"""Direct-media HTTP downloads with DNS, redirect, size, and type guards."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from cove_sensory_mcp.errors import ErrorCode, SensoryError

from .types import ResolvedSource

DnsResolver = Callable[[str], Iterable[str]]


def _system_resolve(host: str) -> Iterable[str]:
    return {cast(str, item[4][0]) for item in socket.getaddrinfo(host, None)}


def _blocked(
    message: str = "The media URL was blocked by the network policy.",
) -> SensoryError:
    return SensoryError(ErrorCode.DOWNLOAD_BLOCKED, message)


@dataclass(frozen=True, slots=True)
class ValidatedUrl:
    url: str
    hostname: str


@dataclass(frozen=True, slots=True)
class NetworkLimits:
    max_bytes: int
    timeout_seconds: float
    max_redirects: int

    def __post_init__(self) -> None:
        if self.max_bytes <= 0 or self.timeout_seconds <= 0 or self.max_redirects < 0:
            raise ValueError("network limits must be positive")


class NetworkPolicy:
    """Validate a URL and every DNS result before any request is sent."""

    def __init__(self, resolver: DnsResolver = _system_resolve) -> None:
        self._resolver = resolver

    def validate_url(self, url: str) -> ValidatedUrl:
        try:
            parsed = urlsplit(url)
            host = parsed.hostname
            _ = parsed.port
        except ValueError:
            raise _blocked() from None
        if (
            parsed.scheme not in {"http", "https"}
            or host is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or host.lower().rstrip(".") == "localhost"
        ):
            raise _blocked()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            try:
                addresses = list(self._resolver(host))
            except (OSError, socket.gaierror):
                raise SensoryError(
                    ErrorCode.DOWNLOAD_FAILED,
                    "The media host could not be resolved.",
                    retryable=True,
                ) from None
        else:
            addresses = [host]
        if not addresses:
            raise _blocked()
        for raw_address in addresses:
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError:
                raise _blocked() from None
            if any(
                (
                    address.is_private,
                    address.is_loopback,
                    address.is_link_local,
                    address.is_multicast,
                    address.is_reserved,
                    address.is_unspecified,
                    bool(getattr(address, "is_site_local", False)),
                )
            ):
                raise _blocked()
        return ValidatedUrl(url=url, hostname=host)


def _detect_media(header: bytes) -> frozenset[str]:
    if header.startswith(b"\xff\xd8\xff"):
        return frozenset({"image/jpeg"})
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return frozenset({"image/png"})
    if header.startswith((b"GIF87a", b"GIF89a")):
        return frozenset({"image/gif"})
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return frozenset({"image/webp"})
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return frozenset({"audio/wav", "audio/x-wav", "audio/wave"})
    if header.startswith(b"ID3") or header[:2] in {
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2",
    }:
        return frozenset({"audio/mpeg", "audio/mp3"})
    if header.startswith(b"fLaC"):
        return frozenset({"audio/flac", "audio/x-flac"})
    if header.startswith(b"OggS"):
        return frozenset({"audio/ogg", "video/ogg"})
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return frozenset({"audio/webm", "video/webm"})
    if len(header) >= 2 and header[0] == 0xFF and header[1] & 0xF6 == 0xF0:
        return frozenset({"audio/aac"})
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return frozenset(
            {
                "audio/mp4",
                "audio/x-m4a",
                "video/mp4",
                "video/quicktime",
            }
        )
    return frozenset()


def _canonical_media_type(content_type: str) -> str:
    return {
        "audio/mp3": "audio/mpeg",
        "audio/wave": "audio/wav",
        "audio/x-flac": "audio/flac",
        "audio/x-m4a": "audio/mp4",
        "audio/x-wav": "audio/wav",
    }.get(content_type, content_type)


class MediaDownloader:
    """Stream one validated direct media response into a request workspace."""

    def __init__(
        self,
        policy: NetworkPolicy | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._policy = policy or NetworkPolicy()
        self._client = client or httpx.AsyncClient(
            follow_redirects=False, trust_env=False
        )

    async def download(
        self, url: str, destination_dir: Path, limits: NetworkLimits
    ) -> ResolvedSource:
        try:
            return await asyncio.wait_for(
                self._download(url, destination_dir, limits),
                timeout=limits.timeout_seconds,
            )
        except TimeoutError:
            raise SensoryError(
                ErrorCode.DOWNLOAD_FAILED,
                "The direct media download timed out.",
                retryable=True,
            ) from None

    async def _download(
        self, url: str, destination_dir: Path, limits: NetworkLimits
    ) -> ResolvedSource:
        destination = destination_dir.resolve(strict=True)
        target = destination / f"download_{uuid.uuid4().hex}.media"
        current = url
        response: httpx.Response | None = None
        completed = False
        try:
            for redirects in range(limits.max_redirects + 1):
                validated = self._policy.validate_url(current)
                request = self._client.build_request("GET", validated.url)
                response = await self._client.send(
                    request,
                    stream=True,
                    follow_redirects=False,
                )
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    await response.aclose()
                    response = None
                    if location is None or redirects >= limits.max_redirects:
                        raise _blocked("The media URL redirected too many times.")
                    current = urljoin(current, location)
                    continue
                if response.status_code == 404:
                    raise SensoryError(
                        ErrorCode.SOURCE_NOT_FOUND, "The media source was not found."
                    )
                response.raise_for_status()
                content_type = (
                    response.headers.get("content-type", "").split(";", 1)[0].lower()
                )
                if not content_type.startswith(("image/", "video/", "audio/")):
                    raise _blocked("The response is not direct media.")
                raw_length = response.headers.get("content-length")
                if raw_length is not None:
                    try:
                        if int(raw_length) > limits.max_bytes:
                            raise SensoryError(
                                ErrorCode.MEDIA_TOO_LARGE, "The media is too large."
                            )
                    except ValueError:
                        raise _blocked("The response length is invalid.") from None
                size = 0
                header = bytearray()
                with target.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > limits.max_bytes:
                            raise SensoryError(
                                ErrorCode.MEDIA_TOO_LARGE, "The media is too large."
                            )
                        if len(header) < 32:
                            header.extend(chunk[: 32 - len(header)])
                        handle.write(chunk)
                detected = _detect_media(bytes(header))
                if content_type not in detected:
                    raise _blocked(
                        "The response media type did not match its contents."
                    )
                canonical_type = _canonical_media_type(content_type)
                parsed = urlsplit(current)
                display_name = Path(unquote(parsed.path)).name or "downloaded-media"
                result = ResolvedSource(
                    path=target,
                    source_kind="download",
                    display_name=display_name,
                    cleanup_required=True,
                    original_size=size,
                    mime_type=canonical_type,
                )
                completed = True
                return result
            raise _blocked("The media URL redirected too many times.")
        except SensoryError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise SensoryError(
                ErrorCode.DOWNLOAD_FAILED,
                "The direct media download failed.",
                retryable=True,
                cause=exc,
            ) from None
        finally:
            if response is not None:
                await response.aclose()
            if target.exists() and not completed:
                target.unlink(missing_ok=True)
