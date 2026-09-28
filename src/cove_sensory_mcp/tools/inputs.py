"""Closed public Pydantic inputs for the four production sensing tools."""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cove_sensory_mcp.models import DetailLevel, ProviderId


class ChatGPTFile(BaseModel):
    """One temporary file handoff supplied by ChatGPT to an MCP tool."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    download_url: str = Field(min_length=1, max_length=16_384)
    file_id: str = Field(min_length=1, max_length=512)
    mime_type: str | None = Field(default=None, min_length=1, max_length=255)
    file_name: str | None = Field(default=None, min_length=1, max_length=1_024)

    @field_validator("download_url")
    @classmethod
    def validate_download_url(cls, value: str) -> str:
        """Accept only credential-free HTTPS URLs suitable for guarded download."""
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError:
            raise ValueError("download_url is invalid") from None
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("download_url is invalid")
        return value

    @field_validator("file_id", "mime_type", "file_name")
    @classmethod
    def reject_control_characters(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("file metadata is invalid")
        return value


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    source: str = Field(min_length=1, max_length=8_192)
    question: str = Field(default="", max_length=8_000)
    detail: DetailLevel = DetailLevel.AUTO
    language: str = Field(default="zh-CN", min_length=2, max_length=64)

    @field_validator("source")
    @classmethod
    def safe_source(cls, value: str) -> str:
        if not value.strip() or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("source is invalid")
        return value


class _RangedInput(_Input):
    start_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_range(self) -> _RangedInput:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("end_seconds must follow start_seconds")
        return self


class SenseImageInput(_Input):
    provider: ProviderId | None = None


class SenseVideoInput(_RangedInput):
    include_audio: bool = True
    visual_provider: ProviderId | None = None
    audio_provider: ProviderId | None = None


class SenseAudioInput(_RangedInput):
    include_transcript: bool = True


class SenseMusicInput(_RangedInput):
    include_lyrics_transcript: bool = False
