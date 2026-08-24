"""Environment-only composition for a private Railway deployment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.config.secrets import KeyringSecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.models import Modality, RouteConfig
from cove_sensory_mcp.services import AppServices

DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
_PROVIDER_ID = "gemini"
_API_KEY_ENV = "GEMINI_API_KEY"


def _provider(model: str) -> ProviderConfig:
    return ProviderConfig(
        adapter="gemini",
        model=model,
        api_key_env=_API_KEY_ENV,
        declared_capabilities={modality: True for modality in Modality},
    )


def _provider_identity(provider: ProviderConfig) -> dict[str, object]:
    return provider.model_dump(
        mode="python",
        exclude={
            "verified_capabilities",
            "verified_joint_capabilities",
            "last_verified_at",
        },
    )


def bootstrap_cloud_config(store: ConfigStore, *, model: str) -> AppConfig:
    """Ensure one env-backed Gemini provider and explicit routes exist."""
    desired = _provider(model)

    def merge(config: AppConfig) -> None:
        existing = config.providers.get(_PROVIDER_ID)
        if existing is None or _provider_identity(existing) != _provider_identity(
            desired
        ):
            config.providers[_PROVIDER_ID] = desired.model_copy(deep=True)
        for modality in Modality:
            route = getattr(config.routes, modality.value)
            if route is None or route.primary == _PROVIDER_ID:
                setattr(
                    config.routes,
                    modality.value,
                    RouteConfig(primary=_PROVIDER_ID),
                )

    return store.update(merge)


def build_cloud_services(
    environ: Mapping[str, str] | None = None,
) -> AppServices:
    """Build Linux-safe services without ever persisting provider credentials."""
    active = os.environ if environ is None else environ
    data_dir = Path(active.get("COVE_DATA_DIR", "/data")).expanduser()
    if not data_dir.is_absolute() or data_dir.parent == data_dir:
        raise ValueError("COVE_DATA_DIR must be a non-root absolute path")
    model = active.get("COVE_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    store = ConfigStore(data_dir / "config.yaml", jobs_dir=data_dir / "jobs")
    bootstrap_cloud_config(store, model=model)
    return AppServices(config_store=store, secret_store=KeyringSecretStore())
