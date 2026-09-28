from __future__ import annotations

from pathlib import Path

import pytest

from cove_sensory_mcp.cloud import (
    DEFAULT_GEMINI_MODEL,
    bootstrap_cloud_config,
    build_cloud_services,
)
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.models import Modality


def test_cloud_bootstrap_creates_env_only_gemini_routes(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml")

    config = bootstrap_cloud_config(store, model=DEFAULT_GEMINI_MODEL)

    provider = config.providers["gemini"]
    assert provider.model == DEFAULT_GEMINI_MODEL
    assert provider.credential_ref is None
    assert provider.api_key_env == "GEMINI_API_KEY"
    assert provider.verified_capabilities == {}
    for modality in Modality:
        assert provider.declared_capabilities[modality] is True
        assert getattr(config.routes, modality.value).primary == "gemini"
    persisted = store.path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in persisted
    assert "secret" not in persisted.lower()


def test_cloud_bootstrap_preserves_verification_for_same_provider(
    tmp_path: Path,
) -> None:
    store = ConfigStore(tmp_path / "config.yaml")
    bootstrap_cloud_config(store, model=DEFAULT_GEMINI_MODEL)

    def verify_image(config) -> None:
        config.providers["gemini"].verified_capabilities[Modality.IMAGE] = True

    store.update(verify_image)
    config = bootstrap_cloud_config(store, model=DEFAULT_GEMINI_MODEL)

    assert config.providers["gemini"].verified_capabilities == {Modality.IMAGE: True}


def test_cloud_bootstrap_resets_verification_when_model_changes(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml")
    bootstrap_cloud_config(store, model="gemini-old")

    def verify_image(config) -> None:
        config.providers["gemini"].verified_capabilities[Modality.IMAGE] = True

    store.update(verify_image)
    config = bootstrap_cloud_config(store, model="gemini-new")

    assert config.providers["gemini"].model == "gemini-new"
    assert config.providers["gemini"].verified_capabilities == {}


@pytest.mark.parametrize("data_dir", ["relative", "/"])
def test_build_cloud_services_requires_safe_absolute_data_path(data_dir: str) -> None:
    with pytest.raises(ValueError, match="non-root absolute"):
        build_cloud_services({"COVE_DATA_DIR": data_dir})
