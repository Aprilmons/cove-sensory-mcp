"""Cloud instructions must match the environment-backed HTTP entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp import Client

from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.server import create_server
from cove_sensory_mcp.services import AppServices
from cove_sensory_mcp.tools.setup import sensory_setup_guide


@pytest.fixture
def services(tmp_path) -> AppServices:
    return AppServices(ConfigStore(tmp_path / "config.yaml"), MemorySecretStore())


@pytest.mark.asyncio
async def test_cloud_mcp_guide_names_environment_setup_and_explicit_verification(
    services: AppServices,
) -> None:
    async with Client(
        create_server(services, cloud_deployment=True), mode="legacy"
    ) as client:
        result = await client.call_tool("sensory_setup_guide", {})

    assert result.is_error is False
    guide = result.structured_content
    assert guide["configuration"]["service"] == "Cove service"
    assert guide["configuration"]["variables"] == [
        "GEMINI_API_KEY",
        "COVE_GEMINI_MODEL",
        "COVE_DATA_DIR",
    ]
    assert "command" not in guide
    assert "cove-sensory-mcp configure" not in str(guide)
    assert guide["next_step"]["tool"] == "sensory_self_test"
    assert set(guide["next_step"]["modalities"]) == {
        "image",
        "video_visual",
        "video_audio",
        "audio",
        "music",
    }
    assert "approval" in guide["next_step"]["notice"]
    assert "quota" in guide["next_step"]["notice"]
    assert "/data" in guide["persistence"]
    assert "Never send API Keys" in guide["security_notice"]
    assert services.config_store.load().providers == {}


@pytest.mark.asyncio
async def test_cloud_mode_keeps_all_seven_tool_schemas_unchanged(
    services: AppServices,
) -> None:
    local_tools = await create_server(services).list_tools()
    cloud_tools = await create_server(services, cloud_deployment=True).list_tools()
    assert len(local_tools) == 7
    assert [tool.model_dump() for tool in cloud_tools] == [
        tool.model_dump() for tool in local_tools
    ]


@pytest.mark.asyncio
async def test_environment_variables_do_not_change_the_local_guide(
    services: AppServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COVE_DATA_DIR", "/data")
    async with Client(create_server(services), mode="legacy") as client:
        result = await client.call_tool("sensory_setup_guide", {})
    assert result.structured_content["command"] == "cove-sensory-mcp configure"


def test_http_entry_point_selects_cloud_instructions(
    services: AppServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cove_sensory_mcp.server as server_module

    selected = []
    transports = []

    def capture_create_server(active_services, *, cloud_deployment=False):
        assert active_services is services
        selected.append(cloud_deployment)
        return SimpleNamespace(run=lambda **options: transports.append(options))

    monkeypatch.setattr(server_module, "create_server", capture_create_server)
    server_module.run_http(services, host="127.0.0.1", port=8000)
    server_module.run_stdio(services)

    assert selected == [True, False]
    assert [options["transport"] for options in transports] == [
        "streamable-http",
        "stdio",
    ]


@pytest.mark.asyncio
async def test_cloud_guide_reads_no_service_state_and_returns_fresh_data() -> None:
    class UnavailableServices:
        def __getattribute__(self, name):
            raise AssertionError("Setup instructions must not read config or secrets")

    services = UnavailableServices()
    first = await sensory_setup_guide(services, cloud_deployment=True)
    first["configuration"]["variables"].append("caller mutation")
    second = await sensory_setup_guide(services, cloud_deployment=True)
    assert "caller mutation" not in second["configuration"]["variables"]
