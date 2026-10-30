"""Offline tests of the real ASGI application, socket ownership and audit gate."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from roop_3d import __version__
from roop_3d.core.hardware import Hardware
from roop_3d.core.settings import Settings
from roop_3d.server.app import create_app
from roop_3d.server.__main__ import PortCollisionError, ReadyServer, reserve_socket


def test_binding_is_strictly_loopback() -> None:
    assert Settings().host == "127.0.0.1"
    for host in ("0.0.0.0", "::", "localhost", "192.168.1.2"):
        with pytest.raises(ValidationError):
            Settings(host=host)


def test_port_defaults_and_development_port_separation() -> None:
    assert Settings().port == 8030
    with pytest.raises(ValidationError, match="reserved"):
        Settings(port=5174)


def test_socket_binds_real_loopback_and_collision_does_not_kill_owner() -> None:
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0))
        owner.listen(1)
        port = owner.getsockname()[1]
        with pytest.raises(PortCollisionError, match="ROOP3D_PORT"):
            reserve_socket(Settings(port=port))
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            assert owner.getsockname()[0] == "127.0.0.1"


def test_real_server_advertises_readiness_after_socket_start_and_shuts_down(capsys) -> None:
    import httpx
    import uvicorn

    async def scenario() -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            listener.setblocking(False)
            settings = Settings(port=listener.getsockname()[1])
            server = ReadyServer(uvicorn.Config(create_app(settings), port=settings.port, log_config=None, access_log=False))
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                for _ in range(100):
                    if server.started:
                        break
                    await asyncio.sleep(0.01)
                assert server.started
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://127.0.0.1:{settings.port}/api/v1/health", timeout=2)
                    assert response.status_code == 200
                    assert response.json()["application"] == "roop_3d"
            finally:
                server.should_exit = True
                await asyncio.wait_for(task, timeout=5)
    asyncio.run(scenario())
    assert '"event": "ready"' in capsys.readouterr().out


def test_cli_collision_exits_with_actionable_json() -> None:
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0))
        owner.listen(1)
        env = dict(os.environ, ROOP3D_PORT=str(owner.getsockname()[1]), ROOP3D_HOST="127.0.0.1")
        result = subprocess.run([sys.executable, "-B", "-m", "roop_3d.server"],
                                cwd=Path(__file__).absolute().parents[2], env=env,
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 2, result.stdout + result.stderr
        assert '"PORT_IN_USE"' in result.stdout
        assert "no existing process was stopped" in result.stdout
        with socket.create_connection(owner.getsockname(), timeout=1):
            pass


def test_environment_overrides_are_roop3d_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROOP3D_PORT", "8130")
    monkeypatch.setenv("ROOP_API_PORT", "9999")
    assert Settings.from_env().port == 8130


def test_health_reports_honest_isolation_and_uptime() -> None:
    with TestClient(create_app(), base_url="http://127.0.0.1:8030") as client:
        first = client.get("/api/v1/health")
        second = client.get("/api/v1/health")
        assert first.status_code == 200
        value = first.json()
        assert value["application"] == "roop_3d"
        assert value["version"] == __version__
        assert 0 <= value["uptime_seconds"] <= second.json()["uptime_seconds"]
        assert value["isolation"]["status"] in ("passed", "failed", "unknown")
        assert value["isolation"]["scope"]


def test_missing_isolation_evidence_is_unknown(tmp_path: Path) -> None:
    with TestClient(create_app(repo_root=tmp_path), base_url="http://127.0.0.1:8030") as client:
        assert client.get("/api/v1/health").json()["isolation"]["status"] == "unknown"


def test_isolation_verifier_executes_real_hash_checks(tmp_path: Path) -> None:
    from roop_3d.audit.isolation import inventory, verify
    legacy = tmp_path / "app"
    legacy.mkdir()
    source = legacy / "main.py"
    source.write_bytes(b"original")
    baseline = inventory(tmp_path)
    assert verify(tmp_path, baseline)["ok"]
    source.write_bytes(b"modified")
    result = verify(tmp_path, baseline)
    assert not result["ok"]
    assert result["violations"][0]["status"] == "HASH_MISMATCH"


def test_hardware_is_probed_without_claiming_inference_or_inventing_cpu_count() -> None:
    with TestClient(create_app(), base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/v1/system/hardware")
        assert response.status_code == 200
        value = Hardware.model_validate(response.json())
        assert value.logical_cpu_count == os.cpu_count()
        assert value.physical_cpu_count is None
        assert value.cuda_runtime_version is None
        assert value.provider_execution_verified is False
        if value.gpu_probe_state == "VERIFIED_LOCAL":
            assert value.gpus
            assert all(g.name and 0 <= g.free_vram_mib <= g.total_vram_mib for g in value.gpus)
        else:
            assert value.gpus == []
        if value.provider_probe_state == "VERIFIED_LOCAL":
            assert value.advertised_providers and value.ort_version
        else:
            assert value.advertised_providers is None


def test_explicit_unknown_hardware_is_valid() -> None:
    value = Hardware(timestamp="probe unavailable", os={}, logical_cpu_count=None)
    assert value.gpus == [] and value.advertised_providers is None
    assert value.gpu_probe_state == "UNKNOWN"


def test_error_envelopes_and_body_limit() -> None:
    with TestClient(create_app(Settings(max_body_bytes=8)), base_url="http://127.0.0.1:8030") as client:
        missing = client.get("/missing")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "HTTP_404"
        assert missing.json()["error"]["request_id"] == missing.headers["x-request-id"]
        large = client.post("/missing", content=b"123456789")
        assert large.status_code == 413
        assert large.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_chunked_body_limit_cannot_be_bypassed_without_content_length() -> None:
    app = create_app(Settings(max_body_bytes=8))
    sent = []
    messages = iter([{"type": "http.request", "body": b"12345", "more_body": True},
                     {"type": "http.request", "body": b"67890", "more_body": False}])

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(app({"type": "http", "method": "POST", "path": "/missing", "headers": [],
                     "query_string": b"", "scheme": "http", "server": ("127.0.0.1", 8030)}, receive, send))
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "REQUEST_TOO_LARGE"


def test_origin_strategy_is_explicit() -> None:
    with TestClient(create_app(), base_url="http://127.0.0.1:8030") as client:
        assert client.get("/api/v1/health", headers={"Origin": "https://example.com"}).status_code == 403
        denied_host = client.get("/api/v1/health", headers={"Host": "attacker.example"})
        assert denied_host.status_code == 400
        assert denied_host.json()["error"]["code"] == "HOST_DENIED"
        assert client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:5174"}).status_code == 403
    with TestClient(create_app(Settings(dev_cors=True)), base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:5174"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"


def test_service_import_does_not_load_heavy_optional_packages() -> None:
    result = subprocess.run([sys.executable, "-B", "-c",
                             "import sys; from roop_3d.server.app import create_app; create_app(); "
                             "assert not ({'torch','diffusers','tensorrt','smplx'} & set(sys.modules))"],
                            cwd=Path(__file__).absolute().parents[2], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
