"""Stage 8: ``face_engine/run.py`` (launcher) and ``face_engine/benchmark.py`` logic.

No GPU work: the engine step is exercised with the compiler and lookups
patched, the API / proxy through in-process ASGI transports. The launcher's
live paths (Uvicorn, the WebSocket proxy, a worker render) were run by hand
on 2026-09-28 (README, Stage 8).
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from face_engine import benchmark
from face_engine import run as launcher


# ---------------------------------------------------------------------------- engines
@pytest.mark.parametrize(("profile", "expected"), [
    ("fast", [("scrfd_10g_bnkps", "fp16"), ("hyperswap_1a_256", "auto")]),
    ("balanced", [("scrfd_10g_bnkps", "fp16"), ("hyperswap_1a_256", "auto"), ("xseg_3", "fp16")]),
    ("cinema", [("scrfd_10g_bnkps", "fp16"), ("hyperswap_1a_256", "auto"), ("xseg_3", "fp16"),
                ("bisenet_resnet34", "fp16"), ("gpen_bfr_512", "fp32")]),
])
def test_wanted_engines_match_what_the_processors_load(profile: str,
                                                       expected: list[tuple[str, str]]) -> None:
    assert launcher.wanted_engines(launcher.profile_params(profile)) == expected


@pytest.fixture
def engine_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patched lookups: which engines exist, what the compiler does."""
    from face_engine.core import execution, trt_compiler

    # The detector engine exists: these tests are about the other engines.
    env: dict[str, Any] = {"built": {("scrfd_10g_bnkps", "fp16")}, "calls": [], "exit": 0}
    monkeypatch.setattr(trt_compiler, "ENGINE_DIR", tmp_path)
    monkeypatch.setattr(trt_compiler, "discover_gpu", lambda device_id=0: SimpleNamespace(sm="89"))
    monkeypatch.setattr(execution.ExecutionEngine, "available_providers",
                        staticmethod(lambda: ["TensorrtExecutionProvider"]))
    monkeypatch.setattr(trt_compiler, "find_engine_for",
                        lambda path, precision, **kw: (tmp_path / f"{Path(path).stem}_{precision}"
                                                       if (Path(path).stem, precision)
                                                       in env["built"] else None))

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        model, precision = cmd[cmd.index("--models") + 1], cmd[cmd.index("--precision") + 1]
        env["calls"].append((model, precision))
        if env["exit"] == 0:
            env["built"].add((model, precision))
        return SimpleNamespace(returncode=env["exit"])

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    return env


def _paths(params: Any) -> dict[str, str]:
    from face_engine.server.processing import required_models

    return {name: f"/models/{name}.onnx" for name in required_models(params)}


def test_missing_engines_are_compiled_present_ones_are_not(engine_env: dict[str, Any]) -> None:
    params = launcher.profile_params("cinema")
    engine_env["built"].add(("hyperswap_1a_256", "fp16"))
    launcher.ensure_engines(params, _paths(params))
    assert engine_env["calls"] == [("xseg_3", "fp16"), ("bisenet_resnet34", "fp16"),
                                   ("gpen_bfr_512", "fp32")]
    launcher.ensure_engines(params, _paths(params))  # everything present now
    assert len(engine_env["calls"]) == 3


def test_a_failed_build_is_not_retried_until_recompile(engine_env: dict[str, Any],
                                                       tmp_path: Path) -> None:
    params = launcher.profile_params("fast")
    engine_env["exit"] = 1
    launcher.ensure_engines(params, _paths(params))
    marker = tmp_path / ".launcher_failed_hyperswap_1a_256_sm89_fp16.json"
    assert json.loads(marker.read_text())["exit_code"] == 1
    tried = [("hyperswap_1a_256", "fp16"), ("hyperswap_1a_256", "fp32")]  # "auto"
    assert engine_env["calls"] == tried
    launcher.ensure_engines(params, _paths(params))
    assert engine_env["calls"] == tried  # neither retried the 2nd time
    engine_env["exit"] = 0
    launcher.ensure_engines(params, _paths(params), recompile=True)
    assert engine_env["calls"] == [*tried, ("hyperswap_1a_256", "fp16")]  # fp16 now builds
    assert not marker.exists()


def test_auto_builds_fp16_then_falls_back_to_fp32(engine_env: dict[str, Any],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    params = launcher.profile_params("fast")
    real = launcher.subprocess.run

    def fp16_fails(cmd: list[str], **kwargs: Any) -> Any:
        engine_env["exit"] = 1 if cmd[cmd.index("--precision") + 1] == "fp16" else 0
        return real(cmd, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", fp16_fails)
    launcher.ensure_engines(params, _paths(params))
    assert engine_env["calls"] == [("hyperswap_1a_256", "fp16"), ("hyperswap_1a_256", "fp32")]
    assert ("hyperswap_1a_256", "fp32") in engine_env["built"]


def test_no_tensorrt_means_no_compile(engine_env: dict[str, Any],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from face_engine.core import execution

    monkeypatch.setattr(execution.ExecutionEngine, "available_providers",
                        staticmethod(lambda: ["CUDAExecutionProvider"]))
    params = launcher.profile_params("balanced")
    launcher.ensure_engines(params, _paths(params))
    assert engine_env["calls"] == []


# ---------------------------------------------------------------------------- services
def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(host="127.0.0.1", port=8765, workspace=str(tmp_path / "ws"))


def test_profile_becomes_the_ui_defaults(tmp_path: Path) -> None:
    params = launcher.profile_params("fast")
    with TestClient(launcher.api_app(_args(tmp_path), params, None)) as client:
        defaults = client.get("/api/options").json()["defaults"]
    assert defaults == params.model_dump()
    assert defaults["detection_stride"] == 3 and defaults["mask_types"] == ["box"]


@pytest.fixture
def ui_dir(tmp_path: Path) -> Path:
    ui = tmp_path / "dist"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>ui</title>", encoding="utf-8")
    (ui / "assets" / "app.js").write_bytes(bytes(range(256)) * 8)
    return ui


def test_all_mode_serves_the_ui_with_ranges(tmp_path: Path, ui_dir: Path) -> None:
    app = launcher.api_app(_args(tmp_path), launcher.profile_params("balanced"), ui_dir)
    with TestClient(app) as client:
        assert "<title>ui</title>" in client.get("/").text
        part = client.get("/assets/app.js", headers={"Range": "bytes=10-19"})
        assert part.status_code == 206 and part.content == bytes(range(10, 20))
        assert part.headers["content-range"] == "bytes 10-19/2048"
        assert client.get("/api/options").status_code == 200


def test_ui_mode_proxies_the_api(tmp_path: Path, ui_dir: Path) -> None:
    import httpx

    api = launcher.api_app(_args(tmp_path), launcher.profile_params("cinema"), None)
    proxy = launcher.ui_proxy_app("http://api.test", ui_dir, transport=httpx.ASGITransport(app=api))
    with TestClient(proxy) as client:
        opts = client.get("/api/options")
        assert opts.status_code == 200
        assert opts.json()["defaults"]["enhancer_model"] == "gpen_bfr_512"
        assert client.get("/api/does-not-exist").status_code == 404  # upstream status kept
        part = client.get("/assets/app.js", headers={"Range": "bytes=0-3"})
        assert part.status_code == 206 and part.content == bytes(range(4))


# ---------------------------------------------------------------------------- benchmark
@pytest.mark.parametrize(("gpu", "profile", "override", "expected"), [
    ("NVIDIA GeForce RTX 4070", "fast", None, 40.0),
    ("NVIDIA GeForce RTX 3080 Ti", "fast", None, 40.0),
    ("NVIDIA GeForce RTX 4070", "balanced", None, None),   # its own target is 30 fps
    ("NVIDIA GeForce RTX 3060 Laptop GPU", "fast", None, None),
    ("NVIDIA GeForce RTX 3060 Laptop GPU", "cinema", 5.0, 5.0),
])
def test_fps_gate(gpu: str, profile: str, override: float | None,
                  expected: float | None) -> None:
    assert benchmark.gate_fps(gpu, profile, override) == expected


def test_sync_counter_counts_only_the_render_threads() -> None:
    import torch

    calls = []
    real = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *a, **k: calls.append(1)  # no device needed
    try:
        with benchmark.SyncCounter() as counter:
            torch.cuda.synchronize()  # set-up on this thread
            worker = threading.Thread(target=torch.cuda.synchronize, name="stream-encode")
            worker.start()
            worker.join()
    finally:
        torch.cuda.synchronize = real
    assert counter.count == 1 and counter.setup == 1 and len(calls) == 2
    assert any("test_launcher.py" in site for site in counter.sites)
