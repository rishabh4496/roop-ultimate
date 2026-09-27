"""Stage 5 backend: REST, WebSocket telemetry, preview, background render, stop, 206 + CORS.

The end-to-end tests drive the real models through the HTTP API on a clip made
from the insightface sample photo (six real faces, slow zoom, a tone for audio),
then check OUTCOMES: the assigned face becomes the source identity, unassigned
faces stay who they were, the file streams with 206 ranges.
"""
from __future__ import annotations

import io
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from face_engine.server.api import create_app
from face_engine.server.state import ServerSettings

ORIGIN = "http://localhost:5173"


def _insightface_image() -> np.ndarray:
    import insightface

    return cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))


_ANALYSIS: list[Any] = []


def _analysis() -> tuple[Any, Any, Any]:
    """CUDA detector + encoder, the same providers the server uses for analysis."""
    if not _ANALYSIS:
        from face_engine.core.config import EngineConfig, Provider
        from face_engine.core.execution import ExecutionEngine
        from face_engine.models.zoo import build_default_registry
        from face_engine.pipeline.detector import SCRFDDetector
        from face_engine.processors.swapper import IdentityEncoder

        registry = build_default_registry()
        engine = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU], strict=True))
        _ANALYSIS.extend([engine, SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps")),
                          IdentityEncoder(engine, registry.ensure("arcface_w600k_r50"))])
    return _ANALYSIS[0], _ANALYSIS[1], _ANALYSIS[2]


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from face_engine.media.tools import find_tool

    d = tmp_path_factory.mktemp("server_media")
    photo = _insightface_image()
    cv2.imwrite(str(d / "group.jpg"), photo)
    # Source: exactly ONE person. A loose crop held two similar-sized faces and
    # two detectors (CUDA vs TensorRT FP16) broke the size tie differently.
    _, det, _ = _analysis()
    faces = sorted(det.detect(photo), key=lambda f: f.bbox[0])
    face = faces[2]
    margin = 0.3 * face.width
    x0, y0, x1, y1 = face.bbox
    crop = photo[int(max(y0 - margin, 0)):int(y1 + margin), int(max(x0 - margin, 0)):int(x1 + margin)]
    cv2.imwrite(str(d / "source.jpg"), crop)
    assert len(det.detect(crop)) == 1
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-loop", "1", "-i", str(d / "group.jpg"),
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=2",
                    "-vf", "scale=960:-2,zoompan=z='1+0.0015*on':d=1:s=960x664:fps=24000/1001",
                    "-t", "2", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(d / "target.mp4")], check=True)
    (d / "notes.txt").write_text("not an image", encoding="utf-8")
    return {"dir": d, "source": d / "source.jpg", "target": d / "target.mp4",
            "group": d / "group.jpg", "text": d / "notes.txt"}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    settings = ServerSettings(workspace=tmp_path_factory.mktemp("workspace"),
                              cors_origins=[ORIGIN])
    with TestClient(create_app(settings)) as c:
        yield c


def _upload(client: TestClient, sources: list[Path], target: Path) -> Any:
    files = [("sources", (p.name, p.read_bytes(), "application/octet-stream")) for p in sources]
    files.append(("target", (target.name, target.read_bytes(), "application/octet-stream")))
    return client.post("/api/project/load", files=files)


def _wait(client: TestClient, timeout: float = 600) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get("/api/pipeline/status").json()["job"]
        if job and job["state"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("render did not finish")


# ------------------------------------------------------------------ validation (no models)
def test_options_and_param_validation(client: TestClient) -> None:
    opts = client.get("/api/options").json()
    assert opts["unavailable"] == {"alphaface_256": "no public model release"}
    assert opts["providers"]["cpu"] is True
    assert set(opts["params"]["pixel_boost"]["enum"]) == {"none", "256x256", "512x512", "1024x1024"}
    for bad in ({"pixel_boost": "2048x2048"}, {"enhancer_blend": 101}, {"mask_types": ["lasso"]},
                {"execution_provider": "rocm"}, {"swapper_model": "simswap"}, {"unknown": 1}):
        assert client.post("/api/pipeline/start", json=bad).status_code == 422, bad


def test_file_endpoints_refuse_traversal(client: TestClient) -> None:
    for name in ("..%2F..%2Fpyproject.toml", "..\\secret.txt", ".hidden", "nope.mp4"):
        assert client.get(f"/api/outputs/{name}").status_code == 404
    assert client.get("/api/thumbnails/..%2Fx.jpg").status_code == 404


# ------------------------------------------------------------------ with models
@pytest.mark.gpu
def test_upload_validation(client: TestClient, media: dict[str, Path]) -> None:
    r = _upload(client, [media["text"]], media["target"])
    assert r.status_code == 422 and "unsupported file type" in r.json()["detail"]
    blank = media["dir"] / "blank.png"
    cv2.imwrite(str(blank), np.full((200, 200, 3), 127, np.uint8))
    r = _upload(client, [blank], media["target"])
    assert r.status_code == 422 and "no face" in r.json()["detail"]
    fake = media["dir"] / "fake.mp4"
    fake.write_bytes(b"\x00" * 1000)
    r = _upload(client, [media["source"]], fake)
    assert r.status_code == 422 and "not a readable video" in r.json()["detail"]


@pytest.fixture(scope="module")
def project(client: TestClient, media: dict[str, Path]) -> dict[str, Any]:
    r = _upload(client, [media["source"]], media["target"])
    assert r.status_code == 200, r.text
    body = r.json()
    # 2 s at 24000/1001 is 47.95 frames: ffmpeg writes 47.
    assert body["target"]["kind"] == "video" and body["target"]["frames"] == 47
    faces = client.get("/api/detect/faces", params={"frames": 4}).json()["faces"]
    return {"source": body["sources"][0]["id"], "faces": faces, "frames": 47}


@pytest.mark.gpu
def test_detection_groups_people(client: TestClient, project: dict[str, Any]) -> None:
    faces = project["faces"]
    assert len(faces) == 6  # six people, each seen on all 4 sampled frames
    assert all(f["count"] == 4 for f in faces)
    thumb = client.get(faces[0]["thumbnail_url"])
    assert thumb.status_code == 200 and cv2.imdecode(np.frombuffer(thumb.content, np.uint8), 1) is not None


def _person_nearest(faces: list[dict[str, Any]], x_center: float) -> dict[str, Any]:
    return min(faces, key=lambda f: abs((f["bbox"][0] + f["bbox"][2]) / 2 - x_center))


@pytest.mark.gpu
def test_preview_frame(client: TestClient, project: dict[str, Any]) -> None:
    body = {"frame_index": 10, "execution_provider": "cuda", "swapper_model": "hyperswap_1a_256",
            "enhancer_model": "none", "pixel_boost": "none", "mask_blur": 0.3,
            "enhancer_blend": 80}
    r = client.post("/api/preview/frame", json=body)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-faces"] == "6" and r.headers["x-swapped"] == "6"  # no assignment: all
    assert r.headers["x-cache"] == "miss"
    for key in ("x-render-ms", "x-decode-ms", "x-process-ms", "x-encode-ms"):
        assert float(r.headers[key]) >= 0
    image = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (664, 960)
    # The miss decoded the GOP up to frame 10 and prefetches the rest of it.
    again = client.post("/api/preview/frame", json={**body, "frame_index": 5})
    assert again.headers["x-cache"] == "hit"
    original = client.post("/api/preview/frame", json={**body, "mode": "original"})
    decoded = cv2.imdecode(np.frombuffer(original.content, np.uint8), cv2.IMREAD_COLOR)
    changed = (np.abs(decoded.astype(int) - image).max(axis=2) > 20).mean()
    assert 0.005 < changed < 0.5  # the faces changed, the rest of the frame did not
    # Validation, and the query-string variant.
    assert client.post("/api/preview/frame", json={**body, "pixel_boost": "3x"}).status_code == 422
    b64 = client.get("/api/preview/frame", params={"frame": 10, "format": "base64",
                                                   "mode": "original"}).json()
    assert b64["image"].startswith("data:image/jpeg;base64,")


@pytest.mark.gpu
def test_render_assigned_face_only(client: TestClient, project: dict[str, Any],
                                   media: dict[str, Path]) -> None:
    from face_engine.media.capturer import VideoSource
    from face_engine.media.ffmpeg_pipe import (
        inspect_output,
        verify_http_range_streaming,
    )

    # Assign the source to the LEFTMOST person only.
    faces = sorted(project["faces"], key=lambda f: f["bbox"][0])
    chosen = faces[0]
    assert client.post("/api/project/assign", json={chosen["id"]: project["source"]}).status_code == 200
    r = client.post("/api/pipeline/start", json={"execution_provider": "cuda",
                                                  "mask_types": ["box", "occlusion"]})
    assert r.status_code == 200, r.text
    assert client.post("/api/pipeline/start", json={}).status_code in (409, 200)
    job = _wait(client)
    assert job["state"] == "completed", job
    assert job["frames_done"] == job["frames_total"] == project["frames"]

    name = job["output"].rsplit("/", 1)[1]
    out = client.app.state.app_state.outputs_dir() / name
    report = inspect_output(out)
    assert report.frames == project["frames"] and report.faststart and report.audio_codec == "aac"
    verify_http_range_streaming(out)

    # Outcome: the chosen person became the source; the others did not change identity.
    _, det, enc = _analysis()
    source_img = cv2.imread(str(media["source"]))
    source = enc.embed(source_img, max(det.detect(source_img), key=lambda f: f.width))
    frame_out = next(VideoSource(out).frames(24, 25)).image
    frame_in = next(VideoSource(media["target"]).frames(24, 25)).image
    faces_out, faces_in = det.detect(frame_out), det.detect(frame_in)
    leftmost_out = min(faces_out, key=lambda f: f.bbox[0])
    assert enc.embed(frame_out, leftmost_out).similarity(source) > 0.6
    for face in sorted(faces_in, key=lambda f: f.bbox[0])[1:]:
        twin = min(faces_out, key=lambda f: float(np.linalg.norm(f.center - face.center)))
        assert enc.embed(frame_out, twin).similarity(enc.embed(frame_in, face)) > 0.8


@pytest.mark.gpu
def test_outputs_serve_ranges_with_cors(client: TestClient) -> None:
    job = client.get("/api/pipeline/status").json()["job"]
    url = job["output"]
    full = client.get(url, headers={"Origin": ORIGIN})
    assert full.status_code == 200 and full.headers["access-control-allow-origin"] == ORIGIN
    part = client.get(url, headers={"Range": "bytes=100-1123", "Origin": ORIGIN})
    assert part.status_code == 206 and part.content == full.content[100:1124]
    assert part.headers["content-range"] == f"bytes 100-1123/{len(full.content)}"
    exposed = part.headers["access-control-expose-headers"].lower()
    assert "content-range" in exposed and "accept-ranges" in exposed
    assert client.get(url, headers={"Range": f"bytes={len(full.content) + 5}-"}).status_code == 416
    preflight = client.options(url, headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET",
                                             "Access-Control-Request-Headers": "range"})
    assert preflight.status_code == 200
    target = client.get("/api/media/target", headers={"Range": "bytes=0-99"})
    assert target.status_code == 206 and len(target.content) == 100


@pytest.mark.gpu
def test_telemetry_websocket(client: TestClient) -> None:
    with client.websocket_connect("/ws/telemetry") as sock:
        msg = sock.receive_json()
    assert msg["type"] == "telemetry" and msg["interval_s"] == 0.25  # 4 Hz
    gpu = msg["gpu"]
    assert gpu is None or {"temperature_c", "vram_used_mb", "vram_total_mb", "utilization_pct",
                           "stale"} <= set(gpu)
    assert msg["job"]["state"] in ("completed", "failed", "cancelled")
    assert {"fps", "elapsed_s", "eta_s", "latency_ms", "frames_done"} <= set(msg["job"])
    render = msg["render"]
    assert {"fps", "frames_done", "frames_total", "progress", "eta_s", "elapsed_s"} <= set(render)
    assert 0.0 <= render["progress"] <= 1.0


def test_stalled_gpu_query_backs_off_without_blocking() -> None:
    """A hung NVML call must not stall telemetry: stale sample, then exponential back-off."""
    import asyncio
    import threading

    from face_engine.server.telemetry import GuardedSampler

    release = threading.Event()

    class Flaky:
        calls = 0

        def query(self) -> dict[str, Any]:
            Flaky.calls += 1
            if Flaky.calls > 1:
                release.wait(5)  # the driver "hangs"
            return {"name": "gpu", "utilization_pct": 50, "vram_used_mb": 1, "vram_total_mb": 2,
                    "temperature_c": 40, "source": "fake"}

    guard = GuardedSampler(Flaky(), timeout=0.05)

    async def run() -> list[Any]:
        first = await guard.sample()
        t0 = time.monotonic()
        second = await guard.sample()          # times out -> stale copy of the first
        third = await guard.sample()           # inside the back-off window: no new query
        return [first, second, third, time.monotonic() - t0]

    first, second, third, took = asyncio.run(run())
    release.set()
    guard.close()
    assert first["stale"] is False and second["stale"] is True and third["stale"] is True
    assert took < 0.5 and Flaky.calls == 2 and guard.failures == 1  # 3rd call: backed off


@pytest.mark.gpu
def test_stop_cancels_and_frees_everything(client: TestClient, project: dict[str, Any]) -> None:
    from face_engine.media import ipc_pool

    r = client.post("/api/pipeline/start", json={"execution_provider": "cuda", "workers": 1,
                                                  "mask_types": ["box", "occlusion", "region"],
                                                  "enhancer_model": "gpen_bfr_512"})
    assert r.status_code == 200, r.text
    deadline = time.monotonic() + 300
    while client.get("/api/pipeline/status").json()["job"]["frames_done"] < 2:
        assert time.monotonic() < deadline, "render never produced a frame"
        time.sleep(0.2)
    stopped = client.post("/api/pipeline/stop").json()
    assert stopped["job"]["state"] == "cancelled"
    assert stopped["shared_memory_segments"] == 0 and not ipc_pool._OWNED
    job_id = stopped["job"]["id"]
    assert not any(job_id in p.name for p in client.app.state.app_state.outputs_dir().iterdir())
    # The server is usable again immediately.
    assert client.post("/api/pipeline/start", json={"execution_provider": "cuda"}).status_code == 200
    assert _wait(client)["state"] == "completed"


@pytest.mark.gpu
def test_stop_cancels_a_segment_pool_render(client: TestClient, project: dict[str, Any]) -> None:
    """workers > 1: keyframe segments in GPU processes; stop kills them and frees everything.

    Lightest model set on purpose: two processes with the full cinema set ran this
    machine out of host RAM (2026-09-28).
    """
    from face_engine.media import ipc_pool

    r = client.post("/api/pipeline/start", json={"execution_provider": "cuda", "workers": 2,
                                                  "mask_types": ["box"]})
    assert r.status_code == 200, r.text
    time.sleep(3.0)  # the workers are starting (spawn + model load)
    stopped = client.post("/api/pipeline/stop").json()
    assert stopped["job"]["state"] in ("cancelled", "completed")
    assert stopped["shared_memory_segments"] == 0 and not ipc_pool._OWNED
    outputs = client.app.state.app_state.outputs_dir()
    assert not any(p.name.startswith(".") and p.is_dir() for p in outputs.iterdir())


@pytest.mark.gpu
def test_image_target(client: TestClient, media: dict[str, Path]) -> None:
    r = _upload(client, [media["source"]], media["group"])
    assert r.status_code == 200 and r.json()["target"]["kind"] == "image"
    client.post("/api/pipeline/start", json={"execution_provider": "cuda"})
    job = _wait(client)
    assert job["state"] == "completed" and job["output"].endswith(".png")
    png = client.get(job["output"]).content
    assert cv2.imdecode(np.frombuffer(png, np.uint8), 1).shape[:2] == _insightface_image().shape[:2]
    assert io.BytesIO(png).getbuffer().nbytes > 0
