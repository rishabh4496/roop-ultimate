"""Contract for routes_angle_scan.py: the /ws/angle-scan pipeline and the
session endpoints the React Biometric Angle HUD drives.

Only the router is mounted (no api.py, no GPU): the detector, landmark and
embedding models are stand-ins, the clip is synthetic, and the head in it turns
from -60 to +60 degrees of yaw so several pose bins fill.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes_angle_scan as ras  # noqa: E402
from roop import face_util as fu  # noqa: E402

W, H, N = 640, 480, 61
SIZE = 120


def _yaw(i):
    return -60 + 2 * i          # -60 .. +60


def _cx(i):
    return 120 + 6 * i          # the face drifts right; its x encodes the frame


def _emb(axis, n=8):
    v = np.zeros(n, np.float32)
    v[axis] = 1
    return v


class Face(dict):
    """insightface.Face-like: attribute and item access."""
    __getattr__ = dict.get


def _face_at(i, identity=0):
    pts = fu._project_reference(_yaw(i), 0)
    pts = (pts - pts.mean(axis=0)) * 100 + (_cx(i), 240)     # eye distance ~67 px
    box = np.array([_cx(i) - SIZE / 2, 240 - SIZE / 2, _cx(i) + SIZE / 2, 240 + SIZE / 2], np.float32)
    # Identity reads high only near frontal, as with a real frontal reference.
    sim = 0.9 if abs(_yaw(i)) <= 20 else 0.3
    emb = sim * _emb(identity) + np.sqrt(1 - sim ** 2) * _emb(5)
    return Face(bbox=box, kps=pts.astype(np.float32), det_score=0.9,
                normed_embedding=emb.astype(np.float32), embedding=(emb * 20).astype(np.float32))


def make_clip(path):
    """The synthetic turning-head clip (also used by check_angle_hud_browser.py)."""
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    rng = np.random.default_rng(0)
    for i in range(N):
        frame = np.full((H, W, 3), 60, np.uint8)
        x0 = int(_cx(i) - SIZE / 2)
        frame[180:300, x0:x0 + SIZE] = rng.integers(40, 220, (SIZE, SIZE, 3), dtype=np.uint8)
        frame[20:30, 0:W] = (0, 0, 255)          # a marker row, unused by the face
        wr.write(frame)
    wr.release()
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return make_clip(str(tmp_path_factory.mktemp("ras") / "clip.avi"))


def _detect(frame):
    # the textured block's left edge -> frame index
    col = frame[240, :, :].astype(int)
    xs = np.nonzero(np.abs(col - 60).sum(axis=1) > 30)[0]
    if xs.size == 0:
        return []
    i = int(round((xs.min() + SIZE / 2 - 120) / 6))
    return [_face_at(max(0, min(N - 1, i)))]


@pytest.fixture()
def client(clip, tmp_path, monkeypatch):
    applied = []
    monkeypatch.setenv("ROOP_TARGET_ANGLE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ras, "resolve_target", lambda payload: {
        "media_path": clip, "media_id": "m1", "person_id": "p1",
        "references": np.stack([_emb(0)])})
    monkeypatch.setattr(ras, "is_busy", lambda: False)
    monkeypatch.setattr(ras, "detect_faces", _detect)
    monkeypatch.setattr(ras, "landmarks_fn", None)
    monkeypatch.setattr(ras, "embed_fn", lambda frame, kps: _emb(0))

    def read(path, idx):
        cap = cv2.VideoCapture(path)
        try:
            for _ in range(idx + 1):
                ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    monkeypatch.setattr(ras, "read_frame", read)

    def apply(person_id, media_id, media_path, picks):
        applied.append((person_id, media_id, media_path, picks))
        return {"added": len(picks)}

    monkeypatch.setattr(ras, "apply_to_person", apply)
    monkeypatch.setattr(ras, "_session", None)
    app = FastAPI()
    app.include_router(ras.router)
    c = TestClient(app)
    c.applied = applied
    return c


def _scan(client, **msg):
    events = []
    with client.websocket_connect("/ws/angle-scan") as ws:
        ws.send_json({"op": "start", "target_person_id": "p1", "step_frames": 1, **msg})
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["event"] in ("result", "error", "cancelled"):
                break
    return events


def test_scan_streams_progress_then_a_portfolio(client):
    events = _scan(client)
    phases = [e["phase"] for e in events if e["event"] == "progress"]
    assert phases[0] == "scan" and "quality" in phases and phases[-1] == "export"
    scan_ev = [e for e in events if e["event"] == "progress" and e["phase"] == "scan" and "scan_fps" in e]
    assert scan_ev and all(k in scan_ev[-1] for k in ("frames_scanned", "tracklet_size", "frame_total"))
    result = events[-1]
    assert result["event"] == "result", events[-1]
    sess = result["session"]
    bins = {b["bin"]: b for b in sess["bins"]}
    # the turning head fills both profiles and the frontal cell
    for name in ("BIN_0_FRONTAL", "BIN_5_PROFILE_LEFT", "BIN_6_PROFILE_RIGHT"):
        assert bins[name]["status"] in ("selected", "nearest"), (name, bins[name])
        assert bins[name]["url"].startswith("/api/angle-scan/image/")
        assert "image" not in bins[name]                       # served by URL, not inline
        assert bins[name]["time_s"] == pytest.approx(bins[name]["frame_idx"] / 30.0, abs=1e-3)
    assert bins["BIN_7_PITCH_UP"]["status"] == "missing"
    assert sess["person_id"] == "p1" and sess["media_id"] == "m1"
    assert sess["frame_total"] == N and sess["fused_embedding"]["available"]
    assert sess["candidates"]["evaluated"] > 0 and sess["tracklets"]
    # the session survives a page reload
    again = client.get("/api/angle-scan/session").json()["session"]
    assert again["cache_key"] == sess["cache_key"]


def test_crop_is_served_with_byte_ranges(client):
    sess = _scan(client)[-1]["session"]
    url = next(b["url"] for b in sess["bins"] if b["url"])
    full = client.get(url)
    assert full.status_code == 200 and full.headers["content-type"].startswith("image/jpeg")
    img = cv2.imdecode(np.frombuffer(full.content, np.uint8), 1)
    assert img.shape == (512, 512, 3)
    part = client.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    etag = full.headers.get("etag")
    if etag:
        assert client.get(url, headers={"If-None-Match": etag}).status_code == 304
    assert client.get("/api/angle-scan/image/zzzz/bin_0.jpg").status_code == 400
    assert client.get("/api/angle-scan/image/0123456789abcdef/../x").status_code in (400, 404)


def test_busy_refuses_the_scan(client, monkeypatch):
    monkeypatch.setattr(ras, "is_busy", lambda: True)
    ev = _scan(client)[-1]
    assert ev["event"] == "error" and ev["code"] == "busy"


def test_a_render_starting_mid_scan_stops_it(client, monkeypatch):
    calls = {"n": 0}

    def busy_after_a_few():
        calls["n"] += 1
        return calls["n"] > 1          # free at start, busy from the first progress tick

    monkeypatch.setattr(ras, "is_busy", busy_after_a_few)
    ev = _scan(client)[-1]
    assert ev["event"] == "error" and ev["code"] == "busy" and "render started" in ev["message"]
    assert client.get("/api/angle-scan/session").json() == {"session": None}


def test_bad_target_is_reported(client, monkeypatch):
    def bad(payload):
        raise ValueError("select a target person first")
    monkeypatch.setattr(ras, "resolve_target", bad)
    ev = _scan(client)[-1]
    assert ev == {"event": "error", "code": "bad_request", "message": "select a target person first"}


def test_cancel_stops_the_scan(client):
    with client.websocket_connect("/ws/angle-scan") as ws:
        ws.send_json({"op": "start", "step_frames": 1})
        ws.send_json({"op": "cancel"})
        while True:
            ev = ws.receive_json()
            if ev["event"] in ("result", "error", "cancelled"):
                break
    # a tiny clip may finish before the cancel lands; either end is legal,
    # but it must END
    assert ev["event"] in ("cancelled", "result")


def test_thresholds_regate_without_rescanning(client):
    _scan(client)
    before = client.get("/api/angle-scan/session").json()["session"]
    strict = client.post("/api/angle-scan/thresholds", json={"min_iod": 900}).json()["session"]
    assert strict["thresholds"]["min_iod"] == 900
    assert strict["candidates"]["valid"] == 0 < before["candidates"]["valid"]
    assert all(b["status"] == "missing" for b in strict["bins"])
    loose = client.post("/api/angle-scan/thresholds", json={"min_iod": 10, "blur_frac": 0}).json()["session"]
    assert loose["candidates"]["valid"] == loose["candidates"]["evaluated"]
    assert client.post("/api/angle-scan/thresholds", json={"min_iod": "x"}).status_code == 422
    assert client.post("/api/angle-scan/thresholds", json={"blur_frac": 3}).status_code == 422


def test_override_pins_the_users_frame(client):
    _scan(client)
    r = client.post("/api/angle-scan/override", json={"bin": "BIN_7_PITCH_UP", "frame_idx": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    b7 = next(b for b in body["session"]["bins"] if b["bin"] == "BIN_7_PITCH_UP")
    assert b7["status"] == "override" and b7["frame_idx"] == 30 and b7["url"]
    assert b7["source_bin"] == "BIN_0_FRONTAL"        # frame 30 is a frontal head
    assert body["session"]["overrides"] == ["BIN_7_PITCH_UP"]
    assert body["faces_in_frame"] == 1 and body["similarity"] == pytest.approx(0.9, abs=1e-3)
    cleared = client.post("/api/angle-scan/override/clear", json={"bin": 7}).json()["session"]
    assert next(b for b in cleared["bins"] if b["bin"] == "BIN_7_PITCH_UP")["status"] == "missing"
    assert client.post("/api/angle-scan/override", json={"bin": "nope", "frame_idx": 1}).status_code == 422
    assert client.post("/api/angle-scan/override", json={"bin": 0, "frame_idx": 10_000}).status_code == 422


def test_override_warns_on_a_stranger(client, monkeypatch):
    _scan(client)
    monkeypatch.setattr(ras, "detect_faces", lambda frame: [_face_at(30, identity=3)])
    body = client.post("/api/angle-scan/override", json={"bin": 1, "frame_idx": 30}).json()
    assert "low_identity_similarity" in body["warnings"]


def test_endpoints_need_a_session(client):
    assert client.post("/api/angle-scan/thresholds", json={"min_iod": 40}).status_code == 404
    assert client.post("/api/angle-scan/override", json={"bin": 0, "frame_idx": 1}).status_code == 404
    assert client.post("/api/angle-scan/apply", json={}).status_code == 404
    assert client.get("/api/angle-scan/session").json() == {"session": None}


def test_apply_hands_every_filled_bin_to_the_bank(client, monkeypatch):
    sess = _scan(client)[-1]["session"]
    filled = [b["bin"] for b in sess["bins"] if b.get("frame_idx") is not None]
    r = client.post("/api/angle-scan/apply", json={})
    assert r.status_code == 200 and r.json()["added"] == len(filled)
    person, media_id, path, picks = client.applied[-1]
    assert (person, media_id) == ("p1", "m1") and path.endswith("clip.avi")
    assert sorted(p["bin"] for p in picks) == sorted(filled)
    assert all(len(p["kps"]) == 5 and len(p["bbox"]) == 4 for p in picks)
    only = client.post("/api/angle-scan/apply", json={"bins": ["BIN_0_FRONTAL"]}).json()
    assert only["added"] == 1
    monkeypatch.setattr(ras, "is_busy", lambda: True)
    assert client.post("/api/angle-scan/apply", json={}).status_code == 409


def test_source_portfolio_endpoints(client, monkeypatch):
    import types
    import roop.globals as g
    from roop import face_util as _fu

    def src_face(yaw, axis):
        pts = _fu._project_reference(yaw, 0) * 120 + 300
        return {"kps": pts.astype(np.float32), "embedding": _emb(axis) * 20, "det_score": 0.9}

    good = types.SimpleNamespace(faces=[src_face(0, 0), src_face(-60, 1), src_face(60, 2)])
    bad = types.SimpleNamespace(faces=[src_face(-60, 1)])
    sources = [good, bad]
    monkeypatch.setattr(ras, "get_source_faceset", lambda i: sources[i])
    monkeypatch.setattr(g, "ANGLE_FRAME_LUT", None, raising=False)

    _scan(client)                                    # a target scan publishes the frame LUT
    assert g.ANGLE_FRAME_LUT is not None and len(g.ANGLE_FRAME_LUT) > 0

    r = client.post("/api/angle-scan/source-portfolio", json={"source_index": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["portfolio"]["profile_left"] and body["portfolio"]["profile_right"]
    assert body["frame_lut"]["available"] and body["frame_lut"]["frames"] == len(g.ANGLE_FRAME_LUT)
    assert good.angle_portfolio is not None
    assert client.get("/api/angle-scan/source-portfolio?index=0").json()["portfolio"]["dim"] == 8

    assert client.post("/api/angle-scan/source-portfolio", json={"source_index": 1}).status_code == 422
    assert getattr(bad, "angle_portfolio", "unset") is None
    assert client.post("/api/angle-scan/source-portfolio", json={"source_index": -1}).status_code == 404
    assert client.post("/api/angle-scan/source-portfolio", json={"source_index": 5}).status_code == 404

    cleared = client.post("/api/angle-scan/source-portfolio/clear", json={"source_index": 0}).json()
    assert cleared["portfolio"] is None and good.angle_portfolio is None


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
