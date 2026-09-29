"""Contract for roop/source_portfolio.py and its render hooks in ProcessMgr:
the SOURCE angle portfolio, the target frame LUT, routing, far-eye damping."""
import os
import sys
import types

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_util as fu  # noqa: E402
from roop import source_portfolio as sp  # noqa: E402
from roop.angle_portfolio import AngleBin as B  # noqa: E402


def _unit(axis, n=8):
    v = np.zeros(n, np.float32)
    v[axis] = 1
    return v


def _kps(yaw, pitch=0.0, centre=(400, 300), scale=120):
    pts = fu._project_reference(yaw, pitch)
    return ((pts - pts.mean(axis=0)) * scale + centre).astype(np.float32)


def _box(kps, pad=60):
    x0, y0 = kps.min(axis=0) - pad
    x1, y1 = kps.max(axis=0) + pad
    return [float(x0), float(y0), float(x1), float(y1)]


def _face(yaw, axis, det=0.9, norm=20.0, lm68=None, pitch=0.0):
    f = {"kps": _kps(yaw, pitch), "embedding": _unit(axis) * norm, "det_score": det}
    if lm68 is not None:
        f["landmark_3d_68"] = lm68
    return f


def _faceset(faces):
    return types.SimpleNamespace(faces=faces)


def _closed_eyes_68():
    pts = np.zeros((68, 3), np.float32)
    for base, x0 in ((36, 300), (42, 360)):
        pts[base:base + 6, :2] = [(x0, 200), (x0 + 10, 199), (x0 + 20, 199), (x0 + 30, 200),
                                   (x0 + 20, 201), (x0 + 10, 201)]
    return pts


# ── source portfolio ─────────────────────────────────────────────────────────
def test_portfolio_bins_each_source_face_and_keeps_the_best():
    fs = _faceset([_face(0, 0, det=0.6), _face(2, 1, det=0.95), _face(-15, 2), _face(18, 3),
                   _face(-60, 4), _face(62, 5)])
    pf = sp.build_source_portfolio(fs)
    assert pf is not None and pf.dim == 8
    assert pf.refs[B.BIN_0_FRONTAL].face_index == 1                     # higher score wins
    assert {B.BIN_1_QUARTER_LEFT, B.BIN_2_QUARTER_RIGHT, B.BIN_5_PROFILE_LEFT,
            B.BIN_6_PROFILE_RIGHT} <= set(pf.refs)
    # fused = frontal + quarters only, unit length, no profile in it
    assert np.linalg.norm(pf.fused) == pytest.approx(1.0, abs=1e-6)
    assert pf.fused[4] == 0 and pf.fused[5] == 0 and pf.fused[1] > 0 and pf.fused[2] > 0
    s = pf.summary()
    assert s["profile_left"] and s["profile_right"] and s["fused_sources"][0] == "BIN_0_FRONTAL"


def test_portfolio_rejections():
    fs = _faceset([_face(0, 0), _face(0, 1, lm68=_closed_eyes_68()), {"kps": _kps(0)},
                   {"kps": None, "embedding": _unit(2) * 20}, _face(0, 3, pitch=-12)])
    pf = sp.build_source_portfolio(fs)
    assert pf.rejected == {"eyes_closed": 1, "no_embedding": 1, "no_pose": 1, "between_bins": 1}
    assert pf.refs[B.BIN_0_FRONTAL].face_index == 0


def test_a_level_photo_that_reads_pitched_is_still_frontal():
    """The benchmark's anshita.png is level and straight-on but solves to pitch
    ~19 deg up; a portfolio of one such face must be a FRONTAL portfolio."""
    fs = _faceset([_face(0, 0, pitch=-19)])                  # _project_reference: -pitch = up
    pf = sp.build_source_portfolio(fs)
    assert pf is not None and set(pf.refs) == {B.BIN_0_FRONTAL}
    assert pf.pitch_neutral == pytest.approx(-19, abs=0.5)
    assert pf.summary()["pitch_neutral"] == pytest.approx(-19, abs=0.5)


def test_face0_reads_its_own_vector_not_the_faceset_average():
    """FaceSet.AverageEmbeddings puts the MEAN in faces[0].embedding (V1 sets)."""
    faces = [_face(-60, 4), _face(0, 0), _face(60, 5)]
    fs = _faceset(faces)
    fs.embeddings_backup = faces[0]["embedding"].copy()
    faces[0]["embedding"] = np.mean([f["embedding"] for f in faces], axis=0)
    pf = sp.build_source_portfolio(fs)
    np.testing.assert_allclose(pf.refs[B.BIN_5_PROFILE_LEFT].embedding, _unit(4))


def test_no_frontal_or_quarter_means_no_portfolio():
    assert sp.build_source_portfolio(_faceset([_face(-60, 0), _face(60, 1)])) is None
    assert sp.build_source_portfolio(_faceset([])) is None


# ── routing ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def pf():
    return sp.build_source_portfolio(_faceset([_face(0, 0), _face(-15, 1), _face(15, 2),
                                               _face(-60, 4), _face(60, 5)]))


@pytest.mark.parametrize("yaw,profile_axis", [(55, 5), (-55, 4)])
def test_profile_frame_blends_the_matching_profile(pf, yaw, profile_axis):
    k = _kps(yaw)
    r = sp.route(pf, k, _box(k))
    expected = 0.7 * pf.refs[B.BIN_6_PROFILE_RIGHT if yaw > 0 else B.BIN_5_PROFILE_LEFT].embedding + 0.3 * pf.fused
    expected /= np.linalg.norm(expected)
    assert r.kind == "profile" and r.pose_from == "live"
    np.testing.assert_allclose(r.embedding, expected, atol=1e-6)
    assert r.embedding[profile_axis] > 0.6
    assert r.far_eye == sp.far_eye_index(k)


@pytest.mark.parametrize("yaw", [0, 20, -30, 34])
def test_frontal_and_oblique_frames_use_the_fused_vector(pf, yaw):
    k = _kps(yaw)
    r = sp.route(pf, k, _box(k))
    assert r.kind == "fused" and np.array_equal(r.embedding, pf.fused)
    assert (r.far_eye is None) == (abs(yaw) <= 35)


def test_profile_without_that_profile_falls_back_to_fused():
    one_sided = sp.build_source_portfolio(_faceset([_face(0, 0), _face(-60, 4)]))
    k = _kps(60)
    r = sp.route(one_sided, k, _box(k))
    assert r.kind == "fused" and r.far_eye is not None


def test_embedding_shape_never_changes(pf):
    shapes = {sp.route(pf, _kps(y), _box(_kps(y))).embedding.shape for y in range(-80, 81, 5)}
    assert shapes == {(8,)}


# ── frame LUT ────────────────────────────────────────────────────────────────
def _scan():
    tracks = [{"track_id": 0, "detections": []}]
    for i, yaw in ((0, 0), (3, 50), (6, -50)):
        k = _kps(yaw, centre=(960, 540), scale=150)
        tracks[0]["detections"].append({"frame_idx": i, "kps": k.tolist(), "bbox": _box(k)})
    return {"video_path": "clip.mp4", "frame_shape": [1080, 1920], "step_frames": 3, "tracks": tracks}


def test_lut_built_from_a_scan_and_read_in_o1(pf):
    lut = sp.build_frame_lut(_scan())
    assert len(lut) == 3 and lut.step == 3 and lut.media_path == "clip.mp4"
    e = lut.entries[3][0]
    assert e.yaw == pytest.approx(50, abs=1.0) and e.bin == B.BIN_6_PROFILE_RIGHT
    k = np.asarray(_scan()["tracks"][0]["detections"][1]["kps"], np.float32)
    frontal_kps = _kps(0, centre=(960, 540), scale=150)
    # the LUT's pose wins over the keypoints handed in (proves the O(1) read ran)
    r = sp.route(pf, frontal_kps, _box(k), frame_idx=3, lut=lut)
    assert r.pose_from == "lut" and r.kind == "profile" and r.yaw == pytest.approx(50, abs=1.0)
    # an unscanned frame between two scans takes the nearest one
    assert sp.route(pf, frontal_kps, _box(k), frame_idx=4, lut=lut).pose_from == "near"


def test_lut_row_is_only_for_the_face_it_measured(pf):
    lut = sp.build_frame_lut(_scan())
    other = _kps(0, centre=(300, 300), scale=100)                    # a second person
    r = sp.route(pf, other, _box(other), frame_idx=3, lut=lut)
    assert r.pose_from == "live" and r.kind == "fused"
    far = _kps(0, centre=(960, 540), scale=150)
    assert sp.route(pf, far, _box(far), frame_idx=100, lut=lut).pose_from == "live"


# ── far-eye damping ──────────────────────────────────────────────────────────
def test_far_eye_is_the_one_nearer_the_nose():
    assert sp.far_eye_index(_kps(60)) == 1 and sp.far_eye_index(_kps(-60)) == 0


def test_damping_pulls_only_the_far_eye_25_percent_back():
    enhanced = np.full((256, 256, 3), 200, np.uint8)
    pre = np.full((128, 128, 3), 100, np.uint8)                       # enhancer upscaled x2
    k = np.array([[80, 100], [170, 100], [150, 140], [90, 180], [160, 180]], np.float32)
    out = sp.damp_far_eye(enhanced, pre, k, eye=1)
    assert out[100, 170].tolist() == pytest.approx([175, 175, 175], abs=1)   # 200 - 0.25 * 100
    assert out[100, 80].tolist() == [200, 200, 200]                   # the near eye untouched
    assert out[230, 20].tolist() == [200, 200, 200]


# ── ProcessMgr hooks ─────────────────────────────────────────────────────────
def test_with_unit_embedding_keeps_norm_and_drops_latents():
    from insightface.app.common import Face
    from roop.ProcessMgr import _with_unit_embedding
    f = Face(embedding=_unit(0) * 23.0, kps=_kps(0), _latent_hyper=np.ones(3))
    out = _with_unit_embedding(f, _unit(3))
    assert np.linalg.norm(out["embedding"]) == pytest.approx(23.0, rel=1e-6)
    np.testing.assert_allclose(out.normed_embedding, _unit(3))
    assert "_latent_hyper" not in out and "_latent_hyper" in f          # source Face untouched
    assert np.array_equal(f["embedding"], _unit(0) * 23.0)


def test_route_source_uses_the_trim_offset_and_the_render_target(pf, monkeypatch):
    import roop.globals as g
    from roop.ProcessMgr import ProcessMgr
    lut = sp.build_frame_lut(_scan())
    monkeypatch.setattr(g, "ANGLE_FRAME_LUT", lut, raising=False)
    monkeypatch.setattr(g, "target_path", "clip.mp4", raising=False)
    k = np.asarray(_scan()["tracks"][0]["detections"][1]["kps"], np.float32)
    face = types.SimpleNamespace(kps=_kps(0, centre=(960, 540), scale=150), bbox=np.asarray(_box(k)))
    stub = types.SimpleNamespace(_tls=types.SimpleNamespace(frame_idx=1), _angle_frame_offset=2)
    r = ProcessMgr._route_source(stub, pf, face)
    assert r.pose_from == "lut" and r.kind == "profile"                 # 2 + 1 = absolute frame 3
    stub._angle_frame_offset = 0
    assert ProcessMgr._route_source(stub, pf, face).pose_from != "lut"
    monkeypatch.setattr(g, "target_path", "another.mp4", raising=False)
    stub._angle_frame_offset = 2
    assert ProcessMgr._route_source(stub, pf, face).pose_from == "live"  # LUT is for another clip


def test_route_stats_count_what_ran():
    st = sp.RouteStats()
    st.add("routed", 50_000)
    st.add("profile")
    snap = st.snapshot()
    assert snap["routed"] == 1 and snap["profile"] == 1 and snap["mean_route_ms"] == pytest.approx(0.05)


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
