"""Contract for roop/angle_portfolio.py: the 9-bin lattice, selection, fusion, export."""
import base64
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_util as fu  # noqa: E402
from roop import angle_portfolio as ap  # noqa: E402
from roop.angle_portfolio import AngleBin as B  # noqa: E402
from roop.pose_quality import CandidateFaceMetric  # noqa: E402


# ── lattice ──────────────────────────────────────────────────────────────────
# (yaw, pitch_up) -> bin; pitch_up is the bins' up-positive pitch (= -pitch).
EDGES = [
    ((0, 0), B.BIN_0_FRONTAL), ((-10, 10), B.BIN_0_FRONTAL), ((10, -10), B.BIN_0_FRONTAL),
    ((-10.01, 0), B.BIN_1_QUARTER_LEFT), ((-25, 0), B.BIN_1_QUARTER_LEFT),
    ((10.01, 0), B.BIN_2_QUARTER_RIGHT), ((25, 0), B.BIN_2_QUARTER_RIGHT),
    ((-25.01, 0), B.BIN_3_HALF_PROFILE_LEFT), ((-45, 14), B.BIN_3_HALF_PROFILE_LEFT),
    ((25.01, 0), B.BIN_4_HALF_PROFILE_RIGHT), ((45, -15), B.BIN_4_HALF_PROFILE_RIGHT),
    ((-45.01, 0), B.BIN_5_PROFILE_LEFT), ((-80, 40), B.BIN_5_PROFILE_LEFT),
    ((45.01, 0), B.BIN_6_PROFILE_RIGHT), ((70, -30), B.BIN_6_PROFILE_RIGHT),
    ((0, 15.01), B.BIN_7_PITCH_UP), ((-20, 40), B.BIN_7_PITCH_UP),
    ((0, -15.01), B.BIN_8_PITCH_DOWN), ((20, -30), B.BIN_8_PITCH_DOWN),
    # the lattice's gaps
    ((0, 12), None), ((-15, -12), None), ((30, 20), None), ((-22, -18), None),
]


@pytest.mark.parametrize("pose,expected", EDGES)
def test_lattice_edges(pose, expected):
    yaw, up = pose
    assert ap.angle_bin(yaw, -up) == expected


def test_bins_follow_the_physical_head_direction():
    """A head whose nose tip rises toward the eye line is looking UP.

    solve_pose_5pt reports that head with NEGATIVE pitch; the bins must still
    call it PITCH_UP. Anchored on the picture, not on either sign convention."""
    frontal = fu._project_reference(0, 0)
    for pitch_arg in (-25, 25):
        pts = fu._project_reference(0, pitch_arg)
        rising = fu.kps_pose_ratios(pts)[1] < fu.kps_pose_ratios(frontal)[1]  # nose nearer the eyes
        yaw, pitch, _roll = fu.solve_pose_5pt(pts)
        expected = B.BIN_7_PITCH_UP if rising else B.BIN_8_PITCH_DOWN
        assert ap.angle_bin(yaw, pitch) == expected
    # and +yaw is the face turned toward the viewer's right (nose right of centre)
    pts = fu._project_reference(35, 0)
    assert pts[2, 0] > (pts[0, 0] + pts[1, 0]) / 2
    assert ap.angle_bin(*fu.solve_pose_5pt(pts)[:2]) == B.BIN_4_HALF_PROFILE_RIGHT


def test_distance_to_bin():
    assert ap.distance_to_bin(0, 0, B.BIN_0_FRONTAL) == 0
    assert ap.distance_to_bin(-30, 0, B.BIN_1_QUARTER_LEFT) == pytest.approx(5)
    assert ap.distance_to_bin(0, -12, B.BIN_0_FRONTAL) == pytest.approx(2)   # pitch_up 12
    assert ap.distance_to_bin(28, -19, B.BIN_4_HALF_PROFILE_RIGHT) == pytest.approx(4)


# ── selection ────────────────────────────────────────────────────────────────
def _m(frame_idx, yaw, pitch_up_deg, score, valid=True, sim=0.8, track=1, kps=None):
    return CandidateFaceMetric(frame_idx=frame_idx, track_id=track, yaw=yaw, pitch=None if pitch_up_deg is None else -pitch_up_deg,
                               roll=0.0, composite_score=score, is_valid=valid,
                               id_similarity=sim, kps=kps, bbox=[0, 0, 10, 10])


def test_best_per_bin_and_invalid_ignored():
    metrics = [_m(1, 0, 0, 0.5), _m(2, 2, 1, 0.9), _m(3, 1, 0, 0.99, valid=False),
               _m(4, -60, 0, 0.3), _m(5, None, None, 0.9)]
    sel = ap.AnglePortfolioSelector()
    pf = sel.select_portfolio(metrics)
    assert pf[B.BIN_0_FRONTAL].frame_idx == 2
    assert pf[B.BIN_5_PROFILE_LEFT].frame_idx == 4
    assert sel.report[B.BIN_0_FRONTAL].status == "selected" and sel.report[B.BIN_0_FRONTAL].candidates == 2
    assert sel.report[B.BIN_7_PITCH_UP].status == "missing" and B.BIN_7_PITCH_UP not in pf
    assert list(pf) == sorted(pf)


def test_tie_breaks_on_similarity():
    pf = ap.AnglePortfolioSelector().select_portfolio([_m(1, 0, 0, 0.7, sim=0.6), _m(2, 0, 0, 0.7, sim=0.9)])
    assert pf[B.BIN_0_FRONTAL].frame_idx == 2


def test_nearest_neighbour_fill_and_limits():
    metrics = [
        _m(1, 0, 0, 0.9),          # frontal winner
        _m(2, 2, 0, 0.8),          # frontal runner-up: may fill a neighbour
        _m(3, 0, 18, 0.6),         # up winner (pitch_up 18)
        _m(4, 30, 21, 0.7),        # gap: 6 deg from BIN_4, 10 from BIN_7
        _m(5, -12, 30, 0.5),       # up runner-up: 20 deg from the empty BIN_1, too far
    ]
    # Level straight-on frames that fail the quality gates still define the
    # person's neutral pitch (0 here), so the geometry below is absolute.
    metrics += [_m(100 + i, 0, 0, 0.1, valid=False) for i in range(6)]
    sel = ap.AnglePortfolioSelector(max_fill_deg=10)
    pf = sel.select_portfolio(metrics)
    # BIN_4 borrows the gap candidate, flagged with its distance and no source bin
    assert pf[B.BIN_4_HALF_PROFILE_RIGHT].frame_idx == 4
    slot = sel.report[B.BIN_4_HALF_PROFILE_RIGHT]
    assert slot.status == "nearest" and slot.distance_deg == pytest.approx(6.0) and slot.source_bin is None
    # BIN_2 (10..25 yaw) borrows the frontal runner-up (8 deg away), never the winner
    assert pf[B.BIN_2_QUARTER_RIGHT].frame_idx == 2
    assert sel.report[B.BIN_2_QUARTER_RIGHT].source_bin == "BIN_0_FRONTAL"
    # no frame appears twice
    ids = [m.frame_idx for m in pf.values()]
    assert len(ids) == len(set(ids))
    # far bins stay missing
    assert sel.report[B.BIN_6_PROFILE_RIGHT].status == "missing"


def test_fill_prefers_the_closer_bin():
    # 3 deg from BIN_3 and 7 deg from BIN_5 -> goes to BIN_3
    pf = ap.AnglePortfolioSelector().select_portfolio([_m(1, -38, 18, 0.5)])
    assert list(pf) == [B.BIN_3_HALF_PROFILE_LEFT]


def test_pitch_is_relative_to_the_persons_neutral():
    """A level, straight-on person whose anatomy reads 19 deg 'up' (the
    benchmark's passport photo does) must still fill FRONTAL, not PITCH_UP."""
    metrics = [_m(i, yaw, 19, 0.8) for i, yaw in enumerate((0, 2, -3, -15, 18))]
    metrics.append(_m(10, 0, 19 + 25, 0.7))                 # genuinely looking up from there
    sel = ap.AnglePortfolioSelector()
    pf = sel.select_portfolio(metrics)
    assert sel.neutral[1] == pytest.approx(-19)             # server convention: pitch = -pitch_up
    assert pf[B.BIN_0_FRONTAL].frame_idx in (0, 1, 2)
    assert pf[B.BIN_1_QUARTER_LEFT].frame_idx == 3 and pf[B.BIN_2_QUARTER_RIGHT].frame_idx == 4
    assert pf[B.BIN_7_PITCH_UP].frame_idx == 10
    assert sel.relative_pitch(pf[B.BIN_7_PITCH_UP]) == pytest.approx(-25)


def test_neutral_pitch_uses_only_near_frontal_faces():
    assert ap.neutral_pitch([(0, 10), (5, 12), (60, -40), (-70, 50)]) == pytest.approx(11)
    assert ap.neutral_pitch([(60, -40)]) == 0.0 and ap.neutral_pitch([]) == 0.0


# ── fusion ───────────────────────────────────────────────────────────────────
def _unit(axis, n=8):
    v = np.zeros(n, np.float32)
    v[axis] = 1
    return v


def test_fused_embedding_is_weighted_normalised_and_frontal_only():
    pf = {B.BIN_0_FRONTAL: _m(1, 0, 0, 0.9), B.BIN_1_QUARTER_LEFT: _m(2, -15, 0, 0.3),
          B.BIN_5_PROFILE_LEFT: _m(3, -70, 0, 1.0)}
    embs = {1: _unit(0) * 20.0, 2: _unit(1) * 3.0, 3: _unit(2)}
    fused = ap.synthesize_fused_embedding(pf, embs)
    assert np.linalg.norm(fused) == pytest.approx(1.0, abs=1e-6)
    assert fused[2] == 0                                        # profile excluded
    # weights are the scores, not the raw vector norms: 0.9 : 0.3
    assert fused[0] / fused[1] == pytest.approx(3.0, rel=1e-5)


def test_fused_embedding_errors():
    with pytest.raises(ValueError):
        ap.synthesize_fused_embedding({B.BIN_5_PROFILE_LEFT: _m(3, -70, 0, 1.0)}, {3: _unit(0)})
    with pytest.raises(ValueError):
        ap.synthesize_fused_embedding({B.BIN_0_FRONTAL: _m(1, 0, 0, 0.9)}, {})
    # zero scores fall back to equal weights
    pf = {B.BIN_0_FRONTAL: _m(1, 0, 0, 0.0), B.BIN_2_QUARTER_RIGHT: _m(2, 15, 0, 0.0)}
    fused = ap.synthesize_fused_embedding(pf, {1: _unit(0), 2: _unit(1)})
    assert fused[0] == pytest.approx(fused[1])


# ── export / payload ─────────────────────────────────────────────────────────
VW, VH, N = 640, 480, 12


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("ap") / "clip.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (VW, VH))
    rng = np.random.default_rng(0)
    for _ in range(N):
        w.write(rng.integers(0, 255, (VH, VW, 3), dtype=np.uint8))
    w.release()
    return path


def _kps(yaw, pitch):
    pts = fu._project_reference(yaw, pitch)
    pts = (pts - pts.mean(axis=0)) * 120 + (VW / 2, VH / 2)
    return pts.tolist()


def _metrics():
    out = []
    for i, (yaw, up, score) in enumerate([(0, 0, 0.9), (-15, 0, 0.7), (18, 0, 0.6), (-60, 0, 0.5), (0, 20, 0.4)]):
        out.append(_m(i * 2, yaw, up, score, kps=_kps(yaw, -up)))
    out.append(_m(N + 20, 30, 0, 0.8, kps=_kps(30, 0)))       # past the end of the clip
    return out


def test_payload_exports_fuses_and_caches(clip, tmp_path):
    calls = []

    def embed(frame, kps):
        calls.append(frame.shape)
        return _unit(len(calls) - 1) * 5

    payload = ap.build_target_angle_payload(clip, _metrics(), embed_fn=embed, cache_root=str(tmp_path))
    assert payload["cached"] is False
    bins = {e["bin"]: e for e in payload["bins"]}
    assert len(bins) == 9
    assert payload["coverage"] == {"selected": 6, "nearest": 0, "missing": 3, "total": 9}
    front = bins["BIN_0_FRONTAL"]
    assert front["status"] == "selected" and front["frame_idx"] == 0 and front["file"] == "bin_0.jpg"
    assert front["image"].startswith("data:image/jpeg;base64,")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(front["image"].split(",", 1)[1]), np.uint8), 1)
    assert img.shape == (512, 512, 3)
    assert os.path.isfile(os.path.join(payload["cache_dir"], "bin_0.jpg"))
    assert bins["BIN_7_PITCH_UP"]["pitch_up"] == pytest.approx(20)
    assert bins["BIN_4_HALF_PROFILE_RIGHT"]["file"] is None
    assert bins["BIN_4_HALF_PROFILE_RIGHT"]["export_error"] == "frame not decodable"
    assert bins["BIN_6_PROFILE_RIGHT"]["status"] == "missing" and "image" not in bins["BIN_6_PROFILE_RIGHT"]

    fe = payload["fused_embedding"]
    assert len(calls) == 3 and fe["available"] and fe["dim"] == 8
    assert fe["sources"] == ["BIN_0_FRONTAL", "BIN_1_QUARTER_LEFT", "BIN_2_QUARTER_RIGHT"]
    fused = np.load(os.path.join(payload["cache_dir"], "fused_embedding.npy"))
    assert np.linalg.norm(fused) == pytest.approx(1.0, abs=1e-6)

    again = ap.build_target_angle_payload(clip, _metrics(), embed_fn=embed, cache_root=str(tmp_path))
    assert again["cached"] is True and len(calls) == 3
    assert again["cache_key"] == payload["cache_key"]
    assert {e["bin"]: e.get("image") for e in again["bins"]} == {e["bin"]: e.get("image") for e in payload["bins"]}


def test_payload_without_embeddings_reports_why(clip, tmp_path):
    payload = ap.build_target_angle_payload(clip, _metrics(), embed_fn=lambda f, k: None,
                                            cache_root=str(tmp_path))
    fe = payload["fused_embedding"]
    assert not fe["available"] and fe["embed_failures"] == 3 and "no frontal" in fe["error"]


def test_app_embedding_uses_the_recognition_model(monkeypatch):
    import contextlib

    class Rec:
        def get(self, img, face):
            self.kps = face.kps
            return np.arange(4, dtype=np.float32)

    class FA:
        models = {"recognition": Rec()}

    monkeypatch.setattr(fu, "lease_face_analyser", lambda: contextlib.nullcontext(FA()))
    out = ap.app_embedding(np.zeros((8, 8, 3), np.uint8), _kps(0, 0))
    assert out.tolist() == [0, 1, 2, 3]


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
