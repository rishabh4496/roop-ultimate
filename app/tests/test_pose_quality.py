"""Contract for roop/pose_quality.py: perspective pose, quality factors, batch API."""
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_util as fu  # noqa: E402
from roop import pose_quality as pq  # noqa: E402

H, W = 1080, 1920
_F = np.diag([1.0, -1.0, -1.0])
_REF = np.asarray(fu._reference_5pt(), float)
_X = _REF - _REF.mean(axis=0)
_IOD = float(np.linalg.norm(_REF[1] - _REF[0]))


def _rot(y, p, r):
    y, p, r = map(math.radians, (y, p, r))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def _project(yaw, pitch, roll, frontal_iod_px, shape=(H, W), centre=None):
    """The reference head at a known pose, perspective-projected so that a
    frontal face would show `frontal_iod_px` between the eyes."""
    h, w = shape
    f = max(h, w)
    cam = (_F @ _rot(yaw, pitch, roll) @ _X.T).T
    cam[:, 2] += f * _IOD / frontal_iod_px
    cx, cy = centre if centre is not None else (w / 2, h / 2)
    return np.stack([f * cam[:, 0] / cam[:, 2] + cx, f * cam[:, 1] / cam[:, 2] + cy], axis=1)


@pytest.mark.parametrize("yaw", [-70, -30, 0, 25, 60])
@pytest.mark.parametrize("pitch", [-25, 0, 20])
@pytest.mark.parametrize("roll", [-30, 0, 15])
def test_pose_recovers_ground_truth_and_agrees_with_weak_perspective(yaw, pitch, roll):
    pts = _project(yaw, pitch, roll, 120)
    got = pq.estimate_head_pose(pts, (H, W))
    assert got == pytest.approx((yaw, pitch, roll), abs=0.05)
    # Same sign convention as the solver the render uses: the two must never
    # disagree about which way a head faces.
    wp = fu.solve_pose_5pt(pts)
    assert np.max(np.abs(np.subtract(got, wp))) < 3.5


def test_pose_accepts_numpy_frame_shape_with_channels():
    pts = _project(30, 10, 5, 100)
    assert pq.estimate_head_pose(pts, (H, W, 3)) == pytest.approx((30, 10, 5), abs=0.05)


@pytest.mark.parametrize("bad", [np.zeros((5, 2)), np.ones((4, 2)), np.full((5, 2), np.nan)])
def test_degenerate_landmarks_read_as_unknown(bad):
    assert all(math.isnan(v) for v in pq.estimate_head_pose(bad, (H, W)))


def _texture(side=200, seed=0, level=128, spread=60):
    rng = np.random.default_rng(seed)
    img = rng.normal(level, spread, (side, side, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_profile_is_not_rejected_as_small():
    """Raw eye distance collapses on a profile; the gate reads frontal-equivalent."""
    pts = _project(75, 0, 0, 100)
    raw = float(np.linalg.norm(pts[1] - pts[0]))
    assert raw < pq.MIN_IOD_PX
    yaw, pitch, roll, frontal = pq._solve(pts, (H, W))
    assert frontal == pytest.approx(100, rel=0.02)
    q, bd = pq.compute_quality_score(_texture(), pts, 0.9, 0.8, roll=roll, frontal_iod=frontal)
    assert "too_small" not in pq.reject_reasons(bd)
    assert bd["iod_px"] == pytest.approx(raw)
    # and a genuinely small frontal face IS rejected
    small = _project(0, 0, 0, 30)
    _, bd_small = pq.compute_quality_score(_texture(), small, 0.9, 0.8)
    assert "too_small" in pq.reject_reasons(bd_small)


def test_quality_factors_move_the_score():
    kps = _project(0, 0, 0, 120)
    sharp = _texture()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    q_sharp, bd_sharp = pq.compute_quality_score(sharp, kps, 0.9, 0.8)
    q_blur, bd_blur = pq.compute_quality_score(blurred, kps, 0.9, 0.8)
    assert bd_sharp["sharpness"] > 10 * bd_blur["sharpness"] and q_sharp > q_blur

    dark = _texture(level=15, spread=5)
    _, bd_dark = pq.compute_quality_score(dark, kps, 0.9, 0.8)
    assert "too_dark" in pq.reject_reasons(bd_dark)

    blown = np.full((200, 200, 3), 255, np.uint8)
    _, bd_blown = pq.compute_quality_score(blown, kps, 0.9, 0.8)
    assert {"too_bright", "clipped"} <= set(pq.reject_reasons(bd_blown))
    assert bd_blown["norm_illum"] == 0.0


def test_roll_penalty_and_missing_identity():
    crop = _texture()
    kps = _project(0, 0, 0, 120)
    q0, _ = pq.compute_quality_score(crop, kps, 0.9, 0.8, roll=0.0, frontal_iod=120)
    q20, _ = pq.compute_quality_score(crop, kps, 0.9, 0.8, roll=20.0, frontal_iod=120)
    assert q0 - q20 == pytest.approx(0.20, abs=1e-6)
    # None drops the id term and renormalises; it is not scored as "wrong person"
    q_none, bd = pq.compute_quality_score(crop, kps, 0.9, None, roll=0.0, frontal_iod=120)
    q_zero, _ = pq.compute_quality_score(crop, kps, 0.9, 0.0, roll=0.0, frontal_iod=120)
    assert q_none > q_zero and math.isnan(bd["id_similarity"])
    assert 0.0 <= q_none <= 1.0


# ── batch path ───────────────────────────────────────────────────────────────
VW, VH, N = 640, 480, 20
BLURRED, DARK = 5, 7
BOX = (220, 140, 420, 340)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("pq") / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (VW, VH))
    for i in range(N):
        frame = np.full((VH, VW, 3), 90, np.uint8)
        patch = _texture(seed=i)
        if i == BLURRED:
            patch = cv2.GaussianBlur(patch, (0, 0), 5)
        frame[BOX[1]:BOX[3], BOX[0]:BOX[2]] = patch
        if i == DARK:
            frame = (frame * 0.08).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _cands():
    kps = _project(20, 0, 0, 90, shape=(VH, VW), centre=(320, 240)).tolist()
    out = [{"frame_idx": i, "track_id": 3, "bbox": list(BOX), "kps": kps,
            "det_score": 0.9, "similarity": 0.8} for i in range(N)]
    out.append({"frame_idx": N + 50, "bbox": list(BOX), "kps": kps, "det_score": 0.5, "similarity": None})
    return out[::-1]   # order must be preserved, not re-sorted


def test_batch_scores_every_candidate_in_order(clip):
    cands = _cands()
    res = pq.evaluate_candidate_frames(clip, cands, landmarks_fn=None)
    assert [m.frame_idx for m in res] == [c["frame_idx"] for c in cands]
    assert all(isinstance(m, pq.CandidateFaceMetric) for m in res)
    by = {m.frame_idx: m for m in res}
    assert by[N + 50].reject_reasons == ["unreadable_frame"] and not by[N + 50].is_valid
    assert "blurred" in by[BLURRED].reject_reasons
    assert "too_dark" in by[DARK].reject_reasons
    good = [by[i] for i in range(N) if i not in (BLURRED, DARK)]
    assert all(m.is_valid for m in good), [m.reject_reasons for m in good if not m.is_valid]
    assert all(m.yaw == pytest.approx(20, abs=0.1) for m in good)
    assert all(m.track_id == 3 and m.composite_score > 0 for m in good)
    assert by[BLURRED].composite_score < min(m.composite_score for m in good)


def test_batch_empty_and_json_roundtrip(clip):
    assert pq.evaluate_candidate_frames(clip, []) == []
    m = pq.evaluate_candidate_frames(clip, _cands()[-1:], landmarks_fn=None)[0]
    assert pq.CandidateFaceMetric.model_validate_json(m.model_dump_json()) == m


# ── eye openness ─────────────────────────────────────────────────────────────
def _eye(x0, y0, opening, width=30.0):
    """Six points p36..p41 order: corner, 2 upper, corner, 2 lower. EAR = opening / width."""
    h = opening / 2.0
    return [(x0, y0), (x0 + 10, y0 - h), (x0 + 20, y0 - h), (x0 + width, y0),
            (x0 + 20, y0 + h), (x0 + 10, y0 + h)]


def _lm68(first_open, second_open, width=30.0):
    pts = np.zeros((68, 2), np.float64)
    pts[36:42] = _eye(300, 200, first_open * width, width)
    pts[42:48] = _eye(360, 200, second_open * width, width)
    return pts


def test_ear_formula_matches_the_brief_and_eyelid_preserver():
    from roop.eyelid_preserver import calculate_ear
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 100, (68, 3))          # (x, y, z) as landmark_3d_68 returns
    p = pts[:, :2]
    brief = (np.linalg.norm(p[37] - p[41]) + np.linalg.norm(p[38] - p[40])) / (2 * np.linalg.norm(p[36] - p[39]))
    first, second = pq.eye_aspect_ratios(pts)
    assert first == pytest.approx(brief)
    assert second == pytest.approx(calculate_ear(p[42:48]), rel=1e-5)
    assert pq.eye_aspect_ratios(_lm68(0.30, 0.10)) == pytest.approx((0.30, 0.10))


@pytest.mark.parametrize("bad", [None, np.zeros((5, 2)), np.full((68, 2), np.nan), np.zeros((68, 2))])
def test_unmeasurable_eyes_are_not_read_as_open(bad):
    assert pq.eye_aspect_ratios(bad) is None and pq.eyes_open(bad) is None


def test_gate_reads_the_more_open_eye():
    assert pq.eyes_open(_lm68(0.12, 0.10)) is False       # blink: both closed
    assert pq.eyes_open(_lm68(0.28, 0.08)) is True        # far eye of a turned head
    assert pq.eyes_open(_lm68(0.19, 0.19)) is False       # just under 0.20
    assert pq.eyes_open(_lm68(0.20, 0.05)) is True


def test_closed_eyes_discarded_before_sharpness_and_illumination(clip, monkeypatch):
    calls = []
    real_sharp, real_illum = pq.sharpness, pq.illumination
    monkeypatch.setattr(pq, "sharpness", lambda c: calls.append("s") or real_sharp(c))
    monkeypatch.setattr(pq, "illumination", lambda c: calls.append("i") or real_illum(c))
    closed = dict(_cands()[-2], landmarks_68=_lm68(0.10, 0.12).tolist())    # frame 0
    m = pq.evaluate_candidate_frames(clip, [closed], landmarks_fn=None)[0]
    assert m.reject_reasons == ["eyes_closed"] and not m.is_valid
    assert m.ear == pytest.approx(0.12) and m.ear_left == pytest.approx(0.10)
    assert m.sharpness == 0.0 and m.composite_score == 0.0 and calls == []
    assert m.yaw == pytest.approx(20, abs=0.1)            # pose still reported

    opened = dict(closed, landmarks_68=_lm68(0.31, 0.29).tolist())
    m = pq.evaluate_candidate_frames(clip, [opened], landmarks_fn=None)[0]
    assert m.is_valid and m.ear == pytest.approx(0.31) and calls == ["s", "i"]


def test_landmarks_fn_used_and_failures_leave_ear_unmeasured(clip, caplog):
    seen = []

    def fn(frame, bbox, kps):
        seen.append(frame.shape)
        return _lm68(0.05, 0.05) if len(seen) == 1 else None

    cands = [c for c in _cands() if c["frame_idx"] in (1, 2, 3)]
    res = pq.evaluate_candidate_frames(clip, cands, landmarks_fn=fn)
    assert len(seen) == 3 and seen[0] == (VH, VW, 3)
    assert [m.reject_reasons for m in res].count(["eyes_closed"]) == 1
    unmeasured = [m for m in res if m.ear is None]
    assert len(unmeasured) == 2 and all(m.is_valid for m in unmeasured)
    assert "unmeasured on 2 of 3" in caplog.text

    def boom(frame, bbox, kps):
        raise RuntimeError("landmark model failed")
    res = pq.evaluate_candidate_frames(clip, cands[:1], landmarks_fn=boom)
    assert res[0].ear is None and res[0].is_valid


def test_app_landmarks_68_uses_the_analysers_model(monkeypatch):
    import contextlib
    from roop import face_util

    class Model:
        def get(self, img, face):
            self.bbox = face.bbox
            return np.hstack([_lm68(0.3, 0.3), np.zeros((68, 1))])   # (68, 3) like the real one

    class FA:
        lm68_model = None
        models = {"landmark_3d_68": Model()}

    fa = FA()
    monkeypatch.setattr(face_util, "lease_face_analyser", lambda: contextlib.nullcontext(fa))
    out = pq.app_landmarks_68(np.zeros((10, 10, 3), np.uint8), [1, 2, 8, 9])
    assert out.shape == (68, 2) and fa.models["landmark_3d_68"].bbox.tolist() == [1, 2, 8, 9]
    fa.models = {}
    assert pq.app_landmarks_68(np.zeros((10, 10, 3), np.uint8), [1, 2, 8, 9]) is None


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
