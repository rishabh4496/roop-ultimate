"""The point-tracking ladder under the flow chain (`roop/flow_backend.py`).

Ground truth is a known similarity applied to a blurred-noise texture (the same fixture family as
test_flow_coast.py), so every assertion is in pixels.

What is pinned, and what is NOT claimed:

* `CvLK` IS the call `FlowCoaster` always made: a coaster built without a tracker produces bit-identical results to
  one built with `CvLK` (no behaviour change for a render that does not ask for another back end);
* `TorchLK` tracks to sub-pixel against ground truth and agrees with the CPU call; it refuses textureless and
  out-of-frame points the way OpenCV reports them (status 0), and it runs the full `FlowCoaster` chain, forward-
  backward check, RANSAC similarity and all;
* the NVOF rung's SAMPLING and DISPATCH are tested against a fake engine. It has never run on an OFA: this
  environment's OpenCV has no CUDA (`nvof_unavailable_reason()` says so), and the module says UNVERIFIED too;
* `select()` never raises and says what it fell back from.
"""
import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import flow_backend as fb                                  # noqa: E402
from roop.flow_coast import FlowCoaster                               # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


H, W = 480, 640
CENTRE = np.array([320.0, 240.0], np.float32)


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 2.2)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _landmarks(n=106, rx=70.0, ry=90.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([CENTRE[0] + rx * np.cos(t), CENTRE[1] + ry * np.sin(t)], 1)
    inner = CENTRE + np.random.default_rng(3).uniform(-0.6, 0.6, (n // 3, 2)) * [rx, ry]
    return np.concatenate([ring[:n - len(inner)], inner]).astype(np.float32)


def _similarity(dx, dy, deg=0.0, scale=1.0):
    a = np.deg2rad(deg)
    c, s = scale * np.cos(a), scale * np.sin(a)
    m = np.array([[c, -s, 0.0], [s, c, 0.0]], np.float32)
    m[:, 2] = CENTRE + [dx, dy] - m[:, :2] @ CENTRE
    return m


def _warp(img, m):
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _apply(m, pts):
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)


def _bbox(pts):
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], np.float32)


def _points(n=70, seed=5):
    rng = np.random.default_rng(seed)
    return (CENTRE + rng.uniform(-0.6, 0.6, (n, 2)) * [70, 90]).astype(np.float32).reshape(-1, 1, 2)


def _torch_tracker(device='cpu'):
    pytest.importorskip('torch')
    return fb.TorchLK(device=device)


# ── cv: the unchanged call ───────────────────────────────────────────────────────────────────────
def test_cv_is_exactly_the_opencv_call():
    img = _texture()
    nxt = _warp(img, _similarity(3.0, -2.0))
    pts = _points()
    got = fb.CvLK().track(img, nxt, pts)
    want = cv2.calcOpticalFlowPyrLK(img, nxt, pts, None, winSize=(15, 15), maxLevel=3,
                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    for a, b in zip(got, want):
        assert np.array_equal(a, b)


def test_a_coaster_with_no_tracker_equals_one_with_cv():
    img, lm = _texture(), _landmarks()
    a, b = FlowCoaster(), FlowCoaster(tracker=fb.CvLK())
    for c in (a, b):
        assert c.anchor(img, _bbox(lm), lm, lm[:5])
    for step in range(1, 4):
        frame = _warp(img, _similarity(2.0 * step, -1.0 * step, deg=0.5 * step))
        ra, rb = a.step(frame), b.step(frame)
        assert ra.consistent == rb.consistent
        assert np.array_equal(ra.landmarks, rb.landmarks) and np.array_equal(ra.bbox, rb.bbox)


# ── torch: against ground truth and against OpenCV ────────────────────────────────────────────────
@pytest.mark.parametrize('dx,dy', [(0.0, 0.0), (0.4, -0.3), (3.0, -2.0), (6.5, 4.0), (-8.0, 1.5)])
def test_torch_lk_recovers_a_known_translation_to_subpixel(dx, dy):
    img = _texture()
    m = _similarity(dx, dy)
    nxt = _warp(img, m)
    pts = _points()
    new, st, err = _torch_tracker().track(img, nxt, pts)
    ok = st.reshape(-1) == 1
    assert ok.mean() > 0.9
    miss = np.linalg.norm(new[ok].reshape(-1, 2) - _apply(m, pts.reshape(-1, 2))[ok], axis=1)
    assert float(np.median(miss)) < 0.15 and float(np.percentile(miss, 95)) < 0.5, (dx, dy)
    assert err.shape == (len(pts), 1) and np.isfinite(err).all()


def test_torch_lk_recovers_rotation_and_scale():
    img = _texture()
    m = _similarity(2.0, 1.0, deg=4.0, scale=1.03)
    nxt = _warp(img, m)
    pts = _points()
    new, st, _ = _torch_tracker().track(img, nxt, pts)
    ok = st.reshape(-1) == 1
    miss = np.linalg.norm(new[ok].reshape(-1, 2) - _apply(m, pts.reshape(-1, 2))[ok], axis=1)
    assert ok.mean() > 0.85 and float(np.median(miss)) < 0.3


def test_torch_lk_agrees_with_the_cpu_call():
    img = _texture(2)
    nxt = _warp(img, _similarity(3.5, -2.5, deg=2.0))
    pts = _points(120, seed=9)
    t_new, t_st, _ = _torch_tracker().track(img, nxt, pts)
    c_new, c_st, _ = fb.CvLK().track(img, nxt, pts)
    both = (t_st.reshape(-1) == 1) & (c_st.reshape(-1) == 1)
    assert both.mean() > 0.85
    d = np.linalg.norm(t_new[both].reshape(-1, 2) - c_new[both].reshape(-1, 2), axis=1)
    assert float(np.median(d)) < 0.15 and float(np.percentile(d, 90)) < 0.6


def test_torch_lk_refuses_what_opencv_refuses():
    flat = np.full((H, W), 128, np.uint8)
    tr = _torch_tracker()
    _, st, _ = tr.track(flat, flat, _points(20))
    assert st.sum() == 0, 'a textureless window carries no motion: it must not be tracked'
    img = _texture()
    edge = np.array([[[W - 1.0, 100.0]], [[3.0, 4.0]], [[320.0, 240.0]]], np.float32)
    _, st, _ = tr.track(img, _warp(img, _similarity(40.0, 0.0)), edge)
    assert st[0, 0] == 0                                     # the point is carried off the right edge
    assert tr.track(img, img[:10], _points(5))[0] is None    # mismatched crops: no answer, not a crash
    new, st, er = tr.track(img, img, np.zeros((0, 1, 2), np.float32))
    assert new.shape == (0, 1, 2) and st.shape == (0, 1)


def test_torch_lk_gpu_agrees_with_torch_lk_on_the_cpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    img = _texture(4)
    nxt = _warp(img, _similarity(3.0, -2.0))
    pts = _points(100)
    a = fb.TorchLK(device='cpu').track(img, nxt, pts)
    b = fb.TorchLK(device='cuda').track(img, nxt, pts)
    both = (a[1].reshape(-1) == 1) & (b[1].reshape(-1) == 1)
    assert both.mean() > 0.9
    assert float(np.abs(a[0][both] - b[0][both]).max()) < 0.05


# ── the whole flow chain through each tracker ─────────────────────────────────────────────────────
@pytest.mark.parametrize('name', ['cv', 'torch'])
def test_the_flow_chain_follows_a_moving_head_through_each_tracker(name):
    tracker = fb.CvLK() if name == 'cv' else _torch_tracker()
    img, lm = _texture(), _landmarks()
    c = FlowCoaster(tracker=tracker)
    assert c.anchor(img, _bbox(lm), lm, lm[:5])
    truth = lm
    for step in range(1, 6):
        m = _similarity(2.5 * step, -1.5 * step, deg=1.0 * step)
        r = c.step(_warp(img, m))
        assert r.consistent, (name, step, r.reason)
        miss = np.linalg.norm(r.landmarks - _apply(m, lm), axis=1)
        assert float(miss.max()) < 1.5, (name, step, float(miss.max()))


@pytest.mark.parametrize('name', ['cv', 'torch'])
def test_the_chain_still_refuses_an_occluded_head_with_either_tracker(name):
    """The refusal rules live in FlowCoaster, not in the tracker: a back end must not weaken them."""
    tracker = fb.CvLK() if name == 'cv' else _torch_tracker()
    img, lm = _texture(), _landmarks()
    c = FlowCoaster(tracker=tracker)
    assert c.anchor(img, _bbox(lm), lm, lm[:5])
    covered = _warp(img, _similarity(3.0, 0.0))
    covered[100:380, 200:460] = _texture(77)[100:380, 200:460]          # an unrelated picture over the face
    r = c.step(covered)
    assert not r.consistent


# ── nvof: dispatch and sampling against a fake engine (never run on an OFA) ───────────────────────────
class TestNvof:
    def test_sampling_moves_each_point_by_the_flow_under_it(self):
        flow = np.zeros((40, 60, 2), np.float32)
        flow[..., 0] = 2.0
        flow[..., 1] = -1.0
        pts = np.array([[[10.0, 10.0]], [[30.5, 20.25]]], np.float32)
        new, st, err = fb.NvofPointTracker.sample(flow, pts, (40, 60))
        assert np.allclose(new.reshape(-1, 2), pts.reshape(-1, 2) + [2.0, -1.0])
        assert st.reshape(-1).tolist() == [1, 1] and err.shape == (2, 1)

    def test_bilinear_between_cells(self):
        flow = np.zeros((10, 10, 2), np.float32)
        flow[:, :, 0] = np.arange(10, dtype=np.float32)[None, :]             # flow x = column index
        new, _, _ = fb.NvofPointTracker.sample(flow, np.array([[[3.5, 4.0]]], np.float32), (10, 10))
        assert new[0, 0, 0] == pytest.approx(3.5 + 3.5)

    def test_outside_the_image_and_non_finite_flow_are_refused(self):
        flow = np.zeros((20, 20, 2), np.float32)
        flow[5, 5] = np.nan
        pts = np.array([[[25.0, 5.0]], [[5.0, 5.0]], [[10.0, 10.0]]], np.float32)
        _, st, _ = fb.NvofPointTracker.sample(flow, pts, (20, 20))
        assert st.reshape(-1).tolist() == [0, 0, 1]

    def test_a_flow_of_the_wrong_shape_is_no_answer(self):
        assert fb.NvofPointTracker.sample(np.zeros((5, 5, 2), np.float32), np.zeros((1, 1, 2), np.float32), (9, 9))[0] is None
        assert fb.NvofPointTracker.sample(None, np.zeros((1, 1, 2), np.float32), (9, 9))[0] is None

    def test_the_engine_is_built_once_per_crop_size(self):
        built = []
        t = fb.NvofPointTracker(engine_factory=lambda size: built.append(size) or object())
        a, b = t._engine((100, 120)), t._engine((100, 120))
        t._engine((90, 120))
        assert a is b and built == [(120, 100), (120, 90)]                  # OpenCV sizes are (width, height)

    def test_track_uses_the_dense_flow_it_is_given(self):
        class Fixed(fb.NvofPointTracker):
            def _dense(self, prev, nxt):
                out = np.zeros(prev.shape + (2,), np.float32)
                out[..., 0] = 1.5
                return out
        img = np.zeros((30, 40), np.uint8)
        new, st, _ = Fixed().track(img, img, np.array([[[10.0, 10.0]]], np.float32))
        assert new[0, 0].tolist() == [11.5, 10.0] and st[0, 0] == 1

    def test_without_a_cuda_opencv_the_reason_is_stated(self):
        reason = fb.nvof_unavailable_reason()
        if fb.cuda_opencv_available() and fb.nvof_class() is not None:
            assert reason is None
        else:
            assert reason and ('CUDA' in reason or 'NvidiaOpticalFlow' in reason)


# ── selection ────────────────────────────────────────────────────────────────────────────────────
class TestSelect:
    def test_auto_is_the_cpu_call_unless_the_engine_is_really_there(self):
        t = fb.select('auto')
        assert t.name == ('nvof' if fb.nvof_unavailable_reason() is None else 'cv')

    def test_torch_is_chosen_only_when_asked_for(self):
        pytest.importorskip('torch')
        assert fb.select('torch').name == 'torch'
        assert fb.select('cv').name == 'cv'

    def test_an_unavailable_nvof_request_says_why_and_falls_down(self):
        if fb.nvof_unavailable_reason() is None:
            pytest.skip('this machine has the engine')
        said = []
        t = fb.select('nvof', log=said.append)
        assert t.name == 'cv' and said and 'nvof is not available' in said[0] and 'CUDA' in said[0]

    def test_the_environment_chooses_and_junk_means_auto(self, monkeypatch):
        monkeypatch.setenv('ROOP_FLOW_BACKEND', 'TORCH')
        assert fb.requested() == 'torch'
        monkeypatch.setenv('ROOP_FLOW_BACKEND', 'warp-drive')
        assert fb.requested() == 'auto'
        monkeypatch.delenv('ROOP_FLOW_BACKEND')
        assert fb.requested() == 'auto'

    def test_the_prepass_selects_through_this_module_and_builds_coasters_with_it(self):
        text = open(os.path.join(APP, 'roop', 'procmgr_tracking.py'), encoding='utf-8').read()
        assert 'flow_backend.select(' in text
        assert text.count('FlowCoaster(tracker=flow_tracker)') == 2 and 'FlowCoaster()' not in text
