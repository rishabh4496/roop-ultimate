"""Multi-frame end-to-end stability checks on the REAL offline components.

Every test here drives a sequence of frames through the production classes
(KpsStabilizer, OcclusionMasker, MultiFaceTracker, depth_order /
subtract_foreground, the production paste path via a stub that only supplies
the mixin) and asserts a measurable property of the output. Nothing is mocked
below the paste/tracker boundary. Synthetic inputs use a fixed seed, so the
numbers are reproducible.

The three scenarios match the stability brief:

  a) hard-angle lateral sweep, -65..+65 degrees yaw
  b) foreign-object sweep across 40% of the face over 30 frames
  c) two faceset-bound tracks crossing with overlapping boxes
  d) Restore Ultra / UltraMax: 256-384 px bypass ramp, temporal EMA, identity
     gate (algebra invariants) and B=4 batched TensorRT parity

Where the shipped code does not meet a stated bound, the case is recorded as a
strict ``xfail`` whose reason carries the measured number. A bound is never
loosened to make a test pass. If a fix lands, the strict xfail turns into a
failure, which is the signal to promote the case to a plain test.

Scope limits: the tracker cases use one-hot identity embeddings, so the
identity signal is perfectly separable. The occluder cases are synthetic
rectangles, not real hands or bars. The seam case uses the blender's
``laplacian`` mode on a constant colour step, without the production colour
match (``skin_lab_match``) in front of it.
"""
import importlib.util
import math
import os
import sys
from types import MappingProxyType, SimpleNamespace  # noqa: F401  (kept for readers)

import cv2
import numpy as np
import pytest

from roop.one_euro import KpsStabilizer
from roop.occlusion_masker import OcclusionMasker
from roop.multi_face_tracker import MultiFaceTracker, locked_source
from roop.face_z_order import depth_order, subtract_foreground
from roop.blender import delta_e76, laplacian_blend
from roop.face_parser import hull_labels
from roop import smart_restorer_bypass as _sb
from roop.enhancer_stabilizer import DEFAULT_BLEND, TemporalEnhancerBlend


# --------------------------------------------------------------------------- #
# (a) hard-angle lateral sweep
# --------------------------------------------------------------------------- #
# Face-frame 5-point template (left eye, right eye, nose, left mouth, right mouth),
# in units of the face radius, with a per-point depth that foreshortens the
# features as the head turns. This is a geometric model, not a detector.
_TEMPLATE = np.array([[-0.45, -0.25], [0.45, -0.25], [0.0, 0.05],
                      [-0.28, 0.45], [0.28, 0.45]], np.float64)
_DEPTH = np.array([0.4, -0.4, 0.0, 0.3, -0.3], np.float64)
_R = 60.0
_CENTRE = np.array([160.0, 150.0])
# Perceptual just-noticeable difference for CIE76 dE (a common rule of thumb).
_JND_DE76 = 2.3


def _landmarks_at_yaw(yaw_deg):
    th = math.radians(yaw_deg)
    x = _TEMPLATE[:, 0] * math.cos(th) - _DEPTH * math.sin(th)
    y = _TEMPLATE[:, 1]
    return (np.stack([x, y], 1) * _R + _CENTRE).astype(np.float64)


def _sweep_frames(n=131, noise_px=0.6, seed=7):
    """Yaw -65..+65 in 1-degree steps: noise-free truth plus detector jitter."""
    rng = np.random.default_rng(seed)
    yaws = np.linspace(-65.0, 65.0, n)
    truth = np.stack([_landmarks_at_yaw(y) for y in yaws])
    noisy = truth + rng.normal(0.0, noise_px, truth.shape)
    return yaws, truth, noisy


def _motion_error_rms(seq, truth, start):
    """RMS over frames and points of the frame-to-frame displacement error,
    i.e. the jitter that remains after the true motion is removed."""
    d = np.diff(seq[start:], axis=0) - np.diff(truth[start:], axis=0)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=-1))))


def test_hard_angle_sweep_landmark_jitter_below_threshold():
    """Frame-to-frame jitter of the STABILIZED 5-point landmarks, measured
    against the noise-free trajectory, across the full -65..+65 degree sweep.
    Must stay below 0.8 px."""
    _, truth, noisy = _sweep_frames()
    stab = KpsStabilizer(min_cutoff=0.05, beta=0.02)
    smoothed = np.stack([stab.apply(noisy[i], float(i)) for i in range(len(noisy))])
    jitter = _motion_error_rms(smoothed, truth, stab.warmup_frames())
    assert np.isfinite(jitter)
    assert jitter < 0.8, f"stabilized landmark jitter {jitter:.3f} px >= 0.8 px"


def test_hard_angle_sweep_stabilizer_beats_raw_detector():
    """The stabilizer must cut jitter relative to the raw detector on the same
    sequence. This guards against a stabilizer that is wired but inert."""
    _, truth, noisy = _sweep_frames()
    stab = KpsStabilizer(min_cutoff=0.05, beta=0.02)
    smoothed = np.stack([stab.apply(noisy[i], float(i)) for i in range(len(noisy))])
    warm = stab.warmup_frames()
    raw = _motion_error_rms(noisy, truth, warm)
    out = _motion_error_rms(smoothed, truth, warm)
    assert out < 0.5 * raw, f"stabilized {out:.3f} px vs raw {raw:.3f} px"


def _paste_stub():
    """The production paste mixin with only the attributes paste_upscale reads.

    Defined here rather than imported from another test module, so this file
    does not depend on how the `tests` package resolves on sys.path."""
    import threading
    import types
    from roop.procmgr_masking import MaskingMixin

    class _Stub(MaskingMixin):
        def __init__(self):
            self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
            self._tls = threading.local()

    return _Stub()


def _skin_labels(yaw, size=96):
    """Landmark-hull skin label map at a given yaw, the same fallback used when no parser is loaded."""
    th = math.radians(yaw)
    x = _TEMPLATE[:, 0] * math.cos(th) - _DEPTH * math.sin(th)
    lm = (np.stack([x, _TEMPLATE[:, 1]], 1) * 28.0 + size / 2).astype(np.float32)
    return hull_labels(np.vstack([lm, lm * 0 + size / 2]), (size, size))


@pytest.mark.parametrize('yaw', [-65, -40, -15, 0, 15, 40, 65])
def test_hard_angle_paste_mask_stays_inside_skin_class(yaw, monkeypatch):
    """At each yaw, the production paste path must change pixels only where the
    semantic skin label is set. Zero changed pixels may fall outside the skin
    class (no double-layer border)."""
    from roop.face_parser import ParseResult
    import roop.globals
    monkeypatch.setenv('ROOP_GRAIN_MATCH', '0')
    monkeypatch.setattr(roop.globals, 'skin_lab_match', 0., raising=False)
    labels = _skin_labels(yaw)
    assert labels is not None and labels.sum() > 0
    plate = np.full((96, 96, 3), (100, 150, 200), np.uint8)
    swap = np.full_like(plate, (120, 170, 220))
    stub = _paste_stub()
    stub._temporal_engine = lambda name: OcclusionMasker()
    result = stub.paste_upscale(
        swap, swap, np.array([[1., 0, 0], [0, 1., 0]]), plate.copy(), 1,
        [0, 0, 0, 0, 0., 0.], target_crop=plate, track_id=1, frame_index=0,
        semantic_parse=ParseResult(labels, 'bisenet'))
    changed = np.any(result != plate, axis=-1)
    outside = int((changed & (labels == 0)).sum())
    assert outside == 0, f"yaw {yaw}: {outside} changed pixels outside the skin class"


def _boundary_pairs(labels):
    """(inside, outside) pixel coordinate pairs across the skin boundary, 4-connected."""
    pairs = []
    H, W = labels.shape
    for y, x in zip(*np.nonzero(labels == 1)):
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and labels[yy, xx] == 0:
                pairs.append(((y, x), (yy, xx)))
    return pairs


def _seam_steps(yaw, sigma=0.0):
    labels = _skin_labels(yaw)
    plate = np.full((96, 96, 3), (100, 150, 200), np.uint8)
    swap = np.full_like(plate, (60, 200, 90))  # a strong colour step on purpose
    alpha = labels.astype(np.float32)
    if sigma:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    blended = laplacian_blend(swap, plate, alpha).astype(np.uint8)
    pairs = _boundary_pairs(labels)
    assert pairs
    raw = [float(delta_e76(swap[p][None, None], plate[q][None, None])[0, 0]) for p, q in pairs]
    out = [float(delta_e76(blended[p][None, None], plate[q][None, None])[0, 0]) for p, q in pairs]
    return float(np.mean(raw)), float(np.mean(out))


@pytest.mark.parametrize('yaw', [-65, 0, 65])
def test_seam_blend_removes_most_of_the_colour_step(yaw):
    """The Laplacian blend must remove most of the colour step across the
    skin boundary, at the extremes and the frontal pose. Measured as the mean
    CIE76 dE over boundary pixel pairs, blended versus raw. A blend that does
    nothing fails here."""
    raw, out = _seam_steps(yaw, sigma=3.0)
    assert raw > 40.0          # the test really is a strong colour step
    assert out < 0.5 * raw, f"yaw {yaw}: seam {out:.2f} dE vs raw {raw:.2f} dE"


@pytest.mark.xfail(strict=True, reason=(
    "Measured gap: the blend leaves a residual boundary step of about 10-21 dE "
    "(CIE76) on a 73 dE colour step, well above the ~2.3 dE JND. Production "
    "colour matching (skin_lab_match) is not in this unit path, so the number "
    "is a bound on the blend alone, not on the full pipeline."))
@pytest.mark.parametrize('yaw', [-65, 65])
def test_seam_residual_step_below_jnd(yaw):
    _, out = _seam_steps(yaw, sigma=3.0)
    assert out < _JND_DE76, f"yaw {yaw}: residual seam step {out:.2f} dE >= JND"


# --------------------------------------------------------------------------- #
# (b) foreign-object sweep: occluder over 40% of the face, 30 frames
# --------------------------------------------------------------------------- #
def _object_iou(mask, gt, thresh=0.5):
    pred = mask > thresh
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union else 0.0


def _run_object_sweep(obj_colour, noise_sd=0.0, soft=False, seed=3):
    """Drive the real OcclusionMasker over a 30-frame sweep. Return per-frame IoU
    and whether the track stayed live on every frame."""
    rng = np.random.default_rng(seed)
    size, face_w = 96, 56
    clean = np.full((size, size, 3), (120, 150, 190), np.uint8)  # skin-like plate
    labels = np.ones((size, size), np.uint8)
    labels[28:34, 22:35] = 4          # natural eye, protected
    clean[28:34, 22:35] = 10
    silhouette = np.ones((size, size), np.float32)
    masker = OcclusionMasker()
    x0 = (size - face_w) // 2
    travel = int(round(0.40 * face_w))  # object covers 40% of face width at the end
    ious, live = [], []
    for i in range(30):
        ox = x0 + int(round(travel * i / 29))
        gt = np.zeros((size, size), bool)
        gt[45:85, ox:ox + 15] = True
        t = clean.astype(np.float32)
        if soft:
            g = cv2.GaussianBlur(gt.astype(np.float32), (0, 0), 1.2)[..., None]
            t = t * (1 - g) + np.array(obj_colour, np.float32) * g
        else:
            t[gt] = obj_colour
        if noise_sd:
            t = t + rng.normal(0.0, noise_sd, t.shape)
        target = np.clip(t, 0, 255).astype(np.uint8)
        m = masker.compute(target, clean, silhouette, labels=labels,
                           track_id=11, frame_index=i)
        ious.append(_object_iou(m, gt))
        live.append(11 in masker._state)
    return ious, all(live)


def test_foreign_object_sweep_preserves_occluder_pixels_without_dropout():
    """A dark bar with a sharp edge and sensor noise crosses 40% of the face
    width over 30 frames. The swap buffer does not contain it, so the occlusion
    mask must cover it. Mean IoU against the ground-truth rectangle must exceed
    0.95, and the track must stay live on every frame."""
    ious, live = _run_object_sweep((25, 25, 25), noise_sd=4.0, soft=False)
    assert live, "occlusion track dropped out during the sweep"
    mean_iou = float(np.mean(ious))
    assert mean_iou > 0.95, f"occluder IoU {mean_iou:.3f} <= 0.95 (min {min(ious):.3f})"


@pytest.mark.xfail(strict=True, reason=(
    "Measured gap: a soft-edged dark bar (Gaussian sigma 1.2 px on the object "
    "boundary) gives mean IoU about 0.84 against the 0.95 bound. The residual "
    "detector thresholds the hard transition, so the soft rim falls below it."))
def test_foreign_object_soft_edge_iou_meets_bound():
    ious, _ = _run_object_sweep((25, 25, 25), noise_sd=4.0, soft=True)
    assert float(np.mean(ious)) > 0.95


@pytest.mark.xfail(strict=True, reason=(
    "Measured gap: an occluder whose colour is close to skin (125,150,185) is "
    "not detected at all, IoU 0.00 on every frame. The residual detector is "
    "colour-contrast based, so a hand that matches the face is invisible to it."))
def test_skin_toned_occluder_is_detected():
    ious, _ = _run_object_sweep((125, 150, 185), noise_sd=4.0, soft=False)
    assert float(np.mean(ious)) > 0.95


# --------------------------------------------------------------------------- #
# (c) dual-face crossing with two faceset bindings and Z-order compositing
# --------------------------------------------------------------------------- #
def _face(x, y, identity, score=.95, size=40):
    return dict(bbox=np.array([x, y, x + size, y + size], np.float32),
                embedding=np.eye(2, dtype=np.float32)[identity], det_score=score)


def test_dual_face_crossing_track_to_faceset_invariant_and_z_ordered():
    """Two faces move toward each other on the same row, fully overlap at the
    middle frame, and separate. Each is bound to its own .fsz faceset before the
    crossing. For every frame, before, during and after, each face must resolve
    to the same track and the same faceset (locked_source). The composited
    ownership in the overlap must follow the depth order."""
    tracker = MultiFaceTracker(max_age=3)
    faceset = {0: 'alice.fsz', 1: 'bob.fsz'}
    bound = None
    for i in range(41):
        ax = 10 + 2 * i          # identity 0 moves right
        bx = 90 - 2 * i          # identity 1 moves left
        dets = [_face(ax, 20, 0), _face(bx, 20, 1)]
        if i % 2:
            dets.reverse()        # detection order must not matter
        result = tracker.update(dets, i)
        assert len(result) == 2, f"frame {i}: expected 2 tracked faces"
        by_identity = {int(np.argmax(f['embedding'])): f for f in result}
        if bound is None:
            bound = {p: by_identity[p]['_track_id'] for p in (0, 1)}
            for p, tid in bound.items():
                tracker.bind(tid, faceset[p])
        allowed = set(faceset.values())
        for p in (0, 1):
            f = by_identity[p]
            assert f['_track_id'] == bound[p], f"frame {i}: identity {p} changed track"
            assert locked_source(tracker.bindings, f, allowed) == faceset[p], (
                f"frame {i}: identity {p} resolves to the wrong faceset")

    # Z-order at the overlap: the larger, lower face is nearer. The nearer
    # face's semantic region must be removed from the farther face's ownership.
    faces = [_face(60, 20, 0, size=48), _face(70, 26, 1, size=40)]
    faces[0]['_track_id'], faces[1]['_track_id'] = bound[0], bound[1]
    shape = (120, 160, 3)
    order = depth_order(faces, shape)
    masks = []
    for f in faces:
        b = f['bbox'].astype(int)
        m = np.zeros(shape[:2], np.float32)
        m[b[1]:b[3], b[0]:b[2]] = 1
        masks.append(m)
    result = subtract_foreground(faces, masks, shape, order=order)
    near, far = order[-1], order[0]
    overlap = (masks[near] > 0) & (masks[far] > 0)
    assert overlap.any()
    assert float(result[far][overlap].max()) == 0.0, "far face still owns the overlap"
    assert float(result[near][overlap].min()) == 1.0, "near face lost its own pixels"


# --------------------------------------------------------------------------- #
# (d) Restore Ultra / UltraMax
# --------------------------------------------------------------------------- #
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_TESTS_DIR)


def _load_sibling(name):
    """Load a harness module that sits beside this file and is not a package member."""
    if _TESTS_DIR not in sys.path:
        sys.path.insert(0, _TESTS_DIR)   # the harnesses import `fixtures` and friends by bare name
    spec = importlib.util.spec_from_file_location(name, os.path.join(_TESTS_DIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_restore_ramp_is_monotone_and_has_no_step_over_256_384():
    """The bypass weight falls from 1 at 256 px to 0 at 384 px, linearly. Monotone, bounded, and
    no adjacent-height step larger than the ramp's own slope (the old 320 px step jumped by 1.0)."""
    heights = np.arange(200.0, 450.0, 0.5)
    w = np.array([_sb.ramp_weight(h) for h in heights])
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert np.all(np.diff(w) <= 1e-12), "ramp weight rises with face height"
    assert _sb.ramp_weight(256.0) == 1.0 and _sb.ramp_weight(384.0) == 0.0
    max_step = float(np.max(np.abs(np.diff(w)))) / 0.5           # per pixel
    assert max_step <= 1.0 / (_sb.RAMP_HI_PX - _sb.RAMP_LO_PX) + 1e-12


def _ultra_texture(seed=1, size=192):
    rng = np.random.default_rng(seed)
    chans = []
    for c in range(3):
        n = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 3.0)
        n = (n - n.min()) / max(float(n.max() - n.min()), 1e-6)
        chans.append(100.0 + 60.0 * n + 5.0 * c)
    return np.stack(chans, -1).astype(np.float32)


def _ultra_redraw(base, seed, sigma=7.0):
    """The same face with an independent high-frequency texture draw each frame: what a restorer does."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.standard_normal(base.shape[:2]).astype(np.float32), (0, 0), 0.7)
    n *= sigma / max(float(n.std()), 1e-6)
    return np.clip(base + n[..., None] + 0.5, 0, 255).astype(np.uint8)


def _hf_variance(frames):
    hp = [f.astype(np.float32) - cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.5) for f in frames]
    return float(np.mean(np.var(np.stack(hp), axis=0)))


def _shimmer_reduction(blend, n=40, skip=8):
    base = _ultra_texture()
    frames = [_ultra_redraw(base, 700 + i) for i in range(n)]
    eng = TemporalEnhancerBlend(blend=blend)
    outs = [eng.blend_crop(f, track_id=1, frame_index=i) for i, f in enumerate(frames)]
    return 1.0 - _hf_variance(outs[skip:]) / _hf_variance(frames[skip:])


@pytest.mark.xfail(strict=True, reason=(
    "Measured gap: the flow-registered EMA at the shipped DEFAULT blend (0.20) cuts high-pass "
    "temporal variance by 33.3% (closed form 1-(1-w)/(1+w)); the >40% bound needs history "
    "weight w >= 0.25. The bound is not loosened; the default is not changed here."))
def test_ema_variance_reduction_exceeds_40pct_at_default_blend():
    assert DEFAULT_BLEND == 0.20
    got = _shimmer_reduction(DEFAULT_BLEND)
    assert got > 0.40, f"default blend reduced HF temporal variance by {got:.3f}"


def test_ema_variance_reduction_exceeds_40pct_at_blend_030():
    got = _shimmer_reduction(0.30)
    assert got > 0.40, f"blend 0.30 reduced HF temporal variance by only {got:.3f}"


def test_identity_gate_algebra_invariants_hold():
    """The identity gate's offline half: unit norm, slerp constancy, keyframe smoothness and the
    de-identification bound, with no model and no GPU. Render arms need real faces and are not run here."""
    bench = _load_sibling("identity_algebra_bench")
    assert bench.algebra_check(seed=0) == 0


@pytest.mark.gpu
def test_ultramax_b4_batched_matches_b1_on_tensorrt():
    """Four faces in one TensorRT call against four single-face calls, on the shipped codeformer.fp16 graph.
    Skips (never passes) when TensorRT or the model is absent."""
    ort = pytest.importorskip("onnxruntime")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        pytest.skip("TensorRT is not available on this machine")
    probe = _load_sibling("ultramax_batch_probe")
    if not os.path.isfile(os.path.join(_APP_DIR, probe.MODEL_REL)):
        pytest.skip("codeformer.fp16.onnx is not present")
    import roop.globals as g
    saved_cwd = os.getcwd()
    saved = (getattr(g, "CFG", None), getattr(g, "execution_providers", None))
    try:
        probe.init_environment()
        if not any("Tensorrt" in str(p) for p in g.execution_providers):
            pytest.skip("TensorRT is not among the decoded execution providers")
        sess = probe.build_session("shipped")
        assert sess.get_providers()[0] == "TensorrtExecutionProvider", "the session fell back off TensorRT"
        r = probe.measure(sess, 4, iters=3, warm=2)
    finally:
        os.chdir(saved_cwd)
        g.CFG, g.execution_providers = saved
    assert r["finite"]
    assert r["max_levels_u8"] <= 2, f"B=4 differs from B=1 by {r['max_levels_u8']} levels"
    assert r["frac_pixels_differing_u8"] <= 0.001
    assert r["batched_ms_per_face"] < 250.0, "too slow to be the TensorRT engine"


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
