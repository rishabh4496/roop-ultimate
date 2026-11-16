"""Bug #2 Stage 2: semantic paste matte, yaw-asymmetric boundary, multi-band blend.

Synthetic parse maps stand in for BiSeNet so the geometry and colour claims can be
asserted exactly and without a GPU. ``test_real_bisenet_*`` runs the real network
when it can be loaded (skipped, with the reason, when it cannot) -- a synthetic
test cannot prove the model path executes (AGENTS.md: "Count the stage, not the
wrapper").
"""
import os
import time

import cv2
import numpy as np
import pytest

from roop import face_parser as fp
from roop import blender as bl
from roop.mask_composer import MaskComposer, debug_channels, publish_debug, take_debug
from roop.unified_matte import yaw_from_kps, inward_alpha

N = 256
SKIN, HAIR, EAR_L, EAR_R, NECK, EYE_L, EYE_R, NOSE, MOUTH, LIP_U, LIP_L, GLASSES = 1, 17, 7, 8, 14, 4, 5, 10, 11, 12, 13, 6


def synth(yaw, n=N, mislabel_core=False):
    """Label map + landmark-style hull of a face turned by ``yaw`` degrees.

    The hull (like a real 106-point hull) sits ~10 px OUTSIDE the skin contour, so
    it contains slivers of hair, ear, neck and background -- the pixels a hull-only
    paste would swap over. Positive yaw: the nose side (+x) is the far cheek.
    """
    s = min(abs(yaw) / 60.0, 1.0)
    sign = 1 if yaw >= 0 else -1
    half_w, half_h = int(n * 0.30 * (1 - 0.45 * s)), int(n * 0.36)
    fx, cy = n // 2 + int(sign * n * 0.06 * s), n // 2
    lab = np.zeros((n, n), np.uint8)
    cv2.ellipse(lab, (fx, cy - int(n * .06)), (half_w + int(n * .09), half_h + int(n * .07)), 0, 0, 360, HAIR, -1)
    cv2.rectangle(lab, (fx - half_w // 2, cy + half_h - 14), (fx + half_w // 2, n - 1), NECK, -1)
    cv2.ellipse(lab, (fx, cy), (half_w, half_h), 0, 0, 360, SKIN, -1)
    near_x = fx - sign * half_w
    cv2.ellipse(lab, (near_x, cy), (int(n * .035), int(n * .06)), 0, 0, 360, EAR_L if sign > 0 else EAR_R, -1)
    cv2.ellipse(lab, (fx - half_w // 3, cy - 20), (12, 6), 0, 0, 360, EYE_L, -1)
    cv2.ellipse(lab, (fx + half_w // 3, cy - 20), (12, 6), 0, 0, 360, EYE_R, -1)
    cv2.ellipse(lab, (fx, cy + 5), (7, 14), 0, 0, 360, NOSE, -1)
    cv2.ellipse(lab, (fx, cy + 45), (20, 5), 0, 0, 360, MOUTH, -1)
    if mislabel_core:    # the s4.mp4 failure: real cheek skin labelled neck, deep inside the face
        cv2.ellipse(lab, (fx, cy + 20), (22, 16), 0, 0, 360, NECK, -1)
    hull = np.zeros((n, n), np.float32)
    cv2.ellipse(hull, (fx, cy), (half_w + 10, half_h + 10), 0, 0, 360, 1.0, -1)
    return lab, hull


def interior_of(lab):
    return fp.class_mask(lab, fp.INTERIOR_CLASSES + ('glasses',))


def parse_of(lab):
    return fp.ParseResult(lab, 'bisenet')


def kps_for_yaw(yaw_deg):
    """5 landmarks (eyes, nose, mouth corners) whose nose offset gives ``yaw_deg``."""
    t = np.tan(np.radians(yaw_deg))
    eye_l, eye_r = np.array([80., 100.]), np.array([176., 100.])
    nose = (eye_l + eye_r) / 2 + np.array([t * 48.0, 50.0])      # asym = 2*dx/eye_len = t
    return np.array([eye_l, eye_r, nose, [95., 170.], [161., 170.]], np.float32)


# ---------------------------------------------------------------- face_parser
def test_class_registry_covers_the_brief_and_every_model_id():
    for name in ('skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose', 'u_lip', 'l_lip',
                 'mouth', 'hair', 'ear_l', 'ear_r', 'neck', 'glasses'):
        assert name in fp.SEMANTIC_CLASSES
    ids = sorted(i for v in fp.SEMANTIC_CLASSES.values() for i in v)
    assert ids == list(range(19)), 'every CelebAMask id belongs to exactly one semantic class'
    assert not set(fp.INTERIOR_CLASSES) & set(fp.OPAQUE_EXCLUDED + fp.AMBIGUOUS_EXCLUDED)


def test_class_mask_matches_isin_and_resize_is_nearest():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 19, (64, 64)).astype(np.uint8)
    want = np.isin(lab, fp.class_ids(('skin', 'neck')))
    assert np.array_equal(fp.class_mask(lab, ('skin', 'neck')), want)
    big = fp.resize_labels(lab, (128, 128))
    assert set(np.unique(big)) <= set(np.unique(lab)), 'nearest resize must not invent classes'


def test_hull_fallback_when_no_weights_never_downloads(monkeypatch):
    monkeypatch.setattr(fp, 'weights_available', lambda: False)
    fp.FaceParser.release()
    pts = []
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        pts.append((128 + 70 * np.cos(a), 128 + 90 * np.sin(a)))
    res = fp.FaceParser().parse(np.zeros((N, N, 3), np.uint8), landmarks=np.array(pts, np.float32))
    assert res is not None and res.source == 'hull' and not res.is_semantic
    assert res.labels[128, 128] == 1 and res.labels[2, 2] == 0
    assert fp.FaceParser().parse(np.zeros((N, N, 3), np.uint8)) is None      # no weights, no landmarks
    fp.FaceParser.release()


def test_hull_parse_drives_the_composer_without_weights():
    lab, hull = synth(0)
    res = MaskComposer().compose((N, N), fp.ParseResult((hull > 0).astype(np.uint8), 'hull'), hull=hull)
    assert res.source == 'hull' and res.alpha.max() > 0.95
    assert res.alpha[hull == 0].max() == 0.0


# ------------------------------------------------------------- mask_composer
@pytest.mark.parametrize('yaw', [55.0, -55.0, 0.0])
def test_zero_bleed_outside_target_skin_classes(yaw):
    """The literal brief: alpha is exactly 0 on every pixel the parser does not call
    skin/brow/eye/nose/lip/mouth (hair, ears, neck, background, ...)."""
    lab, hull = synth(yaw)
    res = MaskComposer(core_protect=False).compose((N, N), parse_of(lab), hull=hull, yaw=yaw)
    outside = ~interior_of(lab)
    assert outside.sum() > 0 and (hull > 0)[outside].sum() > 200, 'the hull must contain non-face pixels to be a test'
    assert float(res.alpha[outside].max()) == 0.0
    assert float(res.alpha[~(hull > 0)].max()) == 0.0
    assert res.alpha.max() > 0.95 and res.alpha[N // 2, N // 2] > 0.9


@pytest.mark.parametrize('yaw', [55.0, -55.0])
def test_hull_alone_would_have_bled_so_the_test_has_teeth(yaw):
    lab, hull = synth(yaw)
    bled = ((hull > 0) & ~interior_of(lab)).sum()
    assert bled > 500, 'a hull-only paste swaps over %d non-face pixels here' % bled


def test_core_protect_keeps_mislabelled_neck_inside_the_face_but_not_hair():
    yaw = 30.0
    lab, hull = synth(yaw, mislabel_core=True)
    cy = N // 2
    strict = MaskComposer(core_protect=False).compose((N, N), parse_of(lab), hull=hull, yaw=0.0)
    safe = MaskComposer().compose((N, N), parse_of(lab), hull=hull, yaw=0.0)
    probe = (cy + 20, N // 2 + int(np.sign(yaw) * N * .06 * .5))
    assert strict.alpha[probe] == 0.0, 'the literal brief swaps nothing over a neck label'
    assert safe.alpha[probe] > 0.5, 'core_protect must not punch a hole in a mislabelled cheek'
    hair = fp.class_mask(lab, ('hair',)) & (hull > 0)
    assert hair.any() and float(safe.alpha[hair].max()) == 0.0, 'hair is opaque and always excluded'


def _edge_profile(res, yaw, row):
    """(near erosion depth, far erosion depth, near band, far band) in px along `row`."""
    a, inter = res.alpha[row], res.interior[row]
    xs = np.flatnonzero(inter)
    left, right = xs.min(), xs.max()

    def cross(vals, thr):
        i = np.flatnonzero(vals >= thr)
        return i[0] if len(i) else None
    lo = cross(a, 0.5) - left
    hi = right - (len(a) - 1 - cross(a[::-1], 0.5))
    lo_band = cross(a, 0.95) - cross(a, 0.05)
    hi_band = cross(a[::-1], 0.95) - cross(a[::-1], 0.05)
    left_d, right_d, left_b, right_b = lo, hi, lo_band, hi_band
    return (right_d, left_d, right_b, left_b) if yaw < 0 else (left_d, right_d, left_b, right_b)


@pytest.mark.parametrize('yaw', [55.0, -55.0])
def test_far_cheek_is_eroded_harder_and_softer_than_the_near_jaw(yaw):
    lab, hull = synth(yaw)
    res = MaskComposer(core_protect=False).compose((N, N), parse_of(lab), hull=hull, yaw=yaw, feather=8.0)
    near_d, far_d, near_b, far_b = _edge_profile(res, yaw, N // 2 - 30)
    assert far_d >= near_d + 3, 'far edge pulled in %d px vs near %d px' % (far_d, near_d)
    assert far_b > near_b, 'far feather band %d px must be wider than near %d px' % (far_b, near_b)


def test_frontal_boundary_is_symmetric():
    lab, hull = synth(0.0)
    res = MaskComposer(core_protect=False).compose((N, N), parse_of(lab), hull=hull, yaw=0.0)
    d1, d2, b1, b2 = _edge_profile(res, 0.0, N // 2 - 30)
    assert abs(d1 - d2) <= 2 and abs(b1 - b2) <= 2


def test_yaw_sign_convention_matches_yaw_from_kps():
    """+yaw (nose toward +x) must put the far edge at +x, the way unified_matte does."""
    assert yaw_from_kps(kps_for_yaw(40.0)) > 0 > yaw_from_kps(kps_for_yaw(-40.0))
    assert yaw_from_kps(kps_for_yaw(40.0)) == pytest.approx(40.0, abs=0.5)


def test_swapped_crop_validity_clips_the_matte():
    lab, hull = synth(0.0)
    swapped = lab.copy()
    # Warp artefacts live at the rim, outside the landmark core (core_protect keeps a
    # background label INSIDE the core, where BiSeNet's frontal priors are the usual culprit).
    swapped[:, 198:] = 0
    with_swap = MaskComposer().compose((N, N), parse_of(lab), swap=parse_of(swapped), hull=hull)
    without = MaskComposer().compose((N, N), parse_of(lab), hull=hull)
    assert with_swap.swap_used and not without.swap_used
    assert without.interior[:, 199:].any(), 'the target parse alone swaps over the invalid strip'
    assert not with_swap.interior[:, 199:].any() and float(with_swap.alpha[:, 199:].max()) == 0.0


def test_unreliable_parse_falls_back_to_the_hull_instead_of_swapping_half_a_face():
    lab = np.zeros((N, N), np.uint8)                  # parser saw nothing but background
    _, hull = synth(0.0)
    res = MaskComposer().compose((N, N), parse_of(lab), hull=hull)
    assert res.source == 'hull-fallback' and res.parse_cover < 0.55
    assert res.alpha[N // 2, N // 2] > 0.9


def test_stray_island_is_dropped_but_ambiguous_split_is_kept():
    lab, hull = synth(0.0)
    lab[10:30, 10:30] = SKIN                          # another face's skin, far away
    hull2 = hull.copy()
    hull2[10:30, 10:30] = 1.0
    res = MaskComposer().compose((N, N), parse_of(lab), hull=hull2)
    assert float(res.alpha[10:30, 10:30].max()) == 0.0 and res.alpha[N // 2, N // 2] > 0.9


def test_base_alpha_and_occluder_keeps_are_respected():
    lab, hull = synth(0.0)
    base = np.ones((N, N), np.float32)
    base[:, : N // 2] = 0.0                           # occluder / user offset removed the left half
    res = MaskComposer().compose((N, N), parse_of(lab), hull=hull, base_alpha=base)
    assert float(res.alpha[:, : N // 2].max()) == 0.0 and res.alpha[N // 2, N // 2 + 30] > 0.9


def test_labels_at_512_resize_to_the_crop():
    lab, hull = synth(30.0, n=512)
    res = MaskComposer(core_protect=False).compose((256, 256), parse_of(lab), hull=cv2.resize(hull, (256, 256)), yaw=30.0)
    assert res.alpha.shape == (256, 256) and res.alpha.dtype == np.float32
    assert 0.0 <= res.alpha.min() and res.alpha.max() <= 1.0


def test_composer_cost_is_small_at_crop_resolution():
    lab, hull = synth(40.0, n=512)
    c = MaskComposer()
    c.compose((512, 512), parse_of(lab), hull=hull, yaw=40.0)
    t = time.perf_counter()
    for _ in range(20):
        c.compose((512, 512), parse_of(lab), hull=hull, yaw=40.0)
    ms = (time.perf_counter() - t) / 20 * 1000
    assert ms < 40.0, 'composer took %.1f ms per 512 crop' % ms


def test_debug_channels_shapes_and_registry():
    lab, hull = synth(20.0)
    crop = np.full((N, N, 3), 120, np.uint8)
    res = MaskComposer().compose((N, N), parse_of(lab), hull=hull, yaw=20.0)
    ch = debug_channels(crop, parse_of(lab), res)
    assert set(ch) == {'parsing_overlay', 'final_alpha_mask'}
    assert ch['parsing_overlay'].shape == (N, N, 3) and ch['parsing_overlay'].dtype == np.uint8
    assert ch['final_alpha_mask'].shape == (N, N) and ch['final_alpha_mask'].max() > 240
    take_debug()
    publish_debug(ch)
    assert set(take_debug()) == set(ch) and take_debug() == {}


# ------------------------------------------------------------------- blender
def _skin_scene(dl=(30, 16, 20), n=200, seed=3):
    """target plate + swapped crop of the same face, the swap a different skin colour."""
    rng = np.random.default_rng(seed)
    tex = rng.normal(0, 3.0, (n, n, 1)).astype(np.float32)
    plate = np.empty((n, n, 3), np.float32)
    plate[:] = (140, 165, 205)
    plate += tex
    swap = np.empty_like(plate)
    swap[:] = np.array((140, 165, 205), np.float32) + np.array(dl, np.float32)
    swap += rng.normal(0, 3.0, (n, n, 1)).astype(np.float32)
    mask = np.zeros((n, n), np.uint8)
    cv2.ellipse(mask, (n // 2, n // 2), (int(n * .36), int(n * .44)), 0, 0, 360, 1, -1)
    alpha = MaskComposer().asymmetric_inward(mask.astype(np.float32), 0.0, 14.0)
    return np.clip(plate, 0, 255), np.clip(swap, 0, 255), mask, alpha


def _seam_step(out, mask, d0=18, d1=22):
    """CIE76 distance between the mean colour d0..d1 px INSIDE and OUTSIDE the contour.

    d0 is beyond the 14 px feather, so the inside ring is full-alpha face: this is the
    colour step the feather has to hide."""
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    a, b = (inside >= d0) & (inside <= d1), (outside >= 3) & (outside <= 7)
    lab = bl._to_lab(out)
    return float(np.linalg.norm(lab[a].mean(0) - lab[b].mean(0)))


def _ring_mismatch(content, plate, alpha):
    """Mean CIE76 between the content being blended and the plate across the transition band."""
    ring = (alpha > 0.05) & (alpha < 0.95)
    return float(bl.delta_e76(content, plate)[ring].mean())


@pytest.mark.parametrize('mode', bl.BLEND_MODES)
def test_every_mode_leaves_the_plate_untouched_where_alpha_is_zero(mode):
    plate, swap, mask, alpha = _skin_scene()
    out = bl.blend(swap, plate, alpha, mode)
    assert out.dtype == np.float32 and out.shape == plate.shape
    assert np.array_equal(out[alpha == 0], plate[alpha == 0])


@pytest.mark.parametrize('mode', bl.BLEND_MODES)
def test_alpha_one_returns_the_paste(mode):
    plate, swap, *_ = _skin_scene()
    out = bl.blend(swap, plate, np.ones(plate.shape[:2], np.float32), mode)
    assert float(np.abs(out - swap).max()) < 1e-3


def test_unknown_mode_is_rejected_and_normalised_to_the_default():
    with pytest.raises(ValueError):
        bl.blend(np.zeros((8, 8, 3), np.float32), np.zeros((8, 8, 3), np.float32), np.zeros((8, 8)), 'bogus')
    assert bl.normalize_blend_mode('Poisson-Fast') == 'poisson_fast' and bl.normalize_blend_mode('x') == 'laplacian'


def test_tiny_roi_does_not_crash_the_pyramid():
    p = np.full((6, 9, 3), 200, np.float32)
    t = np.full((6, 9, 3), 50, np.float32)
    out = bl.laplacian_blend(p, t, np.full((6, 9), .5, np.float32))
    assert out.shape == p.shape and np.isfinite(out).all()


def test_laplacian_keeps_pores_where_a_low_pass_blend_would_not():
    plate, swap, mask, alpha = _skin_scene(dl=(0, 0, 0))
    core = cv2.erode(mask, np.ones((41, 41), np.uint8)).astype(bool)
    out = bl.laplacian_blend(swap, plate, alpha)
    hf = lambda im: im - cv2.GaussianBlur(im, (0, 0), 2.0)
    assert np.allclose(hf(out)[core], hf(swap)[core], atol=0.5)
    assert hf(out)[core].std() > 2.0


def test_skin_lab_match_cuts_the_seam_delta_e():
    plate, swap, mask, alpha = _skin_scene()
    before = _seam_step(bl.alpha_feather_blend(swap, plate, alpha), mask)
    matched, info = bl.skin_lab_match(swap, plate, mask)
    assert info['applied'] and info['pixels'] > 1000
    assert before > 8.0, 'the scene must start with a visible seam (dE %.1f)' % before
    for mode in bl.BLEND_MODES:
        after = _seam_step(bl.blend(matched, plate, alpha, mode), mask)
        assert after < 0.3 * before, '%s: seam dE %.2f -> %.2f' % (mode, before, after)
    # and across the transition band itself
    assert _ring_mismatch(matched, plate, alpha) < 0.3 * _ring_mismatch(swap, plate, alpha)


def test_poisson_fast_alone_also_reduces_the_seam():
    plate, swap, mask, alpha = _skin_scene()
    plain = _seam_step(bl.alpha_feather_blend(swap, plate, alpha), mask)
    assert _seam_step(bl.poisson_fast_blend(swap, plate, alpha), mask) < plain


def test_lab_statistics_come_from_skin_only_and_eyes_and_teeth_are_not_recoloured():
    plate, swap, mask, alpha = _skin_scene()
    n = plate.shape[0]
    skin = mask.copy()
    patch = np.zeros_like(mask)
    cv2.ellipse(patch, (n // 2, n // 2 - 10), (28, 14), 0, 0, 360, 1, -1)       # a mouth with teeth
    skin[patch > 0] = 0
    teeth = (240.0, 240.0, 235.0)
    swap_t, plate_t = swap.copy(), plate.copy()
    swap_t[patch > 0], plate_t[patch > 0] = teeth, (60.0, 60.0, 150.0)          # plate has a dark open mouth

    base, _ = bl.skin_lab_match(swap, plate, skin)
    with_patch, info = bl.skin_lab_match(swap_t, plate_t, skin)
    ring = cv2.erode(skin, np.ones((15, 15), np.uint8)).astype(bool)
    assert float(bl.delta_e76(base, with_patch)[ring].max()) < 1.0, 'teeth/mouth leaked into the statistics'
    centre = np.zeros_like(patch)
    cv2.ellipse(centre, (n // 2, n // 2 - 10), (14, 5), 0, 0, 360, 1, -1)
    assert float(bl.delta_e76(with_patch, swap_t)[centre > 0].max()) < 1.5, 'teeth were recoloured'


def test_lab_match_noops_without_enough_skin():
    plate, swap, mask, _ = _skin_scene()
    tiny = np.zeros_like(mask)
    tiny[5:12, 5:12] = 1
    out, info = bl.skin_lab_match(swap, plate, tiny)
    assert not info['applied'] and np.array_equal(out, swap.astype(np.float32))
    out, info = bl.skin_lab_match(swap, plate, mask, strength=0.0)
    assert not info['applied']


def test_lab_shift_is_capped():
    plate, swap, mask, _ = _skin_scene(dl=(90, 60, 60))
    out, info = bl.skin_lab_match(swap, plate, mask)
    moved = bl._to_lab(out) - bl._to_lab(swap)
    assert float(np.abs(moved).max()) <= float(bl._LAB_MAX_SHIFT.max()) + 0.6


def test_blend_cost_at_face_roi_size():
    n = 640
    p = np.random.default_rng(0).uniform(0, 255, (n, n, 3)).astype(np.float32)
    t = p[::-1].copy()
    a = cv2.GaussianBlur(np.pad(np.ones((n - 160, n - 160), np.float32), 80), (0, 0), 20)
    bl.laplacian_blend(p, t, a)
    t0 = time.perf_counter()
    for _ in range(10):
        bl.laplacian_blend(p, t, a)
    ms = (time.perf_counter() - t0) / 10 * 1000
    assert ms < 60.0, 'laplacian blend took %.1f ms on a 640 px ROI' % ms


# --------------------------------------------------------- real network (opt-in)
def _real_parser():
    if not fp.weights_available():
        pytest.skip('models/resnet18.onnx is not present')
    fp.FaceParser.release()
    parser = fp.FaceParser()
    if not parser.available():
        pytest.skip('BiSeNet session could not be built in this process (needs the app init for providers)')
    return parser


def test_real_bisenet_parses_a_face_and_feeds_the_composer():
    parser = _real_parser()
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, '..', 'facesets', 'akansha.png')
    img = cv2.imread(path)
    if img is None:
        pytest.skip('no sample face image')
    res = parser.parse(img)
    assert res.source == 'bisenet' and res.labels.shape == (512, 512)
    assert fp.class_mask(res.labels, ('skin',)).mean() > 0.08, 'BiSeNet found no skin in a portrait'
    out = MaskComposer().compose((512, 512), res, yaw=0.0)
    assert out.alpha.max() > 0.9 and 0.05 < out.alpha.mean() < 0.9
    t = time.perf_counter()
    again = parser.parse(img)
    assert (time.perf_counter() - t) < 0.02 and np.array_equal(again.labels, res.labels), 'second parse must hit the cache'
    fp.FaceParser.release()


# -------------------------------------------- the real paste_upscale, end to end
import threading
from types import SimpleNamespace

import roop.globals as _g
from roop.procmgr_masking import MaskingMixin


class _Paste(MaskingMixin):
    def __init__(self):
        self.options = SimpleNamespace(blend_ratio=1., show_face_area_overlay=False)
        self._tls = threading.local()


_OFF = 72     # crop (0,0) sits at plate (72,72)


def _paste_scene(yaw):
    lab, hull = synth(yaw)
    ys, xs = np.nonzero(hull > 0.5)
    ang = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    rx, ry = (xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2
    lm = np.stack([cx + rx * np.cos(ang), cy + ry * np.sin(ang)], 1).astype(np.float32) + _OFF
    kps = kps_for_yaw(yaw) + _OFF
    plate = np.full((400, 400, 3), (130, 160, 200), np.uint8)
    swapped = np.full((N, N, 3), (150, 175, 225), np.uint8)
    M = np.array([[1., 0., -_OFF], [0., 1., -_OFF]])
    return lab, plate, swapped, M, lm, kps


def _run_paste(monkeypatch, yaw, *, parse='semantic', **cfg):
    monkeypatch.setenv('ROOP_GRAIN_MATCH', '0')
    settings = dict(semantic_paste_mask=True, blend_mode='laplacian', skin_lab_match=1.0,
                    debug_mask_channels=False)
    settings.update(cfg)
    for k, v in settings.items():
        monkeypatch.setattr(_g, k, v, raising=False)
    lab, plate, swapped, M, lm, kps = _paste_scene(yaw)
    sp = parse_of(lab) if parse == 'semantic' else None
    out = _Paste().paste_upscale(swapped, swapped, M, plate.copy(), 1, [0, 0, 0, 0, 12., 10.],
                                 face_landmarks=lm, face_kps=kps, semantic_parse=sp)
    return lab, plate, out


def _non_face_pixels(lab, plate, out):
    non = ~interior_of(lab)
    view = np.s_[_OFF:_OFF + N, _OFF:_OFF + N]
    return non, plate[view], out[view]


@pytest.mark.parametrize('yaw', [55.0, -55.0])
def test_paste_upscale_leaves_every_non_face_pixel_untouched(monkeypatch, yaw):
    lab, plate, out = _run_paste(monkeypatch, yaw)
    non, p, o = _non_face_pixels(lab, plate, out)
    assert np.array_equal(o[non], p[non]), '%d non-face pixels were painted' % int((o[non] != p[non]).any(-1).sum())
    assert np.any(out != plate), 'nothing was pasted at all'


@pytest.mark.parametrize('yaw', [55.0, -55.0])
def test_without_the_parse_the_hull_paints_over_hair_ear_and_neck(monkeypatch, yaw):
    lab, plate, out = _run_paste(monkeypatch, yaw, semantic_paste_mask=False)
    non, p, o = _non_face_pixels(lab, plate, out)
    painted = int((o[non] != p[non]).any(-1).sum())
    assert painted > 150, 'control: the legacy matte should bleed here (%d px)' % painted


def test_the_semantic_matte_really_runs_in_the_paste(monkeypatch):
    import roop.mask_composer as mc
    calls = []
    real = mc.MaskComposer.compose
    monkeypatch.setattr(mc.MaskComposer, 'compose', lambda self, *a, **k: calls.append(k.get('yaw')) or real(self, *a, **k))
    _run_paste(monkeypatch, 55.0)
    assert len(calls) == 1 and calls[0] > 30.0, 'composer must run once, with the face yaw (%r)' % calls
    calls.clear()
    _run_paste(monkeypatch, 55.0, semantic_paste_mask=False)
    assert calls == []


def test_a_missing_parse_degrades_to_the_hull_through_the_composer(monkeypatch):
    lab, plate, out = _run_paste(monkeypatch, 40.0, parse='none')
    assert np.any(out != plate)
    non, p, o = _non_face_pixels(lab, plate, out)
    assert int((o[non] != p[non]).any(-1).sum()) > 0       # hull-level protection only, by design


def test_blend_modes_all_run_and_preserve_the_plate_outside_the_matte(monkeypatch):
    outs = {}
    for mode in bl.BLEND_MODES:
        lab, plate, outs[mode] = _run_paste(monkeypatch, 40.0, blend_mode=mode)
        far = np.ones(plate.shape[:2], bool)
        far[_OFF:_OFF + N, _OFF:_OFF + N] = False
        assert np.array_equal(outs[mode][far], plate[far])
    assert not np.array_equal(outs['laplacian'], outs['alpha_feather'])
    assert np.isfinite(outs['poisson_fast']).all()


def test_skin_lab_match_pulls_the_pasted_skin_toward_the_plate(monkeypatch):
    _, plate, off = _run_paste(monkeypatch, 0.0, skin_lab_match=0.0)
    _, _, on = _run_paste(monkeypatch, 0.0, skin_lab_match=1.0)
    c = (slice(_OFF + 100, _OFF + 150), slice(_OFF + 100, _OFF + 150))
    d_off = float(bl.delta_e76(off[c], plate[c]).mean())
    d_on = float(bl.delta_e76(on[c], plate[c]).mean())
    assert d_on < 0.6 * d_off, 'skin dE to the plate %.1f -> %.1f' % (d_off, d_on)


def test_debug_channels_are_published_only_on_request(monkeypatch):
    take_debug()
    _run_paste(monkeypatch, 30.0)
    assert take_debug() == {}
    _run_paste(monkeypatch, 30.0, debug_mask_channels=True)
    ch = take_debug()
    assert set(ch) == {'parsing_overlay', 'final_alpha_mask'} and ch['final_alpha_mask'].max() > 200
