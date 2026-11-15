"""Standalone verification of the 2026-11-16 geometry / single-alpha fixes
(.audit/05_changelog.md).

Run either way (use the app interpreter so cv2/onnxruntime resolve):

    app/env/Scripts/python.exe tests/verify_face_swap_fixes.py
    app/env/Scripts/python.exe -m pytest tests/verify_face_swap_fixes.py -q

Everything runs on synthetic frames against the PRODUCTION functions
(`roop.orientation`, `roop.alignment_safety`, `roop.face_util.estimate_norm`,
`roop.unified_matte`, `MaskingMixin.paste_upscale`). No model download is
needed. If `app/models/hyperswap_1a_256.onnx` and onnxruntime are present, one
extra test pushes a REAL swapper output through the same paste (disable with
ROOP_VERIFY_REAL=0).

What synthetic geometry can and cannot prove: these checks establish the
mechanisms (no flips, no background bleed, one alpha boundary). They do not
establish visual quality on real footage; that acceptance gate is still open.
"""
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for _p in (str(REPO), str(APP)):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(APP))
os.environ["ROOP_GRAIN_MATCH"] = "0"      # grain would add legitimate noise

from roop.alignment_safety import (InvalidAlignment, TrackGeometry, affine,  # noqa: E402
                                   landmarks5, unit_latent)
from roop.face_util import estimate_norm, swap_template_points  # noqa: E402
from roop.orientation import resolve_track_rolls, roll_from_kps, wrap180  # noqa: E402
from roop.procmgr_masking import MaskingMixin  # noqa: E402
from roop.unified_matte import inward_alpha, mask2d, warp_premultiplied, yaw_from_kps  # noqa: E402

CROP = 256
TEMPLATE = swap_template_points(CROP, "arcface").astype(np.float32)   # upright 5 pts


# --------------------------------------------------------------------------
# synthetic face builders
# --------------------------------------------------------------------------
def make_kps(roll=0., pitch=0., yaw=0., centre=(200., 200.), scale=.8, jitter=None):
    """5 landmarks (l-eye, r-eye, nose, l-mouth, r-mouth) in frame space.

    roll rotates in plane (positive = chin swings to image +x, matching
    roll_from_kps); pitch compresses the vertical axis (nose/mouth drift);
    yaw shifts the nose along the eye baseline and narrows the horizontal span.
    """
    p = (TEMPLATE - CROP / 2.) * scale
    p[:, 0] *= np.cos(np.radians(yaw)) ** .5            # foreshortened width
    p[2, 0] += np.sin(np.radians(yaw)) * 38. * scale     # nose leads the turn
    p[:, 1] *= np.cos(np.radians(pitch)) ** .5           # foreshortened height
    p[2, 1] += np.sin(np.radians(pitch)) * 20. * scale
    t = np.radians(-roll)       # chin to +x for positive roll (see roll_from_kps)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32)
    p = p @ r.T + np.asarray(centre, np.float32)
    if jitter is not None:
        p = p + jitter
    return p.astype(np.float32)


def bbox_of(kps, pad=.55):
    lo, hi = kps.min(0), kps.max(0)
    s = (hi - lo).max() * pad
    return np.array([lo[0] - s, lo[1] - s, hi[0] + s, hi[1] + s], np.float32)


class Paste(MaskingMixin):
    def __init__(self):
        self.options = SimpleNamespace(blend_ratio=1., show_face_area_overlay=False)
        self._tls = threading.local()


OFFSETS = [0, 0, 0, 0, 12., 10.]
SKIN = np.array([120, 150, 200], np.float32)       # BGR
ENH = np.array([160, 170, 215], np.float32)        # enhancer tint (BGR)
PLATE = np.array([255, 0, 0], np.float32)          # saturated blue background
BLEED = np.array([0, 255, 0], np.uint8)            # green: must never appear


def silhouette(yaw, centre=(128, 128)):
    """Crop-space face silhouette; profile faces are narrower."""
    m = np.zeros((CROP, CROP), np.float32)
    ax = max(18, int(80 * max(np.cos(np.radians(abs(yaw))), .3)))
    cv2.ellipse(m, centre, (ax, 96), 0, 0, 360, 1., -1)
    return m


def crop_with_bleed(colour, sil):
    """Flat face colour inside the silhouette, saturated green outside it."""
    c = np.empty((CROP, CROP, 3), np.uint8)
    c[:] = BLEED
    c[sil > 0] = colour.astype(np.uint8)
    return c


def segment_residual(out, a, b):
    """Per-pixel distance from the segment a->b (a pure two-colour blend)."""
    d = (b - a)
    t = np.clip(((out.astype(np.float32) - a) @ d) / float(d @ d), 0., 1.)
    proj = a + t[..., None] * d
    return np.linalg.norm(out.astype(np.float32) - proj, axis=-1), t


def paste_scene(yaw, roll=0., swap_colour=SKIN, enh_colour=SKIN, plate_size=400,
                centre=(200., 200.), sil=None):
    kps = make_kps(roll=roll, yaw=yaw, centre=centre)
    M = estimate_norm(kps, CROP, "arcface")
    sil = silhouette(yaw) if sil is None else sil
    plate = np.empty((plate_size, plate_size, 3), np.uint8)
    plate[:] = PLATE.astype(np.uint8)
    # high-contrast checker so any plate resampling would show up
    plate[::16] = (0, 0, 0)
    plate[:, ::16] = (255, 255, 255)
    swap = crop_with_bleed(swap_colour, sil)
    enh = crop_with_bleed(enh_colour, sil)
    out = Paste().paste_upscale(swap, enh, M, plate.copy(), 1., OFFSETS,
                                face_kps=kps, crop_alpha=sil)
    foot = cv2.warpAffine(sil, cv2.invertAffineTransform(M),
                          (plate_size, plate_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    return out, plate, foot, M, kps, sil


# ==========================================================================
# 1. ORIENTATION STABILITY
# ==========================================================================
def _sequence(n=60, seed=7):
    """60 frames: roll sweeps -60..+60..-60, pitch +-30, yaw wobble."""
    rng = np.random.default_rng(seed)
    f = np.arange(n)
    roll = 60. * np.sin(2 * np.pi * f / n)
    pitch = 30. * np.sin(2 * np.pi * f / n * 1.5 + .6)
    yaw = 20. * np.sin(2 * np.pi * f / n * 2.)
    kps = [make_kps(roll[i], pitch[i], yaw[i],
                    centre=(200 + 25 * np.sin(i / 9.), 200 + 10 * np.cos(i / 7.)),
                    jitter=rng.normal(0, .35, (5, 2)).astype(np.float32))
           for i in range(n)]
    return roll, kps


def test_orientation_clean_sweep_has_no_180_and_no_chatter():
    roll, kps = _sequence()
    faces = [dict(kps=k.copy()) for k in kps]
    resolve_track_rolls(faces)
    actions = [f["rotation_action"] for f in faces]
    assert "rotate_180" not in actions, actions
    # A -60..+60 sweep may legitimately cross the 54.5 deg cardinal threshold
    # twice per side, but hysteresis forbids per-frame toggling.
    changes = sum(1 for a, b in zip(actions, actions[1:]) if a != b)
    assert changes <= 4, f"{changes} action changes: {actions}"
    # An action may only be active once the head really reached the band.
    for i, (r, a) in enumerate(zip(roll, actions)):
        recent = np.abs(roll[max(0, i - 8):i + 1]).max()
        if a is not None:
            # hysteresis + the 3-frame hold may lag a few frames behind the head
            assert recent > 50., (i, r, a)
        if abs(r) < 25. and recent < 50.:
            assert a is None, (i, r, a)
    # Resolved roll follows the truth (no 90/180 jump in the estimate itself).
    resolved = np.array([f["roll_deg"] for f in faces])
    assert np.abs([wrap180(a - b) for a, b in zip(resolved, roll)]).max() < 12.
    assert np.abs(np.diff(resolved)).max() < 15.


@pytest.mark.parametrize("outlier_frame", [5, 20, 31, 55])
def test_orientation_single_frame_landmark_outlier_never_flips(outlier_frame):
    roll, kps = _sequence()
    kps = [k.copy() for k in kps]
    centre = kps[outlier_frame].mean(0)
    # worst case: a coherent 180 deg misread (ordering stays valid)
    kps[outlier_frame] = (2 * centre - kps[outlier_frame]).astype(np.float32)
    faces = [dict(kps=k) for k in kps]
    resolve_track_rolls(faces)
    actions = [f["rotation_action"] for f in faces]
    assert "rotate_180" not in actions
    resolved = np.array([f["roll_deg"] for f in faces])
    assert np.abs(np.diff(resolved)).max() < 15., "roll estimate jumped on outlier"
    assert not faces[outlier_frame]["roll_trusted"]
    clean = [dict(kps=k.copy()) for k in _sequence()[1]]
    resolve_track_rolls(clean)
    # The outlier is untrusted, so it may delay an engage/release by a frame or
    # two (the 3-observation hold restarts) but must never swap the action.
    ref = [f["rotation_action"] for f in clean]
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(actions, ref)) if a != b]
    assert len(diff) <= 3, diff
    assert all(a is None or b is None for _, a, b in diff), diff


def test_orientation_borderline_tilt_with_outliers_does_not_seed_a_turn():
    # Steady 50 deg tilt (below 54.5) with two isolated 100 deg misreads.
    faces = [dict(kps=make_kps(50.)) for _ in range(30)]
    for i in (6, 17):
        faces[i] = dict(kps=make_kps(150.))
    resolve_track_rolls(faces)
    assert all(f["rotation_action"] is None for f in faces)


def test_orientation_genuine_upside_down_is_still_found():
    faces = [dict(kps=make_kps(180.)) for _ in range(8)]
    resolve_track_rolls(faces)
    assert [f["rotation_action"] for f in faces[:2]] == [None, None]   # needs 3
    assert faces[-1]["rotation_action"] == "rotate_180"


def test_affine_matrix_continuity_over_sequence():
    roll, kps = _sequence()
    geo = TrackGeometry()
    mats, rejected = [], 0
    rng = np.random.default_rng(3)
    outliers = {12: "scale", 25: "nan", 33: "shuffle", 47: "scale"}
    for i, k in enumerate(kps):
        k = k.copy()
        kind = outliers.get(i)
        if kind == "scale":
            k = ((k - k.mean(0)) * .25 + k.mean(0)).astype(np.float32)
        elif kind == "nan":
            k[1] = np.nan
        elif kind == "shuffle":
            k = k[[1, 0, 2, 4, 3]]
        try:
            pts = geo.update(1, i, k, bbox_of(kps[i]))
            m = estimate_norm(pts, CROP, "arcface")
            geo.validate_matrix(1, i, m)
        except InvalidAlignment:
            rejected += 1
            assert kind is not None, f"clean frame {i} was rejected"
            continue
        assert kind is None, f"outlier frame {i} ({kind}) produced a matrix"
        mats.append((i, m))
    assert rejected == len(outliers)
    for (i0, a), (i1, b) in zip(mats, mats[1:]):
        gap = i1 - i0
        ang = lambda m: np.degrees(np.arctan2(m[1, 0], m[0, 0]))
        scale = lambda m: np.sqrt(np.linalg.det(m[:, :2]))
        assert abs(wrap180(ang(b) - ang(a))) < 14. * gap, (i0, i1)
        assert .8 < scale(b) / scale(a) < 1.25, (i0, i1)
        assert np.linalg.norm(b[:, 2] - a[:, 2]) < 90 * gap
        assert np.linalg.det(b[:, :2]) > 0


@pytest.mark.parametrize("roll", [-60, -45, 0, 45, 60])
@pytest.mark.parametrize("pitch", [-30, 0, 30])
def test_every_pose_gives_a_valid_similarity(roll, pitch):
    k = make_kps(roll, pitch)
    m = estimate_norm(k, CROP, "arcface")
    affine(m)
    # real similarity: unit-ratio singular values, mapped points near template
    s = np.linalg.svd(m[:, :2], compute_uv=False)
    assert s[0] / s[1] < 1.02
    # pitch compresses only the vertical axis, so only allow a loose fit there
    mapped = k @ m[:, :2].T + m[:, 2]
    assert np.sqrt(((mapped - TEMPLATE) ** 2).sum(1).mean()) < (.02 if pitch == 0 else .12) * CROP


def test_partial_occlusion_displaced_landmarks_are_guarded():
    k = make_kps(10.)
    occluded = k.copy()
    occluded[3:] += np.array([0., -35.])      # hand over the mouth drags it up
    # either rejected cleanly, or accepted but still a valid, bounded matrix
    try:
        landmarks5(occluded, bbox_of(k))
        m = estimate_norm(occluded, CROP, "arcface")
    except InvalidAlignment:
        return
    affine(m)
    assert abs(wrap180(np.degrees(np.arctan2(m[1, 0], m[0, 0])) -
                       np.degrees(np.arctan2(*estimate_norm(k, CROP)[[1, 0], 0])))) < 30


# ==========================================================================
# 2. SIDE-ANGLE HALO PREVENTION
# ==========================================================================
@pytest.mark.parametrize("yaw", [-65, -50, -40, 40, 50, 65])
@pytest.mark.parametrize("roll", [0, 25])
def test_profile_paste_does_not_touch_plate_outside_silhouette(yaw, roll):
    out, plate, foot, *_ = paste_scene(yaw, roll)
    outside = foot <= 0
    np.testing.assert_array_equal(out[outside], plate[outside])
    assert np.any(out != plate), "nothing was pasted"


@pytest.mark.parametrize("yaw", [-65, -45, 40, 55, 65])
def test_profile_paste_has_no_background_bleed_or_replicated_border(yaw):
    out, plate, foot, M, kps, sil = paste_scene(yaw)
    # Every pasted pixel is a blend of plate and the face colour only. The crop
    # outside the silhouette is saturated green: any green means bleed.
    changed = np.any(out != plate, axis=-1)
    assert changed.any()
    px = out[changed].astype(np.float32)
    plate_px = plate[changed].astype(np.float32)
    # per-pixel two-colour model: out = plate + a*(SKIN - plate)
    d = SKIN - plate_px
    a = np.clip(((px - plate_px) * d).sum(1) / np.maximum((d * d).sum(1), 1e-6), 0, 1)
    resid = np.linalg.norm(px - (plate_px + a[:, None] * d), axis=1)
    assert resid.max() < 3.0, f"foreign colour in output (max residual {resid.max():.1f})"
    assert not np.any(changed & (foot <= 0))


def test_content_outside_alpha_cannot_influence_output():
    """Border-replication guard: differing ONLY where alpha==0, identical out."""
    yaw = 55
    sil = silhouette(yaw)
    kps = make_kps(yaw=yaw)
    M = estimate_norm(kps, CROP, "arcface")
    plate = np.full((400, 400, 3), 90, np.uint8)
    a = crop_with_bleed(SKIN, sil)
    b = a.copy()
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 255, a.shape, dtype=np.uint8)
    b[sil == 0] = noise[sil == 0]
    # also replicate-style border: stripes in the outermost rows/cols
    b[:3] = 255
    b[:, -3:] = 0
    o1 = Paste().paste_upscale(a, a, M, plate.copy(), 1., OFFSETS, face_kps=kps, crop_alpha=sil)
    o2 = Paste().paste_upscale(b, b, M, plate.copy(), 1., OFFSETS, face_kps=kps, crop_alpha=sil)
    np.testing.assert_array_equal(o1, o2)


@pytest.mark.parametrize("yaw", [40, 65])
def test_premultiplied_warp_has_zero_border_and_exact_colour(yaw):
    sil = silhouette(yaw)
    content = crop_with_bleed(SKIN, sil)
    M = estimate_norm(make_kps(yaw=yaw), CROP, "arcface")
    colour, cov = warp_premultiplied(content, sil, cv2.invertAffineTransform(M), (400, 400))
    assert cov.max() <= 1.0 + 1e-5 and cov.min() >= 0
    live = cov > 1e-3
    np.testing.assert_allclose(colour[live], np.tile(SKIN, (live.sum(), 1)), atol=1e-2)
    foot = cv2.warpAffine(sil, cv2.invertAffineTransform(M), (400, 400))
    assert not np.any((cov > 0) & (cv2.dilate((foot > 0).astype(np.uint8), np.ones((3, 3))) == 0))


@pytest.mark.parametrize("yaw", [40, 52, 65])
def test_far_cheek_erosion_is_directional_and_inward_only(yaw):
    sil = silhouette(yaw)
    for s in (1, -1):
        a = inward_alpha(sil, 10., s * yaw)
        assert np.all(a[sil == 0] == 0)
        assert np.all(a <= sil + 1e-6)
    right = inward_alpha(sil, 10., +yaw)
    left = inward_alpha(sil, 10., -yaw)
    assert right.sum() < sil.sum() and left.sum() < sil.sum()
    # mirror symmetry of the proxy: eroding the opposite side by the same amount
    mirrored = inward_alpha(sil[:, ::-1].copy(), 10., -yaw)[:, ::-1]
    np.testing.assert_allclose(right, mirrored, atol=1e-5)
    # Far cheek (+yaw => image right) carries less alpha than the near cheek.
    # (Total mass is not compared with the frontal case: the feather also
    # narrows at profile, which legitimately adds mass near the near edge.)
    # The designed inset is small (<= 2.5 % of the crop at the far edge), so
    # assert direction and a measurable edge inset, not a large ratio.
    assert right[:, 128:].sum() < right[:, :128].sum() - 10.
    assert left[:, :128].sum() < left[:, 128:].sum() - 10.
    row = sil[128] > 0
    xs = np.nonzero(row)[0]
    far_edge, near_edge = xs[-1] - 3, xs[0] + 3        # 3 px inside each edge
    assert right[128, far_edge] < right[128, near_edge]
    front = inward_alpha(sil, 10., 0.)
    np.testing.assert_allclose(front, front[:, ::-1], atol=.3)   # frontal ~ symmetric


def test_yaw_proxy_matches_construction():
    prev = 0.
    for y in (-65, -40, -10, 10, 40, 65):
        v = yaw_from_kps(make_kps(yaw=y))
        assert np.sign(v) == np.sign(y)
    vals = [yaw_from_kps(make_kps(yaw=y)) for y in (40, 50, 65)]
    assert vals == sorted(vals)


@pytest.mark.parametrize("yaw", [-60, 45, 65])
def test_single_alpha_boundary_with_swapper_and_enhancer(yaw):
    """Swapper and enhancer both on: one alpha, one transition, no swap colour
    left on the edge."""
    out, plate, foot, M, kps, sil = paste_scene(yaw, swap_colour=SKIN, enh_colour=ENH)
    changed = np.any(out != plate, axis=-1)
    px = out[changed].astype(np.float32)
    pp = plate[changed].astype(np.float32)
    d = ENH - pp
    a = np.clip(((px - pp) * d).sum(1) / np.maximum((d * d).sum(1), 1e-6), 0, 1)
    resid = np.linalg.norm(px - (pp + a[:, None] * d), axis=1)
    # every edge pixel is plate<->ENHANCED. A second boundary (swap pasted first
    # then enhancer on a different alpha) would leave SKIN-tinted rings.
    assert resid.max() < 3.0, f"swap-only colour on the edge ({resid.max():.1f})"
    # alpha along each scanline crossing the face is unimodal (single boundary)
    cov = np.zeros(out.shape[:2], np.float32)
    flat = changed.nonzero()
    if len(flat[0]):
        full = np.zeros(out.shape[:2], np.float32)
        d_img = ENH - plate.astype(np.float32)
        full = np.clip(((out.astype(np.float32) - plate) * d_img).sum(-1) /
                       np.maximum((d_img ** 2).sum(-1), 1e-6), 0, 1) * changed
        cov = full
    ys, xs = np.nonzero(foot > .5)
    for y in range(ys.min() + 8, ys.max() - 8, 12):
        row = cov[y]
        nz = np.nonzero(row > .02)[0]
        if len(nz) < 6:
            continue
        seg = row[nz[0]:nz[-1] + 1]
        peak = int(np.argmax(seg))
        rise, fall = np.diff(seg[:peak + 1]), np.diff(seg[peak:])
        # monotone up then monotone down, tolerance for 8-bit rounding + checker
        assert rise.min() > -.08 and fall.max() < .08, f"extra boundary at row {y}"
    deep = cv2.erode((foot > .995).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    assert deep.sum() > 200
    # deep interior is exactly the enhanced face (one alpha, fully opaque)
    np.testing.assert_allclose(out[deep].astype(np.float32).mean(0), ENH, atol=1.5)
    np.testing.assert_allclose(out[deep].astype(np.float32).std(0), 0, atol=1.5)


def test_double_paste_negative_control_is_detected():
    """The residual metric must be capable of failing: pasting swap and
    enhancer separately on different alphas leaves a foreign ring."""
    yaw = 50
    sil_small = silhouette(yaw)
    sil_big = cv2.dilate(sil_small, np.ones((15, 15), np.uint8))
    kps = make_kps(yaw=yaw)
    M = estimate_norm(kps, CROP, "arcface")
    plate = np.full((400, 400, 3), PLATE, np.uint8)
    p = Paste()
    mid = p.paste_upscale(crop_with_bleed(SKIN, sil_big), crop_with_bleed(SKIN, sil_big),
                          M, plate.copy(), 1., OFFSETS, face_kps=kps, crop_alpha=sil_big)
    out = p.paste_upscale(crop_with_bleed(ENH, sil_small), crop_with_bleed(ENH, sil_small),
                          M, mid, 1., OFFSETS, face_kps=kps, crop_alpha=sil_small)
    changed = np.any(out != plate, axis=-1)
    px, pp = out[changed].astype(np.float32), plate[changed].astype(np.float32)
    d = ENH - pp
    a = np.clip(((px - pp) * d).sum(1) / np.maximum((d * d).sum(1), 1e-6), 0, 1)
    resid = np.linalg.norm(px - (pp + a[:, None] * d), axis=1)
    assert resid.max() > 10., "negative control not detected: metric is vacuous"


# ==========================================================================
# 3. EDGE CASES
# ==========================================================================
def test_zero_face_frames_are_noops():
    assert resolve_track_rolls([]) == 0
    plate = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    empty = np.zeros((CROP, CROP), np.float32)
    out = Paste().paste_upscale(crop_with_bleed(SKIN, empty), crop_with_bleed(SKIN, empty),
                                np.array([[1., 0, 0], [0, 1., 0]]), plate.copy(), 1., OFFSETS,
                                crop_alpha=empty)
    np.testing.assert_array_equal(out, plate)
    assert inward_alpha(empty, 8., 40.).sum() == 0
    c, a = warp_premultiplied(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.float32),
                              np.eye(2, 3), (16, 16))
    assert a.sum() == 0 and c.sum() == 0
    for bad in (None, np.zeros((0, 2)), np.zeros((5, 2))):
        if bad is None or len(bad) == 0:
            continue
        with pytest.raises(InvalidAlignment):
            landmarks5(bad)
    g = TrackGeometry()
    assert g.update(None, None, make_kps(), None).shape == (5, 2)   # untracked passthrough


@pytest.mark.parametrize("centre", [(10, 200), (390, 200), (200, 5), (-200, -200), (620, 200)])
def test_partially_or_fully_out_of_bounds_faces(centre):
    out, plate, foot, M, kps, sil = paste_scene(0, centre=centre)
    assert out.shape == plate.shape and out.dtype == plate.dtype
    np.testing.assert_array_equal(out[foot <= 0], plate[foot <= 0])
    visible = foot > .99
    if visible.sum():       # in-frame part is painted, with no green replicate bleed
        # no pure-green replicate/bleed pixel anywhere in the frame
        assert not np.any(np.all(out == BLEED, axis=-1))
    if centre in {(-200, -200), (620, 200)}:
        assert visible.sum() == 0
        np.testing.assert_array_equal(out, plate)


def test_out_of_bounds_landmarks_vs_bbox_are_rejected():
    k = make_kps()
    with pytest.raises(InvalidAlignment):
        landmarks5(k + 900., bbox_of(k))
    ok = make_kps(centre=(2., 2.))        # partly off-frame but self-consistent
    landmarks5(ok, bbox_of(ok))


def test_multi_face_tracks_are_isolated_and_stable():
    n, ids = 40, (1, 2, 3)
    paths = {1: lambda i: (80 + 2 * i, 120), 2: lambda i: (250, 90 + 3 * i),
             3: lambda i: (330 - 2 * i, 300)}
    solo = {}
    for tid in ids:                                   # each track alone
        g, outs = TrackGeometry(), []
        for i in range(n):
            k = make_kps(centre=paths[tid](i), scale=.5 + .1 * tid)
            outs.append(g.update(tid, i, k, bbox_of(k)))
        solo[tid] = outs
    shared, inter = TrackGeometry(), {t: [] for t in ids}
    for i in range(n):                                # interleaved + threaded
        def run(tid, i=i):
            k = make_kps(centre=paths[tid](i), scale=.5 + .1 * tid)
            inter[tid].append(shared.update(tid, i, k, bbox_of(k)))
        ts = [threading.Thread(target=run, args=(t,)) for t in ids]
        [t.start() for t in ts]; [t.join() for t in ts]
    for tid in ids:
        np.testing.assert_allclose(np.array(inter[tid]), np.array(solo[tid]), atol=1e-5)
    # per-track roll resolution is independent of the other tracks
    seqs = {t: [dict(kps=make_kps(roll=(t - 2) * 20, centre=paths[t](i))) for i in range(n)]
            for t in ids}
    for t in ids:
        resolve_track_rolls(seqs[t])
        assert all(f["rotation_action"] is None for f in seqs[t])


def test_crossing_faces_swapping_positions_do_not_blend_tracks():
    g = TrackGeometry()
    a0, b0 = make_kps(centre=(100, 200)), make_kps(centre=(300, 200))
    g.update(1, 0, a0, bbox_of(a0)); g.update(2, 0, b0, bbox_of(b0))
    # ids re-assigned to the opposite positions: large displacement => hard reset
    out1 = g.update(1, 1, b0, bbox_of(b0))
    out2 = g.update(2, 1, a0, bbox_of(a0))
    np.testing.assert_allclose(out1, b0, atol=1e-4)
    np.testing.assert_allclose(out2, a0, atol=1e-4)


def test_mask_stabiliser_isolates_tracks_and_streams():
    from roop.one_euro import MaskStabilizer
    st = MaskStabilizer()
    zero, one = np.zeros((64, 64), np.float32), np.ones((64, 64), np.float32)
    kp = TEMPLATE * .25
    for t in range(4):
        np.testing.assert_array_equal(st.apply(zero, kp, t, track_id=1, stream_key="xseg"), zero)
        np.testing.assert_array_equal(st.apply(one, kp, t, track_id=2, stream_key="xseg"), one)
    assert len(st.tracks) == 2


def test_fp16_numeric_edges():
    f16 = np.float16
    big = unit_latent(np.full(512, 65504, f16))            # fp16 max; would overflow in fp16 norm
    assert big.dtype == np.float32 and np.linalg.norm(big) == pytest.approx(1., abs=1e-5)
    tiny = unit_latent(np.full(512, 6e-8, f16))            # fp16 subnormal
    assert tiny.dtype == np.float32 and np.linalg.norm(tiny) == pytest.approx(1., abs=1e-4)
    mixed = np.zeros(512, f16); mixed[0] = 65504; mixed[1] = f16(6e-8)
    assert np.isfinite(unit_latent(mixed)).all()
    for bad in (np.zeros(512, f16), np.full(512, np.nan, f16),
                np.full(512, np.inf, f16), np.full(512, -np.inf, f16)):
        with pytest.raises(ValueError):
            unit_latent(bad)
    # fp16 landmarks / masks / content flow through the geometry + matte code
    k16 = make_kps().astype(f16)
    assert landmarks5(k16).dtype == np.float32
    with pytest.raises(InvalidAlignment):
        landmarks5(np.full((5, 2), 70000., np.float32).astype(f16))     # overflows to inf
    sil16 = silhouette(50).astype(f16)
    a = inward_alpha(sil16.astype(np.float32), 8., 50.)
    assert a.dtype == np.float32 and np.isfinite(a).all()
    assert mask2d(sil16, (CROP, CROP)).dtype == np.float32
    dirty = silhouette(0); dirty[3, 3] = np.nan; dirty[4, 4] = np.inf; dirty[5, 5] = 7.
    clean = mask2d(dirty, (CROP, CROP))
    assert np.isfinite(clean).all() and clean.min() >= 0 and clean.max() <= 1
    c, cov = warp_premultiplied(crop_with_bleed(SKIN, silhouette(0)).astype(f16),
                                silhouette(0).astype(f16),
                                cv2.invertAffineTransform(estimate_norm(make_kps(), CROP)),
                                (400, 400))
    assert np.isfinite(c).all() and np.isfinite(cov).all()


# ==========================================================================
# optional: a REAL swapper ONNX output through the same production paste
# ==========================================================================
def _real_swapper():
    if os.environ.get("ROOP_VERIFY_REAL", "1") == "0":
        return None
    path = APP / "models" / "hyperswap_1a_256.onnx"
    if not path.exists():
        return None
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [i.name for i in s.get_inputs()]
        return s if names == ["source", "target"] else None
    except Exception:
        return None


@pytest.mark.parametrize("yaw", [45, 65])
def test_real_onnx_swapper_output_pastes_without_halo(yaw):
    sess = _real_swapper()
    if sess is None:
        pytest.skip("hyperswap ONNX / onnxruntime not available")
    sil = silhouette(yaw)
    kps = make_kps(yaw=yaw)
    M = estimate_norm(kps, CROP, "arcface")
    plate = np.full((400, 400, 3), 110, np.uint8)
    plate[::16] = 0
    aligned = cv2.warpAffine(plate, M, (CROP, CROP), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    aligned[sil > 0] = SKIN.astype(np.uint8)
    latent = unit_latent(np.random.default_rng(2).normal(size=(1, 512)))
    blob = (aligned[..., ::-1].astype(np.float32) / 255.).transpose(2, 0, 1)[None]
    fake = sess.run(None, {"source": latent.astype(np.float32), "target": blob})[0][0]
    assert np.isfinite(fake).all()
    fake = np.clip(fake.transpose(1, 2, 0)[..., ::-1] * 255., 0, 255).astype(np.uint8)
    out = Paste().paste_upscale(fake, fake, M, plate.copy(), 1., OFFSETS,
                                face_kps=kps, crop_alpha=sil)
    foot = cv2.warpAffine(sil, cv2.invertAffineTransform(M), (400, 400))
    np.testing.assert_array_equal(out[foot <= 0], plate[foot <= 0])
    assert np.any(out != plate)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"] + sys.argv[1:]))
