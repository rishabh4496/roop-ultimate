"""Expression & gaze fidelity (Stage 11): iris retention, micro-expression tension, wide-mouth teeth.

The faces here are synthetic on purpose: the property under test is "the output's iris is where the
PLATE's iris is while keeping the SWAP's colour", and that needs ground truth for both positions,
which real footage cannot give. Every assertion is on a measured pixel quantity.
"""
import math
import os
import sys

import cv2
import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import expression_fidelity as ef  # noqa: E402
from roop import oral_cavity as oc  # noqa: E402

D = 80.0                              # interocular distance, px
KPS = np.array([[120, 100], [200, 100], [160, 135], [130, 175], [190, 175]], np.float32)
SKIN = (150, 170, 205)                # BGR


def _eye(img, cx, cy, iris_dx, iris_color, pupil=18, lid_open=1.0):
    """Draw one eye: sclera ellipse, a coloured iris offset by iris_dx, a dark pupil."""
    ry = max(1, int(round(D * 0.075 * lid_open)))
    cv2.ellipse(img, (int(cx), int(cy)), (int(D * 0.17), ry), 0, 0, 360, (235, 235, 235), -1, cv2.LINE_AA)
    if lid_open > 0.2:
        ir = int(round(D * 0.075))
        cv2.circle(img, (int(cx + iris_dx), int(cy)), ir, iris_color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(cx + iris_dx), int(cy)), max(2, ir // 2), (pupil, pupil, pupil), -1, cv2.LINE_AA)


def _scene(plate_dx, swap_dx, swap_color=(200, 120, 40), plate_color=(40, 60, 90), lid_open=1.0):
    plate = np.full((220, 320, 3), SKIN, np.uint8)
    swap = np.full((220, 320, 3), SKIN, np.uint8)
    for (cx, cy) in KPS[:2]:
        _eye(plate, cx, cy, plate_dx, plate_color, lid_open=lid_open)
        _eye(swap, cx, cy, swap_dx, swap_color)
    return swap, plate


def _iris_centre(img, eye):
    """Centroid of the dark/coloured iris pixels in one eye (independent of the module under test)."""
    cx, cy = eye
    roi = img[int(cy) - 14:int(cy) + 15, int(cx) - 30:int(cx) + 31].astype(np.float32)
    lum = roi @ np.array([0.114, 0.587, 0.299], np.float32)
    wgt = np.clip(190.0 - lum, 0, None)
    wgt[wgt < 40] = 0
    ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    return (wgt * xs).sum() / wgt.sum() + int(cx) - 30, (wgt * ys).sum() / wgt.sum() + int(cy) - 14


# ── iris / pupil ─────────────────────────────────────────────────────────────

def test_locate_iris_finds_the_synthetic_iris():
    swap, plate = _scene(plate_dx=6, swap_dx=-5)
    cx, cy = KPS[0]
    pad = int(D * 0.21 * 1.7) + 3
    y0, x0 = int(cy) - pad, int(cx) - pad
    roi = plate[y0:y0 + 2 * pad + 1, x0:x0 + 2 * pad + 1]
    y = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    inner = ef._ellipse(y.shape, cx - x0, cy - y0, D * 0.21 * 0.78, D * 0.13 * 0.62, 1.0)
    found = ef.locate_iris(y, inner, D * ef.IRIS_R)
    assert found is not None
    assert abs(found[0] - (cx - x0 + 6)) < 1.5 and abs(found[1] - (cy - y0)) < 1.5


def test_strength_zero_is_the_same_array_untouched():
    swap, plate = _scene(6, -5)
    before = swap.copy()
    out = ef.retain_iris_pupil(swap, plate, KPS, 0.0)
    assert out is swap and (swap == before).all()
    assert ef.restore_micro_expression(swap, plate, KPS, 0.0) is swap and (swap == before).all()


def test_iris_moves_to_the_plate_gaze_and_keeps_the_swap_colour():
    swap, plate = _scene(plate_dx=7, swap_dx=-7)
    before_err = [abs(_iris_centre(swap, e)[0] - _iris_centre(plate, e)[0]) for e in KPS[:2]]
    out = ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0)
    after_err = [abs(_iris_centre(out, e)[0] - _iris_centre(plate, e)[0]) for e in KPS[:2]]
    assert min(before_err) > 10                                  # the gazes really did disagree
    assert max(after_err) < 3.0                                  # and now agree to within 3 px
    # identity: the iris is still the SWAP's colour (strong blue channel), not the plate's brown
    for (cx, cy) in KPS[:2]:
        px = out[int(cy), int(cx + 7 - 4)]                      # inside the moved iris, off the pupil
        assert px[0] > 1.5 * px[2] and px[0] > 90, px                 # still the swap's blue, not the plate's brown


def test_a_pale_plate_iris_does_not_brighten_the_swapped_one():
    """Regression, seen on real footage: the plate's irises were pale, the swap's dark, and blending the
    plate's luma washed the swapped eyes out. Brightness must stay the swap's.

    Both irises sit at the SAME place, so nothing is moved or erased and no iris edge shifts: only the
    brightness / contrast step acts, and the ring's mean luma is a clean measurement (when the iris moves,
    a one-pixel offset swamps it with sclera)."""
    swap, plate = _scene(plate_dx=0, swap_dx=0, swap_color=(45, 40, 35), plate_color=(125, 115, 105))

    def ring_mean(img, cx, cy, r0=3.5, r1=5.5):
        ys, xs = np.mgrid[int(cy) - 7:int(cy) + 8, int(cx) - 7:int(cx) + 8]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        keep = (d2 >= r0 * r0) & (d2 <= r1 * r1)
        return float(np.mean([_lum_at(img, x, y) for x, y in zip(xs[keep], ys[keep])]))

    out = ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0)
    assert (out != swap).any(), 'the pass found no iris and did nothing, so this test would check nothing'
    for cx, cy in KPS[:2]:
        swap_l, out_l, plate_l = (ring_mean(i, cx, cy) for i in (swap, out, plate))
        assert plate_l - swap_l > 55, (swap_l, plate_l)               # the two irises really differ
        assert abs(out_l - swap_l) < 0.1 * (plate_l - swap_l), (swap_l, out_l, plate_l)


def _lum_at(img, x, y):
    b, g, r = (float(v) for v in img[y, x])
    return 0.114 * b + 0.587 * g + 0.299 * r


def test_strength_interpolates_the_gaze_shift():
    swap, plate = _scene(plate_dx=7, swap_dx=-7)
    errs = []
    for s in (0.25, 0.5, 1.0):
        out = ef.retain_iris_pupil(swap.copy(), plate, KPS, s)
        errs.append(abs(_iris_centre(out, KPS[0])[0] - _iris_centre(plate, KPS[0])[0]))
    assert errs[0] > errs[2] and errs[1] > errs[2] - 0.5          # more strength never moves it farther


def test_pixels_outside_the_eye_windows_are_bit_exact():
    swap, plate = _scene(7, -7)
    out = ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0)
    keep = np.ones(swap.shape[:2], bool)
    for (cx, cy) in KPS[:2]:
        keep[int(cy) - 25:int(cy) + 26, int(cx) - 40:int(cx) + 41] = False
    assert (out[keep] == swap[keep]).all()


def test_a_blink_in_the_plate_leaves_the_swap_alone():
    swap, plate = _scene(7, -7, lid_open=0.0)                     # plate eyes closed: a flat lid line
    plate[:] = SKIN
    before = swap.copy()
    out = ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0)
    assert (out == before).all()


def test_gaze_pass_fades_out_on_a_turned_head_and_respects_ownership():
    swap, plate = _scene(7, -7)
    assert (ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0, yaw=40.0) == swap).all()
    half = ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0, yaw=31.0)
    full = ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0)
    assert 0 < np.abs(half.astype(int) - swap).sum() < np.abs(full.astype(int) - swap).sum()

    class Own:                                                    # this face owns none of the frame
        def crop(self, x0, y0, x1, y1):
            return np.zeros((y1 - y0, x1 - x0), np.float32)
    assert (ef.retain_iris_pupil(swap.copy(), plate, KPS, 1.0, region=Own()) == swap).all()


def test_degenerate_inputs_return_the_input():
    swap, plate = _scene(7, -7)
    tiny = np.array([[10, 10], [14, 10], [12, 12], [10, 14], [14, 14]], np.float32)
    assert ef.retain_iris_pupil(swap, plate, tiny, 1.0) is swap
    assert ef.retain_iris_pupil(swap, plate[:50], KPS, 1.0) is swap
    assert ef.retain_iris_pupil(swap, plate, None, 1.0) is swap
    assert ef.restore_micro_expression(swap, plate, tiny, 1.0) is swap


# ── micro-expression ─────────────────────────────────────────────────────────

def _creased_pair():
    """A plate with dark nasolabial creases and brow furrows, and a swap that is the same face
    flattened (the creases blurred away) - what an identity swapper hands back."""
    rng = np.random.default_rng(3)
    plate = np.full((260, 320, 3), 175, np.uint8)
    plate = np.clip(plate + rng.normal(0, 2, plate.shape), 0, 255).astype(np.uint8)
    u = np.array([1.0, 0.0])
    nose = KPS[2]
    for sign, corner in ((-1, KPS[3]), (1, KPS[4])):
        wing = nose + sign * 0.20 * D * u + np.array([0, 0.05 * D])
        bow = (wing + corner) / 2 + sign * 0.045 * D * u
        pts = np.round(np.array([wing, bow, corner])).astype(np.int32)
        cv2.polylines(plate, [pts.reshape(-1, 1, 2)], False, (110, 120, 140), 3, cv2.LINE_AA)
    for eye, inward in ((KPS[0], 1), (KPS[1], -1)):
        c = eye + np.array([inward * 0.22 * D, -0.30 * D])
        cv2.line(plate, tuple(np.round(c - [6, 0]).astype(int)), tuple(np.round(c + [6, 0]).astype(int)),
                 (115, 125, 145), 3, cv2.LINE_AA)
    swap = cv2.GaussianBlur(plate, (0, 0), 6.0)                   # creases gone, tone kept
    return swap, plate


def _band(img, d=D):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    return cv2.GaussianBlur(y, (0, 0), 0.02 * d) - cv2.GaussianBlur(y, (0, 0), 0.06 * d)


def _lum(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)


def _crease_depths(img):
    """Luma of each nasolabial crease relative to the skin 9 px either side of it (negative = dark
    line). Sampled on the polyline the fixture draws, so it is independent of the module under test."""
    y = _lum(img)
    out = []
    for sign, corner in ((-1, KPS[3]), (1, KPS[4])):
        wing = KPS[2] + sign * 0.20 * D * np.array([1, 0]) + np.array([0, 0.05 * D])
        bow = (wing + corner) / 2 + sign * 0.045 * D * np.array([1, 0])
        p = (bow + corner) / 2
        px, py = int(round(p[0])), int(round(p[1]))
        out.append(y[py, px] - 0.5 * (y[py, px - 9] + y[py, px + 9]))
    return np.array(out)


def _recovered(plate, swap, out):
    """Fraction of the crease depth the swap lost that the output got back (1 = all of it)."""
    lost = _crease_depths(plate) - _crease_depths(swap)
    return (_crease_depths(out) - _crease_depths(swap)) / lost


def _band_err(ref, img, m):
    """Relative band-pass error inside mask m (used only for RELATIVE comparisons: a DoG re-analysis
    of a transplanted band under-reads it, so the absolute value is not a depth)."""
    a, b = _band(ref)[m], _band(img)[m]
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def _region_mask():
    x0, y0, x1, y1 = 0, 0, 320, 260
    return ef.micro_expression_regions(KPS, (260, 320), (x0, y0, x1, y1))


def test_micro_expression_brings_the_creases_back_without_moving_tone_or_colour():
    swap, plate = _creased_pair()
    m = _region_mask() > 0.5
    assert m.sum() > 500
    out = ef.restore_micro_expression(swap.copy(), plate, KPS, 1.0)
    assert (_crease_depths(plate) < -40).all() and (_crease_depths(swap) > -15).all()   # fixture is honest
    rec = _recovered(plate, swap, out)
    assert (rec > 0.7).all() and (rec < 1.1).all(), rec          # the crease is back, and not overdone
    # tone: the region's mean luma stays put. A window that clips the DoG's bright halos darkens it
    # (6 levels at 0.11 d); the shipped window keeps the drift under 3.
    ya = cv2.cvtColor(swap, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    yb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    assert abs(yb[m].mean() - ya[m].mean()) < 3.0
    # colour: chroma is the swap's own, to rounding
    ca = cv2.cvtColor(swap, cv2.COLOR_BGR2YCrCb)[:, :, 1:].astype(int)
    cb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[:, :, 1:].astype(int)
    assert np.abs(ca - cb).max() <= 2


def test_micro_expression_is_monotone_in_strength_and_bit_exact_outside_its_regions():
    swap, plate = _creased_pair()
    m = _region_mask() > 0.5
    recs = []
    for s in (0.25, 0.5, 1.0):
        out = ef.restore_micro_expression(swap.copy(), plate, KPS, s)
        recs.append(_recovered(plate, swap, out).mean())
    assert recs[0] < recs[1] < recs[2]
    assert recs[1] == pytest.approx(0.5 * recs[2], abs=0.08)       # linear in strength
    out = ef.restore_micro_expression(swap.copy(), plate, KPS, 1.0)
    far = np.ones(swap.shape[:2], bool)
    far[ef.micro_expression_regions(KPS, (260, 320), (0, 0, 320, 260)) > 1e-4] = False
    far = cv2.erode(far.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)   # clear of any blur reach
    assert far.sum() > 10000 and (out[far] == swap[far]).all()


def test_micro_expression_follows_head_roll():
    swap, plate = _creased_pair()
    c = KPS.mean(axis=0)
    rot = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), 25.0, 1.0)
    kps_r = (np.hstack([KPS, np.ones((5, 1), np.float32)]) @ rot.T).astype(np.float32)
    swap_r = cv2.warpAffine(swap, rot, (320, 260), borderMode=cv2.BORDER_REPLICATE)
    plate_r = cv2.warpAffine(plate, rot, (320, 260), borderMode=cv2.BORDER_REPLICATE)
    m = ef.micro_expression_regions(kps_r, (260, 320), (0, 0, 320, 260)) > 0.5
    out = ef.restore_micro_expression(swap_r.copy(), plate_r, kps_r, 1.0)
    e0, e1 = _band_err(plate_r, swap_r, m), _band_err(plate_r, out, m)
    assert e1 < 0.6 * e0, (e0, e1)                                # a rolled head's regions still find the creases


# ── wide-mouth teeth ─────────────────────────────────────────────────────────

def _mouth_face(inner_gap):
    pts = np.zeros((106, 2), np.float32)
    cx, cy, half = 100.0, 100.0, 20.0
    top, bot = cy - inner_gap / 2, cy + inner_gap / 2
    pts[52], pts[61] = (cx - half, cy), (cx + half, cy)
    # (upper, lower) inner-lip pairs sit one above the other, as 2d106det places them
    for up, lo, x in ((66, 54, cx - 10), (62, 60, cx), (70, 57, cx + 10)):
        pts[up], pts[lo] = (x, top), (x, bot)
    pts[65], pts[69] = (cx - 14, top), (cx + 14, top)
    return {"landmark_2d_106": pts, "bbox": (60, 60, 140, 140)}


def _mouth_images(gap):
    """Plate: dark cavity with a bright upper tooth row. Swap: the SAME cavity flipped top-to-bottom,
    so it has identical variance and Laplacian energy (the clamped-lip / blurry-teeth detector sees
    nothing wrong, as with a clean swap) yet different content: the teeth are at the bottom."""
    top, bot = int(100 - gap / 2), int(100 + gap / 2) + 1
    teeth_rows = max(2, int(gap * 0.4))
    plate = np.full((200, 200, 3), (90, 100, 150), np.uint8)
    plate[top:bot, 82:119] = (45, 40, 70)
    plate[top:top + teeth_rows, 82:119] = (235, 235, 240)
    swap = plate.copy()
    swap[top:bot, 82:119] = plate[top:bot, 82:119][::-1]
    return swap, plate


def test_forced_mode_restores_the_cavity_on_a_wide_mouth_only():
    gap = 16.0                                                     # 16 / 40 = 0.40 of the mouth width
    swap, plate = _mouth_images(gap)
    face = _mouth_face(gap)
    assert oc._inner_gap_ratio(face) >= oc.WIDE_GAP_RATIO
    default = oc.reconstruct_inner_mouth_geometry(swap.copy(), plate, face)
    assert (default == swap).all()                                 # swap not broken: default leaves it
    forced = oc.reconstruct_inner_mouth_geometry(swap.copy(), plate, face, force_wide=True)
    teeth = (slice(93, 97), slice(92, 108))                          # the plate's tooth row (top of the aperture)
    assert forced[teeth].mean() > swap[teeth].mean() + 25            # the target's teeth are back
    # outside the mouth box: untouched
    assert (forced[:80] == swap[:80]).all()


def test_forced_mode_ignores_a_narrow_or_closed_mouth():
    for gap in (3.0, 6.0):                                         # 0.075 and 0.15 of the width
        swap, plate = _mouth_images(gap)
        face = _mouth_face(gap)
        assert oc._inner_gap_ratio(face) < oc.WIDE_GAP_RATIO
        out = oc.reconstruct_inner_mouth_geometry(swap.copy(), plate, face, force_wide=True)
        assert (out == swap).all(), gap


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
