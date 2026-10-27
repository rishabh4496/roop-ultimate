"""Vector roto zones (Stage 11): outline, 2-point pin, matte and the two composites.

Every assertion is on pixels or coordinates that a wrong implementation would get wrong; none is
on a call count. The pin tests build a head at two poses and require the zone to keep its place
RELATIVE TO THE FACE, which is the whole point of the pin, rather than checking a matrix entry.
"""
import math
import os
import sys

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import roto_polygons as rp  # noqa: E402


def _poly(points, **kw):
    base = {"id": "a", "mode": "exclude", "points": points, "feather": 0, "frames": None, "pin": None}
    base.update(kw)
    return base


SQUARE = [[40, 40], [80, 40], [80, 80], [40, 80]]


# ── sanitize ─────────────────────────────────────────────────────────────────

def test_sanitize_drops_malformed_and_keeps_good():
    raw = [
        {"points": SQUARE, "mode": "include", "feather": 12},
        {"points": [[1, 2], [3, 4]]},                                    # two points: not a polygon
        {"points": [[0, 0], [1, 1], [float("nan"), 3], [2, 2]]},         # nan point dropped -> 3 left, kept
        {"points": SQUARE, "enabled": False},
        "nonsense",
        {"points": [[0, 0], [1, 1], ["x", 3]]},                          # unusable point -> 2 left: dropped
    ]
    out = rp.sanitize(raw)
    assert [p["mode"] for p in out] == ["include", "exclude"]
    assert out[0]["feather"] == 12.0
    assert len(out[1]["points"]) == 3


def test_sanitize_clamps_and_caps():
    huge = [{"points": SQUARE, "feather": 1e9}] * (rp.MAX_POLYGONS + 10)
    out = rp.sanitize(huge)
    assert len(out) == rp.MAX_POLYGONS
    assert all(p["feather"] == rp.MAX_FEATHER_PX for p in out)
    many = [{"points": [[i, i % 7] for i in range(rp.MAX_POINTS + 100)]}]
    assert len(rp.sanitize(many)[0]["points"]) == rp.MAX_POINTS
    assert rp.sanitize(None) == [] and rp.sanitize({"points": SQUARE}) == []


def test_sanitize_frame_range_and_pin_validation():
    good = {"points": SQUARE, "frames": [90, 10],
            "pin": {"ref_nose": [10, 10], "ref_chin": [10, 30], "source_index": 2}}
    p = rp.sanitize([good])[0]
    assert p["frames"] == (10, 90)
    assert p["pin"]["source_index"] == 2
    degenerate = dict(good, pin={"ref_nose": [5, 5], "ref_chin": [5, 5]})
    assert rp.sanitize([degenerate])[0]["pin"] is None
    off = dict(good, pin={"enabled": False, "ref_nose": [10, 10], "ref_chin": [10, 30]})
    assert rp.sanitize([off])[0]["pin"] is None


# ── flatten ──────────────────────────────────────────────────────────────────

def test_straight_polygon_flattens_to_its_vertices():
    pts = rp.sanitize([{"points": SQUARE}])[0]["points"]
    assert np.allclose(rp.flatten(pts), np.array(SQUARE, np.float32))


def test_bezier_segment_matches_the_cubic_formula():
    pts = [{"x": 0.0, "y": 0.0, "hin": None, "hout": (30.0, 0.0)},
           {"x": 100.0, "y": 0.0, "hin": (-30.0, -40.0), "hout": None},
           {"x": 50.0, "y": 90.0, "hin": None, "hout": None}]
    flat = rp.flatten(pts, steps=16)
    # 16 samples on the curved segment (anchor + 15 interior), then the two straight segments' anchors
    assert flat.shape == (16 + 1 + 1, 2)
    t = 8 / 16
    p0, c1, c2, p3 = np.array([0, 0]), np.array([30, 0]), np.array([70, -40]), np.array([100, 0])
    expect = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3
    assert np.allclose(flat[8], expect, atol=1e-4)
    assert np.allclose(flat[0], p0) and np.allclose(flat[16], p3)


# ── anchors and the 2-point pin ──────────────────────────────────────────────

class _Face:
    def __init__(self, kps, lm=None):
        self.kps = np.asarray(kps, np.float32)
        self.landmark_2d_106 = lm


def _kps(cx, cy, scale=1.0, roll=0.0):
    """Five ArcFace keypoints for a head centred at (cx, cy), rotated by `roll` degrees."""
    base = np.array([[-30, -20], [30, -20], [0, 5], [-22, 38], [22, 38]], np.float32) * scale
    r = math.radians(roll)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]], np.float32)
    return base @ rot.T + np.array([cx, cy], np.float32)


def test_face_anchors_uses_106_chin_when_present_and_extrapolates_otherwise():
    k = _kps(100, 100)
    nose, chin = rp.face_anchors(_Face(k))
    assert np.allclose(nose, k[2])
    mouth = (k[3] + k[4]) / 2
    assert np.allclose(chin, mouth + (mouth - k[2]) * 1.1)
    lm = np.zeros((106, 2), np.float32)
    lm[16] = (100, 190)
    nose2, chin2 = rp.face_anchors(_Face(k, lm))
    assert np.allclose(chin2, (100, 190))
    assert rp.face_anchors(_Face(k[:3])) is None
    assert rp.face_anchors(object()) is None


def test_pin_keeps_the_zone_fixed_relative_to_the_face():
    ref = rp.face_anchors(_Face(_kps(200, 200)))
    # a zone drawn on the cheek: the point's position in the FACE's own frame is what must not change

    def in_face_frame(pt, anchors):
        n, c = anchors
        axis = (c - n)
        axis_len = np.linalg.norm(axis)
        u = axis / axis_len
        v = np.array([-u[1], u[0]])
        d = pt - n
        return np.array([d @ u, d @ v]) / axis_len

    # translate, scale, roll, and all three at once
    for dx, dy, scale, roll in [(25, -10, 1.0, 0), (0, 0, 1.6, 0), (0, 0, 1.0, 40), (-40, 30, 0.7, -25)]:
        cur = rp.face_anchors(_Face(_kps(200 + dx, 200 + dy, scale, roll)))
        cheek = np.array([[236.0, 214.0]], np.float32)
        m = rp.pin_transform(ref[0], ref[1], cur[0], cur[1])
        moved = cheek @ m[:, :2].T + m[:, 2]
        case = (dx, dy, scale, roll)
        assert np.allclose(in_face_frame(cheek[0], ref), in_face_frame(moved[0], cur), atol=1e-3), case
        assert np.allclose(np.array([ref[0], ref[1]]) @ m[:, :2].T + m[:, 2], np.array([cur[0], cur[1]]),
                           atol=1e-3), case


def test_pin_transform_degenerate_returns_none():
    assert rp.pin_transform((1, 1), (1, 1), (0, 0), (5, 5)) is None
    assert rp.pin_transform((0, 0), (5, 5), (3, 3), (3, 3)) is None


def test_pinned_zone_follows_the_chosen_source_and_stays_put_without_a_face():
    ref_face = _Face(_kps(200, 200))
    rn, rc = rp.face_anchors(ref_face)
    poly = rp.sanitize([{"points": [[230, 200], [260, 200], [260, 230], [230, 230]],
                         "pin": {"ref_nose": rn.tolist(), "ref_chin": rc.tolist(), "source_index": 1}}])[0]
    near, far = rp.face_anchors(_Face(_kps(205, 200))), rp.face_anchors(_Face(_kps(400, 200)))
    # the face for source 1 is the FAR one: source index outranks proximity
    out, tracked = rp.placed_outline(poly, [(0, near), (1, far)])
    assert tracked and abs(out[:, 0].mean() - (245 + 200)) < 1.0
    # no source index: nearest nose wins
    poly2 = dict(poly, pin=dict(poly["pin"], source_index=None))
    out2, _ = rp.placed_outline(poly2, [(0, near), (1, far)])
    assert abs(out2[:, 0].mean() - (245 + 5)) < 1.0
    # no faces: the zone stays where it was drawn (and says it is not tracked)
    out3, tracked3 = rp.placed_outline(poly, [])
    assert not tracked3 and np.allclose(out3, rp.flatten(poly["points"]))


def test_active_frame_range_is_inclusive_and_unknown_frame_is_live():
    p = rp.sanitize([{"points": SQUARE, "frames": [10, 20]}])[0]
    assert rp.active(p, 10) and rp.active(p, 20) and not rp.active(p, 9) and not rp.active(p, 21)
    assert rp.active(p, None)
    assert rp.ui_frame(0, 0) == 1 and rp.ui_frame(4, 100) == 105 and rp.ui_frame(None) is None


# ── matte ────────────────────────────────────────────────────────────────────

def test_hard_matte_area_matches_the_polygon():
    m, x0, y0 = rp.polygon_matte(np.array(SQUARE, np.float32), (200, 200), 0)
    assert abs(m.sum() - 40 * 40) < 40          # anti-aliased edge, not exact
    assert m.max() == pytest.approx(1.0, abs=1e-3)
    assert m.shape[0] < 60 and m.shape[1] < 60   # an ROI, never the whole frame


def test_feather_softens_the_edge_without_moving_it():
    sq = np.array([[60, 60], [140, 60], [140, 140], [60, 140]], np.float32)
    hard, _, _ = rp.polygon_matte(sq, (200, 200), 0)
    soft, x0, y0 = rp.polygon_matte(sq, (200, 200), 20)
    assert soft.shape[0] > hard.shape[0]
    # the outline is the 50% line: the matte value on the edge midpoint
    assert soft[100 - y0, 60 - x0] == pytest.approx(0.5, abs=0.12)
    assert soft[100 - y0, 100 - x0] > 0.99                      # deep inside is still opaque
    # total mass is preserved by a symmetric blur, so the zone did not grow or shrink
    assert abs(soft.sum() - hard.sum()) / hard.sum() < 0.02


def test_matte_off_frame_or_degenerate_is_none():
    assert rp.polygon_matte(np.array([[300, 300], [320, 300], [320, 320]], np.float32), (100, 100), 0) is None
    assert rp.polygon_matte(np.zeros((2, 2), np.float32), (100, 100), 0) is None


# ── composites ───────────────────────────────────────────────────────────────

def _frames():
    plate = np.full((120, 160, 3), 200, np.uint8)
    result = np.full((120, 160, 3), 40, np.uint8)
    return result, plate


def test_exclusion_restores_the_plate_inside_and_nothing_outside():
    result, plate = _frames()
    before = result.copy()
    rp.apply_exclusions(result, plate, rp.sanitize([{"points": SQUARE}]))
    assert (result[50:70, 50:70] == 200).all()                  # well inside
    inside = np.zeros(result.shape[:2], bool)
    inside[36:84, 36:84] = True
    assert (result[~inside] == before[~inside]).all()           # bit-exact outside the zone


def test_exclusion_feather_is_a_monotone_ramp():
    result, plate = _frames()
    rp.apply_exclusions(result, plate, rp.sanitize([{"points": [[20, 20], [140, 20], [140, 100], [20, 100]],
                                                     "feather": 24}]))
    row = result[60, 0:40, 0].astype(int)
    assert row[0] < 52 and row[-1] > 150
    assert (np.diff(row) >= -1).all()                           # never steps back


def test_inclusion_cancels_the_exclusion_it_overlaps():
    result, plate = _frames()
    zones = rp.sanitize([
        {"points": [[20, 20], [140, 20], [140, 100], [20, 100]], "mode": "exclude"},
        {"points": [[60, 40], [100, 40], [100, 80], [60, 80]], "mode": "include"},
    ])
    rp.apply_exclusions(result, plate, zones)
    assert (result[50:70, 70:90] == 40).all()                   # the swap is back inside the inclusion
    assert (result[30:35, 30:130] == 200).all()                 # the rest of the exclusion stands


def test_zone_outside_its_frame_range_does_nothing():
    result, plate = _frames()
    before = result.copy()
    rp.apply_exclusions(result, plate, rp.sanitize([{"points": SQUARE, "frames": [5, 9]}]), frame_ui=10)
    assert (result == before).all()
    rp.apply_exclusions(result, plate, rp.sanitize([{"points": SQUARE, "frames": [5, 9]}]), frame_ui=7)
    assert (result[50:70, 50:70] == 200).all()


def test_exclusion_is_a_noop_when_result_is_the_plate_or_shapes_differ():
    result, plate = _frames()
    assert rp.apply_exclusions(plate, plate, rp.sanitize([{"points": SQUARE}])) is plate
    small = result[:50]
    assert rp.apply_exclusions(small, plate, rp.sanitize([{"points": SQUARE}])) is small
    assert (small == 40).all()


def test_pinned_exclusion_moves_with_the_face():
    ref = _Face(_kps(60, 60, 0.6))
    rn, rc = rp.face_anchors(ref)
    zone = [{"points": [[75, 55], [90, 55], [90, 70], [75, 70]],
             "pin": {"ref_nose": rn.tolist(), "ref_chin": rc.tolist()}}]
    shifted = rp.face_anchors(_Face(_kps(60 + 40, 60 + 20, 0.6)))
    result, plate = _frames()
    rp.apply_exclusions(result, plate, rp.sanitize(zone), faces=[(0, shifted)])
    assert (result[78:88, 118:128] == 200).all()                # zone landed 40 right, 20 down
    assert (result[58:68, 78:88] == 40).all()                   # and left its drawn position


def test_inclusion_for_crop_lands_where_the_affine_says():
    zones = rp.sanitize([{"points": [[100, 100], [140, 100], [140, 140], [100, 140]], "mode": "include"}])
    M = np.array([[0.5, 0, -40], [0, 0.5, -40]], np.float32)       # frame -> 128x128 crop
    crop = rp.inclusion_for_crop(zones, (300, 300), (128, 128), M)
    assert crop is not None and crop.shape == (128, 128)
    assert crop[20:30, 20:30].min() > 0.95                         # zone centre (120,120) -> (20,20)... inside
    assert crop[0:5, 0:5].max() < 0.05
    assert rp.inclusion_for_crop(rp.sanitize([{"points": SQUARE, "mode": "exclude"}]), (300, 300), (128, 128), M) is None
    assert rp.inclusion_for_crop([], (300, 300), (128, 128), M) is None


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
