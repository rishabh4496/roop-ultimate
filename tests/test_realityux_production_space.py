"""RealityUX in the shapes production hands it.

The Stage 7 tests fed the engine a 256 crop, a 256 mask, landmarks already in
crop pixels and a hand-written M -- a world with one coordinate space. In
production the crop is cut from the FRAME by a similarity `M`, the detector's
landmarks and boxes are FRAME pixels, and with BiSeNet on the mask is the
parser's 512 map over a 256 crop. Three defects lived in the gap, each measured
on real footage (d4.mp4, 112 faces):

  * neighbour protection drew the crop-space polygon straight into the 512 mask
    (half scale, wrong place): 3.4% of the swap region erased on average, 13 of
    112 faces lost >10%, worst 54%, many with no neighbour in the crop at all;
  * the mouth polygon was clipped to the crop's corner, so the adaptive mouth
    mask was empty on every real face;
  * pixel kernels differed 2x between the 3060 (256 mask) and the 4070 (512).
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import cv2
import numpy as np

from roop.enhancers.occlusion_governor import OcclusionGovernor
from roop.processors.Mask_RealityUX import Mask_RealityUX

CROP = 256


class Face:
    def __init__(self, bbox, lm68=None):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.landmark_3d_68 = lm68
        self.landmark_2d_106 = None
        self.kps = None
        self.normed_embedding = None


def box_in_mask(shape, M, bbox, pad=0.10):
    """Independent oracle: the padded box, mapped frame -> crop -> mask pixels."""
    h, w = shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    quad = np.array([[[x1 - px, y1 - py], [x2 + px, y1 - py],
                      [x2 + px, y2 + py], [x1 - px, y2 + py]]], np.float32)
    pts = cv2.transform(quad, M)[0] * np.array([w / CROP, h / CROP], np.float32)
    out = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(out, np.round(pts).astype(np.int32), 1)
    return out


def make_M(frame_xy, side):
    """Frame -> crop similarity that puts `frame_xy` (a face centre) at the
    crop centre with `side` frame pixels spanning the crop."""
    s = CROP / float(side)
    cx, cy = frame_xy
    return np.array([[s, 0, CROP / 2 - s * cx],
                     [0, s, CROP / 2 - s * cy]], dtype=np.float32)


def engine(parser_on, xseg_value=0.0, labels_value=1):
    r = Mask_RealityUX()
    r._executor = None
    r._parser_enabled = parser_on
    r._xseg.Run = MagicMock(return_value=np.full((CROP, CROP), xseg_value, np.float32))
    r._parser.RunLabels = MagicMock(
        return_value=np.full((512, 512), labels_value, np.int64))
    return r


def run(r, target, faces, M):
    crop = np.full((CROP, CROP, 3), 128, np.uint8)
    return r.Run(crop, "", target_face=target, frame_faces=faces, M=M)


# ── neighbour protection ─────────────────────────────────────────────────────

def test_far_neighbour_costs_the_target_nothing_at_either_mask_size():
    target = Face([600, 300, 800, 520])
    other = Face([1500, 300, 1700, 520])           # nowhere near the crop
    M = make_M((700, 410), 300)
    for parser_on in (False, True):
        m = run(engine(parser_on), target, [target, other], M)
        assert m.shape == ((512, 512) if parser_on else (256, 256))
        assert float(m.max()) == 0.0, f"parser={parser_on}: target face was erased"


def test_neighbour_lands_where_it_really_is_in_the_mask():
    target = Face([600, 300, 800, 520])
    other = Face([760, 320, 900, 500])             # overlaps the crop's right side
    M = make_M((700, 410), 300)
    for parser_on in (False, True):
        m = run(engine(parser_on), target, [target, other], M)
        h = m.shape[0]
        sc = h / float(CROP)
        want = box_in_mask(m.shape, M, other.bbox) > 0
        assert want.any()
        got = m > 0.5
        # protected pixels sit on the neighbour's mapped box, in mask pixels,
        # and the left half of the crop (the target) is untouched.
        assert (got & ~cv2.dilate(want.astype(np.uint8),
                                  np.ones((int(8 * sc),) * 2)).astype(bool)).sum() == 0
        assert not got[:, : int(0.3 * h)].any()
        assert got[int(0.5 * h), int(0.85 * h)]


def test_protected_share_of_the_crop_is_the_same_at_both_mask_sizes():
    target = Face([600, 300, 800, 520])
    other = Face([760, 320, 900, 500])
    M = make_M((700, 410), 300)
    share = {p: float((run(engine(p), target, [target, other], M) > 0.5).mean())
             for p in (False, True)}
    assert abs(share[False] - share[True]) < 0.01, share


def test_unmatched_target_protects_nobody():
    """A target that is not among the detections (a rebound copy) used to fall
    back to 'the first face is the target', filing the REAL target under
    non-targets and erasing its own face."""
    stranger = Face([2000, 2000, 2100, 2100])      # matches nothing
    a = Face([600, 300, 800, 520])
    b = Face([760, 320, 900, 500])
    resolved, non_targets = OcclusionGovernor().arbitrate_multi_face_tracks(
        detected_faces=[a, b], target_reference_embedding=None, target_face=stranger)
    assert resolved is None and non_targets == []


def test_duplicate_detection_is_not_a_neighbour():
    a = Face([600, 300, 800, 520])
    dup = Face([610, 305, 805, 525])               # same face, detected twice
    far = Face([1500, 300, 1700, 520])
    rebound = Face([604, 302, 802, 522])           # target copy, not the same object
    resolved, non_targets = OcclusionGovernor().arbitrate_multi_face_tracks(
        detected_faces=[a, dup, far], target_reference_embedding=None, target_face=rebound)
    assert resolved is a
    assert non_targets == [far]


# ── mouth ────────────────────────────────────────────────────────────────────

def mouth_landmarks_frame(centre, width, gap):
    """68 points; only the inner lip (60..67) matters. FRAME pixels."""
    pts = np.tile(np.asarray(centre, np.float32), (68, 1))
    cx, cy = centre
    pts[60] = [cx - width / 2, cy]
    pts[64] = [cx + width / 2, cy]
    for k, off in zip((61, 62, 63), (-width / 4, 0, width / 4)):
        pts[k] = [cx + off, cy - gap / 2]
    for k, off in zip((67, 66, 65), (-width / 4, 0, width / 4)):
        pts[k] = [cx + off, cy + gap / 2]
    return pts


def test_mouth_mask_lands_on_the_mouth_for_frame_space_landmarks():
    M = make_M((700, 410), 300)
    mouth_frame = (700.0, 480.0)
    face = Face([600, 300, 800, 520], mouth_landmarks_frame(mouth_frame, 60, 40))
    r = engine(False)
    crop = np.full((CROP, CROP, 3), 150, np.uint8)
    mx, my = cv2.transform(np.array([[mouth_frame]], np.float32), M)[0][0]
    crop[int(my) - 14:int(my) + 14, int(mx) - 20:int(mx) + 20] = 235    # teeth
    m = r.Run(crop, "", target_face=face, frame_faces=[face], M=M,
              restore_original_mouth=True, mouth_open_threshold=0.18)
    assert r._last_telemetry["mouth_restored"] is True
    assert m[int(my), int(mx)] > 0.4, "mouth region not kept"
    assert m[40, 128] == 0.0 and m[CROP - 8, CROP - 8] == 0.0, "mask leaked off the mouth"
    # The corner the old code collapsed the polygon into stays swappable.
    assert float(m[:, : int(mx) - 40].max()) == 0.0


# ── resolution independence and thread safety ────────────────────────────────

def test_occlusion_governance_edge_is_the_same_at_both_mask_sizes():
    """Kernels are stated at a 256 reference, so the soft edge covers the same
    share of the crop on the 3060 (256 mask) and the 4070 (512). Measured on the
    band where either mask is partial: 0.071 mean difference before, 0.033 after;
    the threshold sits between."""
    out = {}
    for parser_on in (False, True):
        r = engine(parser_on)
        xs = np.zeros((CROP, CROP), np.float32)
        xs[150:210, 90:160] = 0.9
        r._xseg.Run = MagicMock(return_value=xs)
        crop = np.full((CROP, CROP, 3), 128, np.uint8)
        m = r.Run(crop, "", enable_xseg_occlusion=True, occlusion_erosion=3)
        out[parser_on] = cv2.resize(m, (CROP, CROP), interpolation=cv2.INTER_AREA)
    a, b = out[False], out[True]
    band = ((a > 0.05) & (a < 0.95)) | ((b > 0.05) & (b < 0.95))
    assert float(np.abs(a - b)[band].mean()) < 0.05


def test_telemetry_is_per_thread():
    r = engine(False)
    seen = {}
    gate = threading.Barrier(2)

    def worker(name, mar_open):
        face = Face([600, 300, 800, 520],
                    mouth_landmarks_frame((700, 480), 60, 40 if mar_open else 1))
        gate.wait()
        r.Run(np.full((CROP, CROP, 3), 128, np.uint8), "", target_face=face,
              restore_original_mouth=True, mouth_open_threshold=0.18)
        gate.wait()                     # both have finished before either reads
        seen[name] = r._last_telemetry["mouth_restored"]

    ts = [threading.Thread(target=worker, args=("open", True)),
          threading.Thread(target=worker, args=("closed", False))]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert seen == {"open": True, "closed": False}


# ── the settings reach the engine that reads them ────────────────────────────

def test_surgical_settings_reach_roop_globals(monkeypatch):
    """Mask_RealityUX reads these off roop.globals; until _apply_surgical_settings
    existed only the /enhancer route wrote them, so the Face Swap and Settings
    controls were saved to config.yaml and never consumed."""
    import api
    import roop.globals as g

    class Cfg:
        restore_original_mouth = True
        mouth_open_threshold = 0.22
        enable_xseg_occlusion = True
        occlusion_erosion = 2

    monkeypatch.setattr(g, "CFG", Cfg(), raising=False)
    for k in ("restore_original_mouth", "mouth_open_threshold",
              "enable_xseg_occlusion", "occlusion_erosion"):
        monkeypatch.setattr(g, k, None, raising=False)

    api._apply_surgical_settings({})                       # config is the fallback
    assert (g.restore_original_mouth, g.mouth_open_threshold,
            g.enable_xseg_occlusion, g.occlusion_erosion) == (True, 0.22, True, 2)

    api._apply_surgical_settings({"restore_original_mouth": False,   # request wins, clamped
                                  "mouth_open_threshold": 9, "occlusion_erosion": "99",
                                  "enable_xseg_occlusion": False})
    assert (g.restore_original_mouth, g.mouth_open_threshold,
            g.enable_xseg_occlusion, g.occlusion_erosion) == (False, 0.5, False, 5)

    api._apply_surgical_settings({"mouth_open_threshold": "junk", "occlusion_erosion": None})
    assert (g.mouth_open_threshold, g.occlusion_erosion) == (0.18, 3)
