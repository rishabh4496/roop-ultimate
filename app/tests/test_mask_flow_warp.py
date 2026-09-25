"""MaskStabilizer's flow-warp (setting `mask_flow_warp`) and the XSeg guided
filter (`mask_guided_filter`). Pure CPU; no model is loaded."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.one_euro import MaskStabilizer          # noqa: E402
from roop import procmgr_masking as pm             # noqa: E402

KPS = np.array([[90, 110], [166, 110], [128, 150], [98, 190], [158, 190]],
               np.float32)


def _scene(shift, size=256):
    """A textured crop and its mask with a vertical edge at 128 + shift."""
    rng = np.random.default_rng(3)
    tex = cv2.resize(rng.integers(0, 255, (32, 32), dtype=np.uint8),
                     (size * 2, size), interpolation=cv2.INTER_CUBIC)
    x0 = 64 - shift
    crop = cv2.cvtColor(tex[:, x0:x0 + size], cv2.COLOR_GRAY2BGR)
    mask = np.zeros((size, size, 1), np.float32)
    # texture column c lands at crop x = c - x0 = c - 64 + shift: content moves
    # +shift, so the mask edge (texture column 192) sits at 128 + shift
    mask[:, 128 + shift:] = 1.0
    return crop, mask


def test_unit_channel_mask_keeps_shape_and_track():
    # XSeg returns (H, W, 1). cv2.remap drops the unit channel, and before the
    # reshape the blend broadcast to (H, W, H) and the track stopped matching
    # on every other frame.
    st = MaskStabilizer(strength=0.5, motion_beta=0.0, flow_warp=True)
    for t in range(6):
        crop, mask = _scene(2 * t)
        out = st.apply(mask, KPS, t, guide=crop)
        assert out.shape == mask.shape
    assert len(st.tracks) == 1
    assert st.flow_stats['applied'] == 5
    assert st.flow_stats['declined'] == 0


def test_flow_warp_follows_a_moving_edge_better_than_plain_ema():
    errs = {}
    for flow in (False, True):
        st = MaskStabilizer(strength=0.5, motion_beta=0.0, flow_warp=flow)
        e = []
        for t in range(10):
            crop, mask = _scene(3 * t)
            out = st.apply(mask, KPS, t, guide=crop)
            e.append(float(np.abs(out - mask).mean()))
        errs[flow] = np.mean(e[3:])
    assert errs[True] < errs[False] * 0.6, errs


def test_flow_warp_off_is_the_legacy_arithmetic():
    a = MaskStabilizer(strength=0.5, motion_beta=0.0)
    b = MaskStabilizer(strength=0.5, motion_beta=0.0, flow_warp=False)
    for t in range(5):
        crop, mask = _scene(3 * t)
        oa = a.apply(mask, KPS, t)
        ob = b.apply(mask, KPS, t, guide=crop)
        assert np.array_equal(oa, ob)
    assert b.flow_summary_line() is None


def test_gap_declines_instead_of_warping_across_it():
    st = MaskStabilizer(strength=0.5, motion_beta=0.0, flow_warp=True)
    for t in (0, 1, 3):
        crop, mask = _scene(t)
        st.apply(mask, KPS, t, guide=crop)
    assert st.flow_stats == {'applied': 1, 'declined': 1, 'reset': 0}


def test_shared_stats_cover_every_instance():
    stats = {'applied': 0, 'declined': 0, 'reset': 0}
    for _ in range(2):
        st = MaskStabilizer(strength=0.5, motion_beta=0.0, flow_warp=True,
                            flow_stats=stats)
        for t in range(3):
            crop, mask = _scene(t)
            st.apply(mask, KPS, t, guide=crop)
    assert stats['applied'] == 4
    assert 'applied 4/4' in st.flow_summary_line()


def test_guided_filter_leaves_flat_regions_and_shape_alone():
    crop, mask = _scene(0)
    out = pm.refine_mask_edges(mask, crop)
    assert out.shape == mask.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # far from the edge the mask is flat, so the filter is the identity there
    assert np.allclose(out[:, :100], 0.0, atol=1e-4)
    assert np.allclose(out[:, 160:], 1.0, atol=1e-4)
    flat = np.full((256, 256), 0.25, np.float32)
    assert np.allclose(pm.refine_mask_edges(flat, crop), 0.25, atol=1e-4)


def test_guided_filter_is_gated_by_the_setting():
    import roop.globals as g
    prev = getattr(g, 'mask_guided_filter', False)
    try:
        g.mask_guided_filter = False
        assert pm.mask_guided_filter_enabled() is False
        g.mask_guided_filter = True
        assert pm.mask_guided_filter_enabled() is True
    finally:
        g.mask_guided_filter = prev
