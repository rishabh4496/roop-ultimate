"""Face-swap seam fixes, driven through the real code paths (process_mask, the watchdog, the colour mixin,
Mask_RealityUX.Run, paste_upscale) rather than their helpers. Companion to ``test_face_seam_defects.py``.
"""
import os
import sys
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import roop.globals as _g                                           # noqa: E402
from roop import watchdog                                           # noqa: E402
from roop.procmgr_masking import MaskingMixin                       # noqa: E402
from roop.processors import Mask_RealityUX as _rux                  # noqa: E402
from test_face_seam_defects import _skin_crop, _disk                # noqa: E402
import test_mask_and_blend as _tb                                   # noqa: E402


# --------------------------------------------------------------------------- #
# The occluder family through the real process_mask: no cliff at the old 0.35 step
# --------------------------------------------------------------------------- #
class _StubOccluder:
    processorname = 'mask_occluder'
    type = 'mask'
    pool = None

    def __init__(self, prob):
        self.prob = prob

    def Run(self, frame, keywords):
        return self.prob.copy()


class _MaskMgr(MaskingMixin):
    def __init__(self):
        self.options = SimpleNamespace(blend_ratio=1., show_face_area_overlay=False, masking_text='')
        self._tls = threading.local()
        self._stab_active = False

    def _temporal_engine(self, name):
        return None

    def _cur_mask_stab(self):
        return None


def _occluder_mask(prob_value):
    frame = np.full((256, 256, 3), 120, np.uint8)
    target = np.full((256, 256, 3), 200, np.uint8)
    prob = np.full((256, 256), prob_value, np.float32)
    M = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
    _, mask = _MaskMgr().process_mask(_StubOccluder(prob), frame, target, orig_frame=None,
                                      target_face=None, M=M, defer_composite=True)
    return np.asarray(mask, np.float32)


def test_occluder_mask_is_continuous_across_the_old_threshold():
    """Pre-change: `(p > 0.35)` -> a probability of 0.349 gave 0.0 and 0.351 gave 1.0 (a jump of 1.0), so noise
    around the threshold popped hand<->face. Now a 0.002 change in p moves the mask by a few hundredths."""
    lo, hi = _occluder_mask(0.349), _occluder_mask(0.351)
    assert abs(float(hi.mean()) - float(lo.mean())) < 0.05
    assert 0.35 < float(lo.mean()) < 0.65          # sits on the ramp, not at either rail
    assert _occluder_mask(0.10).max() == 0.0       # clearly face
    assert _occluder_mask(0.80).min() == 1.0       # clearly obstruction


# --------------------------------------------------------------------------- #
# Watchdog: an obstruction that leaves is not a mask collapse
# --------------------------------------------------------------------------- #
@pytest.fixture
def wd():
    watchdog.reset()
    alerts = []
    watchdog.WATCHDOG.add_log_listener(lambda msg: alerts.append(msg))
    yield alerts
    watchdog.reset()


def test_obstruction_leaving_does_not_raise_collapse_or_seam_events(wd):
    present = np.zeros((128, 128), np.float32)
    present[20:100, 20:100] = 1.0                  # a hand in front of the face
    gone = np.zeros((128, 128), np.float32)        # the hand has moved away
    watchdog.check_mask_geometry(10, present, stream_key=('t', 0, 'occ'), mask_role='obstruction')
    ok = watchdog.check_mask_geometry(11, gone, stream_key=('t', 0, 'occ'), mask_role='obstruction')
    assert ok is True
    assert wd == [] and watchdog.WATCHDOG.get_mask_seam_events() == []


def test_a_face_mask_collapse_is_still_reported(wd):
    face = np.ones((128, 128), np.float32)
    watchdog.check_mask_geometry(10, face, stream_key=('t', 0, 'face'), mask_role='face')
    ok = watchdog.check_mask_geometry(11, np.zeros_like(face), stream_key=('t', 0, 'face'), mask_role='face')
    assert ok is False
    assert any('Mask collapse detected' in a for a in wd)
    assert [e['kind'] for e in watchdog.WATCHDOG.get_mask_seam_events()] == ['collapse']


def test_default_role_is_face_so_existing_callers_are_unchanged(wd):
    face = np.ones((64, 64), np.float32)
    watchdog.check_mask_geometry(1, face, stream_key='k')
    assert watchdog.check_mask_geometry(2, np.zeros_like(face), stream_key='k') is False


def test_process_face_labels_occluder_passes_as_obstruction():
    """The call site, not just the watchdog API: ProcessMgr passes the occluder family as the obstruction role."""
    from roop.occlusion_mask import OCCLUDER_FAMILY
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'roop', 'ProcessMgr.py'), encoding='utf-8').read()
    assert "mask_role=('obstruction' if getattr(p, 'processorname', None) in _OCCLUDER_FAMILY" in src
    assert set(OCCLUDER_FAMILY) == {'mask_occluder', 'mask_xseg3'}


# --------------------------------------------------------------------------- #
# Occlusion-aware colour redo (the ColorTransferMixin method ProcessMgr calls)
# --------------------------------------------------------------------------- #
_NOISE = 9.0        # real skin crops measured L std 11-41 (8-bit Lab); the guards' floor is 3


def _colour_scene(with_hand):
    size = 256
    target = _skin_crop(size, noise=_NOISE)
    hand = np.zeros((size, size), bool)
    keep = np.ones((size, size), np.float32)
    if with_hand:
        hand[90:190, 110:215] = True
        target[hand] = (40, 95, 235)
        keep[hand] = 0.0
    swap = _skin_crop(size, seed=11, noise=_NOISE)
    swap = np.clip(swap.astype(np.float32) * (0.82, 0.92, 1.0) + (14, 0, -18), 0, 255).astype(np.uint8)
    skin = _disk(size).astype(np.float32)
    return size, target, swap, skin, keep, hand


def test_colour_redo_fixes_a_contaminated_first_pass_and_leaves_a_clean_one_alone():
    from roop.procmgr_color import ColorTransferMixin
    mixin = ColorTransferMixin()

    size, target, swap, skin, keep, hand = _colour_scene(with_hand=True)
    pass1 = mixin.apply_color_transfer(swap, target, mask=skin)                 # what the swap stage did
    redo = mixin._occlusion_aware_colour(swap, target, skin, keep, skin)
    assert redo is not None, 'a hand inside the sampling mask must trigger the redo'
    fixed, new_mask = redo
    on_face = (skin > 0.5) & ~hand
    truth = cv2.cvtColor(_skin_crop(size, noise=_NOISE), cv2.COLOR_BGR2LAB).astype(np.float32)[on_face].mean(0)

    def err(img):
        return float(np.abs(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[on_face].mean(0) - truth).max())

    assert err(pass1) > 3.0 and err(fixed) < 1.5, (err(pass1), err(fixed))
    assert not new_mask[hand].any()

    size, target, swap, skin, keep, hand = _colour_scene(with_hand=False)
    assert mixin._occlusion_aware_colour(swap, target, skin, keep, skin) is None, \
        'no obstruction: the first pass must stand untouched (bit-identical frames)'


def test_colour_redo_refuses_a_sample_too_small_or_too_flat_to_fit():
    """Found on real footage: a 272-px sample (10% of pass 1's) brightened a face by 69 L levels, and
    near-flat patches (L std < 1) crushed contrast through Reinhard's target_std / source_std."""
    from roop.procmgr_color import ColorTransferMixin
    mixin = ColorTransferMixin()
    size, target, swap, skin, keep, hand = _colour_scene(with_hand=True)

    tiny = np.zeros((size, size), np.float32)
    tiny[120:140, 120:140] = 1.0                       # only a 20x20 sliver of face is "kept"
    assert mixin._occlusion_aware_colour(swap, target, skin, tiny, skin) is None

    flat_target = target.copy()
    flat_target[~hand & (skin > 0.5)] = (118, 150, 196)   # the kept skin is one constant colour: std 0
    assert mixin._occlusion_aware_colour(swap, flat_target, skin, keep, skin) is None


def test_colour_redo_refuses_a_correction_larger_than_occlusion_can_explain():
    from roop import occlusion_color as oc
    base = _skin_crop(128)
    mask = np.ones((128, 128), bool)
    modest = np.clip(base.astype(np.float32) + 6, 0, 255).astype(np.uint8)
    wild = np.clip(base.astype(np.float32) + 70, 0, 255).astype(np.uint8)
    assert oc.correction_is_plausible(base, modest, mask)
    assert not oc.correction_is_plausible(base, wild, mask)


# --------------------------------------------------------------------------- #
# RealityUX accessory gate: hair outside the face core is subtracted, inside is not
# --------------------------------------------------------------------------- #
class _StubXSeg:
    def Run(self, img, kw):
        return np.zeros((256, 256), np.float32)       # XSeg reads "skin / swap" everywhere


class _StubParser:
    def __init__(self, labels):
        self.labels = labels

    def RunLabels(self, img):
        return self.labels


def _ellipse_landmarks(cx, cy, rx, ry, n=106):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1).astype(np.float32)


def _rux_engine(labels):
    eng = _rux.Mask_RealityUX()
    eng._parser_enabled = True
    eng._executor = None
    eng._xseg = _StubXSeg()
    eng._parser = _StubParser(labels)
    return eng


def _rux_scene():
    labels = np.ones((512, 512), np.int32)           # skin everywhere
    labels[0:70, 150:360] = 17                       # hair ABOVE the face outline (forehead fringe)
    labels[200:330, 250:262] = 17                    # a strand of "hair" INSIDE the face core
    face = SimpleNamespace(landmark_2d_106=_ellipse_landmarks(256, 290, 150, 180), kps=None)
    M = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
    crop = np.full((512, 512, 3), 128, np.uint8)
    return labels, face, M, crop


def test_core_protect_map_is_one_inside_zero_outside_and_none_without_landmarks():
    labels, face, M, crop = _rux_scene()
    core = _rux._core_protect_map(face, M, crop.shape, (512, 512))
    assert core is not None and core[290, 256] == pytest.approx(1.0) and core[10, 256] == 0.0
    none_lm = SimpleNamespace(landmark_2d_106=None, landmark_3d_68=None)
    assert _rux._core_protect_map(none_lm, M, crop.shape, (512, 512)) is None
    five = SimpleNamespace(landmark_2d_106=np.zeros((5, 2), np.float32))
    assert _rux._core_protect_map(five, M, crop.shape, (512, 512)) is None
    assert _rux._core_protect_map(face, None, crop.shape, (512, 512)) is None


def test_realityux_subtracts_hair_outside_the_core_and_never_inside(monkeypatch):
    labels, face, M, crop = _rux_scene()
    eng = _rux_engine(labels)

    monkeypatch.setattr(_g, 'realityux_outer_accessory', False, raising=False)
    off = np.asarray(eng.Run(crop, '', target_face=face, M=M), np.float32)
    monkeypatch.setattr(_g, 'realityux_outer_accessory', True, raising=False)
    on = np.asarray(eng.Run(crop, '', target_face=face, M=M), np.float32)

    # premise (the priority inversion): XSeg reads skin there, so the old gate let the fringe through
    assert float(off[10:60, 200:310].max()) < 0.05
    # fix: BiSeNet's call stands outside the face outline
    assert float(on[10:60, 200:310].mean()) > 0.9
    # inside the core the gate is exactly what it was
    assert np.array_equal(on[200:330, 250:262], off[200:330, 250:262])
    core = _rux._core_protect_map(face, M, crop.shape, (512, 512))
    assert not (np.abs(on - off) > 1e-6)[core > 0.95].any()
    # and nothing BiSeNet did not label as accessory moved
    far = np.ones((512, 512), bool)
    far[0:90, 130:380] = False
    assert np.array_equal(on[far], off[far])


# --------------------------------------------------------------------------- #
# paste_upscale: restorer tone step at the rim, degraded (no Lab match) path
# --------------------------------------------------------------------------- #
class _PasteMgr(MaskingMixin):
    def __init__(self):
        self.options = SimpleNamespace(blend_ratio=1., show_face_area_overlay=False)
        self._tls = threading.local()


def _paste_rim(monkeypatch, yaw, regrade, parse, rim_lock=True, lab=1.0):
    monkeypatch.setenv('ROOP_GRAIN_MATCH', '0')
    for k, v in dict(semantic_paste_mask=True, blend_mode='laplacian', skin_lab_match=lab,
                     debug_mask_channels=False, boundary_illumination_strength=0.0,
                     restorer_rim_lock=rim_lock).items():
        monkeypatch.setattr(_g, k, v, raising=False)
    lab_img, plate, swapped, M, lm, kps = _tb._paste_scene(yaw)
    swapped = np.full_like(swapped, (130, 160, 200))                     # tone-matched to the plate
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.normal(0, 6, swapped.shape).astype(np.float32), (0, 0), 1.0)
    swapped = np.clip(swapped + tex, 0, 255).astype(np.uint8)
    restored = np.clip(swapped.astype(np.float32) + regrade, 0, 255).astype(np.uint8)
    sp = _tb.parse_of(lab_img) if parse == 'semantic' else None
    out = _PasteMgr().paste_upscale(swapped, restored, M, plate.copy(), 1, [0, 0, 0, 0, 12., 10.],
                                    face_landmarks=lm, face_kps=kps, semantic_parse=sp)
    return plate, out


def _rim_step(plate, out):
    def gray(im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.GaussianBlur(gray(out) - gray(plate), (0, 0), 1.0)
    foot = np.abs(d) > 0.5
    g2 = np.maximum(np.abs(np.diff(d, axis=1))[:-1, :], np.abs(np.diff(d, axis=0))[:, :-1])
    return float(g2[foot[:g2.shape[0], :g2.shape[1]]].max())


@pytest.mark.parametrize('yaw', [0.0, 40.0])
def test_restorer_regrade_does_not_leave_a_tone_step_at_the_rim_without_the_lab_match(monkeypatch, yaw):
    floor = _rim_step(*_paste_rim(monkeypatch, yaw, 0.0, 'none'))
    stepped = _rim_step(*_paste_rim(monkeypatch, yaw, 18.0, 'none', rim_lock=False))
    locked = _rim_step(*_paste_rim(monkeypatch, yaw, 18.0, 'none', rim_lock=True))
    assert stepped > floor + 1.0, \
        'premise: an 18-level restorer regrade must leave a visible rim step (%.2f vs %.2f)' % (stepped, floor)
    assert locked < floor + 0.6, \
        'tone lock should bring the step back to the no-regrade floor (%.2f vs %.2f)' % (locked, floor)


def test_the_default_semantic_path_is_byte_identical_with_the_lock_on_or_off(monkeypatch):
    """The Lab match runs there, so the lock must stand down entirely."""
    _, off = _paste_rim(monkeypatch, 40.0, 18.0, 'semantic', rim_lock=False)
    _, on = _paste_rim(monkeypatch, 40.0, 18.0, 'semantic', rim_lock=True)
    assert np.array_equal(on, off)


def test_the_lock_runs_when_the_user_turns_the_lab_match_off(monkeypatch):
    """skin_lab_match=0 is the 'keep the source's own pigment' choice; the rim must still not step."""
    _, off = _paste_rim(monkeypatch, 40.0, 18.0, 'semantic', rim_lock=False, lab=0.0)
    _, on = _paste_rim(monkeypatch, 40.0, 18.0, 'semantic', rim_lock=True, lab=0.0)
    assert not np.array_equal(on, off)
