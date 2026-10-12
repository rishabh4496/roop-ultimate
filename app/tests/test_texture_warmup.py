"""Block warm-up for the ordered stabilisation path: texture/mask filters take a looser seed bound than geometric ones.

`_stab_warmup_frames` used to ask every active filter for the frames needed to forget its seed to <= 1% and took the
slowest. For the geometric filters (kps, landmarks, the temporal engines) that bound is right: a wrong seed there is a
POSITION OFFSET that survives into the paste. For the texture/mask filters (enhancer, mask, HF carry) a wrong seed is
just a block-start frame that has not been smoothed yet, i.e. the raw frame, the level the un-stabilised path shows on
every frame. Measured on a 1500-frame clip with enhancer + mask smoothing on: warm-up 7f (the 1% rule) 15.5 fps /
excess shimmer 1.56; 3f 17.7 / 1.56; 0f 19.8 / 1.60; smoothing off 22.5 / 2.07. The replay was most of the cost and
bought nothing visible, and the frames at block starts were no noisier than the rest.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.one_euro import EnhancerStabilizer, KpsStabilizer, MaskStabilizer      # noqa: E402
from roop.procmgr_stabilization import StabilizationSchedulingMixin             # noqa: E402


def _warmup(kps=None, enh=None, mask=None, hf=None, env=None):
    stub = types.SimpleNamespace(kps_stabilizer=kps, enh_stabilizer=enh, mask_stabilizer=mask, _hf_stabilizer=hf)
    stub_env = {k: v for k, v in os.environ.items() if k not in ('ROOP_STAB_WARMUP', 'ROOP_STAB_TEXTURE_EPS')}
    stub_env.update(env or {})
    with patch.dict(os.environ, stub_env, clear=True):
        return StabilizationSchedulingMixin._stab_warmup_frames(stub)


class _FakeEngine:
    enabled = True

    def __init__(self, frames):
        self.frames = frames
        self.asked = []

    def warmup_frames(self, eps=0.01):
        self.asked.append(eps)
        return self.frames


class TextureWarmup(unittest.TestCase):
    def test_enhancer_only_no_longer_pays_the_one_percent_rule(self):
        enh = EnhancerStabilizer(strength=0.6)
        strict = _warmup(enh=enh, env={'ROOP_STAB_TEXTURE_EPS': '0.01'})
        loose = _warmup(enh=enh)
        self.assertEqual(strict, enh.warmup_frames(0.01))
        self.assertGreaterEqual(strict, 6)                  # the 7-frame replay the measured arms paid for
        self.assertLessEqual(loose, 1)
        self.assertLess(loose, strict)

    def test_mask_and_enhancer_together(self):
        enh, mask = EnhancerStabilizer(strength=0.6), MaskStabilizer(strength=0.5)
        self.assertLessEqual(_warmup(enh=enh, mask=mask), 1)
        self.assertGreater(_warmup(enh=enh, mask=mask, env={'ROOP_STAB_TEXTURE_EPS': '0.01'}), 1)

    def test_a_geometric_filter_keeps_the_strict_bound(self):
        kps = KpsStabilizer(min_cutoff=0.05, beta=0.02)
        self.assertEqual(_warmup(kps=kps), kps.warmup_frames(0.01))
        self.assertGreater(_warmup(kps=kps), 4)

    def test_geometric_still_sets_the_floor_next_to_a_texture_filter(self):
        kps, enh = KpsStabilizer(min_cutoff=0.05, beta=0.02), EnhancerStabilizer(strength=0.6)
        self.assertEqual(_warmup(kps=kps, enh=enh), kps.warmup_frames(0.01))

    def test_each_filter_is_asked_with_its_own_bound(self):
        geo, tex = _FakeEngine(9), _FakeEngine(3)
        stub = types.SimpleNamespace(kps_stabilizer=geo, enh_stabilizer=tex, mask_stabilizer=None, _hf_stabilizer=None)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ROOP_STAB_TEXTURE_EPS', None)
            os.environ.pop('ROOP_STAB_WARMUP', None)
            StabilizationSchedulingMixin._stab_warmup_frames(stub)
        self.assertEqual(geo.asked, [0.01])
        self.assertEqual(tex.asked, [0.5])

    def test_the_hf_carry_takes_the_texture_bound(self):
        hf = _FakeEngine(4)
        _warmup(hf=hf)
        self.assertEqual(hf.asked, [0.5])

    def test_the_texture_bound_is_never_tighter_than_the_requested_one(self):
        # `eps` is the caller's floor: a tighter-than-default ask must still bind the texture filters.
        enh = _FakeEngine(5)
        stub = types.SimpleNamespace(kps_stabilizer=None, enh_stabilizer=enh, mask_stabilizer=None, _hf_stabilizer=None)
        with patch.dict(os.environ, {'ROOP_STAB_TEXTURE_EPS': '0.001'}):
            os.environ.pop('ROOP_STAB_WARMUP', None)
            StabilizationSchedulingMixin._stab_warmup_frames(stub, eps=0.02)
        self.assertEqual(enh.asked, [0.02])

    def test_the_explicit_override_still_wins(self):
        self.assertEqual(_warmup(enh=EnhancerStabilizer(strength=0.6), env={'ROOP_STAB_WARMUP': '5'}), 5)
        self.assertEqual(_warmup(enh=EnhancerStabilizer(strength=0.6), env={'ROOP_STAB_WARMUP': '0'}), 0)

    def test_no_active_filter_needs_no_warmup(self):
        self.assertEqual(_warmup(), 0)


if __name__ == '__main__':
    unittest.main()
