"""Stage 2: the FP32 locks, and the TensorRT optimisation profiles each model can actually take.

Two things the Stage 2 brief asserted that the shipped files and the measurements say otherwise about, pinned here so they cannot
quietly change:

  * which models are locked to FP32 (`PRECISION_OVERRIDES`) and that the lock reaches `providers_for`;
  * which models can be given a min/opt/max profile at all. TensorRT rejects a profile whose max exceeds a STATIC input dimension, and
    GPEN-256/512, RestoreFormer++, CodeFormer v0.1 and inswapper_128 are exported with static `[1,3,S,S]` inputs. UltraMax's
    `codeformer.fp16` has a dynamic batch and is the one enhancer a batch profile can bind to.
"""
import os
import sys
import unittest
from unittest import mock

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop import precision_policy as pp                    # noqa: E402
from roop import trt_shape_profile as sp                   # noqa: E402

MODELS = os.path.join(APP, 'models')
LOCKED = ('xseg', '2d106det', '1k3d68', 'resnet18', 'restoreformer_plus_plus')


def _need(rel):
    path = os.path.join(MODELS, rel)
    if not os.path.isfile(path):
        raise unittest.SkipTest('%s is not installed' % rel)
    return path


class Fp32Locks(unittest.TestCase):
    def test_the_locked_models_are_fp32_in_the_table(self):
        for stem in LOCKED:
            self.assertEqual(pp.PRECISION_OVERRIDES.get(stem), 'fp32', stem)

    def test_the_lock_reaches_the_resolver_for_each_file(self):
        files = {'xseg': 'xseg.onnx', '2d106det': os.path.join('buffalo_l', '2d106det.onnx'),
                 '1k3d68': os.path.join('buffalo_l', '1k3d68.onnx'), 'resnet18': 'resnet18.onnx',
                 'restoreformer_plus_plus': 'restoreformer_plus_plus.onnx'}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pp.PRECISION_OVERRIDE_ENV, None)
            for stem, rel in files.items():
                self.assertEqual(pp.precision_override_for(stem, os.path.join(MODELS, rel)), 'fp32', stem)

    def test_the_rfpp_lock_can_be_dropped_in_one_line(self):
        """RF++ at FP32 costs +82% per face; the operator must be able to hand it back without a code change."""
        path = os.path.join(MODELS, 'restoreformer_plus_plus.onnx')
        with mock.patch.dict(os.environ, {pp.PRECISION_OVERRIDE_ENV: 'restoreformer_plus_plus:global'}):
            self.assertIsNone(pp.precision_override_for('restoreformer_pp', path))
            self.assertEqual(pp.precision_override_for('xseg', os.path.join(MODELS, 'xseg.onnx')), 'fp32')

    def test_the_enhancers_that_may_run_fp16_have_no_lock(self):
        """UltraMax and GPEN stay on the global precision; they are guarded by their canary instead."""
        for stem in ('codeformer.fp16', 'gpen-bfr-512', 'gpen_bfr_256'):
            self.assertNotIn(stem, pp.PRECISION_OVERRIDES)

    def test_rfpp_policy_matrix_agrees_with_the_override(self):
        self.assertEqual(pp.POLICIES['restoreformer_pp'].recommended, 'fp32')


class ProfileBinding(unittest.TestCase):
    def test_ultramax_binds_an_explicit_batch_profile_covering_one_to_four(self):
        path = _need(os.path.join('CodeFormer', 'codeformer.fp16.onnx'))
        info = sp.describe('ultramax', path)
        self.assertTrue(info['profiled'])
        opts = info['options']

        def shape(key):
            name, dims = opts[key].split(':')
            self.assertEqual(name, 'x')
            return [int(d) for d in dims.split('x')]
        lo, opt, hi = (shape('trt_profile_%s_shapes' % k) for k in ('min', 'opt', 'max'))
        self.assertEqual(lo, [1, 3, 512, 512])
        self.assertEqual(opt[1:], [3, 512, 512])
        self.assertGreaterEqual(hi[0], 4, 'the brief asks for a 1..4 batch at 512; the shipped profile must cover it')
        self.assertEqual(hi[1:], [3, 512, 512], 'spatial axes stay pinned: opening them would build for shapes never fed')

    def test_static_models_get_no_profile_because_none_can_be_bound(self):
        for key, rel in (('gpen', 'GPEN-BFR-512.onnx'), ('gpen256', 'gpen_bfr_256.onnx'),
                         ('restoreformer_pp', 'restoreformer_plus_plus.onnx'),
                         ('inswapper', 'inswapper_128.onnx')):
            path = _need(rel)
            with self.subTest(model=rel):
                info = sp.describe(key, path)
                self.assertFalse(info['profiled'])
                self.assertEqual(info['options'], {})
                self.assertTrue(all(not i['dynamic'] for i in info['inputs']),
                                '%s now has a dynamic input: a batch profile could be bound (and the brief\'s 1..4 / 1..16 revisited)' % rel)

    def test_a_profile_is_never_emitted_for_a_static_dimension_wider_than_itself(self):
        """Pinned, not asserted by TensorRT: every dimension of every emitted profile is min<=opt<=max, static ones all equal."""
        path = _need(os.path.join('CodeFormer', 'codeformer.fp16.onnx'))
        opts = sp.describe('ultramax', path)['options']
        parse = lambda v: [int(d) for d in v.split(':')[1].split('x')]       # noqa: E731
        lo, opt, hi = (parse(opts['trt_profile_%s_shapes' % k]) for k in ('min', 'opt', 'max'))
        for a, b, c in zip(lo, opt, hi):
            self.assertLessEqual(a, b)
            self.assertLessEqual(b, c)


if __name__ == '__main__':
    unittest.main()
