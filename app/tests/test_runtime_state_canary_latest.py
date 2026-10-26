"""PRECISION.canary_verdicts reports the LATEST verdict per model, and `nan_inf_free` is established, not asserted.

The canary store keeps one verdict per engine hash ever verified (xseg 21, restoreformer++ 11, gpen-bfr-512 5 on the dev
box) and the report keyed them by model name with the last dict entry winning. A build that failed under a floor since
relaxed showed FAILED for a model whose current engine passes; the reverse - an old pass hiding a live failure - was
equally possible. `nan_inf_free` was a literal True. Found by test_pipeline_3stage_sync_and_fps.py.
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import runtime_state  # noqa: E402


def _info(verdicts):
    with mock.patch('roop.aux_canary._load_store', return_value={'verdicts': verdicts}):
        return runtime_state._surgeon_info()


def _v(stem, passed, when):
    return {'stem': stem, 'passed': passed, 'reason': 'ok' if passed else 'diverges', 'when': when, 'metrics': {}}


class LatestVerdictPerModel(unittest.TestCase):
    def test_an_old_failure_does_not_mask_a_newer_pass(self):
        info = _info({'b': _v('xseg', True, '2026-10-25 23:42:23'), 'a': _v('xseg', False, '2026-10-10 20:03:33')})   # old one LAST
        self.assertIs(info['canary_verdicts']['xseg']['passed'], True)
        self.assertEqual(info['canary_verdicts']['xseg']['recorded'], 2)

    def test_an_old_pass_does_not_hide_a_newer_failure(self):
        info = _info({'a': _v('xseg', False, '2026-10-25 23:42:23'), 'z': _v('xseg', True, '2026-10-10 20:03:33')})   # old one LAST
        self.assertIs(info['canary_verdicts']['xseg']['passed'], False)             # dict order must not decide it

    def test_models_are_independent(self):
        info = _info({'a': _v('xseg', True, '2026-10-01 00:00:00'), 'b': _v('1k3d68', False, '2026-10-02 00:00:00')})
        self.assertEqual({k: v['passed'] for k, v in info['canary_verdicts'].items()}, {'xseg': True, '1k3d68': False})


class NanInfFreeIsEstablishedNotAsserted(unittest.TestCase):
    def gate(self, verdicts):
        return _info(verdicts)['verification_gates']['nan_inf_free']

    def test_no_verdicts_is_unknown_not_true(self):
        self.assertIsNone(self.gate({}))

    def test_every_latest_verdict_passing_establishes_it(self):
        self.assertIs(self.gate({'a': _v('xseg', True, '1'), 'b': _v('1k3d68', True, '1')}), True)

    def test_a_failing_latest_verdict_does_not_establish_it(self):
        self.assertIsNone(self.gate({'a': _v('xseg', True, '1'), 'b': _v('1k3d68', False, '1')}))

    def test_a_stale_failure_does_not_block_it(self):
        self.assertIs(self.gate({'b': _v('xseg', True, '2'), 'a': _v('xseg', False, '1')}), True)


if __name__ == '__main__':
    unittest.main()
