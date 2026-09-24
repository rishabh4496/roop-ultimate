"""The per-render VRAM governor (roop/vram_governor.py).

What must hold:
  * the planner steps the cheap thing down first (the swap batch) and the
    look-changing thing (GPEN resolution) only once the batch is exhausted;
  * it does NOTHING on the two machines this project ships to, at their live
    configurations and ordinary free VRAM -- a safety net that fires on a
    healthy 4070 or 3060 is a silent perf regression;
  * its outputs actually reach the two consumers (the batcher and the GPEN
    size), and stop reaching them once the render ends.
"""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import roop.globals                                    # noqa: E402
from roop import vram_governor as vg                   # noqa: E402


def job(**kw):
    base = dict(width=1920, height=1080, faces=2, enhancer='GPEN 256 Pro',
                swap_model='realswap', mask_engines=('mask_xseg',), threads=12,
                batch_cap=8, swap_contexts=2, detmask_contexts=2,
                enhancer_contexts=2, nvdec=True, detector_size=512)
    base.update(kw)
    return vg.JobSpec(**base)


class Planner(unittest.TestCase):

    def test_4070_live_config_is_left_alone(self):
        # 12GB card with the desktop idling at ~1.1GB used (measured 09-24).
        plan = vg.plan_job(job(), free_mb=10900, total_mb=12282, margin_gb=1.5)
        self.assertEqual(plan.actions, [], plan.as_dict())
        self.assertEqual(plan.batch_cap, 8)

    def test_prior_reproduces_the_3060_measurement(self):
        # 2346MB measured for the whole unpooled process on the RTX 3060.
        budget = vg.plan_job(job(threads=8, batch_cap=1, swap_contexts=1,
                                 detmask_contexts=1, enhancer_contexts=1, nvdec=False),
                             free_mb=1e6, total_mb=1e6).budget_mb
        self.assertLess(abs(budget - 2346) / 2346, 0.15, budget)

    def test_3060_live_config_is_left_alone(self):
        # 6GB laptop: single contexts, batch 4, GPEN 256 Pro, ~5.2GB free.
        plan = vg.plan_job(job(threads=8, batch_cap=4, swap_contexts=1,
                               detmask_contexts=1, enhancer_contexts=1),
                           free_mb=5300, total_mb=6144, margin_gb=1.5)
        self.assertEqual(plan.actions, [], plan.as_dict())

    def test_batch_steps_down_before_the_enhancer(self):
        j = job(enhancer='GPEN 2048', width=3840, height=2160)
        roomy = vg.plan_job(j, free_mb=1e6, total_mb=1e6).budget_mb
        batch1 = vg.plan_job(job(enhancer='GPEN 2048', width=3840, height=2160,
                                 batch_cap=1), free_mb=1e6, total_mb=1e6).budget_mb
        # Enough room for batch 1 at 2048, not for batch 8.
        free = (roomy + batch1) / 2 + 1.5 * 1024
        plan = vg.plan_job(j, free_mb=free, total_mb=12282, margin_gb=1.5)
        self.assertLess(plan.batch_cap, 8)
        self.assertEqual(plan.gpen_size, 2048, 'GPEN must not drop while a batch step remains')
        self.assertTrue(all('batch' in a for a in plan.actions))

    def test_gpen_steps_2048_1024_512_when_the_batch_is_exhausted(self):
        plan = vg.plan_job(job(enhancer='GPEN 2048'), free_mb=2000, total_mb=12282)
        self.assertEqual(plan.batch_cap, 1)
        self.assertEqual(plan.gpen_size, 512)
        self.assertEqual([a for a in plan.actions if 'GPEN' in a],
                         ['GPEN 2048 -> 1024', 'GPEN 1024 -> 512'])
        self.assertFalse(plan.fits)

    def test_an_unsized_enhancer_is_never_resized(self):
        plan = vg.plan_job(job(enhancer='GPEN 256 Pro'), free_mb=1000, total_mb=6144)
        self.assertIsNone(plan.gpen_size)
        self.assertFalse(any('GPEN' in a for a in plan.actions))

    def test_margin_is_clamped_to_the_slider_range(self):
        self.assertEqual(vg.clamp_margin_gb(0.0), 0.5)
        self.assertEqual(vg.clamp_margin_gb(9), 4.0)
        self.assertEqual(vg.clamp_margin_gb('junk'), vg.DEFAULT_MARGIN_GB)
        self.assertEqual(vg.clamp_margin_gb(float('nan')), vg.DEFAULT_MARGIN_GB)

    def test_a_bigger_margin_steps_down_sooner(self):
        free = vg.plan_job(job(), free_mb=1e6, total_mb=1e6).budget_mb + 2.0 * 1024
        self.assertEqual(vg.plan_job(job(), free, 12282, margin_gb=1.5).actions, [])
        self.assertTrue(vg.plan_job(job(), free, 12282, margin_gb=4.0).actions)

    def test_resolution_raises_the_budget(self):
        small = vg.plan_job(job(width=1280, height=720), 1e6, 1e6).budget_mb
        large = vg.plan_job(job(width=3840, height=2160), 1e6, 1e6).budget_mb
        self.assertGreater(large, small)

    def test_the_learned_ratio_scales_the_budget(self):
        one = vg.plan_job(job(), 1e6, 1e6, ratio=1.0).budget_mb
        two = vg.plan_job(job(), 1e6, 1e6, ratio=2.0).budget_mb
        self.assertAlmostEqual(two, one * 2, delta=1)
        # A polluted sample cannot steer the next render past the band.
        self.assertAlmostEqual(vg.plan_job(job(), 1e6, 1e6, ratio=50).budget_mb,
                               one * 3.0, delta=1)

    def test_signature_ignores_what_the_planner_varies(self):
        self.assertEqual(job(batch_cap=8).signature(), job(batch_cap=1).signature())
        self.assertEqual(job(enhancer='GPEN 1024').signature(),
                         job(enhancer='GPEN 2048').signature())
        self.assertNotEqual(job().signature(), job(width=3840, height=2160).signature())


class Calibration(unittest.TestCase):

    def test_record_then_load_is_an_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cal.json')
            self.assertEqual(vg.load_ratio('sig', path), 1.0)
            vg.record_peak('sig', estimate_mb=4000, peak_mb=6000, path=path)
            self.assertAlmostEqual(vg.load_ratio('sig', path), 1.5)
            vg.record_peak('sig', estimate_mb=4000, peak_mb=4000, path=path)
            self.assertAlmostEqual(vg.load_ratio('sig', path), 1.25)
            row = json.load(open(path))['sig']
            self.assertEqual(row['samples'], 2)


class RenderSession(unittest.TestCase):
    """admit() / finish() and the two consumers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cal = mock.patch.object(vg, '_CALIBRATION_FILE',
                                     os.path.join(self.tmp.name, 'c.json'))
        self.cal.start()

    def tearDown(self):
        vg.finish()
        self.cal.stop()
        self.tmp.cleanup()

    def _admit(self, free_mb):
        with mock.patch.object(vg, 'query_vram_mb', return_value=(free_mb, 1000.0, 12282.0)):
            return vg.admit(job(enhancer='GPEN 2048'), 1.5)

    def test_no_plan_means_requested_values(self):
        self.assertIsNone(vg.current_plan())
        self.assertEqual(vg.governed_batch_cap(8), 8)
        self.assertEqual(vg.governed_gpen_size(2048), 2048)

    def test_a_tight_plan_reaches_both_consumers_then_clears(self):
        plan = self._admit(free_mb=2000)
        self.assertEqual(vg.governed_batch_cap(8), plan.batch_cap)
        self.assertEqual(vg.governed_gpen_size(2048), 512)
        vg.finish()
        self.assertEqual(vg.governed_batch_cap(8), 8)
        self.assertEqual(vg.governed_gpen_size(2048), 2048)

    def test_gpen_size_reaches_get_processing_plugins(self):
        from roop.core import get_processing_plugins
        saved = getattr(roop.globals, 'selected_enhancer', None)
        roop.globals.selected_enhancer = 'GPEN 2048'
        try:
            self.assertEqual(get_processing_plugins(None)['gpen'], {'size': 2048})
            self._admit(free_mb=2000)
            self.assertEqual(get_processing_plugins(None)['gpen'], {'size': 512})
        finally:
            roop.globals.selected_enhancer = saved

    def test_no_gpu_means_no_plan(self):
        with mock.patch.object(vg, 'query_vram_mb', return_value=None):
            self.assertIsNone(vg.admit(job(), 1.5))
        self.assertIsNone(vg.current_plan())


class BatcherHonoursTheGovernor(unittest.TestCase):

    def setUp(self):
        from roop import ProcessMgr as pm
        self.pm = pm
        self._env = dict(os.environ)
        self._batch = pm._BATCH_SWAP
        os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        os.environ.pop('ROOP_BATCH_SWAP_MAX', None)
        pm._BATCH_SWAP = True
        roop.globals.swap_model_mask_strength = 0.0

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        self.pm._BATCH_SWAP = self._batch
        vg._active = None

    def _make(self, threads=8):
        swapper = SimpleNamespace(type='swap', pool=None, model_has_mask=False,
                                  RunBatchMulti=lambda requests: [])
        stub = SimpleNamespace(processors=[swapper], _runtime_swap_batch_size=2,
                               runtime_profile=SimpleNamespace(
                                   hardware=SimpleNamespace(vram_total_gb=12.0)))
        return self.pm.ProcessMgr._make_swap_batcher(stub, threads)

    def _plan(self, cap):
        vg._active = (SimpleNamespace(batch_cap=cap, gpen_size=None), None, 0.0)

    def test_governor_cap_lowers_the_batch(self):
        self._plan(4)
        b = self._make()
        try:
            self.assertEqual(b._max_batch, 4)
            self.assertEqual(roop.globals.last_swap_batch_max, 4)
        finally:
            b.stop()

    def test_governor_cap_beats_an_explicit_setting(self):
        os.environ['ROOP_BATCH_SWAP_MAX'] = '8'
        self._plan(2)
        b = self._make()
        try:
            self.assertEqual(b._max_batch, 2)
        finally:
            b.stop()

    def test_governor_cap_of_one_turns_batching_off(self):
        self._plan(1)
        self.assertIsNone(self._make())
        self.assertEqual(roop.globals.last_swap_batch_max, 1)

    def test_explicit_batch_one_means_no_batcher(self):
        os.environ['ROOP_BATCH_SWAP_MAX'] = '1'
        self.assertIsNone(self._make(), 'an explicit 1 used to be floored to 2')


if __name__ == '__main__':
    unittest.main()
