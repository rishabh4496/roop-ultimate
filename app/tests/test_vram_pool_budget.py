"""Keeping peak VRAM under ~90% of the card without changing a model or a precision.

Three mechanisms, pinned here:

  1. After a replayed render's pre-pass the FaceAnalysis pool and every hybrid detector
     pool shrink to width 1 (`face_util.shrink_analysis_pools`). The swap phase only calls
     the detector for verification and rescue, so the pre-pass's width is idle memory.
     The kept instance is the first one, `_ensure_face_analyser` does not rebuild the pool
     at its configured width behind the shrink, and the ceiling ends with the pool / the
     next run.
  2. The render's VRAM plan (`vram_governor.plan_job`) lowers pool widths -- one context of
     one pool at a time, whichever frees most -- AFTER the swap batch and BEFORE the
     look-changing GPEN size, holds the projected peak to 90% of the card whatever the
     margin slider says, and publishes the widths to `TensorRTResourceManager`, which
     enforces them on every pool (explicit settings too) until the render ends.
  3. On Windows a one-time warning names the driver setting that turns a VRAM over-commit
     (paging, which reads as a hang) into a failure, when use crosses 95%.
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from queue import Queue
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals                                                  # noqa: E402
from roop import face_util, session_pool                             # noqa: E402
from roop import vram_governor as vg                                 # noqa: E402


def _lease_pool(n):
    items = ['inst%d' % i for i in range(n)]
    q = Queue()
    for it in items:
        q.put(it)
    return {'items': items, 'q': q}


class ShrinkLeasePool(unittest.TestCase):

    def test_keeps_the_first_instances_and_returns_the_rest(self):
        pool = _lease_pool(4)
        dropped = session_pool.shrink_lease_pool(pool, 1)
        self.assertEqual(dropped, ['inst1', 'inst2', 'inst3'])
        self.assertEqual(pool['items'], ['inst0'])
        self.assertEqual(pool['q'].qsize(), 1)
        self.assertEqual(pool['q'].get_nowait(), 'inst0')

    def test_a_pool_already_that_narrow_is_untouched(self):
        pool = _lease_pool(1)
        self.assertEqual(session_pool.shrink_lease_pool(pool, 1), [])
        self.assertEqual(session_pool.shrink_lease_pool(pool, 3), [])
        self.assertEqual(pool['items'], ['inst0'])

    def test_an_instance_in_use_is_never_torn_down(self):
        pool = _lease_pool(2)
        held = pool['q'].get()                       # a worker is mid-call
        self.assertEqual(session_pool.shrink_lease_pool(pool, 1, timeout=0.2), [])
        self.assertEqual(pool['items'], ['inst0', 'inst1'], 'a refused shrink changed the pool')
        pool['q'].put(held)
        self.assertEqual(pool['q'].qsize(), 2)
        self.assertEqual(session_pool.shrink_lease_pool(pool, 1, timeout=0.2), ['inst1'])

    def test_waits_for_a_lease_that_returns(self):
        pool = _lease_pool(2)
        held = pool['q'].get()
        threading.Timer(0.1, lambda: pool['q'].put(held)).start()
        self.assertEqual(session_pool.shrink_lease_pool(pool, 1, timeout=5.0), ['inst1'])


class _FakeAnalyser:
    pass


class ShrinkAnalysisPools(unittest.TestCase):

    def setUp(self):
        self._saved = {k: getattr(face_util, k) for k in (
            'FACE_ANALYSER_POOL', 'FACE_ANALYSER', '_ANALYSER_Q', '_ANALYSER_DET_SIZE',
            '_ANALYSER_DET_THRESH', '_ANALYSER_ENGINE', '_ANALYSER_LM68_LAZY',
            '_ANALYSER_POOL_CEILING')}
        self._globals = (roop.globals.g_current_face_analysis, roop.globals.g_desired_face_analysis,
                         getattr(roop.globals, 'is_preview', False))
        modules = ['landmark_2d_106', 'detection', 'recognition']
        roop.globals.g_desired_face_analysis = modules
        roop.globals.g_current_face_analysis = modules
        roop.globals.is_preview = False
        self.pool = [_FakeAnalyser(), _FakeAnalyser()]
        face_util.FACE_ANALYSER_POOL = list(self.pool)
        face_util.FACE_ANALYSER = self.pool[0]
        q = Queue()
        for fa in self.pool:
            q.put(fa)
        face_util._ANALYSER_Q = q
        face_util._ANALYSER_DET_SIZE = face_util._desired_det_size()
        face_util._ANALYSER_DET_THRESH = getattr(roop.globals, 'face_detector_threshold', 0.60)
        face_util._ANALYSER_ENGINE = face_util._current_engine()
        face_util._ANALYSER_LM68_LAZY = bool(getattr(roop.globals, 'lm68_lazy', False))
        face_util._ANALYSER_POOL_CEILING = None
        self.built = []

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(face_util, k, v)
        (roop.globals.g_current_face_analysis, roop.globals.g_desired_face_analysis,
         roop.globals.is_preview) = self._globals

    def _build(self):
        fa = _FakeAnalyser()
        self.built.append(fa)
        return fa

    def _ensure(self, configured=2):
        with mock.patch('roop.session_pool.detmask_pool_size', return_value=configured), \
                mock.patch('roop.session_pool.detmask_pooling_enabled', return_value=configured >= 2), \
                mock.patch('roop.face_util._build_face_analyser', side_effect=self._build), \
                mock.patch('roop.face_util._cleanup_fa_pool'):
            return face_util._ensure_face_analyser()

    def test_shrinks_to_the_first_instance(self):
        with mock.patch('roop.face_util._cleanup_fa_pool') as cleanup:
            out = face_util.shrink_analysis_pools(1)
        self.assertEqual(out['analyser'], 1)
        self.assertEqual(face_util.FACE_ANALYSER_POOL, [self.pool[0]])
        self.assertIs(face_util.FACE_ANALYSER, self.pool[0])
        cleanup.assert_called_once_with([self.pool[1]])
        self.assertFalse(face_util.analysis_pooled())

    def test_the_pool_is_not_rebuilt_at_its_configured_width_behind_the_shrink(self):
        with mock.patch('roop.face_util._cleanup_fa_pool'):
            face_util.shrink_analysis_pools(1)
        res = self._ensure(configured=2)
        self.assertEqual(self.built, [], 'the shrunk pool was rebuilt at width 2')
        self.assertIs(res, self.pool[0])

    def test_without_the_ceiling_the_old_behaviour_would_rebuild(self):
        """The control: it is the ceiling, not a coincidence, that stops the rebuild."""
        face_util.FACE_ANALYSER_POOL = [self.pool[0]]
        face_util.FACE_ANALYSER = self.pool[0]
        self._ensure(configured=2)
        self.assertEqual(len(self.built), 2)

    def test_clearing_the_ceiling_restores_the_configured_width(self):
        with mock.patch('roop.face_util._cleanup_fa_pool'):
            face_util.shrink_analysis_pools(1)
        face_util.clear_analysis_pool_ceiling()
        self.assertIsNone(face_util._ANALYSER_POOL_CEILING)
        self._ensure(configured=2)
        self.assertEqual(len(self.built), 2)

    def test_releasing_the_pool_ends_the_ceiling(self):
        with mock.patch('roop.face_util._cleanup_fa_pool'):
            face_util.shrink_analysis_pools(1)
            face_util.release_face_analyser()
        self.assertIsNone(face_util._ANALYSER_POOL_CEILING)
        self.assertEqual(face_util.FACE_ANALYSER_POOL, [])

    def test_hybrid_detector_pools_shrink_too(self):
        from roop import yoloface
        saved = yoloface._pool
        yoloface._pool = _lease_pool(3)
        try:
            with mock.patch('roop.face_util._cleanup_fa_pool'):
                out = face_util.shrink_analysis_pools(1)
            self.assertEqual(out['detectors'], 2)
            self.assertEqual(yoloface._pool['items'], ['inst0'])
        finally:
            yoloface._pool = saved

    def test_a_pool_already_at_width_one_reports_nothing_dropped(self):
        face_util.FACE_ANALYSER_POOL = [self.pool[0]]
        out = face_util.shrink_analysis_pools(1)
        self.assertEqual(out, {'analyser': 0, 'detectors': 0})


class BudgetCaps(unittest.TestCase):

    def setUp(self):
        self.mgr = session_pool.TensorRTResourceManager()
        self.live = mock.patch('roop.session_pool._live_vram_mb', return_value=(11000.0, 12282.0))
        self.live.start()

    def tearDown(self):
        self.live.stop()

    def _select(self, key, requested=2, explicit=False):
        return self.mgr.select_pool_size(requested, key, (1, 3, 256, 256), 1, explicit=explicit)

    def test_no_caps_changes_nothing(self):
        self.assertEqual(self._select('swapper:hyperswap'), 2)

    def test_a_cap_lowers_that_family_only(self):
        self.mgr.set_budget_caps({'swap': 1})
        self.assertEqual(self._select('swapper:hyperswap'), 1)
        self.assertEqual(self._select('enhancer:gpen_256'), 2)
        self.assertEqual(self._select('mask:xseg'), 2)

    def test_detector_and_mask_pools_share_the_detmask_cap(self):
        self.mgr.set_budget_caps({'detmask': 1})
        self.assertEqual(self._select('mask:xseg'), 1)
        self.assertEqual(self._select('detector:retinaface'), 1)
        self.assertEqual(self._select('swapper:hyperswap'), 2)

    def test_every_enhancer_family_answers_to_the_enhancer_cap(self):
        self.mgr.set_budget_caps({'enhancer': 1})
        for key in ('enhancer:gpen_256', 'enhancer:restoreformer', 'enhancer:codeformer',
                    'enhancer:ultramax'):
            self.assertEqual(self._select(key), 1, key)

    def test_an_explicit_setting_is_capped_too(self):
        """The cap is physical (measured free memory), not a policy opinion."""
        self.mgr.set_budget_caps({'swap': 1})
        self.assertEqual(self._select('swapper:hyperswap', requested=4, explicit=True), 1)

    def test_clearing_restores_the_width_so_the_cap_was_never_memoised(self):
        self.mgr.set_budget_caps({'swap': 1})
        self.assertEqual(self._select('swapper:hyperswap'), 1)
        self.mgr.set_budget_caps(None)
        self.assertEqual(self.mgr.budget_caps(), {})
        self.assertEqual(self._select('swapper:hyperswap'), 2)

    def test_a_cap_never_widens(self):
        self.mgr.set_budget_caps({'swap': 6})
        self.assertEqual(self._select('swapper:hyperswap'), 2)

    def test_the_cap_is_announced_once(self):
        self.mgr.set_budget_caps({'swap': 1})
        with mock.patch('builtins.print') as p:
            for _ in range(3):
                self._select('swapper:hyperswap')
        self.assertEqual(sum('VRAM plan' in str(c) for c in p.call_args_list), 1)


def job(**kw):
    base = dict(width=1920, height=1080, faces=2, enhancer='GPEN 256 Pro',
                swap_model='realswap', mask_engines=('mask_xseg',), threads=12,
                batch_cap=8, swap_contexts=2, detmask_contexts=2,
                enhancer_contexts=2, nvdec=True, detector_size=512)
    base.update(kw)
    return vg.JobSpec(**base)


def budget(j, batch=None):
    return vg.plan_job(vg.JobSpec(**{**j.__dict__, 'batch_cap': batch or j.batch_cap}),
                       free_mb=1e9, total_mb=1e9).budget_mb


class PlannerPools(unittest.TestCase):

    def _free_for(self, target_budget_mb, total=12282, margin_gb=1.5):
        return target_budget_mb + max(margin_gb * 1024.0, 0.10 * total)

    def test_a_card_with_room_is_left_alone(self):
        plan = vg.plan_job(job(), free_mb=10900, total_mb=12282)
        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.pool_caps, {})

    def test_pools_step_down_after_the_batch_and_before_the_gpen_size(self):
        j = job(enhancer='GPEN 2048', width=3840, height=2160)
        plan = vg.plan_job(j, free_mb=self._free_for(budget(j, batch=1) - 200), total_mb=12282)
        kinds = ['batch' if 'batch' in a else 'pool' if 'pool' in a else 'gpen' for a in plan.actions]
        self.assertIn('pool', kinds, plan.actions)
        self.assertEqual(kinds, sorted(kinds, key={'batch': 0, 'pool': 1, 'gpen': 2}.get),
                         'steps out of order: %s' % plan.actions)
        self.assertEqual(plan.gpen_size, 2048, 'the look-changing step ran while a pool step could fit')
        self.assertTrue(plan.fits, plan.as_dict())

    def test_one_pool_at_a_time_the_one_that_frees_most_first(self):
        j = job(enhancer='GPEN 2048', width=3840, height=2160)
        needed = budget(j, batch=1)
        freed = {}
        for name, fld in (('enhancer', 'enhancer_contexts'), ('swap', 'swap_contexts'),
                          ('detmask', 'detmask_contexts')):
            freed[name] = needed - budget(vg.JobSpec(**{**j.__dict__, fld: 1}), batch=1)
        biggest = max(freed, key=freed.get)
        # Short by less than any single step frees: exactly one pool step is planned.
        plan = vg.plan_job(j, free_mb=self._free_for(needed - 0.5 * min(freed.values())),
                           total_mb=12282)
        pool_steps = [a for a in plan.actions if 'pool' in a]
        self.assertEqual(len(pool_steps), 1, plan.actions)
        self.assertEqual(pool_steps[0], '%s pool 2 -> 1' % biggest)
        self.assertEqual(plan.pool_caps, {biggest: 1})
        # And the other pools stay at the width they were asked for.
        self.assertTrue(set(plan.pool_caps) < {'enhancer', 'swap', 'detmask'})

    def test_not_all_pools_stay_at_the_full_width_when_the_budget_is_exceeded(self):
        j = job(enhancer='GPEN 2048', width=3840, height=2160)
        plan = vg.plan_job(j, free_mb=self._free_for(budget(j, batch=1) - 1500), total_mb=12282)
        self.assertGreaterEqual(len(plan.pool_caps), 2, plan.as_dict())

    def test_a_pool_whose_models_are_not_in_the_job_is_not_touched(self):
        j = job(enhancer='None', mask_engines=())
        plan = vg.plan_job(j, free_mb=self._free_for(budget(j, batch=1) - 300), total_mb=12282)
        self.assertNotIn('enhancer', plan.pool_caps, plan.as_dict())

    def test_no_pool_goes_below_one(self):
        plan = vg.plan_job(job(), free_mb=500, total_mb=12282)
        self.assertTrue(all(v >= 1 for v in plan.pool_caps.values()))
        self.assertFalse(plan.fits)

    def test_the_ceiling_is_ninety_percent_whatever_the_slider_says(self):
        total = 40000.0
        j = job()
        b = budget(j)
        # 0.5 GB margin alone would call this plan fine; 10% of a 40 GB card is 4 GB.
        free = b + 1024.0
        self.assertEqual(vg.plan_job(j, free, total, margin_gb=0.5).fits, False)
        loose = vg.plan_job(j, b + 0.10 * total + 10, total, margin_gb=0.5)
        self.assertTrue(loose.fits)
        self.assertGreaterEqual(loose.margin_mb, 0.10 * total - 1)

    def test_the_4070_and_3060_live_configs_are_still_left_alone(self):
        self.assertEqual(vg.plan_job(job(), 10900, 12282, margin_gb=1.5).actions, [])
        self.assertEqual(vg.plan_job(job(threads=8, batch_cap=4, swap_contexts=1,
                                         detmask_contexts=1, enhancer_contexts=1),
                                     5300, 6144, margin_gb=1.5).actions, [])


class RatioFallback(unittest.TestCase):

    def test_a_narrower_pool_borrows_the_widest_known_ratio_of_its_configuration(self):
        wide = job().signature()
        narrow = job(swap_contexts=1).signature()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cal.json')
            vg.record_peak(wide, 4000, 6400, path=path)                    # ratio 1.6
            vg.record_peak(job(detmask_contexts=1).signature(), 3500, 4200, path=path)   # 1.2
            self.assertAlmostEqual(vg.load_ratio(narrow, path), 1.6)
            self.assertAlmostEqual(vg.load_ratio(wide, path), 1.6)

    def test_an_exact_row_beats_a_neighbour(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cal.json')
            vg.record_peak(job().signature(), 4000, 6400, path=path)
            vg.record_peak(job(swap_contexts=1).signature(), 3000, 3300, path=path)
            self.assertAlmostEqual(vg.load_ratio(job(swap_contexts=1).signature(), path), 1.1)

    def test_a_different_configuration_is_not_a_neighbour(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cal.json')
            vg.record_peak(job(width=3840, height=2160).signature(), 4000, 6400, path=path)
            self.assertEqual(vg.load_ratio(job(swap_contexts=1).signature(), path), 1.0)

    def test_a_signature_without_the_context_fields_still_loads_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cal.json')
            vg.record_peak('sig', 4000, 6000, path=path)
            self.assertAlmostEqual(vg.load_ratio('sig', path), 1.5)
            self.assertEqual(vg.load_ratio('other', path), 1.0)


class AdmitPublishesCaps(unittest.TestCase):

    def tearDown(self):
        vg.finish()
        session_pool.resource_manager().set_budget_caps(None)

    def test_admit_publishes_the_plans_widths_and_finish_clears_them(self):
        j = job(enhancer='GPEN 2048', width=3840, height=2160)
        tight = budget(j, batch=1) - 400 + 1.5 * 1024
        with mock.patch.object(vg, 'query_vram_mb', return_value=(tight, 12282 - tight, 12282.0)), \
                mock.patch.object(vg, 'load_ratio', return_value=1.0):
            plan = vg.admit(j, 1.5)
        self.assertTrue(plan.pool_caps, plan.as_dict())
        self.assertEqual(session_pool.resource_manager().budget_caps(), plan.pool_caps)
        with mock.patch.object(vg, 'record_peak', return_value=None):
            vg.finish()
        self.assertEqual(session_pool.resource_manager().budget_caps(), {})

    def test_a_roomy_card_publishes_nothing(self):
        with mock.patch.object(vg, 'query_vram_mb', return_value=(10900.0, 1382.0, 12282.0)), \
                mock.patch.object(vg, 'load_ratio', return_value=1.0):
            plan = vg.admit(job(), 1.5)
        self.assertEqual(plan.pool_caps, {})
        self.assertEqual(session_pool.resource_manager().budget_caps(), {})


class CriticalWarning(unittest.TestCase):

    def setUp(self):
        vg._reset_critical_warning()

    def tearDown(self):
        vg._reset_critical_warning()

    def test_warns_once_on_windows_at_95_percent_and_names_the_setting(self):
        with mock.patch.object(vg, '_say') as say:
            self.assertTrue(vg.warn_if_vram_critical(11700, 12282, 'frame 500', platform='win32'))
            self.assertFalse(vg.warn_if_vram_critical(12000, 12282, 'frame 600', platform='win32'))
        text = ' '.join(str(c.args[0]) for c in say.call_args_list)
        for needle in ('95%', 'NVIDIA Control Panel', 'CUDA - Sysmem Fallback Policy',
                       'Prefer No Sysmem Fallback', sys.executable, 'frame 500'):
            self.assertIn(needle, text)
        self.assertEqual(len(say.call_args_list), 1)

    def test_silent_below_the_threshold(self):
        with mock.patch.object(vg, '_say') as say:
            self.assertFalse(vg.warn_if_vram_critical(11000, 12282, platform='win32'))
            self.assertFalse(vg.warn_if_vram_critical(0.95 * 12282 - 1, 12282, platform='win32'))
        say.assert_not_called()

    def test_threshold_is_inclusive_at_exactly_95_percent(self):
        with mock.patch.object(vg, '_say'):
            self.assertTrue(vg.warn_if_vram_critical(0.95 * 12282, 12282, platform='win32'))

    def test_silent_off_windows_where_the_setting_does_not_exist(self):
        with mock.patch.object(vg, '_say') as say:
            self.assertFalse(vg.warn_if_vram_critical(12200, 12282, platform='linux'))
        say.assert_not_called()

    def test_unknown_total_is_not_a_warning(self):
        self.assertFalse(vg.warn_if_vram_critical(100, 0, platform='win32'))

    def test_it_is_written_through_the_bar_safe_writer(self):
        with mock.patch('roop.procmgr_runtime.bar_write') as bar:
            vg.warn_if_vram_critical(11900, 12282, 'x', platform='win32')
        self.assertEqual(bar.call_count, 1)
        self.assertIn('Sysmem Fallback', bar.call_args.args[0])

    def test_the_sampler_raises_it_during_a_render(self):
        with mock.patch.object(vg, 'query_vram_mb', return_value=(300.0, 11982.0, 12282.0)), \
                mock.patch('sys.platform', 'win32'), mock.patch.object(vg, '_say') as say:
            sampler = vg._PeakSampler(0, 1000.0, period=0.01)
            sampler.start()
            time.sleep(0.15)
            sampler.stop()
        self.assertEqual(sum('Sysmem Fallback' in str(c) for c in say.call_args_list), 1)

    def test_admit_raises_it_when_the_card_is_already_that_full(self):
        with mock.patch.object(vg, 'query_vram_mb', return_value=(300.0, 11982.0, 12282.0)), \
                mock.patch.object(vg, 'load_ratio', return_value=1.0), \
                mock.patch('sys.platform', 'win32'), mock.patch.object(vg, '_say') as say, \
                mock.patch('builtins.print'):
            vg.admit(job(), 1.5)
        try:
            self.assertEqual(sum('Sysmem Fallback' in str(c) for c in say.call_args_list), 1)
        finally:
            vg.finish()


if __name__ == '__main__':
    unittest.main()
