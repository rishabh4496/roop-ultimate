"""Proof that ProcessMgr actually REACHES roop/temporal_smoother.py.

This file exists because of `memory/adaptive-controller-was-unreachable.md`:
the safe adaptive controller was constructed, configured, consulted in tests
and reported "no effect" for two GPU campaigns, because its only hook was
wired to the sequential encoder loop while production renders through
`_run_stab_parallel`. A flag that is read, an engine that is constructed and a
method that is defined are all perfectly consistent with never running.

So these do not test the filters (test_temporal_smoother.py does). They test
the WIRING: the enable conditions, the accessor every mutating site must go
through, the per-block clone list, the warm-up worst-case, and the fact that
the summary lines are emitted at all.
"""

import ast
import inspect
import unittest
from pathlib import Path

import numpy as np

import roop.ProcessMgr as PM
import roop.procmgr_masking as masking
from roop.temporal_smoother import (AdaptiveLandmarkSmoother,
                                    HighFrequencyFlowStabilizer)

APP = Path(__file__).resolve().parents[1]


def _source(obj):
    return inspect.getsource(obj)


class _Face(dict):
    """Minimal stand-in for insightface's Face: a dict with attribute access,
    whose __getattr__ returns None for a missing key exactly as the real one
    does -- the behaviour that made DMDNet's unguarded landmark read fail late
    rather than raise (Session Log 2026-08-31 Part 2)."""

    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class _Mgr:
    """Just enough ProcessMgr to exercise the two methods under test."""

    _apply_landmark_stab = PM.ProcessMgr._apply_landmark_stab
    _temporal_engine = PM.ProcessMgr._temporal_engine
    _temporal_track_id = staticmethod(PM.ProcessMgr._temporal_track_id)
    _report_smoother_summaries = PM.ProcessMgr._report_smoother_summaries

    def __init__(self, smoother=None, hf=None, frame_idx=0):
        import threading
        self._tls = threading.local()
        self._tls.frame_idx = frame_idx
        self._landmark_smoother = smoother or AdaptiveLandmarkSmoother()
        self._hf_stabilizer = hf or HighFrequencyFlowStabilizer(enabled=False)


def _kps(dx=0.0):
    return np.array([[30., 40.], [70., 40.], [50., 60.], [35., 80.], [65., 80.]],
                    dtype=np.float32) + np.array([dx, 0.], np.float32)


class TheDenseLandmarksActuallyGetSmoothed(unittest.TestCase):
    def test_landmark_2d_106_is_replaced_on_a_contiguous_frame(self):
        mgr = _Mgr()
        d0 = np.zeros((106, 2), np.float32)
        face = _Face(kps=_kps(), landmark_2d_106=d0, _track_id=7)
        mgr._tls.frame_idx = 0
        mgr._apply_landmark_stab(face)

        d1 = np.full((106, 2), 20.0, np.float32)
        face2 = _Face(kps=_kps(dx=1.0), landmark_2d_106=d1, _track_id=7)
        mgr._tls.frame_idx = 1
        mgr._apply_landmark_stab(face2)

        # Pulled back toward the previous frame's landmarks, i.e. smoothed.
        self.assertTrue(np.all(face2.landmark_2d_106 < d1))
        self.assertTrue(np.all(face2.landmark_2d_106 > d0))
        self.assertEqual(mgr._landmark_smoother.stats()['dense_applied'], 1)

    def test_both_the_attribute_and_the_dict_entry_are_written(self):
        """`Face` is a dict subclass. Consumers read it both ways, so writing
        only one leaves the two disagreeing about where the face is."""
        mgr = _Mgr()
        for i in (0, 1):
            face = _Face(kps=_kps(dx=float(i)),
                         landmark_2d_106=np.full((106, 2), float(i), np.float32),
                         _track_id=3)
            mgr._tls.frame_idx = i
            mgr._apply_landmark_stab(face)
        np.testing.assert_array_equal(face.landmark_2d_106,
                                      face['landmark_2d_106'])

    def test_a_face_without_dense_landmarks_is_handled(self):
        """landmark_2d_106 is absent unless the 106 model ran; the real Face
        returns None rather than raising, so this must not assume an array."""
        mgr = _Mgr()
        face = _Face(kps=_kps(), _track_id=1)
        mgr._apply_landmark_stab(face)          # must not raise
        self.assertIsNone(face.landmark_2d_106)

    def test_a_disabled_smoother_touches_nothing(self):
        mgr = _Mgr(smoother=AdaptiveLandmarkSmoother(enabled=False))
        d = np.zeros((106, 2), np.float32)
        face = _Face(kps=_kps(), landmark_2d_106=d, _track_id=1)
        mgr._apply_landmark_stab(face)
        self.assertIs(face.landmark_2d_106, d)

    def test_an_untracked_face_declines_rather_than_sharing_state(self):
        """Without a track id two people in one frame would share a history."""
        mgr = _Mgr()
        face = _Face(kps=_kps(), landmark_2d_106=np.zeros((106, 2), np.float32))
        mgr._apply_landmark_stab(face)
        self.assertEqual(mgr._landmark_smoother.stats()['skipped_no_key'], 1)


class MutatingSitesGoThroughTheAccessor(unittest.TestCase):
    """`_temporal_engine` is what returns the per-BLOCK clone in the parallel
    path. Reading `self._landmark_smoother` directly inside a block would
    advance one shared history from several workers at once."""

    def test_apply_landmark_stab_uses_temporal_engine(self):
        src = _source(PM.ProcessMgr._apply_landmark_stab)
        self.assertIn("_temporal_engine('landmark_smoother')", src)
        self.assertNotIn("self._landmark_smoother", src)

    def test_the_hf_call_site_uses_temporal_engine(self):
        src = _source(PM.ProcessMgr.process_face)
        self.assertIn("_temporal_engine('hf_stabilizer')", src)
        self.assertNotIn("self._hf_stabilizer.stabilize", src)


class BlockLifecycleIsRegistered(unittest.TestCase):
    """Three lists in ProcessMgr must all name the new engines, and each omission
    is a different silent defect."""

    def _run_stab_parallel_src(self):
        return _source(PM.ProcessMgr._run_stab_parallel)

    def test_both_engines_are_cloned_per_block(self):
        """Omission = two blocks thrash one track's state between two frame
        indices, and the HF filter shares cv2 DIS handles across threads."""
        src = self._run_stab_parallel_src()
        clone_loop = src[src.index("for _tname in ("):]
        for name in ("'landmark_smoother'", "'hf_stabilizer'"):
            self.assertIn(name, clone_loop[:600])

    def test_both_engines_take_part_in_the_warmup_worst_case(self):
        """Omission = a block starts from the wrong seed and shows a step at
        every block boundary."""
        src = _source(PM.ProcessMgr._stab_warmup_frames)
        self.assertIn("'_landmark_smoother'", src)
        self.assertIn("'_hf_stabilizer'", src)

    def test_neither_engine_forces_ordered_single_threaded_execution(self):
        """These decline safely out of order, so they must NOT join the ordered
        set -- that would pin threads=1 and cost ~3x for nothing (Session Log
        2026-08-25 Part 3)."""
        src = _source(PM.ProcessMgr)
        ordered = src[src.index("_want_temporal_ordered = "):]
        ordered = ordered[:ordered.index("\n\n")]
        self.assertNotIn("landmark_smoother", ordered)
        self.assertNotIn("hf_stabilizer", ordered)

    def test_the_warmups_do_not_widen_the_block_beyond_the_existing_filters(self):
        """A filter needing a longer warm-up than the ones already shipped
        would silently widen every parallel block."""
        from roop.one_euro import EnhancerStabilizer
        existing = EnhancerStabilizer(strength=1.0).warmup_frames()
        self.assertLessEqual(AdaptiveLandmarkSmoother().warmup_frames(), existing)
        self.assertLessEqual(HighFrequencyFlowStabilizer().warmup_frames(), existing)


class TheSummaryIsActuallyPrinted(unittest.TestCase):
    """`TemporalMaskSmoother.summary_line` and `tracker.summary_line` are both
    defined and called by nothing. A reporting method with no caller reports
    nothing, which is the state these counters exist to prevent."""

    def test_report_is_invoked_at_the_end_of_a_run(self):
        src = _source(PM.ProcessMgr)
        self.assertIn("self._report_smoother_summaries()", src)

    def test_an_enabled_engine_prints_its_line(self):
        import io
        from contextlib import redirect_stdout
        sm = AdaptiveLandmarkSmoother()
        for i in range(4):
            sm.smooth(_kps(dx=float(i)), None, track_id=1, frame_index=i)
        mgr = _Mgr(smoother=sm)
        buf = io.StringIO()
        with redirect_stdout(buf):
            mgr._report_smoother_summaries()
        self.assertIn('[LandmarkSmooth]', buf.getvalue())
        self.assertIn('applied 3/4', buf.getvalue())

    def test_a_disabled_engine_prints_nothing(self):
        import io
        from contextlib import redirect_stdout
        mgr = _Mgr(smoother=AdaptiveLandmarkSmoother(enabled=False))
        buf = io.StringIO()
        with redirect_stdout(buf):
            mgr._report_smoother_summaries()
        self.assertEqual(buf.getvalue(), '')


class EnableConditions(unittest.TestCase):
    def test_the_hf_filter_is_decided_after_the_processors_are_built(self):
        """The small-card policy can strip the enhancer AFTER the user asked
        for it. Reading the REQUEST would leave the filter enabled with nothing
        to damp -- active-looking, and doing nothing."""
        src = _source(PM.ProcessMgr.initialize)
        want = src.index("_want_hf_stabilize")
        decide = src.index("self._hf_stabilizer.enabled = _has_enhancer")
        ready = src.index("self.processors = newprocessors")
        self.assertLess(want, ready, "the request is recorded before the loader")
        self.assertLess(ready, decide, "the decision must follow the loader")

    def test_landmark_smoothing_rides_on_stabilize_face(self):
        src = _source(PM.ProcessMgr.initialize)
        block = src[src.index("self._landmark_smoother.enabled"):][:400]
        self.assertIn("stabilize_face", block)
        self.assertIn("stabilize_landmarks", block)


class ExactlyOnePathOwnsLandmarkSmoothing(unittest.TestCase):
    """The lesson this whole file was written for, and it still caught me.

    `stabilize_landmarks` has TWO possible owners and they must never both be
    live, nor both be dead:

      * `temporal_detection` ON  (the shipped default) -> the tracking pre-pass
        `_build_temporal_faces` smooths kps and lm106 together;
      * `temporal_detection` OFF -> `ProcessMgr._apply_stab`.

    The first version of this feature hung ONLY off `_apply_stab`. That method
    is reachable only through `self.kps_stabilizer`, which `run_batch_inmem`
    sets to None whenever `_temporal_mode` is on -- so on the shipped config the
    filter was enabled, constructed, covered by tests, and never once called.
    The suite was green; only a rendered clip showed it.
    """

    def test_temporal_mode_hands_ownership_to_the_prepass(self):
        src = _source(PM.ProcessMgr.run_batch_inmem)
        block = src[src.index("if self._temporal_mode:"):][:900]
        self.assertIn("self.kps_stabilizer = None", block)
        self.assertIn("self._landmark_smoother.enabled = False", block,
                      "the ProcessMgr-level smoother must be switched off when "
                      "the pre-pass owns the work, or it reports 'enabled but "
                      "never invoked' on every default render")

    def test_apply_stab_is_the_only_route_to_the_processmgr_instance(self):
        """If a second call site appears, the ownership split above stops being
        a split and the two paths can double-smooth."""
        src = _source(PM.ProcessMgr)
        self.assertEqual(src.count("_apply_landmark_stab(face)"), 1)

    def test_the_prepass_couples_the_two_arrays_under_one_smoother(self):
        import roop.procmgr_tracking as tracking
        src = _source(tracking.ProcessMgrTrackingMixin._build_temporal_faces) \
            if hasattr(tracking, 'ProcessMgrTrackingMixin') else \
            (APP / 'roop' / 'procmgr_tracking.py').read_text(encoding='utf-8')
        self.assertIn('AdaptiveLandmarkSmoother', src)
        # bbox keeps its own filter: it is a detection rectangle, not part of
        # the alignment/hull pair the coupling exists to hold together.
        self.assertIn("_smooth('bbox'", src)

    def test_the_prepass_reports_what_it_did(self):
        src = (APP / 'roop' / 'procmgr_tracking.py').read_text(encoding='utf-8')
        self.assertIn('_landmark_smooth_lines', src)
        self.assertIn('summary_line()', src)


class MaskEdgeModeIsWired(unittest.TestCase):
    def test_default_mode_leaves_blur_area_on_the_gaussian_path(self):
        """`face_mask_blend` was calibrated against the Gaussian (30 -> 12 on
        2026-08-22 to kill a halo). The default must not change the ramp shape
        under every existing config."""
        import roop.globals as g
        self.assertEqual(getattr(g, 'mask_edge_mode', 'gaussian'), 'gaussian')
        self.assertEqual(float(getattr(g, 'boundary_illumination_strength', 0.0)),
                         0.0)

    def test_the_distance_mode_changes_the_ramp_and_gaussian_does_not(self):
        import cv2
        import roop.globals as g

        class _Opt:
            pass

        mgr = masking.MaskingMixin()
        matte = np.zeros((256, 256), np.uint8)
        cv2.circle(matte, (128, 128), 70, 255, -1)
        prev = getattr(g, 'mask_edge_mode', 'gaussian')
        try:
            g.mask_edge_mode = 'gaussian'
            a = mgr.blur_area(matte.copy(), 12.0)
            g.mask_edge_mode = 'distance'
            b = mgr.blur_area(matte.copy(), 12.0)
        finally:
            g.mask_edge_mode = prev
        self.assertEqual(a.dtype, b.dtype)
        self.assertGreater(int(np.abs(a.astype(int) - b.astype(int)).max()), 8,
                           "the distance mode produced the Gaussian's ramp -- "
                           "it is not reaching soft_distance_matte")

    def test_a_zero_blend_still_short_circuits_in_both_modes(self):
        import roop.globals as g
        mgr = masking.MaskingMixin()
        matte = np.zeros((64, 64), np.uint8)
        matte[20:44, 20:44] = 255
        prev = getattr(g, 'mask_edge_mode', 'gaussian')
        try:
            g.mask_edge_mode = 'distance'
            out = mgr.blur_area(matte.copy(), 0.0)
        finally:
            g.mask_edge_mode = prev
        # face_mask_blend <= 0 means "no feather"; the distance branch must not
        # claim it, or the setting stops being able to turn feathering off.
        self.assertGreater(int(out.max()), 200)

    def test_the_fallback_warning_is_bounded(self):
        """These sit on the per-face path: an unbounded print is one line per
        face per frame, ~120k on a long render."""
        import io
        from contextlib import redirect_stdout
        masking.reset_mask_warnings()
        buf = io.StringIO()
        with redirect_stdout(buf):
            for _ in range(50):
                masking._warn_once('k', 'boom')
        self.assertEqual(buf.getvalue().count('boom'), 1)
        self.assertEqual(masking.mask_warn_counts()['k'], 50)
        masking.reset_mask_warnings()


class SettingsReachTheRunPath(unittest.TestCase):
    def test_process_options_carries_the_new_keys(self):
        from roop.ProcessOptions import ProcessOptions
        sig = inspect.signature(ProcessOptions.__init__).parameters
        for key in ('stabilize_landmarks', 'stabilize_hf_texture',
                    'stabilize_hf_texture_weight'):
            self.assertIn(key, sig)

    def test_the_api_run_path_populates_them(self):
        """A ProcessOptions parameter nothing passes silently takes its default
        forever, so the control looks wired and is not."""
        src = (APP / 'api.py').read_text(encoding='utf-8')
        for key in ('stabilize_landmarks', 'stabilize_hf_texture',
                    'stabilize_hf_texture_weight'):
            self.assertIn('%s=' % key, src, '%s never reaches ProcessOptions' % key)
            self.assertIn('payload.get("%s"' % key, src,
                          '%s is passed but never read off the payload' % key)

    def test_the_preview_and_run_paths_both_set_the_mask_edge_globals(self):
        """These two DO change a single frame, so a preview that omitted them
        would show the user something the render does not produce."""
        src = (APP / 'api.py').read_text(encoding='utf-8')
        for key in ('mask_edge_mode', 'boundary_illumination_strength'):
            self.assertEqual(src.count('roop_globals.%s = ' % key), 2,
                             '%s must be set on BOTH the preview and run paths'
                             % key)

    def test_globals_defined_so_config_sync_can_reach_them(self):
        """tests/config_sync.py copies a config key onto globals only when
        globals already DEFINES it. A key it does not define is silently not
        synced, so every bench would render the module default."""
        import roop.globals as g
        for key in ('mask_edge_mode', 'boundary_illumination_strength'):
            self.assertTrue(hasattr(g, key))

    def test_settings_both_load_and_save_the_new_keys(self):
        """Loaded but not saved is the classic drift: the value works this
        session and is dropped the next time anything writes settings."""
        import settings
        cfg = settings.Settings(str(APP / 'config.yaml'))
        save_src = inspect.getsource(settings.Settings.save)
        for key in ('stabilize_landmarks', 'stabilize_hf_texture',
                    'stabilize_hf_texture_weight', 'mask_edge_mode',
                    'boundary_illumination_strength'):
            self.assertTrue(hasattr(cfg, key), '%s is not loaded' % key)
            self.assertIn("'%s'" % key, save_src, '%s is not saved' % key)


class DefaultsAreNoOps(unittest.TestCase):
    """Everything but the landmark smoother ships off, and 'off' has to mean
    bit-identical, not nearly."""

    def test_hf_texture_is_opt_in(self):
        import settings
        # Defaults must be tested independently of the user's persisted config.
        cfg = settings.Settings(str(APP / '__missing_defaults_test__.yaml'))
        self.assertFalse(cfg.stabilize_hf_texture)

    def test_landmark_smoothing_ships_on(self):
        """It IS the boundary-crawl fix rather than an experiment, and it
        declines safely wherever frames are non-contiguous."""
        import settings
        cfg = settings.Settings(str(APP / '__missing_defaults_test__.yaml'))
        self.assertTrue(cfg.stabilize_landmarks)

    def test_react_defaults_agree_with_the_backend(self):
        import json
        import re
        src = (APP.parent / 'react-ui' / 'src' / 'components' / 'faceswap'
               / 'defaults.js').read_text(encoding='utf-8')
        for key, expected in (('stabilize_landmarks', 'true'),
                              ('stabilize_hf_texture', 'false'),
                              ('stabilize_hf_texture_weight', '0.15'),
                              ('boundary_illumination_strength', '0')):
            m = re.search(r'^\s*%s:\s*([^,]+),' % key, src, re.M)
            self.assertIsNotNone(m, '%s missing from defaults.js' % key)
            self.assertEqual(m.group(1).strip(), expected)
        self.assertIn("mask_edge_mode: 'gaussian'", src)


if __name__ == '__main__':
    unittest.main()
