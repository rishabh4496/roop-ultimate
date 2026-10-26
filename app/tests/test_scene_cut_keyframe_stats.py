"""The hardware cut detector must not call a keyframe a scene cut because it IS a keyframe.

The keyframe pre-scan samples only I-frames. An I-frame decodes as ~100% intra coding units and at a lower QP than the
P-frames around it, on every keyframe, so the detector's intra-ratio (>= 0.85) and QP-jump (>= 6) cues fired for every
sampled keyframe: a 1080p clip with two real cuts reported four (the extra two were forced GOP boundaries whose luma
differed by 2.4 and 3.0 against 30-60 at the real cuts). The false cuts became Stage 2 cut hints (a forced full detection
at each one, defeating the K stride), HUD "scene cuts", and chapter markers. Found by test_pipeline_3stage_sync_and_fps.py.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.scene_detector import HardwareBitstreamCutDetector  # noqa: E402

# what ParseDecodeStats returned on the real clip (frame 45: an I-frame in the middle of a shot)
I_FRAME = {'cu_type': [0] * 400, 'qp_luma': [6.4] * 400, 'mv0_x': [0.0] * 400, 'mv0_y': [0.0] * 400}
P_FRAME = {'cu_type': [1] * 399 + [0], 'qp_luma': [16.0] * 400, 'mv0_x': [0.5] * 400, 'mv0_y': [0.5] * 400}
ALL_INTRA_P = {'cu_type': [0] * 400, 'qp_luma': [16.0] * 400, 'mv0_x': [0.0] * 400, 'mv0_y': [0.0] * 400}


class KeyframeStatsAreNotCutEvidence(unittest.TestCase):
    def test_a_keyframe_mid_shot_is_not_a_cut_however_intra_it_looks(self):
        d = HardwareBitstreamCutDetector()
        self.assertFalse(d.observe(0, stats=I_FRAME, keyframe=True))
        for idx in (45, 105, 165):                                   # a forced GOP boundary every 45 frames
            self.assertFalse(d.observe(idx, stats=I_FRAME, keyframe=True), idx)
        self.assertEqual(d.scene_cuts, [])

    def test_the_old_behaviour_is_what_the_bug_was(self):
        d = HardwareBitstreamCutDetector()
        d.observe(0, stats=I_FRAME)                                  # caller does not say: legacy path
        self.assertTrue(d.observe(45, stats=I_FRAME), 'the cue that produced the false cuts no longer exists')

    def test_a_non_key_frame_that_is_all_intra_is_still_evidence_of_a_cut(self):
        d = HardwareBitstreamCutDetector()
        d.observe(30, stats=P_FRAME, keyframe=False)
        self.assertTrue(d.observe(40, stats=ALL_INTRA_P, keyframe=False))
        self.assertEqual(d.scene_cuts, [40])

    def test_keyframe_qp_does_not_poison_the_qp_baseline_of_p_frames(self):
        d = HardwareBitstreamCutDetector()
        d.observe(45, stats=I_FRAME, keyframe=True)                  # QP 6.4 must not become the baseline
        self.assertIsNone(d._last_mean_qp)
        self.assertFalse(d.observe(50, stats=P_FRAME, keyframe=False))
        self.assertEqual(d._last_mean_qp, 16.0)                      # the P-frame sets it

    def test_hw_frames_pass_their_own_keyframe_flag(self):
        class Frame:
            index, decode_stats, tensor, is_keyframe = 45, I_FRAME, None, True
        d = HardwareBitstreamCutDetector()
        self.assertFalse(d.observe_hw_frame(Frame()))
        Frame.is_keyframe = False
        d = HardwareBitstreamCutDetector()
        d.observe(30, stats=P_FRAME, keyframe=False)
        Frame.decode_stats = ALL_INTRA_P
        self.assertTrue(d.observe_hw_frame(Frame()))


class CutLocalization(unittest.TestCase):
    """A cut with no keyframe at it is only visible at the next sampled keyframe; bisection finds its first frame."""

    def test_every_possible_cut_position_is_found_exactly(self):
        from roop.sparse_range_scan import locate_cut_frame
        lo, hi = 105, 150                                   # the keyframes either side of the cut at 120 in the e2e clip
        for cut in range(lo + 1, hi + 1):                   # `cut` = first frame of the new shot
            found, probes = locate_cut_frame(lo, hi, lambda f, c=cut: f >= c)
            self.assertEqual(found, cut)
            self.assertLessEqual(probes, 6)                 # ceil(log2(45)) decodes, not 45

    def test_adjacent_samples_need_no_probe(self):
        from roop.sparse_range_scan import locate_cut_frame
        self.assertEqual(locate_cut_frame(59, 60, lambda f: True), (60, 0))

    def test_relocate_moves_only_the_last_cut_earlier(self):
        d = HardwareBitstreamCutDetector()
        d.scene_cuts = [60, 150]
        d.relocate_last_cut(120)
        self.assertEqual(d.scene_cuts, [60, 120])
        d.relocate_last_cut(130)                            # never later than what was observed
        self.assertEqual(d.scene_cuts, [60, 120])


class EdgeSamples(unittest.TestCase):
    """The keyframe schedule left the first and last GOP of a trim unsampled: 122 of 155 face frames got a range."""

    def test_both_ends_of_the_interval_are_always_sampled(self):
        from roop.sparse_range_scan import with_edge_samples
        self.assertEqual(with_edge_samples([35, 80, 125], 155), [0, 35, 80, 125, 154])

    def test_no_duplicates_and_nothing_outside_the_interval(self):
        from roop.sparse_range_scan import with_edge_samples
        self.assertEqual(with_edge_samples([0, 50, 154, 200, -3], 155), [0, 50, 154])

    def test_a_single_frame_interval(self):
        from roop.sparse_range_scan import with_edge_samples
        self.assertEqual(with_edge_samples([], 1), [0])

    def test_a_face_in_every_frame_is_covered_end_to_end(self):
        """Positives at the trim's two ends plus the keyframes between them leave no uncovered frame."""
        from roop.sparse_range_scan import build_active_ranges, with_edge_samples
        samples = with_edge_samples([35, 80, 125], 155)
        ranges, mask = build_active_ranges(samples, 155, 6)
        self.assertEqual(ranges, [(0, 154)])
        self.assertTrue(mask.all())
        ranges, mask = build_active_ranges([35, 80, 125], 155, 6)       # without them: the bug
        self.assertEqual(ranges, [(25, 135)])


@unittest.skipUnless(os.environ.get('ROOP_TEST_LIGHT') != '1', 'needs torch')
class LumaStillDecidesAtAKeyframe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            if not torch.cuda.is_available():
                raise unittest.SkipTest('UNVERIFIED - no CUDA device for the VRAM luma path')
        except ImportError:
            raise unittest.SkipTest('UNVERIFIED - torch not installed')
        cls.torch = torch

    def _frame(self, level):
        return self.torch.full((3, 72, 128), float(level), device='cuda')

    def test_a_real_cut_at_a_keyframe_is_still_found(self):
        d = HardwareBitstreamCutDetector()
        d.observe(0, tensor=self._frame(200), stats=I_FRAME, keyframe=True)
        d.observe(45, tensor=self._frame(201), stats=I_FRAME, keyframe=True)          # same shot: no cut
        self.assertTrue(d.observe(60, tensor=self._frame(60), stats=I_FRAME, keyframe=True))   # 141 levels away
        self.assertEqual(d.scene_cuts, [60])

    def test_a_probe_does_not_become_the_baseline_of_the_next_sample(self):
        """The scan inspects frames BEFORE the one that flagged a cut. Committing them made the next keyframe, in the
        same shot as the flagging one, get compared with a frame of the shot before it: a false cut after every cut."""
        def run(commit_probes):
            d = HardwareBitstreamCutDetector()
            d.observe(45, tensor=self._frame(200), keyframe=True)
            self.assertTrue(d.observe(60, tensor=self._frame(60), keyframe=True))          # the real cut
            d.observe(52, tensor=self._frame(200), keyframe=False, commit=commit_probes)   # a probe, still shot A
            d.observe(46, tensor=self._frame(200), keyframe=False, commit=commit_probes)
            return d.observe(105, tensor=self._frame(61), keyframe=True), d.scene_cuts      # same shot as 60
        self.assertEqual(run(commit_probes=False), (False, [60]))
        flagged, cuts = run(commit_probes=True)                                            # the old behaviour
        self.assertTrue(flagged and cuts == [60, 105], 'the false cut after a real one no longer reproduces')


if __name__ == '__main__':
    unittest.main()
