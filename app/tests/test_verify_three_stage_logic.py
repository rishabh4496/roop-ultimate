"""The logic of `verify_three_stage_pipeline.py` that needs no GPU: scene truth, the five gates on
synthetic run records, and the sanitiser step. A gate that cannot fail proves nothing, so each gate is
shown failing as well as passing (the harness's real renders are in `docs/perf/stage5_verify/`).
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import verify_three_stage_pipeline as v  # noqa: E402


def run(plan, pre=867, stage3=0, fa3=0):
    return {'plan': plan, 'detector_calls': {'pre': pre, 'stage3': stage3}, 'fa_get': {'pre': 0, 'stage3': fa3},
            'get_all_faces': {'pre': 5, 'stage3': 0}, 'retry_rotated_stage3': 0}


GOOD_PLAN = sorted(v.TRUTH_TARGET | set(range(191, 200)) | set(range(451, 460)))
AUDIT = {'target_detected_frames': sorted(v.TRUTH_TARGET)}


class Truth(unittest.TestCase):
    def test_regions_tile_the_clip(self):
        spans = [v.EMPTY_HEAD, v.DISTRACTOR_RUN, v.FLASH, v.GAP, v.TARGET_RUN, v.OUTRO]
        covered = [n for a, b in spans for n in range(a, b + 1)]
        self.assertEqual(covered, list(range(v.N_FRAMES)))

    def test_truth_sets_are_the_briefs(self):
        self.assertEqual(len(v.TRUTH_TARGET), 5 + 151)
        self.assertTrue({200, 204, 300, 450} <= v.TRUTH_TARGET)
        self.assertFalse({199, 205, 299, 451} & v.TRUTH_TARGET)
        self.assertEqual(v.MUST_BE_EMPTY, set(range(0, 191)) | set(range(460, 600)))
        self.assertFalse(v.MUST_BE_EMPTY & v.TRUTH_TARGET)

    def test_profile_sweep_runs_from_70_deg_to_frontal_and_then_holds(self):
        self.assertEqual(v.source_frame_for(0), v.PROFILE_SRC[0])
        seq = [v.source_frame_for(k) for k in range(0, 114)]
        self.assertEqual(seq, sorted(seq))
        self.assertEqual(seq[-1], v.PROFILE_SRC[1])
        hold = {v.source_frame_for(k) for k in range(114, 151)}
        self.assertTrue(all(v.PROFILE_SRC[1] <= f <= v.SRC_LAST for f in hold))

    def test_landscape_is_deterministic_and_the_right_size(self):
        a, b = v.landscape(), v.landscape()
        self.assertEqual(a.shape, (v.H, v.W, 3))
        self.assertTrue(np.array_equal(a, b))

    def test_ranges_of(self):
        self.assertEqual(v.ranges_of([3, 4, 5, 9, 11, 12]), ['3-5', '9', '11-12'])
        self.assertEqual(v.ranges_of([]), [])


class Gates(unittest.TestCase):
    def test_recall_passes_on_a_complete_plan_and_names_the_missing_frames(self):
        self.assertTrue(v.inv_recall(run(GOOD_PLAN), AUDIT)['pass'])
        gap = [n for n in GOOD_PLAN if not 373 <= n <= 379]
        r = v.inv_recall(run(gap), AUDIT)
        self.assertFalse(r['pass'])
        self.assertEqual(r['missing'], ['373-379'])
        self.assertEqual(r['missing_but_single_frame_detectable'], ['373-379'])

    def test_recall_flags_a_lost_flash(self):
        r = v.inv_recall(run([n for n in GOOD_PLAN if not 200 <= n <= 204]), AUDIT)
        self.assertFalse(r['pass'])
        self.assertEqual(r['missing_flash'], ['200-204'])

    def test_recall_needs_a_plan_at_all(self):
        self.assertFalse(v.inv_recall(run(None), AUDIT)['pass'])

    def test_rejection_catches_a_distractor_or_outro_frame(self):
        self.assertTrue(v.inv_rejection(run(GOOD_PLAN))['pass'])
        for leaked in (150, 0, 190, 460, 599):
            r = v.inv_rejection(run(sorted(GOOD_PLAN + [leaked])))
            self.assertFalse(r['pass'], leaked)
            self.assertEqual(r['leaked_count'], 1)

    def test_rejection_allows_the_dilation_margin(self):
        self.assertTrue(v.inv_rejection(run(sorted(GOOD_PLAN + [191, 195, 459])))['pass'])

    def test_no_redetect_passes_only_with_a_counter_that_saw_the_prepass(self):
        self.assertTrue(v.inv_no_redetect(run(GOOD_PLAN))['pass'])
        self.assertFalse(v.inv_no_redetect(run(GOOD_PLAN, stage3=1))['pass'])
        self.assertFalse(v.inv_no_redetect(run(GOOD_PLAN, fa3=1))['pass'])
        vacuous = v.inv_no_redetect(run(GOOD_PLAN, pre=0))
        self.assertFalse(vacuous['pass'], 'a counter that never fired is not evidence')

    def test_sanitation_gate_runs_the_real_functions(self):
        r = v.inv_sanitation()
        self.assertTrue(r['pass'], r)
        self.assertEqual(r['type_errors'], [])
        self.assertIn('sanitize_face_attributes', r['steps'])
        self.assertEqual(r['steps']['occluded_frac_reader'], 0.0)

    def test_sanitation_gate_fails_when_the_sanitiser_raises(self):
        from unittest import mock
        from roop import face_sanity
        with mock.patch.object(face_sanity, 'sanitize_face_attributes', side_effect=TypeError('NoneType vs float')):
            r = v.inv_sanitation()
        self.assertFalse(r['pass'])
        self.assertTrue(any('TypeError' in e for e in r['type_errors']), r)

    def test_sanitation_gate_fails_when_the_sanitiser_does_nothing(self):
        # a no-op leaves det_score None, and the comparison every consumer makes raises
        from unittest import mock
        from roop import face_sanity
        with mock.patch.object(face_sanity, 'sanitize_face_attributes', return_value=False):
            r = v.inv_sanitation()
        self.assertFalse(r['pass'], r)
        self.assertTrue(any('det_score_comparable' in e for e in r['type_errors']), r)


if __name__ == '__main__':
    unittest.main()
