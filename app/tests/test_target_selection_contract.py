"""Regression tests for the explicit target-person selection contract."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop.target_selection import (  # noqa: E402
    normalize_target_selection,
    selection_diagnostic_for_mode,
    selection_face_indices,
    selection_group_ids,
)


class TargetSelectionContractTests(unittest.TestCase):
    def test_one_selected_person_only_its_group_is_eligible(self):
        target_groups = [10, 20]
        selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 0}, person_count=2)
        self.assertEqual(selection_group_ids(target_groups, selection), {10})
        self.assertEqual(selection_face_indices(target_groups, selection), [0])

    def test_changing_selected_person_changes_only_the_eligible_group(self):
        target_groups = [10, 20]
        first = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 0}, person_count=2)
        second = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 1}, person_count=2)
        self.assertEqual(selection_face_indices(target_groups, first), [0])
        self.assertEqual(selection_face_indices(target_groups, second), [1])

    def test_multiple_reference_angles_stay_in_one_person_bank(self):
        target_groups = [7, 7, 9]
        selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 0}, person_count=2)
        self.assertEqual(selection_group_ids(target_groups, selection), {7})
        self.assertEqual(selection_face_indices(target_groups, selection), [0, 1])

    def test_multi_person_mode_is_explicit_and_keeps_both_people(self):
        target_groups = [7, 7, 9]
        selection = normalize_target_selection(
            {"selection_mode": "multi_person", "person_ids": [0, 1]}, person_count=2)
        self.assertTrue(selection["valid"])
        self.assertEqual(selection_group_ids(target_groups, selection), {7, 9})
        self.assertEqual(selection_face_indices(target_groups, selection), [0, 1, 2])

    def test_no_target_person_selected_has_no_eligible_faces(self):
        target_groups = [7, 9]
        selection = normalize_target_selection(
            {"selection_mode": "selected"}, person_count=2)
        self.assertFalse(selection["valid"])
        self.assertEqual(selection["diagnostic"], "selection_required")
        self.assertEqual(selection_face_indices(target_groups, selection), [])

    def test_invalid_person_index_is_diagnostic_and_never_redirected(self):
        target_groups = [7, 9]
        selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 99}, person_count=2)
        self.assertFalse(selection["valid"])
        self.assertEqual(selection["diagnostic"], "invalid_person_id")
        self.assertIsNone(selection["person_id"])
        self.assertEqual(selection_face_indices(target_groups, selection), [])

    def test_backend_admission_diagnostics_do_not_fallback(self):
        selected = normalize_target_selection(
            {"selection_mode": "selected", "person_id": 0}, person_count=1)
        self.assertEqual(
            selection_diagnostic_for_mode("selected", selected, 0),
            "target_required",
        )
        self.assertIsNone(selection_diagnostic_for_mode("selected", selected, 1))
        self.assertEqual(
            selection_diagnostic_for_mode(
                "selected", normalize_target_selection({}, person_count=1), 1),
            "selection_required",
        )

    def test_processmgr_uses_canonical_selection_filter(self):
        source = (ROOT / "app" / "roop" / "ProcessMgr.py").read_text(encoding="utf-8")
        self.assertIn("selection_group_ids", source)
        self.assertIn("g in self.selected_target_groups", source)
        self.assertIn('("selected", "selected_multi")', source)

    def test_stable_person_ids_select_all_angles_for_the_selected_person(self):
        target_groups = [0, 0, 1]
        person_ids = ["tp-a", "tp-b"]
        selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": "tp-b"},
            person_count=2, target_person_ids=person_ids)
        self.assertEqual(selection_group_ids(target_groups, selection, person_ids), {1})
        self.assertEqual(selection_face_indices(target_groups, selection, person_ids), [2])

    def test_parallel_angle_ids_do_not_turn_second_angle_into_a_new_person(self):
        target_groups = [0, 0, 1]
        parallel_ids = ["tp-a", "tp-a", "tp-b"]
        selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": "tp-a"},
            person_count=2, target_person_ids=["tp-a", "tp-b"])
        self.assertEqual(selection_group_ids(target_groups, selection, parallel_ids), {0})
        self.assertEqual(selection_face_indices(target_groups, selection, parallel_ids), [0, 1])


    def test_stable_person_ids_select_all_angles_for_the_selected_person(self):
        target_groups = [0, 0, 1]
        person_ids = ["tp-a", "tp-b"]
        selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": "tp-b"},
            person_count=2, target_person_ids=person_ids)
        self.assertEqual(selection_group_ids(target_groups, selection, person_ids), {1})
        self.assertEqual(selection_face_indices(target_groups, selection, person_ids), [2])

    def test_parallel_angle_ids_do_not_turn_second_angle_into_a_new_person(self):
        target_groups = [0, 0, 1]
        parallel_ids = ["tp-a", "tp-a", "tp-b"]
        selection = normalize_target_selection(
            {"selection_mode": "selected", "person_id": "tp-a"},
            person_count=2, target_person_ids=["tp-a", "tp-b"])
        self.assertEqual(selection_group_ids(target_groups, selection, parallel_ids), {0})
        self.assertEqual(selection_face_indices(target_groups, selection, parallel_ids), [0, 1])


class LegacyDirectCallerSelection(unittest.TestCase):
    """A caller with NO canonical request (Gradio, virtualcam, every bench that
    builds ProcessOptions itself) means "the captured people" by "selected".

    Since Stage 13 that path normalized to selection_mode "none" and selected
    nobody, so four standing benches rendered untouched video for a week and
    reported fps for it (`no track entry matched 807 100.0%`). The API path,
    which always carries a request, keeps its explicit-selection contract.
    """

    def _options(self, swap_mode, **kw):
        from roop.ProcessOptions import ProcessOptions
        options = ProcessOptions([], 0.6, 0.8, swap_mode, 0, "", None, 1, 256,
                                 False, False, **kw)
        options.legacy_target_face_groups = [0, 1]
        return options

    def test_no_request_selected_mode_selects_every_captured_person(self):
        from roop.target_selection import resolve_processing_selection
        groups, selection, selected = resolve_processing_selection(self._options("selected"), 2)
        self.assertEqual(groups, [0, 1])
        self.assertTrue(selection["valid"])
        self.assertEqual(selected, {0, 1})

    def test_no_request_all_faces_mode_is_unchanged(self):
        from roop.target_selection import resolve_processing_selection
        _g, selection, selected = resolve_processing_selection(self._options("all"), 2)
        self.assertEqual(selection["selection_mode"], "none")
        self.assertEqual(selected, set())

    def test_no_request_with_no_captured_person_selects_nobody(self):
        from roop.target_selection import resolve_processing_selection
        _g, _s, selected = resolve_processing_selection(self._options("selected"), 0)
        self.assertEqual(selected, set())

    def test_an_explicit_legacy_selection_is_respected(self):
        from roop.target_selection import resolve_processing_selection
        options = self._options("selected", selection_state={
            "selection_mode": "selected", "person_id": 1})
        _g, _s, selected = resolve_processing_selection(options, 2)
        self.assertEqual(selected, {1})

    def test_an_explicit_legacy_no_selection_is_respected(self):
        from roop.target_selection import resolve_processing_selection
        options = self._options("selected", selection_state={
            "selection_mode": "none"})
        _g, selection, selected = resolve_processing_selection(options, 2)
        self.assertEqual(selection["selection_mode"], "none")
        self.assertEqual(selected, set())

    def test_a_request_that_selects_nobody_still_selects_nobody(self):
        """The API contract: an invalid/absent selection is a refusal, never a
        silent widening to everyone."""
        from roop.target_selection import resolve_processing_selection
        request = {"target_groups": [0, 1], "target_person_ids": ["tp_a", "tp_b"],
                   "selection_state": {"selection_mode": "selected", "person_id": None}}
        options = self._options("selected", processing_request=request)
        _g, selection, selected = resolve_processing_selection(options, 2)
        self.assertFalse(selection["valid"])
        self.assertEqual(selected, set())


if __name__ == "__main__":
    unittest.main()
