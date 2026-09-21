"""Source faceset mapping stays separate from target-person identity state.

These tests exercise the pure mapping contract used by both the React payload
builder and the API boundary.  They deliberately use source indices 0 and 1
as distinguishable identities so an accidental compaction/fallback is visible.
"""

import os
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop.processing_request import (  # noqa: E402
    normalize_processing_request,
    normalize_source_index_mapping,
    remap_source_index_mapping_after_move,
    remap_source_index_mapping_after_removal,
    resolve_selected_source_index,
    source_index_mapping_errors,
)


class SourceFacesetMappingTests(unittest.TestCase):
    def test_removing_a_source_skips_that_binding_and_compacts_later_sources(self):
        self.assertEqual(
            remap_source_index_mapping_after_removal([0, 1, 2], 0),
            [-1, 0, 1],
        )

    def test_reordering_a_source_follows_the_source_not_the_array_slot(self):
        # Move old source 0 after old source 2. Person bindings follow the
        # source identities, so old [0, 1, 2] becomes [2, 0, 1].
        self.assertEqual(
            remap_source_index_mapping_after_move([0, 1, 2], 0, 2),
            [2, 0, 1],
        )

    def test_source_zero_does_not_redirect_when_another_source_is_removed(self):
        self.assertEqual(
            remap_source_index_mapping_after_removal([0, 1], 1),
            [0, -1],
        )

    def test_invalid_mapping_is_skip_with_a_diagnostic_not_source_zero(self):
        raw = [None, "", "bad", 4, -1, 1]
        self.assertEqual(
            normalize_source_index_mapping(raw, 2, "selected"),
            [-1, -1, -1, -1, -1, 1],
        )
        self.assertEqual(source_index_mapping_errors(raw, 2, "selected"), [0, 1, 2, 3])
        self.assertEqual(resolve_selected_source_index([-1, -1], 0), -1)

    def test_selected_index_is_source_gallery_space_not_target_person_space(self):
        request = normalize_processing_request(
            {
                "detection": "Selected face",
                "selection_state": {
                    "selection_mode": "selected",
                    "person_id": 1,
                },
                "face_mapping": [-1, 0],
            },
            target_groups=[0, 1],
            source_count=1,
            selected_source_gallery_index=0,
        )
        self.assertEqual(request["selection_state"]["person_id"], 1)
        self.assertEqual(request["selected_source_gallery_index"], 0)
        # The selected source is in gallery slot 0, but maps to target-person
        # slot 1. These are intentionally different namespaces.
        self.assertEqual(request["source_index"], 1)
        self.assertNotEqual(
            request["selected_source_gallery_index"],
            request["selection_state"]["person_id"],
        )

    def test_selected_people_with_only_the_second_person_mapped_uses_that_persons_slot(self):
        # Two people captured, source 0 = "person_g", source 1 = "person_j".
        # The user maps ONLY the second person (B -> person_j). ProcessMgr treats a
        # one-person selection as `single_person` in EITHER selected mode and
        # reads `selected_index` as the mapped-list slot, so that slot has to be
        # B's rank (1) -- not the gallery-highlighted source, which maps to
        # nobody here and resolved to -1: B was refused "no source faceset"
        # while A, whom the user had NOT mapped, was the only one able to swap.
        people = ["tp_a", "tp_b"]
        request = normalize_processing_request(
            {
                "detection": "Selected people",
                "selection_state": {"selection_mode": "multi_person", "person_ids": ["tp_b"]},
                "target_person_source_mapping": {"tp_b": "src_person_j"},
            },
            target_groups=[0, 1],
            source_count=2,
            selected_source_gallery_index=0,
            current_source_ids=["src_person_g", "src_person_j"],
            target_person_ids=people,
        )
        self.assertEqual(request["swap_mode"], "selected_multi")
        self.assertEqual(request["source_index_mapping"], [-1, 1])
        self.assertEqual(request["source_index"], 1)

        # Both mapped: two people, so ProcessMgr indexes by rank and the single
        # slot is irrelevant -- but it must still not be a lie (-1 is honest).
        both = normalize_processing_request(
            {
                "detection": "Selected people",
                "selection_state": {"selection_mode": "multi_person", "person_ids": people},
                "target_person_source_mapping": {"tp_a": "src_person_g", "tp_b": "src_person_j"},
            },
            target_groups=[0, 1], source_count=2, selected_source_gallery_index=0,
            current_source_ids=["src_person_g", "src_person_j"], target_person_ids=people,
        )
        self.assertEqual(both["source_index_mapping"], [0, 1])

        # One person mapped to a source that no longer exists: honest skip.
        gone = normalize_processing_request(
            {
                "detection": "Selected people",
                "selection_state": {"selection_mode": "multi_person", "person_ids": ["tp_b"]},
                "target_person_source_mapping": {"tp_b": "src_removed"},
            },
            target_groups=[0, 1], source_count=2, selected_source_gallery_index=0,
            current_source_ids=["src_person_g", "src_person_j"], target_person_ids=people,
        )
        self.assertEqual(gone["source_index"], -1)

    def test_preview_and_render_receive_identical_source_mapping(self):
        payload = {
            "detection": "Selected people",
            "selection_state": {
                "selection_mode": "multi_person",
                "person_ids": [0, 1],
            },
            "face_mapping": [1, 0],
            "target_index": 2,
        }
        common = {
            "target_groups": [0, 1, 1],
            "source_count": 2,
            "selected_source_gallery_index": 1,
            "target_media_index": 2,
            "request_id": "mapping-parity",
        }
        preview = normalize_processing_request(payload, **common)
        render = normalize_processing_request(payload, **common)
        for key in (
            "selection_state", "target_groups", "target_media_index",
            "face_mapping", "source_index_mapping", "selected_source_gallery_index",
            "source_index", "source_mapping_errors",
        ):
            self.assertEqual(preview[key], render[key], key)
        self.assertEqual(preview["source_index_mapping"], [1, 0])
        self.assertEqual(preview["source_index"], 0)

    def test_empty_mapping_does_not_change_all_faces_gallery_order(self):
        request = normalize_processing_request(
            {
                "detection": "All faces",
                "face_mapping": [],
                "source_mapping_ids": [],
            },
            target_groups=[],
            source_count=2,
            selected_source_gallery_index=0,
            current_source_ids=["source-a", "source-b"],
        )
        self.assertIsNone(request["source_index_mapping"])
        self.assertEqual(request["source_index"], 0)

    def test_queued_source_names_survive_a_gallery_reorder(self):
        payload = {
            "detection": "Selected people",
            "selection_state": {
                "selection_mode": "multi_person",
                "person_ids": [0, 1],
            },
            "face_mapping": [0, 1],
            "source_mapping_names": ["Alice", "Bob"],
            "selected_source_name": "Alice",
        }
        request = normalize_processing_request(
            payload,
            target_groups=[0, 1],
            source_count=2,
            selected_source_gallery_index=0,
            current_source_names=["Bob", "Alice"],
        )
        self.assertEqual(request["source_index_mapping"], [1, 0])
        self.assertEqual(request["selected_source_gallery_index"], 1)
        # Alice is now gallery slot 1 and is mapped to person slot 0.
        self.assertEqual(request["source_index"], 0)

    def test_stable_source_ids_survive_duplicate_names_and_reorder(self):
        request = normalize_processing_request(
            {
                "detection": "Selected people",
                "selection_state": {
                    "selection_mode": "multi_person",
                    "person_ids": [0, 1],
                },
                "face_mapping": [0, 1],
                "source_mapping_ids": ["source-a#face-0", "source-a#face-1"],
                "selected_source_id": "source-a#face-0",
                "source_mapping_names": ["same.png", "same.png"],
            },
            target_groups=[0, 1],
            source_count=2,
            selected_source_gallery_index=0,
            current_source_names=["same.png", "same.png"],
            current_source_ids=["source-a#face-1", "source-a#face-0"],
        )
        self.assertEqual(request["source_index_mapping"], [1, 0])
        self.assertEqual(request["selected_source_gallery_index"], 1)
        self.assertEqual(request["source_index"], 0)


if __name__ == "__main__":
    unittest.main()
