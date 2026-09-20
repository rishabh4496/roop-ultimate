"""Preview/render parity tests for the canonical processing request."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop.processing_request import (  # noqa: E402
    normalize_processing_request,
    selection_log_line,
)
from roop.target_selection import selection_face_indices  # noqa: E402


class PreviewRenderSelectionContractTests(unittest.TestCase):
    def test_same_normalized_request_makes_same_person_eligible(self):
        payload = {
            "detection": "Selected face",
            "selection_state": {
                "selection_mode": "selected",
                "person_id": 1,
                "target_reference_index": 2,
            },
            "face_mapping": [-1, 0],
            "source_mapping_names": [None, "source-a"],
            "selected_source_name": "source-a",
            "target_index": 0,
        }
        common = {
            "target_groups": [0, 1, 1],
            "source_count": 1,
            "selected_source_gallery_index": 0,
            "current_source_names": ["source-a"],
            "target_media_index": 0,
            "request_id": "parity-test",
        }

        preview_request = normalize_processing_request(payload, **common)
        render_request = normalize_processing_request(payload, **common)

        # These are the fields consumed by live_swap and batch rendering. The
        # objects can have the same request id without sharing mutable state.
        for key in (
            "swap_mode", "selection_state", "target_groups",
            "target_face_count", "target_person_count", "target_media_index",
            "face_mapping", "source_index_mapping", "source_index",
        ):
            self.assertEqual(preview_request[key], render_request[key], key)

        eligible_preview = selection_face_indices(
            preview_request["target_groups"],
            preview_request["selection_state"],
        )
        eligible_render = selection_face_indices(
            render_request["target_groups"],
            render_request["selection_state"],
        )
        self.assertEqual(eligible_preview, [1, 2])
        self.assertEqual(eligible_preview, eligible_render)
        self.assertEqual(preview_request["source_index"], 1)

    def test_selection_diagnostic_log_contains_canonical_state_and_request_id(self):
        request = normalize_processing_request(
            {
                "detection": "Selected face",
                "selection_state": {
                    "selection_mode": "selected",
                    "person_id": 0,
                },
                "face_mapping": [0],
            },
            target_groups=[0],
            source_count=1,
            target_media_index=0,
            request_id="req-123",
        )
        line = selection_log_line(request, "preview")
        self.assertIn("[Selection]", line)
        self.assertIn("request=req-123", line)
        self.assertIn("mode=selected", line)
        self.assertIn("person=0", line)
        self.assertIn("mapping=[0]", line)
        self.assertIn("target_media=0", line)


if __name__ == "__main__":
    unittest.main()
