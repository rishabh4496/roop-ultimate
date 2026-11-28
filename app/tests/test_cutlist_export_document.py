"""React UI 2.0's cut-list export (react-ui-v2/src/engine/interchange.js cutListDocument) against the
engine's real exporters. 1.0 sent face_segments: [] and read keys the engine never returns, so no
test ever fed segments through these; this pins the document shape 2.0 sends."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import nle_interchange  # noqa: E402

# What cutListDocument builds for presence Ann [1,40],[100,160] and Bob [30,90], In 1 / Out 150, 25 fps.
DOC = {
    "name": "clip",
    "source_media": "clip.mp4",
    "total_frames": 300,
    "timeline": {
        "fps": 25, "frame_start": 0, "frame_end": 150,
        "face_segments": [
            {"in": 0, "out": 40, "face_id": "Ann", "speaker_cluster_id": "p-ann", "track": 1},
            {"in": 29, "out": 90, "face_id": "Bob", "speaker_cluster_id": "p-bob", "track": 2},
            {"in": 99, "out": 150, "face_id": "Ann", "speaker_cluster_id": "p-ann", "track": 1},
        ],
    },
}


class CutListExportTest(unittest.TestCase):
    def test_edl_has_the_source_on_v1_and_one_swap_event_per_segment(self):
        edl = nle_interchange.export_cmx3600_edl(DOC, fps=25)
        self.assertIn("FROM CLIP NAME: clip.mp4", edl)
        self.assertEqual(edl.count("* TRACK: V2"), 3)
        # Bob's stretch: frame 29 at 25 fps is 00:00:01:04, frame 90 is 00:00:03:15.
        self.assertIn("00:00:01:04 00:00:03:15", edl)
        self.assertIn("* SPEAKER: p-bob", edl)

    def test_otio_places_each_segment_at_its_frame(self):
        import json
        otio = json.loads(nle_interchange.export_otio_json(DOC, fps=25))
        text = json.dumps(otio)
        self.assertIn('"value": 29', text)   # Bob's in
        self.assertIn('"value": 61', text)   # Bob's duration 90 - 29


if __name__ == "__main__":
    unittest.main()
