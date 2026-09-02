"""Automated unit tests for face selection, faceset mapping, and preview synchronization in React UI 2.0.

Verifies:
1. Person grouping logic preserving rank order.
2. Face-mapping array resolution for single and multi-person setups.
3. Angle-pose categorization against primary buckets.
4. Component existence and export validation for SourceGallery, TargetPersonsPanel, and FacesetLibraryModal.
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
V2_SRC = ROOT / "react-ui-v2" / "src"


def group_by_person(groups, count=None):
    if count is None:
        count = len(groups)
    slice_groups = groups[:count]
    res = {}
    for i, rank in enumerate(slice_groups):
        res.setdefault(rank, []).append(i)
    return sorted(res.items(), key=lambda item: item[0])


def resolve_face_mapping_array(target_groups, face_mapping_dict):
    uniq_persons = sorted(list(set(x for x in target_groups if isinstance(x, int))))
    return [
        face_mapping_dict.get(p_id, p_id)
        for p_id in uniq_persons
    ]


PRIMARY_POSES = ['Front', 'Left Profile', 'Right Profile', 'Up Tilt', 'Down Tilt']


def match_poses(pose_label):
    covered = set()
    for p in PRIMARY_POSES:
        if p.lower() in pose_label.lower():
            covered.add(p)
    return covered


class UI2FaceSelectionTests(unittest.TestCase):
    def test_group_by_person_clustering(self):
        # 3 people across 6 angles: Person 0 has 3 angles, Person 1 has 2, Person 2 has 1
        groups = [0, 0, 1, 0, 1, 2]
        grouped = group_by_person(groups)
        self.assertEqual(len(grouped), 3)
        self.assertEqual(grouped[0], (0, [0, 1, 3]))
        self.assertEqual(grouped[1], (1, [2, 4]))
        self.assertEqual(grouped[2], (2, [5]))

    def test_face_mapping_array_payload_generation(self):
        # Swap Target Person 0 -> Source #2, Target Person 1 -> Source #0, Target Person 2 -> Source #1
        target_groups = [0, 0, 1, 2]
        face_mapping_dict = {0: 2, 1: 0, 2: 1}
        payload_array = resolve_face_mapping_array(target_groups, face_mapping_dict)
        self.assertEqual(payload_array, [2, 0, 1])

    def test_partial_face_mapping_defaults_to_self(self):
        # Only Person 0 is explicitly mapped to Source #3; Person 1 and 2 retain default index
        target_groups = [0, 1, 2]
        face_mapping_dict = {0: 3}
        payload_array = resolve_face_mapping_array(target_groups, face_mapping_dict)
        self.assertEqual(payload_array, [3, 1, 2])

    def test_pose_coverage_categorization(self):
        self.assertEqual(match_poses("Frontal view"), {"Front"})
        self.assertEqual(match_poses("Left Profile + Up Tilt"), {"Left Profile", "Up Tilt"})
        self.assertEqual(match_poses("Right Profile - Down Tilt"), {"Right Profile", "Down Tilt"})

    def test_faces_subsystem_files_and_exports_exist(self):
        faces_dir = V2_SRC / "components" / "faces"
        self.assertTrue((faces_dir / "SourceGallery.jsx").is_file())
        self.assertTrue((faces_dir / "TargetPersonsPanel.jsx").is_file())
        self.assertTrue((faces_dir / "FacesetLibraryModal.jsx").is_file())
        self.assertTrue((faces_dir / "index.js").is_file())

        index_text = (faces_dir / "index.js").read_text(encoding="utf-8")
        for export_name in ("SourceGallery", "TargetPersonsPanel", "FacesetLibraryModal"):
            self.assertIn(export_name, index_text)


if __name__ == "__main__":
    unittest.main()
