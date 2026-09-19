"""Regression checks for the React selected-target/source mapping contract.

The mapping is intentionally checked at the source level here because FaceSwap.jsx
is bundled by Vite and is not a directly importable Python module. These assertions
protect the two failure modes that produced random or no swaps: the highlighted
target person was ignored, and an implicit person rank addressed a nonexistent
source faceset.
"""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
FACE_SWAP = ROOT / "react-ui" / "src" / "components" / "FaceSwap.jsx"
PERSON_GROUPS = ROOT / "react-ui" / "src" / "components" / "PersonGroups.jsx"


class SelectedTargetMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face_swap = FACE_SWAP.read_text(encoding="utf-8")
        cls.person_groups = PERSON_GROUPS.read_text(encoding="utf-8")

    def test_selected_mode_uses_highlighted_target_person(self):
        self.assertIn("p.face_detection_mode === 'Selected face'", self.face_swap)
        self.assertIn("pId !== selectedPerson", self.face_swap)
        self.assertIn("return Math.min(selSource, sourceFaces.length - 1);", self.face_swap)

    def test_unselected_people_are_explicit_skips(self):
        self.assertIn("return -1;", self.face_swap)
        self.assertIn("faceSelection={p.face_detection_mode}", self.face_swap)
        self.assertIn("selectedSource={selSource}", self.face_swap)

    def test_explicit_mapping_still_wins(self):
        self.assertIn("if (mappedSrc !== undefined)", self.face_swap)
        self.assertIn("explicit === -1 || (explicit >= 0 && explicit < sourceFaces.length)", self.face_swap)
        self.assertIn("faceSelection === 'Selected face'", self.person_groups)
        self.assertIn("rank === selRank ? selectedSource : -1", self.person_groups)

    def test_implicit_mapping_never_sends_out_of_range_source(self):
        self.assertIn("pId >= 0 && pId < sourceFaces.length ? pId : -1", self.face_swap)


if __name__ == "__main__":
    unittest.main()
