"""Architecture guard for the FaceSwap comparison-grid boundary."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACESWAP = ROOT / "react-ui" / "src" / "components" / "FaceSwap.jsx"
GRID_PANEL = ROOT / "react-ui" / "src" / "components" / "faceswap" / "ComparisonGridPanel.jsx"


class FaceSwapComponentBoundaryTest(unittest.TestCase):
    def test_comparison_chrome_is_not_reimplemented_in_faceswap(self):
        face_swap = FACESWAP.read_text(encoding="utf-8")
        panel = GRID_PANEL.read_text(encoding="utf-8")
        self.assertIn("ComparisonGridPanel", face_swap)
        self.assertIn("const toggleItem", panel)
        self.assertIn("<CompareGrid", panel)
        self.assertNotIn("comparingEnhancers ? (() => {", face_swap)
        self.assertNotIn("comparingMasks ? (() => {", face_swap)
        self.assertNotIn("comparingSwappers ? (() => {", face_swap)
        self.assertNotIn("comparingUpscalers ? (() => {", face_swap)

    def test_the_model_catalog_remains_owned_by_faceswap(self):
        """The extraction must not duplicate the catalog in four grid branches."""
        face_swap = FACESWAP.read_text(encoding="utf-8")
        self.assertEqual(1, face_swap.count("const AI_UPSCALE_MODELS = ["))
        self.assertIn("availableItems={AI_UPSCALE_MODELS.map", face_swap)


if __name__ == "__main__":
    unittest.main()
