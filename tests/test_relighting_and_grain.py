"""Self-verification for SH relighting and scene-matched sensor grain."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for path in (str(REPO_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from roop.processors.frame import face_swapper


class RelightingAndGrainTest(unittest.TestCase):
    SIZE = 192

    def _face_mask(self):
        mask = np.zeros((self.SIZE, self.SIZE), dtype=np.float32)
        cv2.ellipse(mask, (self.SIZE // 2, self.SIZE // 2), (58, 72), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(mask, (9, 9), 0)

    def test_directional_scene_shading_changes_ambient_scale(self):
        """A high-contrast left key light must brighten the left cheek."""
        scene = np.empty((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        scene[:, :self.SIZE // 2] = (245, 245, 245)
        scene[:, self.SIZE // 2:] = (20, 20, 20)
        skin = self._face_mask()

        coefficients = face_swapper.estimate_scene_sh_coefficients(scene, skin)
        self.assertEqual(coefficients.shape, (9,))
        normals = face_swapper.estimate_face_normals(scene.shape[:2])
        scale = face_swapper.estimate_sh_lighting_scale(coefficients, normals, skin)

        left_cheek, right_cheek = scale[self.SIZE // 2, 58], scale[self.SIZE // 2, 134]
        self.assertGreater(left_cheek, right_cheek + 0.10)
        swap = np.full_like(scene, 128)
        relit = face_swapper.apply_sh_lighting_transfer(swap, scene, skin, skin)
        self.assertGreater(float(relit[self.SIZE // 2, 58].mean()),
                           float(relit[self.SIZE // 2, 134].mean()) + 10.0)

    def test_injected_grain_variance_matches_background_within_ten_percent(self):
        """Synthesized skin noise follows the measured background residual sigma."""
        source_rng = np.random.default_rng(17)
        background = np.full((self.SIZE, self.SIZE, 3), 128.0, dtype=np.float32)
        background += source_rng.normal(0.0, 12.0, size=background.shape)
        background = np.clip(background, 0.0, 255.0).astype(np.uint8)
        skin = self._face_mask()
        _, target_sigma = face_swapper.estimate_background_grain(background, skin)

        restored = np.full_like(background, 128)
        output = face_swapper.inject_film_grain(
            restored, background, np.ones_like(skin), skin,
            rng=np.random.default_rng(99))
        generated_sigma = float(np.std(output.astype(np.float32) - restored.astype(np.float32)))

        self.assertGreater(target_sigma, 1.0)
        self.assertLessEqual(abs(generated_sigma - target_sigma) / target_sigma, 0.10)


if __name__ == '__main__':
    unittest.main()
