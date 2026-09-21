"""The occlusion admission gate is judged on the face, not the jaw contour.

Real-file finding (<MEDIA_DIR>/target_clip.mp4, frame 1): a fully visible face was
marked `partial` at hidden fraction 0.208 and its swap thrown away; all 22
"hidden" landmarks were 2d106 contour points (0-13, 18-24, 32) under a fur
collar and hair.  The 8% gate is unchanged; the population it reads is the
interior landmarks (33-105) on the 106-point layout.
"""

import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop.tracker import (  # noqa: E402
    CONTOUR_106,
    STATE_PARTIAL,
    STATE_VISIBLE,
    occlusion_gate_population,
    occlusion_hidden_fraction,
    occlusion_state_for,
)


class OcclusionGatePopulation(unittest.TestCase):
    def _visible106(self, hidden):
        visible = np.ones(106, dtype=bool)
        visible[list(hidden)] = False
        return visible

    def test_target_person_frame_1_reading_is_visible(self):
        hidden = list(range(0, 14)) + list(range(18, 25)) + [32]
        visible = self._visible106(hidden)
        self.assertAlmostEqual(1.0 - visible.mean(), 22 / 106, places=6)  # the old reading, 0.208
        self.assertEqual(occlusion_state_for(visible), STATE_VISIBLE)
        self.assertEqual(occlusion_hidden_fraction(visible), 0.0)

    def test_whole_contour_hidden_is_still_visible(self):
        self.assertEqual(occlusion_state_for(self._visible106(range(CONTOUR_106))), STATE_VISIBLE)

    def test_interior_occluder_still_trips_the_gate(self):
        # A hand across the mouth: 8 of the 73 interior landmarks (11%) hidden.
        visible = self._visible106(range(84, 92))
        self.assertEqual(occlusion_state_for(visible), STATE_PARTIAL)
        self.assertGreaterEqual(occlusion_hidden_fraction(visible), 0.08)
        # Five interior points (6.8%) sit under the 8% gate, as before.
        self.assertEqual(occlusion_state_for(self._visible106(range(84, 89))), STATE_VISIBLE)

    def test_other_layouts_are_judged_whole(self):
        for n in (5, 20, 68):
            visible = np.ones(n, dtype=bool)
            visible[:max(1, int(n * 0.1))] = False
            self.assertEqual(occlusion_gate_population(visible).size, n)
            self.assertEqual(occlusion_state_for(visible), STATE_PARTIAL)
        self.assertEqual(occlusion_state_for(np.ones(20, dtype=bool)), STATE_VISIBLE)
        self.assertEqual(occlusion_state_for(None), STATE_VISIBLE)
        self.assertEqual(occlusion_hidden_fraction(None), 0.0)

    def test_coasted_still_wins(self):
        self.assertEqual(occlusion_state_for(np.ones(106, dtype=bool), coasted=True), "coasted")


if __name__ == "__main__":
    unittest.main()
