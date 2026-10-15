import unittest
import numpy as np
from roop.orientation import action_for_roll, resolve_track_rolls
from roop.FaceSet import FaceSet
from roop.ProcessMgr import ProcessMgr


class AngleFlickerFixTest(unittest.TestCase):
    def test_action_for_roll_hysteresis_lower_bound(self):
        # Without prev_action, -40 deg is None, -50 deg is rotate_anticlockwise
        self.assertIsNone(action_for_roll(-40.0, prev_action=None))
        self.assertEqual(action_for_roll(-50.0, prev_action=None), "rotate_anticlockwise")

        # With prev_action='rotate_anticlockwise', -40 deg stays rotate_anticlockwise (deadband 36..54)
        self.assertEqual(action_for_roll(-40.0, prev_action="rotate_anticlockwise"), "rotate_anticlockwise")
        # Until it clears -36 deg towards 0
        self.assertIsNone(action_for_roll(-30.0, prev_action="rotate_anticlockwise"))

        # With prev_action=None, -40 is None, -48 is rotate_anticlockwise (standard 45 deg cutoff)
        self.assertIsNone(action_for_roll(-40.0, prev_action=None))
        self.assertEqual(action_for_roll(-48.0, prev_action=None), "rotate_anticlockwise")

    def test_action_for_roll_hysteresis_upper_bound(self):
        # Inverted / upside-down boundary around 135 deg
        # Without prev_action, 130 is rotate_clockwise, 140 is rotate_180
        self.assertEqual(action_for_roll(130.0, prev_action=None), "rotate_clockwise")
        self.assertEqual(action_for_roll(140.0, prev_action=None), "rotate_180")

        # With prev_action='rotate_180', 130 deg stays rotate_180 (deadband 126..144)
        self.assertEqual(action_for_roll(130.0, prev_action="rotate_180"), "rotate_180")
        # Only drops out of rotate_180 if tilt drops below 126 deg
        self.assertEqual(action_for_roll(120.0, prev_action="rotate_180"), "rotate_clockwise")

        # With prev_action='rotate_clockwise', 140 deg stays rotate_clockwise
        self.assertEqual(action_for_roll(140.0, prev_action="rotate_clockwise"), "rotate_clockwise")
        # Transitions to rotate_180 when tilt exceeds 144 deg
        self.assertEqual(action_for_roll(150.0, prev_action="rotate_clockwise"), "rotate_180")

    def test_resolve_track_rolls_eliminates_boundary_chatter(self):
        # Sequence oscillating around 135 deg: [134, 137, 133, 138, 135, 136]
        # In memoryless evaluation this flips 5 times.
        rolls = [134.0, 137.0, 133.0, 138.0, 135.0, 136.0]
        faces = [{"roll_deg_test": r} for r in rolls]

        # Mock roll_from_face to return our sequence
        from unittest.mock import patch
        with patch("roop.orientation.roll_from_face", side_effect=rolls):
            resolve_track_rolls(faces)

        actions = [f["rotation_action"] for f in faces]
        # First action chosen should persist without flipping back and forth every frame
        self.assertEqual(len(set(actions)), 1, f"Expected 0 flips in boundary oscillation, got {actions}")

    def test_faceset_canonical_face_index(self):
        fs = FaceSet()
        fs.faces = [0, 1, 2, 3, 4]
        # Poses matching riya kansal.fsz: Face 0 is profile (82 deg), Face 2 is frontal (2.6 deg)
        fs.face_poses = [(82.3, 31.9), (34.2, 3.8), (2.6, 1.1), (35.3, 4.0), (81.4, 25.9)]
        self.assertEqual(fs.canonical_face_index(), 2)

    def test_faceset_select_reference_index_hysteresis(self):
        fs = FaceSet()
        fs.faces = [0, 1, 2, 3, 4]
        fs.face_poses = [(82.3, 31.9), (34.2, 3.8), (2.6, 1.1), (35.3, 4.0), (81.4, 25.9)]

        # Face 2 is at 2.6 deg. Face 1 is at 34.2 deg. Midpoint is ~18.4 deg.
        # At 20 deg, Face 1 is slightly closer, but if previous was Face 2, hysteresis should keep Face 2.
        sel_without_prev = fs.select_reference_index(pose=(20.0, 2.0), previous_index=None)
        self.assertEqual(sel_without_prev, 1)

        sel_with_prev = fs.select_reference_index(pose=(20.0, 2.0), previous_index=2)
        self.assertEqual(sel_with_prev, 2, "Hysteresis should keep previous index 2 when within margin")

        # When yaw reaches 35 deg, candidate beats previous well beyond margin -> switches to 1
        sel_switched = fs.select_reference_index(pose=(35.0, 2.0), previous_index=2)
        self.assertEqual(sel_switched, 1)

    def test_processmgr_prefers_track_rotation_action(self):
        mgr = ProcessMgr(progress=lambda *args: None)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Face with explicit track-resolved rotation_action
        face_dict = {"rotation_action": "rotate_clockwise", "roll_deg": 0.0}
        self.assertEqual(mgr.rotation_action(face_dict, frame), "rotate_clockwise")


if __name__ == "__main__":
    unittest.main()
