"""Discoverable wrapper for the real-frame Selected-Face harness.

The heavy part -- real detector + recogniser + swapper on the composed frame --
is gated behind ``ROOP_RUN_SELECTED_INTEGRATION=1`` so the default unittest run
stays fast and GPU-free. The runnable harness is the primary deliverable:

    env/Scripts/python.exe tests/integration_selected_face_regression.py
    env/Scripts/python.exe tests/integration_selected_face_regression.py --providers cpu,cuda,tensorrt

This file keeps two always-on, model-free checks (the harness imports and the
composed frame is well-formed) and one opt-in end-to-end check that runs the
full single-provider assertion set and requires it to pass.
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, _HERE)                       # for the sibling harness import

import integration_selected_face_regression as H  # noqa: E402


class HarnessIsWellFormed(unittest.TestCase):
    def test_canvas_has_room_for_three_faces(self):
        # No model load: just the composed frame's geometry. The portraits are
        # real local facesets named by ROOP_TEST_PERSON_A/_B/_BYSTANDER; without
        # them this is a SKIP, never a pass.
        try:
            canvas = H.build_canvas()
        except SystemExit as exc:
            self.skipTest(f"{exc} (set ROOP_TEST_PERSON_A/_B/_BYSTANDER)")
        self.assertEqual(canvas.ndim, 3)
        self.assertEqual(canvas.shape[2], 3)
        # three cells wide -> clearly wider than tall
        self.assertGreater(canvas.shape[1], canvas.shape[0] * 2)

    def test_scenario_uses_four_distinct_identities(self):
        names = {H.SOURCE_FACESET, H.PERSON_A_FACESET,
                 H.PERSON_B_FACESET, H.BYSTANDER_FACESET}
        self.assertEqual(len(names), 4,
                         "source and the three on-frame people must be distinct, "
                         "or a correct swap could not be told from a broken one")


@unittest.skipUnless(
    os.environ.get("ROOP_RUN_SELECTED_INTEGRATION") == "1",
    "real-frame Selected-Face integration is opt-in: set "
    "ROOP_RUN_SELECTED_INTEGRATION=1 (needs the facesets and a working "
    "detector/swapper), or run tests/integration_selected_face_regression.py")
class RealFrameSelectedFace(unittest.TestCase):
    def test_single_provider_assertions_all_pass(self):
        from roop.backend_manager import canonical_provider_decision
        prov = ("cuda" if "cuda" in canonical_provider_decision("cuda").active.lower()
                else "cpu")
        out = os.path.join(H.APP, "output", "selected_face_integration_test")
        ok, lines, _ = H.check_single_provider(prov, out)
        self.assertTrue(ok, "\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
