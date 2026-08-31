"""Phase 8 unit tests: target expression state, asymmetry and integration hooks."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.temporal_expression import (  # noqa: E402
    LEFT_BROW, RIGHT_BROW, TemporalExpressionEngine, measure_expression,
)


def expression_landmarks(left=0.30, right=0.30, mouth=0.04,
                         brow_shift=0.0, jaw_shift=0.0):
    points = np.zeros((106, 2), dtype=np.float32)
    points[:, 0] = np.linspace(80.0, 240.0, 106)
    points[:, 1] = 150.0

    def eye(indices, cx, aperture):
        w = 32.0
        h = float(aperture) * w
        values = ((cx - w / 2.0, 120.0), (cx - w * .2, 120.0 - h / 2.0),
                  (cx + w * .2, 120.0 - h / 2.0), (cx + w / 2.0, 120.0),
                  (cx + w * .2, 120.0 + h / 2.0), (cx - w * .2, 120.0 + h / 2.0))
        points[list(indices)] = np.asarray(values, dtype=np.float32)

    eye((33, 35, 36, 37, 39, 42), 130.0, left)
    eye((87, 89, 90, 91, 93, 96), 190.0, right)
    points[53] = (130.0, 170.0)
    points[59] = (190.0, 170.0)
    points[52] = (160.0, 170.0 - mouth * 30.0)
    points[61] = (160.0, 170.0 + mouth * 30.0)
    points[list(LEFT_BROW), 1] = 94.0 - brow_shift
    points[list(RIGHT_BROW), 1] = 94.0 - brow_shift
    points[:33, 1] += jaw_shift
    return points


class TemporalExpressionTests(unittest.TestCase):
    def test_measurement_reports_requested_target_channels(self):
        measured = measure_expression(
            expression_landmarks(left=.28, right=.14, mouth=.40),
            kps=np.asarray([[130, 120], [190, 120], [160, 145], [145, 170], [175, 170]], dtype=np.float32),
            bbox=[80, 70, 240, 230], detection_confidence=.9)
        self.assertAlmostEqual(measured["left_eye_openness"], .28, places=4)
        self.assertAlmostEqual(measured["right_eye_openness"], .14, places=4)
        self.assertAlmostEqual(measured["mouth_aspect_ratio"], .40, places=4)
        self.assertGreater(measured["confidence"], .5)

    def test_slow_blink_has_continuous_states(self):
        engine = TemporalExpressionEngine(enabled=True)
        sequence = [.30, .25, .12, .05, .12, .25, .30]
        states = []
        for index, value in enumerate(sequence):
            state = engine.update(4, index, expression_landmarks(left=value, right=value),
                                  confidence=1.0)
            states.append(state.blink_state)
        self.assertIn("closing", states)
        self.assertIn("closed", states)
        self.assertIn("opening", states)
        self.assertEqual(states[0], "open")
        self.assertNotIn("unknown", states)

    def test_fast_blink_does_not_randomly_reopen(self):
        engine = TemporalExpressionEngine(enabled=True)
        values = [.30, .04, .30]
        states = [engine.update(5, i, expression_landmarks(left=v, right=v),
                                confidence=1.0).blink_state
                  for i, v in enumerate(values)]
        self.assertEqual(states[0], "open")
        self.assertIn(states[1], ("closing", "closed"))
        self.assertIn(states[2], ("opening", "open"))
        self.assertNotEqual(states, ["open", "closed", "open"])

    def test_asymmetric_blink_and_wink_keep_independent_strengths(self):
        engine = TemporalExpressionEngine(enabled=True)
        engine.update(6, 0, expression_landmarks(), confidence=1.0)
        state = engine.update(6, 1, expression_landmarks(left=.04, right=.30),
                              confidence=1.0)
        plan = engine.plan(6)
        self.assertIn(state.blink_state, ("wink_left", "closing"))
        self.assertGreater(plan["eye_strengths"][0], plan["eye_strengths"][1])

    def test_mouth_and_jaw_events_are_target_driven(self):
        engine = TemporalExpressionEngine(enabled=True)
        engine.update(7, 0, expression_landmarks(mouth=.03, jaw_shift=0), confidence=1.0)
        state = engine.update(7, 1, expression_landmarks(mouth=.45, jaw_shift=12,
                                                          brow_shift=8), confidence=1.0)
        plan = engine.plan(7)
        self.assertGreater(state.mouth_aspect_ratio, .03)
        self.assertNotEqual(state.jaw_movement, 0.0)
        self.assertNotEqual(state.eyebrow_movement, 0.0)
        self.assertGreater(plan["mouth_strength"], 0.0)

    def test_low_confidence_smooths_noise_but_does_not_drop_channels(self):
        engine = TemporalExpressionEngine(enabled=True)
        first = engine.update(8, 0, expression_landmarks(mouth=.04), confidence=1.0)
        second = engine.update(8, 1, expression_landmarks(mouth=.10), confidence=.1)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertGreater(second.expression_confidence, 0.0)
        self.assertLess(second.mouth_openness, .10)

    def test_disabled_engine_is_a_noop(self):
        engine = TemporalExpressionEngine(enabled=False)
        self.assertIsNone(engine.update(1, 0, expression_landmarks()))
        self.assertIsNone(engine.plan(1))

    def test_process_and_masking_hooks_are_regional(self):
        with open(os.path.join(APP, "roop", "ProcessMgr.py"), encoding="utf-8") as handle:
            process_source = handle.read()
        with open(os.path.join(APP, "roop", "procmgr_masking.py"), encoding="utf-8") as handle:
            masking_source = handle.read()
        with open(os.path.join(APP, "roop", "temporal_expression.py"), encoding="utf-8") as handle:
            expression_source = handle.read()
        self.assertIn("ROOP_TEMPORAL_EXPRESSION", expression_source)
        self.assertIn("eye_strengths=_eye_strengths", process_source)
        self.assertIn("current target's eyelids/mouth carry motion", process_source)
        self.assertIn("eye_strengths=None", masking_source)
        self.assertIn("strength:float=1.0", masking_source)


if __name__ == "__main__":
    unittest.main()
