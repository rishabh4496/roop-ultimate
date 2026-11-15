"""B-03: a face-selection mode that does not exist must be refused, not guessed as "all".

`normalize_swap_mode` used to be `LABELS.get(label, "all")`. The dict is keyed by UI labels, so the canonical
values the swap loop branches on ("first", "selected", ...) were not keys: an API client sending "first", a typo, or a
renamed label silently swapped EVERY face in the frame.
"""

import os
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop.processing_request import (  # noqa: E402
    SWAP_MODE_LABELS,
    SWAP_MODES,
    UnknownSwapMode,
    normalize_processing_request,
    normalize_swap_mode,
)


class NormalizeSwapModeTests(unittest.TestCase):
    def test_every_ui_label_maps_to_its_canonical_mode(self):
        for label, mode in SWAP_MODE_LABELS.items():
            self.assertEqual(normalize_swap_mode(label), mode, label)

    def test_every_canonical_mode_round_trips(self):
        # This is the case the old fallback got wrong: "first" came back as "all".
        for mode in SWAP_MODES:
            self.assertEqual(normalize_swap_mode(mode), mode, mode)
        self.assertEqual(normalize_swap_mode("first"), "first")
        self.assertEqual(normalize_swap_mode("selected"), "selected")

    def test_the_default_label_is_all(self):
        self.assertEqual(normalize_swap_mode("All faces"), "all")

    def test_absent_means_default(self):
        for value in (None, "", "   "):
            self.assertEqual(normalize_swap_mode(value), "all", repr(value))

    def test_case_and_surrounding_space_do_not_matter(self):
        self.assertEqual(normalize_swap_mode("  first FOUND "), "first")
        self.assertEqual(normalize_swap_mode("SELECTED"), "selected")

    def test_unknown_values_are_refused(self):
        for value in ("Frist found", "everyone", "selected-face", 3, ["first"], {"mode": "first"}):
            with self.assertRaises(UnknownSwapMode, msg=repr(value)):
                normalize_swap_mode(value)

    def test_refusal_is_a_value_error_naming_the_choices(self):
        with self.assertRaises(ValueError) as caught:
            normalize_swap_mode("nobody")
        self.assertIn("'nobody'", str(caught.exception))
        self.assertIn("First found", str(caught.exception))


class RequestBoundaryTests(unittest.TestCase):
    def _request(self, detection=None, **extra):
        payload = dict(extra)
        if detection is not None:
            payload["detection"] = detection
        return normalize_processing_request(payload, source_count=2)

    def test_canonical_mode_survives_the_request(self):
        self.assertEqual(self._request("first")["swap_mode"], "first")

    def test_missing_detection_still_defaults_to_all(self):
        self.assertEqual(self._request()["swap_mode"], "all")

    def test_unknown_detection_is_not_turned_into_all(self):
        with self.assertRaises(UnknownSwapMode):
            self._request("Frist found")


class ApiHandlerTests(unittest.TestCase):
    def test_the_api_turns_the_refusal_into_a_400(self):
        try:
            import api  # noqa: F401
        except Exception as exc:  # the API pulls in the GPU stack; the light profile has none
            self.skipTest("api module not importable here: %s" % type(exc).__name__)
        handlers = api.app.exception_handlers
        self.assertIn(UnknownSwapMode, handlers)


if __name__ == "__main__":
    unittest.main()
