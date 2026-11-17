"""The still/preview path must actually run its frame operation.

WHY THIS EXISTS.  `pause_aware` gates every decorated frame operation on
`roop.globals.processing`, and `PauseController.begin` REFUSES when that flag
is False.  The flag is owned by the batch run (`core.batch_process` and the
API's swap route); it is False for `core.live_swap`, which serves the
single-image swap and the UI preview button.

The decorator's refusal path returns the INPUT FRAME, so the still path loaded
every model, reported success, and handed back the untouched plate -- 0.00/255
face-region delta and zero identity movement, reproduced on both the RTX 4070
and the RTX 3060.  Nothing raised, no return code changed, and the whole test
suite stayed green, because the defect lived entirely in admission.

These tests assert admission, not pixels: whether the wrapped function is
CALLED.  A pixel assertion needs a GPU, a faceset and a fixture; the bug was
that the call never happened at all, and that is checkable anywhere.
"""

import os
import sys
import unittest

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import roop.globals  # noqa: E402
from roop.procmgr_runtime import pause_aware, pause_controller  # noqa: E402


class _Mgr:
    """Minimal stand-in carrying only what the decorator reads."""

    def __init__(self, is_preview):
        self.is_preview = is_preview
        self.calls = 0

    @pause_aware
    def process_frame(self, frame, frame_idx=None):
        self.calls += 1
        return "SWAPPED"


class _RealShapeMgr:
    """Stand-in with the SAME signature as `ProcessMgr.process_frame`, recording what arrives.

    `_Mgr` above declares only `frame_idx`, so it could not see the wrapper dropping the other two
    parameters: `core.live_swap` calls `process_frame(frame, frame_idx=..., frame_absolute=True,
    source_video=...)`, and a wrapper with a fixed `(frame, frame_idx, output_pending)` signature raised
    `TypeError: ... unexpected keyword argument 'frame_absolute'` on every UI preview and single-image swap
    (found by driving the live UI against the real backend, 2026-11-18).
    """

    def __init__(self, is_preview):
        self.is_preview = is_preview
        self.seen = None

    @pause_aware
    def process_frame(self, frame, frame_idx=None, frame_absolute=False, source_video=None):
        self.seen = {"frame_idx": frame_idx, "frame_absolute": frame_absolute, "source_video": source_video}
        return "SWAPPED"


class WrapperForwardsEveryArgumentTests(unittest.TestCase):
    def setUp(self):
        self._processing = roop.globals.processing
        pause_controller.finish()
        self.addCleanup(self._restore)

    def _restore(self):
        roop.globals.processing = self._processing
        pause_controller.finish()

    def test_preview_call_exactly_as_live_swap_makes_it(self):
        roop.globals.processing = False
        mgr = _RealShapeMgr(is_preview=True)
        out = mgr.process_frame("PLATE", frame_idx=7, frame_absolute=True, source_video="clip.mp4")
        self.assertEqual(out, "SWAPPED")
        self.assertEqual(mgr.seen, {"frame_idx": 7, "frame_absolute": True, "source_video": "clip.mp4"})

    def test_render_call_inside_a_run_forwards_them_too(self):
        roop.globals.processing = True
        mgr = _RealShapeMgr(is_preview=False)
        mgr.process_frame("PLATE", frame_idx=3, frame_absolute=True, source_video="clip.mp4")
        self.assertEqual(mgr.seen, {"frame_idx": 3, "frame_absolute": True, "source_video": "clip.mp4"})

    def test_defaults_are_untouched_when_the_caller_passes_nothing_extra(self):
        roop.globals.processing = True
        mgr = _RealShapeMgr(is_preview=False)
        mgr.process_frame("PLATE", frame_idx=1)
        self.assertEqual(mgr.seen, {"frame_idx": 1, "frame_absolute": False, "source_video": None})

    def test_a_refused_frame_still_returns_the_plate_and_never_calls_the_body(self):
        roop.globals.processing = False
        mgr = _RealShapeMgr(is_preview=False)
        self.assertEqual(mgr.process_frame("PLATE", frame_idx=1, frame_absolute=True), "PLATE")
        self.assertIsNone(mgr.seen)

    def test_core_live_swap_passes_only_names_the_wrapper_forwards(self):
        """Source-level drift guard: every keyword `live_swap` sends must be accepted by the real wrapper."""
        core = os.path.join(_APP, "roop", "core.py")
        with open(core, encoding="utf-8") as handle:
            text = handle.read()
        start = text.find("_preview_process_mgr.process_frame(")
        self.assertGreater(start, 0, "live_swap's process_frame call moved; update this guard")
        call = text[start:text.find(")", start) + 1]
        sent = {kw for kw in ("frame_idx", "frame_absolute", "source_video") if kw + "=" in call}
        mgr = _RealShapeMgr(is_preview=True)
        roop.globals.processing = False
        mgr.process_frame("PLATE", **{kw: (True if kw == "frame_absolute" else 1) for kw in sent})
        self.assertEqual(set(k for k, v in mgr.seen.items() if v not in (None, False)), sent)


class PreviewAdmissionTests(unittest.TestCase):
    def setUp(self):
        self._processing = roop.globals.processing
        pause_controller.finish()
        self.addCleanup(self._restore)

    def _restore(self):
        roop.globals.processing = self._processing
        pause_controller.finish()

    def test_preview_outside_a_run_actually_processes(self):
        """The regression: no run in progress, so nothing to pause or stop."""
        roop.globals.processing = False
        mgr = _Mgr(is_preview=True)
        self.assertEqual(mgr.process_frame("PLATE"), "SWAPPED")
        self.assertEqual(mgr.calls, 1)

    def test_render_worker_outside_a_run_is_still_refused(self):
        """A stopped run must still drop work -- that is what the gate is for."""
        roop.globals.processing = False
        mgr = _Mgr(is_preview=False)
        self.assertEqual(mgr.process_frame("PLATE"), "PLATE")
        self.assertEqual(mgr.calls, 0)

    def test_render_worker_inside_a_run_processes(self):
        roop.globals.processing = True
        mgr = _Mgr(is_preview=False)
        self.assertEqual(mgr.process_frame("PLATE"), "SWAPPED")
        self.assertEqual(mgr.calls, 1)

    def test_preview_during_a_live_run_keeps_run_scoped_admission(self):
        """The bypass must not hand a paused render extra GPU work."""
        roop.globals.processing = True
        mgr = _Mgr(is_preview=True)
        self.assertEqual(mgr.process_frame("PLATE"), "SWAPPED")
        self.assertEqual(mgr.calls, 1)

    def test_live_swap_uses_a_preview_marked_manager(self):
        """The bypass is keyed on `is_preview`; core must actually set it.

        Source-level, because importing ProcessMgr for real pulls the model
        stack in.  If the marker moves, the still path silently regresses to
        returning the plate again.
        """
        core = os.path.join(_APP, "roop", "core.py")
        with open(core, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("_preview_process_mgr.is_preview = True", text)

    def test_live_swap_skips_batch_fast_bypass_during_preview(self):
        """Preview must not run a second detector pass and return raw pixels.

        The dead-code guard (`and not is_preview`) was removed in favour of a
        comment that explicitly documents the intent and the absence of the
        fast-path call.  This test now checks that the bypass is absent rather
        than that a vacuous condition exists.
        """
        core = os.path.join(_APP, "roop", "core.py")
        with open(core, encoding="utf-8") as handle:
            text = handle.read()
        # fast_path_bypass must not be called from within live_swap.
        # Locate the function body between its def and the next top-level def.
        live_swap_start = text.find("def live_swap(")
        self.assertGreater(live_swap_start, 0, "live_swap not found in core.py")
        next_def = text.find("\ndef ", live_swap_start + 1)
        body = text[live_swap_start:next_def] if next_def > 0 else text[live_swap_start:]
        self.assertNotIn(
            "fast_path_bypass",
            body,
            "live_swap must not call fast_path_bypass — preview must go through "
            "ProcessMgr so selected-face fallback and tracking work correctly",
        )

    def test_api_serializes_overlapping_preview_requests(self):
        """Fast control changes must not interleave process-wide preview globals."""
        api = os.path.join(_APP, "api.py")
        with open(api, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("_preview_request_lock = threading.Lock()", text)
        self.assertIn("_preview_request_lock.acquire()", text)
        self.assertIn("_preview_request_lock.release()", text)


if __name__ == "__main__":
    unittest.main()
