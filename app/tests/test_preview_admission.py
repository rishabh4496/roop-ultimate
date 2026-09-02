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


if __name__ == "__main__":
    unittest.main()
