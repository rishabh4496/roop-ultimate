"""Regression tests for three bugs found in the extended post-fix audit.

BUG 1 – Deferred preview flag lost during scrub / playback
  The useEffect that fires preview after a run finished cleared
  previewDeferredRef.current even when the user was mid-scrub or mid-play,
  discarding the "needs refresh after run" intent permanently.  The flag is
  now kept alive until scrubbing and playback end.

BUG 2 – Unsupported source file types silently discarded
  /api/source/add accepted any file type but only processed images and .fsz
  facesets.  Videos, GIFs, and other formats were saved to disk and then
  silently ignored, leaving the user with a "no face detected" toast when the
  real error was a wrong file type.  The endpoint now returns an "unsupported"
  list.

BUG 3 – Frontend showed "no face detected" for unsupported source types
  onAddSource displayed the "No face detected" error message whenever
  res.source_faces gained nothing — including the silent-discard case above.
  Now it inspects res.unsupported first and shows a clear type-mismatch
  message.

BUG 4 – Preview lock: is_preview flag set outside the try block
  _preview_request_lock was acquired, then roop_globals.is_preview was set
  True BEFORE the try block, so an exception on that assignment (however
  unlikely) would have released neither the lock nor the flag.  Both are now
  inside the try block that owns the finally.
"""

import os
import re
import sys
import unittest

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_APP)
_SRC = os.path.join(_ROOT, "react-ui", "src")
_FACESWAP = os.path.join(_SRC, "components", "FaceSwap.jsx")
_API_PY = os.path.join(_APP, "api.py")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1: Deferred preview flag survives scrub / play
# ─────────────────────────────────────────────────────────────────────────────

class DeferredPreviewFlagTests(unittest.TestCase):
    """The deferred-preview flag must not be cleared while scrubbing or playing."""

    def setUp(self):
        self._src = _read(_FACESWAP)

    def test_isScrubbing_guard_before_flag_clear(self):
        """Effect must early-return when isScrubbing is true."""
        self.assertRegex(
            self._src,
            r'if\s*\(\s*isScrubbing\s*\|\|\s*isPlaying\s*\)\s*return',
            "The deferred-preview effect must guard against scrubbing/playing "
            "BEFORE it clears previewDeferredRef so the flag survives.",
        )

    def test_isScrubbing_in_dependency_array(self):
        """isScrubbing must be in the dependency array so the effect re-fires."""
        # Locate the effect that clears previewDeferredRef
        chunk = self._src.split("previewDeferredRef.current = false", 1)
        self.assertEqual(len(chunk), 2,
                         "previewDeferredRef.current = false not found in FaceSwap.jsx")
        # The dependency array follows the effect body
        tail = chunk[1].split("}, [", 1)
        self.assertEqual(len(tail), 2,
                         "Could not find the dependency array after the deferred-preview effect")
        dep_section = tail[1].split("]", 1)[0]
        self.assertIn(
            "isScrubbing", dep_section,
            "isScrubbing must be listed in the deferred-preview effect's "
            "dependency array so it re-fires when scrubbing ends",
        )
        self.assertIn(
            "isPlaying", dep_section,
            "isPlaying must be listed in the deferred-preview effect's "
            "dependency array so it re-fires when playback ends",
        )

    def test_targets_check_after_not_before_scrub_guard(self):
        """targets.length === 0 guard must come AFTER the scrub/play guard.

        Both patterns appear in multiple effects; we must search within the
        specific deferred-preview effect that owns previewDeferredRef.current.
        """
        src = self._src
        # Anchor on the flag clear — unique to the deferred-preview effect
        flag_clear_pos = src.find("previewDeferredRef.current = false")
        self.assertGreater(flag_clear_pos, 0,
                           "previewDeferredRef.current = false not found")
        # The effect body runs from just before the flag clear to the closing
        # dependency array.  Search only inside this window.
        effect_window = src[flag_clear_pos - 500: flag_clear_pos + 500]

        scrub_rel = effect_window.find("if (isScrubbing || isPlaying) return")
        targets_rel = effect_window.find("if (targets.length === 0) return")
        self.assertGreater(
            scrub_rel, 0,
            "Scrub guard not found inside the deferred-preview effect",
        )
        self.assertGreater(
            targets_rel, 0,
            "targets.length guard not found inside the deferred-preview effect",
        )
        self.assertGreater(
            targets_rel, scrub_rel,
            "targets.length === 0 check must come AFTER the scrub/play guard "
            "so the flag is preserved while the user scrubs",
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG 2: Backend tracks unsupported source file types
# ─────────────────────────────────────────────────────────────────────────────

class UnsupportedSourceTypeBackendTests(unittest.TestCase):
    """source_add must collect non-image/non-fsz names and return them."""

    def setUp(self):
        self._src = _read(_API_PY)

    def test_skipped_names_list_initialised(self):
        """skipped_names list must be initialised before the upload loop."""
        func = self._src.split("def source_add(", 1)[1].split("\n\n\n", 1)[0]
        self.assertIn(
            "skipped_names = []", func,
            "source_add must initialise skipped_names before iterating files",
        )

    def test_unsupported_branch_appends_basename(self):
        """The else branch for unrecognised file types must append the basename."""
        func = self._src.split("def source_add(", 1)[1].split("\n\n\n", 1)[0]
        self.assertRegex(
            func,
            r'skipped_names\.append\(os\.path\.basename\(path\)\)',
            "source_add must append os.path.basename(path) to skipped_names "
            "for files that are neither images nor .fsz facesets",
        )

    def test_unsupported_key_added_to_payload(self):
        """Payload must carry 'unsupported' when skipped_names is non-empty."""
        func = self._src.split("def source_add(", 1)[1].split("\n\n\n", 1)[0]
        self.assertIn(
            'payload["unsupported"] = skipped_names',
            func,
            "source_add must set payload['unsupported'] to the list of "
            "skipped filenames so the frontend can display a useful message",
        )

    def test_unsupported_key_conditional(self):
        """The 'unsupported' key must only be added when the list is non-empty."""
        func = self._src.split("def source_add(", 1)[1].split("\n\n\n", 1)[0]
        # The guard must appear before the assignment
        guard_pos = func.find("if skipped_names:")
        assign_pos = func.find('payload["unsupported"] = skipped_names')
        self.assertGreater(
            guard_pos, 0,
            "source_add must guard the unsupported assignment with 'if skipped_names:'",
        )
        self.assertGreater(
            assign_pos, guard_pos,
            "The 'unsupported' assignment must be inside the 'if skipped_names:' block",
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG 3: Frontend shows correct error for unsupported source types
# ─────────────────────────────────────────────────────────────────────────────

class UnsupportedSourceTypeFrontendTests(unittest.TestCase):
    """onAddSource must check res.unsupported before showing 'no face detected'."""

    def setUp(self):
        self._src = _read(_FACESWAP)

    def test_unsupported_check_present(self):
        """onAddSource must inspect res.unsupported?.length."""
        func = self._src.split("const onAddSource = ", 1)[1].split("\n  };", 1)[0]
        self.assertIn(
            "res.unsupported?.length",
            func,
            "onAddSource must check res.unsupported?.length so videos/GIFs "
            "get a type-mismatch error instead of 'No face detected'",
        )

    def test_unsupported_shows_error_toast(self):
        """The unsupported branch must call notify with 'error' severity."""
        func = self._src.split("const onAddSource = ", 1)[1].split("\n  };", 1)[0]
        # The branch: if (res.unsupported?.length) { notify(..., 'error') }
        self.assertRegex(
            func,
            r"res\.unsupported\?\.length[\s\S]{0,200}notify\(",
            "onAddSource must call notify() inside the unsupported branch",
        )
        # The severity must be 'error', not 'success' or omitted
        unsupported_section = func.split("res.unsupported?.length", 1)[1].split("else if", 1)[0]
        self.assertIn(
            "'error'",
            unsupported_section,
            "The unsupported notify call must use 'error' severity",
        )

    def test_unsupported_message_mentions_accepted_types(self):
        """The error message must name the accepted formats."""
        func = self._src.split("const onAddSource = ", 1)[1].split("\n  };", 1)[0]
        unsupported_section = func.split("res.unsupported?.length", 1)[1].split("else if", 1)[0]
        self.assertRegex(
            unsupported_section,
            r'\.png|\.jpg|\.jpeg|\.webp|\.fsz',
            "The unsupported-type error must list the accepted extensions "
            "so the user knows what to use instead",
        )

    def test_no_face_detected_not_shown_for_unsupported(self):
        """'No face detected' must only fire when res.unsupported is absent."""
        func = self._src.split("const onAddSource = ", 1)[1].split("\n  };", 1)[0]
        # 'No face detected' must be in an else / else if branch
        self.assertRegex(
            func,
            r'else\s+(if\s*\(.*\)\s*)?\bnotify\([^\)]*No face detected',
            "The 'No face detected' notify must be in an else branch that is "
            "skipped when res.unsupported is set",
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG 4: Preview lock safety – is_preview inside the try block
# ─────────────────────────────────────────────────────────────────────────────

class PreviewLockSafetyTests(unittest.TestCase):
    """roop_globals.is_preview must be set INSIDE the try block that owns the lock."""

    def setUp(self):
        self._src = _read(_API_PY)

    def test_is_preview_inside_try(self):
        """is_preview = True must appear inside the try: block, not before it."""
        # Locate the preview endpoint's lock/try structure
        preview_fn = self._src.split("def preview(payload: dict = Body(...))", 1)[1]
        # Trim to the function body (up to the next top-level @app decorator)
        preview_fn = preview_fn.split("\n@app.", 1)[0]

        acquire_pos = preview_fn.find("_preview_request_lock.acquire()")
        try_pos = preview_fn.find("try:", acquire_pos)
        is_preview_pos = preview_fn.find("roop_globals.is_preview = True", acquire_pos)

        self.assertGreater(acquire_pos, 0, "_preview_request_lock.acquire() not found")
        self.assertGreater(try_pos, acquire_pos, "try: not found after lock.acquire()")
        self.assertGreater(
            is_preview_pos, try_pos,
            "roop_globals.is_preview = True must be INSIDE the try block "
            "(after 'try:') so an exception cannot leak a held lock",
        )

    def test_is_preview_reset_in_finally(self):
        """is_preview must be reset to False in the finally block."""
        preview_fn = self._src.split("def preview(payload: dict = Body(...))", 1)[1]
        preview_fn = preview_fn.split("\n@app.", 1)[0]
        finally_section = preview_fn.split("finally:", 1)
        self.assertEqual(len(finally_section), 2, "No finally: block in preview()")
        self.assertIn(
            "roop_globals.is_preview = False",
            finally_section[1],
            "preview() finally block must reset is_preview to False",
        )

    def test_lock_released_in_finally(self):
        """Lock must be released in the finally block."""
        preview_fn = self._src.split("def preview(payload: dict = Body(...))", 1)[1]
        preview_fn = preview_fn.split("\n@app.", 1)[0]
        finally_section = preview_fn.split("finally:", 1)
        self.assertEqual(len(finally_section), 2, "No finally: block in preview()")
        self.assertIn(
            "_preview_request_lock.release()",
            finally_section[1],
            "preview() finally block must release _preview_request_lock",
        )


if __name__ == "__main__":
    unittest.main()
