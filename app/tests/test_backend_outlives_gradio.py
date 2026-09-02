"""A Gradio failure must not take the FastAPI backend down with it.

WHAT HAPPENED.  A second launcher instance collided on the Gradio port -- both
React launchers derive it as ROOP_API_PORT + 2 -- and the React UI went to
ECONNREFUSED on every poll against a backend that had started perfectly:

    [Backend] listening on http://127.0.0.1:42003     <- API up and healthy
    * Running on local URL:  http://127.0.0.1:42005
    Exception When localhost is not accessible, a shareable link must be
      created. ... when launching Gradio Server!
    Closing server running on port: 42005
    (env) (base) G:\pinokio\api\roop-ultimate\app>   <- process exited

THE TRAP, and the reason this file asserts what it asserts.  `run.py` starts the
API on a DAEMON thread and then calls `core.run()` on the main thread, so the API
lives exactly as long as `core.run()` does.  And `ui/main.py:543` CATCHES the
Gradio exception -- it prints, sets `run_server = False`, closes the UI and
RETURNS NORMALLY.  So:

  * the process does not crash, it exits cleanly with status 0;
  * nothing is raised, so a `try/except` around `core.run()` never fires;
  * the API's own health was never in question -- it had already bound.

The condition to defend against is therefore "core.run() RETURNED", not
"core.run() raised". These tests drive that exact shape.
"""

import os
import re
import unittest

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_APP)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class BackendOutlivesGradioTests(unittest.TestCase):
    def setUp(self):
        self.run_py = _read(os.path.join(_APP, "run.py"))

    def test_run_py_keeps_serving_after_core_run_returns(self):
        """The guard must key on the RETURN, not on an exception."""
        tail = self.run_py[self.run_py.index("core.run()"):]
        self.assertIn("ROOP_REACT_CLIENT", tail,
                      "run.py must decide whether to outlive Gradio")
        self.assertIn("api_thread", tail,
                      "run.py must hold the process open on the API thread")
        self.assertRegex(tail, r"api_thread\.join",
                         "run.py must block on the API thread after core.run()")

    def test_the_api_thread_is_reachable_after_core_run(self):
        """A bare `threading.Thread(...).start()` cannot be joined later."""
        self.assertRegex(
            self.run_py,
            r"api_thread\s*=\s*threading\.Thread\(\s*target=run_api,\s*daemon=True\s*\)",
            "the API thread must be bound to a name so it can be joined")

    def test_gradio_still_owns_the_process_for_the_legacy_launcher(self):
        """Where Gradio IS the app, its shutdown must still end the process."""
        legacy = _read(os.path.join(_ROOT, "start_legacy.js"))
        self.assertNotIn("ROOP_REACT_CLIENT", legacy)

    def test_both_react_launchers_declare_themselves(self):
        for name in ("start_react.js", "start_react_v2.js"):
            text = _read(os.path.join(_ROOT, name))
            self.assertIn('ROOP_REACT_CLIENT: "1"', text, name)

    def test_ui_main_swallows_the_gradio_error(self):
        """The premise: if ui/main.py ever re-raises, run.py needs a try too.

        This is the assumption the whole design rests on, so it is asserted
        rather than remembered.
        """
        main = _read(os.path.join(_APP, "ui", "main.py"))
        launch = main[main.index("ui.queue().launch("):]
        handler = launch[:400]
        self.assertRegex(handler, r"except Exception as \w+:",
                         "ui/main.py is expected to CATCH the launch failure")
        self.assertIn("when launching Gradio Server!", handler)
        self.assertNotIn("raise", handler.split("except Exception")[1][:200])


if __name__ == "__main__":
    unittest.main()
