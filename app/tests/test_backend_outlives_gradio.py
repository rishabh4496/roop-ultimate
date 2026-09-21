"""A Gradio failure must not take the FastAPI backend down with it.

WHAT HAPPENED.  A second launcher instance collided on the Gradio port -- both
React launchers derive it as ROOP_API_PORT + 2 -- and the React UI went to
ECONNREFUSED on every poll against a backend that had started perfectly:

    [Backend] listening on http://127.0.0.1:42003     <- API up and healthy
    * Running on local URL:  http://127.0.0.1:42005
    Exception When localhost is not accessible, a shareable link must be
      created. ... when launching Gradio Server!
    Closing server running on port: 42005
    (env) (base) <PINOKIO_HOME>/api/roop-ultimate\app>   <- process exited

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

    def test_react_url_is_published_only_after_backend_readiness(self):
        """Pinokio must not open the webview during the CFG/API startup race."""
        helper_start = self.run_py.index("def _announce_react_backend_when_ready")
        helper_end = self.run_py.index("\n\nif __name__ == '__main__':", helper_start)
        helper = self.run_py[helper_start:helper_end]
        # The readiness gate moved into roop.startup_state_machine: run.py's
        # helper delegates to execute_api_ready (CFG published + loopback
        # socket accepts + API thread alive) and only then execute_ui_ready
        # emits the URL Pinokio captures. Assert the gate where it lives.
        self.assertIn('execute_api_ready', helper)
        self.assertIn('execute_ui_ready', helper)
        self.assertLess(helper.index('execute_api_ready'), helper.index('execute_ui_ready'))
        sm = _read(os.path.join(_APP, "roop", "startup_state_machine.py"))
        gate = sm[sm.index("def execute_api_ready"):sm.index("def execute_ui_ready")]
        self.assertIn('getattr(roop_globals, "CFG", None)', gate)
        self.assertIn('socket.create_connection', gate)
        self.assertIn('api_thread.is_alive()', gate)

        main = self.run_py[self.run_py.index("api_thread = threading.Thread"):]
        self.assertIn('_announce_react_backend_when_ready', main)
        pre_core = main[:main.index('core.run()')]
        self.assertNotIn('http://', pre_core,
                         "run.py must not emit http:// before core.run() publishes CFG")
        tail = main[main.index('core.run()'):]
        self.assertNotIn('http://', tail[:tail.index('while api_thread.is_alive()')],
                         "run.py post-core.run() block must not emit http:// before socket readiness")

    def test_run_py_accepts_ui_and_react_flags(self):
        """CLI arguments --ui and --react must configure the active client mode."""
        self.assertIn("'--ui'", self.run_py)
        self.assertIn("'--react'", self.run_py)
        main = self.run_py[self.run_py.index("args = parser.parse_args()"):self.run_py.index("def _run_cli_benchmark")]
        self.assertIn("ROOP_REACT_CLIENT", main)

    def test_gradio_still_owns_the_process_for_the_legacy_launcher(self):
        """Where Gradio IS the app, its shutdown must still end the process."""
        legacy = _read(os.path.join(_ROOT, "start_legacy.js"))
        self.assertNotIn("ROOP_REACT_CLIENT", legacy)

    def test_the_react_launcher_declares_itself(self):
        text = _read(os.path.join(_ROOT, "start_react.js"))
        self.assertIn('ROOP_REACT_CLIENT: "1"', text)

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

    def test_ui_main_skips_gradio_when_react_client_active(self):
        """When the React client is active, Gradio UI must not launch or emit a URL."""
        main = _read(os.path.join(_APP, "ui", "main.py"))
        self.assertIn('os.environ.get("ROOP_REACT_CLIENT") == "1"', main)
        react_check = main[main.index('ROOP_REACT_CLIENT'):]
        self.assertIn("return", react_check[:200],
                      "ui/main.py must return early before building Gradio blocks")
        self.assertLess(main.index('ROOP_REACT_CLIENT'), main.index('gr.Blocks'),
                        "Gradio blocks must not be constructed in React mode")

    def test_run_py_legacy_branch_does_not_emit_http_url(self):
        """Legacy branch must not print http:// so start_legacy.js captures Gradio."""
        main = self.run_py[self.run_py.index("api_thread = threading.Thread"):]
        else_block = main[main.index('ROOP_REACT_CLIENT'):main.index('core.run()')]
        self.assertNotIn('http://', else_block,
                         "run.py legacy branch must not emit http:// before Gradio launches")


if __name__ == "__main__":
    unittest.main()
