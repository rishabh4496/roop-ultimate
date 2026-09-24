"""The React frame transport + player pipeline, run through node.

react-ui/.render-check/player-check.mjs exercises the REAL client modules
(transport/frameProtocol.js, components/player/framePipeline.js,
store/telemetryStore.js) with a fake decoder and renderer: the header offsets
the server writes, latest-wins decoding, one draw per display tick, every
ImageBitmap closed once, and a clear() that a late decode cannot undo.
test_frames_ws.py pins the server side of the same bytes.

Skipped (not failed) where node is not installed, like the other JS suites.
"""
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fixtures  # noqa: E402

REACT_UI = Path(HERE).parent.parent / "react-ui"
CHECKS = REACT_UI / ".render-check"


def _node():
    found = shutil.which("node")
    if found:
        return found
    home = Path(fixtures.pinokio_home() or "")
    for cand in (home / "bin" / "miniconda" / "node.exe",
                 home / "bin" / "miniforge" / "node.exe"):
        if cand.exists():
            return str(cand)
    return None


class FramePlayerJS(unittest.TestCase):
    def _run(self, script):
        node = _node()
        if not node:
            self.skipTest("node not available to run the JS checks")
        if not (REACT_UI / "node_modules").is_dir():
            self.skipTest("react-ui dependencies are not installed")
        path = CHECKS / script
        self.assertTrue(path.exists(), f"missing checker: {path}")
        return subprocess.run([node, str(path)], cwd=str(REACT_UI), capture_output=True,
                              text=True, timeout=300, encoding="utf-8")

    def test_player_pipeline_checks_pass(self):
        proc = self._run("player-check.mjs")
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("ALL GREEN", proc.stdout)

    def test_processing_render_checks_pass_against_the_store(self):
        """The Processing tab's live pieces read the telemetry store; the
        render harness seeds it the way App does and must stay green."""
        proc = self._run("run.mjs")
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("ALL GREEN", proc.stdout)


if __name__ == "__main__":
    unittest.main()
