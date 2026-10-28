"""Stage 17: the React model (`mouthCustody.js`) and the backend (`roop/mouth_custody.py`) agree on the same cases.

`react-ui/.render-check/mouth-custody.cases.json` is the shared reference: a legacy recipe must open as the mode it always
meant, and the slip clamps to the same range, whichever side reads it. The JS side of the same file runs under node when
node is on PATH (CI's python-light job has it); the Python side always runs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import mouth_custody as mc  # noqa: E402

CASES = os.path.join(ROOT, "react-ui", ".render-check", "mouth-custody.cases.json")


def _cases():
    with open(CASES, encoding="utf-8") as fh:
        return json.load(fh)


class SharedCasesTest(unittest.TestCase):
    def test_derive_cases(self):
        for case in _cases()["derive"]:
            self.assertEqual(mc.derive_mode(case["params"], lambda k, d: d), case["mode"], case["params"])

    def test_clamp_cases(self):
        for case in _cases()["clamp"]:
            self.assertEqual(mc.clamp_offset_ms(case["in"]), float(case["out"]), case["in"])

    def test_the_cases_are_not_empty(self):
        cases = _cases()
        self.assertGreaterEqual(len(cases["derive"]), 6)
        self.assertGreaterEqual(len(cases["clamp"]), 8)


class NodeSideTest(unittest.TestCase):
    def test_the_react_model_passes_the_same_cases_under_node(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not on PATH")
        done = subprocess.run([node, os.path.join(".render-check", "run-module.mjs"), "render-mouth-custody.mjs"],
                              cwd=os.path.join(ROOT, "react-ui"), capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-1500:] + done.stderr[-500:])
        self.assertIn("all passed", done.stdout)


if __name__ == "__main__":
    unittest.main()
