"""The Media Canvas role badges / "Exclude (never swap)" decision, run through node.

react-ui/.render-check/face-roles-check.mjs imports the REAL module
(src/components/faceswap/faceRoles.js) and pins:
  * the defect that motivated it: "Ignore / Skip" deleted the person's key, which only
    skips in "Selected face" - in "Selected people" / "All faces" the person kept
    swapping with their rank-default source. An explicit null entry skips in every mode;
  * the badge a box shows always agrees with the payload the job sends.
test_target_exclusion_mapping.py pins the backend half of the same `null`.

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
CHECKER = REACT_UI / ".render-check" / "face-roles-check.mjs"


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


class FaceRolesJS(unittest.TestCase):
    def test_role_and_exclusion_checks_pass(self):
        node = _node()
        if not node:
            self.skipTest("node not available to run the JS checks")
        self.assertTrue(CHECKER.exists(), f"missing checker: {CHECKER}")
        proc = subprocess.run([node, str(CHECKER)], cwd=str(REACT_UI), capture_output=True,
                              text=True, timeout=300, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
        self.assertIn("ALL GREEN", proc.stdout)


if __name__ == "__main__":
    unittest.main()
