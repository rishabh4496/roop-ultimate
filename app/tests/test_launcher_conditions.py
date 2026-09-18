"""Launcher `when` expressions must never throw.

Pinokio evaluates a step's `when` as a JavaScript expression. If the expression
raises, the step is skipped -- silently. That is not hypothetical: torch.js
briefly used

    gpu === 'nvidia' || (Array.isArray(gpus) && gpus.includes('nvidia')) || which('nvidia-smi')

and on any machine where `gpu` was not exactly 'nvidia', evaluation continued to
`gpus`, which is not guaranteed to be in scope, and threw a ReferenceError. Every
branch of torch.js was skipped, so the install produced a venv with no torch and
no onnxruntime. `import onnxruntime` then resolved to an implicit NAMESPACE
package -- a module whose `__file__` is None and which has no
`get_available_providers` -- and startup reported:

    onnxruntime is installed but exposes no provider API (loaded from None)

So this test evaluates every `when` in the launcher scripts against the minimal
variable scope, and fails if any of them throws.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCHERS = ("torch.js", "install.js", "update.js", "start_react.js")
WHEN = re.compile(r'"?when"?\s*:\s*"\{\{([\s\S]*?)\}\}"')

# Only what the reference launcher relies on. `gpus` and `which` are
# deliberately absent: a condition that needs them is the bug this pins.
SCOPES = [
    {"platform": "win32", "arch": "x64", "gpu": "nvidia"},
    {"platform": "win32", "arch": "x64", "gpu": "amd"},
    {"platform": "win32", "arch": "x64", "gpu": ""},
    {"platform": "win32", "arch": "x64", "gpu": None},
    {"platform": "linux", "arch": "x64", "gpu": "nvidia"},
    {"platform": "linux", "arch": "x64", "gpu": "amd"},
    {"platform": "linux", "arch": "x64", "gpu": ""},
    {"platform": "darwin", "arch": "arm64", "gpu": ""},
    {"platform": "darwin", "arch": "x64", "gpu": ""},
]

NODE = shutil.which("node")

EVAL_JS = r"""
const vm = require('node:vm');
const payload = JSON.parse(process.argv[1]);
const failures = [];
for (const item of payload.exprs) {
  for (const scope of payload.scopes) {
    const ctx = Object.assign({}, scope, {
      exists: () => false,
      running: () => false,
      path: require('node:path'),
      envs: {},
    });
    try {
      vm.runInNewContext(item.expr, ctx);
    } catch (e) {
      failures.push(`${item.file}: ${e.name}: ${e.message} :: ${item.expr.slice(0, 120)}`);
    }
  }
}
console.log(JSON.stringify(failures));
"""


def _expressions():
    found = []
    for name in LAUNCHERS:
        path = os.path.join(ROOT, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for match in WHEN.finditer(handle.read()):
                found.append({"file": name, "expr": match.group(1)})
    return found


@unittest.skipUnless(NODE, "node is not on PATH")
class LauncherConditionsNeverThrow(unittest.TestCase):
    def test_expressions_were_actually_found(self):
        """A rename of the `when` key would make this file vacuously pass."""
        self.assertTrue(_expressions(),
                        "parsed no `when` expressions out of the launcher scripts")

    def test_no_when_expression_throws_in_a_minimal_scope(self):
        exprs = _expressions()
        payload = json.dumps({"exprs": exprs, "scopes": SCOPES})
        result = subprocess.run([NODE, "-e", EVAL_JS, payload],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        failures = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(
            failures, [],
            "a launcher `when` expression throws, so Pinokio will silently skip "
            "that step:\n  " + "\n  ".join(failures))

    def test_torch_js_does_not_depend_on_undeclared_globals(self):
        """The specific regression: gpus/which inside torch.js conditions."""
        with open(os.path.join(ROOT, "torch.js"), encoding="utf-8") as handle:
            conditions = " ".join(m.group(1) for m in WHEN.finditer(handle.read()))
        for name in ("gpus", "which("):
            self.assertNotIn(
                name, conditions,
                f"torch.js conditions reference `{name}`, which is not guaranteed "
                "to be in scope; when it is absent every branch throws and the "
                "GPU dependencies are never installed")


class StartRepairsAnEnvironmentWithoutOnnxRuntime(unittest.TestCase):
    """A venv with no onnxruntime must not be left for the user to debug."""

    def test_start_reinstalls_gpu_dependencies_when_onnxruntime_is_absent(self):
        with open(os.path.join(ROOT, "start_react.js"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("site-packages/onnxruntime/__init__.py", text,
                      "start_react.js does not check for a missing onnxruntime")
        self.assertIn("torch.js", text,
                      "start_react.js must repair through torch.js, which owns "
                      "the per-platform dependency versions")


if __name__ == "__main__":
    unittest.main()
