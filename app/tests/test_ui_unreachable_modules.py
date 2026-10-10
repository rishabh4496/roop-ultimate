"""No module under react-ui/src may be unreachable from the app's entry point.

Six component folders (studio, preview, timeline, facebank, queue, telemetry:
6,771 lines in 23 modules) sat in src/ for two weeks with no importer. Each had its
own render-check script and each script passed, so the suite was green over code
the app could not run: the project's recurring failure of something reporting
success while not running. The build cannot see it (Vite only bundles what is
imported), lint cannot, and a test that imports a module directly proves nothing
about whether the app does.

`react-ui/scripts/unreachable-modules.mjs` is the scan (`npm run lint:unreachable`,
part of `npm run check`). This runs it from the Python suite too, and checks the
scan itself against a fixture tree, because a scanner that reports 0 is worthless
unless it is known to fire.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "react-ui" / "scripts" / "unreachable-modules.mjs"


def _node():
    found = shutil.which("node")
    if found:
        return found
    # <PINOKIO_HOME>/api/roop-ultimate is this repo; Pinokio's own node lives under bin/.
    home = ROOT.parents[1]
    for cand in (home / "bin" / "miniconda" / "node.exe", home / "bin" / "miniforge" / "node.exe"):
        if cand.exists():
            return str(cand)
    return None


def _scan(script):
    node = _node()
    proc = subprocess.run([node, str(script), "--json"], capture_output=True, text=True, timeout=60)
    if proc.returncode not in (0, 1):
        raise AssertionError(f"scan crashed ({proc.returncode}): {proc.stderr}")
    return proc.returncode, json.loads(proc.stdout)


@unittest.skipUnless(_node(), "node is not available")
class UnreachableModules(unittest.TestCase):
    def test_every_module_in_src_is_reachable_from_main(self):
        code, report = _scan(SCRIPT)
        self.assertGreater(report["files"], 80, "the scan found almost no modules; it has drifted from the source")
        self.assertEqual(
            report["unreachable"], [],
            "modules nothing imports (import them or delete them; there is no allowlist): "
            + ", ".join(f"{u['file']} ({u['lines']} lines)" for u in report["unreachable"]))
        self.assertEqual(code, 0)

    def test_the_scan_fires_on_an_orphan_and_follows_every_kind_of_import(self):
        """Run the real script against a tree it can be wrong about."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "react-ui"
            (pkg / "scripts").mkdir(parents=True)
            (pkg / "src" / "components" / "sub").mkdir(parents=True)
            shutil.copy(SCRIPT, pkg / "scripts" / "unreachable-modules.mjs")
            files = {
                # static import, side-effect import, re-export through an index, dynamic import,
                # a worker by URL, and an import that only appears in a COMMENT.
                "main.jsx": "import './index.css'\nimport App from './App.jsx'\n",
                "index.css": "body{}",
                "App.jsx": ("import { a } from './components/sub'\n"
                            "const Lazy = () => import('./components/Lazy')\n"
                            "new Worker(new URL('./components/job.worker.js', import.meta.url))\n"
                            "// import mentioned from './components/OnlyInAComment'\n"
                            "/* import blockcomment from './components/OnlyInABlockComment' */\n"
                            "const url = `${proto}//${host}/ws/x`  // a template with // is not a comment start\n"),
                "components/sub/index.js": "export { a } from './a'\n",
                "components/sub/a.js": "export const a = 1\n",
                "components/Lazy.jsx": "export default () => null\n",
                "components/job.worker.js": "self.onmessage = () => {}\n",
                "components/Orphan.jsx": "export default () => null\n",
                "components/OnlyInAComment.jsx": "export default 1\n",
                "components/OnlyInABlockComment.jsx": "export default 1\n",
                "components/ImportsOnlyOrphans.jsx": "import Orphan from './Orphan'\nexport default Orphan\n",
            }
            for rel, body in files.items():
                (pkg / "src" / rel).write_text(body, encoding="utf-8")
            code, report = _scan(pkg / "scripts" / "unreachable-modules.mjs")
            self.assertEqual(code, 1, "an orphan must make the scan exit non-zero")
            self.assertEqual(
                sorted(u["file"] for u in report["unreachable"]),
                ["components/ImportsOnlyOrphans.jsx", "components/OnlyInABlockComment.jsx",
                 "components/OnlyInAComment.jsx", "components/Orphan.jsx"])


if __name__ == "__main__":
    unittest.main()
