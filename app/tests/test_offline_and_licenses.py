"""Offline bundle, standalone install, and licence metadata audit.

Three groups:

  1. The React app must not fetch from another origin. Source and production
     bundle are scanned with the same policy as react-ui/scripts/verify-offline-bundle.mjs:
     a fetch position (script/link/img/iframe/CSS url or @import) may never name an
     http(s) origin, and every other http(s) URL must be on a short text-only host list
     (XML namespace, React's error-doc link, Tailwind's licence banner). Those strings
     are never requested. A literal "zero http(s) strings" rule cannot hold: React's
     SVG output contains the SVG namespace identifier.

  2. test_standalone_install.py must pass. It is run here through unittest and asserted.

  3. Licence metadata for the modules and models added in the recent stage commits
     (the four modules added since f9833cc). The brief asked for permissive-only
     licences (MIT, Apache-2.0, BSD-3-Clause). The repository does not meet that yet,
     and NOTICE.md says so for some models. Those cases are strict xfails with the
     evidence in the reason. They are not loosened and not claimed as passing.
"""
import io
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
UI = ROOT / "react-ui"
SRC = UI / "src"
DIST = UI / "dist"

TEXT_SUFFIXES = {".html", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".css", ".json",
                 ".svg", ".map", ".txt", ".webmanifest"}
TEXT_HOSTS = {"www.w3.org", "react.dev", "tailwindcss.com"}
FETCH_PATTERNS = [
    re.compile(r"<link\b[^>]*\bhref\s*=\s*[\"']?\s*(?:https?:)?//[^\s\"'>]+", re.I),
    re.compile(r"<script\b[^>]*\bsrc\s*=\s*[\"']?\s*(?:https?:)?//[^\s\"'>]+", re.I),
    re.compile(r"<(?:img|iframe|source|video|audio|embed)\b[^>]*\bsrc\s*=\s*[\"']?\s*(?:https?:)?//[^\s\"'>]+", re.I),
    re.compile(r"@import\s+(?:url\()?\s*[\"']?(?:https?:)?//[^\s\"')]+", re.I),
    re.compile(r"url\(\s*[\"']?(?:https?:)?//[^\s\"')]+", re.I),
]
# A real host has a dot and a TLD; this skips string fragments such as 'http://' in code.
URL = re.compile(r"https?://([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})[^\s\"'<>)`]*")

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-3-Clause"}


def _text_files(root):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def _scan(root):
    fetches, hosts = [], {}
    for path in _text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        for pattern in FETCH_PATTERNS:
            for m in pattern.finditer(text):
                fetches.append(f"{rel}: {m.group(0)[:120]}")
        for m in URL.finditer(text):
            hosts.setdefault(m.group(1).lower(), []).append(rel)
    return fetches, hosts


class TestOfflineBundle(unittest.TestCase):
    def test_production_bundle_is_built(self):
        self.assertTrue((DIST / "index.html").is_file(),
                        "react-ui/dist/index.html is missing; run the vite build first")

    def test_no_fetch_position_names_another_origin(self):
        for root in (SRC, DIST):
            with self.subTest(root=root.name):
                fetches, _ = _scan(root)
                self.assertEqual(fetches, [], "a fetch position reaches another origin")

    def test_every_other_url_host_is_text_only(self):
        for root in (SRC, DIST):
            with self.subTest(root=root.name):
                _, hosts = _scan(root)
                foreign = {h: files for h, files in hosts.items() if h not in TEXT_HOSTS}
                self.assertEqual(foreign, {}, "external host outside the text-only list")

    def test_text_hosts_are_only_namespace_licence_and_error_links(self):
        """Every text-only host really is text in the bundle: the namespace, the banner and the error link."""
        _, hosts = _scan(DIST)
        self.assertTrue(set(hosts) <= TEXT_HOSTS, sorted(set(hosts) - TEXT_HOSTS))
        self.assertIn("www.w3.org", hosts)  # the SVG namespace React emits

    def test_repo_verifier_agrees_when_node_is_available(self):
        # ROOP_NODE overrides PATH lookup, for machines where node is not on PATH for pytest.
        node = os.environ.get("ROOP_NODE") or shutil.which("node")
        if not node:
            self.skipTest("node is not on PATH (set ROOP_NODE); the Python scan above still ran")
        proc = subprocess.run([node, str(UI / "scripts" / "verify-offline-bundle.mjs")],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class TestStandaloneInstall(unittest.TestCase):
    def test_standalone_install_suite_passes(self):
        # Loaded by path: `tests` resolves to the root tests/ package when both trees are collected.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "standalone_install_check", APP / "tests" / "test_standalone_install.py")
        sa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa)
        suite = unittest.TestLoader().loadTestsFromTestCase(sa.TestStandaloneInstall)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
        self.assertTrue(result.wasSuccessful(), stream.getvalue())
        self.assertGreater(result.testsRun, 0)


# Modules added since f9833cc (git diff --diff-filter=A f9833cc HEAD -- app/roop).
NEW_MODULES = [
    "batch_matrix.py",
    "enhancer_stabilizer.py",
    "realism_matcher.py",
    "timeline_state.py",
]
_HAS_HEADER = {"timeline_state.py"}  # the others carry no SPDX line at HEAD


def _spdx(path):
    m = re.search(r"SPDX-License-Identifier:\s*([A-Za-z0-9.+-]+)", path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


_module_params = []
for name in NEW_MODULES:
    marks = []
    if name not in _HAS_HEADER:
        marks = [pytest.mark.xfail(strict=True, reason=(
            f"{name} has no SPDX-License-Identifier line; the repository licence is "
            "AGPL-3.0 (LICENSE, NOTICE.md) and no permissive licence has been chosen for this module."))]
    _module_params.append(pytest.param(name, marks=marks, id=name))


@pytest.mark.parametrize("name", _module_params)
def test_new_module_declares_a_permissive_spdx_licence(name):
    spdx = _spdx(APP / "roop" / name)
    assert spdx in PERMISSIVE, f"{name}: SPDX identifier is {spdx!r}"


@pytest.mark.xfail(strict=True, reason=(
    "The model registry (app/model_manifest.json) has no licence field on any of its 7 entries, "
    "so the licence cannot be checked from the registry. NOTICE.md records non-permissive terms "
    "for models in use: FaceFusion HyperSwap / Alpha are 'research and non-commercial usage "
    "restrictions as specified upstream'; ReSwapper is 'InsightFace non-commercial research terms'. "
    "Those are not MIT, Apache-2.0 or BSD-3-Clause."))
def test_every_registry_model_declares_a_permissive_licence():
    import json
    manifest = json.loads((APP / "model_manifest.json").read_text(encoding="utf-8"))
    bad = [m["file"] for m in manifest["models"] if m.get("license") not in PERMISSIVE]
    assert bad == [], f"models without a permissive licence: {bad}"


def test_notice_states_the_project_licence():
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    assert "AGPL-3.0" in notice
    assert (ROOT / "LICENSE").is_file()
