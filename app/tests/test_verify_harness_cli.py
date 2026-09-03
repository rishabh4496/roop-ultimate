"""The verify harness's entry point must actually start.

verify_roop_keep.py shipped once with --base-dir defaulting to None and no
fallback applied, so every invocation died on the first line of main() with
`TypeError: _getfullpathname: path should be string ... not NoneType`.
default_base() had been tested directly and passed. The entry point had not
been run -- the same "tested the function, not the wiring" failure this repo
keeps hitting.

These exercise the argument-resolution path without loading any model.
"""

import argparse
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from verify_roop_keep import default_base, resolve_base, resolve_faceset  # noqa: E402


class TestBaseResolution(unittest.TestCase):

    def test_default_base_is_a_string_path(self):
        base = default_base()
        self.assertIsInstance(base, str)
        self.assertTrue(base, "default_base() returned an empty path")

    def test_resolve_base_survives_an_unset_flag(self):
        """The exact shipped defect: --base-dir unset must not be None."""
        args = argparse.Namespace(base_dir=None)
        base = resolve_base(args)
        self.assertIsInstance(base, str)
        self.assertTrue(os.path.isabs(base))

    def test_resolve_base_honours_an_explicit_flag(self):
        args = argparse.Namespace(base_dir=os.path.join("some", "corpus"))
        self.assertTrue(resolve_base(args).endswith(os.path.join("some", "corpus")))

    def test_no_drive_letter_is_baked_in(self):
        """The 3060 has no G: drive; the root must be resolved, not literal."""
        import inspect
        import verify_roop_keep
        source = inspect.getsource(verify_roop_keep)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "roop-keep" not in stripped:
                continue
            self.assertNotRegex(
                stripped, r"['\"][A-Za-z]:[\/]{1,2}pinokio",
                "a drive-lettered corpus path is baked into code: %s" % stripped)


class TestFacesetResolution(unittest.TestCase):

    def test_a_missing_faceset_is_reported_not_silently_substituted(self):
        name, note = resolve_faceset("definitely_not_a_faceset_xyz")
        self.assertIsNone(name)
        self.assertIn("not found", note)

    def test_a_transposition_resolves_but_says_so(self):
        """mehak -> mahek must never happen quietly: a verification report
        naming the wrong person is worse than a failed run."""
        name, note = resolve_faceset("mehak")
        if name is not None and name != "mehak":
            self.assertIsNotNone(note)
            self.assertIn("SPELLING", note)
            self.assertIn(name, note)


if __name__ == "__main__":
    unittest.main()
