"""The two test suites must never share a module basename.

There are two suites in this repository:

    app/tests/   the application suite (a package: it has __init__.py)
    tests/       repo-root benchmark and integration harnesses (not a package)

Both are named ``tests``, and for a while both contained files with the SAME
basenames -- test_occlusion_mask.py, test_face_reference.py, verify_s5_profile.py
and seven more. That is not a cosmetic duplication:

  * ``unittest`` and pytest's default "prepend" import mode key test modules by
    BASENAME. Two files called ``test_occlusion_mask.py`` therefore resolve to
    one module, and whichever directory is reached first wins. The other file
    is silently never run.
  * The root copies were frozen snapshots of older app/ versions, so the suite
    that won was asserting against code that had since been fixed elsewhere.

This test fails the moment a basename exists in both trees again, which is the
only cheap way to stop the situation from rebuilding itself.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_TESTS = REPO_ROOT / "app" / "tests"
ROOT_TESTS = REPO_ROOT / "tests"


def _module_names(directory):
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.glob("*.py") if p.name != "__init__.py"}


class TestSuitesDoNotCollide(unittest.TestCase):
    def test_no_shared_basenames_between_the_two_suites(self):
        shared = _module_names(APP_TESTS) & _module_names(ROOT_TESTS)
        self.assertEqual(
            shared, set(),
            "these filenames exist in BOTH app/tests/ and tests/, so one "
            "shadows the other under basename-keyed test collection and only "
            "one of them actually runs:\n"
            + "\n".join(f"  {name}" for name in sorted(shared))
            + "\nKeep the application test in app/tests/ and give the "
              "root-level harness a distinct name.")

    def test_the_check_can_actually_see_both_suites(self):
        """Guard against the guard passing because it found nothing."""
        self.assertTrue(_module_names(APP_TESTS),
                        "app/tests/ has no modules; the collision check above "
                        "would pass vacuously")
        self.assertTrue(_module_names(ROOT_TESTS),
                        "tests/ has no modules; the collision check above "
                        "would pass vacuously")

    def test_no_root_level_roop_package_shadows_the_application_package(self):
        """`roop` must exist only under app/.

        A second, partial `roop/` package at the repo root used to extend its
        own __path__ into app/roop. Any module present at the root won, so
        stale copies of face_analyser/face_reference were imported instead of
        the maintained ones.
        """
        self.assertFalse(
            (REPO_ROOT / "roop").exists(),
            "a `roop/` directory exists at the repository root. The real "
            "package is app/roop; a root-level one shadows it and silently "
            "swaps in stale modules.")


if __name__ == "__main__":
    unittest.main()
