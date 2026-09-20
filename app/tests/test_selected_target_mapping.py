"""Selected-face target mapping: UI payload contract + backend behaviour.

Two halves, because the bug spanned both sides of the wire:

1. The UI decision itself is JavaScript, so it is exercised by running the real
   module through Node (react-ui/.render-check/face-mapping-check.mjs). This
   test shells out to that runner so the mapping is covered by the normal
   Python test sweep too, and skips when Node is unavailable rather than
   silently passing.
2. The payload the UI now emits is fed through the actual backend consumers,
   app.api.mapped_facesets / mapped_selected_index, to confirm that "only the
   highlighted person swaps" survives translation into facesets.
"""
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REACT_UI = ROOT / "react-ui"
CHECKER = REACT_UI / ".render-check" / "face-mapping-check.mjs"

sys.path.insert(0, str(ROOT / "app"))


def _node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    for cand in (
        Path("G:/pinokio/bin/miniconda/node.exe"),
        Path("G:/pinokio/bin/miniforge/node.exe"),
    ):
        if cand.exists():
            return str(cand)
    return None


class SelectedTargetMappingUITests(unittest.TestCase):
    """Runs the real FaceSwap mapping module, not a copy of its source text."""

    def test_face_mapping_behaviour_suite_passes(self):
        node = _node()
        if not node:
            self.skipTest("node not available to run the JS mapping suite")
        self.assertTrue(CHECKER.exists(), f"missing checker: {CHECKER}")
        proc = subprocess.run(
            [node, str(CHECKER)],
            cwd=str(REACT_UI),
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"JS mapping checks failed:\n{proc.stdout}\n{proc.stderr}",
        )
        self.assertIn("ALL GREEN", proc.stdout)


class MappedFacesetsBackendTests(unittest.TestCase):
    """The emitted payload, run through the backend that consumes it."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("ROOP_SKIP_STARTUP", "1")
        try:
            from roop.typing import FaceSet  # noqa: F401
        except Exception as exc:  # pragma: no cover - import env problem
            raise unittest.SkipTest(f"roop not importable: {exc}")

    def _mapped(self, mapping, source_count, swap_mode="selected"):
        import roop.globals as roop_globals
        from roop.typing import FaceSet
        import api

        prev = roop_globals.INPUT_FACESETS
        try:
            facesets = []
            for i in range(source_count):
                fs = FaceSet()
                fs._probe_id = i  # marker so we can tell the facesets apart
                facesets.append(fs)
            roop_globals.INPUT_FACESETS = facesets
            mapped = api.mapped_facesets(mapping, swap_mode)
            ids = None
            if mapped is not None:
                ids = [getattr(fs, "_probe_id", None) for fs in mapped]
            return mapped, ids
        finally:
            roop_globals.INPUT_FACESETS = prev

    def test_skip_entry_yields_empty_faceset_for_unselected_person(self):
        # [-1, 0] is what the fixed UI sends when person 1 is highlighted.
        mapped, ids = self._mapped([-1, 0], source_count=1)
        self.assertIsNotNone(mapped)
        self.assertEqual(len(mapped), 2)
        self.assertIsNone(ids[0], "unselected person must get an empty FaceSet")
        self.assertEqual(ids[1], 0, "highlighted person must get source 0")

    def test_old_payload_addressed_a_nonexistent_source(self):
        # The pre-fix payload for the same scenario. Person 1's entry pointed at
        # a source that does not exist, which is the empty-swap the user saw.
        mapped, ids = self._mapped([0, 1], source_count=1)
        self.assertEqual(ids[0], 0)
        self.assertIsNone(ids[1], "old payload silently produced an empty FaceSet")

    def test_selected_index_translates_into_mapped_space(self):
        import api
        mapping = [-1, 0]
        mapped, _ = self._mapped(mapping, source_count=1)
        # Gallery source 0 is person rank 1 in the mapped list.
        self.assertEqual(api.mapped_selected_index(mapping, mapped, 0), 1)

    def test_all_input_mode_opts_out_of_the_mapping(self):
        mapped, _ = self._mapped([-1, 0], source_count=2, swap_mode="all_input")
        self.assertIsNone(mapped, "all_input must keep gallery order")

    def test_every_emitted_entry_is_accepted_by_the_backend(self):
        # Sweep the shapes the fixed UI can emit; none may raise, and a -1 must
        # always mean "empty faceset" rather than an index.
        for mapping in ([-1], [0], [-1, 0], [0, -1], [-1, -1], [0, 1], [-1, 1]):
            for source_count in (1, 2):
                if any(x >= source_count for x in mapping):
                    continue
                mapped, ids = self._mapped(mapping, source_count)
                self.assertEqual(len(mapped), len(mapping))
                for want, got in zip(mapping, ids):
                    self.assertEqual(got, None if want < 0 else want)


if __name__ == "__main__":
    unittest.main()
