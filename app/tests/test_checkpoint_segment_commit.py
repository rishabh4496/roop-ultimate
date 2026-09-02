"""`_checkpoint_segment` must actually reach `update_checkpoint`.

It is wrapped in `except Exception` -- deliberately, because a failed
checkpoint must not kill a render -- and it called
`_project_checkpoint.manifest_path`, a function that has never existed on that
module (it belongs to `roop.segment_writer`, which owns the manifest).

So every segment commit raised AttributeError and was swallowed as one line of
output: "[Resume] project checkpoint failed: module 'project_checkpoint' has
no attribute 'manifest_path'". `update_checkpoint` was never reached from the
writer path at all. The visible consequence is that EVERY project on disk
reports `segments: 0` no matter how many parts it wrote, so the one durable
record of what survives an interruption was always empty -- and a resume had
nothing to look at except `safe_frame`, which is not backed by anything on
disk.

A broad `except` around a call means the call's spelling has no test coverage
unless something asserts the happy path is reachable. That is what this is.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import project_checkpoint
from roop import segment_writer


class TheFunctionLivesWhereItIsCalledFrom(unittest.TestCase):
    def test_manifest_path_is_segment_writers(self):
        self.assertTrue(callable(getattr(segment_writer, "manifest_path", None)),
                        "segment_writer owns the manifest and must expose its path")

    def test_project_checkpoint_does_not_define_it(self):
        """Pinning the premise. If it is ever added there, this test says so
        rather than letting two spellings drift apart again."""
        self.assertFalse(hasattr(project_checkpoint, "manifest_path"))

    def test_api_calls_it_on_segment_writer(self):
        source_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "api.py")
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("segment_writer.manifest_path(", source)
        self.assertNotIn("_project_checkpoint.manifest_path(", source)


class EveryProjectCheckpointAttributeApiUsesExists(unittest.TestCase):
    """The general form. `_project_checkpoint.<name>` is only ever verified at
    runtime inside a broad except, so a typo is silent."""

    def test_no_api_call_names_a_missing_project_checkpoint_attribute(self):
        # Parsed, not grepped: the fix's own comment discusses the old wrong
        # spelling, and a guard that reads prose would fail on the explanation
        # of the bug it exists to prevent.
        import ast
        source_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "api.py")
        with open(source_path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        used = {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "_project_checkpoint"}
        self.assertTrue(used, "expected api.py to use project_checkpoint")
        missing = sorted(n for n in used if not hasattr(project_checkpoint, n))
        self.assertEqual([], missing,
                         f"api.py calls project_checkpoint attributes that do not "
                         f"exist: {missing}")


if __name__ == "__main__":
    unittest.main()
