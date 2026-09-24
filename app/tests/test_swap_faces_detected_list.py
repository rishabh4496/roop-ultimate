"""`faces` inside ProcessMgr.swap_faces is THIS FRAME'S detected faces.

4bd577d (2026-09-24) reused the name for a source faceset's faces while
building `allowed_source_indices`:

    faces = getattr(src_data, 'faces', None)

The per-face loop further down then iterated the SOURCE photo's face instead of
the target, refused it on distance, and never looked at the real face. Every
selected-mode render (the default) came out untouched, the audit read
"faces seen 1 ... NOT swapped", and all 3454 tests passed. The regression
benchmark (`run.py --benchmark --benchmark-mode regression`) is what caught it.

Source-level on purpose: swap_faces needs the whole model stack to run, and
the defect is a NAME collision, which the AST shows exactly. Every assignment
to `faces` must derive from detection, the prepass replay, or `faces` itself.
"""

import ast
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]

# The expressions `faces` may be assigned from, and only these. A new one is
# fine if it is still the frame's faces -- add it here deliberately.
ALLOWED = {
    "list(_tfaces)",
    "get_all_faces(frame)",
    "recovered",
    "_tracker.update(faces, frame_idx)",
    "list(faces) + list(_coasted)",
}


def _swap_faces():
    tree = ast.parse((APP / "roop" / "ProcessMgr.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "swap_faces":
            return node
    raise AssertionError("ProcessMgr.swap_faces not found")


def _assignments_to_faces(fn):
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets, value = [node.target], node.value
        elif isinstance(node, (ast.For, ast.comprehension)):
            targets, value = [node.target], node.iter
        elif isinstance(node, (ast.With,)):
            targets = [i.optional_vars for i in node.items if i.optional_vars is not None]
            value = None
        else:
            continue
        for target in targets:
            if any(isinstance(n, ast.Name) and n.id == "faces" for n in ast.walk(target)):
                yield node.lineno, value


class DetectedFacesName(unittest.TestCase):
    def test_faces_is_only_ever_the_frames_faces(self):
        found = list(_assignments_to_faces(_swap_faces()))
        self.assertTrue(found, "no assignment to `faces` found -- has swap_faces moved?")
        for lineno, value in found:
            text = ast.unparse(value) if value is not None else "<with/for target>"
            if text.startswith("sorted(faces"):
                continue   # the claim-order sort: same faces, reordered
            with self.subTest(line=lineno):
                self.assertIn(text, ALLOWED,
                              "ProcessMgr.py:%d assigns `faces = %s`. `faces` is the frame's "
                              "detected faces; use another name." % (lineno, text))

    def test_no_source_data_reaches_the_name(self):
        for lineno, value in _assignments_to_faces(_swap_faces()):
            if value is None:
                continue
            names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(value) if isinstance(n, ast.Attribute)}
            with self.subTest(line=lineno):
                self.assertFalse({"src_data", "source_data"} & names)
                self.assertNotIn("input_face_datas", attrs)


if __name__ == "__main__":
    unittest.main()
