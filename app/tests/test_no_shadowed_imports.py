"""A late import inside a function makes the name local for the WHOLE function.

`ProcessMgr.process_face` used the module-level `estimate_norm` at the top of
the enhancer re-align, and 200 lines further down re-imported the same name
inside a `try:` for the canonical-mask path. That second import is what binds
the name in the function's scope, so every earlier read raised

    local variable 'estimate_norm' referenced before assignment

on every single face — swallowed by the re-align's `except`, which quietly fell
back to the unaligned crop. Nothing crashed and no test failed; the feature was
just permanently off.

`test_no_undefined_names` cannot see this: it treats every binding as covering
the whole module, which is exactly the assumption Python breaks here. So this
check looks at scope and order instead — a function-local import of a name the
module already imports, read anywhere earlier in that same function.
"""

import ast
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

SKIP_DIRS = {"env", "tests", "clip", "installer", "models", "output", "temp",
             "__pycache__", "facesets", "docs", "sidecar_keep", "ui"}

FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _module_files():
    for root in (APP, APP / "roop", APP / "roop" / "processors"):
        yield from sorted(root.glob("*.py"))


def _module_imports(tree):
    """Names bound by imports at module level."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _own_scope(func):
    """Nodes belonging to *func* itself, excluding nested function scopes."""
    stack, out = list(func.body), []
    while stack:
        node = stack.pop()
        out.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FUNCS + (ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)
    return out


def shadowed_imports(path):
    """(function, name, use_line, import_line) for every module-level name that a
    function re-imports after already reading it."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    module = _module_imports(tree)
    found = []
    for func in ast.walk(tree):
        if not isinstance(func, FUNCS):
            continue
        nodes = _own_scope(func)
        imported = {}
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    if name in module:
                        # The EARLIEST local import is the one that matters: a
                        # read below it is already bound, so only reads above
                        # the first one can hit the unbound window.
                        imported[name] = min(imported.get(name, node.lineno),
                                             node.lineno)
        if not imported:
            continue
        for node in nodes:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
                        and sub.id in imported and sub.lineno < imported[sub.id]):
                    found.append((func.name, sub.id, sub.lineno, imported[sub.id]))
    return sorted(set(found))


class TestNoShadowedImports(unittest.TestCase):
    def test_no_function_reads_a_name_it_reimports_later(self):
        broken = {}
        for path in _module_files():
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            hits = shadowed_imports(path)
            if hits:
                broken[str(path.relative_to(APP))] = hits
        self.assertEqual(
            broken, {},
            "a later function-local import makes the name local for the whole "
            "function, so these earlier reads raise UnboundLocalError:\n"
            + "\n".join(
                f"  {f}: {fn}() reads {name} at line {use}, "
                f"re-imports it at line {imp}"
                for f, hits in sorted(broken.items())
                for fn, name, use, imp in hits))

    def test_the_check_actually_detects_the_shadow(self):
        """Guard against the guard silently passing everything."""
        import tempfile
        src = ("from os import sep\n"
               "def f():\n"
               "    a = sep\n"
               "    from os import sep\n"
               "    return a, sep\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shadow.py"
            p.write_text(src, encoding="utf-8")
            self.assertEqual(shadowed_imports(p), [("f", "sep", 3, 4)])

    def test_a_local_import_before_every_use_is_fine(self):
        """The common, harmless pattern must not be flagged."""
        import tempfile
        src = ("from os import sep\n"
               "def f():\n"
               "    from os import sep\n"
               "    return sep\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ok.py"
            p.write_text(src, encoding="utf-8")
            self.assertEqual(shadowed_imports(p), [])

    def test_a_second_redundant_import_below_does_not_accuse_the_first(self):
        """ProcessMgr.initialize() imports get_first_face in two sibling
        branches; the read between them is bound by the one above it."""
        import tempfile
        src = ("from os import sep\n"
               "def f():\n"
               "    from os import sep\n"
               "    a = sep\n"
               "    from os import sep\n"
               "    return a, sep\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "twice.py"
            p.write_text(src, encoding="utf-8")
            self.assertEqual(shadowed_imports(p), [])


if __name__ == "__main__":
    unittest.main()
