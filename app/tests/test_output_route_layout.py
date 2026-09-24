"""Keep API output routes behind one explicit router boundary."""

import ast
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
API = APP / "api.py"
OUTPUT = APP / "routes_output.py"


def _paths(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "router"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)):
                verb = decorator.func.attr.upper()
                if verb == "API_ROUTE":
                    # @router.api_route(path, methods=[...]) -- one entry per
                    # method, so a GET+HEAD media route reads as its verbs.
                    methods = []
                    for kw in decorator.keywords:
                        if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                            methods = [m.value for m in kw.value.elts if isinstance(m, ast.Constant)]
                    for m in methods:
                        result.add((str(m).upper(), decorator.args[0].value))
                    continue
                result.add((verb, decorator.args[0].value))
    return result


class OutputRouteBoundaryTest(unittest.TestCase):
    def test_output_handlers_are_owned_by_the_output_router(self):
        paths = _paths(OUTPUT)
        media = {"/api/file", "/outputs/{filename:path}",
                 "/api/media/{filename:path}", "/static/outputs/{filename:path}",
                 # The original target of the latest output (compare view).
                 "/api/output/source"}
        expected = {
            ("GET", "/api/output"),
            ("POST", "/api/output/delete"),
            ("POST", "/api/reveal"),
        }
        # Every media path answers HEAD as well as GET (a <video> probes with
        # HEAD before it ranges the body).
        for path in media:
            expected.add(("GET", path))
            expected.add(("HEAD", path))
        self.assertEqual(expected, paths)

    def test_api_registers_one_router_and_keeps_import_compatibility_only(self):
        source = API.read_text(encoding="utf-8")
        self.assertIn("app.include_router(_routes_output.router)", source)
        self.assertIn("list_output = _routes_output.list_output", source)
        self.assertNotIn('@app.get("/api/output")', source)
        self.assertNotIn('@app.get("/api/file")', source)
        self.assertNotIn('@app.post("/api/output/delete")', source)
        self.assertNotIn('@app.post("/api/reveal")', source)


if __name__ == "__main__":
    unittest.main()
