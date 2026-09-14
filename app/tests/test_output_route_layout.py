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
                result.add((decorator.func.attr.upper(), decorator.args[0].value))
    return result


class OutputRouteBoundaryTest(unittest.TestCase):
    def test_output_handlers_are_owned_by_the_output_router(self):
        paths = _paths(OUTPUT)
        self.assertEqual({
            ("GET", "/api/output"),
            ("POST", "/api/output/delete"),
            ("POST", "/api/reveal"),
            ("GET", "/api/file"),
        }, paths)

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
