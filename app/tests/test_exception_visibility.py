"""Regression guards for the repository-wide fallback observability contract.

The production tree contains compatibility fallbacks that must keep rendering,
but none should disappear errors without either reporting them or raising them.
This test keeps future broad handlers from silently reintroducing the original
"feature is off but the render succeeds" failure mode.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop import degrade  # noqa: E402


def _production_files():
    return sorted(
        path for path in APP.rglob("*.py")
        if "env" not in path.parts
        and "tests" not in path.parts
        and "__pycache__" not in path.parts
        and (path.parent == APP or "roop" in path.parts)
    )


def _is_broad(handler):
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in {"Exception", "BaseException"}
    if isinstance(handler.type, ast.Tuple):
        return any(isinstance(item, ast.Name)
                   and item.id in {"Exception", "BaseException"}
                   for item in handler.type.elts)
    return False


def _is_observable(handler):
    if any(isinstance(node, ast.Raise) for node in ast.walk(handler)):
        return True
    reporting_names = {
        "print", "warn", "warning", "logger", "logging", "traceback",
        "notify", "bar_write", "update_status", "record_degradation",
        "swallowed", "_swallowed", "print_exc", "exception", "error",
    }
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        names = set()
        if isinstance(func, ast.Name):
            names.add(func.id.lower())
        elif isinstance(func, ast.Attribute):
            names.add(func.attr.lower())
            owner = func.value
            while isinstance(owner, ast.Attribute):
                names.add(owner.attr.lower())
                owner = owner.value
            if isinstance(owner, ast.Name):
                names.add(owner.id.lower())
        if names & reporting_names:
            return True
    return False


class TestProductionFallbackVisibility(unittest.TestCase):
    def test_every_broad_handler_reports_or_raises(self):
        silent = []
        for path in _production_files():
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                             filename=str(path))
            for handler in ast.walk(tree):
                if (isinstance(handler, ast.ExceptHandler)
                        and _is_broad(handler)
                        and not _is_observable(handler)):
                    silent.append(f"{path.relative_to(APP)}:{handler.lineno}")
        self.assertEqual([], silent,
                         "silent broad exception handlers remain:\n"
                         + "\n".join(f"  {item}" for item in silent))


class TestFallbackReporter(unittest.TestCase):
    def setUp(self):
        degrade.reset()
        self.addCleanup(degrade.reset)

    def test_counts_repeated_sites_without_repeating_the_record(self):
        for _ in range(3):
            degrade.swallowed("test.site", ValueError("bad"), "test detail")
        self.assertEqual(3, degrade.total_swallowed())
        self.assertEqual([{
            "site": "test.site",
            "count": 3,
            "detail": "test detail",
            "first_error": "ValueError: bad",
        }], degrade.report())

    def test_strict_mode_reraises_the_original_exception(self):
        error = RuntimeError("diagnose me")
        with patch.dict(os.environ, {"ROOP_STRICT_FALLBACK": "1"}):
            with self.assertRaises(RuntimeError) as caught:
                degrade.swallowed("test.strict", error)
            self.assertIs(error, caught.exception)
        self.assertEqual(1, degrade.total_swallowed())


if __name__ == "__main__":
    unittest.main()
