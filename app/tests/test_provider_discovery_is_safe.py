"""No production code may call `get_available_providers` unguarded.

A broken ONNX Runtime install is not hypothetical here. When the package is
absent but a directory of that name is importable -- a half-removed install, or
an install step that never ran -- Python resolves `onnxruntime` to an implicit
NAMESPACE package: `__file__` is None and the module has no attributes. Every
bare `ort.get_available_providers()` then raises AttributeError.

That is what took the app down twice:

    AttributeError: module 'onnxruntime' has no attribute 'get_available_providers'

`roop/core.py` was the worst case, because its call was at MODULE level and
therefore raised during `from roop import core`, where no caller could handle
it. `roop.ort_support.available_providers()` returns [] instead, which every
one of these code paths already handles as "no GPU provider".
"""
from __future__ import annotations

import ast
import os
import sys
import types
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files whose provider discovery runs on a normal startup or update path.
PRODUCTION = (
    os.path.join("roop", "core.py"),
    os.path.join("roop", "utilities.py"),
    os.path.join("roop", "optimized_processor.py"),
    os.path.join("roop", "trt_session_builder.py"),
    os.path.join("roop", "runtime_optimizer.py"),
    os.path.join("roop", "backend_manager.py"),
    os.path.join("roop", "predictor.py"),
    os.path.join("roop", "bench.py"),
    os.path.join("ui", "main.py"),
    "update_health.py",
    "update_manager.py",
)


def _calls(tree):
    """Every `<something>.get_available_providers()` call node."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_available_providers"):
            yield node


def _guarding_handlers(tree, lineno):
    """Exception types guarding `lineno`, or None when it is unguarded."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = {n.lineno for b in node.body for n in ast.walk(b)
                if hasattr(n, "lineno")}
        if lineno not in body:
            continue
        names = []
        for handler in node.handlers:
            if handler.type is None:
                names.append("bare")
            elif isinstance(handler.type, ast.Name):
                names.append(handler.type.id)
            elif isinstance(handler.type, ast.Tuple):
                names += [e.id for e in handler.type.elts
                          if isinstance(e, ast.Name)]
        return names
    return None


class ProviderDiscoveryIsAlwaysSafe(unittest.TestCase):
    def test_the_helper_exists_and_degrades_to_an_empty_list(self):
        sys.path.insert(0, APP)
        try:
            from roop.ort_support import available_providers, onnxruntime_is_usable
        finally:
            sys.path.remove(APP)

        broken = types.ModuleType("onnxruntime")      # namespace-package shape
        broken.__file__ = None
        from roop.gpu_preflight import clear_preflight_cache
        clear_preflight_cache()
        saved = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = broken
        try:
            self.assertEqual(available_providers(), [])
            self.assertFalse(onnxruntime_is_usable())
        finally:
            if saved is not None:
                sys.modules["onnxruntime"] = saved
            else:
                sys.modules.pop("onnxruntime", None)
            clear_preflight_cache()

    def test_no_production_call_is_unguarded(self):
        offenders = []
        for relative in PRODUCTION:
            path = os.path.join(APP, relative)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            for call in _calls(tree):
                handlers = _guarding_handlers(tree, call.lineno)
                broad = handlers is not None and any(
                    name in ("Exception", "BaseException", "AttributeError", "bare")
                    for name in handlers)
                if not broad:
                    offenders.append(f"{relative}:{call.lineno}")
        self.assertEqual(
            offenders, [],
            "unguarded get_available_providers() calls (use "
            "roop.ort_support.available_providers instead):\n  "
            + "\n  ".join(offenders))

    def test_core_does_not_discover_providers_at_import_time_unguarded(self):
        """core.py's call ran at module level; an exception there is fatal."""
        with open(os.path.join(APP, "roop", "core.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:                    # module level only
            if isinstance(node, (ast.Try, ast.FunctionDef, ast.ClassDef)):
                continue
            for call in _calls(ast.Module(body=[node], type_ignores=[])):
                self.fail(
                    f"roop/core.py line {call.lineno} discovers providers at "
                    "module level without a guard; a broken onnxruntime makes "
                    "`from roop import core` raise")


class TheSmallGpuTensorRTGateIsConsistent(unittest.TestCase):
    """One env var, one default, or a 6GB card gets contradictory answers.

    backend_manager admits TensorRT on a sub-7GB card, so a profiler that
    still defaults the same flag to OFF reports "backend admission remains
    CUDA/CPU" and skips the capability probe that feeds precision selection.
    """

    def test_every_reader_uses_the_same_default(self):
        pattern = "ROOP_ALLOW_TRT_SMALL_GPU"
        defaults = {}
        for relative in (os.path.join("roop", "backend_manager.py"),
                         os.path.join("roop", "runtime_optimizer.py"),
                         os.path.join("roop", "face_util.py")):
            path = os.path.join(APP, relative)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == pattern):
                    fallback = (node.args[1].value
                                if len(node.args) > 1
                                and isinstance(node.args[1], ast.Constant)
                                else "<none>")
                    defaults.setdefault(relative, set()).add(fallback)

        self.assertTrue(defaults, f"no reader of {pattern} was found")
        distinct = {value for values in defaults.values() for value in values}
        self.assertEqual(
            len(distinct), 1,
            f"{pattern} is read with conflicting defaults {defaults}; a sub-7GB "
            "card then gets TensorRT admitted by one module and refused by "
            "another")


if __name__ == "__main__":
    unittest.main()
