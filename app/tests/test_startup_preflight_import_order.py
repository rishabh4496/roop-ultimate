"""Startup must not break the `onnxruntime` module it then inspects.

A previous change added `from settings import _enable_tensorrt_runtime` ABOVE
`import onnxruntime` in run.py. `settings` imports `roop.runtime_optimizer`,
which imports onnxruntime itself, so by the time `run_preflight_checks()` ran
the name `ort` could be bound to a partially initialised module. Startup then
died with:

    AttributeError: module 'onnxruntime' has no attribute 'get_available_providers'

on installs where TensorRT, CUDA and onnxruntime-gpu were all present and
healthy. Pinokio reported it as a failed launch and terminated the shell.

These tests pin the two properties that prevent it:
  1. run.py's DLL registration is self-contained (no settings/roop import
     before onnxruntime is imported).
  2. the preflight check degrades to a warning instead of raising when the
     provider API is genuinely unavailable.
"""
from __future__ import annotations

import ast
import contextlib
import io
import types
import unittest
from pathlib import Path


RUN_PY = Path(__file__).resolve().parents[1] / "run.py"


def _module_source() -> str:
    return RUN_PY.read_text(encoding="utf-8")


def _imports_before_onnxruntime(tree: ast.Module) -> list[str]:
    """Module-level import names that execute before `import onnxruntime`."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "onnxruntime":
                    return names
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "onnxruntime":
                return names
            names.append(node.module)
    return names


class PreflightImportOrdering(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(_module_source())

    def test_no_app_module_is_imported_before_runtime_registration(self):
        """DLL registration must happen before the first app-module import."""
        source = _module_source()
        registration_line = next(
            node.lineno for node in self.tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_register_gpu_runtime_dirs"
        )
        offenders = []
        for node in self.tree.body:
            if node.lineno >= registration_line:
                continue
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            offenders.extend(
                name for name in names
                if name == "settings" or name == "roop" or name.startswith("roop.")
            )
        self.assertEqual(
            offenders, [],
            "run.py imports %s before DLL registration" % offenders)

    def test_dll_registration_helper_exists_and_is_self_contained(self):
        """The helper must resolve DLL directories without the app packages."""
        helper = next(
            (n for n in ast.walk(self.tree)
             if isinstance(n, ast.FunctionDef)
             and n.name == "_register_gpu_runtime_dirs"), None)
        self.assertIsNotNone(
            helper, "run.py must register GPU runtime DLL directories itself")

        imported: list[str] = []
        for node in ast.walk(helper):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)
                  and node.func.id == "__import__"
                  and node.args
                  and isinstance(node.args[0], ast.Constant)):
                imported.append(str(node.args[0].value))

        forbidden = [n for n in imported
                     if n == "settings" or n == "roop" or n.startswith("roop.")]
        self.assertEqual(
            forbidden, [],
            f"_register_gpu_runtime_dirs imports {forbidden}; it must stay "
            "independent of the model stack")

    def test_authoritative_preflight_owns_onnxruntime_import(self):
        """The central preflight must own the guarded ORT import."""
        source = _module_source()
        self.assertNotIn("import onnxruntime as ort", source)
        self.assertIn(
            "from roop.gpu_preflight import get_preflight_result", source)


class PreflightDegradesInsteadOfCrashing(unittest.TestCase):
    """`run_preflight_checks` is the first thing the launcher executes."""

    def _preflight(self, result):
        import sys

        namespace = {
            "sys": sys,
            "np": types.SimpleNamespace(__version__="1.26.4"),
        }
        function = next(n for n in ast.parse(_module_source()).body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "run_preflight_checks")
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(RUN_PY), "exec"), namespace)

        fake_preflight = types.ModuleType("roop.gpu_preflight")
        fake_preflight.get_preflight_result = lambda: result
        saved = sys.modules.get("roop.gpu_preflight")
        sys.modules["roop.gpu_preflight"] = fake_preflight
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                namespace["run_preflight_checks"]()
        finally:
            if saved is not None:
                sys.modules["roop.gpu_preflight"] = saved
            else:
                sys.modules.pop("roop.gpu_preflight", None)
        return buffer.getvalue()

    def test_a_module_without_the_provider_api_is_a_warning_not_a_crash(self):
        """This is the exact shape that terminated the Pinokio shell."""
        broken = {
            "onnxruntime_importable": True,
            "available_providers": [],
            "active_provider": "none",
            "failure_stage": "provider_not_compiled",
            "failure_reason": "get_available_providers is missing",
        }
        with self.assertRaisesRegex(SystemExit, "provider_not_compiled"):
            self._preflight(broken)

    def test_a_failing_provider_query_is_a_warning_not_a_crash(self):
        result = {
            "onnxruntime_importable": True,
            "available_providers": [],
            "active_provider": "none",
            "failure_stage": "provider_not_compiled",
            "failure_reason": "provider registry unavailable",
        }
        with self.assertRaisesRegex(SystemExit, "provider registry unavailable"):
            self._preflight(result)

    def test_a_healthy_tensorrt_runtime_is_reported_ok(self):
        result = {
            "onnxruntime_importable": True,
            "available_providers": [
                "TensorrtExecutionProvider", "CUDAExecutionProvider",
                "CPUExecutionProvider"],
            "active_provider": "TensorrtExecutionProvider",
            "tensorrt_session_usable": True,
            "failure_stage": None,
            "failure_reason": None,
        }
        self.assertIn("[OK] TensorRT minimal session verified as active.",
                      self._preflight(result))

    def test_a_cuda_only_runtime_still_starts_with_a_warning(self):
        result = {
            "onnxruntime_importable": True,
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "active_provider": "CUDAExecutionProvider",
            "tensorrt_session_usable": False,
            "failure_stage": "tensorrt_session_construction_failure",
            "failure_reason": "TensorRT failed to initialize",
        }
        output = self._preflight(result)
        self.assertIn("CUDA is the validated fallback", output)
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
