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

    def test_no_app_module_is_imported_before_onnxruntime(self):
        """settings/roop pull in onnxruntime themselves; importing them first
        is what left `ort` half-initialised."""
        offenders = [
            name for name in _imports_before_onnxruntime(self.tree)
            if name == "settings" or name == "roop" or name.startswith("roop.")
        ]
        self.assertEqual(
            offenders, [],
            "run.py imports %s before onnxruntime; that is the exact ordering "
            "that produced 'module onnxruntime has no attribute "
            "get_available_providers' at startup" % offenders)

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

    def test_the_helper_runs_before_onnxruntime_is_imported(self):
        """Registering the directories after the import would be a no-op."""
        source = _module_source()
        call_at = source.find("\n_register_gpu_runtime_dirs()")
        import_at = source.find("\nimport onnxruntime")
        self.assertNotEqual(call_at, -1, "the helper is never called")
        self.assertNotEqual(import_at, -1, "onnxruntime is never imported")
        self.assertLess(call_at, import_at,
                        "DLL directories must be registered before ORT loads")


class PreflightDegradesInsteadOfCrashing(unittest.TestCase):
    """`run_preflight_checks` is the first thing the launcher executes."""

    def _preflight(self, ort_module):
        namespace = {
            "sys": __import__("sys"),
            "np": types.SimpleNamespace(__version__="1.26.4"),
            "ort": ort_module,
        }
        function = next(n for n in ast.parse(_module_source()).body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "run_preflight_checks")
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(RUN_PY), "exec"), namespace)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            namespace["run_preflight_checks"]()
        return buffer.getvalue()

    def test_a_module_without_the_provider_api_is_a_warning_not_a_crash(self):
        """This is the exact shape that terminated the Pinokio shell."""
        broken = types.ModuleType("onnxruntime")     # no get_available_providers
        broken.__file__ = "/somewhere/onnxruntime/__init__.py"
        output = self._preflight(broken)
        self.assertIn("WARNING", output)
        self.assertNotIn("Traceback", output)

    def test_a_failing_provider_query_is_a_warning_not_a_crash(self):
        def explode():
            raise RuntimeError("provider registry unavailable")

        module = types.ModuleType("onnxruntime")
        module.get_available_providers = explode
        output = self._preflight(module)
        self.assertIn("WARNING", output)
        self.assertIn("provider registry unavailable", output)

    def test_a_healthy_tensorrt_runtime_is_reported_ok(self):
        module = types.ModuleType("onnxruntime")
        module.get_available_providers = lambda: [
            "TensorrtExecutionProvider", "CUDAExecutionProvider",
            "CPUExecutionProvider"]
        self.assertIn("[OK] TensorrtExecutionProvider registered.",
                      self._preflight(module))

    def test_a_cuda_only_runtime_still_starts_with_a_warning(self):
        module = types.ModuleType("onnxruntime")
        module.get_available_providers = lambda: [
            "CUDAExecutionProvider", "CPUExecutionProvider"]
        output = self._preflight(module)
        self.assertIn("TensorrtExecutionProvider not found", output)
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
