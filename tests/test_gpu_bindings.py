"""GPU binding smoke test: DLL registration, ONNX Runtime providers, torch CUDA.

Guards the failure this project keeps hitting -- a runtime that *reports* it is
on the GPU while it is not. On Windows a CUDA/cuDNN DLL that cannot be resolved
surfaces as Win32 error 126 and onnxruntime then quietly hands the session to
the CPU provider; the process stays up, the picture is valid, only the speed
(and the TensorRT/CUDA claim) is wrong.

Two parts:

* ``TestRegistrationHelper`` -- stdlib only, runs in the light CI profile. Checks
  the directory ordering and the PATH/add_dll_directory side effects of
  ``app/run.py::_register_gpu_runtime_dirs`` against a fake site-packages tree.
* ``TestGpuBindings`` (marker ``gpu``) -- one fresh interpreter registers the
  directories exactly as run.py does, THEN imports onnxruntime and torch, so
  "registered before the first import" is tested rather than assumed.

Run: ``app/env/Scripts/python.exe -m pytest tests/test_gpu_bindings.py``
Stop the app first if it is rendering; the dummy session is tiny but a render
holds 12-15 GB.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RUN_PY = Path(__file__).resolve().parents[1] / "app" / "run.py"

# The two machines this project ships on: RTX 3060 Laptop (Ampere) and RTX 4070
# (Ada). A different capability is not "wrong", but it is not what any default
# here was measured on, so it fails loudly instead of passing by accident.
SUPPORTED_COMPUTE_CAPABILITIES = {(8, 6), (8, 9)}

_REGISTRATION_NAMES = {
    "_NVIDIA_DLL_PACKAGES", "_GPU_DLL_HANDLES",
    "_gpu_runtime_dirs", "_register_gpu_runtime_dirs",
}


def _registration_nodes() -> list[ast.stmt]:
    """The registration helpers lifted out of run.py without executing run.py.

    run.py runs preflight checks and imports the whole app at module level, so
    it cannot be imported here; the helpers are self-contained by contract
    (test_startup_preflight_import_order enforces that).
    """
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _REGISTRATION_NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in _REGISTRATION_NAMES
                for target in node.targets):
            nodes.append(node)
    found = {
        n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in nodes
    }
    assert found == _REGISTRATION_NAMES, (
        f"run.py is missing {sorted(_REGISTRATION_NAMES - found)}")
    return nodes


def _load_registration() -> dict:
    namespace = {"os": os, "sys": sys}
    module = ast.Module(body=_registration_nodes(), type_ignores=[])
    exec(compile(module, str(RUN_PY), "exec"), namespace)
    return namespace


def _make_site_packages(root: Path) -> dict[str, Path]:
    """A fake site-packages with every directory the helper knows about."""
    dirs = {
        "torch": root / "torch" / "lib",
        "trt_libs": root / "tensorrt_libs",
        "trt": root / "tensorrt",
        "cuda_runtime": root / "nvidia" / "cuda_runtime" / "bin",
        "cublas": root / "nvidia" / "cublas" / "bin",
        "cudnn": root / "nvidia" / "cudnn" / "bin",
        "curand": root / "nvidia" / "curand" / "bin",
        "cufft": root / "nvidia" / "cufft" / "bin",
        "cusparse": root / "nvidia" / "cusparse" / "bin",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True)
    # import libraries live next to the bin dirs and must never be registered
    (root / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
    return dirs


class TestRegistrationHelper:
    def test_directory_order(self, tmp_path):
        dirs = _make_site_packages(tmp_path)
        ns = _load_registration()
        found = ns["_gpu_runtime_dirs"]([str(tmp_path)])
        assert found == [str(dirs[k]) for k in (
            "torch", "trt_libs", "trt",
            "cuda_runtime", "cublas", "cudnn", "curand", "cufft",
            "cusparse",
        )]

    def test_missing_directories_are_skipped(self, tmp_path):
        (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
        ns = _load_registration()
        found = ns["_gpu_runtime_dirs"]([str(tmp_path), str(tmp_path / "nowhere")])
        assert found == [str(tmp_path / "nvidia" / "cudnn" / "bin")]

    def test_registration_prepends_path_and_registers_each_dir(self, tmp_path, monkeypatch):
        dirs = _make_site_packages(tmp_path)
        registered: list[str] = []
        monkeypatch.setattr(
            os, "add_dll_directory", lambda p: registered.append(p) or object(),
            raising=False)
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        elsewhere = str(tmp_path / "elsewhere")
        monkeypatch.setenv("PATH", os.pathsep.join([elsewhere, str(dirs["cudnn"])]))

        ns = _load_registration()
        result = ns["_register_gpu_runtime_dirs"]()

        assert registered == result and len(result) == len(dirs)
        entries = os.environ["PATH"].split(os.pathsep)
        assert entries[:len(result)] == result
        assert entries[len(result):] == [elsewhere], (
            "the pre-existing PATH entry must survive once, after ours")

    def test_registration_is_idempotent(self, tmp_path, monkeypatch):
        _make_site_packages(tmp_path)
        monkeypatch.setattr(os, "add_dll_directory", lambda p: object(), raising=False)
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        monkeypatch.setenv("PATH", "keep")

        ns = _load_registration()
        ns["_register_gpu_runtime_dirs"]()
        once = os.environ["PATH"]
        ns["_register_gpu_runtime_dirs"]()
        assert os.environ["PATH"] == once

    def test_no_add_dll_directory_leaves_path_alone(self, tmp_path, monkeypatch):
        _make_site_packages(tmp_path)
        monkeypatch.delattr(os, "add_dll_directory", raising=False)
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        monkeypatch.setenv("PATH", "untouched")

        assert _load_registration()["_register_gpu_runtime_dirs"]() == []
        assert os.environ["PATH"] == "untouched"

    def test_helpers_import_no_gpu_package(self):
        """Resolving a directory must not import the package it belongs to."""
        banned = {"torch", "tensorrt", "tensorrt_libs", "onnxruntime", "cv2",
                  "insightface", "nvidia", "roop", "settings"}
        imported: set[str] = set()
        for node in ast.walk(ast.Module(body=_registration_nodes(), type_ignores=[])):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "__import__" and node.args
                  and isinstance(node.args[0], ast.Constant)):
                imported.add(str(node.args[0].value).split(".")[0])
        assert not (imported & banned), sorted(imported & banned)


# ---------------------------------------------------------------------------
# GPU part: everything below needs the real environment.
# ---------------------------------------------------------------------------

_CHILD = r"""
import ast, json, os, sys, traceback

run_py = sys.argv[1]
wanted = set(sys.argv[2].split(","))
tree = ast.parse(open(run_py, encoding="utf-8").read())
body = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        body.append(node)
    elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
        body.append(node)
ns = {"os": os, "sys": sys}
exec(compile(ast.Module(body=body, type_ignores=[]), run_py, "exec"), ns)

report = {"registered": ns["_register_gpu_runtime_dirs"]()}
# Registration must have happened with none of these loaded yet.
report["loaded_before_import"] = sorted(
    m for m in ("onnxruntime", "torch", "cv2", "insightface", "tensorrt")
    if m in sys.modules)


def section(name, fn):
    try:
        report[name] = fn()
    except BaseException:
        report[name + "_error"] = traceback.format_exc()


def providers():
    import onnxruntime as ort
    report["ort_version"] = ort.__version__
    return list(ort.get_available_providers())


def dummy_cuda_session():
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    options = ort.SessionOptions()
    # Without this, a provider that fails to load (Win32 error 126) silently
    # leaves the session on CPU and the test would pass.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CUDAExecutionProvider"])
    out = session.run(None, {"x": np.arange(4, dtype=np.float32).reshape(1, 4)})[0]
    return {"providers": session.get_providers(), "output": out.tolist()}


def torch_cuda():
    import torch
    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)) if available else None,
        "device": torch.cuda.get_device_name(0) if available else None,
    }


section("providers", providers)
section("session", dummy_cuda_session)
section("torch", torch_cuda)
print("@@REPORT@@" + json.dumps(report))
"""


@pytest.fixture(scope="module")
def child_report() -> dict:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD, str(RUN_PY), ",".join(sorted(_REGISTRATION_NAMES))],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    marker = "@@REPORT@@"
    lines = [l for l in completed.stdout.splitlines() if l.startswith(marker)]
    assert lines, (
        f"child produced no report (exit {completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout[-3000:]}\n--- stderr ---\n{completed.stderr[-3000:]}")
    report = json.loads(lines[-1][len(marker):])
    # onnxruntime reports the real cause ("LoadLibrary failed with error 126")
    # on stderr as a warning; the exception it raises only says "CPU fallback".
    report["stderr"] = completed.stderr[-2500:]
    return report


def _fail_if_errored(report: dict, section: str) -> None:
    error = report.get(section + "_error")
    assert error is None, (
        f"{section} failed:\n{error}\n--- child stderr ---\n{report.get('stderr', '')}")
    assert "error 126" not in report.get("stderr", "").lower(), (
        f"Win32 error 126 (DLL not found) in the child's stderr:\n{report['stderr']}")


@pytest.mark.gpu
class TestGpuBindings:
    def test_registration_runs_before_any_gpu_package_is_imported(self, child_report):
        assert child_report["loaded_before_import"] == []
        assert child_report["registered"], (
            "no GPU runtime directory was found on sys.path; is this the app env?")

    def test_ort_lists_cuda_and_tensorrt_providers(self, child_report):
        _fail_if_errored(child_report, "providers")
        providers = child_report["providers"]
        assert "CUDAExecutionProvider" in providers, providers
        assert "TensorrtExecutionProvider" in providers, providers

    def test_dummy_cuda_session_loads_without_win32_error_126(self, child_report):
        _fail_if_errored(child_report, "session")
        session = child_report["session"]
        assert session["providers"][0] == "CUDAExecutionProvider", session["providers"]
        assert session["output"] == [[0.0, 2.0, 4.0, 6.0]]

    def test_torch_cuda_is_available_on_a_supported_architecture(self, child_report):
        _fail_if_errored(child_report, "torch")
        info = child_report["torch"]
        assert info["available"] is True, "torch.cuda.is_available() is False"
        assert tuple(info["capability"]) in SUPPORTED_COMPUTE_CAPABILITIES, (
            f"{info['device']} reports compute capability {info['capability']}; "
            f"expected one of {sorted(SUPPORTED_COMPUTE_CAPABILITIES)}")
