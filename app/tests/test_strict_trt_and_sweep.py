"""The strict TensorRT assertion, and verify_trt_fps.py's batch sweep.

assert_strict_tensorrt_session used to require get_providers() == ["TensorrtExecutionProvider"].
ONNX Runtime appends CPUExecutionProvider to EVERY session (ORT 1.23.2: providers=[TRT]
reports [TRT, CPU]), so the old check rejected every healthy session - even an already
batch-dynamic model - and only passed against test doubles that returned the bare list.
"""
from __future__ import annotations

import argparse
import sys
import types

import pytest

from roop import trt_session_builder as builder
from roop.trt_session_builder import TensorRTSessionError, assert_strict_tensorrt_session

TRT, CUDA, CPU = "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"


class _Session:
    def __init__(self, active):
        self._active = active

    def get_providers(self):
        return list(self._active)


@pytest.mark.parametrize("active", [[TRT], [TRT, CPU]])
def test_tensorrt_first_with_ortss_own_trailing_cpu_entry_is_accepted(active):
    assert_strict_tensorrt_session(_Session(active), "m.onnx")


@pytest.mark.parametrize("active", [[], [CPU], [CUDA, TRT], [TRT, CUDA], [TRT, CUDA, CPU], [CUDA, CPU]])
def test_anything_that_is_not_tensorrt_first_without_cuda_is_rejected(active):
    with pytest.raises(TensorRTSessionError, match="strict TensorRT session rejected"):
        assert_strict_tensorrt_session(_Session(active), "m.onnx")


def test_build_tensorrt_session_forbids_cpu_nodes_at_construction(tmp_path, monkeypatch):
    """'No CPU fallback' is enforced where ORT can enforce it, not by inspecting the list."""
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    entries, built = {}, {}

    class Options:
        graph_optimization_level = None

        def add_session_config_entry(self, key, value):
            entries[key] = value

    class Ort(types.ModuleType):
        GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)
        SessionOptions = Options

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):  # noqa: N802
            built["providers"] = providers
            return _Session([TRT, CPU])

    monkeypatch.setitem(sys.modules, "onnxruntime", Ort("onnxruntime"))
    monkeypatch.setattr(builder, "prepare_tensorrt_runtime", lambda: None)
    monkeypatch.setattr("roop.ort_support.available_providers", lambda: [TRT, CUDA, CPU])
    monkeypatch.setattr(builder.TensorRTSessionConfig, "provider_options", lambda self, path: {"device_id": 0})

    session = builder.build_tensorrt_session(model)
    assert session.get_providers() == [TRT, CPU]
    assert entries == {"session.disable_cpu_ep_fallback": "1"}
    assert [p[0] for p in built["providers"]] == [TRT], "TensorRT is the only provider requested"


# ---------------------------------------------------------------------------
# verify_trt_fps.py sweep helpers (the module imports torch + the runner)
# ---------------------------------------------------------------------------

verify = pytest.importorskip("verify_trt_fps")


def test_batch_sizes_are_parsed_sorted_unique_and_must_form_a_dynamic_profile():
    assert verify.parse_batch_sizes("8, 1,4,4") == [1, 4, 8]
    for bad in ("", "0,4", "x", "1"):
        with pytest.raises(argparse.ArgumentTypeError):
            verify.parse_batch_sizes(bad)


def test_scaling_is_relative_to_the_smallest_batch():
    rows = [{"batch_size": 1, "fps": 100.0}, {"batch_size": 4, "fps": 250.0}, {"batch_size": 8, "fps": 300.0}]
    assert verify.scaling_vs_first(rows) == {"1": 1.0, "4": 2.5, "8": 3.0}
    assert verify.scaling_vs_first([]) == {}


def test_a_sweep_over_a_static_batch_1_model_points_at_the_converter(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("sweep() checks CUDA before anything else")
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "static.onnx"
    onnx.save(model, str(path))
    args = argparse.Namespace(model=str(path), batch_sizes=[1, 4, 8], shape=None, cache_dir=str(tmp_path / "c"),
                              device_id=0, warmup=0, iterations=1)
    with pytest.raises(RuntimeError, match=r"convert_dynamic_shapes\.py .*static\.onnx"):
        verify.sweep(args)


def load_tests(loader, tests, pattern):
    from app.tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
