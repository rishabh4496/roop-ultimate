"""roop.onnx_cuda_safe: the asymmetric-ConvTranspose rewrite is exact, cached, and scoped.

Real ``onnx`` / ``onnxruntime`` (CPU provider) prove the rewrite is output-identical to
the original graph; the factory wiring is checked with the fake ORT from
``test_strict_session`` so no GPU is needed. The 6.8x speed-up itself was measured on
an RTX 4070 (see CUDA_EXECUTION_POLICY.md) and is not asserted here.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from roop import onnx_cuda_safe, strict_session  # noqa: E402
from roop.strict_session import CPU_EP, CUDA_EP, TENSORRT_EP  # noqa: E402


def _model(pads=None, auto_pad=None, output_shape=None, stride=2, kernel=3, name="ct"):
    rng = np.random.default_rng(7)
    w = rng.standard_normal((4, 3, kernel, kernel)).astype(np.float32)   # (Cin, Cout, kH, kW)
    b = rng.standard_normal((3,)).astype(np.float32)
    attrs = {"strides": [stride, stride], "kernel_shape": [kernel, kernel]}
    if pads is not None:
        attrs["pads"] = pads
    if auto_pad is not None:
        attrs["auto_pad"] = auto_pad
    if output_shape is not None:
        attrs["output_shape"] = output_shape
    node = helper.make_node("ConvTranspose", ["x", "w", "b"], ["y"], name=name, **attrs)
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def _run(model):
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).standard_normal((1, 4, 8, 8)).astype(np.float32)
    return s.run(None, {"x": x})[0]


def _ops(model):
    return [n.op_type for n in model.graph.node]


@pytest.mark.parametrize("pads", [[0, 0, 1, 1], [1, 0, 0, 1], [1, 2, 2, 1], [0, 1, 1, 0]])
def test_rewrite_is_output_identical(pads):
    original = _model(pads=pads)
    rewritten = _model(pads=pads)
    assert onnx_cuda_safe.rewrite_model(rewritten) == 1
    assert _ops(rewritten) == ["ConvTranspose", "Slice"]
    ct = rewritten.graph.node[0]
    assert list({a.name: a for a in ct.attribute}["pads"].ints) == [0, 0, 0, 0]
    a, b = _run(original), _run(rewritten)
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)          # a crop moves no values: bit-exact on the same provider
    onnx.checker.check_model(rewritten)


@pytest.mark.parametrize("kwargs", [
    {"pads": [1, 1, 1, 1]},                           # symmetric: nothing to do
    {"pads": None},                                   # no pads attribute
    {"auto_pad": "SAME_UPPER"},                       # not explicit
    {"pads": [0, 0, 1, 1], "output_shape": [16, 16]}, # output_shape wins over pads; leave alone
])
def test_rewrite_leaves_other_nodes_alone(kwargs):
    m = _model(**kwargs)
    assert onnx_cuda_safe.rewrite_model(m) == 0
    assert _ops(m) == ["ConvTranspose"]


def test_cache_hit_does_not_reload_the_model(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / "net.onnx")
    onnx.save(_model(pads=[0, 0, 1, 1]), path)

    first = onnx_cuda_safe.cuda_safe_model_path(path)
    assert first != path and os.path.dirname(first).endswith(".cuda_safe") and os.path.isfile(first)
    assert "[CudaSafeONNX] net.onnx: 1 asymmetric" in capsys.readouterr().out
    np.testing.assert_array_equal(_run(onnx.load(path)), _run(onnx.load(first)))

    def boom(*a, **k):
        raise AssertionError("cached model must not be re-read")
    monkeypatch.setattr(onnx, "load", boom)
    assert onnx_cuda_safe.cuda_safe_model_path(path) == first


def test_model_with_nothing_to_rewrite_is_not_copied_and_is_cached(tmp_path, monkeypatch):
    path = str(tmp_path / "plain.onnx")
    onnx.save(_model(pads=[1, 1, 1, 1]), path)
    assert onnx_cuda_safe.cuda_safe_model_path(path) == path
    assert not [f for f in os.listdir(tmp_path / ".cuda_safe") if f.endswith(".onnx")]
    monkeypatch.setattr(onnx, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rescan")))
    assert onnx_cuda_safe.cuda_safe_model_path(path) == path


def test_changed_file_is_rescanned(tmp_path):
    path = str(tmp_path / "net.onnx")
    onnx.save(_model(pads=[1, 1, 1, 1]), path)
    assert onnx_cuda_safe.cuda_safe_model_path(path) == path
    onnx.save(_model(pads=[0, 0, 1, 1]), path)            # new size/mtime: stamp differs
    os.utime(path, ns=(1, 2_000_000_000))
    assert onnx_cuda_safe.cuda_safe_model_path(path) != path


def test_unreadable_model_falls_back_to_original(tmp_path):
    path = str(tmp_path / "broken.onnx")
    with open(path, "wb") as fh:
        fh.write(b"not a protobuf")
    assert onnx_cuda_safe.cuda_safe_model_path(path) == path


def test_kill_switch(tmp_path, monkeypatch):
    path = str(tmp_path / "net.onnx")
    onnx.save(_model(pads=[0, 0, 1, 1]), path)
    monkeypatch.setenv("ROOP_CUDA_SAFE_ONNX", "0")
    assert onnx_cuda_safe.cuda_safe_model_path(path) == path
    assert not os.path.exists(tmp_path / ".cuda_safe")


# --------------------------------------------------------------------------
# factory wiring (fake ORT, no GPU)
# --------------------------------------------------------------------------

class _FakeSession:
    def __init__(self, model, options, providers):
        self.model, self.providers = model, providers

    def get_providers(self):
        return [p[0] if isinstance(p, tuple) else p for p in self.providers] + [CPU_EP]


class _FakeOrt(types.ModuleType):
    __version__ = "0.0-fake"

    def __init__(self):
        super().__init__("onnxruntime")
        self.models = []

    def get_available_providers(self):
        return [TENSORRT_EP, CUDA_EP, CPU_EP]

    def InferenceSession(self, model, options=None, providers=None):  # noqa: N802
        self.models.append(model)
        return _FakeSession(model, options, providers)


@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    from roop import predictor
    fake = _FakeOrt()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(strict_session, "_device_summary", lambda device_id: "FakeGPU")
    monkeypatch.setattr(onnx_cuda_safe, "cuda_safe_model_path", lambda p: p + ".safe")
    monkeypatch.delenv("ROOP_STRICT_PROVIDER", raising=False)
    strict_session.reset_announcements()
    predictor.reset()
    yield fake
    strict_session.reset_announcements()
    predictor.reset()


def test_cuda_only_chain_gets_the_rewritten_model(fake_ort):
    strict_session.create_strictly_accelerated_session(
        "m/face_occluder.onnx", providers=[(CUDA_EP, {"device_id": 0}), CPU_EP], sess_options=object(), tag="t1")
    assert fake_ort.models == ["m/face_occluder.onnx.safe"]


def test_tensorrt_chain_keeps_the_original_model(fake_ort):
    strict_session.create_strictly_accelerated_session(
        "m/face_occluder.onnx", providers=[(TENSORRT_EP, {"device_id": 0}), (CUDA_EP, {"device_id": 0}), CPU_EP],
        sess_options=object(), tag="t2")
    assert fake_ort.models == ["m/face_occluder.onnx"]


def test_in_memory_model_is_never_rewritten(fake_ort):
    strict_session.create_strictly_accelerated_session(
        b"bytes", providers=[(CUDA_EP, {"device_id": 0}), CPU_EP], sess_options=object(), tag="t3")
    assert fake_ort.models == [b"bytes"]
