"""roop.trt_build_guard: torch's idle VRAM is released immediately before and after a TensorRT build or canary run.

Each test drives the real call site (the strict session factory, the canary entry, the context manager) with a
fake torch and a fake onnxruntime that write to ONE shared event log, so the assertion is about ORDER: the
release happens before the build starts and after it ends, not "somewhere in the same function".
"""
from __future__ import annotations

import sys
import types

import pytest

from roop import predictor, strict_session
from roop.strict_session import CUDA_EP, CPU_EP, TENSORRT_EP, create_strictly_accelerated_session
from roop.trt_build_guard import around_trt_build, release_torch_cache

TRT_OPTS = {"device_id": 0, "trt_max_workspace_size": 2 * 1024 ** 3, "trt_engine_cache_path": "models/trt_cache/mixed"}
CUDA_OPTS = {"device_id": 0}


def _fake_torch(events, *, available=True, initialized=True, boom=None):
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: available
    cuda.is_initialized = lambda: initialized

    def empty_cache():
        if boom:
            raise boom
        events.append("empty_cache")
    cuda.empty_cache = empty_cache
    torch.cuda = cuda
    return torch


class _Session:
    def __init__(self, active):
        self._active = active

    def get_providers(self):
        return list(self._active)


def _fake_ort(events, *, fail=None):
    ort = types.ModuleType("onnxruntime")
    ort.__version__ = "0.0-fake"
    ort.get_available_providers = lambda: [TENSORRT_EP, CUDA_EP, CPU_EP]

    def InferenceSession(model, options=None, providers=None):  # noqa: N802
        events.append("build:start")
        if fail:
            events.append("build:fail")
            raise fail
        events.append("build:end")
        first = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        return _Session([first, CPU_EP])
    ort.InferenceSession = InferenceSession
    return ort


@pytest.fixture
def events(monkeypatch):
    log = []
    monkeypatch.delenv("ROOP_STRICT_PROVIDER", raising=False)
    monkeypatch.setattr(strict_session, "_device_summary", lambda device_id: "FakeGPU")
    strict_session.reset_announcements()
    predictor.reset()
    yield log
    strict_session.reset_announcements()
    predictor.reset()


class _Opts:
    def add_session_config_entry(self, key, value):
        pass


def test_release_runs_before_and_after_a_tensorrt_session_build(events, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(events))
    create_strictly_accelerated_session("models/scrfd.onnx", providers=[(TENSORRT_EP, TRT_OPTS), CPU_EP],
                                        sess_options=_Opts())
    assert events == ["empty_cache", "build:start", "build:end", "empty_cache"]


def test_release_runs_after_a_build_that_fails(events, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(events, fail=RuntimeError("OutOfMemory")))
    with pytest.raises(RuntimeError, match="OutOfMemory"):
        create_strictly_accelerated_session("models/scrfd.onnx", providers=[(TENSORRT_EP, TRT_OPTS), CPU_EP],
                                            sess_options=_Opts())
    assert events == ["empty_cache", "build:start", "build:fail", "empty_cache"]


def test_a_plain_cuda_session_does_not_pay_the_device_sync(events, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(events))
    create_strictly_accelerated_session("models/scrfd.onnx", providers=[(CUDA_EP, CUDA_OPTS), CPU_EP],
                                        sess_options=_Opts())
    assert events == ["build:start", "build:end"]


def test_canary_run_is_bracketed(events, monkeypatch):
    from roop import aux_canary, pipeline_status
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))

    def fake_check(stem, session, model_file, reference_factory=None):
        events.append("canary:body")
        return aux_canary.CanaryResult("aux:" + stem, None, "stub")
    monkeypatch.setattr(aux_canary, "_check_session", fake_check)
    monkeypatch.setattr(pipeline_status, "session_precision", lambda session: "fp16")
    aux_canary.check_session("xseg", object(), "models/xseg.onnx")
    assert events == ["empty_cache", "canary:body", "empty_cache"]


def test_native_engine_serialisation_is_bracketed(events, monkeypatch):
    from roop import trt_graph_surgery
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))

    class Builder:
        def build_serialized_network(self, network, config):
            events.append("build")
            return b"engine"
    assert trt_graph_surgery._serialize(Builder(), object(), object()) == b"engine"
    assert events == ["empty_cache", "build", "empty_cache"]


def test_no_torch_loaded_means_nothing_is_imported_or_released(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert release_torch_cache("x") is False
    assert "torch" not in sys.modules


@pytest.mark.parametrize("kw", [{"available": False}, {"initialized": False}])
def test_no_cuda_context_means_no_release(monkeypatch, kw):
    events = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events, **kw))
    assert release_torch_cache("x") is False
    assert events == []


def test_a_failing_release_never_breaks_the_build(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(events, boom=RuntimeError("driver gone")))
    with around_trt_build("t"):
        events.append("body")
    assert events == ["body"]


def load_tests(loader, tests, pattern):
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
