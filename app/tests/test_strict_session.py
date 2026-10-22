"""roop.strict_session: a GPU session is on the GPU or it does not exist.

The factory is exercised against a fake ``onnxruntime`` injected into
``sys.modules`` so the whole contract -- no CPU rung, hard errors, the assertion
on the BUILT session, the confirmation line -- is checked in the light CI
profile. ``tests/test_gpu_bindings.py`` covers the real library.
"""
from __future__ import annotations

import sys
import types

import pytest

from roop import predictor, strict_session
from roop.strict_session import (CPU_EP, CUDA_EP, TENSORRT_EP, StrictSessionError,
                                 create_strictly_accelerated_session, without_cpu)

CUDA_OPTS = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested",
             "gpu_mem_limit": 10 * 1024 ** 3, "cudnn_conv_algo_search": "HEURISTIC"}
TRT_OPTS = {"device_id": 0, "trt_max_workspace_size": 2 * 1024 ** 3,
            "trt_fp16_enable": True, "trt_engine_cache_path": "models/trt_cache/mixed"}


class FakeOptions:
    def __init__(self):
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class FakeSession:
    def __init__(self, model, options, providers, active):
        self.model, self.options, self.requested = model, options, providers
        self._active = active

    def get_providers(self):
        return list(self._active)


class FakeOrt(types.ModuleType):
    """Records every construction attempt; ``plan`` decides what each one does."""

    __version__ = "0.0-fake"

    def __init__(self, available, plan=None):
        super().__init__("onnxruntime")
        self.available = available
        self.attempts = []
        self.plan = plan or (lambda chain: [_name(chain[0]), CPU_EP])

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, model, options=None, providers=None):  # noqa: N802
        self.attempts.append(list(providers))
        outcome = self.plan(list(providers))
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeSession(model, options, providers, outcome)


def _name(entry):
    return entry[0] if isinstance(entry, tuple) else entry


@pytest.fixture
def ort(monkeypatch):
    def install(available=(TENSORRT_EP, CUDA_EP, CPU_EP), plan=None):
        fake = FakeOrt(available, plan)
        monkeypatch.setitem(sys.modules, "onnxruntime", fake)
        return fake
    monkeypatch.delenv("ROOP_STRICT_PROVIDER", raising=False)
    monkeypatch.setattr(strict_session, "_device_summary", lambda device_id: "FakeGPU (cc 8.9, 12000 MiB)")
    strict_session.reset_announcements()
    predictor.reset()
    yield install
    strict_session.reset_announcements()
    predictor.reset()


OPTS = FakeOptions()


def build(chain, **kw):
    return create_strictly_accelerated_session(
        "models/xseg.onnx", providers=chain, sess_options=kw.pop("sess_options", OPTS), **kw)


def test_cpu_is_removed_from_a_gpu_chain(ort):
    fake = ort()
    session = build([(CUDA_EP, CUDA_OPTS), CPU_EP])
    assert fake.attempts == [[(CUDA_EP, CUDA_OPTS)]]
    assert session.get_providers()[0] == CUDA_EP


def test_options_the_caller_resolved_are_passed_through_untouched(ort):
    """The factory enforces; it does not re-decide cuDNN algo, arena or TensorRT build options."""
    fake = ort()
    build([(TENSORRT_EP, TRT_OPTS), (CUDA_EP, CUDA_OPTS), CPU_EP])
    trt, cuda = fake.attempts[0]
    assert trt == (TENSORRT_EP, TRT_OPTS) and trt[1] is TRT_OPTS
    assert cuda == (CUDA_EP, CUDA_OPTS) and cuda[1] is CUDA_OPTS


def test_a_build_failure_is_raised_once_with_no_retry_ladder(ort):
    """Loaders run their own policy on this exception (the swapper's FP32 rebuild
    sets _trt_disabled and rebuilds its pool); a silent step down in here would hand
    them a CUDA session they believe is TensorRT."""
    fake = ort(plan=lambda chain: RuntimeError("engine build failed"))
    with pytest.raises(RuntimeError, match="engine build failed"):
        build([(TENSORRT_EP, TRT_OPTS), (CUDA_EP, CUDA_OPTS), CPU_EP])
    assert [[_name(p) for p in attempt] for attempt in fake.attempts] == [[TENSORRT_EP, CUDA_EP]]


def test_cpu_is_never_in_any_attempt(ort):
    fake = ort(plan=lambda chain: RuntimeError("CUDA failure"))
    for chain in ([(CUDA_EP, CUDA_OPTS), CPU_EP], [(TENSORRT_EP, TRT_OPTS), CPU_EP]):
        with pytest.raises(RuntimeError):
            build(chain)
    assert fake.attempts
    assert all(CPU_EP not in [_name(p) for p in attempt] for attempt in fake.attempts)


def test_missing_cuda_provider_is_a_hard_error_with_instructions(ort):
    fake = ort(available=(CPU_EP,))
    with pytest.raises(StrictSessionError) as caught:
        build([(CUDA_EP, CUDA_OPTS), CPU_EP], tag="mask:xseg")
    message = str(caught.value)
    assert isinstance(caught.value, RuntimeError)
    assert "mask:xseg" in message and CUDA_EP in message
    assert "onnxruntime-gpu" in message and "provision_runtime.py" in message
    assert "available providers : CPUExecutionProvider" in message
    assert fake.attempts == [], "no session may be built when the provider is missing"


def test_a_session_that_came_up_cpu_only_is_rejected(ort):
    """ORT's silent drop: construction succeeds, the GPU provider is simply not active."""
    ort(plan=lambda chain: [CPU_EP])
    with pytest.raises(predictor.ProviderAssertionError, match="CPU-only"):
        build([(CUDA_EP, CUDA_OPTS), CPU_EP])


def test_missing_tensorrt_ep_uses_cuda_and_says_so(ort, capsys):
    fake = ort(available=(CUDA_EP, CPU_EP))
    build([(TENSORRT_EP, TRT_OPTS), (CUDA_EP, CUDA_OPTS), CPU_EP])
    assert [_name(p) for p in fake.attempts[0]] == [CUDA_EP]
    assert "TensorrtExecutionProvider is not available" in capsys.readouterr().out


def test_a_chain_with_no_gpu_provider_is_a_configuration_not_a_fallback(ort, capsys):
    fake = ort(plan=lambda chain: [CPU_EP])
    session = build([CPU_EP], tag="mask:xseg")
    assert fake.attempts == [[CPU_EP]]
    assert session.get_providers() == [CPU_EP]
    assert "by configuration" in capsys.readouterr().out


def test_asking_for_cuda_by_name_never_builds_a_cpu_session(ort, monkeypatch):
    fake = ort()
    monkeypatch.setattr(strict_session, "_resolve_chain", lambda provider: [CPU_EP])
    with pytest.raises(StrictSessionError, match="holds no CUDA/TensorRT"):
        create_strictly_accelerated_session("m.onnx", "cuda", sess_options=OPTS)
    assert fake.attempts == []


def test_provider_name_selects_the_tuned_chain(ort, monkeypatch):
    fake = ort()
    asked = []
    monkeypatch.setattr(strict_session, "_resolve_chain",
                        lambda provider: asked.append(provider) or [(CUDA_EP, CUDA_OPTS), CPU_EP])
    create_strictly_accelerated_session("m.onnx", "cuda", sess_options=OPTS)
    assert asked == ["cuda"] and fake.attempts == [[(CUDA_EP, CUDA_OPTS)]]


def test_unknown_provider_name_is_rejected():
    with pytest.raises(ValueError, match="'cuda' or 'tensorrt'"):
        strict_session._resolve_chain("directml")


def test_confirmation_line_names_provider_device_and_memory_limits_once(ort, capsys):
    ort()
    build([(CUDA_EP, CUDA_OPTS), CPU_EP], tag="mask:xseg")
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("[Session] mask:xseg"))
    assert "active=CUDAExecutionProvider" in line
    assert "FakeGPU (cc 8.9, 12000 MiB)" in line
    assert "arena=kSameAsRequested" in line and "gpu_mem_limit=10.0 GiB" in line
    assert "model=xseg.onnx" in line
    build([(CUDA_EP, CUDA_OPTS), CPU_EP], tag="mask:xseg")
    assert "[Session] mask:xseg" not in capsys.readouterr().out, "one line per tag"


def test_confirmation_line_reports_tensorrt_workspace_and_cache(ort, capsys):
    ort(plan=lambda chain: [TENSORRT_EP, CUDA_EP, CPU_EP])
    build([(TENSORRT_EP, TRT_OPTS), (CUDA_EP, CUDA_OPTS), CPU_EP], tag="swapper:x")
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("[Session] swapper:x"))
    assert "active=TensorrtExecutionProvider" in line
    assert "trt_workspace=2.00 GiB" in line and "trt_fp16=True" in line
    assert "engine_cache=mixed" in line


def test_a_model_passed_as_bytes_needs_a_tag_and_is_not_stringified(ort):
    fake = ort()
    with pytest.raises(ValueError, match="tag is required"):
        create_strictly_accelerated_session(
            b"\x08\x07", providers=[(CUDA_EP, CUDA_OPTS)], sess_options=OPTS)
    session = create_strictly_accelerated_session(
        b"\x08\x07", providers=[(CUDA_EP, CUDA_OPTS)], sess_options=OPTS, tag="swapper:bytes")
    assert session.model == b"\x08\x07" and fake.attempts


def test_forbid_cpu_nodes_sets_the_ort_config_entry(ort):
    ort()
    options = FakeOptions()
    build([(CUDA_EP, CUDA_OPTS)], sess_options=options, forbid_cpu_nodes=True)
    assert options.entries == {"session.disable_cpu_ep_fallback": "1"}
    plain = FakeOptions()
    build([(CUDA_EP, CUDA_OPTS)], sess_options=plain, tag="other")
    assert plain.entries == {}


def test_without_cpu_handles_bare_names_and_tuples():
    assert without_cpu([(CUDA_EP, {}), CPU_EP, (CPU_EP, {})]) == [(CUDA_EP, {})]


def test_native_runner_loaded_model_excludes_ort_tensorrt(ort):
    fake = ort()
    from roop.trt_native_runner import register_native_loaded, unregister_native_loaded, is_native_loaded
    register_native_loaded("xseg.onnx")
    try:
        session = build([(TENSORRT_EP, TRT_OPTS), (CUDA_EP, CUDA_OPTS), CPU_EP])
        attempted = [_name(p) for p in fake.attempts[-1]]
        assert TENSORRT_EP not in attempted
        assert CUDA_EP in attempted
        assert session.get_providers()[0] == CUDA_EP
    finally:
        unregister_native_loaded("xseg.onnx")


def test_native_runner_alias_normalization_and_backend_fallback():
    from roop.trt_native_runner import register_native_loaded, unregister_native_loaded, is_native_loaded
    from roop.backend_manager import build_session_with_fallback

    engine_mock_path = "models/restoreformer_plus_plus/engines/isl_q_enchigh.engine"
    register_native_loaded(engine_mock_path)
    try:
        assert is_native_loaded("isl_q_enchigh")
        assert is_native_loaded("restoreformer++")
        assert is_native_loaded("restoreformer_plus_plus")
        assert is_native_loaded("restoreformer_single_output")
        assert is_native_loaded("enhancer:restoreformer++")
        assert not is_native_loaded("inswapper_128")

        # Test backend_manager.build_session_with_fallback excludes TensorRT
        calls = []
        def fake_builder(chain):
            calls.append(list(chain))
            class S:
                def get_providers(self):
                    return [_name(chain[0])]
            return S()

        sess, used = build_session_with_fallback(
            fake_builder,
            [("TensorrtExecutionProvider", {}), ("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
            tag="restoreformer++"
        )
        assert not any("tensorrt" in str(_name(p)).lower() for p in used)
        assert any("cuda" in str(_name(p)).lower() for p in used)
    finally:
        unregister_native_loaded(engine_mock_path)
        assert not is_native_loaded("restoreformer++")




# ---------------------------------------------------------------------------
# Real onnxruntime, real GPU. One fresh interpreter per test so DLL registration
# is exactly run.py's (the helpers are lifted out of run.py without running it).
# ---------------------------------------------------------------------------

_REAL = r'''
import ast, json, os, sys, tempfile
app, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, app); os.chdir(app)
tree = ast.parse(open(os.path.join(app, "run.py"), encoding="utf-8").read())
want = {"_NVIDIA_DLL_PACKAGES", "_GPU_DLL_HANDLES", "_gpu_runtime_dirs", "_register_gpu_runtime_dirs"}
body = [n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in want)
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in want for t in n.targets))]
ns = {"os": os, "sys": sys}
exec(compile(ast.Module(body=body, type_ignores=[]), "run.py", "exec"), ns)
ns["_register_gpu_runtime_dirs"]()
import onnxruntime as ort
from onnx import TensorProto, helper
graph = helper.make_graph([helper.make_node("Add", ["x", "x"], ["y"])], "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]); model.ir_version = 8
path = os.path.join(tempfile.mkdtemp(), "tiny.onnx"); open(path, "wb").write(model.SerializeToString())
from roop.strict_session import create_strictly_accelerated_session
cuda = ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 10 * 1024 ** 3, "do_copy_in_default_stream": True})
out = {}
if mode == "healthy":
    s = create_strictly_accelerated_session(path, providers=[cuda, "CPUExecutionProvider"], tag="probe:healthy")
    out["active"] = s.get_providers()
else:
    bad = [("TensorrtExecutionProvider", {"device_id": 0, "trt_fp16_enable": "1"}), cuda, "CPUExecutionProvider"]
    try:
        out["raw_active"] = ort.InferenceSession(path, providers=bad).get_providers()
    except Exception as exc:
        out["raw_raised"] = type(exc).__name__
    try:
        out["factory_active"] = create_strictly_accelerated_session(path, providers=bad, tag="probe:bad").get_providers()
    except Exception as exc:
        out["factory_raised"] = type(exc).__name__
print("@@" + json.dumps(out))
'''


def _real(mode: str) -> dict:
    import json
    import subprocess
    from pathlib import Path
    app = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-I", "-c", _REAL, str(app), mode],
                          capture_output=True, text=True, timeout=300)
    lines = [l for l in done.stdout.splitlines() if l.startswith("@@")]
    assert lines, "child failed (exit %s)\n%s\n%s" % (done.returncode, done.stdout[-2000:], done.stderr[-2000:])
    result = json.loads(lines[-1][2:])
    result["stdout"] = done.stdout
    return result


@pytest.mark.gpu
class TestRealOnnxRuntime:
    def test_a_healthy_chain_is_on_cuda_and_announces_device_and_limits(self):
        result = _real("healthy")
        assert result["active"][0] == "CUDAExecutionProvider"
        line = next(l for l in result["stdout"].splitlines() if l.startswith("[Session] probe:healthy"))
        assert "active=CUDAExecutionProvider" in line and "gpu_mem_limit=10.0 GiB" in line
        assert "arena=kSameAsRequested" in line and "NVIDIA" in line

    def test_a_rejected_tensorrt_option_is_an_error_where_raw_ort_returns_a_cpu_session(self):
        """ORT 1.23 rejects trt_fp16_enable='1', drops TensorRT AND CUDA, and returns a CPU session."""
        result = _real("rejected-option")
        assert "factory_active" not in result, (
            "the factory returned a session on %s" % result.get("factory_active"))
        assert result.get("factory_raised"), result


# ---------------------------------------------------------------------------
# optimized_processor.create_onnx_session: its last candidate used to be a bare
# [CPUExecutionProvider], which passes its own "GPU was requested but is not
# active" check (that candidate requested no GPU) and was returned.
# ---------------------------------------------------------------------------

def _op_fake(monkeypatch, plan):
    from roop import optimized_processor as op
    fake = FakeOrt((TENSORRT_EP, CUDA_EP, CPU_EP), plan)
    fake.SessionOptions = lambda: types.SimpleNamespace(
        graph_optimization_level=None, intra_op_num_threads=1, inter_op_num_threads=1)
    fake.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)
    monkeypatch.setattr(op, "ort", fake)
    return op, fake


def test_create_onnx_session_never_ends_on_a_bare_cpu_session_after_a_gpu_request(monkeypatch):
    op, fake = _op_fake(monkeypatch, lambda chain: RuntimeError("no GPU today"))
    with pytest.raises(RuntimeError, match="unable to create ONNX Runtime session"):
        op.create_onnx_session("m.onnx", [(CUDA_EP, CUDA_OPTS), CPU_EP])
    assert [CPU_EP] not in [[_name(p) for p in attempt] for attempt in fake.attempts]


def test_create_onnx_session_rejects_a_gpu_request_that_came_up_cpu_only(monkeypatch):
    op, fake = _op_fake(monkeypatch, lambda chain: [CPU_EP])
    with pytest.raises(RuntimeError):
        op.create_onnx_session("m.onnx", [(CUDA_EP, CUDA_OPTS), CPU_EP])


def test_create_onnx_session_still_builds_cpu_when_cpu_was_asked_for(monkeypatch):
    op, fake = _op_fake(monkeypatch, lambda chain: [CPU_EP])
    assert op.create_onnx_session("m.onnx", [CPU_EP]).get_providers() == [CPU_EP]


def load_tests(loader, tests, pattern):
    from app.tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
