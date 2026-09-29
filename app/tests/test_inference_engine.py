"""Stage 3: OptimizedInferenceSession, IO binding, FP16 conversion.

Each test here pins something that went wrong while building it:
  * ORT REJECTS '1'/'0' for TRT boolean options and has_user_compute_stream on
    the TRT EP -- and a rejected option does not raise: ORT drops TensorRT AND
    CUDA and retries on CPU. The provider check must catch that.
  * onnxconverter-common names inserted Casts after the consuming node, so a
    graph with unnamed nodes (hififace: 38) converts to an invalid model.
  * keep_io_types re-casts a graph output that the graph ALSO reads internally
    (hififace's `mask`), leaving a float32/float16 type clash.
  * a full FP16 conversion of hififace loads, runs, and returns NaN.
Engines go to the app's real cache root (default_cache_root): a cold
TensorRT build measured ~107 s per engine here (432 s for the four TRT
sessions), a warm load ~6 s. The engines are the ones the app itself builds
with these options, so sharing the cache costs nothing and only a machine's
FIRST run pays. The one test that must not leave engines behind (the rejected
option) uses a temp directory.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

MODELS = APP_DIR / "models"
HIFI = MODELS / "hififace_unofficial_256.onnx"
HYPER = MODELS / "hyperswap_1a_256.onnx"


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _cuda(), reason="no CUDA")


# ── light ──────────────────────────────────────────────────────────────────────

class _Meta:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class TestProfileAndCache:
    def _ie(self):
        pytest.importorskip("onnx")
        from roop.processors.frame import inference_engine
        return inference_engine

    def test_fixed_profile_for_a_symbolic_batch(self):
        ie = self._ie()
        prof = ie._fixed_profile([_Meta("target", ["batch_size", 3, 256, 256]),
                                  _Meta("source", ["batch_size", 512])])
        assert prof == ("target:1x3x256x256,source:1x512",) * 3     # min = opt = max

    def test_no_profile_for_a_static_graph(self):
        ie = self._ie()
        assert ie._fixed_profile([_Meta("source", [1, 512]), _Meta("target", [1, 3, 256, 256])]) is None

    def test_no_profile_for_a_dynamic_spatial_dim(self):
        ie = self._ie()
        assert ie._fixed_profile([_Meta("x", [None, 3, None, None])]) is None

    def test_cache_root_env_and_home(self, monkeypatch, tmp_path):
        ie = self._ie()
        monkeypatch.delenv("ROOP_TRT_CACHE_ROOT", raising=False)
        assert ie.default_cache_root() == Path.home() / ".cache" / "roop-ultimate" / "trt_cache"
        monkeypatch.setenv("ROOP_TRT_CACHE_ROOT", str(tmp_path))
        assert ie.default_cache_root() == tmp_path

    def test_namespace_separates_precision(self):
        pytest.importorskip("onnxruntime")
        ie = self._ie()
        a, b = ie.cache_namespace("fp16"), ie.cache_namespace("fp32")
        assert a != b and a.startswith("fp16_") and "ort" in a
        assert all(c.isalnum() or c in "._-" for c in a)

    def test_argument_validation(self):
        pytest.importorskip("onnxruntime")
        ie = self._ie()
        with pytest.raises(ValueError):
            ie.OptimizedInferenceSession(str(HIFI), provider="rocm")
        with pytest.raises(ValueError):
            ie.OptimizedInferenceSession(str(HIFI), provider="cpu", precision="int8")
        with pytest.raises(FileNotFoundError):
            ie.OptimizedInferenceSession("no/such/model.onnx", provider="cpu")


# ── FP16 conversion on small synthetic graphs ──────────────────────────────────

def _tiny_graph(path, overflow=False):
    """x[1,4] -> (y = x*2 + 1) and (m = sigmoid(y)); z = y - m reads the
    OUTPUT m internally; nodes unnamed. overflow=True adds mean(x*1e3^2),
    which is +inf in float16."""
    import onnx
    from onnx import TensorProto, helper
    nodes = [
        helper.make_node("Mul", ["x", "two"], ["a"]),
        helper.make_node("Add", ["a", "one"], ["y"]),
        helper.make_node("Sigmoid", ["y"], ["m"]),
        helper.make_node("Sub", ["y", "m"], ["z"]),
    ]
    inits = [helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]),
             helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])]
    outs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("m", TensorProto.FLOAT, [1, 4])]
    if overflow:
        nodes += [helper.make_node("Mul", ["x", "big"], ["b"]),
                  helper.make_node("Mul", ["b", "b"], ["b2"]),
                  helper.make_node("ReduceSum", ["b2"], ["s"], keepdims=0)]
        inits.append(helper.make_tensor("big", TensorProto.FLOAT, [], [1000.0]))
        outs.append(helper.make_tensor_value_info("s", TensorProto.FLOAT, []))
    g = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                          outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


class TestConvertFp16:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("onnxconverter_common")
        pytest.importorskip("onnxruntime")
        from roop.processors.frame import inference_engine
        self.ie = inference_engine

    def test_output_read_internally_and_unnamed_nodes(self, tmp_path):
        import onnxruntime as ort
        src = _tiny_graph(tmp_path / "t.onnx")
        dst = self.ie.convert_onnx_fp16(str(src), str(tmp_path / "t16.onnx"), op_block_list=[])
        assert dst.endswith("t16.onnx")
        s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        assert [i.type for i in s.get_inputs()] == ["tensor(float)"]          # IO kept float32
        x = np.array([[0.1, -0.5, 0.7, 2.0]], np.float32)
        z, m = s.run(None, {"x": x})
        y = x * 2 + 1
        np.testing.assert_allclose(m, 1 / (1 + np.exp(-y)), atol=2e-3)
        np.testing.assert_allclose(z, y - 1 / (1 + np.exp(-y)), atol=5e-3)

    def test_overflow_is_refused_not_returned(self, tmp_path):
        src = _tiny_graph(tmp_path / "o.onnx", overflow=True)
        with pytest.raises(ValueError, match="non-finite"):
            self.ie.convert_onnx_fp16(str(src), str(tmp_path / "o16.onnx"), op_block_list=[])

    @pytest.mark.skipif(not HYPER.is_file(), reason="hyperswap absent")
    def test_already_fp16_is_not_converted(self, tmp_path):
        assert self.ie.is_fp16_graph(str(HYPER))
        assert self.ie.convert_onnx_fp16(str(HYPER), str(tmp_path / "x.onnx")) == str(HYPER)
        assert not (tmp_path / "x.onnx").exists()

    @pytest.mark.gpu
    @pytest.mark.skipif(not HIFI.is_file(), reason="hififace absent")
    def test_hififace_converts_finite_and_close(self, tmp_path):
        import onnxruntime as ort
        from roop.processors.frame.inference_engine import _prepare_runtime
        _prepare_runtime()
        dst = self.ie.convert_onnx_fp16(str(HIFI), str(tmp_path / "h16.onnx"))
        rng = np.random.default_rng(0)
        v = rng.normal(size=(1, 512))
        feed = {"target": rng.uniform(-1, 1, (1, 3, 256, 256)).astype(np.float32),
                "source": (v / np.linalg.norm(v)).astype(np.float32)}
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        a = ort.InferenceSession(str(HIFI), providers=prov).run(None, feed)
        b = ort.InferenceSession(dst, providers=prov).run(None, feed)
        for x, y in zip(a, b):
            assert np.isfinite(y).all()
            assert np.abs(x - y).mean() < 0.01        # measured 0.044/255 x 2 on real faces

    def test_default_block_list_is_what_fixed_hififace(self):
        from onnxconverter_common import float16
        assert set(self.ie.FP16_EXTRA_BLOCK) == {"ReduceMean", "Pow", "Sqrt", "Div"}
        assert "ReduceMean" not in float16.DEFAULT_OP_BLOCK_LIST      # else the extra is moot


# ── sessions on the GPU ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cache_root():
    pytest.importorskip("onnx")
    from roop.processors.frame.inference_engine import default_cache_root
    return default_cache_root()


def _feed(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(1, 512))
    return {"target": rng.uniform(-1, 1, (1, 3, 256, 256)).astype(np.float32),
            "source": (v / np.linalg.norm(v)).astype(np.float32)}


@pytest.fixture(scope="module")
def sessions(cache_root):
    if not (_cuda() and HIFI.is_file() and HYPER.is_file()):
        pytest.skip("needs CUDA and both model files")
    from roop.processors.frame.inference_engine import OptimizedInferenceSession
    out = {}
    for tag, path in (("hifi", HIFI), ("hyper", HYPER)):
        for prov, prec in (("cuda", "fp32"), ("tensorrt", "fp16"), ("tensorrt", "fp32")):
            out[(tag, prov, prec)] = OptimizedInferenceSession(str(path), prov, prec, strict=True,
                                                               cache_root=cache_root)
    return out


@pytest.mark.gpu
class TestOptimizedSession:
    def test_every_arm_on_its_requested_provider(self, sessions):
        for (tag, prov, prec), s in sessions.items():
            assert s.active_provider == prov, (tag, prov, prec)
            assert s.session.get_providers()[0].lower().startswith(prov[:4])

    def test_trt_options(self, sessions):
        hifi = dict(sessions[("hifi", "tensorrt", "fp16")].providers[0][1])
        assert hifi["trt_fp16_enable"] == "True" and hifi["trt_engine_cache_enable"] == "True"
        assert hifi["trt_profile_min_shapes"] == hifi["trt_profile_max_shapes"] \
            == "target:1x3x256x256,source:1x512"
        hyper = dict(sessions[("hyper", "tensorrt", "fp32")].providers[0][1])
        assert hyper["trt_fp16_enable"] == "False"
        assert "trt_profile_min_shapes" not in hyper          # static graph

    def test_engines_cached_per_precision(self, sessions, cache_root):
        a = sessions[("hifi", "tensorrt", "fp16")].cache_dir
        b = sessions[("hifi", "tensorrt", "fp32")].cache_dir
        assert a != b and a.parent == b.parent == Path(cache_root)
        assert list(a.glob("*.engine")) and list(b.glob("*.engine"))

    def test_run_equals_run_binding(self, sessions):
        import torch
        feed = _feed(1)
        for key, s in sessions.items():
            host = s.run(feed)[0]
            dev = s.run_binding({k: torch.as_tensor(v, device="cuda") for k, v in feed.items()})
            np.testing.assert_array_equal(host, dev.cpu().numpy(), err_msg=str(key))

    def test_zero_copy_binds_the_callers_pointer(self, sessions):
        import torch
        s = sessions[("hifi", "tensorrt", "fp16")]
        t = {k: torch.as_tensor(v, device="cuda") for k, v in _feed(2).items()}
        out1 = s.run_binding(t)
        for name, tensor in t.items():
            assert s._bound_inputs[name] == tensor.data_ptr()      # no staging copy
        out2 = s.run_binding(t)
        assert out1.data_ptr() == out2.data_ptr()                  # persistent output buffer
        kept = s.run_binding(t, clone=True)
        assert kept.data_ptr() != out2.data_ptr()

    def test_non_contiguous_input_is_staged_on_device(self, sessions):
        import torch
        s = sessions[("hyper", "tensorrt", "fp16")]
        feed = _feed(3)
        want = s.run(feed)[0]
        nhwc = torch.as_tensor(feed["target"], device="cuda").permute(0, 2, 3, 1).contiguous()
        view = nhwc.permute(0, 3, 1, 2)                            # right shape, not contiguous
        assert not view.is_contiguous()
        got = s.run_binding({"target": view, "source": torch.as_tensor(feed["source"], device="cuda")})
        assert s._bound_inputs["target"] == s._staging["target"].data_ptr()
        np.testing.assert_array_equal(got.cpu().numpy(), want)

    def test_inputs_written_by_torch_kernels_are_seen(self, sessions):
        # stream ordering: ORT runs on its own stream, which must wait for the
        # kernel that produced the input on the caller's stream
        import torch
        s = sessions[("hifi", "cuda", "fp32")]
        g = torch.Generator(device="cuda").manual_seed(0)
        for _ in range(5):
            target = torch.rand((1, 3, 256, 256), device="cuda", generator=g) * 2 - 1
            source = torch.nn.functional.normalize(torch.randn((1, 512), device="cuda", generator=g), dim=1)
            got = s.run_binding({"target": target, "source": source}, clone=True)
            want = s.run({"target": target.cpu().numpy(), "source": source.cpu().numpy()})[0]
            np.testing.assert_array_equal(got.cpu().numpy(), want)

    def test_input_errors(self, sessions):
        import torch
        s = sessions[("hifi", "cuda", "fp32")]
        with pytest.raises(ValueError, match="fixed"):
            s.run_binding({"target": torch.zeros((2, 3, 256, 256), device="cuda"),
                           "source": torch.zeros((2, 512), device="cuda")})
        with pytest.raises(ValueError, match="missing"):
            s.run_binding({"target": torch.zeros((1, 3, 256, 256), device="cuda")})
        with pytest.raises(TypeError):
            s.run_binding({"target": np.zeros((1, 3, 256, 256), np.float32),
                           "source": torch.zeros((1, 512), device="cuda")})

    def test_rejected_option_cpu_fallback_is_caught(self, tmp_path, monkeypatch):
        cache_root = tmp_path
        # The trap this module hit: a bad option value makes ORT drop TRT AND
        # CUDA without raising. strict must turn that into an error.
        from roop.processors.frame import inference_engine as ie
        real = ie.OptimizedInferenceSession._providers

        def bad(self, provider, model_arg):
            chain = real(self, provider, model_arg)
            name, opts = chain[0]
            return [(name, dict(opts, trt_fp16_enable="1"))] + chain[1:]
        monkeypatch.setattr(ie.OptimizedInferenceSession, "_providers", bad)
        with pytest.raises(ie.ProviderFallbackError, match="running on cpu"):
            ie.OptimizedInferenceSession(str(HYPER), "tensorrt", "fp16", strict=True, cache_root=cache_root)
        loose = ie.OptimizedInferenceSession(str(HYPER), "tensorrt", "fp16", strict=False,
                                             cache_root=cache_root)
        assert loose.active_provider == "cpu"
        with pytest.raises(RuntimeError, match="running on cpu"):
            import torch
            loose.run_binding({"source": torch.zeros((1, 512), device="cuda"),
                               "target": torch.zeros((1, 3, 256, 256), device="cuda")})


# ── the swappers on the engine ─────────────────────────────────────────────────

@pytest.mark.gpu
@needs_cuda
class TestSwapperOnEngine:
    @pytest.mark.parametrize("name", ["hififace_256", "hyperswap_1a_256"])
    def test_torch_infer_matches_numpy_infer(self, name, cache_root, monkeypatch):
        import torch
        from roop.processors.frame import model_registry as mr
        monkeypatch.setenv("ROOP_TRT_CACHE_ROOT", str(cache_root))
        sw = mr.create_swapper(name)
        try:
            sw.initialize_session(None, "tensorrt", precision="fp16", strict=True)
        except FileNotFoundError:
            pytest.skip("model files absent")
        assert sw.engine.active_provider == "tensorrt"
        crop = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        feed = sw.pre_process(crop, np.random.default_rng(1).normal(0, 20, 512).astype(np.float32))
        host = sw.infer(feed)
        host_mask = sw.last_mask
        dev = sw.infer({k: torch.as_tensor(v, device="cuda") for k, v in feed.items()})
        assert torch.is_tensor(dev) and dev.is_cuda
        np.testing.assert_array_equal(dev.cpu().numpy(), host)
        np.testing.assert_array_equal(sw.last_mask.cpu().numpy(), host_mask)

    def test_torch_input_needs_a_gpu_session(self):
        import torch
        from roop.processors.frame import model_registry as mr
        sw = mr.create_swapper("hyperswap_1a_256")
        try:
            sw.initialize_session(None, "cpu")
        except FileNotFoundError:
            pytest.skip("model absent")
        with pytest.raises(RuntimeError, match="cuda"):
            sw.infer({sw.image_input_name: torch.zeros((1, 3, 256, 256), device="cuda"),
                      sw.embed_input_name: torch.zeros((1, 512), device="cuda")})

    def test_validate_tensor_torch(self):
        import torch
        from roop.processors.frame.swapper_base import TensorSpec, validate_tensor_torch
        spec = TensorSpec("target", ("b", 3, 256, 256))
        with pytest.raises(ValueError):
            validate_tensor_torch(torch.zeros((1, 3, 255, 256), device="cuda"), spec)
        with pytest.raises(ValueError):
            validate_tensor_torch(torch.zeros((1, 3, 256, 256), device="cuda", dtype=torch.float64), spec)
        with pytest.raises(ValueError, match="CUDA"):
            validate_tensor_torch(torch.zeros((1, 3, 256, 256)), spec)
        v = torch.zeros((1, 256, 256, 3), device="cuda").permute(0, 3, 1, 2)
        assert validate_tensor_torch(v, spec).is_contiguous()
