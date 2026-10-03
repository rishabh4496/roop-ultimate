"""Runtime tests for face_engine: provider grant, device buffers, caching, registry.

The GPU tests build a tiny ONNX graph in memory, so they need no model
download. They assert what ORT *granted* (``session.get_providers()``), never
what was requested: ORT returns a working CPU session when a GPU provider
fails to load, and a test that only checks "it ran" passes on CPU.

Env:
    FACE_ENGINE_REQUIRE_TRT=1  fail (instead of recording) when TensorRT is not granted.
"""
from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import (
    ExecutionEngine,
    ProviderFallbackError,
    SessionCreationError,
    ShapeProfile,
)
from face_engine.core.registry import (
    ModelRegistry,
    ModelSpec,
    ModelTask,
    ModelUnavailableError,
)
from face_engine.models.zoo import MODEL_ZOO, build_default_registry
from face_engine.utils import downloads
from face_engine.utils.downloads import IntegrityError, download_file, verify_file

HAS_CUDA_EP = Provider.CUDA.value in ort.get_available_providers()
HAS_TRT_EP = Provider.TENSORRT.value in ort.get_available_providers()

_SCALE = np.linspace(0.5, 2.0, 3, dtype=np.float32).reshape(1, 3, 1, 1)
_BIAS = np.float32(0.25)


def _expected(x: np.ndarray) -> np.ndarray:
    return x * _SCALE + _BIAS


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``y = x * scale + bias`` with a dynamic batch axis, opset 17."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 16, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3, 16, 16])
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "scale"], ["scaled"]),
         helper.make_node("Add", ["scaled", "bias"], ["y"])],
        "tiny", [x], [y],
        initializer=[numpy_helper.from_array(_SCALE, "scale"),
                     numpy_helper.from_array(np.array(_BIAS, dtype=np.float32), "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("models") / "tiny.onnx"
    onnx.save(model, str(path))
    return path
@pytest.fixture(scope="module")
def tiny_model_fp16(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``y = x * scale + bias`` in FLOAT16 with a dynamic batch axis, opset 17."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["batch", 3, 16, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, ["batch", 3, 16, 16])
    scale_fp16 = _SCALE.astype(np.float16)
    bias_fp16 = np.array(_BIAS, dtype=np.float16)
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "scale"], ["scaled"]),
         helper.make_node("Add", ["scaled", "bias"], ["y"])],
        "tiny_fp16", [x], [y],
        initializer=[numpy_helper.from_array(scale_fp16, "scale"),
                     numpy_helper.from_array(bias_fp16, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("models") / "tiny_fp16.onnx"
    onnx.save(model, str(path))
    return path



@pytest.fixture()
def config(tmp_path: Path) -> EngineConfig:
    return EngineConfig(cache_dir=tmp_path / "cache", models_dir=tmp_path / "models")


@pytest.fixture()
def engine(config: EngineConfig) -> Iterator[ExecutionEngine]:
    with ExecutionEngine(config) as eng:
        yield eng


def _cfg(config: EngineConfig, **changes: object) -> EngineConfig:
    return config.model_copy(update=changes)


# ---------------------------------------------------------------- provider grant + buffers
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA_EP, reason="onnxruntime build has no CUDAExecutionProvider")
def test_cuda_session_initializes_and_allocates_device_buffers(
        tiny_model: Path, config: EngineConfig) -> None:
    strict = _cfg(config, providers=[Provider.CUDA, Provider.CPU], strict=True)
    with ExecutionEngine(strict) as engine:
        handle = engine.get_session(tiny_model)
        assert handle.primary_provider == Provider.CUDA.value, handle.granted
        assert not handle.fell_back and handle.on_gpu

        options = handle.session.get_provider_options()[Provider.CUDA.value]
        assert options["arena_extend_strategy"] == "kNextPowerOfTwo"
        assert options["cudnn_conv_algo_search"] == "HEURISTIC"
        assert options["use_tf32"] == "0"
        # The limit is derived from the VRAM free when the session was BUILT;
        # compare with what the session recorded, not a fresh query (other
        # processes allocate in between).
        limit = int(options["gpu_mem_limit"])
        assert limit == handle.provider_options[Provider.CUDA.value]["gpu_mem_limit"]
        from face_engine.core.execution import query_vram
        vram = query_vram(strict.device_id)
        assert vram is not None
        assert 0 < limit <= int(vram[1] * strict.cuda.vram_fraction)

        x = np.random.default_rng(0).standard_normal((4, 3, 16, 16)).astype(np.float32)
        x_dev = engine.allocate(handle, x)
        y_dev = engine.empty(handle, x.shape, np.float32)
        assert x_dev.device_name() == "cuda" and y_dev.device_name() == "cuda"
        assert y_dev.shape() == list(x.shape)
        engine.run_bound(handle, {"x": x_dev}, {"y": y_dev})
        np.testing.assert_allclose(y_dev.numpy(), _expected(x), rtol=1e-6, atol=1e-6)

        # Host-memory path on the same session.
        (y_host,) = handle.run({"x": x})
        np.testing.assert_allclose(y_host, _expected(x), rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_TRT_EP, reason="onnxruntime build has no TensorrtExecutionProvider")
def test_tensorrt_chain_reports_honestly(tiny_model: Path, config: EngineConfig) -> None:
    profile = ShapeProfile(min_shapes={"x": (1, 3, 16, 16)}, opt_shapes={"x": (2, 3, 16, 16)},
                           max_shapes={"x": (4, 3, 16, 16)})
    with ExecutionEngine(config) as engine:
        handle = engine.get_session(tiny_model, shape_profile=profile)
        assert handle.wanted == Provider.TENSORRT.value
        assert handle.fell_back == (handle.primary_provider != Provider.TENSORRT.value)
        if os.environ.get("FACE_ENGINE_REQUIRE_TRT") == "1":
            assert handle.primary_provider == Provider.TENSORRT.value, handle.granted
        if handle.primary_provider == Provider.TENSORRT.value:
            trt = handle.session.get_provider_options()[Provider.TENSORRT.value]
            assert trt["trt_fp16_enable"] == "1"
            assert trt["trt_engine_cache_enable"] == "1"
            assert int(trt["trt_max_workspace_size"]) == 4 * 1024 ** 3
            assert Path(trt["trt_engine_cache_path"]) == config.resolved_trt_cache_dir()
            assert trt["trt_profile_opt_shapes"] == "x:2x3x16x16"
        x = np.ones((3, 3, 16, 16), dtype=np.float32)
        (y,) = handle.run({"x": x})
        np.testing.assert_allclose(y, _expected(x), rtol=1e-3, atol=1e-3)  # FP16 tolerance

@pytest.mark.gpu
@pytest.mark.skipif(not (HAS_CUDA_EP or HAS_TRT_EP), reason="onnxruntime has neither CUDA nor TensorRT EP")
@pytest.mark.parametrize("provider_chain", [
    pytest.param([Provider.TENSORRT, Provider.CUDA, Provider.CPU], id="trt",
                 marks=pytest.mark.skipif(not HAS_TRT_EP, reason="no TRT")),
    pytest.param([Provider.CUDA, Provider.CPU], id="cuda",
                 marks=pytest.mark.skipif(not HAS_CUDA_EP, reason="no CUDA")),
])
def test_zero_copy_iobinding_fp16_cuda_tensorrt_latency(
        tiny_model_fp16: Path, config: EngineConfig, provider_chain: list[Provider]) -> None:
    """Verifies dummy FP16 tensor loaded on CUDA runs through ExecutionEngine.run_binding()
    using TensorRT/CUDA without copying back to CPU RAM, measuring forward execution latency in ms."""
    import torch
    assert torch.cuda.is_available(), "CUDA device required for zero-copy IOBinding test"

    engine_cfg = config.model_copy(update={"providers": provider_chain})
    with ExecutionEngine(engine_cfg) as engine:
        handle = engine.get_session(tiny_model_fp16, trt_fp16=True)
        assert handle.on_gpu, f"Session must run on GPU, granted: {handle.granted}"

        # 1. Prepare dummy FP16 tensor directly on CUDA
        batch_size = 2
        dummy_x = torch.randn(batch_size, 3, 16, 16, dtype=torch.float16, device="cuda:0")
        assert dummy_x.is_cuda and dummy_x.dtype == torch.float16

        # 2. Run through ExecutionEngine.run_binding()
        outputs = engine.run_binding(handle, {"x": dummy_x})

        # 3. Assert zero-copy device residence (never transferred back to CPU host)
        assert "y" in outputs
        out_y = outputs["y"]
        assert isinstance(out_y, torch.Tensor)
        assert out_y.is_cuda
        assert out_y.device.type == "cuda"
        assert out_y.dtype == torch.float16
        assert tuple(out_y.shape) == (batch_size, 3, 16, 16)

        # Verify numerical accuracy
        expected = (_SCALE.astype(np.float16) * dummy_x.cpu().numpy() + np.float16(_BIAS))
        np.testing.assert_allclose(out_y.cpu().numpy(), expected, rtol=1e-2, atol=1e-2)

        # 4. Chain tensors in VRAM without CPU roundtrip
        chained = engine.run_binding(handle, {"x": out_y})
        assert chained["y"].is_cuda
        assert chained["y"].device.type == "cuda"
        assert chained["y"].dtype == torch.float16

        # 5. Measure forward execution latency in milliseconds
        # Warmup
        for _ in range(10):
            engine.run_binding(handle, {"x": dummy_x})
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        iterations = 50

        start_event.record()
        for _ in range(iterations):
            engine.run_binding(handle, {"x": dummy_x})
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event) / iterations
        assert latency_ms > 0.0, f"Expected positive execution latency, got {latency_ms}"
        print(f"\n[Zero-Copy IOBinding] {handle.primary_provider} FP16 forward latency: {latency_ms:.4f} ms", flush=True)


# ---------------------------------------------------------------- fallback (CPU-only)
def _fail_gpu_providers(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    real = ort.InferenceSession
    attempts: list[list[str]] = []

    def fake(path: str, sess_options: object = None, providers: list[str] = (),
             provider_options: object = None) -> ort.InferenceSession:
        attempts.append(list(providers))
        if providers[0] != Provider.CPU.value:
            raise RuntimeError(f"simulated {providers[0]} load failure")
        return real(path, sess_options=sess_options, providers=providers,
                    provider_options=provider_options)

    monkeypatch.setattr(ort, "InferenceSession", fake)
    monkeypatch.setattr(ExecutionEngine, "available_providers",
                        staticmethod(lambda: [p.value for p in Provider]))
    return attempts


def test_construction_failure_falls_back_down_the_chain(
        tiny_model: Path, engine: ExecutionEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = _fail_gpu_providers(monkeypatch)
    handle = engine.get_session(tiny_model)
    assert [a[0] for a in attempts] == [p.value for p in Provider]
    assert handle.primary_provider == Provider.CPU.value
    assert handle.wanted == Provider.TENSORRT.value and handle.fell_back


def test_strict_mode_refuses_a_fallback(tiny_model: Path, config: EngineConfig,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_gpu_providers(monkeypatch)
    with ExecutionEngine(_cfg(config, strict=True)) as engine:
        with pytest.raises(ProviderFallbackError):
            engine.get_session(tiny_model)


def test_every_chain_failing_raises(tiny_model: Path, engine: ExecutionEngine,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(ort, "InferenceSession", always_fail)
    with pytest.raises(SessionCreationError, match="boom"):
        engine.get_session(tiny_model, providers=[Provider.CPU])


# ---------------------------------------------------------------- session cache
def test_cuda_limit_ignores_free_vram_by_default(engine: ExecutionEngine,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from face_engine.core import execution

    total = 12 * 1024 ** 3
    monkeypatch.setattr(execution, "query_vram", lambda _device=0: (0, total))  # busy WDDM card
    assert engine.cuda_mem_limit() == int(total * 0.8)
    capped = ExecutionEngine(engine.config.model_copy(update={
        "cuda": engine.config.cuda.model_copy(update={"free_vram_fraction": 0.9})}))
    assert capped.cuda_mem_limit() == 1  # the opt-in cap is what starves the arena


def test_identical_requests_share_one_session(tiny_model: Path, engine: ExecutionEngine) -> None:
    a = engine.get_session(tiny_model, providers=[Provider.CPU])
    b = engine.get_session(str(tiny_model), providers=[Provider.CPU])
    assert a is b and a.session is b.session
    assert len(engine.cached_sessions) == 1


def test_a_different_shape_profile_is_a_different_session(
        tiny_model: Path, engine: ExecutionEngine) -> None:
    p1 = ShapeProfile({"x": (1, 3, 16, 16)}, {"x": (1, 3, 16, 16)}, {"x": (2, 3, 16, 16)})
    p2 = ShapeProfile({"x": (1, 3, 16, 16)}, {"x": (4, 3, 16, 16)}, {"x": (8, 3, 16, 16)})
    a = engine.get_session(tiny_model, providers=[Provider.CPU], shape_profile=p1)
    b = engine.get_session(tiny_model, providers=[Provider.CPU], shape_profile=p2)
    c = engine.get_session(tiny_model, providers=[Provider.CPU], shape_profile=p1)
    assert a is not b and a is c


def test_release_and_cleanup(tiny_model: Path, engine: ExecutionEngine) -> None:
    engine.get_session(tiny_model, providers=[Provider.CPU])
    assert engine.release(tiny_model) == 1
    assert engine.cached_sessions == []
    engine.cleanup_vram()  # must not raise with or without torch/CUDA


def test_shape_profile_validation() -> None:
    with pytest.raises(ValueError):
        ShapeProfile({"x": (4, 3)}, {"x": (2, 3)}, {"x": (8, 3)})
    with pytest.raises(ValueError):
        ShapeProfile({"x": (1,)}, {"y": (1,)}, {"x": (1,)})

    # Dynamic batching presets:
    bbox = ShapeProfile.bounding_box_mask_profile("input")
    assert bbox.min_shapes["input"] == (1, 3, 256, 256)
    assert bbox.opt_shapes["input"] == (4, 3, 256, 256)
    assert bbox.max_shapes["input"] == (8, 3, 256, 256)
    assert "input:1x3x256x256" in bbox.trt_options()["trt_profile_min_shapes"]

    superres = ShapeProfile.super_resolution_profile("input")
    assert superres.min_shapes["input"] == (1, 3, 512, 512)
    assert superres.opt_shapes["input"] == (2, 3, 512, 512)
    assert superres.max_shapes["input"] == (4, 3, 512, 512)
    assert "input:1x3x512x512" in superres.trt_options()["trt_profile_min_shapes"]


def test_default_config_matches_the_stage1_contract(config: EngineConfig) -> None:
    assert config.providers == [Provider.TENSORRT, Provider.CUDA, Provider.CPU]
    assert config.cuda.arena_extend_strategy == "kNextPowerOfTwo"
    assert config.cuda.cudnn_conv_algo_search == "HEURISTIC"
    assert config.cuda.use_tf32 is False
    assert config.cuda.do_copy_in_default_stream is True
    trt = config.tensorrt_provider_options()
    assert trt["device_id"] == 0
    assert trt["trt_max_workspace_size"] == 4294967296
    assert trt["trt_fp16_enable"] is True and trt["trt_engine_cache_enable"] is True
    default = EngineConfig()
    # Anchored at the repository root, not the working directory.
    root = Path(__file__).resolve().parents[2]
    if "FACE_ENGINE_CACHE_DIR" not in os.environ:
        assert default.resolved_trt_cache_dir() == (root / ".cache" / "trt_engines").resolve()


# ---------------------------------------------------------------- registry + downloads
def test_zoo_declarations_are_complete() -> None:
    expected = {"scrfd_10g_bnkps", "retinaface_r50", "yoloface_8n", "hrffa", "2dfan4", "arcface_w600k_r50",
                "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "alphaface_256",
                "inswapper_128", "inswapper_128_fp16", "xseg", "xseg_3", "bisenet_resnet34", "gpen_bfr_512", "gpen_bfr_1024",
                "gpen_bfr_2048", "restoreformer_plus_plus",
                # added after this list was written (all pinned + sized, so the integrity
                # assertions below cover them): the hififace / hyperswap-256 swappers and their
                # CrossFace embedder, the DFL XSeg v2 / face-occluder v3 / SAM2 occluders and
                # the BiSeNet-34 parser.
                "hififace_256", "crossface_hififace", "hyperswap_256", "dfl_xseg_v2",
                "face_occluder_v3", "face_parser_bisenet34", "sam2_hiera_tiny",
                *(f"liveportrait_{k}" for k in ("appearance", "motion", "warping", "stitching",
                                                "eye", "landmark"))}
    assert set(MODEL_ZOO) == expected
    # Two specs may share a file only as ALIASES of one artifact (hyperswap_256 = hyperswap_1a_256,
    # face_occluder_v3 = xseg_3, dfl_xseg_v2 = xseg, face_parser_bisenet34 = bisenet_resnet34):
    # identical bytes, so a download of either can never clobber the other. Different models
    # writing one filename is the bug this guards.
    by_file: dict[str, list] = {}
    for s in MODEL_ZOO.values():
        by_file.setdefault(s.filename, []).append(s)
    for filename, specs in by_file.items():
        assert len({(s.sha256, s.size, s.urls) for s in specs}) == 1, (
            f"{filename}: {[s.name for s in specs]} are different artifacts under one filename")
    for spec in MODEL_ZOO.values():
        # A URL without a pinned hash would download unverified bytes.
        assert spec.downloadable == spec.pinned, spec.name
        assert spec.downloadable == (spec.size is not None), spec.name
    assert {s.name for s in MODEL_ZOO.values() if not s.downloadable} == {"hrffa", "alphaface_256"}
    assert MODEL_ZOO["scrfd_10g_bnkps"].dynamic_axes
    assert not MODEL_ZOO["yoloface_8n"].dynamic_axes


def test_unavailable_model_fails_clearly(tmp_path: Path) -> None:
    registry = build_default_registry(tmp_path)
    with pytest.raises(ModelUnavailableError, match="hrffa"):
        registry.ensure("hrffa", show_progress=False)
    # hyperswap 1a/1b/1c/256, inswapper 128 (+fp16), alphaface, hififace
    assert len(registry.by_task(ModelTask.SWAP)) == 8


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload, self.status_code = payload, status
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise downloads.requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        stream = io.BytesIO(self._payload)
        yield from iter(lambda: stream.read(chunk_size), b"")


class _FakeSession:
    def __init__(self, by_url: dict) -> None:
        self.by_url, self.calls = by_url, []

    def get(self, url: str, **_kwargs: object) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(self.by_url[url])

    def close(self) -> None:
        return None


def test_download_rejects_bad_bytes_then_takes_the_verified_mirror(tmp_path: Path) -> None:
    good = b"model-bytes" * 1000
    digest = hashlib.sha256(good).hexdigest()
    http = _FakeSession({"https://bad/m.onnx": b"x" * len(good), "https://good/m.onnx": good})
    target = tmp_path / "m.onnx"
    download_file(["https://bad/m.onnx", "https://good/m.onnx"], target, sha256=digest,
                  size=len(good), show_progress=False, session=http)  # type: ignore[arg-type]
    assert target.read_bytes() == good
    assert not target.with_name("m.onnx.part").exists()
    assert verify_file(target, digest, len(good))
    # Verified -> no second fetch.
    download_file(["https://good/m.onnx"], target, sha256=digest, size=len(good),
                  show_progress=False, session=http)  # type: ignore[arg-type]
    assert http.calls == ["https://bad/m.onnx", "https://good/m.onnx"]

    with pytest.raises(IntegrityError):
        download_file(["https://bad/m.onnx"], tmp_path / "other.onnx", sha256=digest,
                      size=len(good), show_progress=False, session=http)  # type: ignore[arg-type]


def test_registry_ensure_verifies_a_local_file(tmp_path: Path) -> None:
    payload = b"\x00\x01" * 512
    (tmp_path / "local.onnx").write_bytes(payload)
    spec = ModelSpec(name="local", task=ModelTask.DETECTION, filename="local.onnx",
                     urls=("https://unused/local.onnx",),
                     sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
    registry = ModelRegistry(tmp_path, [spec])
    assert registry.verify("local")
    assert registry.ensure("local", show_progress=False) == tmp_path / "local.onnx"
    (tmp_path / "local.onnx").write_bytes(payload[:-1] + b"\xff")
    assert not registry.verify("local")



def test_sidecar_trusts_only_files_older_than_their_hash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good, bad = b"a" * 4096, b"b" * 4096
    digest = hashlib.sha256(good).hexdigest()
    path = tmp_path / "m.onnx"
    calls: list[Path] = []
    real = downloads.sha256_file
    monkeypatch.setattr(downloads, "sha256_file", lambda p, *a: calls.append(p) or real(p, *a))

    # A freshly written file is re-hashed every time: NTFS keeps mtime_ns
    # across a fast same-size rewrite, so (size, mtime) cannot tell them apart.
    # Force the collision instead of hoping for one: restore the fresh mtime.
    path.write_bytes(good)
    fresh = path.stat().st_mtime_ns
    assert verify_file(path, digest, len(good))
    path.write_bytes(bad)
    os.utime(path, ns=(fresh, fresh))
    assert not verify_file(path, digest, len(good))
    path.write_bytes(good)
    os.utime(path, ns=(fresh, fresh))
    assert verify_file(path, digest, len(good))
    assert len(calls) == 3

    # A file last modified well before it was hashed is trusted from the sidecar.
    old = path.stat().st_mtime_ns - 10_000_000_000
    os.utime(path, ns=(old, old))
    assert verify_file(path, digest, len(good))  # hashes, writes a trusted sidecar
    calls.clear()
    assert verify_file(path, digest, len(good))
    assert calls == []
