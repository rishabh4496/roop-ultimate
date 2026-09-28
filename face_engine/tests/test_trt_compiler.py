"""Stage 6: AOT TensorRT compilation, verification, lookup and use.

The builder / runner / precision-pinning tests build TINY synthetic ONNX
models (seconds). The integration test uses the real HyperSwap engine when
``tools/compile_engines.py`` has produced one, and skips otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper

from face_engine.core import trt_compiler as tc
from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine, register_gpu_runtime_dirs

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import compile_engines

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]


@pytest.fixture(scope="module", autouse=True)
def _runtime() -> None:
    register_gpu_runtime_dirs()


# ------------------------------------------------------------------ pure
def test_cli_parses_sizes_and_groups() -> None:
    assert compile_engines.parse_size("4GB") == 4 << 30
    assert compile_engines.parse_size("1.5GiB") == int(1.5 * (1 << 30))
    assert compile_engines.parse_size("512MB") == 512 << 20
    assert {s.model for s in tc.select_specs("swapper")} == {
        "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "inswapper_128"}
    assert {s.group for s in tc.select_specs("all")} == {"swapper", "enhancer", "masker",
                                                           "detector"}
    assert [s.model for s in tc.select_specs("xseg_3")] == ["xseg_3"]
    with pytest.raises(KeyError):
        tc.select_specs("nonsense")


def test_specs_use_the_models_real_inputs() -> None:
    """Names / layouts are the graphs', not the brief's (source, NHWC XSeg, GPEN batch 1)."""
    hs = tc.ENGINE_SPECS["hyperswap_1a_256"]
    assert hs.inputs == {"source": (512,), "target": (3, 256, 256)} and hs.batch == (1, 2, 8)
    assert tc.ENGINE_SPECS["xseg_3"].inputs == {"input": (256, 256, 3)}
    assert tc.ENGINE_SPECS["gpen_bfr_512"].batch == (1, 1, 1)
    assert tc.ENGINE_SPECS["gpen_bfr_1024"].inputs == {"input": (3, 1024, 1024)}


def test_gpu_discovery_matches_cuda() -> None:
    gpu = tc.discover_gpu(0)
    major, minor = torch.cuda.get_device_capability(0)
    assert gpu.sm == f"{major}{minor}" and gpu.total_vram_mb > 1000
    path = tc.engine_path(tc.ENGINE_SPECS["hyperswap_1a_256"], gpu, "fp16", Path("x"))
    assert path.name == f"hyperswap_1a_256_sm{gpu.sm}_fp16_b8.engine"


# ------------------------------------------------------------------ tiny builds
def _tiny_conv(path: Path) -> np.ndarray:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["y"], ["out"])], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 16, 16])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 4, 16, 16])],
        [numpy_helper.from_array(w, "w")])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return w


def test_build_serialize_run_and_timing_cache(tmp_path: Path) -> None:
    src = tmp_path / "tiny.onnx"
    w = _tiny_conv(src)
    spec = tc.EngineSpec("tiny", "test", {"x": (3, 16, 16)}, (1, 2, 4), batched_graph=False)
    gpu = tc.discover_gpu(0)
    result = tc.build_engine(spec, src, "fp32", gpu, root=tmp_path, workspace_bytes=256 << 20)
    assert result.path.exists() and (tmp_path / tc.TIMING_CACHE).stat().st_size > 0
    engine = tc.TensorRTEngine(result.path)
    assert engine.input_names == ("x",) and engine.output_names == ("out",)
    assert engine.max_batch == 4
    for batch in (1, 3, 4):  # anywhere inside the profile
        x = np.random.default_rng(batch).standard_normal((batch, 3, 16, 16)).astype(np.float32)
        out = engine.run_binding({"x": torch.from_numpy(x).cuda()})["out"]
        torch.cuda.synchronize()
        xt = torch.from_numpy(x)
        ref = torch.relu(torch.nn.functional.conv2d(xt, torch.from_numpy(w), padding=1)).numpy()
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-3)
    # A second build reuses the timing cache (no error on an existing cache).
    tc.build_engine(spec, src, "fp32", gpu, root=tmp_path, workspace_bytes=256 << 20)


def test_decomposed_instance_norm_is_pinned_to_fp32(tmp_path: Path) -> None:
    """The FP32 pin on a decomposed InstanceNorm: the layers really are FP32 under FP16."""
    from face_engine.utils.onnx_batch import decompose_instance_norm

    c = 8
    graph = helper.make_graph(
        [helper.make_node("InstanceNormalization", ["x", "s", "b"], ["y"], name="in0")],
        "norm", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", c, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", c, 4, 4])],
        [numpy_helper.from_array(np.ones(c, np.float32), "s"),
         numpy_helper.from_array(np.zeros(c, np.float32), "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    assert decompose_instance_norm(model) == 1
    path = tmp_path / "norm.onnx"
    onnx.save(model, str(path))
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(0)
    assert trt.OnnxParser(network, trt.Logger(trt.Logger.ERROR)).parse_from_file(str(path))
    pinned = tc.pin_decomposed_norms(network)
    assert pinned >= 6  # mean, sub, square, mean, add, sqrt, div, scale, bias
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if any(tc._IN_MARK in layer.get_output(j).name for j in range(layer.num_outputs)) \
                and layer.get_output(0).dtype in (trt.float32, trt.float16):
            assert layer.precision == trt.float32


def test_verification_rejects_nan_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NaN > gate is False in Python: a non-finite engine (GPEN-1024 FP16) once passed."""
    spec = tc.ENGINE_SPECS["xseg_3"]
    result = tc.BuildResult(spec, tmp_path / "x.engine", "fp16", tc.discover_gpu(0), "g", 0.0, 0, 0.0)
    monkeypatch.setattr(tc, "TensorRTEngine", lambda *a, **k: object())
    monkeypatch.setattr(tc, "check_fidelity", lambda *a, **k: {"output.mean": float("nan")})
    monkeypatch.setattr(tc, "measure_latency", lambda *a, **k: 1.0)
    monkeypatch.setattr(tc, "write_sidecar", lambda *a, **k: None)
    verified = tc.verify(result, tmp_path / "src.onnx")
    assert not verified.ok and "non-finite" in verified.problems[0]


# ------------------------------------------------------------------ lookup
def test_find_engine_refuses_anything_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "xseg_3.onnx"
    src.write_bytes(b"model v1")
    gpu = tc.discover_gpu(0)
    spec = tc.ENGINE_SPECS["xseg_3"]
    path = tc.engine_path(spec, gpu, "fp16", tmp_path)
    path.write_bytes(b"engine")
    record = {"ok": True, "tensorrt": trt.__version__, "gpu": {"sm": gpu.sm},
              "source": tc._stamp(src)}
    side = path.with_suffix(".json")
    side.write_text(json.dumps(record), encoding="utf-8")
    assert tc.find_engine("xseg_3", src, "fp16", root=tmp_path) == path
    assert tc.find_engine_for(src, "fp16", root=tmp_path) == path
    assert tc.find_engine("xseg_3", src, "fp32", root=tmp_path) is None  # other precision
    for bad in ({"ok": False}, {"tensorrt": "9.0"}, {"gpu": {"sm": "75"}}):
        side.write_text(json.dumps({**record, **bad}), encoding="utf-8")
        assert tc.find_engine("xseg_3", src, "fp16", root=tmp_path) is None, bad
    side.write_text(json.dumps(record), encoding="utf-8")
    src.write_bytes(b"model v2, a different size")  # the model changed
    assert tc.find_engine("xseg_3", src, "fp16", root=tmp_path) is None


# ------------------------------------------------------------------ real engine in the pipeline
def _hyperswap_engine() -> tuple[Path, Path] | None:
    from face_engine.models.zoo import build_default_registry

    src = Path(build_default_registry().ensure("hyperswap_1a_256", show_progress=False))
    found = tc.find_engine("hyperswap_1a_256", src, "fp16")
    return (found, src) if found else None


@pytest.mark.skipif(_hyperswap_engine() is None,
                    reason="no compiled hyperswap_1a_256 fp16 engine (run tools/compile_engines.py)")
def test_batched_swapper_uses_the_aot_engine() -> None:
    import cv2
    import insightface

    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.detector import SCRFDDetector
    from face_engine.processors import BatchedFaceSwapper

    _, src = _hyperswap_engine()  # type: ignore[misc]
    trt_engine = ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT, Provider.CUDA]))
    cuda_engine = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU]))
    aot = BatchedFaceSwapper(trt_engine, "hyperswap_1a_256", src, precision="fp16")
    assert aot.aot is not None and aot.batched and aot.max_batch == 8
    assert aot.session.primary_provider.startswith("TensorRT (AOT")
    # A user who chose CUDA gets ONNX Runtime, not the TensorRT plan.
    ort = BatchedFaceSwapper(cuda_engine, "hyperswap_1a_256", src, precision="fp32")
    assert ort.aot is None
    # The default ("auto"): the FP16 engine where it exists, else fp32 on ONNX Runtime.
    default = BatchedFaceSwapper(trt_engine, "hyperswap_1a_256", src)
    assert default.precision == "fp16" and default.aot is not None
    assert BatchedFaceSwapper(cuda_engine, "hyperswap_1a_256", src).precision == "fp32"

    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    frame = torch.from_numpy(image).cuda().permute(2, 0, 1)[None].float()
    det = SCRFDDetector(cuda_engine, build_default_registry().ensure("scrfd_10g_bnkps",
                                                                     show_progress=False))
    kps = det.detect_cuda(frame).kps
    rng = np.random.default_rng(0)
    src_emb = torch.as_tensor(rng.standard_normal((len(kps), 512)), dtype=torch.float32).cuda()
    src_emb = src_emb / src_emb.norm(dim=1, keepdim=True)
    a = aot.swap(frame, kps, source=src_emb)
    b = ort.swap(frame, kps, source=src_emb)
    assert bool(a.ok.all())
    diff = (a.crops - b.crops).abs()
    assert float(diff.mean()) < 1.0 and float(diff.max()) < 40  # FP16 engine vs FP32 ORT, levels
