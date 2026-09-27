"""Stage 3, CUDA-resident: batched swapping, Pixel Boost, restoration, colour, expression.

Needs a CUDA device and the model files (``FACE_ENGINE_MODELS_DIR`` or
``./.cache/models``). Sessions use the CUDA execution provider so the tests
do not wait on TensorRT engine builds; precision is exercised by
``eval_stage3_precision.py`` / ``bench_stage3.py``. Real faces are the six in
``insightface``'s ``t1.jpg``.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.aligner import warp_face_inverse_cuda
from face_engine.pipeline.detector import GPUDetections, SCRFDDetector
from face_engine.processors import (
    BatchedExpressionRestorer,
    BatchedFaceEnhancer,
    BatchedFaceSwapper,
    ColorMode,
    ExpressionRestorer,
    FaceEnhancer,
    FaceSwapper,
    GPUIdentityEncoder,
    Identity,
    IdentityEncoder,
    transfer_color,
    transfer_color_cuda,
)
from face_engine.processors.swapper import (
    explode_pixel_boost,
    explode_pixel_boost_cuda,
    implode_pixel_boost,
    implode_pixel_boost_cuda,
)
from face_engine.utils import onnx_batch

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]


def _to_cuda(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).cuda().permute(2, 0, 1)[None].float()


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return t.clamp(0, 255).round().byte().permute(1, 2, 0).cpu().numpy()


@pytest.fixture(scope="module")
def registry():  # type: ignore[no-untyped-def]
    return build_default_registry()


@pytest.fixture(scope="module")
def engine() -> Iterator[ExecutionEngine]:
    eng = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU], strict=True))
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    import insightface

    return cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))


@pytest.fixture(scope="module")
def frame(image: np.ndarray) -> torch.Tensor:
    return _to_cuda(image)


@pytest.fixture(scope="module")
def detector(engine: ExecutionEngine, registry) -> SCRFDDetector:  # type: ignore[no-untyped-def]
    return SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps", show_progress=False))


@pytest.fixture(scope="module")
def faces(detector: SCRFDDetector, frame: torch.Tensor) -> GPUDetections:
    return detector.detect_cuda(frame)


@pytest.fixture(scope="module")
def encoders(engine: ExecutionEngine, registry) -> tuple[IdentityEncoder, GPUIdentityEncoder]:  # type: ignore[no-untyped-def]
    path = registry.ensure("arcface_w600k_r50", show_progress=False)
    return IdentityEncoder(engine, path), GPUIdentityEncoder(engine, path)


@pytest.fixture(scope="module")
def identities(encoders, image: np.ndarray, faces: GPUDetections) -> np.ndarray:  # type: ignore[no-untyped-def]
    return np.stack([encoders[0].embed(image, f).embedding for f in faces.to_faces()[0]])


@pytest.fixture(scope="module")
def hyperswap(engine: ExecutionEngine, registry) -> BatchedFaceSwapper:  # type: ignore[no-untyped-def]
    return BatchedFaceSwapper(engine, "hyperswap_1a_256",
                              registry.ensure("hyperswap_1a_256", show_progress=False))


def _sources(identities: np.ndarray) -> torch.Tensor:
    """Face i gets the identity of face i+1."""
    return torch.as_tensor(identities[[(i + 1) % 6 for i in range(6)]], device="cuda")


# ------------------------------------------------------------------ batching
def test_models_that_batch_and_models_that_do_not(registry) -> None:  # type: ignore[no-untyped-def]
    """The verified rewrite: which models batch, and the ones refused stay refused."""
    batchable = ("hyperswap_1a_256", "inswapper_128", "arcface_w600k_r50", "restoreformer_plus_plus",
                 "liveportrait_motion", "liveportrait_appearance",
                 "liveportrait_stitching", "liveportrait_eye")
    for name in batchable:
        assert onnx_batch.batched_model(registry.ensure(name, show_progress=False)) is not None, name
    # GPEN: StyleGAN2 modulated convs fold the batch into Conv groups.
    # LivePortrait landmark: its rewrite runs but flattens the batch together.
    for name in ("gpen_bfr_512", "liveportrait_landmark"):
        assert onnx_batch.batched_model(registry.ensure(name, show_progress=False)) is None, name
    # A rewritten InstanceNorm does not survive a TensorRT FP16 build: FP16 callers
    # get no batched graph for those models (and do for the ones without).
    for name in ("hyperswap_1a_256", "restoreformer_plus_plus"):
        path = registry.ensure(name, show_progress=False)
        assert onnx_batch.batched_model(path, fp16=True) is None, name
    assert onnx_batch.batched_model(registry.ensure("inswapper_128", show_progress=False),
                                    fp16=True) is not None


def test_verifier_catches_instance_norm_cross_talk(registry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Without the InstanceNorm decomposition, HyperSwap's batch rows bleed into
    each other on CUDA (CPU is fine). The verifier must refuse that graph."""
    import onnx

    source = registry.ensure("hyperswap_1a_256", show_progress=False)
    model = onnx.load(str(source))
    onnx_batch.make_batch_dynamic(model)  # no decompose_instance_norm
    broken = tmp_path / "hyperswap_broken.onnx"
    onnx.save(model, str(broken))
    with pytest.raises(ValueError, match="differ from the reference"):
        onnx_batch.verify_batch_equivalence(source, broken)


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_pixel_boost_layout_matches_the_host(factor: int) -> None:
    rng = np.random.default_rng(factor)
    crop = rng.integers(0, 256, (factor * 8, factor * 8, 3)).astype(np.float32)
    tiles_host = implode_pixel_boost(crop, 8, factor)
    tiles = implode_pixel_boost_cuda(torch.as_tensor(crop).permute(2, 0, 1)[None].cuda(), 8, factor)
    np.testing.assert_array_equal(tiles.permute(0, 2, 3, 1).cpu().numpy(), tiles_host)
    back = explode_pixel_boost_cuda(tiles, 8, factor)[0].permute(1, 2, 0).cpu().numpy()
    np.testing.assert_array_equal(back, explode_pixel_boost(tiles_host, 8, factor))
    np.testing.assert_array_equal(back, crop)


# ------------------------------------------------------------------ identity
def test_gpu_encoder_matches_host(encoders, frame: torch.Tensor, faces: GPUDetections,  # type: ignore[no-untyped-def]
                                  identities: np.ndarray) -> None:
    got = encoders[1].embed(frame, faces.kps)
    assert got.device.type == "cuda" and encoders[1].batched
    np.testing.assert_allclose(torch.norm(got, dim=1).cpu().numpy(), 1.0, atol=1e-5)
    assert ((got.cpu().numpy() * identities).sum(1) > 0.999).all()


def test_source_latent_is_cached_and_projected(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                               identities: np.ndarray) -> None:
    path = registry.ensure("inswapper_128", show_progress=False)
    gpu, host = BatchedFaceSwapper(engine, "inswapper_128", path), FaceSwapper(engine, "inswapper_128", path)
    latent = gpu.set_source(Identity(identities[0], 1.0))
    assert latent.device.type == "cuda" and latent.shape == (1, 512)
    np.testing.assert_allclose(latent.cpu().numpy(), host.latent(Identity(identities[0], 1.0)),
                               atol=1e-5)


# ------------------------------------------------------------------ swapping
def test_batched_swap_equals_one_face_at_a_time(hyperswap: BatchedFaceSwapper,
                                                frame: torch.Tensor, faces: GPUDetections,
                                                identities: np.ndarray) -> None:
    src = _sources(identities)
    batch = hyperswap.swap(frame, faces.kps, source=src)
    assert hyperswap.batched and batch.crops.shape == (6, 3, 256, 256)
    assert bool(batch.ok.all()) and batch.model_mask.shape == (6, 1, 256, 256)
    for i in range(6):
        one = hyperswap.swap(frame, faces.kps[i:i + 1], source=src[i:i + 1])
        assert float((batch.crops[i] - one.crops[0]).abs().max()) < 1.0  # levels


@pytest.mark.parametrize("boost", [256, 512, 768])
def test_batched_swap_matches_the_host_swapper(hyperswap: BatchedFaceSwapper, engine,  # type: ignore[no-untyped-def]
                                               registry, image: np.ndarray, frame: torch.Tensor,
                                               faces: GPUDetections, identities: np.ndarray,
                                               encoders, detector: SCRFDDetector,
                                               boost: int) -> None:
    """Same identity transfer as the host path (bicubic GPU cut vs Lanczos host cut)."""
    host = FaceSwapper(engine, "hyperswap_1a_256", registry.ensure("hyperswap_1a_256",
                                                                     show_progress=False))
    res = hyperswap.swap(frame, faces.kps, source=_sources(identities), pixel_boost=boost)
    assert res.crops.shape[-1] == boost and res.pixel_boost == boost
    enc = encoders[0]
    got, ref = [], []
    for i, face in enumerate(faces.to_faces()[0]):
        target = Identity(identities[(i + 1) % 6], 1.0)
        h = host.swap(image, face, target, pixel_boost=boost)
        assert np.abs(_to_uint8(res.crops[i]).astype(int) - h.crop).mean() < 4.0
        pasted = _to_uint8(warp_face_inverse_cuda(frame, res.crops[i:i + 1],
                                                  res.matrices[i:i + 1])[0])
        for out, dest in ((pasted, got), (host.paste(image, h), ref)):
            e = enc.embed(out, min(detector.detect(out),
                                   key=lambda q: np.abs(q.bbox - face.bbox).sum()))
            dest.append(e.similarity(target))
    assert abs(np.mean(got) - np.mean(ref)) < 0.01
    assert np.mean(got) > 0.65


def test_non_finite_output_keeps_the_target(hyperswap: BatchedFaceSwapper, frame: torch.Tensor,
                                            faces: GPUDetections, identities: np.ndarray,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    original = hyperswap.run_crops

    def poisoned(crops: torch.Tensor, latents: torch.Tensor) -> Any:
        image, mask = original(crops, latents)
        image[1] = float("nan")
        return image, mask

    monkeypatch.setattr(hyperswap, "run_crops", poisoned)
    res = hyperswap.swap(frame, faces.kps[:3], source=_sources(identities)[:3])
    assert res.ok.tolist() == [True, False, True]
    assert torch.equal(res.crops[1], res.target_crops[1].clamp(0, 255))


# ------------------------------------------------------------------ restoration & colour
def test_batched_gpen_matches_the_host_enhancer(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                                image: np.ndarray, frame: torch.Tensor,
                                                faces: GPUDetections) -> None:
    path = registry.ensure("gpen_bfr_512", show_progress=False)
    gpu = BatchedFaceEnhancer(engine, "gpen_bfr_512", path, precision="fp32")
    res = gpu.enhance(frame, faces.kps)
    assert not gpu.batched and bool(res.ok.all()) and res.frames.device.type == "cuda"
    host = FaceEnhancer(engine, "gpen_bfr_512", path)
    ref = image
    for face in faces.to_faces()[0]:
        ref = host.enhance(ref, face).frame
    assert np.abs(_to_uint8(res.frames[0]).astype(int) - ref).mean() < 0.5


def test_collapsed_restoration_is_rejected(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                           frame: torch.Tensor, faces: GPUDetections,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = BatchedFaceEnhancer(engine, "gpen_bfr_512",
                              registry.ensure("gpen_bfr_512", show_progress=False))
    original = gpu.restore

    def flat(crops: torch.Tensor) -> Any:
        restored, ok = original(crops)
        return restored, ok & (torch.arange(len(ok), device=ok.device) != 0)

    monkeypatch.setattr(gpu, "restore", flat)
    res = gpu.enhance(frame, faces.kps[:2])
    assert res.ok.tolist() == [False, True]
    before = frame[0]
    x0, y0, x1, y1 = faces.boxes[0].round().int().tolist()
    assert float((res.frames[0][:, y0:y1, x0:x1] - before[:, y0:y1, x0:x1]).abs().max()) == 0.0


@pytest.mark.parametrize("mode", [ColorMode.LAB_MEAN, ColorMode.REINHARD, ColorMode.KEEP_CHROMA])
def test_gpu_colour_transfer_matches_the_host(image: np.ndarray, frame: torch.Tensor,
                                              mode: ColorMode) -> None:
    tinted = np.clip(image.astype(int) + [12, -6, 20], 0, 255).astype(np.uint8)
    host = transfer_color(tinted, image, mode)
    got = _to_uint8(transfer_color_cuda(_to_cuda(tinted), frame, mode)[0])
    assert np.abs(got.astype(int) - host).mean() < 1.0
    assert np.abs(got.astype(int) - image).mean() < 0.3 * np.abs(tinted.astype(int) - image).mean()


def test_colour_transfer_needs_a_region() -> None:
    img = torch.rand(2, 3, 32, 32, device="cuda") * 255
    ref = torch.rand(2, 3, 32, 32, device="cuda") * 255
    region = torch.zeros(2, 1, 32, 32, device="cuda")
    region[1, :, :8, :8] = 1  # 64 px: enough; sample 0 has none
    out = transfer_color_cuda(img, ref, ColorMode.REINHARD, region)
    assert torch.equal(out[0], img[0]) and not torch.equal(out[1], img[1])


# ------------------------------------------------------------------ expression
def test_batched_expression_matches_the_host(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                             hyperswap: BatchedFaceSwapper, frame: torch.Tensor,
                                             faces: GPUDetections, identities: np.ndarray) -> None:
    res = hyperswap.swap(frame, faces.kps, source=_sources(identities), pixel_boost=512)
    gpu = BatchedExpressionRestorer.from_registry(engine, registry)
    assert set(gpu._batched) == {"appearance", "motion", "eye", "stitching"}
    got = gpu.restore(res.crops, res.target_crops)
    assert gpu.failures == 0
    host = ExpressionRestorer.from_registry(engine, registry)
    for i in range(6):
        ref = host.restore(_to_uint8(res.crops[i]), _to_uint8(res.target_crops[i]))
        assert np.abs(_to_uint8(got[i]).astype(int) - ref).mean() < 2.0
    assert float((got - res.crops).abs().mean()) > 2.0  # it did something


def test_expression_failure_returns_the_input(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = BatchedExpressionRestorer.from_registry(engine, registry)
    crops = torch.rand(2, 3, 256, 256, device="cuda") * 255

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("model failure")

    monkeypatch.setattr(gpu, "_restore", boom)
    assert gpu.restore(crops, crops) is crops
    assert gpu.failures == 1  # counted, so a benchmark can refuse the row


# ------------------------------------------------------------------ residency
@pytest.fixture
def no_host_copies(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Tensor -> host copies raise (scalar reads stay allowed)."""
    calls: list[str] = []

    def guard(name: str, original: Any) -> Any:
        def wrapped(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            if self.is_cuda:
                calls.append(name)
                raise AssertionError(f"Tensor.{name}() copied a CUDA tensor to the host")
            return original(self, *args, **kwargs)
        return wrapped

    for name in ("cpu", "numpy", "tolist", "__array__"):
        monkeypatch.setattr(torch.Tensor, name, guard(name, getattr(torch.Tensor, name)))
    original_to = torch.Tensor.to

    def to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = original_to(self, *args, **kwargs)
        if self.is_cuda and not out.is_cuda:
            calls.append("to")
            raise AssertionError("Tensor.to() moved a CUDA tensor to the host")
        return out

    monkeypatch.setattr(torch.Tensor, "to", to)
    yield calls


def test_stage3_chain_never_leaves_the_gpu(engine: ExecutionEngine, registry,  # type: ignore[no-untyped-def]
                                           hyperswap: BatchedFaceSwapper, encoders,
                                           frame: torch.Tensor, faces: GPUDetections,
                                           identities: np.ndarray,
                                           no_host_copies: list[str]) -> None:
    enhancer = BatchedFaceEnhancer(engine, "gpen_bfr_512",
                                   registry.ensure("gpen_bfr_512", show_progress=False))
    restorer = BatchedExpressionRestorer.from_registry(engine, registry)
    source = encoders[1].embed(frame, faces.kps[:1])
    hyperswap.set_source(source[0])
    res = hyperswap.swap(frame, faces.kps[1:5], pixel_boost=512)
    crops = restorer.restore(res.crops, res.target_crops)
    pasted = warp_face_inverse_cuda(frame, crops, res.matrices, res.model_mask)
    out = enhancer.enhance(pasted, faces.kps[1:5], reference=frame)
    assert restorer.failures == 0
    for t in (source, res.crops, res.matrices, res.model_mask, crops, pasted, out.frames,
              out.crops_out, out.ok):
        assert t.device.type == "cuda"
    assert bool(res.ok.all()) and bool(out.ok.all())
    assert no_host_copies == []
