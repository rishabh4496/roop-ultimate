"""The boosted pipeline: RetinaFace R50, XSeg feathering, dual-stream processor, engines.

GPU tests use the insightface sample photo (six faces) and the zoo's models;
they skip without CUDA.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face_engine.boosted import retinaface_gpu as rg
from face_engine.boosted import trt_compiler as btc
from face_engine.boosted.ultra_pipeline import BOOSTED_PRESETS, preset_params
from face_engine.boosted.xseg_masker import feather
from face_engine.server.processing import required_models

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


# ---------------------------------------------------------------------------- pure
def test_priors_match_biubug6_layout() -> None:
    p = rg.priors(640)
    assert p.shape == ((80 * 80 + 40 * 40 + 20 * 20) * 2, 4)
    # first anchor: cell (0, 0) of stride 8, min size 16 then 32
    assert np.allclose(p[0], [4 / 640, 4 / 640, 16 / 640, 16 / 640])
    assert np.allclose(p[1], [4 / 640, 4 / 640, 32 / 640, 32 / 640])
    assert np.allclose(p[2, :2], [12 / 640, 4 / 640])  # x runs fastest
    assert np.allclose(p[-1], [624 / 640, 624 / 640, 512 / 640, 512 / 640])  # (19 + .5) * 32


def test_presets_use_r50_and_differ_only_by_the_restorer() -> None:
    boosted, ultra = preset_params("boosted"), preset_params("ultra")
    assert required_models(boosted)[0] == "retinaface_r50"
    assert "scrfd_10g_bnkps" not in required_models(boosted)
    assert boosted.enhancer_model == "none" and ultra.enhancer_model == "gpen_bfr_1024"
    restorer = ("enhancer_model", "enhancer_blend")
    strip = lambda d: {k: v for k, v in d.items() if k not in restorer}
    assert strip(BOOSTED_PRESETS["ultra"]) == strip(BOOSTED_PRESETS["boosted"])
    assert boosted.detection_stride == 3 and boosted.mask_types == ["box", "occlusion"]


def test_boosted_engine_list_and_skip_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from face_engine.core import trt_compiler as core

    assert [m for m, _ in btc.BOOSTED_ENGINES] == ["retinaface_r50", "hyperswap_1a_256",
                                                   "xseg_3", "gpen_bfr_1024"]
    assert all(p == "fp16" for _, p in btc.BOOSTED_ENGINES)
    assert core.ENGINE_SPECS["gpen_bfr_1024"].fp32_layers  # the measured overflow blocks
    built = []
    monkeypatch.setattr(core, "discover_gpu", lambda device_id=0: SimpleNamespace(sm="89"))
    monkeypatch.setattr(core, "find_engine", lambda model, *a, **k: Path(f"{model}.engine"))
    monkeypatch.setattr(core, "build_engine", lambda *a, **k: built.append(a))
    monkeypatch.setattr("face_engine.models.zoo.build_default_registry",
                        lambda: SimpleNamespace(ensure=lambda name, **k: Path(f"{name}.onnx")))
    reports = btc.compile_all_engines()
    assert [r.status for r in reports] == ["ready"] * 4 and built == []


# ---------------------------------------------------------------------------- GPU
@cuda
def test_feather_erodes_then_blurs_within_unit_range() -> None:
    m = torch.zeros(1, 1, 64, 64, device="cuda")
    m[..., 16:48, 16:48] = 1.0
    out = feather(m)
    assert out.shape == m.shape and float(out.min()) >= 0 and float(out.max()) <= 1
    assert float(out.sum()) < float(m.sum())          # the edge moved inward
    assert float(out[0, 0, 32, 32]) > 0.99            # the interior is untouched
    assert float(out[0, 0, 16, 32]) < 0.5             # the old edge is feathered away


@pytest.fixture(scope="module")
def sample() -> tuple[np.ndarray, Any, Any]:
    import cv2
    import insightface

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.models.zoo import build_default_registry

    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    registry = build_default_registry()
    return image, registry, ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT,
                                                                    Provider.CUDA]))


@cuda
@pytest.mark.gpu
def test_r50_finds_the_six_sample_faces_like_scrfd(sample: tuple[Any, Any, Any]) -> None:
    from torchvision.ops import box_iou

    from face_engine.pipeline.detector import SCRFDDetector

    image, registry, engine = sample
    frame = torch.from_numpy(image).cuda().permute(2, 0, 1)[None]
    r50 = rg.RetinaFaceR50Detector(engine, registry.ensure("retinaface_r50"))
    found = r50.detect_cuda(frame)
    ref = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps")).detect_cuda(frame)
    assert len(found) == len(ref) == 6
    iou = box_iou(found.boxes, ref.boxes).max(1).values
    assert float(iou.min()) > 0.6
    h, w = image.shape[:2]
    kps = found.kps
    assert bool(((kps[..., 0] >= 0) & (kps[..., 0] <= w) & (kps[..., 1] >= 0)
                 & (kps[..., 1] <= h)).all())
    inside = ((kps[..., 0] >= found.boxes[:, None, 0] - 5) & (kps[..., 0] <= found.boxes[:, None, 2] + 5))
    assert bool(inside.all())
    # the host path is the same code
    faces = r50.detect(image)
    assert len(faces) == 6 and np.allclose(faces[0].bbox, found.boxes[0].cpu().numpy(), atol=1e-3)


@cuda
@pytest.mark.gpu
def test_r50_tensorrt_matches_onnx_runtime(sample: tuple[Any, Any, Any]) -> None:
    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine

    image, registry, engine = sample
    trt = rg.RetinaFaceR50Detector(engine, registry.ensure("retinaface_r50"))
    if not trt.uses_tensorrt_engine:
        pytest.skip("no compiled retinaface_r50 engine (tools/compile_engines.py)")
    ort = rg.RetinaFaceR50Detector(ExecutionEngine(EngineConfig(providers=[Provider.CUDA])),
                                   registry.ensure("retinaface_r50"))
    frame = torch.from_numpy(image).cuda().permute(2, 0, 1)[None]
    a, b = trt.detect_cuda(frame), ort.detect_cuda(frame)
    assert len(a) == len(b)
    assert float((a.boxes - b.boxes).abs().max()) < 2.0
    assert float((a.kps - b.kps).abs().max()) < 2.0


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize("preset", ["boosted", "ultra"])
def test_dual_stream_output_equals_single_stream(sample: tuple[Any, Any, Any],
                                                 preset: str) -> None:
    from face_engine.boosted.ultra_pipeline import BoostedFrameProcessor
    from face_engine.processors.swapper import IdentityEncoder
    from face_engine.server.processing import ProcessorConfig, model_paths

    image, registry, engine = sample
    frame = torch.from_numpy(image).cuda().permute(2, 0, 1)[None]
    r50 = rg.RetinaFaceR50Detector(engine, registry.ensure("retinaface_r50"))
    face = r50.detect(image)[0]
    emb = IdentityEncoder(engine, registry.ensure("arcface_w600k_r50")).embed(image, face).embedding
    params = preset_params(preset)
    config = ProcessorConfig(params, model_paths(required_models(params), registry), {"s": emb})
    outs = {}
    for dual in (True, False):
        proc = BoostedFrameProcessor(config, dual_stream=dual)
        try:
            out, stats = proc.process_tensor(frame)
            torch.cuda.synchronize()
            outs[dual] = (out, stats)
        finally:
            proc.close()
    assert outs[True][1].swapped == outs[False][1].swapped == 6
    assert torch.equal(outs[True][0], outs[False][0])
    changed = float(((outs[True][0] - frame.float()).abs().amax(1) > 2).float().mean())
    assert changed > 0.01  # the six face regions were swapped
