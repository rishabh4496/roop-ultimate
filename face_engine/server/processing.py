"""Per-frame GPU processing shared by the render job and the live preview.

One :class:`GpuFrameProcessor` serves both, so a preview can never show
something the render would not produce. Frames are ``(1, 3, H, W)`` float BGR
tensors on the GPU; nothing is copied to the host between stages.

Per frame: detect faces (SCRFD on the GPU; the render may stride detection
with optical-flow tracking between) -> choose each face's source identity ->
batched swap of every chosen face (Pixel Boost) -> composite mask (box x
XSeg x BiSeNet x valid, whichever layers are enabled) -> paste -> optionally
restore (GPEN / RestoreFormer++) with the original frame as the colour
reference.

Choosing the source for a face:

* No assignments: every detected face gets the first source ("swap all").
* With assignments (target person -> source): every face's ArcFace embedding
  is compared (one batched matmul) with each assigned target person; the best
  match at cosine >= ``match_threshold`` gets that person's source, anything
  else is left alone. Per-frame identity matching: a sharply turned head can
  fall below the threshold for a few frames (roop-ultimate measured
  same-person profile frames 0.7-1.0 cosine DISTANCE from frontal).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

SwapperName = Literal["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "alphaface_256",
                      "inswapper_128"]
PixelBoost = Literal["none", "256x256", "512x512", "1024x1024"]
MaskType = Literal["box", "occlusion", "region"]
EnhancerName = Literal["none", "gpen_bfr_512", "gpen_bfr_1024", "restoreformer_plus_plus"]
ProviderName = Literal["cuda", "tensorrt", "cpu"]


class RenderParams(BaseModel):
    """Everything ``POST /api/pipeline/start`` accepts (and the preview uses)."""

    model_config = ConfigDict(extra="forbid")

    swapper_model: SwapperName = "hyperswap_1a_256"
    pixel_boost: PixelBoost = "none"
    mask_types: list[MaskType] = Field(default_factory=lambda: ["box", "occlusion"])
    enhancer_model: EnhancerName = "none"
    enhancer_blend: int = Field(default=80, ge=0, le=100)
    execution_provider: ProviderName = "tensorrt"
    # Box layer geometry (fractions of the crop) and feathering.
    mask_padding_top: float = Field(default=0.0, ge=0.0, le=0.5)
    mask_padding_bottom: float = Field(default=0.0, ge=0.0, le=0.5)
    mask_padding_left: float = Field(default=0.0, ge=0.0, le=0.5)
    mask_padding_right: float = Field(default=0.0, ge=0.0, le=0.5)
    mask_blur: float = Field(default=0.3, ge=0.0, le=1.0)
    match_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # Render only: full detection every N frames, optical flow between.
    detection_stride: int = Field(default=1, ge=1, le=8)
    # Render only: >1 renders keyframe segments in that many GPU processes.
    workers: int = Field(default=1, ge=1, le=8)

    @property
    def boost_size(self) -> int | None:
        return None if self.pixel_boost == "none" else int(self.pixel_boost.split("x")[0])


# Hardware profile presets. ``measured_fps`` is what the preset rendered on the
# reference machine, not a promise: RTX 4070, 1080p H.264 with two faces per
# frame, steady state after the first batch (face_engine/tests/bench_presets.py,
# 2026-09-28). The spec's targets were 60+ / 30 / 12; measured 43.0 / 34.8 / 8.3.
PRESETS: dict[str, dict[str, Any]] = {
    # HyperSwap, not inswapper: inswapper's TensorRT FP16 engine ran 12.4 ms for
    # two faces vs HyperSwap FP32's 10.1 ms (2026-09-28), so it saves nothing.
    # The savings are the box-only mask (2.1 vs 5.9 ms) and strided detection.
    "ultra_fast": {
        "label": "Ultra Fast",
        "params": {"swapper_model": "hyperswap_1a_256", "pixel_boost": "none",
                   "mask_types": ["box"], "enhancer_model": "none", "detection_stride": 3},
        "measured_fps": 43.0,
    },
    "balanced": {
        "label": "Balanced",
        "params": {"swapper_model": "hyperswap_1a_256", "pixel_boost": "none",
                   "mask_types": ["box", "occlusion"], "enhancer_model": "none",
                   "detection_stride": 1},
        "measured_fps": 34.8,
    },
    "cinema": {
        "label": "High-Fidelity Cinema",
        "params": {"swapper_model": "hyperswap_1a_256", "pixel_boost": "512x512",
                   "mask_types": ["box", "occlusion", "region"],
                   "enhancer_model": "gpen_bfr_512", "enhancer_blend": 80,
                   "detection_stride": 1},
        "measured_fps": 8.3,
    },
}


def required_models(params: RenderParams) -> list[str]:
    """Zoo names a render with ``params`` needs."""
    names = ["scrfd_10g_bnkps", "arcface_w600k_r50", params.swapper_model]
    if "occlusion" in params.mask_types:
        names.append("xseg_3")
    if "region" in params.mask_types:
        names.append("bisenet_resnet34")
    if params.enhancer_model != "none":
        names.append(params.enhancer_model)
    return names


@dataclass
class FrameStats:
    faces: int = 0
    swapped: int = 0


@dataclass
class ProcessorConfig:
    """Picklable description of a :class:`GpuFrameProcessor` (crosses process boundaries).

    Attributes:
        params: Render parameters.
        model_paths: Zoo name -> local path for every model in ``required_models``.
        sources: Source id -> unit 512-d embedding.
        target_refs: Target person id -> unit embedding (from face detection).
        assignments: Target person id -> source id. Empty: swap every face
            with the first source.
    """

    params: RenderParams
    model_paths: dict[str, str]
    sources: dict[str, np.ndarray]
    target_refs: dict[str, np.ndarray] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)


def _providers(name: str) -> list[Any]:
    from face_engine.core.config import Provider

    return {"tensorrt": [Provider.TENSORRT, Provider.CUDA, Provider.CPU],
            "cuda": [Provider.CUDA, Provider.CPU], "cpu": [Provider.CPU]}[name]


class GpuFrameProcessor:
    """Builds the GPU models once, then processes frames that live on the GPU.

    Args:
        config: :class:`ProcessorConfig`.
        device: CUDA device.
        tracking: Use ``params.detection_stride`` (a render walking frames in
            order). The preview jumps around the timeline and always detects.
    """

    def __init__(self, config: ProcessorConfig, device: Any = "cuda",
                 tracking: bool = False) -> None:
        import torch

        from face_engine.core.config import EngineConfig
        from face_engine.core.execution import ExecutionEngine
        from face_engine.pipeline.detector import SCRFDDetector
        from face_engine.pipeline.masker import (
            BoxMaskConfig,
            GPUMasker,
            MaskerConfig,
            RegionConfig,
            XSegConfig,
        )
        from face_engine.pipeline.tracker import StridedFaceTracker, TrackerConfig
        from face_engine.processors import (
            BatchedFaceEnhancer,
            BatchedFaceSwapper,
            GPUIdentityEncoder,
        )

        if not config.sources:
            raise ValueError("no source identity")
        self.config = config
        self.device = torch.device(device)
        p = config.params
        paths = config.model_paths
        self.engine = ExecutionEngine(EngineConfig(providers=_providers(p.execution_provider)))
        # Detection runs on the CUDA EP: its dynamic canvas would rebuild TensorRT
        # engines, and SCRFD is 4.7 ms per frame there (Stage 2).
        self.detect_engine = ExecutionEngine(EngineConfig(providers=_providers(
            "cpu" if p.execution_provider == "cpu" else "cuda")))
        self.detector = SCRFDDetector(self.detect_engine, paths["scrfd_10g_bnkps"])
        self.tracker = (StridedFaceTracker(self.detector,
                                           TrackerConfig(detection_stride=p.detection_stride))
                        if tracking and p.detection_stride > 1 else None)
        self.encoder = GPUIdentityEncoder(self.engine, paths["arcface_w600k_r50"])
        self.swapper = BatchedFaceSwapper(self.engine, p.swapper_model, paths[p.swapper_model])
        size = p.boost_size or self.swapper.spec.size
        box = BoxMaskConfig(padding_top=p.mask_padding_top, padding_bottom=p.mask_padding_bottom,
                            padding_left=p.mask_padding_left, padding_right=p.mask_padding_right,
                            blur=p.mask_blur if "box" in p.mask_types else 0.0)
        # The mask must use the swap crop's own template (arcface_128), at the
        # Pixel Boost size, so the two line up pixel for pixel.
        self.masker = GPUMasker(
            self.engine, paths.get("xseg_3") if "occlusion" in p.mask_types else None,
            paths.get("bisenet_resnet34") if "region" in p.mask_types else None,
            MaskerConfig(crop_size=size, template=self.swapper.spec.template, box=box,
                         xseg=XSegConfig(), regions=RegionConfig(), concurrent=False))
        self.enhancer = (BatchedFaceEnhancer(self.engine, p.enhancer_model,
                                             paths[p.enhancer_model])
                         if p.enhancer_model != "none" else None)
        ids = list(config.sources)
        self._source_ids = ids
        self._sources = torch.as_tensor(np.stack([np.asarray(config.sources[i], np.float32)
                                                  for i in ids]), device=self.device)
        refs = [r for r in config.target_refs if r in config.assignments]
        self._refs = (torch.as_tensor(np.stack([np.asarray(config.target_refs[r], np.float32)
                                                for r in refs]), device=self.device)
                      if refs else None)
        self._ref_source = (torch.as_tensor([ids.index(config.assignments[r]) for r in refs],
                                            device=self.device) if refs else None)

    # ------------------------------------------------------------------ faces
    def _faces(self, frame: torch.Tensor) -> Any:
        if self.tracker is not None:
            return self.tracker.update(frame).detections
        return self.detector.detect_cuda(frame)

    def _choose_sources(self, frame: torch.Tensor, kps: torch.Tensor) -> tuple[Any, Any]:
        """``(kept_face_indices, source_embeddings)`` for the faces to swap."""
        import torch

        n = kps.shape[0]
        if self._refs is None:
            return torch.arange(n, device=self.device), self._sources[:1].expand(n, -1)
        emb = self.encoder.embed(frame, kps)
        sims = emb @ self._refs.T  # (faces, refs)
        best_sim, best = sims.max(dim=1)
        keep = torch.nonzero(best_sim >= self.config.params.match_threshold)[:, 0]
        return keep, self._sources[self._ref_source[best[keep]]]

    # ------------------------------------------------------------------ frame
    def process_tensor(self, frame: torch.Tensor) -> tuple[torch.Tensor, FrameStats]:
        """``(1, 3, H, W)`` BGR frame on the GPU -> processed float frame + stats."""
        from face_engine.pipeline.aligner import warp_face_inverse_cuda

        f = frame.float()
        faces = self._faces(f)
        stats = FrameStats(faces=len(faces))
        if len(faces) == 0:
            return f, stats
        keep, sources = self._choose_sources(f, faces.kps)
        if keep.shape[0] == 0:
            return f, stats
        kps = faces.kps[keep]
        p = self.config.params
        result = self.swapper.swap(f, kps, source=sources, pixel_boost=p.boost_size)
        mask = self.masker.generate(f, kps).mask * result.ok.float().view(-1, 1, 1, 1)
        out = warp_face_inverse_cuda(f, result.crops, result.matrices, mask)
        if self.enhancer is not None:
            out = self.enhancer.enhance(out, kps, reference=f,
                                        alpha=p.enhancer_blend / 100.0).frames
        stats.swapped = int(keep.shape[0])
        return out, stats

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, FrameStats]:
        """Host convenience: ``(H, W, 3)`` uint8 BGR in and out (image targets)."""
        import torch

        t = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device).permute(2, 0, 1)[None]
        out, stats = self.process_tensor(t)
        return out[0].round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy(), stats

    def close(self) -> None:
        self.engine.close()
        self.detect_engine.close()


def pool_processor(device: Any = "cuda", **config: Any) -> Any:
    """``SegmentWorkerPool`` factory: a :class:`GpuFrameProcessor` over batches."""
    import torch

    cfg = ProcessorConfig(params=RenderParams.model_validate(config["params"]),
                          model_paths=config["model_paths"],
                          sources={k: np.asarray(v, np.float32)
                                   for k, v in config["sources"].items()},
                          target_refs={k: np.asarray(v, np.float32)
                                       for k, v in config["target_refs"].items()},
                          assignments=config["assignments"])
    processor = GpuFrameProcessor(cfg, device=device, tracking=True)

    def run(batch: Any) -> torch.Tensor:
        return torch.cat([processor.process_tensor(f[None])[0] for f in batch.frames])

    return run


def config_kwargs(config: ProcessorConfig) -> dict[str, Any]:
    """:class:`ProcessorConfig` as plain data for :func:`pool_processor`."""
    return {"params": config.params.model_dump(), "model_paths": config.model_paths,
            "sources": {k: np.asarray(v).tolist() for k, v in config.sources.items()},
            "target_refs": {k: np.asarray(v).tolist() for k, v in config.target_refs.items()},
            "assignments": config.assignments}


def model_paths(names: list[str], registry: Any) -> dict[str, str]:
    """Verify (download if needed) and return local paths."""
    return {name: str(Path(registry.ensure(name, show_progress=False))) for name in names}
