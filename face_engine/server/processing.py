"""Per-frame processing shared by the render workers and the live preview.

One :class:`FrameProcessor` implementation serves both, so a preview can never
show something the render would not produce.

Per frame: detect faces (SCRFD) -> pick each face's source identity -> swap
(Pixel Boost) -> composite mask (box x XSeg x BiSeNet, whichever layers are
enabled) -> paste -> optionally restore (GPEN / RestoreFormer++) with the
original frame as the colour reference.

Choosing the source for a face:

* No assignments: every detected face gets the first source ("swap all").
* With assignments (target person -> source): the face's ArcFace embedding is
  compared with each assigned target person; the best match at cosine >=
  ``match_threshold`` gets that person's source, anything else is left alone.
  This is per-frame identity matching with no temporal tracking: a sharply
  turned head can fall below the threshold for a few frames (roop-ultimate
  measured same-person profile frames 0.7-1.0 cosine DISTANCE from frontal).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
    workers: int = Field(default=1, ge=1, le=8)

    @property
    def boost_size(self) -> int | None:
        return None if self.pixel_boost == "none" else int(self.pixel_boost.split("x")[0])


def required_models(params: RenderParams) -> list[str]:
    """Zoo names a render with ``params`` needs."""
    names = ["scrfd_10g_bnkps", "arcface_w600k_r50", params.swapper_model]
    if "occlusion" in params.mask_types:
        names.append("xseg")
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
    """Picklable description of a :class:`FrameProcessor` (crosses process boundaries).

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


class FrameProcessor:
    """Builds the models once, then processes frames."""

    def __init__(self, config: ProcessorConfig) -> None:
        from face_engine.core.config import EngineConfig
        from face_engine.core.execution import ExecutionEngine
        from face_engine.pipeline.detector import SCRFDDetector
        from face_engine.pipeline.masker import (
            BoxMaskConfig,
            CompositeMasker,
            MaskerConfig,
            RegionConfig,
            XSegConfig,
        )
        from face_engine.processors.enhancer import FaceEnhancer
        from face_engine.processors.swapper import (
            FaceSwapper,
            Identity,
            IdentityEncoder,
        )

        if not config.sources:
            raise ValueError("no source identity")
        self.config = config
        p = config.params
        paths = config.model_paths
        self.engine = ExecutionEngine(EngineConfig(providers=_providers(p.execution_provider)))
        self.detector = SCRFDDetector(self.engine, paths["scrfd_10g_bnkps"])
        self.encoder = IdentityEncoder(self.engine, paths["arcface_w600k_r50"])
        self.swapper = FaceSwapper(self.engine, p.swapper_model, paths[p.swapper_model])
        box = BoxMaskConfig(padding_top=p.mask_padding_top, padding_bottom=p.mask_padding_bottom,
                            padding_left=p.mask_padding_left, padding_right=p.mask_padding_right,
                            blur=p.mask_blur if "box" in p.mask_types else 0.0)
        self.masker = CompositeMasker(
            self.engine, paths.get("xseg") if "occlusion" in p.mask_types else None,
            paths.get("bisenet_resnet34") if "region" in p.mask_types else None,
            MaskerConfig(box=box, xseg=XSegConfig(), regions=RegionConfig(), concurrent=False))
        self.enhancer = (FaceEnhancer(self.engine, p.enhancer_model, paths[p.enhancer_model])
                         if p.enhancer_model != "none" else None)
        self.identities = {k: Identity(np.asarray(v, np.float32), 1.0)
                           for k, v in config.sources.items()}
        self._default = next(iter(self.identities))
        refs = [(ref, np.asarray(emb, np.float32)) for ref, emb in config.target_refs.items()
                if ref in config.assignments]
        self._ref_ids = [r for r, _ in refs]
        self._ref_matrix = np.stack([e for _, e in refs]) if refs else None

    def source_for(self, frame: np.ndarray, face: Any) -> Any | None:
        """The identity to put on ``face``, or None to leave it untouched."""
        if self._ref_matrix is None:
            return self.identities[self._default]
        emb = self.encoder.embed(frame, face).embedding
        sims = self._ref_matrix @ emb
        best = int(np.argmax(sims))
        if sims[best] < self.config.params.match_threshold:
            return None
        return self.identities[self.config.assignments[self._ref_ids[best]]]

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, FrameStats]:
        """Return the processed frame (a new array) and what was done."""
        faces = self.detector.detect(frame)
        stats = FrameStats(faces=len(faces))
        out = frame
        swapped = []
        for face in faces:
            identity = self.source_for(frame, face)
            if identity is None:
                continue
            result = self.swapper.swap(frame, face, identity, pixel_boost=self.config.params.boost_size)
            mask = self.masker.generate(frame, face)
            out = self.swapper.paste(out, result, mask.crop_mask)
            swapped.append(face)
        if self.enhancer is not None:
            alpha = self.config.params.enhancer_blend / 100.0
            for face in swapped:
                out = self.enhancer.enhance(out, face, alpha=alpha, reference_frame=frame).frame
        stats.swapped = len(swapped)
        return (out if out is not frame else frame.copy()), stats

    def close(self) -> None:
        self.masker.close()
        self.engine.close()


@dataclass
class FaceSwapWorker:
    """:class:`~face_engine.media.ipc_pool.FramePipeline` worker (picklable)."""

    config: ProcessorConfig

    def init(self) -> FrameProcessor:
        return FrameProcessor(self.config)

    def __call__(self, src: np.ndarray, dst: np.ndarray, seq: int,
                 state: FrameProcessor) -> None:
        out, _ = state.process(src)
        dst[...] = out


def model_paths(names: list[str], registry: Any) -> dict[str, str]:
    """Verify (download if needed) and return local paths."""
    return {name: str(Path(registry.ensure(name, show_progress=False))) for name in names}
