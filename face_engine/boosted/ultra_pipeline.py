"""The boosted render: R50 + HyperSwap-256 || DFL XSeg on two CUDA streams (+ GPEN-1024 "Ultra").

Two presets (:data:`BOOSTED_PRESETS`), one processor:

* ``boosted``: RetinaFace R50 (TensorRT FP16) every 3rd frame with optical-flow
  tracking between, HyperSwap-1a-256 (pinned FP16 engine), box x XSeg mask
  (FP16 engine, eroded + feathered: :mod:`.xseg_masker`), no restorer.
* ``ultra``: the same plus GPEN-BFR-1024 on every face: the mixed FP16 engine
  (encoder final linear + style pixel-norm pinned to FP32; 32.7 vs 57.2 ms per
  face FP32, identity 0.9344 vs 0.9343), which re-crops the swapped face at
  1024 with GPU bicubic, and a LAB colour transfer against the original frame
  (``ColorMode.LAB_MEAN``, the Stage 3 choice: REINHARD's std matching cost
  11% of the restored detail).

:class:`BoostedFrameProcessor` is a :class:`GpuFrameProcessor` (so the Stage 7
:class:`~face_engine.core.cuda_streams.CUDAStreamPipeline` drives it: software
decode + GPU colour, inference, compositing, NVENC ``-preset p4 -cq 19``,
remux). What it changes is :meth:`_swap_and_mask`: after detection, the swap
runs on ``stream_swap`` and the mask (crop warp + XSeg engine + feathering)
on ``stream_mask``, both waiting on one event recorded on the inference
stream, which waits on both before blending. Tensors cross streams with
``record_stream`` so the caching allocator cannot recycle them early.

VRAM: :func:`render` caps torch's caching allocator at 90% of the device
(``torch.cuda.set_per_process_memory_fraction(0.90)``), so a runaway
allocation fails in torch instead of spilling into shared system memory on
Windows. TensorRT engines and ONNX Runtime allocate outside torch and are not
covered by that cap; the benchmark reports device-level peak VRAM for that
reason.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from face_engine.server.processing import (
    GpuFrameProcessor,
    ProcessorConfig,
    RenderParams,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

VRAM_FRACTION = 0.90
_COMMON = {"detector_model": "retinaface_r50", "swapper_model": "hyperswap_1a_256",
           "pixel_boost": "none", "mask_types": ["box", "occlusion"], "detection_stride": 3}
BOOSTED_PRESETS: dict[str, dict[str, Any]] = {
    "boosted": {**_COMMON, "enhancer_model": "none"},
    "ultra": {**_COMMON, "enhancer_model": "gpen_bfr_1024", "enhancer_blend": 80},
}


def preset_params(name: str) -> RenderParams:
    return RenderParams(**BOOSTED_PRESETS[name])


class BoostedFrameProcessor(GpuFrameProcessor):
    """:class:`GpuFrameProcessor` with the swap and the XSeg mask on two CUDA streams.

    Args (beyond the parent's):
        dual_stream: False runs swap then mask on the calling stream (the A/B arm).
    """

    def __init__(self, config: ProcessorConfig, device: Any = "cuda", tracking: bool = False,
                 dual_stream: bool = True) -> None:
        import torch

        super().__init__(config, device=device, tracking=tracking)
        self.dual_stream = dual_stream
        self.stream_swap = torch.cuda.Stream(device=self.device)
        self.stream_mask = torch.cuda.Stream(device=self.device)

    def _build_masker(self, cls: Any, xseg: str | None, bisenet: str | None, config: Any) -> Any:
        from face_engine.boosted.xseg_masker import XSegFeatherMasker

        return XSegFeatherMasker(self.engine, xseg, bisenet, config)

    def _swap_and_mask(self, f: torch.Tensor, kps: torch.Tensor,
                       sources: torch.Tensor) -> tuple[Any, torch.Tensor]:
        import torch

        if not self.dual_stream:
            return super()._swap_and_mask(f, kps, sources)
        main = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(main)
        for t in (f, kps, sources):  # made on `main`, read on both side streams
            t.record_stream(self.stream_swap)
            t.record_stream(self.stream_mask)
        with torch.cuda.stream(self.stream_swap):
            self.stream_swap.wait_event(ready)
            result = self.swapper.swap(f, kps, source=sources,
                                       pixel_boost=self.config.params.boost_size)
            swapped = torch.cuda.Event()
            swapped.record(self.stream_swap)
        with torch.cuda.stream(self.stream_mask):
            self.stream_mask.wait_event(ready)
            mask = self.masker.generate(f, kps).mask
            masked = torch.cuda.Event()
            masked.record(self.stream_mask)
        main.wait_event(swapped)
        main.wait_event(masked)
        for t in (result.crops, result.target_crops, result.matrices, result.model_mask,
                  result.ok, mask):
            if t is not None:
                t.record_stream(main)
        return result, mask


def render(target: str | Path, output: str | Path, source_image: str | Path,
           preset: str = "boosted", *, dual_stream: bool = True, max_frames: int | None = None,
           on_progress: Any = None) -> Any:
    """Render ``target`` with ``preset`` through the Stage 7 stream pipeline; returns its stats."""
    import torch

    from face_engine.benchmark import source_embedding
    from face_engine.core.cuda_streams import CUDAStreamPipeline
    from face_engine.core.execution import register_gpu_runtime_dirs
    from face_engine.models.zoo import build_default_registry
    from face_engine.server.processing import model_paths, required_models

    register_gpu_runtime_dirs()
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    params = preset_params(preset)
    registry = build_default_registry()
    config = ProcessorConfig(params, model_paths(required_models(params), registry),
                             {"source": source_embedding(Path(source_image), Path(target),
                                                         registry)})
    processor = BoostedFrameProcessor(config, tracking=True, dual_stream=dual_stream)
    try:
        return CUDAStreamPipeline().run(target, output, processor, max_frames=max_frames,
                                        on_progress=on_progress)
    finally:
        processor.close()
