"""The boosted pipeline: RetinaFace R50 + HyperSwap-256 + DFL XSeg (+ GPEN-1024 "Ultra").

A thin layer over the rest of ``face_engine``: new here are the RetinaFace R50
GPU detector (:mod:`.retinaface_gpu`), XSeg with morphological feathering
(:mod:`.xseg_masker`), a frame processor that runs the swap and the mask on
two CUDA streams (:mod:`.ultra_pipeline`), its presets and its benchmark
(:mod:`.benchmark_boosted`). Engines come from ``tools/compile_engines.py``
(:mod:`.trt_compiler` builds the four this pipeline uses); decoding, the
stream pipeline, NVENC and colour transfer are the Stage 3-8 modules.
"""
from __future__ import annotations

from face_engine.boosted.retinaface_gpu import RetinaFaceR50Detector, strided_tracker

__all__ = ["RetinaFaceR50Detector", "strided_tracker"]
