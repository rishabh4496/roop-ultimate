"""Processor implementation for the VFX / Master Quality Tile Diffusion Synthesizer.

Runs overlapping tiled inference (512x512 with 64px Gaussian feathered margins)
constrained strictly to the high-frequency residual domain:
- Global geometry and color are fixed from the swapper.
- High-frequency micro-textures (skin pores, hair follicles, lip crevices, sclera vessels)
  are synthesized via few-step latent diffusion.
- Tiled execution guarantees an 8 GB VRAM budget is never exceeded, even on 4K crops.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

import roop.globals
from roop.typing import Face, Frame, FaceSet
from roop.processors.enhance_common import is_usable, sized, exclusive
from roop.tile_diffusion import (
    TileInferenceManager,
    FrequencyResidualSynthesizer,
    TileDiffusionEngine
)


class Enhance_TileDiffusion:
    processorname = 'tile_diffusion'
    self_excluding = True
    type = 'enhance'
    model_template = 'ffhq_512'

    _session_lock = threading.Lock()

    def __init__(self):
        self.plugin_options: Optional[Dict[str, Any]] = None
        self.devicename: str = 'cuda'
        self.engine: Optional[TileDiffusionEngine] = None
        self.tiler: Optional[TileInferenceManager] = None
        self._lock = threading.Lock()

    def Initialize(self, plugin_options: Dict[str, Any]):
        with self._lock:
            self.plugin_options = plugin_options
            dev = plugin_options.get("devicename", "cuda")
            if "mps" in dev:
                dev = "cpu"
            self.devicename = dev

            steps = int(getattr(roop.globals, 'tile_diffusion_steps', 2) or 2)
            tile_size = int(getattr(roop.globals, 'tile_diffusion_tile_size', 512) or 512)
            overlap = int(getattr(roop.globals, 'tile_diffusion_overlap', 64) or 64)

            self.engine = TileDiffusionEngine(
                device=self.devicename,
                fp16=True,
                vram_budget_gb=8.0,
                steps=steps
            )
            self.engine.initialize()
            self.tiler = TileInferenceManager(tile_size=tile_size, overlap=overlap, vram_budget_gb=8.0)

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Tuple[Frame, int]:
        """Execute high-fidelity diffusion tile restoration pass with micro-texture injection."""
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1

        input_size = temp_frame.shape[1]
        fallback_bgr = temp_frame

        if self.engine is None or self.tiler is None:
            self.Initialize({"devicename": getattr(roop.globals, "execution_providers", ["CUDAExecutionProvider"])[0]})

        # Configurable restoration parameters
        strength = float(getattr(roop.globals, 'tile_diffusion_strength', 0.65))
        steps = int(getattr(roop.globals, 'tile_diffusion_steps', 2))
        if self.engine:
            self.engine.steps = steps

        # 1. Overlapping tiled inference (512x512 with 64px Gaussian feathered margins)
        # Keeps peak VRAM strictly under 8 GB regardless of frame/crop resolution (even 4K).
        with self._session_lock:
            tiled_output = self.tiler.process_tiled(
                temp_frame,
                tile_fn=lambda tile: self.engine.infer_tile(tile)
            )

        if not is_usable(tiled_output):
            print("[TileDiffusion] Non-finite output in tile diffusion — using unenhanced crop")
            return sized(fallback_bgr, input_size)

        # 2. High-Frequency Residual Injection:
        # Constrain diffusion pass strictly to the high-frequency residual domain:
        # Keep low-frequency color and global geometry fixed from the swapper,
        # while injecting synthesized micro-textures (skin pores, hair follicles, lip crevices, sclera vessels).
        restored = FrequencyResidualSynthesizer.inject_residual(
            swapper_crop=temp_frame,
            diffusion_output=tiled_output,
            strength=strength,
            sigma=2.5
        )

        return sized(restored, input_size)

    def Release(self):
        with self._lock:
            if self.engine is not None:
                self.engine.release()
                self.engine = None
            self.tiler = None
            self.plugin_options = None
