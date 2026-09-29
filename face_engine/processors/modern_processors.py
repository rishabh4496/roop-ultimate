"""Modernized concrete processors adhering to BaseProcessor.

Adapts model zoo specifications into standardized BaseProcessor instances
for swappers (inswapper, hyperswap, hififace) and maskers (xseg, bisenet, sam2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from face_engine.core.base_processor import BaseProcessor, ExecutionOptions
from face_engine.core.execution import ExecutionEngine, ManagedSession
from face_engine.core.config import EngineConfig, Provider
from face_engine.pipeline.aligner import (
    estimate_similarity_transform,
    template_points,
    warp_face_by_translation,
)
from face_engine.pipeline.detector import Face, as_bgr


class SwapperProcessor(BaseProcessor):
    """Concrete BaseProcessor for face swapper ONNX models."""

    def __init__(
        self,
        model_path: str | Path,
        execution_provider: str = "TensorrtExecutionProvider",
        execution_options: ExecutionOptions | Mapping[str, Any] | None = None,
        model_name: str = "hyperswap_256",
    ) -> None:
        super().__init__(model_path, execution_provider, execution_options)
        self._model_name = model_name
        self._engine: ExecutionEngine | None = None
        self._session: ManagedSession | None = None
        self._emap: np.ndarray | None = None

        # Resolve native resolution from model name / file
        name_lower = str(model_name).lower()
        if "128" in name_lower:
            self._res = 128
        elif "512" in name_lower:
            self._res = 512
        else:
            self._res = 256

        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._denormalize = True
        self._embedding_mode = "normed"

        if "inswapper" in name_lower:
            self._mean = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1, 1)
            self._std = np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape(3, 1, 1)
            self._denormalize = False
            self._embedding_mode = "normed_emap"
        elif "hififace" in name_lower:
            self._embedding_mode = "converted_norm"

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def task(self) -> str:
        return "swap"

    @property
    def native_resolution(self) -> int:
        return self._res

    @property
    def metadata(self) -> dict[str, Any]:
        meta = super().metadata
        meta.update({
            "parameters_schema": {
                "blend_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0, "label": "Blend Ratio"},
                "verify_tol": {"type": "float", "min": 0.3, "max": 1.2, "default": 0.79 if self._res == 256 else 0.65, "label": "Outcome Guard Tolerance"},
            }
        })
        return meta

    def initialize(self) -> None:
        if self._initialized:
            return

        provider_enum = Provider.TENSORRT if "Tensorrt" in self.execution_provider else Provider.CUDA
        cfg = EngineConfig(
            device_id=self.execution_options.device_id,
            providers=[provider_enum, Provider.CUDA, Provider.CPU],
        )
        self._engine = ExecutionEngine(cfg)
        self._session = self._engine.get_session(
            self.model_path,
            trt_fp16=self.execution_options.trt_fp16_enable,
        )
        self._initialized = True

    def process(self, target_crop: np.ndarray, source_latent: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Runs the swap on a pre-aligned target crop."""
        if not self._initialized:
            self.initialize()

        rgb = target_crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(((rgb - self._mean) / self._std)[None], dtype=np.float32)

        feeds: dict[str, Any] = {}
        for inp_name in self._session.input_names:
            if inp_name == "source":
                feeds[inp_name] = source_latent.reshape(1, -1).astype(np.float32)
            else:
                feeds[inp_name] = blob

        outs = self._session.session.run(None, feeds)
        raw_img = np.asarray(outs[0], dtype=np.float32).reshape(3, self._res, self._res)
        if self._denormalize:
            raw_img = (raw_img + 1.0) * 0.5
        bgr = raw_img.transpose(1, 2, 0)[:, :, ::-1] * 255.0
        swapped = np.clip(bgr, 0, 255).astype(np.uint8)

        mask = None
        if len(outs) > 1 and np.asarray(outs[1]).size == self._res * self._res:
            mask = np.clip(np.asarray(outs[1], np.float32).reshape(self._res, self._res), 0.0, 1.0)

        return swapped, mask

    def release(self) -> None:
        if self._engine:
            self._engine.release()
            self._engine.cleanup_vram()
            self._engine = None
        self._session = None
        self._initialized = False


class MaskProcessor(BaseProcessor):
    """Concrete BaseProcessor for facial occlusion and parsing models."""

    def __init__(
        self,
        model_path: str | Path,
        execution_provider: str = "TensorrtExecutionProvider",
        execution_options: ExecutionOptions | Mapping[str, Any] | None = None,
        model_name: str = "face_occluder_v3",
    ) -> None:
        super().__init__(model_path, execution_provider, execution_options)
        self._model_name = model_name
        self._engine: ExecutionEngine | None = None
        self._session: ManagedSession | None = None

        name_lower = str(model_name).lower()
        if "bisenet" in name_lower or "512" in name_lower:
            self._res = 512
            self._task = "parsing"
        elif "sam2" in name_lower:
            self._res = 1024
            self._task = "occlusion"
        else:
            self._res = 256
            self._task = "occlusion"

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def task(self) -> str:
        return self._task

    @property
    def native_resolution(self) -> int:
        return self._res

    @property
    def metadata(self) -> dict[str, Any]:
        meta = super().metadata
        meta.update({
            "parameters_schema": {
                "mask_blur": {"type": "float", "min": 0.0, "max": 64.0, "default": 12.0, "label": "Mask Blur (px)"},
                "feathering": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0, "label": "Feathering Sigma"},
                "threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35, "label": "Threshold"},
            }
        })
        return meta

    def initialize(self) -> None:
        if self._initialized:
            return

        provider_enum = Provider.TENSORRT if "Tensorrt" in self.execution_provider else Provider.CUDA
        cfg = EngineConfig(
            device_id=self.execution_options.device_id,
            providers=[provider_enum, Provider.CUDA, Provider.CPU],
        )
        self._engine = ExecutionEngine(cfg)
        self._session = self._engine.get_session(
            self.model_path,
            trt_fp16=self.execution_options.trt_fp16_enable,
        )
        self._initialized = True

    def process(self, crop: np.ndarray, threshold: float = 0.35, feather_sigma: float = 1.0) -> np.ndarray:
        """Run mask extraction on aligned face crop."""
        if not self._initialized:
            self.initialize()

        s = self._res
        resized = cv2.resize(crop, (s, s), interpolation=cv2.INTER_CUBIC)

        if self._task == "occlusion":
            # NHWC [0, 1] BGR input
            blob = (resized.astype(np.float32) / 255.0)[None, ...]
            inp_name = self._session.input_names[0]
            outs = self._session.session.run(None, {inp_name: blob})
            raw_mask = np.asarray(outs[0]).reshape(s, s)
            # High on visible face; invert so high = occluder
            mask = 1.0 - np.clip(raw_mask, 0.0, 1.0)
            if threshold > 0:
                mask = np.where(mask > threshold, mask, 0.0)
            if feather_sigma > 0:
                k = int(round(feather_sigma * 4)) | 1
                mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), feather_sigma)
            return np.clip(mask, 0.0, 1.0).astype(np.float32)

        elif self._task == "parsing":
            # NCHW ImageNet normalized
            rgb = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1) * 255.0
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1) * 255.0
            blob = ((rgb - mean) / std)[None, ...]
            inp_name = self._session.input_names[0]
            outs = self._session.session.run(None, {inp_name: blob})
            labels = np.asarray(outs[0]).argmax(axis=1)[0]  # (512, 512)
            # Retain standard face skin + eyes + nose + lips classes (1..13 excluding glasses/hair/earring)
            face_classes = {1, 2, 3, 4, 5, 10, 11, 12, 13}
            mask = np.isin(labels, list(face_classes)).astype(np.float32)
            if feather_sigma > 0:
                k = int(round(feather_sigma * 4)) | 1
                mask = cv2.GaussianBlur(mask, (k, k), feather_sigma)
            return np.clip(mask, 0.0, 1.0)

        return np.ones((s, s), dtype=np.float32)

    def release(self) -> None:
        if self._engine:
            self._engine.release()
            self._engine.cleanup_vram()
            self._engine = None
        self._session = None
        self._initialized = False
