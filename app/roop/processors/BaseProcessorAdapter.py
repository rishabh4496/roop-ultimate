"""Bridge adapter wrapping BaseProcessor for ProcessMgr plugin compatibility."""
from __future__ import annotations

from typing import Any
import numpy as np

from face_engine.core.base_processor import BaseProcessor, ExecutionOptions
from face_engine.models.zoo import build_default_registry


class BaseProcessorPluginAdapter:
    """Wraps any BaseProcessor to implement the legacy roop Processor contract.

    Provides Initialize(plugin_options), Run(img1, keywords), processorname, type,
    and Release() for seamless routing inside ProcessMgr.
    """

    def __init__(self, processor_name: str, processor_type: str = "mask") -> None:
        self.processorname = processor_name
        self.type = processor_type
        self.plugin_options: dict[str, Any] = {}
        self._processor: BaseProcessor | None = None
        self._registry = build_default_registry()

    def Initialize(self, plugin_options: dict[str, Any]) -> None:
        self.plugin_options = dict(plugin_options or {})
        devicename = self.plugin_options.get("devicename", "cuda").lower()
        provider = "TensorrtExecutionProvider" if "trt" in devicename or "tensorrt" in devicename else "CUDAExecutionProvider"

        # Resolve model file
        model_name = self.processorname
        if model_name.startswith("mask_"):
            model_name = model_name[5:]

        # Name mapping
        name_map = {
            "xseg3": "face_occluder_v3",
            "xseg": "dfl_xseg_v2",
            "faceparser": "face_parser_bisenet34",
            "occluder": "face_occluder_v3",
            "sam2": "sam2_hiera_tiny",
        }
        resolved_name = name_map.get(model_name, model_name)

        if resolved_name in self._registry:
            model_path = self._registry.local_path(resolved_name)
        else:
            # Fallback path lookup
            from roop.utilities import resolve_relative_path
            model_path = resolve_relative_path(f"../models/{model_name}.onnx")

        exec_opts = ExecutionOptions(
            device_id=self.plugin_options.get("cuda_device_id", 0),
            provider=provider,
            trt_fp16_enable=bool(self.plugin_options.get("trt_fp16_enable", True)),
        )

        if self.type == "swap":
            from face_engine.processors.modern_processors import SwapperProcessor
            self._processor = SwapperProcessor(
                model_path=model_path,
                execution_provider=provider,
                execution_options=exec_opts,
                model_name=resolved_name,
            )
        else:
            from face_engine.processors.modern_processors import MaskProcessor
            self._processor = MaskProcessor(
                model_path=model_path,
                execution_provider=provider,
                execution_options=exec_opts,
                model_name=resolved_name,
            )

        self._processor.initialize()

    def Run(self, img1: np.ndarray, keywords: str = "") -> np.ndarray:
        if self._processor is None:
            return np.ones(img1.shape[:2], dtype=np.float32)

        if self.type == "mask":
            blur = float(self.plugin_options.get("mask_blur", 12.0) or 12.0)
            threshold = float(self.plugin_options.get("threshold", 0.35) or 0.35)
            return self._processor.process(img1, threshold=threshold, feather_sigma=blur / 4.0)

        return img1

    def Release(self) -> None:
        if self._processor is not None:
            self._processor.release()
            self._processor = None
