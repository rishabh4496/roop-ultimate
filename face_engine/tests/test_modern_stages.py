"""Tests for modernized Stage 1, Stage 2, and Stage 3 components."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from face_engine.core.base_processor import BaseProcessor, ExecutionOptions
from face_engine.core.registry import ModelRegistry, ModelSpec, ModelTask
from face_engine.core.zero_copy_engine import (
    ZeroCopyExecutionEngine,
    normalize_bgr_to_rgb_cuda,
    get_default_face_shape_profile,
)
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.hybrid_masker import (
    HybridVideoMasker,
    soft_dilation_cuda,
    gaussian_blur_cuda,
)
from face_engine.processors.modern_processors import SwapperProcessor, MaskProcessor


def test_registry_has_requested_models_and_metadata():
    reg = build_default_registry()
    catalog = reg.catalog()

    # Requested swappers
    for name in ["inswapper_128", "inswapper_128_fp16", "hyperswap_256", "hififace_256"]:
        assert name in catalog, f"Missing swapper: {name}"
        assert catalog[name]["task"] == "swap"
        assert catalog[name]["native_resolution"] in (128, 256)
        assert "blend_ratio" in catalog[name]["parameters_schema"]

    # Requested maskers
    for name in ["dfl_xseg_v2", "face_parser_bisenet34", "face_occluder_v3", "sam2_hiera_tiny"]:
        assert name in catalog, f"Missing masker: {name}"
        assert catalog[name]["task"] in ("occlusion", "parsing")
        assert catalog[name]["native_resolution"] in (256, 512, 1024)
        assert len(catalog[name]["parameters_schema"]) > 0


def test_base_processor_contract():
    class DummyProcessor(BaseProcessor):
        @property
        def name(self) -> str:
            return "dummy"

        @property
        def task(self) -> str:
            return "test"

        def initialize(self) -> None:
            self._initialized = True

        def process(self, x: np.ndarray) -> np.ndarray:
            return x * 2

        def release(self) -> None:
            self._initialized = False

    proc = DummyProcessor("dummy.onnx")
    assert proc.name == "dummy"
    assert proc.task == "test"
    with proc as p:
        assert p._initialized
        assert p.process(np.array([2]))[0] == 4
    assert not proc._initialized


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU kernels")
def test_cuda_normalization_and_morphology():
    bgr = torch.randint(0, 256, (1, 3, 256, 256), device="cuda", dtype=torch.uint8)
    norm = normalize_bgr_to_rgb_cuda(bgr, fp16=True)
    assert norm.shape == (1, 3, 256, 256)
    assert norm.dtype == torch.float16
    assert norm.device.type == "cuda"

    mask = torch.zeros((1, 1, 256, 256), device="cuda", dtype=torch.float32)
    mask[:, :, 100:150, 100:150] = 1.0
    dilated = soft_dilation_cuda(mask, radius=3)
    blurred = gaussian_blur_cuda(dilated, sigma=1.5)
    assert dilated.shape == (1, 1, 256, 256)
    assert blurred.shape == (1, 1, 256, 256)
    assert dilated.max().item() == 1.0
    assert blurred.max().item() <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ZeroCopyEngine")
def test_zero_copy_engine_and_hybrid_masker():
    xseg_path = Path("app/models/xseg_3.onnx")
    if not xseg_path.is_file():
        pytest.skip("xseg_3.onnx not found locally")

    masker = HybridVideoMasker(xseg3_path=xseg_path)
    masker.initialize()

    dummy_crop = np.zeros((256, 256, 3), dtype=np.uint8)
    dummy_kps = np.array([[80, 100], [176, 100], [128, 140], [90, 180], [166, 180]], dtype=np.float32)

    res = masker.step(0, dummy_crop, dummy_kps)
    assert res.shape == (1, 1, 256, 256)
    assert res.device.type == "cuda"
    assert 0.0 <= res.min().item() <= res.max().item() <= 1.0
    masker.release()
