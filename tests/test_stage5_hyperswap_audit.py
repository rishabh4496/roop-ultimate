"""Unit and Integration Tests for Stage 5 — HyperSwap Quality and Performance Audit."""

import os
import sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, "app")
for p in (ROOT, APP):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop.hyperswap_optimizer import (
    AdaptiveVRAMBatcher,
    HyperSwapQualityAuditor,
    HyperSwapSourceCache,
    HyperSwapVRAMProfile,
    get_hyperswap_source_cache,
)


class DummyFace(dict):
    """Mock Face object simulating InsightFace Face container."""
    def __init__(self, embedding):
        super().__init__()
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self["embedding"] = self.embedding

    @property
    def normed_embedding(self):
        norm = np.linalg.norm(self.embedding)
        return self.embedding / norm if norm > 1e-12 else self.embedding


# =====================================================================
# 1. Source Latent Caching Verification
# =====================================================================

def test_source_latent_caching_eliminates_redundancy():
    cache = HyperSwapSourceCache(max_size=32)
    cache.clear()

    np.random.seed(42)
    raw_emb = np.random.normal(0, 1.0, 512).astype(np.float32)
    face = DummyFace(raw_emb)

    # First call: cache miss, compute and cache
    latent1 = cache.get_latent(face, model_key="hyperswap", embedding_mode="normed")
    assert cache.misses == 1
    assert cache.hits == 0
    assert latent1.shape == (1, 512)
    assert np.isclose(np.linalg.norm(latent1), 1.0, atol=1e-5)

    # Second call on same face: cache hit from face object dictionary
    latent2 = cache.get_latent(face, model_key="hyperswap", embedding_mode="normed")
    assert cache.hits == 1
    assert np.array_equal(latent1, latent2)

    # Third call with fresh face container but identical embedding: cache hit from internal LRU
    fresh_face = DummyFace(raw_emb.copy())
    latent3 = cache.get_latent(fresh_face, model_key="hyperswap", embedding_mode="normed")
    assert cache.hits == 2
    assert np.array_equal(latent1, latent3)


def test_source_latent_caching_emap_mode():
    cache = HyperSwapSourceCache(max_size=32)
    cache.clear()

    np.random.seed(42)
    raw_emb = np.random.normal(0, 1.0, 512).astype(np.float32)
    face = DummyFace(raw_emb)
    emap = np.eye(512, dtype=np.float32)

    l1 = cache.get_latent(face, model_key="inswapper", embedding_mode="normed_emap", emap=emap)
    l2 = cache.get_latent(face, model_key="inswapper", embedding_mode="normed_emap", emap=emap)
    assert cache.hits == 1
    assert np.array_equal(l1, l2)


# =====================================================================
# 2. Adaptive VRAM Batching Logic
# =====================================================================

def test_adaptive_vram_telemetry_live():
    profile = AdaptiveVRAMBatcher.get_vram_telemetry(device_id=0)
    assert isinstance(profile, HyperSwapVRAMProfile)
    assert profile.total_vram_mb > 0.0
    assert profile.recommended_batch_size in (1, 2, 4, 8)


def test_adaptive_vram_batch_constraints_low_tier():
    # Simulate low VRAM device (< 7000 MB total, e.g. RTX 3060 Laptop 6GB)
    # The batcher must strictly enforce recommended_batch_size = 1
    from unittest.mock import patch
    with patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_mem:
        class MockMem:
            total = int(6.0 * 1024 * 1024 * 1024)  # 6GB
            free = int(3.5 * 1024 * 1024 * 1024)   # 3.5GB
            used = int(2.5 * 1024 * 1024 * 1024)
        mock_mem.return_value = MockMem()
        profile = AdaptiveVRAMBatcher.get_vram_telemetry(device_id=0)
        assert profile.is_low_vram_tier is True
        assert profile.recommended_batch_size == 1


def test_adaptive_vram_batch_constraints_high_tier():
    # Simulate high VRAM device (12 GB total, > 7GB free, e.g. RTX 4070 Desktop)
    from unittest.mock import patch
    with patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_mem:
        class MockMem:
            total = int(12.0 * 1024 * 1024 * 1024)  # 12GB
            free = int(8.5 * 1024 * 1024 * 1024)    # 8.5GB free
            used = int(3.5 * 1024 * 1024 * 1024)
        mock_mem.return_value = MockMem()
        profile = AdaptiveVRAMBatcher.get_vram_telemetry(device_id=0)
        assert profile.is_low_vram_tier is False
        assert profile.recommended_batch_size == 8


# =====================================================================
# 3. Quality & Geometric Verification Metrics
# =====================================================================

def test_identity_similarity_metric():
    np.random.seed(99)
    emb1 = np.random.normal(0, 1.0, 512).astype(np.float32)
    # Identical
    sim_self = HyperSwapQualityAuditor.compute_identity_similarity(emb1, emb1)
    assert np.isclose(sim_self, 1.0, atol=1e-5)

    # Orthogonal
    emb2 = np.random.normal(0, 1.0, 512).astype(np.float32)
    emb_orth = emb2 - (np.dot(emb2, emb1) / np.dot(emb1, emb1)) * emb1
    sim_orth = HyperSwapQualityAuditor.compute_identity_similarity(emb1, emb_orth)
    assert abs(sim_orth) < 1e-4


def test_skin_detail_laplacian_variance():
    # Uniform flat image -> 0 variance
    flat = np.full((128, 128, 3), 120, dtype=np.uint8)
    var_flat = HyperSwapQualityAuditor.measure_skin_detail(flat)
    assert var_flat == 0.0

    # Textured image -> positive variance
    textured = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    var_tex = HyperSwapQualityAuditor.measure_skin_detail(textured)
    assert var_tex > 100.0


def test_geometric_alignment_error_metric():
    kps_target = np.array([
        [30.0, 40.0],
        [70.0, 40.0],
        [50.0, 60.0],
        [35.0, 80.0],
        [65.0, 80.0]
    ], dtype=np.float32)

    # Shifted swapped face (+2 px shift)
    kps_swapped = kps_target + np.array([2.0, 0.0], dtype=np.float32)
    errors = HyperSwapQualityAuditor.evaluate_geometric_alignment_error(kps_target, kps_swapped)
    assert np.isclose(errors["eye_error_px"], 2.0, atol=1e-3)
    assert np.isclose(errors["mouth_error_px"], 2.0, atol=1e-3)
    assert np.isclose(errors["mean_error_px"], 2.0, atol=1e-3)


# =====================================================================
# 4. Model Topology Integrity
# =====================================================================

def test_hyperswap_model_file_exists_and_topology():
    hyperswap_path = os.path.join(APP, "models", "hyperswap_1a_256.onnx")
    if not os.path.exists(hyperswap_path):
        pytest.skip("hyperswap_1a_256.onnx not present on disk")

    import onnx
    model = onnx.load(hyperswap_path)
    inputs = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}
    outputs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in model.graph.output}

    assert "target" in inputs
    assert inputs["target"] == [1, 3, 256, 256]
    assert "source" in inputs
    assert inputs["source"] == [1, 512]

    assert "output" in outputs
    assert outputs["output"] == [1, 3, 256, 256]
    assert "mask" in outputs
    assert outputs["mask"] == [1, 1, 256, 256]
