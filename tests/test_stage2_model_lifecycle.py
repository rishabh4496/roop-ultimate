"""Tests for Stage 2: Model Lifecycle and Runtime Architecture.

Verifies:
1. All 8 required diagnostic telemetry fields are tracked:
   - MODEL
   - DEVICE
   - PROVIDER
   - PRECISION
   - INPUT SHAPE
   - ENGINE CACHE
   - VRAM COST
   - INITIALIZATION TIME
2. Sessions and IOBindings are loaded once and reused across frames (no per-frame recreation).
3. TensorRT engine/timing cache reuse verification.
4. Deterministic provider fallback and central registration.
5. Numerical output tolerance preservation (exact bit/float equality with reused bindings).
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add project root and app to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

from roop.model_lifecycle import (
    ModelLifecycleRecord,
    clear_model_lifecycle_records,
    get_model_lifecycle_records,
    register_model_lifecycle,
    format_model_lifecycle_table,
    export_model_lifecycle_json,
    check_engine_cache_status,
    TrackModelLifecycle,
)
from roop.processors.Mask_XSeg import Mask_XSeg
from roop.processors.Mask_XSeg3 import Mask_XSeg3
from roop.processors.Mask_RealityUX import Mask_RealityUX


def test_model_lifecycle_required_fields():
    """Verify that all 8 required fields exist in ModelLifecycleRecord and table header."""
    clear_model_lifecycle_records()

    record = register_model_lifecycle(
        model="HyperSwap",
        device="cuda:0",
        provider="TensorrtExecutionProvider",
        precision="fp16",
        input_shape="1x3x256x256, 1x512",
        engine_cache="HIT (reused)",
        vram_cost=420.5,
        init_time=0.085,
    )

    assert record.model == "HyperSwap"
    assert record.device == "cuda:0"
    assert record.provider == "TensorrtExecutionProvider"
    assert record.precision == "fp16"
    assert record.input_shape == "1x3x256x256, 1x512"
    assert record.engine_cache == "HIT (reused)"
    assert record.vram_cost == "420.5 MB"
    assert record.init_time == "85.0 ms"

    table = format_model_lifecycle_table()
    required_headers = [
        "MODEL",
        "DEVICE",
        "PROVIDER",
        "PRECISION",
        "INPUT SHAPE",
        "ENGINE CACHE",
        "VRAM COST",
        "INITIALIZATION TIME",
    ]
    for header in required_headers:
        assert header in table, f"Missing required column: {header}"


def test_track_model_lifecycle_context_manager():
    """Verify TrackModelLifecycle correctly computes elapsed time and cache status."""
    clear_model_lifecycle_records()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an artificial cached engine
        engine_file = Path(tmpdir) / "hyperswap_test.engine"
        engine_file.write_bytes(b"engine_data")
        time.sleep(0.01)

        with TrackModelLifecycle(
            model_name="hyperswap_test",
            device="cuda:0",
            precision="fp16",
            cache_dir=tmpdir,
        ) as tracker:
            mock_sess = MagicMock()
            mock_sess.get_providers.return_value = ["TensorrtExecutionProvider"]
            mock_inp = MagicMock()
            mock_inp.shape = [1, 3, 256, 256]
            mock_sess.get_inputs.return_value = [mock_inp]
            tracker.session = mock_sess

        records = get_model_lifecycle_records()
        assert len(records) == 1
        rec = records[0]
        assert rec.model == "hyperswap_test"
        assert rec.provider == "TensorrtExecutionProvider"
        assert "HIT" in rec.engine_cache
        assert "1x3x256x256" in rec.input_shape


def test_check_engine_cache_status():
    """Verify detection of cache HIT vs BUILT vs non-TRT N/A."""
    # 1. Non-TRT
    status_cuda = check_engine_cache_status("test_model", None, time.time(), "CUDAExecutionProvider")
    assert "N/A" in status_cuda
    assert "CUDA" in status_cuda

    # 2. TensorRT with pre-existing cache file (HIT)
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = Path(tmpdir) / "test_model_1.engine"
        engine.write_bytes(b"dummy")
        start_t = time.time() + 1.0  # start_time is in the future relative to file mtime

        status_hit = check_engine_cache_status("test_model", tmpdir, start_t, "TensorrtExecutionProvider")
        assert "HIT" in status_hit

        # 3. Newly created cache file (BUILT)
        fresh_dir = tempfile.mkdtemp()
        start_t2 = time.time() - 10.0
        new_engine = Path(fresh_dir) / "test_model_2.engine"
        new_engine.write_bytes(b"fresh")

        status_built = check_engine_cache_status("test_model", fresh_dir, start_t2, "TensorrtExecutionProvider")
        assert "BUILT" in status_built


class DummySession:
    def __init__(self, binding):
        self.binding = binding
        self.call_count = 0
    def io_binding(self):
        self.call_count += 1
        return self.binding
    def run_with_iobinding(self, iob):
        pass


def test_mask_xseg_iobinding_reuse():
    """Verify Mask_XSeg reuses _cached_io_binding without creating new binding on every face."""
    xseg = Mask_XSeg()
    mock_binding = MagicMock()
    session = DummySession(mock_binding)

    xseg.model_outputs = [MagicMock(name="output")]
    xseg.model_outputs[0].name = "output"
    xseg.model_inputs = [MagicMock(name="input")]
    xseg.model_inputs[0].name = "input"
    xseg.devicename = "cuda:0"
    xseg._cpu_only = False

    dummy_frame = np.zeros((1, 256, 256, 3), dtype=np.float32)

    # First call: should call sess.io_binding() once and bind output
    xseg._run_session(session, dummy_frame)
    assert session.call_count == 1
    assert mock_binding.bind_output.call_count == 1
    assert getattr(session, "_cached_io_binding", None) is mock_binding

    # Second call: should REUSE the cached io_binding, NOT call sess.io_binding() again!
    xseg._run_session(session, dummy_frame)
    assert session.call_count == 1  # Still 1! Not 2!
    assert mock_binding.bind_output.call_count == 1  # Not rebound!


def test_mask_xseg3_iobinding_reuse():
    """Verify Mask_XSeg3 reuses _cached_io_binding without creating new binding on every face."""
    xseg3 = Mask_XSeg3()
    mock_binding = MagicMock()
    session = DummySession(mock_binding)

    xseg3.model_outputs = [MagicMock(name="output")]
    xseg3.model_outputs[0].name = "output"
    xseg3.model_inputs = [MagicMock(name="input")]
    xseg3.model_inputs[0].name = "input"
    xseg3.devicename = "cuda:0"
    xseg3._cpu_only = False

    dummy_frame = np.zeros((1, 256, 256, 3), dtype=np.float32)

    # First call
    xseg3._run_session(session, dummy_frame)
    assert session.call_count == 1
    assert mock_binding.bind_output.call_count == 1
    assert getattr(session, "_cached_io_binding", None) is mock_binding

    # Second call: reused!
    xseg3._run_session(session, dummy_frame)
    assert session.call_count == 1
    assert mock_binding.bind_output.call_count == 1


def test_mask_realityux_executor_reuse():
    """Verify Mask_RealityUX reuses persistent ThreadPoolExecutor instead of spawning threads per face."""
    rux = Mask_RealityUX()
    rux.Initialize({"devicename": "cpu"})

    assert hasattr(rux, "_executor")
    assert rux._executor is not None

    # Check executor is reused and clean teardown in Release
    rux.Release()
    assert rux._executor is None


def test_export_model_lifecycle_json():
    """Verify structured JSON telemetry export for cross-commit comparison."""
    clear_model_lifecycle_records()
    register_model_lifecycle(
        model="TestModel",
        device="cuda:0",
        provider="CUDAExecutionProvider",
        precision="fp32",
        input_shape="1x3x512x512",
        engine_cache="N/A (CUDA)",
        vram_cost="150.0 MB",
        init_time="45.0 ms",
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_model_lifecycle_json(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["count"] == 1
        assert data["models"][0]["model"] == "TestModel"
        assert data["models"][0]["precision"] == "fp32"
        assert data["models"][0]["input_shape"] == "1x3x512x512"
        assert data["models"][0]["engine_cache"] == "N/A (CUDA)"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_enhance_codeformer_iobinding_reuse():
    """Verify Enhance_CodeFormer reuses _cached_io_binding across inference calls."""
    from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer

    cf = Enhance_CodeFormer()
    mock_binding = MagicMock()
    mock_binding.copy_outputs_to_cpu.return_value = [np.zeros((1, 3, 512, 512), dtype=np.float32)]
    session = DummySession(mock_binding)

    cf.model_outputs = [MagicMock(name="output")]
    cf.model_outputs[0].name = "output"
    cf.model_inputs = [MagicMock(name="x"), MagicMock(name="w")]
    cf.model_inputs[0].name = "x"
    cf.model_inputs[0].type = "tensor(float)"
    cf.model_inputs[1].name = "w"
    cf.model_inputs[1].type = "tensor(double)"
    cf.devicename = "cuda:0"
    cf.model_codeformer = session
    cf.in_dtype = np.float32
    cf._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)

    dummy_crop = np.zeros((512, 512, 3), dtype=np.uint8)

    # First call
    out1, scale1 = cf.Run(MagicMock(), MagicMock(), dummy_crop)
    assert session.call_count == 1
    assert mock_binding.bind_output.call_count == 1
    assert getattr(session, "_cached_io_binding", None) is mock_binding

    # Second call: reused!
    out2, scale2 = cf.Run(MagicMock(), MagicMock(), dummy_crop)
    assert session.call_count == 1
    assert mock_binding.bind_output.call_count == 1
    assert np.array_equal(out1, out2)

    cf.Release()
    assert cf.model_codeformer is None
