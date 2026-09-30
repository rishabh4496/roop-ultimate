"""Model Lifecycle and Runtime Architecture Diagnostics.

Tracks, validates, and reports runtime telemetry for all model and session
initialization paths across ONNX Runtime, TensorRT, and CUDA/PyTorch:
- Model deduplication and reuse verification
- Session creation vs reuse tracking (preventing per-frame recreations)
- TensorRT engine and timing cache reuse verification
- Centralized provider resolution and deterministic fallback logging
- Explicit device placement and precision policy verification
- Per-model VRAM cost and initialization latency measurements
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class ModelLifecycleRecord:
    """Runtime diagnostics record for a single model / session initialization."""

    model: str
    device: str
    provider: str
    precision: str
    input_shape: str
    engine_cache: str
    vram_cost: str
    init_time: str
    reused: bool = False
    session_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_REGISTRY_LOCK = threading.RLock()
_MODEL_REGISTRY: List[ModelLifecycleRecord] = []
_ACTIVE_SESSIONS: Dict[str, int] = {}  # model_key -> session id


def clear_model_lifecycle_records() -> None:
    """Clear recorded model lifecycle events."""
    with _REGISTRY_LOCK:
        _MODEL_REGISTRY.clear()
        _ACTIVE_SESSIONS.clear()


def get_model_lifecycle_records() -> List[ModelLifecycleRecord]:
    """Retrieve all recorded model lifecycle entries."""
    with _REGISTRY_LOCK:
        return list(_MODEL_REGISTRY)


def register_model_lifecycle(
    model: str,
    device: str,
    provider: str,
    precision: str,
    input_shape: str,
    engine_cache: str,
    vram_cost: Union[str, float],
    init_time: Union[str, float],
    reused: bool = False,
    session_id: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ModelLifecycleRecord:
    """Register a model lifecycle record into the central diagnostic repository."""
    if hasattr(precision, 'effective'):
        precision_str = str(precision.effective)
    elif isinstance(precision, str):
        precision_str = precision
    else:
        precision_str = str(precision)

    if isinstance(vram_cost, (int, float)):
        vram_cost_str = f"{vram_cost:.1f} MB" if vram_cost >= 0.1 else "0.0 MB"
    else:
        vram_cost_str = str(vram_cost)

    if isinstance(init_time, (int, float)):
        init_time_str = f"{init_time * 1000.0:.1f} ms" if init_time < 1.0 else f"{init_time:.2f} s"
    else:
        init_time_str = str(init_time)

    record = ModelLifecycleRecord(
        model=str(model),
        device=str(device),
        provider=str(provider),
        precision=precision_str,
        input_shape=str(input_shape),
        engine_cache=str(engine_cache),
        vram_cost=vram_cost_str,
        init_time=init_time_str,
        reused=bool(reused),
        session_id=session_id,
        extra=extra or {},
    )

    with _REGISTRY_LOCK:
        _MODEL_REGISTRY.append(record)
        if session_id is not None:
            _ACTIVE_SESSIONS[model] = session_id

    return record


def get_cuda_vram_mb(device_id: int = 0) -> float:
    """Return currently allocated CUDA memory in MB, or 0.0 if not on CUDA."""
    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            return torch.cuda.memory_allocated(device_id) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def check_engine_cache_status(
    model_name: str,
    cache_dir: Optional[Union[str, Path]],
    start_time: float,
    active_provider: str,
) -> str:
    """Determine whether TensorRT engine cache was HIT, BUILT, or N/A."""
    is_trt = "tensorrt" in str(active_provider).lower()
    if not is_trt:
        clean_ep = str(active_provider).replace("ExecutionProvider", "").strip()
        return f"N/A ({clean_ep or 'CPU'})"

    if not cache_dir:
        return "ENABLED (default cache)"

    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return "ENABLED (directory pending)"

    # Look for serialized engine files or timing caches (.engine, .bin, .timing)
    clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.lower())
    found_engine = False
    hit = False

    try:
        for f in cache_path.iterdir():
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(ext in name_lower for ext in (".engine", ".bin", ".timing")):
                # Check stem or cache match
                if any(part in name_lower for part in clean_stem.split("_") if len(part) > 2):
                    found_engine = True
                    mtime = f.stat().st_mtime
                    if mtime < start_time:
                        hit = True
                        break
    except Exception:
        pass

    if hit:
        return "HIT (reused)"
    elif found_engine:
        return "BUILT (cached)"
    return "ENABLED (timing/engine cache active)"


def format_shape_from_session(session: Any) -> str:
    """Format input shapes from an ONNX Runtime InferenceSession or metadata."""
    if session is None or not hasattr(session, "get_inputs"):
        return "unknown"
    try:
        inputs = session.get_inputs()
        parts = []
        for inp in inputs:
            shape = getattr(inp, "shape", None)
            if shape:
                dims = [str(d) for d in shape]
                parts.append("x".join(dims))
            else:
                parts.append("dynamic")
        return ", ".join(parts) if parts else "none"
    except Exception:
        return "dynamic"


class TrackModelLifecycle:
    """Context manager to measure and register a model's initialization lifecycle."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        precision: str = "fp16",
        cache_dir: Optional[Union[str, Path]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.cache_dir = cache_dir
        self.extra = extra or {}
        self.device_id = 0
        match = re.search(r"cuda:?(\d+)", device.lower())
        if match:
            self.device_id = int(match.group(1))

        self.t0 = 0.0
        self.t0_wall = 0.0
        self.vram0 = 0.0
        self.session: Any = None
        self.provider: Optional[str] = None
        self.input_shape: Optional[str] = None
        self.record: Optional[ModelLifecycleRecord] = None

    def __enter__(self) -> "TrackModelLifecycle":
        self.vram0 = get_cuda_vram_mb(self.device_id)
        self.t0 = time.perf_counter()
        self.t0_wall = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - self.t0
        vram1 = get_cuda_vram_mb(self.device_id)
        vram_cost = max(0.0, vram1 - self.vram0)

        # Detect active provider if not explicitly given
        provider = self.provider
        if not provider and self.session is not None and hasattr(self.session, "get_providers"):
            try:
                providers = self.session.get_providers()
                if providers:
                    provider = providers[0]
            except Exception:
                pass
        if not provider:
            provider = "CUDAExecutionProvider" if "cuda" in self.device.lower() else "CPUExecutionProvider"

        # Detect input shape if not explicitly given
        input_shape = self.input_shape
        if not input_shape and self.session is not None:
            input_shape = format_shape_from_session(self.session)
        if not input_shape:
            input_shape = "dynamic"

        # Check engine cache status
        engine_cache = check_engine_cache_status(
            self.model_name, self.cache_dir, self.t0_wall, provider
        )

        sess_id = id(self.session) if self.session is not None else None

        self.record = register_model_lifecycle(
            model=self.model_name,
            device=self.device,
            provider=provider,
            precision=self.precision,
            input_shape=input_shape,
            engine_cache=engine_cache,
            vram_cost=vram_cost,
            init_time=elapsed,
            reused=False,
            session_id=sess_id,
            extra=self.extra,
        )


def format_model_lifecycle_table(records: Optional[Sequence[ModelLifecycleRecord]] = None) -> str:
    """Format recorded model telemetry into a structured ASCII table."""
    recs = list(records if records is not None else get_model_lifecycle_records())
    if not recs:
        return "No models initialized or registered."

    headers = [
        "MODEL",
        "DEVICE",
        "PROVIDER",
        "PRECISION",
        "INPUT SHAPE",
        "ENGINE CACHE",
        "VRAM COST",
        "INITIALIZATION TIME",
    ]

    rows = []
    for r in recs:
        rows.append([
            r.model,
            r.device,
            r.provider,
            r.precision,
            r.input_shape,
            r.engine_cache,
            r.vram_cost,
            r.init_time,
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    # Construct borders
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [sep, hdr, sep]
    for row in rows:
        line = "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        lines.append(line)
    lines.append(sep)

    return "\n".join(lines)


def print_model_lifecycle_table(records: Optional[Sequence[ModelLifecycleRecord]] = None) -> None:
    """Print the model lifecycle table directly to stdout."""
    print(format_model_lifecycle_table(records), flush=True)


def export_model_lifecycle_json(file_path: Union[str, Path]) -> None:
    """Export all recorded model lifecycles to structured JSON."""
    records = get_model_lifecycle_records()
    payload = {
        "count": len(records),
        "timestamp": time.time(),
        "models": [r.to_dict() for r in records],
    }
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
