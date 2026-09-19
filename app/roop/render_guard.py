"""Resource checks that keep a full render from looking like a deadlock.

TensorRT engine construction and CUDA allocations can wait for a long time when
another process has consumed nearly all of the card.  Preview requests should
remain best-effort, but a full video render must fail early with an actionable
message instead of creating a partial output and leaving workers stalled.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _minimum_free_vram_gb(total_gb: float) -> float:
    """Return a conservative pre-render headroom floor.

    This is deliberately based on *free* memory at render admission, not total
    VRAM.  It never disqualifies TensorRT by card size.  A 6GB card gets the
    same 1GB floor as a larger card, while a 12GB card gets about 1.4GB.
    ``ROOP_RENDER_MIN_FREE_VRAM_GB`` is an escape hatch for a measured setup.
    Set it to ``0`` to disable the guard explicitly.
    """
    configured = os.environ.get("ROOP_RENDER_MIN_FREE_VRAM_GB")
    if configured is not None and str(configured).strip() != "":
        try:
            return max(0.0, float(configured))
        except (TypeError, ValueError):
            pass
    return max(1.0, min(2.5, float(total_gb) * 0.12))


def check_render_gpu_headroom(
    provider: str,
    device_id: int = 0,
    *,
    torch_module: Optional[Any] = None,
) -> Optional[Dict[str, float]]:
    """Validate free GPU memory before a full CUDA/TensorRT render.

    Returns a small telemetry dictionary when a CUDA probe succeeds, ``None``
    for CPU/non-CUDA providers or when the optional probe is unavailable.  A
    low-memory condition raises ``RuntimeError`` before model/session creation.
    """
    normalized = str(provider or "").strip().lower().replace("executionprovider", "")
    if normalized not in {"cuda", "tensorrt"}:
        return None

    try:
        torch = torch_module or __import__("torch")
        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info(int(device_id))
        free_gb = float(free_bytes) / (1024 ** 3)
        total_gb = float(total_bytes) / (1024 ** 3)
    except Exception:
        # A partial CUDA install must continue through the existing provider
        # diagnostics.  This guard is an additional safety net, not admission.
        return None

    required_gb = _minimum_free_vram_gb(total_gb)
    if required_gb > 0.0 and free_gb < required_gb:
        raise RuntimeError(
            "Full render paused before model initialization: "
            f"{free_gb:.2f} GB of {total_gb:.2f} GB VRAM is free, but "
            f"{required_gb:.2f} GB is required for the {normalized.upper()} "
            "runtime. Close other GPU workloads or wait for the card to "
            "clear, then retry. TensorRT remains qualified for this GPU."
        )

    return {
        "free_vram_gb": round(free_gb, 3),
        "total_vram_gb": round(total_gb, 3),
        "required_free_vram_gb": round(required_gb, 3),
    }
