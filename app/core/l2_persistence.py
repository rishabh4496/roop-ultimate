"""CUDA L2 cache persistence (access-policy windows) for the post-processing working set.

Ada's L2 is 36 MiB on the RTX 4070 (``cudaDevAttrL2CacheSize``), of which up to 24.75 MiB can be set aside as PERSISTING
lines (``cudaDevAttrMaxPersistingL2CacheSize``); a stream's access-policy window says which address range should live there.
Both numbers are queried, never assumed: a 3060 laptop has a smaller L2, and whether a driver honours the carve-out at all
(WDDM vs TCC) is something to MEASURE (``tests/probe_l2_persistence.py``), not to read off the API's return code.

API notes against the brief: torch exposes only ``L2_cache_size`` (the persisting maximum is a ``cudaDeviceGetAttribute``);
the runtime call is ``cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)`` (the driver equivalent is ``cuCtxSetLimit``;
``cudaCtxSetLimit`` does not exist). ``cudaStreamSetAttribute`` does exist, with ``cudaStreamAttributeAccessPolicyWindow``.

A window is ONE contiguous ``(base_ptr, num_bytes)`` range, so two buffers can only share it if they are carved from one
allocation: :class:`PersistentArena` hands out views of a single block (the persistent source-embedding buffer and the active
``[B, 3, S, S]`` ROI working buffer), and the window covers the arena. Full-frame decode buffers get the OPPOSITE hint on
their own stream (``cudaAccessPropertyStreaming``) so a 4K frame passing through does not evict the arena.

Everything is a no-op that says so (``available`` False, a reason string) when ``cuda-python`` or a CUDA device is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    from cuda.bindings import runtime as _rt
    HAVE_CUDA_PYTHON = True
    CUDA_PYTHON_IMPORT_ERROR = ""
except Exception as _cuda_python_error:                  # pragma: no cover - the light profile
    _rt = None
    HAVE_CUDA_PYTHON = False
    CUDA_PYTHON_IMPORT_ERROR = f"{type(_cuda_python_error).__name__}: {_cuda_python_error}"[:300]

MiB = 1 << 20
ALIGN = 512


@dataclass(frozen=True)
class L2Info:
    l2_bytes: int
    max_persisting_bytes: int
    max_window_bytes: int
    device: int


def _ok(result, what: str):
    err = result[0] if isinstance(result, tuple) else result
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def available() -> Tuple[bool, str]:
    if not HAVE_CUDA_PYTHON:
        return False, "cuda-python is not importable (" + (CUDA_PYTHON_IMPORT_ERROR or "not installed") + ")"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    return True, ""


def query_l2(device: int = 0) -> L2Info:
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    A = _rt.cudaDeviceAttr
    get = lambda attr: int(_ok(_rt.cudaDeviceGetAttribute(attr, device), "cudaDeviceGetAttribute"))
    return L2Info(l2_bytes=get(A.cudaDevAttrL2CacheSize), max_persisting_bytes=get(A.cudaDevAttrMaxPersistingL2CacheSize),
                  max_window_bytes=get(A.cudaDevAttrMaxAccessPolicyWindowSize), device=device)


class PersistentArena:
    """One CUDA allocation carved into named views, so a single access-policy window can cover all of them."""

    def __init__(self, nbytes: int, device: int = 0):
        self.nbytes = int(-(-nbytes // ALIGN) * ALIGN)
        self.device = torch.device("cuda", device)
        self.block = torch.zeros(self.nbytes, dtype=torch.uint8, device=self.device)
        self._cursor = 0
        self.views: Dict[str, torch.Tensor] = {}

    def alloc(self, name: str, shape, dtype=torch.float16) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= int(s)
        size = n * torch.empty((), dtype=dtype).element_size()
        start = -(-self._cursor // ALIGN) * ALIGN
        if start + size > self.nbytes:
            raise MemoryError(f"arena of {self.nbytes} B cannot hold {name!r} ({size} B) after {start} B")
        view = self.block[start:start + size].view(dtype).view(*shape)
        self._cursor = start + size
        self.views[name] = view
        return view

    @property
    def base_ptr(self) -> int:
        return int(self.block.data_ptr())

    @property
    def used(self) -> int:
        return self._cursor


def _window_attr_id():
    """``cudaStreamAttributeAccessPolicyWindow`` is a C-header alias of the launch attribute; cuda-python 12.9 only
    exposes the launch-attribute spelling (same numeric value), so accept either."""
    ids = _rt.cudaStreamAttrID
    return getattr(ids, "cudaStreamAttributeAccessPolicyWindow", None) or ids.cudaLaunchAttributeAccessPolicyWindow


def _window(base_ptr: int, num_bytes: int, hit_ratio: float, hit_prop, miss_prop):
    attr = _rt.cudaStreamAttrValue()
    w = attr.accessPolicyWindow
    w.base_ptr, w.num_bytes, w.hitRatio, w.hitProp, w.missProp = int(base_ptr), int(num_bytes), float(hit_ratio), hit_prop, miss_prop
    return attr


def set_persisting_limit(nbytes: int, device: int = 0) -> int:
    """Ask the driver to set aside ``nbytes`` of L2 for persisting lines; returns what it actually granted."""
    info = query_l2(device)
    want = min(int(nbytes), info.max_persisting_bytes)
    _ok(_rt.cudaDeviceSetLimit(_rt.cudaLimit.cudaLimitPersistingL2CacheSize, want), "cudaDeviceSetLimit(PersistingL2CacheSize)")
    return int(_ok(_rt.cudaDeviceGetLimit(_rt.cudaLimit.cudaLimitPersistingL2CacheSize), "cudaDeviceGetLimit"))


def pin_arena(stream: torch.cuda.Stream, arena: PersistentArena, persisting_bytes: int = 16 * MiB, hit_ratio: float = 1.0) -> dict:
    """Make ``arena``'s used range persist in L2 for kernels launched on ``stream`` (``Stream_Infer``). Returns what was set.

    The window is clamped to the carve-out the driver granted: a window larger than the persisting set thrashes it, which is
    the failure this exists to prevent, so ``hit_ratio`` is scaled down to ``granted / window`` when it would not fit.
    """
    ok, why = available()
    if not ok:
        return dict(applied=False, reason=why)
    granted = set_persisting_limit(persisting_bytes, arena.device.index or 0)
    num_bytes = min(arena.used or arena.nbytes, query_l2(arena.device.index or 0).max_window_bytes)
    ratio = float(hit_ratio) if num_bytes <= granted else float(hit_ratio) * granted / num_bytes
    attr = _window(arena.base_ptr, num_bytes, ratio, _rt.cudaAccessProperty.cudaAccessPropertyPersisting,
                   _rt.cudaAccessProperty.cudaAccessPropertyStreaming)
    _ok(_rt.cudaStreamSetAttribute(stream.cuda_stream, _window_attr_id(), attr),
        "cudaStreamSetAttribute(AccessPolicyWindow)")
    return dict(applied=True, granted_persisting_bytes=granted, window_bytes=num_bytes, hit_ratio=ratio)


def mark_streaming(stream: torch.cuda.Stream, *buffers: torch.Tensor) -> dict:
    """Tell ``stream`` that ``buffers`` (full-frame decode buffers) are use-once: they must not displace persisting lines.

    A stream carries one window, so this covers the single contiguous range spanning the given tensors; hand it the decode
    ring as one allocation.
    """
    ok, why = available()
    if not ok:
        return dict(applied=False, reason=why)
    lo = min(int(t.data_ptr()) for t in buffers)
    hi = max(int(t.data_ptr()) + t.numel() * t.element_size() for t in buffers)
    num = min(hi - lo, query_l2().max_window_bytes)
    attr = _window(lo, num, 1.0, _rt.cudaAccessProperty.cudaAccessPropertyStreaming, _rt.cudaAccessProperty.cudaAccessPropertyStreaming)
    _ok(_rt.cudaStreamSetAttribute(stream.cuda_stream, _window_attr_id(), attr),
        "cudaStreamSetAttribute(AccessPolicyWindow, streaming)")
    return dict(applied=True, window_bytes=num)


def clear(stream: Optional[torch.cuda.Stream] = None) -> None:
    """Drop the window on ``stream``, reset every persisting line, and give the carve-out back."""
    ok, _ = available()
    if not ok:
        return
    if stream is not None:
        attr = _window(0, 0, 0.0, _rt.cudaAccessProperty.cudaAccessPropertyNormal, _rt.cudaAccessProperty.cudaAccessPropertyNormal)
        _rt.cudaStreamSetAttribute(stream.cuda_stream, _window_attr_id(), attr)
    _rt.cudaCtxResetPersistingL2Cache()
    _rt.cudaDeviceSetLimit(_rt.cudaLimit.cudaLimitPersistingL2CacheSize, 0)
