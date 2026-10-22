"""Hand PyTorch's cached-but-idle VRAM back around a TensorRT engine build or canary run.

A TensorRT build asks the driver for scratch (builder workspace, the HybridGlobWriter's
tactic buffers, virtual-memory buffers) at a moment when this process's torch caching
allocator may be holding gigabytes it is not using.  When the card is full the build fails
with `Error Code 2: OutOfMemory (Requested size was 1024 bytes)` -- a tiny request refused
because everything else is already spoken for, not because the request was big.
`torch.cuda.empty_cache()` returns the idle cached blocks; it never touches a live tensor.

It is called BEFORE (so the build has the room) and AFTER (so scratch the build left behind
in torch's pool does not stay resident beside the engine it produced).  It synchronises the
device, which is why it belongs here, around events that take seconds to minutes, and not on
the per-frame path (`runtime_scheduler` rate-limits its own calls for that reason).

Torch is never imported from here: if it is not already loaded there is no torch cache to
release, and importing it just to ask would load the CUDA runtime in a CPU-only process.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

from roop.degrade import swallowed as _swallowed


def release_torch_cache(tag: str) -> bool:
    """`torch.cuda.empty_cache()` if torch is loaded and CUDA is initialised.  True when it ran."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        cuda = torch.cuda
        if not cuda.is_available() or not cuda.is_initialized():
            return False
        cuda.empty_cache()
        return True
    except Exception as exc:
        _swallowed("roop/trt_build_guard.py:release_torch_cache", exc,
                   "torch cache not released around %s; the build proceeds" % tag)
        return False


@contextmanager
def around_trt_build(tag: str) -> Iterator[None]:
    """Release torch's idle VRAM immediately before and immediately after the body.

    The "after" runs on failure too: a build that died halfway is exactly when scratch is stranded.
    """
    release_torch_cache(tag + " (before)")
    try:
        yield
    finally:
        release_torch_cache(tag + " (after)")
