"""Hybrid CPU topology detection and thread affinity pinning across Intel P-Cores and E-Cores.

Enforces strict OS thread affinity:
- Performance Cores (P-Cores, high priority): CUDA/TensorRT inference and PyNvVideoCodec decode/encode.
- Efficiency Cores (E-Cores, normal/low priority): FastAPI loopback server, React UI telemetry,
  checkpoint serialization, and FFmpeg audio muxing.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional

from roop.degrade import swallowed as _swallowed

# Windows Win32 constants
_RELATION_PROCESSOR_CORE = 0
_THREAD_PRIORITY_HIGHEST = 2
_THREAD_PRIORITY_ABOVE_NORMAL = 1
_THREAD_PRIORITY_NORMAL = 0
_THREAD_PRIORITY_BELOW_NORMAL = -1
_THREAD_PRIORITY_LOWEST = -2
_THREAD_PRIORITY_IDLE = -15


@dataclass(frozen=True)
class CpuTopology:
    """CPU topology metrics for heterogeneous (hybrid) architectures."""
    is_hybrid: bool
    total_logical_cpus: int
    p_core_count: int
    e_core_count: int
    p_core_mask: int
    e_core_mask: int
    p_core_cpus: tuple[int, ...]
    e_core_cpus: tuple[int, ...]

    def summary(self) -> str:
        if not self.is_hybrid:
            return (f"Homogeneous CPU: {self.total_logical_cpus} logical threads "
                    f"(mask: 0x{self.p_core_mask:08x})")
        return (f"Hybrid CPU detected: {self.p_core_count} P-Cores "
                f"({len(self.p_core_cpus)} threads, mask: 0x{self.p_core_mask:08x}), "
                f"{self.e_core_count} E-Cores ({len(self.e_core_cpus)} threads, mask: 0x{self.e_core_mask:08x})")


_TOPOLOGY_CACHE: Optional[CpuTopology] = None


def _mask_to_cpus(mask: int) -> tuple[int, ...]:
    cpus: List[int] = []
    bit = 0
    while mask > 0:
        if mask & 1:
            cpus.append(bit)
        mask >>= 1
        bit += 1
    return tuple(cpus)


def _detect_windows_topology() -> Optional[CpuTopology]:
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_info_ex = getattr(kernel32, "GetLogicalProcessorInformationEx", None)
        if get_info_ex is None:
            return None

        # Determine buffer size needed
        buf_len = ctypes.c_uint32(0)
        get_info_ex(_RELATION_PROCESSOR_CORE, None, ctypes.byref(buf_len))
        if buf_len.value == 0:
            return None

        buf = (ctypes.c_byte * buf_len.value)()
        if not get_info_ex(_RELATION_PROCESSOR_CORE, buf, ctypes.byref(buf_len)):
            return None

        offset = 0
        cores: List[tuple[int, int]] = []  # (efficiency_class, affinity_mask)

        while offset < buf_len.value:
            rel_type = ctypes.c_uint32.from_buffer_copy(buf, offset).value
            size = ctypes.c_uint32.from_buffer_copy(buf, offset + 4).value
            if size == 0:
                break
            if rel_type == _RELATION_PROCESSOR_CORE:
                # In PROCESSOR_RELATIONSHIP:
                # Flags at offset + 8 (1 byte), EfficiencyClass at offset + 9 (1 byte)
                # GroupMask count at offset + 30 (2 bytes)
                # GroupMask[0].Mask is ULONG_PTR at offset + 32
                eff_class = ctypes.c_uint8.from_buffer_copy(buf, offset + 9).value
                mask = ctypes.c_size_t.from_buffer_copy(buf, offset + 32).value
                cores.append((eff_class, mask))
            offset += size

        if not cores:
            return None

        eff_classes = {c[0] for c in cores}
        # If there are multiple efficiency classes, highest is P-core, lowest is E-core
        if len(eff_classes) > 1:
            max_eff = max(eff_classes)
            p_mask = 0
            e_mask = 0
            p_count = 0
            e_count = 0
            for eff, mask in cores:
                if eff == max_eff:
                    p_mask |= mask
                    p_count += 1
                else:
                    e_mask |= mask
                    e_count += 1

            p_cpus = _mask_to_cpus(p_mask)
            e_cpus = _mask_to_cpus(e_mask)
            total = len(p_cpus) + len(e_cpus)
            return CpuTopology(
                is_hybrid=True,
                total_logical_cpus=total,
                p_core_count=p_count,
                e_core_count=e_count,
                p_core_mask=p_mask,
                e_core_mask=e_mask,
                p_core_cpus=p_cpus,
                e_core_cpus=e_cpus,
            )

        # Non-hybrid Windows machine
        full_mask = 0
        for _, mask in cores:
            full_mask |= mask
        all_cpus = _mask_to_cpus(full_mask)
        return CpuTopology(
            is_hybrid=False,
            total_logical_cpus=len(all_cpus),
            p_core_count=len(cores),
            e_core_count=0,
            p_core_mask=full_mask,
            e_core_mask=full_mask,
            p_core_cpus=all_cpus,
            e_core_cpus=all_cpus,
        )
    except Exception as exc:
        _swallowed("cpu_topology:detect_windows", exc, "falling back to generic CPU topology")
        return None


def _detect_linux_topology() -> Optional[CpuTopology]:
    if not sys.platform.startswith("linux"):
        return None
    try:
        capacities: dict[int, int] = {}
        cpu_dir = "/sys/devices/system/cpu"
        if not os.path.exists(cpu_dir):
            return None

        for item in os.listdir(cpu_dir):
            if item.startswith("cpu") and item[3:].isdigit():
                cpu_id = int(item[3:])
                cap_file = os.path.join(cpu_dir, item, "cpu_capacity")
                if os.path.exists(cap_file):
                    with open(cap_file, "r") as f:
                        capacities[cpu_id] = int(f.read().strip())

        if not capacities:
            return None

        distinct_caps = set(capacities.values())
        if len(distinct_caps) > 1:
            max_cap = max(distinct_caps)
            p_cpus = sorted([cid for cid, cap in capacities.items() if cap == max_cap])
            e_cpus = sorted([cid for cid, cap in capacities.items() if cap < max_cap])
            p_mask = sum(1 << cid for cid in p_cpus)
            e_mask = sum(1 << cid for cid in e_cpus)
            return CpuTopology(
                is_hybrid=True,
                total_logical_cpus=len(capacities),
                p_core_count=len(p_cpus),
                e_core_count=len(e_cpus),
                p_core_mask=p_mask,
                e_core_mask=e_mask,
                p_core_cpus=tuple(p_cpus),
                e_core_cpus=tuple(e_cpus),
            )
    except Exception as exc:
        _swallowed("cpu_topology:detect_linux", exc, "falling back to generic CPU topology")
    return None


def detect_topology() -> CpuTopology:
    """Detect CPU topology and whether a hybrid (P-core / E-core) layout is present."""
    global _TOPOLOGY_CACHE
    if _TOPOLOGY_CACHE is not None:
        return _TOPOLOGY_CACHE

    topo = _detect_windows_topology()
    if topo is None:
        topo = _detect_linux_topology()

    if topo is None:
        # Generic homogeneous fallback
        count = os.cpu_count() or 4
        all_mask = (1 << count) - 1
        all_cpus = tuple(range(count))
        topo = CpuTopology(
            is_hybrid=False,
            total_logical_cpus=count,
            p_core_count=count,
            e_core_count=0,
            p_core_mask=all_mask,
            e_core_mask=all_mask,
            p_core_cpus=all_cpus,
            e_core_cpus=all_cpus,
        )

    _TOPOLOGY_CACHE = topo
    return topo


def get_topology() -> CpuTopology:
    """Get the cached CpuTopology singleton."""
    return detect_topology()


def set_current_thread_affinity(mask: int) -> Optional[int]:
    """Set the affinity mask of the calling thread. Returns previous mask if successful."""
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
            h_thread = kernel32.GetCurrentThread()
            old = kernel32.SetThreadAffinityMask(h_thread, ctypes.c_size_t(mask))
            return int(old) if old else None
        except Exception as exc:
            _swallowed("cpu_topology:set_affinity_win", exc, "thread affinity unchanged")
            return None
    elif hasattr(os, "sched_setaffinity"):
        try:
            cpus = _mask_to_cpus(mask)
            old_cpus = os.sched_getaffinity(0)
            os.sched_setaffinity(0, cpus)
            return sum(1 << c for c in old_cpus)
        except Exception as exc:
            _swallowed("cpu_topology:set_affinity_posix", exc, "thread affinity unchanged")
            return None
    return None


def set_current_thread_priority(priority: int) -> bool:
    """Set OS thread priority of calling thread."""
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            kernel32.SetThreadPriority.restype = ctypes.c_bool
            h_thread = kernel32.GetCurrentThread()
            return bool(kernel32.SetThreadPriority(h_thread, ctypes.c_int(priority)))
        except Exception as exc:
            _swallowed("cpu_topology:set_priority_win", exc, "thread priority unchanged")
            return False
    return False


def pin_current_thread_to_p_cores(high_priority: bool = True) -> bool:
    """Pin calling thread to Performance Cores (P-Cores).

    Optionally raises thread scheduling priority to high so CUDA/TensorRT kernel
    dispatch and hardware video decode/encode are never preempted.
    """
    topo = get_topology()
    old = set_current_thread_affinity(topo.p_core_mask)
    if high_priority:
        set_current_thread_priority(_THREAD_PRIORITY_HIGHEST)
    return old is not None


def pin_current_thread_to_e_cores(low_priority: bool = False) -> bool:
    """Pin calling thread to Efficiency Cores (E-Cores).

    Used for background I/O: FastAPI loopback server, React telemetry,
    checkpoint serialization, and FFmpeg audio muxing.
    """
    topo = get_topology()
    if not topo.is_hybrid:
        return True
    old = set_current_thread_affinity(topo.e_core_mask)
    if low_priority:
        set_current_thread_priority(_THREAD_PRIORITY_BELOW_NORMAL)
    else:
        set_current_thread_priority(_THREAD_PRIORITY_NORMAL)
    return old is not None


@contextlib.contextmanager
def p_core_scope(high_priority: bool = True) -> Iterator[None]:
    """Scoped execution strictly on P-Cores, restoring prior affinity on exit."""
    topo = get_topology()
    prev_mask = set_current_thread_affinity(topo.p_core_mask)
    if high_priority:
        set_current_thread_priority(_THREAD_PRIORITY_HIGHEST)
    try:
        yield
    finally:
        if prev_mask:
            set_current_thread_affinity(prev_mask)
            set_current_thread_priority(_THREAD_PRIORITY_NORMAL)


@contextlib.contextmanager
def e_core_scope(low_priority: bool = False) -> Iterator[None]:
    """Scoped execution strictly on E-Cores, restoring prior affinity on exit."""
    topo = get_topology()
    prev_mask = None
    if topo.is_hybrid:
        prev_mask = set_current_thread_affinity(topo.e_core_mask)
        if low_priority:
            set_current_thread_priority(_THREAD_PRIORITY_BELOW_NORMAL)
    try:
        yield
    finally:
        if prev_mask:
            set_current_thread_affinity(prev_mask)
            set_current_thread_priority(_THREAD_PRIORITY_NORMAL)
