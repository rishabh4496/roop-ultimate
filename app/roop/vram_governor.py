"""Per-render VRAM governor: budget the job, then step it down to fit.

WHAT IT DOES. Before a full render builds any session, `admit()` reads free
VRAM (NVML first: it sees every process on the card; torch as the fallback),
estimates what THIS job will allocate -- swapper, detector, masks and enhancer
at their input sizes and context counts, the cross-frame swap batch, and the
NVDEC decode surfaces at the video's resolution -- and compares the
two against the user's safety margin (`vram_safety_margin_gb`, default 1.5).
When the projected headroom is below the margin it steps the job down, cheapest
change first:

    1. the cross-frame swap batch cap, 8 -> 4 -> 2 -> 1   (throughput only)
    2. GPEN 2048 -> 1024 -> 512                           (changes the look)

Each step is printed as a `[VramGovernor]` line; the plan's `batch_cap` is read
by `ProcessMgr._make_swap_batcher` and its `gpen_size` by
`core.get_processing_plugins`. Nothing else consumes it. Preview never goes
through admission, so it is never governed.

WHAT IT DOES NOT DO. It does not refuse a render -- `render_guard` does that,
against a floor of its own -- and it does not resize TensorRT pools, which
`session_pool.TensorRTResourceManager` already admits against live free VRAM.

NOT TO BE CONFUSED WITH `optimized_processor.VramGovernor`, which governs the
vectorized pipeline -- reached only from benchmark_comparison.py and
verify_trt_fps.py, never from a render.

WHY IT LEARNS. The per-model numbers come from `session_pool._resource_spec`,
which that module itself calls "intentionally modest" scheduling budgets, not
measurements. So a sampler thread records the real device-memory peak over the
render, and `finish()` stores peak/estimate per settings signature in
`vram_calibration.json`. The next render with that signature scales its
estimate by the learned ratio. Every render prints both numbers, so a
wrong prior is visible on the first run rather than trusted silently.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from roop.degrade import swallowed as _swallowed

DEFAULT_MARGIN_GB = 1.5
MARGIN_RANGE_GB = (0.5, 4.0)
BATCH_STEPS = (8, 4, 2, 1)
GPEN_STEPS = (2048, 1024, 512)
# A learned ratio outside this band means the sample was polluted (another
# process allocated mid-render, or the render died early); clamp rather than
# let one bad render steer every later one.
_RATIO_BAND = (0.5, 3.0)
_EMA = 0.5
_CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'vram_calibration.json')
# CUDA context + cuDNN/cuBLAS handles + ORT arena slack for the process, which
# no per-model spec carries. Charged once.
_PROCESS_OVERHEAD_MB = 400.0
# session_pool's specs are sized as SCHEDULING budgets for pool admission and
# overstate a whole pipeline: summed raw for the RTX 3060's unpooled
# configuration they give ~3.1GB of models against 2346MB MEASURED for the whole
# process (Settings: "the pooled config needs 4100MB against 2346MB"). The
# prior is scaled onto that measurement.
#
# The direction of the error is chosen, not incidental: an underestimate leaves
# the governor inert (exactly the behaviour before it existed) and the learned
# ratio corrects it after one render; an overestimate would silently cut the
# batch -- or the enhancer's resolution -- on a card that had room.
_PRIOR_SCALE = 0.6
# NVDEC decode surfaces (NV12, 1.5 B/px) held on the device by the pipe reader.
_NVDEC_SURFACES = 16

# Enhancer name -> (resource-spec key, network input side). GPEN sizes are
# resolved from the plan, so they are not listed here.
_ENHANCER_INPUT = {
    'GFPGAN': ('enhancer:gfpgan', 512),
    'Codeformer': ('enhancer:codeformer', 512),
    'Codeformer (fp16)': ('enhancer:codeformer', 512),
    'UltraMax': ('enhancer:ultramax', 512),
    'Restoreformer++': ('enhancer:restoreformer', 512),
    'Restore Ultra': ('enhancer:restoreformer', 512),
    'GPEN': ('enhancer:gpen', 512),
    'GPEN Ultimate': ('enhancer:gpen', 512),
    'GPEN 256': ('enhancer:gpen', 256),
    'GPEN 256 Pro': ('enhancer:gpen', 256),
    'GPEN 256 Ultra': ('enhancer:gpen', 256),
    'GPEN Realistic': ('enhancer:gpen', 256),
    'DMDNet': ('enhancer:dmdnet', 512),
}
_GPEN_SIZED = {'GPEN 1024': 1024, 'GPEN 2048': 2048}
_SWAP_INPUT = {'inswapper': 128, 'reswapper_128': 128}


def clamp_margin_gb(value: Any) -> float:
    try:
        margin = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MARGIN_GB
    if margin != margin:  # NaN
        return DEFAULT_MARGIN_GB
    return max(MARGIN_RANGE_GB[0], min(MARGIN_RANGE_GB[1], margin))


def gpen_size_for(enhancer: Any) -> Optional[int]:
    """The GPEN resolution an enhancer name asks for, or None if not sized."""
    return _GPEN_SIZED.get(str(enhancer or ''))


# ── measurement ────────────────────────────────────────────────────────────

def _nvml_device(device_id: int):
    import pynvml
    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(int(device_id))


def query_vram_mb(device_id: int = 0) -> Optional[Tuple[float, float, float]]:
    """(free, used, total) device memory in MiB, or None when there is no GPU.

    NVML first: it reports the whole device, other processes included, which is
    what an OOM is decided against. `torch.cuda.mem_get_info` is also
    device-wide but initialises a CUDA context as a side effect, so it is only
    the fallback.
    """
    try:
        nvml, handle = _nvml_device(device_id)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        mib = 1024.0 ** 2
        return info.free / mib, info.used / mib, info.total / mib
    except Exception as _degrade_error:
        _swallowed("roop/vram_governor.py:query_vram_mb", _degrade_error,
                   "NVML unavailable, trying torch")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(int(device_id))
            mib = 1024.0 ** 2
            return free / mib, (total - free) / mib, total / mib
    except Exception as _degrade_error:
        _swallowed("roop/vram_governor.py:query_vram_mb", _degrade_error,
                   "no CUDA device")
    return None


# ── the job and its budget ─────────────────────────────────────────────────

@dataclass
class JobSpec:
    width: int = 1920
    height: int = 1080
    faces: int = 1
    enhancer: str = 'None'
    swap_model: str = 'realswap'
    mask_engines: Tuple[str, ...] = ()
    threads: int = 8
    batch_cap: int = 8
    swap_contexts: int = 1
    detmask_contexts: int = 1
    enhancer_contexts: int = 1
    nvdec: bool = True
    detector_size: int = 640

    def signature(self, gpen_size: Optional[int] = None) -> str:
        """What the learned ratio is keyed by. Batch and GPEN size are left out
        on purpose: they are what the planner varies, so a ratio learned at one
        batch must apply to the others or it could never be used."""
        enh = self.enhancer if gpen_size_for(self.enhancer) is None else 'GPEN-sized'
        masks = '+'.join(sorted(m for m in self.mask_engines if m))
        return '|'.join(str(v) for v in (
            self.swap_model, enh, masks or '-', self.swap_contexts,
            self.detmask_contexts, self.enhancer_contexts,
            _resolution_class(self.width, self.height)))


def _resolution_class(width: int, height: int) -> str:
    pixels = int(width) * int(height)
    if pixels <= 1280 * 720:
        return '720p'
    if pixels <= 1920 * 1080:
        return '1080p'
    if pixels <= 2560 * 1440:
        return '1440p'
    return '4k'


def _slot_mb(key: str, side: int, batch: int = 1) -> float:
    from roop.session_pool import _resource_spec
    spec = _resource_spec(key)
    return float(spec.slot_mb((batch, 3, int(side), int(side)), batch))


def estimate_budget_mb(job: JobSpec, batch_cap: int, gpen_size: Optional[int]) -> Dict[str, float]:
    """Per-component VRAM estimate in MiB, before the learned correction.

    Only the NVDEC term depends on the video's resolution: every network runs
    on a fixed-size crop or a fixed detector input, so a 4K clip costs the
    models no more than a 720p one.
    """
    parts: Dict[str, float] = {}
    swap_side = _SWAP_INPUT.get(str(job.swap_model), 256)
    # Faces in flight at the swap: the batcher coalesces crops from worker
    # threads, so a batch can never exceed threads x faces-per-frame.
    in_flight = max(1, min(int(batch_cap), int(job.threads) * max(1, int(job.faces))))
    swap_nets = 2 if str(job.swap_model) == 'realswap' else 1   # hyperswap + hififace band
    parts['swapper'] = (swap_nets * _slot_mb('swap:' + str(job.swap_model), swap_side, in_flight)
                        * max(1, int(job.swap_contexts)))
    parts['detector'] = _slot_mb('detector', job.detector_size) * max(1, int(job.detmask_contexts))
    masks = [m for m in job.mask_engines if m and m != 'None']
    parts['masks'] = sum(_slot_mb('mask:' + m, 256) for m in masks) * max(1, int(job.detmask_contexts))
    if gpen_size:
        parts['enhancer'] = _slot_mb('enhancer:gpen', gpen_size) * max(1, int(job.enhancer_contexts))
    elif str(job.enhancer) in _ENHANCER_INPUT:
        key, side = _ENHANCER_INPUT[str(job.enhancer)]
        parts['enhancer'] = _slot_mb(key, side) * max(1, int(job.enhancer_contexts))
    parts = {k: v * _PRIOR_SCALE for k, v in parts.items()}
    parts['process'] = _PROCESS_OVERHEAD_MB
    if job.nvdec:
        parts['nvdec'] = job.width * job.height * 1.5 * _NVDEC_SURFACES / (1024.0 ** 2)
    return parts


# ── the plan ───────────────────────────────────────────────────────────────

@dataclass
class VramPlan:
    free_mb: float
    total_mb: float
    margin_mb: float
    budget_mb: float
    headroom_mb: float
    batch_cap: int
    gpen_size: Optional[int]
    requested_batch_cap: int
    requested_gpen_size: Optional[int]
    ratio: float
    signature: str
    components: Dict[str, float] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    fits: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def plan_job(job: JobSpec, free_mb: float, total_mb: float,
             margin_gb: float = DEFAULT_MARGIN_GB, ratio: float = 1.0) -> VramPlan:
    """Pure: the smallest step-down that leaves `margin_gb` free, if any does."""
    margin_mb = clamp_margin_gb(margin_gb) * 1024.0
    ratio = max(_RATIO_BAND[0], min(_RATIO_BAND[1], float(ratio or 1.0)))
    requested_gpen = gpen_size_for(job.enhancer)
    batch = max(1, int(job.batch_cap))
    gpen = requested_gpen
    actions: List[str] = []

    def total(b, g):
        parts = estimate_budget_mb(job, b, g)
        return sum(parts.values()) * ratio, parts

    budget, parts = total(batch, gpen)
    # Step 1: the batch. Only down the fixed ladder, and only below the request.
    while free_mb - budget < margin_mb and batch > 1:
        lower = next((s for s in BATCH_STEPS if s < batch), 1)
        actions.append(f"swap batch {batch} -> {lower}")
        batch = lower
        budget, parts = total(batch, gpen)
    # Step 2: the enhancer resolution -- only once the batch is exhausted,
    # because this one changes the look of the output.
    while free_mb - budget < margin_mb and gpen and gpen > GPEN_STEPS[-1]:
        lower = next(s for s in GPEN_STEPS if s < gpen)
        actions.append(f"GPEN {gpen} -> {lower}")
        gpen = lower
        budget, parts = total(batch, gpen)
    return VramPlan(
        free_mb=round(free_mb, 1), total_mb=round(total_mb, 1),
        margin_mb=round(margin_mb, 1), budget_mb=round(budget, 1),
        headroom_mb=round(free_mb - budget, 1), batch_cap=batch, gpen_size=gpen,
        requested_batch_cap=max(1, int(job.batch_cap)),
        requested_gpen_size=requested_gpen, ratio=round(ratio, 3),
        signature=job.signature(), components={k: round(v * ratio, 1) for k, v in parts.items()},
        actions=actions, fits=(free_mb - budget) >= margin_mb)


# ── calibration store ──────────────────────────────────────────────────────

_store_lock = threading.Lock()


def load_ratio(signature: str, path: str = _CALIBRATION_FILE) -> float:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            row = (json.load(handle) or {}).get(signature) or {}
        return float(row.get('ratio', 1.0))
    except FileNotFoundError:
        return 1.0
    except Exception as _degrade_error:
        _swallowed("roop/vram_governor.py:load_ratio", _degrade_error, "ratio 1.0")
        return 1.0


def record_peak(signature: str, estimate_mb: float, peak_mb: float,
                path: str = _CALIBRATION_FILE) -> Optional[float]:
    """Fold one render's peak/estimate into the stored ratio (EMA)."""
    if estimate_mb <= 0 or peak_mb <= 0:
        return None
    observed = max(_RATIO_BAND[0], min(_RATIO_BAND[1], peak_mb / estimate_mb))
    with _store_lock:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle) or {}
        except FileNotFoundError:
            data = {}
        except Exception as _degrade_error:
            _swallowed("roop/vram_governor.py:record_peak", _degrade_error, "starting fresh")
            data = {}
        row = data.get(signature) or {}
        prior = row.get('ratio')
        ratio = observed if prior is None else (_EMA * observed + (1 - _EMA) * float(prior))
        data[signature] = {'ratio': round(ratio, 4), 'samples': int(row.get('samples', 0)) + 1,
                           'last_peak_mb': round(peak_mb, 1),
                           'last_estimate_mb': round(estimate_mb, 1),
                           'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    return ratio


# ── the render-time session ────────────────────────────────────────────────

class _PeakSampler:
    """Device-used peak over the render, relative to the admission baseline."""

    def __init__(self, device_id: int, baseline_used_mb: float, period: float = 0.5):
        self.device_id = device_id
        self.baseline = baseline_used_mb
        self.period = period
        self.peak_used = baseline_used_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='vram-governor-peak',
                                        daemon=True)

    def _run(self):
        while not self._stop.is_set():
            sample = query_vram_mb(self.device_id)
            if sample:
                self.peak_used = max(self.peak_used, sample[1])
            self._stop.wait(self.period)

    def start(self):
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return max(0.0, self.peak_used - self.baseline)


_active_lock = threading.Lock()
_active: Optional[Tuple[VramPlan, _PeakSampler, float]] = None


def current_plan() -> Optional[VramPlan]:
    """The plan of the render in flight, or None (preview, CPU, no render)."""
    active = _active
    return active[0] if active else None


def governed_batch_cap(requested: int) -> int:
    plan = current_plan()
    if plan is None:
        return requested
    return max(1, min(int(requested), int(plan.batch_cap)))


def governed_gpen_size(requested: int) -> int:
    plan = current_plan()
    if plan is None or plan.gpen_size is None:
        return requested
    return min(int(requested), int(plan.gpen_size))


def admit(job: JobSpec, margin_gb: float, device_id: int = 0) -> Optional[VramPlan]:
    """Plan the render and start the peak sampler. None when there is no GPU."""
    global _active
    reading = query_vram_mb(device_id)
    if reading is None:
        return None
    free_mb, used_mb, total_mb = reading
    ratio = load_ratio(job.signature())
    plan = plan_job(job, free_mb, total_mb, margin_gb, ratio)
    sampler = _PeakSampler(device_id, used_mb)
    with _active_lock:
        _active = (plan, sampler, used_mb)
    sampler.start()
    parts = ' '.join(f"{k}={v:.0f}" for k, v in plan.components.items())
    print(f"[VramGovernor] {plan.free_mb:.0f}MB free of {plan.total_mb:.0f}MB, "
          f"job estimate {plan.budget_mb:.0f}MB (x{plan.ratio} learned) [{parts}], "
          f"margin {plan.margin_mb:.0f}MB -> headroom {plan.headroom_mb:.0f}MB; "
          f"batch cap {plan.batch_cap}"
          + (f", GPEN {plan.gpen_size}" if plan.gpen_size else ''), flush=True)
    for action in plan.actions:
        print(f"[VramGovernor] step-down: {action}", flush=True)
    if not plan.fits:
        print("[VramGovernor] WARNING: still short of the margin after every "
              "step-down; the render continues on the smallest plan.", flush=True)
    return plan


def finish() -> Optional[Dict[str, float]]:
    """Stop sampling, fold the measured peak into calibration, clear the plan."""
    global _active
    with _active_lock:
        active, _active = _active, None
    if not active:
        return None
    plan, sampler, _baseline = active
    peak = sampler.stop()
    raw_estimate = plan.budget_mb / max(plan.ratio, 1e-6)
    ratio = None
    try:
        # A render that allocated almost nothing died before the models loaded;
        # it says nothing about the estimate.
        if peak > 256.0:
            ratio = record_peak(plan.signature, raw_estimate, peak)
    except Exception as _degrade_error:
        _swallowed("roop/vram_governor.py:finish", _degrade_error, "calibration not saved")
    print(f"[VramGovernor] measured peak {peak:.0f}MB vs estimate {plan.budget_mb:.0f}MB"
          + (f"; learned ratio now x{ratio:.2f} for this configuration" if ratio else ''),
          flush=True)
    return {'peak_mb': round(peak, 1), 'estimate_mb': plan.budget_mb,
            'ratio': round(ratio, 4) if ratio else None}


# ── building the job from the render's own state ────────────────────────────

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, '') or default)
    except ValueError:
        return default


def job_from_render(files, masking_engine, swap_model, facesets, total_vram_gb: float) -> JobSpec:
    """Describe a render from the arguments `batch_process_regular` received."""
    import roop.globals as g
    # The largest target in the batch decides the frame term; 1080p when none
    # can be read (an image list, a missing file).
    width, height, seen = 1920, 1080, False
    for entry in files or ():
        path = getattr(entry, 'filename', None)
        if not path:
            continue
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0 and (not seen or w * h > width * height):
                width, height, seen = w, h, True
        except Exception as _degrade_error:
            _swallowed("roop/vram_governor.py:job_from_render", _degrade_error, "assuming 1080p")
    engines = masking_engine if isinstance(masking_engine, (list, tuple)) else [masking_engine]
    small = total_vram_gb < 7.0
    # Context counts follow session_pool's tiers: an explicit ROOP_* pool wins,
    # otherwise one context under 7GB and two above. A 0 means "single context".
    swap_ctx = max(1, _int_env('ROOP_TRT_POOL', 1 if small else 2))
    detmask_ctx = max(1, _int_env('ROOP_DETMASK_POOL', 1 if small else 2))
    threads = max(1, int(getattr(g, 'execution_threads', 8) or 8))
    batch = _int_env('ROOP_BATCH_SWAP_MAX', 4 if small else 8)
    try:
        det = int(str(getattr(g, 'face_detector_size', '640')).split('x')[0])
    except ValueError:
        det = 640
    return JobSpec(
        width=width, height=height, faces=max(1, len(facesets or ()) or 1),
        enhancer=str(getattr(g, 'selected_enhancer', 'None') or 'None'),
        swap_model=str(swap_model or 'realswap'),
        mask_engines=tuple(str(e) for e in engines if e),
        threads=threads, batch_cap=max(1, min(batch, threads)),
        swap_contexts=swap_ctx, detmask_contexts=detmask_ctx,
        enhancer_contexts=1 if small else 2,
        nvdec=os.environ.get('ROOP_NVDEC', '1') != '0',
        detector_size=det)
