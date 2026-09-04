"""Optional per-model session pooling to break TensorRT's single-context
serialization.

TensorRT's execution context is NOT thread-safe (concurrent enqueue on one
context corrupts the CUDA context -> error 999), so the pipeline normally
serialises *all* GPU inference behind one global lock (see ProcessMgr._gpu_guard).
That caps GPU utilisation well below 100% even with many worker threads.

A SessionPool holds N independent onnxruntime sessions for the same model, each
with its own TensorRT engine + execution context. Because the contexts are
distinct, N worker threads can run that model concurrently and safely. Different
models running on different contexts across threads is also safe; only reuse of
the *same* context must be serialised, which the lease/return queue guarantees.

Enable with the env var ROOP_TRT_POOL=<N> (N>=2). Default (unset / <2) keeps the
original single-session behaviour byte-for-byte, so this is a no-op unless opted
in. VRAM cost scales ~N x per pooled model, so keep N small on limited GPUs.
"""
import os
import contextlib
import threading
from dataclasses import dataclass
from queue import Empty, Queue


def _detect_vram_gb() -> float:
    """Best-effort total VRAM of the active CUDA device, in GB (0 if unknown).

    Used to auto-tune the pool sizes so the same install runs on cards of very
    different capacity. Detection is deferred to first use (not import time) so
    torch's CUDA context is already initialised and import-order is irrelevant.

    ROOP_VRAM_GB overrides it. That exists because this project runs on two
    cards -- a 4070 12GB and a 3060 6GB -- and every tier below is a decision
    about the SMALLER one that could previously only be exercised by physically
    being on it. A policy you cannot run is a policy you cannot test, and the
    6GB tier had drifted into disabling both pools on the strength of a
    measurement taken when the alternative was 4 or 8 contexts. Set it to
    simulate a card and take the same code path; it changes nothing else.
    """
    forced = os.environ.get('ROOP_VRAM_GB')
    if forced:
        try:
            return float(forced)
        except ValueError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            import psutil
            return (psutil.virtual_memory().total / (1024 ** 3)) * 0.5
    except Exception:
        pass
    return 0.0


def _auto_pool_defaults():
    """VRAM-tiered defaults for (swapper pool, detmask pool).

    Each pooled instance holds its OWN TensorRT engine + context, so VRAM scales
    ~Nx per pooled model. The pools raise GPU concurrency/throughput (the swapper
    pool was validated at +46% fps on a large card) but small cards can't afford
    them — on a 6GB card the extra engines OOM / trigger an endless engine-build
    thrash that drops throughput below 1fps. So: pools OFF on small cards, the
    validated multi-context settings on large cards.

        < 7 GB    (e.g. RTX 3060 6GB)       -> 0 / 0  (single context + lock)
        7-11.5 GB (e.g. 3080 10GB)          -> 2 / 2
        11.5-15.5 GB (e.g. RTX 4070 12GB)   -> 2 / 2
        >= 15.5 GB (e.g. RTX 3090 24GB)     -> 8 / 8

    Note the 11.5 boundary: a nominal "12GB" card reports ~11.99GB to torch
    (RTX 4070 = 12282 MiB), so the large-card tier must sit just below 12 to
    catch them — otherwise a 12GB card gets demoted to 2/2 and loses throughput.

    2026-08-16 correction: the 12GB tier used to return 4/4. Measured directly
    against the real pipeline (not a synthetic benchmark) on an RTX 4070: the
    detect/mask pre-pass ran at 2-2.5 fps at pool=8 (VRAM/PCIe thrashing — see
    _advisory_pool_size below) and detection specifically showed ZERO improvement
    going 2->4 (compute/GIL-bound, not context-bound; see the
    detect-mask-deserialized project memory), while pool=2 measured 45.3 fps
    for the same stage. So 4 bought nothing over 2 on this tier and was pure
    downside risk — lowered the tier default to 2 to match what was actually
    validated, not what seemed proportional.
    """
    gb = _detect_vram_gb()
    if gb <= 0:
        return 0, 0          # unknown / CPU-only -> safest
    if gb < 7:
        return 0, 0
    if gb < 11.5:
        return 2, 2
    if gb < 15.5:
        return 2, 2          # 12GB cards (e.g. RTX 4070): validated 2 swapper, 2 detmask (9.1GB VRAM, peak compute)
    return 4, 4              # 16GB+ cards (e.g. RTX 3090/4080/4090): 4 swapper, 4 detmask


def _advisory_pool_size(gb: float, auto_value: int) -> int:
    """The pool size above which an explicit override gets a WARNING — not a clamp.

    This used to be a hard ceiling that silently overrode the user. It no longer
    is: an explicit ROOP_*_POOL / perf_*_pool value is now honoured exactly, so
    the machine runs at whatever the operator asked for. A setting that quietly
    does something other than what it says is worse than a fast setting and a
    slow one — this project has been bitten repeatedly by controls that looked
    wired and were not.

    The number below is kept because the MEASUREMENT behind it is still true and
    is the only reason anyone would want the warning:

      - Each pooled instance holds its OWN TensorRT engine and execution context,
        and TensorRT allocates that context's device memory lazily on the FIRST
        INFERENCE, not at session-build time (see the
        trt-allocates-on-first-inference note). So nothing observes an
        over-large pool until real frames are already flowing.
      - Measured on an RTX 4070 (12GB) against the real pipeline: pool=8 ran the
        detect/mask pre-pass at 2-2.5 fps; pool=2 ran the SAME stage on the SAME
        clip at 45.3 fps — 18-20x. pool=4 showed no gain over pool=2 for detect
        specifically (compute/GIL-bound, not context-bound).
      - 2026-08-25, whole-render fps on the same card, GPEN 256 Pro, 10 threads,
        counterbalanced, identical 401 faces enhanced in every arm:

            trt/detmask   2/2  21.71 fps   GPU 36.5%
                          2/4  18.44 fps   GPU 29.1%    -15%
                          4/4  17.69 fps   GPU 35.9%    -19%
                          6/6   5.90 fps   GPU 62.4%    -73%

        THE THRESHOLD MOVED BECAUSE OF THAT SECOND ROW. detmask=4 sits exactly
        AT the old advisory (auto 2 x 2) and so printed nothing, while costing
        15% — so the advisory now fires above the auto default itself rather
        than above twice it. A number that only warns once a setting is
        catastrophic is not much of an advisory.
      - The 6/6 arm is also the clearest statement of what this number is NOT
        about: one of its two runs reported 94.5% GPU UTILISATION at 1.51 fps,
        against 43.0% at 23.44 fps for 2/2 — 14x slower at more than twice the
        utilisation. Utilisation is time-coverage, not work completed, and
        paging TensorRT contexts over PCIe covers a great deal of time.
      - The failure mode is the card pinned near 100% VRAM with the driver paging
        contexts over PCIe: 100% "utilisation" at a third of the power limit,
        which is thrashing, not compute. A ~100 ms frame becomes minutes, and it
        is indistinguishable from a hang.

    That last point is the whole reason the warning survives the clamp's removal:
    someone who sets 8, sees 0.2 fps and concludes the app is broken deserves the
    pointer. It prints once and then gets out of the way.
    """
    if gb <= 0:
        return max(auto_value, 1)
    return max(auto_value, 1)


_warned = set()


def _warn_if_oversubscribed(env_name, requested, auto_value, gb):
    """One line, once per knob, when a value is past what was measured safe."""
    advisory = _advisory_pool_size(gb, auto_value)
    if requested <= advisory or env_name in _warned:
        return
    _warned.add(env_name)
    print(f"[SessionPool] {env_name}={requested} is above the measured-safe "
          f"{advisory} for this GPU ({gb:.1f}GB VRAM, auto default {auto_value}). "
          f"Honouring it. If throughput collapses (measured on a 12GB card: "
          f"pool=8 gave 2-2.5 fps against 45.3 fps at pool=2 for the same stage), "
          f"that is TensorRT context/VRAM thrashing, not a hang — lower this "
          f"knob. Re-measure with roop/bench.py or tests/two_face_video.py.")


def _resolve(env_name, auto_value, gb) -> int:
    """An explicit override wins, exactly as given. Unset/blank uses the
    VRAM-tiered auto default. The sub-7GB safety tier is an exception: its
    single-context/global-lock contract is a hard admission boundary, so a
    stale benchmark recommendation cannot turn into an OOM-prone pool after a
    restart."""
    raw = os.environ.get(env_name)
    if raw is None or raw == '':
        # RuntimeOptimizer publishes workload-derived values separately from
        # user-facing environment overrides.  They are still only hints: the
        # resource manager applies its normal model/VRAM admission below.
        raw = os.environ.get('ROOP_RUNTIME_' + env_name[5:])
    if raw is not None and raw != '':
        try:
            requested = max(0, int(raw))
        except ValueError:
            return auto_value
        if gb > 0 and gb < 7.0 and env_name in (
                'ROOP_TRT_POOL', 'ROOP_DETMASK_POOL', 'ROOP_EXPR_POOL'):
            if requested:
                print(f"[SessionPool] {env_name}={requested} rejected by the "
                      f"sub-7GB single-context safety guard ({gb:.1f}GB VRAM); "
                      "using 0", flush=True)
            return 0
        _warn_if_oversubscribed(env_name, requested, auto_value, gb)
        return requested
    return auto_value


_pool_cache = {}


# ---------------------------------------------------------------------------
# TensorRT resource accounting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelResourceSpec:
    """Conservative per-context estimate used before creating extra sessions.

    ORT/TensorRT allocates engine and execution-context memory lazily, often on
    the first inference.  The numbers here are therefore a scheduling budget,
    not a claim about the exact CUDA allocation.  The runtime manager combines
    them with the live free-memory reading and the measured resident state.
    ``workspace_mb`` is deliberately bounded: TensorRT's configured workspace
    is a build-time upper bound and charging the full value to every running
    context would reject safe pools on otherwise healthy cards.
    """

    model_key: str
    engine_mb: float
    workspace_mb: float
    context_mb: float
    activation_mb: float
    reference_pixels: int = 512 * 512
    safety_mb: float = 256.0
    # Six is the largest context count Phase 4 measures. This is a candidate
    # ceiling, not a recommendation: live resident-memory admission and a
    # matching benchmark knee can reduce it for a particular workload.
    max_contexts: int = 6

    def slot_mb(self, input_shape=None, batch_size=1) -> float:
        shape = tuple(input_shape or ())
        pixels = 0
        if len(shape) >= 2:
            dims = [int(v) for v in shape[2:] if isinstance(v, int) and v > 0]
            if len(dims) >= 2:
                pixels = dims[-1] * dims[-2]
        scale = (max(0.25, pixels / float(self.reference_pixels)) ** 0.75
                 if pixels else 1.0)
        batch = max(1, int(batch_size or 1))
        return (self.engine_mb + self.workspace_mb + self.context_mb +
                self.activation_mb * batch * scale)


def _resource_spec(model_key, input_shape=None):
    """Return a model-family estimate, never a VRAM/constant rule.

    Model families have materially different context footprints.  These
    defaults are intentionally modest and are corrected by benchmark data when
    available; the live free-memory guard remains authoritative at runtime.
    """
    key = str(model_key or 'unknown').lower()
    if 'ultramax' in key or 'codeformer' in key or 'restoreformer' in key:
        values = (260.0, 384.0, 180.0, 96.0)
        family = 'enhancer-heavy'
    elif 'gpen' in key or 'enhancer' in key:
        values = (220.0, 320.0, 150.0, 72.0)
        family = 'enhancer'
    elif 'swap' in key or 'realswap' in key or 'inswapper' in key:
        values = (180.0, 320.0, 140.0, 64.0)
        family = 'swapper'
    elif 'detector' in key or 'retina' in key or 'yolo' in key or 'yunet' in key:
        values = (120.0, 256.0, 96.0, 48.0)
        family = 'detector'
    elif 'mask' in key or 'xseg' in key or 'sam' in key or 'bisenet' in key:
        values = (160.0, 256.0, 128.0, 64.0)
        family = 'mask'
    elif 'expr' in key or 'liveportrait' in key:
        values = (360.0, 384.0, 220.0, 128.0)
        family = 'expression'
    else:
        # Unknown models are allowed, but one conservative context is the safe
        # automatic choice until a real benchmark supplies a measured cost.
        values = (256.0, 512.0, 256.0, 128.0)
        family = 'unknown'
    return ModelResourceSpec(family, *values, max_contexts=6)


def _benchmark_stage_aliases(model_key):
    """Return benchmark stage keys that can describe a live model key."""
    key = str(model_key or '').lower()
    aliases = {key}
    if key.startswith(('swapper:', 'swap:')):
        aliases.add('swap')
    if key.startswith('enhancer:'):
        aliases.add('enhance')
    if key.startswith('detector:'):
        aliases.add('detect')
    if key.startswith('mask:'):
        aliases.add(key.split(':', 1)[1])
    if key.startswith(('expression:', 'expr:')):
        aliases.add('expression')
    return aliases


def _matching_benchmark_knee(model_key, input_shape=None):
    """Return a validated automatic context knee, or ``None``.

    Persisted benchmark output is advisory. It is accepted only for the same
    GPU, TensorRT precision/provider, current model settings, and TensorRT
    tuning identity. Explicit environment values are handled by
    ``select_pool_size`` before this helper.
    """
    try:
        import roop.globals as _globals
        cfg = getattr(_globals, 'CFG', None)
        result = getattr(cfg, 'benchmark_results', None) if cfg else None
        if not isinstance(result, dict) or result.get('status') not in (None, 'success'):
            return None
        device = result.get('device') or {}
        if not device.get('gpu_name'):
            return None
        # The GPU name/VRAM checks below are useful diagnostics, but they are
        # not a sufficient cache boundary: driver, CUDA/ORT/TensorRT versions,
        # exposed precision modes, and encoder/decoder capabilities can all
        # change the measured context knee. Real benchmark output therefore
        # has to carry the canonical profile key and match this runtime before
        # it can influence automatic pool sizing.
        recorded_key = result.get('hardware_profile_key')
        if not recorded_key:
            return None
        current_key = None
        if cfg is not None:
            current_hardware = getattr(cfg, 'hardware', None) or {}
            current_key = current_hardware.get('hardware_profile_key') \
                if isinstance(current_hardware, dict) else None
        if not current_key:
            try:
                from roop.runtime_optimizer import shared_hardware_profile
                current_key = shared_hardware_profile(
                    getattr(_globals, 'cuda_device_id', 0) or 0).as_dict().get(
                        'hardware_profile_key')
            except Exception:
                current_key = None
        if not current_key or str(recorded_key) != str(current_key):
            return None
        import torch
        if not torch.cuda.is_available():
            return None
        device_id = getattr(_globals, 'cuda_device_id', 0)
        gpu = str(torch.cuda.get_device_name(device_id))
        if str(device.get('gpu_name')) != gpu:
            return None
        total = float(torch.cuda.get_device_properties(device_id).total_memory) / (1024 ** 3)
        measured_total = float(device.get('total_vram_gb') or 0.0)
        if measured_total <= 0 or abs(total - measured_total) > max(0.5, total * 0.08):
            return None
        settings = result.get('settings_measured') or {}
        measured_tuning = settings.get('trt_tuning') or {}
        if measured_tuning:
            # A context knee is coupled to builder tactics, auxiliary streams,
            # and CUDA-graph capture. Do not reuse it after any of those knobs
            # change; the engine cache namespace already protects engine
            # reuse, while this check protects the advisory pool policy.
            try:
                current_builder = max(0, min(5, int(
                    os.environ.get('ROOP_TRT_BUILDER_OPT_LEVEL', '3'))))
            except (TypeError, ValueError):
                current_builder = 3
            try:
                current_aux = max(-1, min(8, int(
                    os.environ.get('ROOP_TRT_AUX_STREAMS', '-1'))))
            except (TypeError, ValueError):
                current_aux = -1
            current_graph = str(os.environ.get('ROOP_TRT_CUDA_GRAPH', '0')).strip().lower() in (
                '1', 'true', 'yes', 'on')
            expected_tuning = {
                'builder_optimization_level': current_builder,
                'auxiliary_streams': current_aux,
                'cuda_graph': current_graph,
            }
            for field, current in expected_tuning.items():
                if field in measured_tuning:
                    measured = measured_tuning[field]
                    if field == 'cuda_graph':
                        measured = str(measured).strip().lower() in (
                            '1', 'true', 'yes', 'on')
                    else:
                        try:
                            measured = int(measured)
                        except (TypeError, ValueError):
                            return None
                    if measured != current:
                        return None
        if cfg is not None:
            for field, setting in (('provider', 'provider'),
                                   ('trt_precision', 'trt_precision')):
                measured = settings.get(field)
                current = getattr(cfg, setting, None)
                if measured not in (None, '') and current not in (None, ''):
                    if str(measured).lower() != str(current).lower():
                        return None
            key = str(model_key or '').lower()
            if key.startswith(('swapper:', 'swap:')):
                measured = str(settings.get('swap_model') or '').lower()
                current = str(getattr(cfg, 'swap_model', '') or '').lower()
                if measured and current and measured != current:
                    return None
            elif key.startswith('enhancer:'):
                measured = str(settings.get('enhancer') or '').lower().replace(' ', '')
                current = str(getattr(cfg, 'selected_enhancer', '') or '').lower().replace(' ', '')
                if measured and current and measured != current:
                    return None
            elif key.startswith('detector:'):
                measured = str(settings.get('detector_engine') or '').lower()
                current = str(getattr(cfg, 'detector_engine', '') or '').lower()
                if measured and current and measured != current:
                    return None
        expected_shape = tuple(input_shape or ())
        aliases = _benchmark_stage_aliases(model_key)
        for stage in result.get('stages') or []:
            if not isinstance(stage, dict) or stage.get('key') not in aliases:
                continue
            measured_shape = tuple(stage.get('input_shape') or ())
            if expected_shape and measured_shape and expected_shape != measured_shape:
                continue
            scaling = stage.get('scaling') or []
            if not scaling:
                continue
            try:
                candidates = [int(row['n']) for row in scaling
                              if row.get('stable', True) and int(row['n']) >= 1]
                knee = int(stage.get('best_n') or 0)
                if knee < 1 or knee not in candidates:
                    knee = max(candidates) if candidates else 0
                if knee:
                    return knee
            except (TypeError, ValueError):
                continue
    except Exception:
        # A persisted diagnostic must never make model startup fail.
        return None
    return None


def _live_vram_mb():
    """Return (free, total) device memory in MiB, or (0, 0) if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 ** 2), total / (1024 ** 2)
    except Exception:
        pass
    return 0.0, 0.0


class TensorRTResourceManager:
    """Tracks resident pools and admits work without destroying active contexts.

    ``select_pool_size`` is called before extra ORT sessions are built.  It
    considers the model family, input shape, batch, workspace/context budget,
    currently resident pools, live free VRAM and a safety margin.  A user-set
    pool is returned unchanged; explicit controls remain authoritative.
    """

    SAFETY_FRACTION = 0.10
    SAFETY_FLOOR_MB = 768.0

    def __init__(self):
        self._lock = threading.RLock()
        self._pools = {}
        self._baseline_free_mb = 0.0

    def _safety_mb(self, total_mb):
        return max(self.SAFETY_FLOOR_MB,
                   total_mb * self.SAFETY_FRACTION if total_mb else 0.0)

    def _resident_unobserved_mb(self, free_mb):
        """Account for tracked engines not visible in a stale memory reading."""
        if not self._baseline_free_mb:
            return 0.0
        observed = max(0.0, self._baseline_free_mb - free_mb)
        tracked = sum(p['slot_mb'] * p['size'] for p in self._pools.values())
        return max(0.0, tracked - observed)

    def select_pool_size(self, requested, model_key, input_shape=None,
                         batch_size=1, explicit=False):
        requested = max(0, int(requested or 0))
        if requested < 2 or explicit:
            return requested
        spec = _resource_spec(model_key, input_shape)
        slot_mb = spec.slot_mb(input_shape, batch_size)
        free_mb, total_mb = _live_vram_mb()
        with self._lock:
            if free_mb and not self._baseline_free_mb:
                self._baseline_free_mb = free_mb
            if not free_mb:
                # Unknown VRAM is not permission to allocate a wide pool.
                return 1
            safety = max(self._safety_mb(total_mb), spec.safety_mb)
            usable = free_mb - safety - self._resident_unobserved_mb(free_mb)
            affordable = int(max(0.0, usable) // max(1.0, slot_mb))
            measured_knee = _matching_benchmark_knee(model_key, input_shape)
            policy_cap = min(spec.max_contexts, measured_knee) if measured_knee else spec.max_contexts
            selected = min(requested, policy_cap, max(1, affordable))
            if selected != requested:
                print(f"[SessionPool] {model_key}: auto context budget {requested}"
                      f" -> {selected} (slot~{slot_mb:.0f}MB, free~{free_mb:.0f}MB,"
                      f" safety~{safety:.0f}MB, resident pools={len(self._pools)},"
                      f" measured knee={measured_knee or 'none'})")
            return selected

    def register(self, pool, model_key, size, input_shape=None, batch_size=1):
        spec = _resource_spec(model_key, input_shape)
        with self._lock:
            free_mb, _ = _live_vram_mb()
            if free_mb and not self._baseline_free_mb:
                self._baseline_free_mb = free_mb
            self._pools[id(pool)] = {
                'pool': pool, 'model_key': str(model_key or 'unknown'),
                'size': int(size), 'slot_mb': spec.slot_mb(input_shape, batch_size),
                'spec': spec,
            }

    def unregister(self, pool):
        with self._lock:
            self._pools.pop(id(pool), None)

    def pressure(self, pool):
        with self._lock:
            rec = self._pools.get(id(pool))
        if not rec:
            return False
        free_mb, total_mb = _live_vram_mb()
        if not free_mb:
            return False
        # A lazy ORT/TensorRT context may allocate on its first inference.  A
        # free-memory value barely above the safety floor is therefore already
        # pressure: admit no additional context until an existing lease returns.
        return free_mb < (max(self._safety_mb(total_mb), rec['spec'].safety_mb)
                          + rec['slot_mb'])

    def admission_limit(self, pool, size, active):
        # Existing leases are allowed to finish.  Under pressure, admission is
        # reduced to the number already in flight (or one when idle), so no
        # active ORT/TensorRT context is destroyed or interrupted.
        if not self.pressure(pool):
            return size
        return max(1, min(int(size), int(active or 0) or 1))

    def describe(self):
        with self._lock:
            free_mb, total_mb = _live_vram_mb()
            return {
                'free_vram_mb': round(free_mb, 1),
                'total_vram_mb': round(total_mb, 1),
                'resident_pools': len(self._pools),
                'resident_contexts': sum(p['size'] for p in self._pools.values()),
                'resident_budget_mb': round(sum(
                    p['slot_mb'] * p['size'] for p in self._pools.values()), 1),
            }


_resource_manager = TensorRTResourceManager()


def resource_manager():
    """Return the process-wide TensorRT resource manager."""
    return _resource_manager


def _pool_explicit(kind):
    names = {
        'trt': 'ROOP_TRT_POOL', 'detmask': 'ROOP_DETMASK_POOL',
        'detector': 'ROOP_DETECTOR_POOL', 'expr': 'ROOP_EXPR_POOL',
    }
    raw = os.environ.get(names.get(kind, ''))
    return raw is not None and raw != ''


def _resolve_pools():
    if not _pool_cache:
        gb = _detect_vram_gb()
        auto_trt, auto_detmask = _auto_pool_defaults()
        trt = _resolve('ROOP_TRT_POOL', auto_trt, gb)
        detmask = _resolve('ROOP_DETMASK_POOL', auto_detmask, gb)
        _pool_cache['trt'] = trt
        _pool_cache['detmask'] = detmask
        # Tests and embedders may seed _pool_cache directly.  In that case the
        # value is an already-resolved operator choice and model-specific
        # admission must not silently rewrite it.  Normal resolution records
        # provenance so only genuinely automatic values are budgeted.
        _pool_cache['_auto_trt'] = not bool(os.environ.get('ROOP_TRT_POOL'))
        _pool_cache['_auto_detmask'] = not bool(os.environ.get('ROOP_DETMASK_POOL'))
        print(f"[SessionPool] detected {gb:.1f}GB VRAM -> "
              f"ROOP_TRT_POOL={trt}, ROOP_DETMASK_POOL={detmask} "
              f"(explicit values are honoured exactly; these are the auto "
              f"defaults where none was set)")
    return _pool_cache


def pool_size(model_key=None, input_shape=None, batch_size=1) -> int:
    requested = _resolve_pools()['trt']
    if not model_key:
        return requested
    return _resource_manager.select_pool_size(
        requested, model_key, input_shape, batch_size,
        explicit=_pool_explicit('trt') or
        not _pool_cache.get('_auto_trt', False))


def providers_without_tensorrt(providers):
    """Return *providers* with the TensorRT EP removed (keeping CUDA/CPU).

    Some ONNX models don't run under the TensorRT execution provider — e.g. the
    MobileSAM decoder feeds a float `orig_im_size`, which TRT treats as a shape
    tensor and rejects ("must have type Int32 or Int64. Type is Float"). Routing
    such models to CUDA avoids the breakage; TRT's benefit on these small per-crop
    models is marginal anyway. Each entry may be a string or an (name, opts) tuple.
    """
    out = []
    for p in (providers or []):
        name = p[0] if isinstance(p, (list, tuple)) else p
        if 'Tensorrt' in name or 'TensorRT' in name:
            continue
        out.append(p)
    return out or ['CPUExecutionProvider']


def pooling_enabled() -> bool:
    return pool_size() >= 2


# Separate, opt-in pool for the face-analysis (detection/landmark/recognition)
# and mask models. Profiling showed those two stages are ~90% of video time and
# run single-threaded behind the global GPU lock (their per-call cost is dominated
# by lock-wait, not compute). Giving each its OWN pool of independent TensorRT
# contexts lets N worker threads run them concurrently — keeping TRT's fast FP16
# per-call (CUDA FP32 was benchmarked slower) while removing the serialisation.
#
# Kept distinct from ROOP_TRT_POOL (the swapper pool) so it can be tuned / turned
# off independently: each FaceAnalysis instance loads 5 small models, so VRAM
# scales with the pool size. The default is auto-tuned by VRAM (see
# _auto_pool_defaults): 0 on small cards = original single-instance + global lock
# behaviour, byte-for-byte. Set ROOP_DETMASK_POOL explicitly to override.
def detmask_pool_size(model_key=None, input_shape=None, batch_size=1) -> int:
    requested = _resolve_pools()['detmask']
    if not model_key:
        return requested
    return _resource_manager.select_pool_size(
        requested, model_key, input_shape, batch_size,
        explicit=_pool_explicit('detmask') or
        not _pool_cache.get('_auto_detmask', False))


def detmask_pooling_enabled() -> bool:
    return detmask_pool_size() >= 2


def detector_pool_size(model_key=None, input_shape=None, batch_size=1) -> int:
    """How many independent instances of the SELECTED detector to build.

    The hybrid engines (retinaface / yoloface / yunet) bring their own detector
    and only borrow buffalo_l's aux models, so widening ROOP_DETMASK_POOL alone
    parallelises the aux models while the detector itself stays single-file. One
    instance per detect worker is what actually removes that serialisation.

    ROOP_DETECTOR_POOL overrides, and is the knob to turn DOWN first when VRAM is
    tight: retinaface_r50.onnx is ~104MB per instance (yoloface_8n is ~9MB and
    yunet ~350KB, so those are close to free).
    """
    raw = os.environ.get('ROOP_DETECTOR_POOL')
    runtime_raw = raw is None or raw == ''
    if runtime_raw:
        raw = os.environ.get('ROOP_RUNTIME_DETECTOR_POOL')
    if raw:
        try:
            requested = max(1, int(raw))
        except ValueError:
            requested = None
        if requested is not None:
            gb = _detect_vram_gb()
            _auto_trt, auto_detmask = _auto_pool_defaults()
            _warn_if_oversubscribed('ROOP_DETECTOR_POOL', requested,
                                    max(auto_detmask, 1), gb)
            if gb > 0 and gb < 7.0 and requested > 1:
                print(f"[SessionPool] ROOP_DETECTOR_POOL={requested} "
                      f"clamped to 1 by the sub-7GB safety guard "
                      f"({gb:.1f}GB VRAM)", flush=True)
                requested = 1
            if not model_key:
                return requested
            return _resource_manager.select_pool_size(
                requested, model_key, input_shape, batch_size,
                explicit=not runtime_raw)
    try:
        requested = max(1, detmask_pool_size())
        if not model_key:
            return requested
        return _resource_manager.select_pool_size(
            requested, model_key, input_shape, batch_size,
            explicit=_pool_explicit('detmask') or
            not _pool_cache.get('_auto_detmask', False))
    except Exception:
        return 1


# Separate, opt-in pool for the LivePortrait expression restorer. It is the only
# GPU stage still running one-wide while the swapper, mask and detect stages run
# N-wide, so on a chunk where it is enabled its cost adds almost entirely in
# series. Measured on an RTX 4070: a 192-frame chunk went 9.00s with expression
# off to 13.53s with it on, i.e. ~4.5s of serialised work.
#
# This is now the LAST of the serialisation fixes, not the first to reach for.
# The restorer already (a) holds the GPU lock around its session runs only, not
# its CPU conversion, (b) skips the global lock entirely, since its own
# lock/lease is what keeps its contexts exclusive, and (c) overlaps its three
# independent front-half calls (ROOP_EXPR_PARALLEL). Those cost no VRAM. A pool
# is what remains for making two whole restores concurrent, and it is the only
# one that has to be paid for in engines.
#
# Measured on an RTX 4070, 100 restores over 4 threads, 256px crops, TRT FP16
# (all four produce bit-identical pixels):
#
#     old: global lock over the whole call   29.2 faces/s   34.23 ms   baseline
#     (a)+(b)+(c), no pool                   33.3 faces/s   30.06 ms      +14%
#     ROOP_EXPR_POOL=2                       37.3 faces/s   26.84 ms      +28%   +654 MiB
#     ROOP_EXPR_POOL=2 + PARALLEL=2          37.8 faces/s   26.44 ms      +29%   +899 MiB
#
# Note where that stops. A third slot measured no faster than the second: the
# stage is GPU-bound, not lock-bound — warping_spade alone is 23.4 ms of the
# 34 ms (68%), one 421 MB generator producing 512x512, and no amount of
# concurrency makes the card do that work faster. Scheduling was worth ~29% and
# that is the ceiling of it; the rest would have to come out of the model.
#
# VRAM-tiered like the other two, and for the same reason: a fixed value is
# wrong on somebody else's card. This started as a hardcoded ROOP_EXPR_POOL=2 in
# start_react.js, which is tracked and ships to every install — so a 6GB card,
# where _auto_pool_defaults deliberately turns the other two pools OFF because
# extra engines OOM or thrash below 1fps, would still have been handed two extra
# restorer contexts. Tiering it puts that decision back on the machine running
# it.
#
#     < 11.5 GB   -> 0   single context; the measured +28% is not worth being
#                        the allocation that pushes a mid-size card into an
#                        engine-rebuild thrash, and these are the LARGEST models
#                        of any pool here (~537 MB of weights per slot).
#     >= 11.5 GB  -> 2   the configuration measured above. 3 was no faster.
#
# The boundary is 11.5 for the same reason as _auto_pool_defaults': a nominal
# 12GB card reports ~11.99GB. Costs nothing on a card that never enables
# expression restore — ProcessMgr builds the restorer lazily, only once a run
# actually asks for a non-zero strength. ROOP_EXPR_POOL overrides either way.
def _auto_expression_pool() -> int:
    return 2 if _detect_vram_gb() >= 11.5 else 0


def expression_pool_size(model_key=None, input_shape=None, batch_size=1) -> int:
    # `_resolve` takes (env_name, auto_value, gb). This passed only two for as
    # long as it has existed, so ANY call raised TypeError -- which nothing
    # caught. It stayed hidden because the expression stage only initialises
    # when `expression_restore_strength > 0`, i.e. exactly when a user turns the
    # feature on. Turning it on crashed the render.
    requested = _resolve('ROOP_EXPR_POOL', _auto_expression_pool(), _detect_vram_gb())
    if not model_key:
        return requested
    return _resource_manager.select_pool_size(
        requested, model_key, input_shape, batch_size,
        explicit=_pool_explicit('expr'))


def expression_pooling_enabled() -> bool:
    return expression_pool_size() >= 2


class SessionPool:
    """A fixed set of interchangeable per-model resources (e.g. an onnxruntime
    session, optionally paired with its own io_binding). `lease()` hands one
    resource to exactly one thread for the duration of a GPU call, then returns
    it to the pool, so each underlying TensorRT context is only ever touched by
    one thread at a time."""

    def __init__(self, build_fn, size, model_key=None, input_shape=None,
                 batch_size=1, warmup_fn=None):
        self._build_fn = build_fn
        self._model_key = str(model_key or 'unknown')
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._warmup_fn = warmup_fn
        self._cv = threading.Condition()
        self._active = 0
        self._closing = False
        # Pool-wide transitions block new leases while contexts are warmed or
        # resized. This closes the race where a lease could arrive between the
        # idle check and the queue/resource replacement.
        self._transition = False
        self._admission_limit = max(1, int(size))
        self._configured_admission_limit = self._admission_limit
        self._items = [build_fn(i) for i in range(size)]
        self._q = Queue(maxsize=max(1, int(size)))
        for it in self._items:
            self._q.put(it)
        if self._items and self._model_key != 'unknown':
            _resource_manager.register(self, self._model_key, len(self._items),
                                       input_shape, batch_size)

    @property
    def size(self):
        return len(self._items)

    @property
    def active(self):
        with self._cv:
            return self._active

    @property
    def admission_limit(self):
        with self._cv:
            return self._admission_limit

    def stats(self):
        with self._cv:
            return {
                'size': len(self._items), 'active': self._active,
                'admission_limit': self._admission_limit,
                'closing': self._closing,
                'model_key': self._model_key,
            }

    def set_admission_limit(self, limit):
        """Throttle new leases; never interrupts existing leases."""
        with self._cv:
            self._configured_admission_limit = max(
                1, min(len(self._items), int(limit)))
            self._admission_limit = self._configured_admission_limit
            self._cv.notify_all()

    def refresh_pressure(self):
        with self._cv:
            limit = _resource_manager.admission_limit(
                self, len(self._items), self._active)
            self._admission_limit = max(
                1, min(self._configured_admission_limit, limit))
            self._cv.notify_all()
            return self._admission_limit

    @contextlib.contextmanager
    def lease(self):
        with self._cv:
            while True:
                if self._closing:
                    raise RuntimeError('TensorRT session pool is closed')
                if self._transition:
                    self._cv.wait()
                    continue
                self.refresh_pressure()
                if self._active < self._admission_limit:
                    try:
                        item = self._q.get_nowait()
                    except Empty:
                        self._cv.wait()
                        continue
                    self._active += 1
                    break
                self._cv.wait()
        try:
            yield item
        finally:
            with self._cv:
                self._q.put(item)
                self._active -= 1
                self._cv.notify_all()

    def warmup(self, warmup_fn=None):
        """Warm every context with caller-provided real-shape inputs.

        TensorRT engines are commonly built lazily by ORT on first inference,
        so a generic fake tensor would be unsafe.  The callback receives
        ``(item, index)`` and is responsible for using the exact production
        input shape and precision.  Warmup is serialized per context and is
        safe to call before workers start.
        """
        fn = warmup_fn or self._warmup_fn
        if fn is None:
            return 0
        with self._cv:
            if self._active or self._closing or self._transition:
                raise RuntimeError('cannot warm an active/closed session pool')
            self._transition = True
            items = list(self._items)
        try:
            for index, item in enumerate(items):
                fn(item, index)
        finally:
            with self._cv:
                self._transition = False
                self._cv.notify_all()
        return len(items)

    def resize(self, new_size):
        """Resize only while idle; callers must rebuild safely around this API."""
        new_size = max(1, int(new_size))
        with self._cv:
            if self._closing or self._transition:
                raise RuntimeError('TensorRT session pool is closed')
            if self._active:
                raise RuntimeError('cannot resize an active TensorRT session pool')
            self._transition = True
            old_size = len(self._items)
        try:
            if new_size > old_size:
                additions = [self._build_fn(i) for i in range(old_size, new_size)]
                with self._cv:
                    self._items.extend(additions)
                    self._q = Queue(maxsize=max(1, new_size))
                    for it in self._items:
                        self._q.put(it)
            elif new_size < old_size:
                with self._cv:
                    kept = self._items[:new_size]
                    self._items = kept
                    self._q = Queue(maxsize=max(1, new_size))
                    for it in kept:
                        self._q.put(it)
            with self._cv:
                self._configured_admission_limit = min(
                    self._configured_admission_limit, new_size)
                self._admission_limit = min(self._admission_limit, new_size)
        finally:
            with self._cv:
                self._transition = False
                self._cv.notify_all()
        _resource_manager.unregister(self)
        _resource_manager.register(self, self._model_key, new_size,
                                   self._input_shape, self._batch_size)
        return new_size

    def release(self):
        """Close after all leases return; active contexts are never torn down."""
        with self._cv:
            self._closing = True
            while self._active or self._transition:
                self._cv.wait()
            items, self._items = self._items, []
            self._q = Queue(maxsize=1)
            self._cv.notify_all()
        _resource_manager.unregister(self)
        items.clear()
