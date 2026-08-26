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
from queue import Queue


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
    VRAM-tiered auto default."""
    raw = os.environ.get(env_name)
    if raw is not None and raw != '':
        try:
            requested = max(0, int(raw))
        except ValueError:
            return auto_value
        _warn_if_oversubscribed(env_name, requested, auto_value, gb)
        return requested
    return auto_value


_pool_cache = {}


def _resolve_pools():
    if not _pool_cache:
        gb = _detect_vram_gb()
        auto_trt, auto_detmask = _auto_pool_defaults()
        trt = _resolve('ROOP_TRT_POOL', auto_trt, gb)
        detmask = _resolve('ROOP_DETMASK_POOL', auto_detmask, gb)
        _pool_cache['trt'] = trt
        _pool_cache['detmask'] = detmask
        print(f"[SessionPool] detected {gb:.1f}GB VRAM -> "
              f"ROOP_TRT_POOL={trt}, ROOP_DETMASK_POOL={detmask} "
              f"(explicit values are honoured exactly; these are the auto "
              f"defaults where none was set)")
    return _pool_cache


def pool_size() -> int:
    return _resolve_pools()['trt']


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
def detmask_pool_size() -> int:
    return _resolve_pools()['detmask']


def detmask_pooling_enabled() -> bool:
    return detmask_pool_size() >= 2


def detector_pool_size() -> int:
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
            return requested
    try:
        return max(1, detmask_pool_size())
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


def expression_pool_size() -> int:
    # `_resolve` takes (env_name, auto_value, gb). This passed only two for as
    # long as it has existed, so ANY call raised TypeError -- which nothing
    # caught. It stayed hidden because the expression stage only initialises
    # when `expression_restore_strength > 0`, i.e. exactly when a user turns the
    # feature on. Turning it on crashed the render.
    return _resolve('ROOP_EXPR_POOL', _auto_expression_pool(), _detect_vram_gb())


def expression_pooling_enabled() -> bool:
    return expression_pool_size() >= 2


class SessionPool:
    """A fixed set of interchangeable per-model resources (e.g. an onnxruntime
    session, optionally paired with its own io_binding). `lease()` hands one
    resource to exactly one thread for the duration of a GPU call, then returns
    it to the pool, so each underlying TensorRT context is only ever touched by
    one thread at a time."""

    def __init__(self, build_fn, size):
        self._items = [build_fn(i) for i in range(size)]
        self._q = Queue()
        for it in self._items:
            self._q.put(it)

    @contextlib.contextmanager
    def lease(self):
        item = self._q.get()
        try:
            yield item
        finally:
            self._q.put(item)

    def release(self):
        items, self._items = self._items, []
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass
        items.clear()
