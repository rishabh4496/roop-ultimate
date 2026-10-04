"""Measurement-only probes for the performance baseline.

Nothing in this module changes what a render does. It answers four questions the
STAGE TIMING table cannot, so that every later optimisation stage has a reference
it must match:

  1. WHAT ran.       `log_session` -- once per distinct session: model, the provider
                     ORT actually bound (`get_providers()[0]`, NOT the requested
                     chain), whether the TensorRT EP has FP16 on, the input shape.
  2. HOW MUCH VRAM.  `vram_snapshot` -- torch's `mem_get_info` and nvidia-smi side
                     by side at pre-pass start, after the pre-pass, after each
                     processor Initialize and every 500 finished frames.
  3. HOW MANY CPU    `log_threading` -- `threadpoolctl.threadpool_info()` plus the
     threads.        OpenCV / torch / env settings, at render start and again from
                     the first worker thread (OpenCV's thread count is not
                     guaranteed to read the same there).
  4. HOW MUCH WORK.  `count` -- per-call counters: raw detector executions (and who
                     called them), pyramid triggers, every rescue attempted/gained,
                     aux model calls, stabilizer warm-up frames. Split by phase
                     (setup / prepass / main / main+warmup) so a number can be
                     attributed to the pass that spent it.

GATING. `log_session` is always on (one line per distinct session, built once).
Everything else is gated on ROOP_PROFILE=1, the flag the STAGE TIMING report already
uses, and is a single dict test when off.

WHY COUNTERS INSTEAD OF READING THE CODE. This project's recurring failure is
"reports success while not running". A flag that is read, a rescue that is defined
and a pyramid that is configured are all consistent with never executing. Every
counter here sits on the call that DOES the work (the detector, the aux model's own
`get`), not on the wrapper that decides whether to.

Every function here must be incapable of raising into the render.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

from roop.degrade import swallowed as _swallowed

_FORCED = None          # tests: True/False overrides the environment


def enabled() -> bool:
    if _FORCED is not None:
        return _FORCED
    return os.environ.get('ROOP_PROFILE', '0') == '1'


def set_enabled(value) -> None:
    """Force the gate for a test (None returns control to the environment)."""
    global _FORCED
    _FORCED = value


# ── counters ─────────────────────────────────────────────────────────────────
PHASES = ('setup', 'prepass', 'main', 'main+warmup')

_lock = threading.Lock()
_counts = defaultdict(lambda: defaultdict(int))     # key -> phase -> n
_phase = ['setup']
_tls = threading.local()
_frames = [0]
_worker_logged = [False]
FRAME_SNAPSHOT_EVERY = 500


def _phase_name() -> str:
    if getattr(_tls, 'warmup', False) and _phase[0] == 'main':
        return 'main+warmup'
    return _phase[0]


def count(key: str, n: int = 1) -> None:
    if not enabled():
        return
    try:
        ph = _phase_name()
        with _lock:
            _counts[key][ph] += n
    except Exception as exc:        # a probe must never stop a render
        _swallowed("roop/baseline_probe.py:count", exc, "counter dropped")


def set_phase(name: str) -> None:
    """'prepass' for the tracking / smoothing scans, 'main' for the frame loop."""
    if name in PHASES:
        _phase[0] = name


def reset() -> None:
    """Per clip, alongside `_prof_reset`: counters describe one run."""
    with _lock:
        _counts.clear()
    _phase[0] = 'setup'
    _frames[0] = 0
    _worker_logged[0] = False


def warmup_enter() -> None:
    _tls.warmup = True


def warmup_exit() -> None:
    _tls.warmup = False


@contextmanager
def warmup_scope():
    warmup_enter()
    try:
        yield
    finally:
        warmup_exit()


def snapshot() -> dict:
    """{key: {phase: n}} -- a copy, for tests and for the machine-readable line."""
    with _lock:
        return {k: dict(v) for k, v in _counts.items()}


def total(key: str) -> int:
    with _lock:
        return sum(_counts.get(key, {}).values())


def rescue(name: str, faces) -> None:
    """One rescue pass ran; `faces` is what it returned (None / [] = nothing)."""
    if not enabled():
        return
    count('rescue.%s.attempted' % name)
    n = len(faces) if faces else 0
    if n:
        count('rescue.%s.succeeded' % name)
        count('rescue.%s.gained' % name, n)


def rescue_skipped(name: str, engine: str) -> None:
    """A rescue was SKIPPED because `has_multiscale` read true.

    The second counter is the interesting one: it counts skips on an engine that
    never runs a pyramid at all, i.e. where the skip is bought by a config string
    alone (`detector_scale_pyramid: 'auto'` is truthy).
    """
    if not enabled():
        return
    count('rescue.%s.skipped_multiscale' % name)
    if engine not in ('retinaface', 'retinaface_r50'):
        count('rescue.%s.skipped_with_no_pyramid_engine' % name)


def wrap_counter(obj, method: str, key: str) -> bool:
    """Count calls to `obj.method` (instance attribute; the class is untouched).

    Counts the model's own `get` / `detect`, so it holds for every path that
    reaches it: `FaceAnalysis.get`, `_hybrid_detector_faces`, the CLAHE rescue,
    `ensure_landmark_3d_68`. Idempotent.
    """
    if not enabled():
        return False
    try:
        orig = getattr(obj, method)
        if getattr(orig, '_bp_counted', False):
            return False

        def counted(*args, **kwargs):
            count(key)
            return orig(*args, **kwargs)

        counted._bp_counted = True
        setattr(obj, method, counted)
        return True
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:wrap_counter", exc, "model not counted")
        return False


def caller_name(depth: int = 2) -> str:
    try:
        return sys._getframe(depth).f_code.co_name
    except Exception as _exc:
        _swallowed("roop/baseline_probe.py:caller_name", _exc, "probe value unavailable")
        return '?'


# ── sessions ─────────────────────────────────────────────────────────────────
_sessions = {}          # signature -> record
_sessions_lock = threading.Lock()


def _names(providers):
    out = []
    for p in providers or ():
        out.append(str(p[0] if isinstance(p, (tuple, list)) else p))
    return out


def _fmt_shape(shape) -> str:
    try:
        return 'x'.join(str(d) for d in shape)
    except Exception as _exc:
        _swallowed("roop/baseline_probe.py:fmt_shape", _exc, "probe value unavailable")
        return '?'


def trt_fp16_state(sess) -> str:
    """'on' / 'off' read from the LIVE session, 'n/a' when TensorRT is not the
    provider ORT actually bound. Read from the session, not the request: the
    request is exactly what 'a bad option silently drops' makes untrustworthy."""
    try:
        active = sess.get_providers()
        if not active or active[0] != 'TensorrtExecutionProvider':
            return 'n/a'
        opts = (sess.get_provider_options() or {}).get('TensorrtExecutionProvider') or {}
        raw = str(opts.get('trt_fp16_enable', '')).strip().lower()
        if raw in ('1', 'true'):
            return 'on'
        if raw in ('0', 'false'):
            return 'off'
        return 'unreported(%s)' % raw
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:trt_fp16_state", exc, "fp16 state unknown")
        return 'unknown'


def log_session(tag: str, sess, requested=None, model_file=None) -> None:
    """Print one line the first time a distinct session is built.

    Distinct = (tag, bound provider, fp16, input shape). A pool of N identical
    sessions prints once and counts N instances; a pool extra that came up on a
    different provider prints its own line -- that line is the whole point.
    """
    if sess is None or not hasattr(sess, 'get_providers'):
        return
    try:
        active = list(sess.get_providers() or [])
        bound = active[0] if active else 'none'
        fp16 = trt_fp16_state(sess)
        shapes = ','.join('%s:%s' % (i.name, _fmt_shape(i.shape))
                          for i in sess.get_inputs())
        req = _names(requested)
        if model_file is None:
            model_file = getattr(sess, '_model_path', None)
        model_file = os.path.basename(str(model_file)) if model_file else '-'
        sig = (tag, bound, fp16, shapes)
        with _sessions_lock:
            rec = _sessions.get(sig)
            if rec is not None:
                rec['instances'] += 1
                return
            rec = {'tag': tag, 'file': model_file, 'provider': bound,
                   'trt_fp16': fp16, 'input': shapes, 'requested': req,
                   'instances': 1}
            _sessions[sig] = rec
        note = ''
        if req and bound != req[0]:
            note = '  !! bound %s but %s was requested first' % (bound, req[0])
        print('[Session] %s file=%s provider=%s trt_fp16=%s input=%s requested=%s%s'
              % (tag, model_file, bound, fp16, shapes or '-',
                 ','.join(r.replace('ExecutionProvider', '') for r in req) or '-',
                 note), flush=True)
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:log_session", exc, "session not logged")


def sessions() -> list:
    with _sessions_lock:
        return [dict(v) for v in _sessions.values()]


def clear_sessions() -> None:
    with _sessions_lock:
        _sessions.clear()


_analyser_builds = [0]


def log_analyser_build(fa, allowed_modules) -> None:
    """One line per FaceAnalysis construction, NOT deduplicated.

    `log_session` prints each distinct session once, which would hide that the
    analyser is built more than once per process (the pre-render pool is built
    with every module, the render pool with only the ones the options need) --
    so a model such as genderage appears in the session table without ever being
    called in the render. This line says which build owned what.
    """
    try:
        _analyser_builds[0] += 1
        models = sorted(getattr(fa, 'models', {}) or {})
        lazy = getattr(fa, 'lm68_model', None) is not None
        print('[Session] buffalo_l analyser build #%d allowed_modules=%s models=%s '
              'lm68_lazy_model=%s' % (_analyser_builds[0],
                                      'all' if allowed_modules is None else sorted(allowed_modules),
                                      models, lazy), flush=True)
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:log_analyser_build", exc, "build not logged")


# ── VRAM ─────────────────────────────────────────────────────────────────────
_vram_log = []
_vram_lock = threading.Lock()
_NVSMI_TIMEOUT_S = 8.0


def _device_id() -> int:
    try:
        import roop.globals as g
        return int(getattr(g, 'cuda_device_id', 0) or 0)
    except Exception as _exc:
        _swallowed("roop/baseline_probe.py:device_id", _exc, "probe value unavailable")
        return 0


def nvidia_smi_used_mib(device_id=None):
    """Device-wide used VRAM (MiB) as the driver reports it, or None."""
    dev = _device_id() if device_id is None else int(device_id)
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = 0x08000000        # CREATE_NO_WINDOW
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
             '-i', str(dev)],
            capture_output=True, text=True, timeout=_NVSMI_TIMEOUT_S, **kwargs)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0])
    except Exception as _exc:
        _swallowed("roop/baseline_probe.py:nvidia_smi_used_mib", _exc, "probe value unavailable")
        return None


def _torch_vram():
    try:
        from roop import predictor
        return predictor.device_memory()
    except Exception as _exc:
        _swallowed("roop/baseline_probe.py:torch_vram", _exc, "probe value unavailable")
        return None


def _rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 30
    except Exception as _exc:
        _swallowed("roop/baseline_probe.py:rss_gb", _exc, "probe value unavailable")
        return None


def _emit_vram(label, torch_mem, rss, background):
    smi = nvidia_smi_used_mib()
    with _vram_lock:
        prev = _vram_log[-1] if _vram_log else None
        rec = {'label': label, 't': time.time(),
               'torch_used_mib': None if torch_mem is None else round(torch_mem['used'], 1),
               'torch_free_mib': None if torch_mem is None else round(torch_mem['free'], 1),
               'torch_total_mib': None if torch_mem is None else round(torch_mem['total'], 1),
               'nvidia_smi_used_mib': smi,
               'rss_gb': None if rss is None else round(rss, 3)}
        _vram_log.append(rec)
    delta = ''
    if prev is not None and smi is not None and prev.get('nvidia_smi_used_mib') is not None:
        delta = ' (%+.0f MiB vs %s)' % (smi - prev['nvidia_smi_used_mib'], prev['label'])
    print('[VRAM] %s | torch used %s / total %s MiB, free %s | nvidia-smi used %s MiB%s | rss %s GB%s'
          % (label,
             rec['torch_used_mib'], rec['torch_total_mib'], rec['torch_free_mib'],
             smi, delta, rec['rss_gb'], ' [async]' if background else ''), flush=True)


def vram_snapshot(label: str, background: bool = False) -> None:
    """torch + nvidia-smi VRAM at a labelled point.

    `background=True` takes torch's reading now (microseconds) and hands the
    nvidia-smi subprocess (~100 ms on Windows) to a daemon thread, so the
    every-500-frames sample never stalls a render worker.
    """
    if not enabled():
        return
    try:
        torch_mem = _torch_vram()
        rss = _rss_gb()
        if background:
            threading.Thread(target=_emit_vram, args=(label, torch_mem, rss, True),
                             name='baseline_vram', daemon=True).start()
        else:
            _emit_vram(label, torch_mem, rss, False)
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:vram_snapshot", exc, "vram sample dropped")


def vram_log() -> list:
    with _vram_lock:
        return [dict(r) for r in _vram_log]


def frame_tick() -> None:
    """Once per finished frame (ProcessMgr.update_progress). Every
    FRAME_SNAPSHOT_EVERY-th frame takes a VRAM sample; the first call made on a
    worker thread logs that thread's own OpenCV setting."""
    if not enabled():
        return
    try:
        with _lock:
            _frames[0] += 1
            n = _frames[0]
        if n % FRAME_SNAPSHOT_EVERY == 0:
            vram_snapshot('frame %d' % n, background=True)
        if not _worker_logged[0] and threading.current_thread() is not threading.main_thread():
            _worker_logged[0] = True
            log_threading('first-worker-thread (%s)' % threading.current_thread().name,
                          pools=False)
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:frame_tick", exc, "tick dropped")


# ── threads ──────────────────────────────────────────────────────────────────
_THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'ROOP_CV_THREADS', 'ROOP_RUNTIME_CV_THREADS',
               'ORT_NUM_THREADS')
_threading_log = []


def log_threading(label: str, pools: bool = True) -> None:
    """threadpoolctl's view of every native thread pool, plus OpenCV and torch."""
    if not enabled():
        return
    try:
        info = {}
        try:
            import cv2
            info['cv2.getNumThreads'] = cv2.getNumThreads()
            info['cv2.getNumberOfCPUs'] = cv2.getNumberOfCPUs()
            info['cv2.useOptimized'] = bool(cv2.useOptimized())
        except Exception as _exc:
            _swallowed("roop/baseline_probe.py:log_threading.cv2", _exc, "probe value unavailable")
            pass
        torch = sys.modules.get('torch')
        if torch is not None:
            try:
                info['torch.get_num_threads'] = torch.get_num_threads()
                info['torch.get_num_interop_threads'] = torch.get_num_interop_threads()
            except Exception as _exc:
                _swallowed("roop/baseline_probe.py:log_threading.torch", _exc, "probe value unavailable")
                pass
        info['os.cpu_count'] = os.cpu_count()
        try:
            import psutil
            info['psutil.physical'] = psutil.cpu_count(logical=False)
        except Exception as _exc:
            _swallowed("roop/baseline_probe.py:log_threading.psutil", _exc, "probe value unavailable")
            pass
        env = {k: os.environ[k] for k in _THREAD_ENV if k in os.environ}
        print('[Threads] %s | %s | env %s' % (
            label, ' '.join('%s=%s' % kv for kv in info.items()),
            env if env else '{} (none set)'), flush=True)
        rec = {'label': label, **info, 'env': env, 'pools': []}
        if pools:
            try:
                from threadpoolctl import threadpool_info
                for p in threadpool_info():
                    row = {'user_api': p.get('user_api'),
                           'internal_api': p.get('internal_api'),
                           'num_threads': p.get('num_threads'),
                           'version': p.get('version'),
                           'threading_layer': p.get('threading_layer'),
                           'file': os.path.basename(str(p.get('filepath', '')))}
                    rec['pools'].append(row)
                    print('[Threads]   pool %(user_api)s/%(internal_api)s '
                          'num_threads=%(num_threads)s version=%(version)s '
                          'layer=%(threading_layer)s file=%(file)s' % row, flush=True)
                if not rec['pools']:
                    print('[Threads]   threadpoolctl reports no native thread pools', flush=True)
            except Exception as exc:
                print('[Threads]   threadpoolctl unavailable: %s: %s'
                      % (type(exc).__name__, exc), flush=True)
        _threading_log.append(rec)
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:log_threading", exc, "thread info dropped")


def threading_log() -> list:
    return list(_threading_log)


# ── stabilizer geometry ──────────────────────────────────────────────────────
_stab_geometry = []


def log_stab_geometry(info: dict) -> None:
    """One structured line beside the existing `[Stabilize] parallel:` text.

    `redundant_wu_per_block` is WU / block, the ratio the brief asks for: warm-up
    frames recomputed per useful frame. `redundant_share` is WU / (block + WU), the
    share of ALL processed frames that is thrown away, because a block's warm-up is
    EXTRA work in front of its `block` frames (ProcessMgr._process_block), not part
    of it. The measured share (counters) is printed in the end-of-run report.
    """
    try:
        wu = float(info.get('wu', 0))
        block = float(info.get('block', 0)) or 1.0
        info = dict(info)
        info['redundant_wu_per_block'] = round(wu / block, 4)
        info['redundant_share'] = round(wu / (block + wu), 4)
        try:
            import psutil
            vm = psutil.virtual_memory()
            info['psutil_available_mb'] = round(vm.available / 2 ** 20, 1)
            info['psutil_total_mb'] = round(vm.total / 2 ** 20, 1)
        except Exception as _exc:
            _swallowed("roop/baseline_probe.py:stab_geometry.psutil", _exc, "probe value unavailable")
            pass
        _stab_geometry.append(info)
        print('[StabGeometry] ' + ' '.join('%s=%s' % kv for kv in info.items()), flush=True)
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:log_stab_geometry", exc, "geometry not logged")


def stab_geometry() -> list:
    return list(_stab_geometry)


# ── report ───────────────────────────────────────────────────────────────────
def report() -> None:
    """The counter table, printed beside STAGE TIMING (ROOP_PROFILE only), plus a
    single machine-readable JSON line the harness parses verbatim."""
    if not enabled():
        return
    try:
        import json
        snap = snapshot()
        if snap:
            keys = sorted(snap)
            w = max(34, max(len(k) for k in keys))
            print('\n==== BASELINE COUNTERS (ROOP_PROFILE) — calls that DID the work, by phase ====',
                  flush=True)
            print('  %-*s %9s %9s %9s %11s %9s' % (w, 'counter', *PHASES, 'total'), flush=True)
            for k in keys:
                row = snap[k]
                print('  %-*s %9d %9d %9d %11d %9d' % (
                    w, k, *(row.get(p, 0) for p in PHASES), sum(row.values())), flush=True)
            wu = total('stab.warmup_frames')
            out = total('stab.output_frames')
            if wu or out:
                print('  measured stabilizer warm-up: %d discarded vs %d output frames '
                      '= %.1f%% of processed frames (%.3f per useful frame)'
                      % (wu, out, 100.0 * wu / max(1, wu + out), wu / max(1, out)),
                      flush=True)
            print('=' * 79, flush=True)
            print('[BaselineCounters.json] ' + json.dumps(snap, sort_keys=True), flush=True)
        recs = sessions()
        if recs:
            print('==== SESSIONS BUILT (process lifetime) ====', flush=True)
            for r in sorted(recs, key=lambda r: r['tag']):
                print('  %-34s x%-2d %-26s fp16=%-6s %s' % (
                    r['tag'], r['instances'], r['provider'], r['trt_fp16'], r['input']),
                    flush=True)
            print('=' * 43, flush=True)
    except Exception as exc:
        _swallowed("roop/baseline_probe.py:report", exc, "report dropped")
