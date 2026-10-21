"""A Stage 3 frame that dies of a device out-of-memory error is rerun alone, once (Phase 8, roop/gpu_recovery.py).

The claims, each with a negative control where one is possible:
  * which errors count as memory errors, and that a deterministic one is NOT retried
  * the gate: a recovery runs with NO other frame inside, and new frames wait while it is pending
  * the retry gets its own input; a second failure re-raises the ORIGINAL error so the caller's handler still runs
  * a stopped render does not hang a recovery behind a parked worker
  * the two ProcessMgr call sites go through run_frame and hand the retry a copy of the pristine snapshot
"""
import os
import sys
import threading
import time

import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import gpu_recovery as gr  # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    gr.reset()
    monkeypatch.setattr(gr, 'flush_gpu_caches', lambda: {'gc': True, 'torch_cache': False})
    monkeypatch.delenv('ROOP_GPU_RECOVERY', raising=False)
    yield
    gr.reset()


# ── classification ───────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('msg', [
    'onnxruntime: [ONNXRuntimeError] : 1 : FAIL : CUDA failure 2: out of memory ; GPU=0',
    'CUDA error: out of memory',
    'TensorRT: Error Code 2: OutOfMemory (Requested size was 805306368 bytes.)',
    'cudaErrorMemoryAllocation',
    'CUDNN_STATUS_ALLOC_FAILED',
    'BFCArena::AllocateRawInternal Failed to allocate memory for requested size 1572864',
])
def test_memory_errors_are_classified_oom(msg):
    assert gr.classify(RuntimeError(msg)) == 'oom'


def test_torch_oom_class_and_host_memoryerror_are_oom():
    class OutOfMemoryError(RuntimeError):
        pass
    assert gr.classify(OutOfMemoryError('x')) == 'oom'
    assert gr.classify(MemoryError()) == 'oom'


def test_the_cause_chain_is_walked():
    try:
        try:
            raise RuntimeError('CUDA failure 2: out of memory')
        except RuntimeError as inner:
            raise ValueError('swap failed') from inner
    except ValueError as outer:
        assert gr.classify(outer) == 'oom'


@pytest.mark.parametrize('msg', [
    'index 3 is out of bounds for axis 0 with size 3',
    "'NoneType' object has no attribute 'bbox'",
    'CUDA error: an illegal memory access was encountered',          # sticky: a retry cannot help
    'CUDA error: device-side assert triggered',
])
def test_deterministic_and_sticky_errors_are_not_oom(msg):
    assert gr.classify(RuntimeError(msg)) is None


def test_shape_binding_errors_are_named_but_not_oom():
    e = RuntimeError('onnxruntime Got invalid dimensions for input: input for the following indices index: 0 Got: 4')
    assert gr.classify(e) == 'shape'


# ── run_frame ────────────────────────────────────────────────────────────────────────────────────────────────
def test_success_runs_once_and_touches_nothing():
    calls = []
    assert gr.run_frame(lambda: calls.append(1) or 'ok', site='t') == 'ok'
    assert calls == [1] and gr.stats()['oom_seen'] == 0


def test_oom_is_retried_once_alone_with_the_retry_callable():
    seen = []

    def first():
        seen.append('first')
        raise RuntimeError('CUDA failure 2: out of memory')

    def retry():
        seen.append('retry')
        return 'recovered'
    assert gr.run_frame(first, retry, site='t') == 'recovered'
    assert seen == ['first', 'retry']
    s = gr.stats()
    assert s['recovered'] == 1 and s['oom_seen'] == 1 and s['flushes'] == 1 and s['failed_after_retry'] == 0


def test_a_non_memory_error_is_raised_without_a_retry_or_a_flush():
    calls = []

    def first():
        calls.append(1)
        raise ValueError('bad shape in my code')
    with pytest.raises(ValueError):
        gr.run_frame(first, lambda: calls.append(2), site='t')
    assert calls == [1] and gr.stats()['flushes'] == 0


def test_a_shape_error_is_counted_and_not_retried():
    calls = []

    def first():
        calls.append(1)
        raise RuntimeError('onnxruntime Got invalid dimensions for input')
    with pytest.raises(RuntimeError):
        gr.run_frame(first, lambda: calls.append(2), site='t')
    assert calls == [1] and gr.stats()['shape_seen'] == 1 and gr.stats()['flushes'] == 0


def test_a_failed_retry_raises_the_original_error_and_counts_it():
    def first():
        raise RuntimeError('CUDA failure 2: out of memory (first)')

    def retry():
        raise RuntimeError('CUDA failure 2: out of memory (second)')
    with pytest.raises(RuntimeError) as info:
        gr.run_frame(first, retry, site='t')
    assert 'first' in str(info.value)
    s = gr.stats()
    assert s['recovered'] == 0 and s['failed_after_retry'] == 1


def test_the_retry_defaults_to_first():
    n = {'c': 0}

    def first():
        n['c'] += 1
        if n['c'] == 1:
            raise MemoryError()
        return n['c']
    assert gr.run_frame(first, site='t') == 2


def test_disabled_by_env_restores_the_old_path(monkeypatch):
    monkeypatch.setenv('ROOP_GPU_RECOVERY', '0')
    calls = []

    def first():
        calls.append(1)
        raise RuntimeError('CUDA failure 2: out of memory')
    with pytest.raises(RuntimeError):
        gr.run_frame(first, lambda: calls.append(2), site='t')
    assert calls == [1]


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_a_recovery_runs_with_no_other_frame_inside_and_new_frames_wait():
    inside = {'n': 0, 'max_during_retry': None}
    lock = threading.Lock()
    release_workers = threading.Event()
    entered = threading.Barrier(4)
    results = []

    def worker():
        def body():
            with lock:
                inside['n'] += 1
            try:
                entered.wait(5)
                release_workers.wait(5)
                return 'w'
            finally:
                with lock:
                    inside['n'] -= 1
        results.append(gr.run_frame(body, site='worker'))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()

    def failing():
        entered.wait(5)                                # all three workers are inside their frames now
        raise RuntimeError('CUDA failure 2: out of memory')

    def retry():
        with lock:
            inside['max_during_retry'] = inside['n']
        return 'recovered'
    out = {}
    rec = threading.Thread(target=lambda: out.setdefault('r', gr.run_frame(failing, retry, site='rec')))
    rec.start()
    time.sleep(0.4)
    assert 'r' not in out                              # still waiting for the workers to leave their frames
    late_started = threading.Event()
    late_done = threading.Event()

    def late():
        late_started.set()
        gr.run_frame(lambda: late_done.set(), site='late')
    lt = threading.Thread(target=late)
    lt.start()
    late_started.wait(2)
    time.sleep(0.2)
    assert not late_done.is_set()                      # a pending recovery holds NEW frames back
    release_workers.set()
    for t in threads + [rec, lt]:
        t.join(10)
    assert out['r'] == 'recovered' and inside['max_during_retry'] == 0
    assert late_done.is_set() and len(results) == 3


def test_a_stopped_render_does_not_hang_behind_a_parked_worker():
    parked = threading.Event()
    stop = threading.Event()

    def worker():
        gr.run_frame(lambda: (parked.set(), stop.wait(5)), site='parked')
    t = threading.Thread(target=worker)
    t.start()
    parked.wait(2)

    def failing():
        raise RuntimeError('CUDA failure 2: out of memory')
    started = time.perf_counter()
    with pytest.raises(RuntimeError):
        gr.run_frame(failing, lambda: 'never', site='rec', abort=lambda: True)
    assert time.perf_counter() - started < 2.0
    stop.set()
    t.join(5)


def test_the_gate_is_released_after_a_recovery():
    def failing():
        raise MemoryError()
    gr.run_frame(failing, lambda: 1, site='t')
    done = threading.Event()
    threading.Thread(target=lambda: (gr.run_frame(lambda: 1, site='t'), done.set())).start()
    assert done.wait(2)


# ── the wiring ───────────────────────────────────────────────────────────────────────────────────────────────
def _src(rel):
    with open(os.path.join(APP, rel), encoding='utf-8') as fh:
        return fh.read().replace('\r\n', '\n')


def test_both_per_frame_paths_go_through_run_frame_with_a_pristine_retry_copy():
    s = _src('roop/ProcessMgr.py')
    assert s.count('_gpu_recovery.run_frame(') == 3                      # sequential, scheduler, stabilisation
    assert s.count('self._process_frame_uncounted(fallback_frame.copy()') == 2     # the retry never reuses the mutated frame
    assert 'self.process_frame(fallback_frame.copy()' not in s                      # ...and is never counted twice (Phase 9)
    assert "site='process_videoframes'" in s and "site='scheduler'" in s


def test_the_stabilisation_path_is_wrapped_for_counted_frames_only():
    s = _src('roop/ProcessMgr.py')
    i = s.index('def _process_stab_frame')
    body = s[i:i + 3200]
    assert "site='stabilization'" in body and 'pristine.copy()' in body
    assert 'if pristine is None:' in body                                # warm-up frames pass no snapshot: no retry
    assert s.count('pristine=fallback_frame') == 1                       # only the counted-frame call site hands one
    assert 'before_wait=_dedup.abandon' in s


def test_before_wait_runs_after_leaving_the_gate_and_before_the_drain():
    order = []
    holder_in = threading.Event()
    release = threading.Event()

    def holder():
        gr.run_frame(lambda: (holder_in.set(), release.wait(5)), site='holder')
    h = threading.Thread(target=holder)
    h.start()
    holder_in.wait(2)

    def failing():
        raise RuntimeError('CUDA failure 2: out of memory')

    def retry():
        order.append('retry')
        return 1
    out = {}
    t = threading.Thread(target=lambda: out.setdefault('r', gr.run_frame(
        failing, retry, site='x', before_wait=lambda: order.append('before_wait'))))
    t.start()
    time.sleep(0.3)
    assert order == ['before_wait']                                      # ran while the holder still occupied the gate
    release.set()
    t.join(5)
    h.join(5)
    assert order == ['before_wait', 'retry']


def test_the_audit_prints_and_resets_the_recovery_counters():
    s = _src('roop/procmgr_runtime.py')
    assert 'gpu_recovery.summary_lines()' in s and '_gpu_recovery.reset()' in s
