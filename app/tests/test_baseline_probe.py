"""The baseline probes (roop/baseline_probe.py): measurement only, never behaviour.

Two things are pinned and they are different kinds of thing:

  * the PROBE's own contract -- gated off by default, phase attribution, one
    `[Session]` line per distinct session, fp16 read from the live session, the
    every-500-frames VRAM sample, the redundant-fraction arithmetic;
  * that the instrumented `_detect_faces` ladder still takes EXACTLY the decisions
    it took before, and that a counter moves only when the real call ran. A counter
    that reads on a path that never executed is this project's recurring failure
    mode ("reports success while not running"), so each rescue counter is checked
    against a stub that records whether it was really called.
"""

import contextlib
import os
import re
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import baseline_probe as bp  # noqa: E402


@pytest.fixture(autouse=True)
def _probe_on():
    bp.set_enabled(True)
    bp.reset()
    bp.clear_sessions()
    yield
    bp.set_enabled(None)
    bp.reset()
    bp.clear_sessions()


class _In:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeSess:
    def __init__(self, active, options=None, inputs=(('target', [1, 3, 256, 256]),)):
        self._active = list(active)
        self._opts = options or {}
        self._inputs = [_In(n, s) for n, s in inputs]

    def get_providers(self):
        return list(self._active)

    def get_provider_options(self):
        return dict(self._opts)

    def get_inputs(self):
        return list(self._inputs)


TRT = 'TensorrtExecutionProvider'
CUDA = 'CUDAExecutionProvider'
CPU = 'CPUExecutionProvider'


# ── gating and phases ────────────────────────────────────────────────────────

def test_counters_are_inert_when_profile_is_off():
    bp.set_enabled(False)
    bp.count('raw.total')
    bp.rescue('upscaled', [1])
    assert bp.snapshot() == {}


def test_enabled_follows_the_environment_when_not_forced(monkeypatch):
    bp.set_enabled(None)
    monkeypatch.setenv('ROOP_PROFILE', '1')
    assert bp.enabled()
    monkeypatch.setenv('ROOP_PROFILE', '0')
    assert not bp.enabled()


def test_counts_are_attributed_to_the_phase_they_ran_in():
    bp.count('a')
    bp.set_phase('prepass')
    bp.count('a', 2)
    bp.set_phase('main')
    bp.count('a', 4)
    with bp.warmup_scope():
        bp.count('a', 8)
    bp.count('a', 16)
    assert bp.snapshot()['a'] == {'setup': 1, 'prepass': 2, 'main': 20, 'main+warmup': 8}
    assert bp.total('a') == 31


def test_warmup_flag_is_per_thread_and_cleared_on_early_return():
    import threading
    bp.set_phase('main')
    seen = {}

    def worker():
        with bp.warmup_scope():
            bp.count('w')
        bp.count('w')
        seen['done'] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    bp.count('w')           # this thread was never in warm-up
    assert bp.snapshot()['w'] == {'main+warmup': 1, 'main': 2}

    def early_return():
        bp.warmup_enter()
        try:
            return
        finally:
            bp.warmup_exit()

    early_return()
    bp.count('after')
    assert bp.snapshot()['after'] == {'main': 1}


def test_reset_clears_counts_and_phase():
    bp.set_phase('main')
    bp.count('x')
    bp.reset()
    assert bp.snapshot() == {}
    bp.count('y')
    assert bp.snapshot()['y'] == {'setup': 1}


# ── rescue accounting ────────────────────────────────────────────────────────

def test_rescue_counts_attempt_success_and_gain_separately():
    bp.rescue('padded', [])
    bp.rescue('padded', None)
    bp.rescue('padded', ['f1', 'f2'])
    s = bp.snapshot()
    assert sum(s['rescue.padded.attempted'].values()) == 3
    assert sum(s['rescue.padded.succeeded'].values()) == 1
    assert sum(s['rescue.padded.gained'].values()) == 2


def test_skip_is_split_into_pyramid_engine_and_config_only():
    bp.rescue_skipped('downscaled', 'scrfd')
    bp.rescue_skipped('downscaled', 'retinaface_r50')
    s = bp.snapshot()
    assert sum(s['rescue.downscaled.skipped_multiscale'].values()) == 2
    # only the scrfd skip was bought by a config string alone
    assert sum(s['rescue.downscaled.skipped_with_no_pyramid_engine'].values()) == 1


# ── wrap_counter ─────────────────────────────────────────────────────────────

def test_wrap_counter_counts_calls_returns_the_value_and_is_idempotent():
    class Model:
        def get(self, img, face):
            return ('emb', img, face)

    m = Model()
    assert bp.wrap_counter(m, 'get', 'aux.m.get') is True
    assert bp.wrap_counter(m, 'get', 'aux.m.get') is False
    assert m.get(1, 2) == ('emb', 1, 2)
    m.get(3, 4)
    assert bp.total('aux.m.get') == 2
    assert 'get' not in Model.__dict__ or Model.get is not m.get   # class untouched


def test_wrap_counter_does_nothing_when_off():
    class Model:
        def get(self):
            return 1

    bp.set_enabled(False)
    m = Model()
    assert bp.wrap_counter(m, 'get', 'k') is False
    assert 'get' not in vars(m)


# ── sessions ─────────────────────────────────────────────────────────────────

def test_session_logged_once_with_provider_fp16_and_shape(capsys):
    opts = {TRT: {'trt_fp16_enable': 'True'}}
    for _ in range(3):
        bp.log_session('swapper:hyperswap', FakeSess([TRT, CUDA, CPU], opts), [TRT, CUDA, CPU],
                       model_file='C:/m/hyperswap_1a_256.onnx')
    out = capsys.readouterr().out
    assert out.count('[Session] swapper:hyperswap') == 1
    assert 'file=hyperswap_1a_256.onnx' in out
    assert 'provider=TensorrtExecutionProvider' in out
    assert 'trt_fp16=on' in out
    assert 'target:1x3x256x256' in out
    assert bp.sessions()[0]['instances'] == 3


@pytest.mark.parametrize('raw,expected', [('True', 'on'), ('1', 'on'), ('False', 'off'),
                                          ('0', 'off'), ('', 'unreported()')])
def test_fp16_is_read_from_the_live_session(raw, expected):
    s = FakeSess([TRT, CPU], {TRT: {'trt_fp16_enable': raw}})
    assert bp.trt_fp16_state(s) == expected


def test_fp16_is_not_applicable_when_tensorrt_is_not_the_bound_provider():
    # TensorRT was requested but ORT bound CUDA: the option must not be reported
    s = FakeSess([CUDA, CPU], {TRT: {'trt_fp16_enable': 'True'}})
    assert bp.trt_fp16_state(s) == 'n/a'


def test_a_pool_extra_on_a_different_provider_gets_its_own_line(capsys):
    bp.log_session('mask:xseg', FakeSess([TRT, CPU], {TRT: {'trt_fp16_enable': 'True'}}), [TRT, CPU])
    bp.log_session('mask:xseg', FakeSess([CPU]), [TRT, CPU])
    out = capsys.readouterr().out
    assert out.count('[Session] mask:xseg') == 2
    assert 'provider=CPUExecutionProvider' in out
    # the silent drop to CPU is called out beside the line it happened on
    assert '!! bound CPUExecutionProvider but TensorrtExecutionProvider was requested first' in out


def test_no_session_is_a_noop_not_a_swallowed_error(capsys):
    bp.log_session('detector', None, [CPU])
    assert capsys.readouterr().out == ''


# ── VRAM and the every-500-frames hook ───────────────────────────────────────

def test_vram_snapshot_pairs_torch_with_nvidia_smi(monkeypatch, capsys):
    monkeypatch.setattr(bp, '_torch_vram', lambda: {'used': 4000.0, 'free': 8000.0, 'total': 12000.0})
    monkeypatch.setattr(bp, 'nvidia_smi_used_mib', lambda device_id=None: 4021.0)
    monkeypatch.setattr(bp, '_rss_gb', lambda: 5.5)
    bp._vram_log.clear()
    bp.vram_snapshot('phase3:temporal-prepass-start')
    monkeypatch.setattr(bp, 'nvidia_smi_used_mib', lambda device_id=None: 5021.0)
    bp.vram_snapshot('phase3:temporal-prepass-complete')
    out = capsys.readouterr().out
    assert 'torch used 4000.0 / total 12000.0 MiB, free 8000.0 | nvidia-smi used 4021.0 MiB' in out
    assert '(+1000 MiB vs phase3:temporal-prepass-start)' in out
    log = bp.vram_log()
    assert [r['label'] for r in log][-2:] == ['phase3:temporal-prepass-start',
                                              'phase3:temporal-prepass-complete']


def test_frame_tick_samples_vram_every_500_frames(monkeypatch):
    labels = []
    monkeypatch.setattr(bp, 'vram_snapshot',
                        lambda label, background=False: labels.append((label, background)))
    monkeypatch.setattr(bp, 'log_threading', lambda *a, **k: None)
    for _ in range(1200):
        bp.frame_tick()
    # background=True: the nvidia-smi subprocess must never stall a render worker
    assert labels == [('frame 500', True), ('frame 1000', True)]


def test_first_worker_thread_logs_its_own_opencv_setting(monkeypatch):
    import threading
    calls = []
    monkeypatch.setattr(bp, 'log_threading', lambda label, pools=True: calls.append((label, pools)))
    bp.frame_tick()                              # main thread: nothing
    assert calls == []
    t = threading.Thread(target=bp.frame_tick, name='swap_proc_7')
    t.start()
    t.join()
    t = threading.Thread(target=bp.frame_tick, name='swap_proc_8')
    t.start()
    t.join()
    assert calls == [('first-worker-thread (swap_proc_7)', False)]    # once, no pool dump


def test_log_threading_survives_a_missing_threadpoolctl(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'threadpoolctl', None)
    bp.log_threading('render-start')
    out = capsys.readouterr().out
    assert '[Threads] render-start' in out
    assert 'threadpoolctl unavailable' in out


# ── stabilizer geometry ──────────────────────────────────────────────────────

def test_redundant_fraction_is_reported_both_ways(capsys):
    bp._stab_geometry.clear()
    bp.log_stab_geometry({'workers': 10, 'blocks_per_chunk': 20, 'block': 24, 'wu': 6})
    rec = bp.stab_geometry()[-1]
    assert rec['redundant_wu_per_block'] == pytest.approx(0.25)      # WU / block
    assert rec['redundant_share'] == pytest.approx(0.2)              # WU / (block + WU)
    assert 'psutil_available_mb' in rec
    assert '[StabGeometry]' in capsys.readouterr().out


def test_report_prints_measured_warmup_share(capsys):
    bp.set_phase('main')
    bp.count('stab.warmup_frames', 60)
    bp.count('stab.output_frames', 240)
    bp.report()
    out = capsys.readouterr().out
    assert 'BASELINE COUNTERS' in out
    assert '60 discarded vs 240 output frames = 20.0%' in out
    assert '[BaselineCounters.json]' in out


def test_report_is_silent_when_off(capsys):
    bp.count('x')
    bp.set_enabled(False)
    bp.report()
    assert capsys.readouterr().out == ''


# ── the call sites are still wired ───────────────────────────────────────────

def _src(rel):
    with open(os.path.join(APP, rel), encoding='utf-8') as fh:
        return fh.read()


@pytest.mark.parametrize('rel,needle', [
    ('roop/face_util.py', "_bp.log_session('buffalo_l:%s' % task"),
    ('roop/retinaface.py', "baseline_probe.log_session(f'retinaface:{model_type}'"),
    ('roop/processors/Mask_XSeg.py', "baseline_probe.log_session('mask:xseg'"),
    ('roop/processors/FaceSwapInsightFace.py', "baseline_probe.log_session(f'swapper:{swap_model}'"),
    ('roop/processors/Enhance_RestoreFormerPPlus.py',
     "baseline_probe.log_session('enhancer:restoreformer++'"),
    ('roop/ProcessMgr.py', "_bp.vram_snapshot(f'initialize:{key}')"),
    ('roop/ProcessMgr.py', "_bp.vram_snapshot(entry['stage'])"),
    ('roop/ProcessMgr.py', "_bp.frame_tick()"),
    ('roop/ProcessMgr.py', "_bp.log_stab_geometry(_geo)"),
    ('roop/procmgr_batch.py', "_bp.log_threading('render-start"),
    ('roop/procmgr_runtime.py', "_bp.report()"),
    ('roop/procmgr_runtime.py', "_bp.reset()"),
])
def test_probe_call_sites_are_present(rel, needle):
    assert needle in _src(rel), '%s lost its baseline probe: %s' % (rel, needle)


def test_session_logging_sits_in_the_builder_so_pool_extras_are_covered():
    # `_build` is what the pool calls for its (N-1) extras, so the log has to be
    # inside it, not after the primary's `_build()` call.
    for rel in ('roop/processors/Mask_XSeg.py', 'roop/processors/FaceSwapInsightFace.py',
                'roop/processors/Enhance_RestoreFormerPPlus.py'):
        src = _src(rel)
        m = re.search(r'def _build\(_i=0\):(.*?)return ', src, re.S)
        assert m and 'log_session' in m.group(1), rel


# ── the instrumented ladder decides exactly what it decided before ───────────

fu = pytest.importorskip('roop.face_util')


def _face(x=10.0):
    return types.SimpleNamespace(
        bbox=np.array([x, 10, x + 40, 50], dtype=np.float32),
        kps=np.array([[x + 10, 20], [x + 30, 20], [x + 20, 30], [x + 12, 40], [x + 28, 40]],
                     dtype=np.float32),
        det_score=0.9)


@pytest.fixture
def ladder(monkeypatch):
    """_detect_faces with every detector/rescue replaced by a recording stub."""
    import roop.globals as g
    calls = []
    raw_results = []

    def raw(frame, det_size=None, det_thresh=None, aux=True, unclamped=False):
        calls.append('raw')
        return list(raw_results.pop(0)) if raw_results else []

    @contextlib.contextmanager
    def no_aux_models():
        # the rotated rescues lease an analyser to run the aux models on survivors
        yield types.SimpleNamespace(models={})

    def mk(name, result):
        def stub(frame, **kw):
            calls.append(name)
            return list(result) if result else None
        return stub

    state = types.SimpleNamespace(calls=calls, raw_results=raw_results, mk=mk)
    monkeypatch.setattr(fu, '_detect_faces_raw', raw)
    monkeypatch.setattr(fu, 'lease_face_analyser', no_aux_models)
    monkeypatch.setattr(fu, '_enrich_detected_faces', lambda frame, faces: faces)
    monkeypatch.setattr(g, 'rescue_small_faces', True, raising=False)
    monkeypatch.setattr(g, 'detector_engine', 'scrfd', raising=False)
    monkeypatch.setattr(g, 'detector_scale_pyramid', 'auto', raising=False)
    monkeypatch.setattr(g, 'TARGET_FACE_GROUP', [], raising=False)
    monkeypatch.setattr(g, 'INPUT_FACESETS', [], raising=False)
    return state


FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def test_first_pass_hit_runs_no_rescue(ladder):
    ladder.raw_results.append([_face()])
    out = fu._detect_faces(FRAME)
    assert len(out) == 1
    s = bp.snapshot()
    assert bp.total('detect.calls') == 1
    assert bp.total('detect.first_pass_faces') == 1
    assert not any(k.startswith('rescue.') for k in s)


def test_truthy_auto_pyramid_skips_two_rescues_on_an_engine_that_runs_no_pyramid(ladder, monkeypatch):
    """`bool('auto')` is True, so with detector_scale_pyramid='auto' the close-up and
    padding rescues never run -- even on SCRFD, which has no pyramid. The counters
    must say so, and the stubs prove the rescues really were not called."""
    ladder.raw_results.append([])
    monkeypatch.setattr(fu, '_rescue_upscaled', ladder.mk('upscaled', []))
    monkeypatch.setattr(fu, '_rescue_downscaled', ladder.mk('downscaled', [_face()]))
    monkeypatch.setattr(fu, '_rescue_padded', ladder.mk('padded', [_face()]))
    monkeypatch.setattr(fu, '_rescue_rotated', ladder.mk('rotated', [_face()]))
    monkeypatch.setattr(fu, '_rescue_clahe', ladder.mk('clahe', []))
    out = fu._detect_faces(FRAME)
    assert len(out) == 1
    assert ladder.calls == ['raw', 'upscaled', 'rotated']            # behaviour
    assert bp.total('rescue.upscaled.attempted') == 1                 # counters agree
    assert bp.total('rescue.upscaled.succeeded') == 0
    assert bp.total('rescue.downscaled.attempted') == 0
    assert bp.total('rescue.downscaled.skipped_multiscale') == 1
    assert bp.total('rescue.downscaled.skipped_with_no_pyramid_engine') == 1
    assert bp.total('rescue.padded.skipped_multiscale') == 1
    assert bp.total('rescue.rotated.attempted') == 1
    assert bp.total('rescue.rotated.gained') == 1
    assert bp.total('rescue.clahe.attempted') == 0                    # never reached
    assert bp.total('detect.first_pass_empty') == 1


def test_without_a_pyramid_setting_every_rescue_runs_in_order(ladder, monkeypatch):
    import roop.globals as g
    monkeypatch.setattr(g, 'detector_scale_pyramid', None, raising=False)
    ladder.raw_results.append([])
    for name in ('upscaled', 'downscaled', 'padded', 'rotated', 'clahe'):
        monkeypatch.setattr(fu, '_rescue_' + name, ladder.mk(name, []))
    assert fu._detect_faces(FRAME) == []
    assert ladder.calls == ['raw', 'upscaled', 'downscaled', 'padded', 'rotated', 'clahe']
    for name in ('upscaled', 'downscaled', 'padded', 'rotated', 'clahe'):
        assert bp.total('rescue.%s.attempted' % name) == 1, name
        assert bp.total('rescue.%s.skipped_multiscale' % name) == 0, name


def test_rescue_false_skips_the_ladder_and_says_so(ladder, monkeypatch):
    ladder.raw_results.append([])
    monkeypatch.setattr(fu, '_rescue_upscaled', ladder.mk('upscaled', [_face()]))
    assert fu._detect_faces(FRAME, rescue=False) == []
    assert ladder.calls == ['raw']
    assert bp.total('detect.calls_norescue') == 1
    assert bp.total('detect.calls') == 0


def test_partial_miss_counts_each_turn_it_really_tried(ladder, monkeypatch):
    # first pass finds 1 of 2 expected; the 180 turn finds nothing new, the
    # clockwise turn finds the second person and the loop stops there
    ladder.raw_results.extend([[_face(10.0)], [], [_face(200.0)]])
    out = fu._detect_faces(FRAME, expected_count=2)
    assert len(out) == 2
    assert ladder.calls == ['raw', 'raw', 'raw']
    assert bp.total('rescue.partial_miss.entered') == 1
    assert bp.total('rescue.partial_180.attempted') == 1
    assert bp.total('rescue.partial_180.gained') == 0
    assert bp.total('rescue.partial_clockwise.attempted') == 1
    assert bp.total('rescue.partial_clockwise.gained') == 1
    assert bp.total('rescue.partial_anticlockwise.attempted') == 0     # stopped early


def test_raw_detect_counts_executions_and_names_the_caller(monkeypatch):
    import roop.globals as g

    class FakeFA:
        models = {}
        det_model = types.SimpleNamespace(input_size=(512, 512), det_thresh=0.5)

        def get(self, frame):
            return [_face()]

    @contextlib.contextmanager
    def lease():
        yield FakeFA()

    monkeypatch.setattr(fu, 'lease_face_analyser', lease)
    monkeypatch.setattr(g, 'detector_engine', 'scrfd', raising=False)
    out = fu._detect_faces_raw(FRAME)
    assert len(out) == 1
    assert bp.total('raw.total') == 1
    assert bp.total('raw.aux') == 1
    assert bp.total('raw.engine.scrfd') == 1
    assert bp.total('raw.faces_out') == 1
    assert bp.total('raw.caller.test_raw_detect_counts_executions_and_names_the_caller') == 1


def test_hybrid_aux_models_are_counted_on_the_model_not_the_caller(monkeypatch):
    """Both `_hybrid_detector_faces` and FaceAnalysis.get reach `model.get`, so a
    counter on the model sees the call whichever path made it."""
    from insightface.app.common import Face  # noqa: F401  (the helper imports it)

    class Rec:
        def get(self, img, face):
            return None

    fa = types.SimpleNamespace(models={'detection': Rec(), 'recognition': Rec(),
                                       'landmark_2d_106': Rec()}, lm68_model=None, det_model=None)
    for task, m in fa.models.items():
        if task != 'detection':
            bp.wrap_counter(m, 'get', 'aux.buffalo_l.%s.get' % task)
    boxes = np.array([[10, 10, 60, 60, 0.9], [100, 10, 150, 60, 0.9]], dtype=np.float32)
    kps = np.stack([_face(10.0).kps, _face(100.0).kps])
    faces = fu._hybrid_detector_faces(FRAME, fa, boxes, kps, aux=True)
    assert len(faces) == 2
    assert bp.total('aux.buffalo_l.recognition.get') == 2
    assert bp.total('aux.buffalo_l.landmark_2d_106.get') == 2
    fu._hybrid_detector_faces(FRAME, fa, boxes, kps, aux=False)         # no aux, no count
    assert bp.total('aux.buffalo_l.recognition.get') == 2


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
