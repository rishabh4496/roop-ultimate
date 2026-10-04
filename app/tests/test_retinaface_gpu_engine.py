"""detector_engine 'retinaface_r50_gpu' (roop/retinaface_gpu_engine.py).

Everything here pins the LOGIC around the network with a fake `_infer`: when the pyramid
triggers, that the single pass is reused as the 1.0 level, the geometry of the levels,
that the shared NMS rules (not torchvision's) decide what survives, the pool/lease, the
provider mapping, the registration points and the failure modes. The network itself is
exercised by tests/ab_gpu_engine.py on real footage.
"""

import contextlib
import os
import queue
import sys
import threading
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

torch = pytest.importorskip('torch')
import roop.globals  # noqa: E402
from roop import baseline_probe as bp  # noqa: E402
from roop import face_detector as fd  # noqa: E402
from roop import retinaface_gpu_engine as eng  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    bp.set_enabled(True)
    bp.reset()
    monkeypatch.setattr(roop.globals, 'detector_scale_pyramid', 'auto', raising=False)
    monkeypatch.setattr(roop.globals, 'face_detector_nms', 0.40, raising=False)
    yield
    bp.set_enabled(None)
    bp.reset()


def box(x0, y0, x1, y1, score=0.9):
    return [x0, y0, x1, y1, score]


def kp_for(b):
    x0, y0, x1, y1 = b[:4]
    return [[x0 + 0.3 * (x1 - x0), y0 + 0.35 * (y1 - y0)], [x0 + 0.7 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
            [x0 + 0.5 * (x1 - x0), y0 + 0.55 * (y1 - y0)], [x0 + 0.35 * (x1 - x0), y0 + 0.8 * (y1 - y0)],
            [x0 + 0.65 * (x1 - x0), y0 + 0.8 * (y1 - y0)]]


def dets(*boxes):
    if not boxes:
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    return (np.array(boxes, np.float32), np.array([kp_for(b) for b in boxes], np.float32))


class FakeInstance(eng._Instance):
    """An instance with the network replaced: `_infer` returns what the test planned for
    that call and records the image shapes it was handed."""

    def __init__(self, plan):
        self.plan, self.calls = list(plan), []
        self.max_batch, self.tensorrt, self.device, self.runner, self.det = 2, False, None, None, None

    def _infer(self, images, thresh, audit=True):
        self.calls.append([tuple(im.shape[-2:]) for im in images])
        out = self.plan.pop(0)
        assert len(out) == len(images), 'the test planned %d results for %d images' % (len(out), len(images))
        return out


H, W = 1080, 1920
FRAME = np.zeros((H, W, 3), dtype=np.uint8)
PAD = eng.context_pad(H, W)
HP, WP = H + 2 * PAD, W + 2 * PAD


# ── padding ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('h,w', [(480, 640), (720, 1280), (1080, 1920), (2160, 3840), (200, 120), (66, 90)])
def test_context_pad_matches_the_cpu_engines(h, w):
    _, offs = fd.apply_context_padding(np.zeros((h, w, 3), np.uint8))
    assert eng.context_pad(h, w) == offs[0]


def test_torch_reflect_pad_is_opencv_reflect_101():
    import cv2
    import torch.nn.functional as F
    rng = np.random.RandomState(1)
    a = rng.randint(0, 255, (50, 70, 3)).astype(np.uint8)
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    got = F.pad(t, (9, 9, 9, 9), mode='reflect')[0].permute(1, 2, 0).numpy()
    assert np.array_equal(got, cv2.copyMakeBorder(a, 9, 9, 9, 9, cv2.BORDER_REFLECT_101))


# ── trigger rules and reuse ──────────────────────────────────────────────────

@needs_cuda
def test_a_normal_frame_is_one_network_image_and_unpadded():
    face = box(400 + PAD, 200 + PAD, 520 + PAD, 340 + PAD)
    inst = FakeInstance([[dets(face)]])
    b, k = inst.detect(FRAME, 0.5, 0.4)
    assert inst.calls == [[(HP, WP)]]
    assert np.allclose(b[0, :4], [400, 200, 520, 340])               # padding removed
    assert np.allclose(k[0], np.array(kp_for([400, 200, 520, 340]), np.float32), atol=1e-3)
    assert bp.total('gpudet.path.single_scale') == 1
    assert bp.total('gpudet.pyramid_levels') == 0


@needs_cuda
def test_no_face_is_one_network_image_and_empty():
    inst = FakeInstance([[dets()]])
    b, k = inst.detect(FRAME, 0.5, 0.4)
    assert b.shape == (0, 5) and k.shape == (0, 5, 2)
    assert len(inst.calls) == 1                                       # an empty pass never triggers the pyramid


@needs_cuda
def test_a_closeup_triggers_the_pyramid_and_reuses_the_single_pass():
    big = box(800, 300, 1100, 900)                                    # 600 px tall >= 500
    inst = FakeInstance([[dets(big)],                                 # the adaptive single pass
                         [dets(box(405, 148, 560, 458)),              # level 0.5, in ITS coordinates
                          dets(box(610, 222, 830, 676))]])            # level 0.75
    b, k = inst.detect(FRAME, 0.5, 0.4)
    assert len(inst.calls) == 2                                       # single + ONE batched call for the other levels
    assert inst.calls[0] == [(HP, WP)]
    assert inst.calls[1] == [(max(16, round(HP * 0.5)), max(16, round(WP * 0.5))),
                             (max(16, round(HP * 0.75)), max(16, round(WP * 0.75)))]
    assert bp.total('gpudet.single_pass_reused') == 1
    assert bp.total('gpudet.trigger.initial_closeup') == 1
    assert bp.total('gpudet.pyramid_levels') == 3


@needs_cuda
def test_the_merge_is_the_shared_diou_rule_over_levels_in_order():
    big = box(800, 300, 1100, 900)
    l05, l075 = box(405, 148, 560, 458, 0.7), box(610, 222, 830, 676, 0.8)
    inst = FakeInstance([[dets(big)], [dets(l05), dets(l075)]])
    got_b, got_k = inst.detect(FRAME, 0.5, 0.4)

    parts_b, parts_k = [], []
    for d, k, sc in ((dets(l05)[0], dets(l05)[1], (round(WP * 0.5) / WP, round(HP * 0.5) / HP)),
                     (dets(l075)[0], dets(l075)[1], (round(WP * 0.75) / WP, round(HP * 0.75) / HP)),
                     (dets(big)[0], dets(big)[1], (1.0, 1.0))):
        rb, rk = fd.rescale_detections(d, k, scale_factor=sc)
        parts_b.append(rb)
        parts_k.append(rk)
    kept, kept_k, _ = fd.diou_nms(np.vstack(parts_b), kpss=np.vstack(parts_k), iou_thresh=0.40, offset=1.0)
    ref_b, ref_k = fd.remove_context_padding(kept, kept_k, (PAD, PAD, PAD, PAD))
    assert np.array_equal(got_b, ref_b) and np.array_equal(got_k, ref_k)


@needs_cuda
def test_explicit_single_scale_never_builds_a_pyramid(monkeypatch):
    monkeypatch.setattr(roop.globals, 'detector_scale_pyramid', 'none', raising=False)
    inst = FakeInstance([[dets(box(800, 300, 1100, 900))]])
    inst.detect(FRAME, 0.5, 0.4)
    assert len(inst.calls) == 1
    assert bp.total('gpudet.path.explicit_single_scale') == 1


@needs_cuda
def test_configured_scales_have_no_single_pass_and_run_every_level_in_one_batch():
    inst = FakeInstance([[dets(box(200, 200, 300, 330)), dets(box(400, 400, 600, 660))]])
    inst.detect(FRAME, 0.5, 0.4, scales='0.5,1.0')
    assert len(inst.calls) == 1 and len(inst.calls[0]) == 2          # both levels, including 1.0, one call
    assert bp.total('gpudet.single_pass_reused') == 0


@needs_cuda
def test_estimated_height_trigger_skips_the_single_pass():
    inst = FakeInstance([[dets(), dets()]])                           # the two non-1.0 levels... plus 1.0 below
    inst.plan = [[dets(), dets(), dets()]]
    inst.detect(FRAME, 0.5, 0.4, estimated_face_height=600.0)
    assert len(inst.calls) == 1 and len(inst.calls[0]) == 3
    assert bp.total('gpudet.trigger.estimated_face_height') == 1


@needs_cuda
def test_trigger_thresholds_are_the_cpu_pyramids():
    # the module uses the shared function itself, so these are pinned in test_pyramid_reuse
    assert eng.should_trigger_pyramid is fd.should_trigger_pyramid
    assert eng.DEFAULT_PYRAMID_SCALES is fd.DEFAULT_PYRAMID_SCALES


# ── NMS: the app's rule, not torchvision's ───────────────────────────────────

@needs_cuda
def test_duplicates_collapse_but_a_partly_hidden_second_face_survives():
    a = box(500, 300, 640, 440, 0.95)
    dup = box(503, 303, 643, 443, 0.90)         # the same face fired twice: concentric, collapses
    behind = box(540, 300, 680, 440, 0.80)      # IoU 0.56 with `a`, centres 0.29 face-widths apart: a second face
    cands = np.array([a, dup, behind], np.float32)
    inst = FakeInstance([[(cands, np.array([kp_for(x) for x in (a, dup, behind)], np.float32))]])
    b, _ = inst.detect(FRAME, 0.5, 0.4)
    assert len(b) == 2 and np.allclose(sorted(b[:, 0]), [500 - PAD, 540 - PAD])      # `a` and `behind` (padding removed); `dup` gone
    # the shared rule's verdict, computed directly...
    assert len(eng.nms_keep(cands, 0.4, offset=1.0)) == 2
    # ...and the plain-IoU rule face_engine uses by default would have deleted the second face
    from torchvision.ops import nms
    plain = nms(torch.tensor(cands[:, :4]), torch.tensor(cands[:, 4]), 0.4)
    assert len(plain) == 1


# ── preprocessing: the squash geometry ───────────────────────────────────────

class _CanvasDet:
    """Stands in for RetinaFaceR50Detector: blob_cuda is never reached in squash mode, and
    decode_cuda returns what a perfect network would for one face drawn on the 640 canvas."""

    def __init__(self, box640):
        self.box640 = box640

    def _priors_cuda(self, device):
        return torch.zeros((8, 4), device=device)

    def decode_cuda(self, loc, conf, landms, info):
        x0, y0, x1, y1 = self.box640
        boxes = torch.zeros((1, 8, 4), device=loc.device)
        boxes[0, 0] = torch.tensor([x0, y0, x1, y1])
        kps = torch.zeros((1, 8, 5, 2), device=loc.device)
        kps[0, 0] = torch.tensor([[x0, y0], [x1, y0], [(x0 + x1) / 2, (y0 + y1) / 2], [x0, y1], [x1, y1]])
        scores = torch.zeros((1, 8), device=loc.device)
        scores[0, 0] = 0.99
        # identity letterbox info: the canvas IS the output
        assert info.scale == 1.0 and info.pad_x == 0.0 and info.pad_y == 0.0
        return boxes, kps, scores


class _Runner:
    max_batch = 2
    input_names = ('input',)
    output_names = ('loc', 'conf', 'landms')

    def run_binding(self, inputs, output_shapes=None):
        return {n: torch.zeros(s, device=inputs['input'].device) for n, s in output_shapes.items()}


def _squash_instance(box640):
    inst = eng._Instance.__new__(eng._Instance)
    inst.det, inst.runner, inst.max_batch, inst.tensorrt, inst.device = _CanvasDet(box640), _Runner(), 2, False, None
    return inst


@needs_cuda
def test_squash_maps_the_canvas_back_with_per_axis_scales(monkeypatch):
    pytest.importorskip('face_engine', reason='needs the repo root on sys.path')
    eng._import_face_engine()
    monkeypatch.setenv('ROOP_R50_GPU_PREPROCESS', 'squash')
    inst = _squash_instance((100, 200, 200, 400))
    img = torch.zeros((1, 3, 360, 1280), dtype=torch.uint8, device='cuda')          # 16:9: x x2, y x0.5625
    (d, k), = inst._infer([img], 0.5)
    assert np.allclose(d[0, :4], [100 * 1280 / 640, 200 * 360 / 640, 200 * 1280 / 640, 400 * 360 / 640])
    assert np.allclose(k[0, 0], [100 * 2.0, 200 * 0.5625]) and np.allclose(k[0, 4], [200 * 2.0, 400 * 0.5625])


@needs_cuda
def test_squash_blob_is_the_cv2_resize_the_existing_engine_applies(monkeypatch):
    import cv2
    pytest.importorskip('face_engine', reason='needs the repo root on sys.path')
    eng._import_face_engine()
    rng = np.random.RandomState(3)
    frame = rng.randint(0, 255, (540, 960, 3)).astype(np.uint8)
    inst = _squash_instance((0, 0, 1, 1))
    blob, info = inst._squash_blob(torch.from_numpy(frame).cuda().permute(2, 0, 1)[None])
    ref = cv2.resize(frame, (640, 640)).astype(np.float32) - np.array([104.0, 117.0, 123.0], np.float32)
    got = blob[0].permute(1, 2, 0).cpu().numpy()
    assert np.abs(got - ref).max() <= 1.0           # cv2 rounds to uint8; torch stays float
    assert (info.scale, info.pad_x, info.pad_y) == (1.0, 0.0, 0.0) and info.frame_size == (540, 960)


def test_the_default_is_squash_and_letterbox_is_an_explicit_choice(monkeypatch):
    monkeypatch.delenv('ROOP_R50_GPU_PREPROCESS', raising=False)
    assert eng.preprocess_mode() == 'squash' == eng.DEFAULT_PREPROCESS
    monkeypatch.setenv('ROOP_R50_GPU_PREPROCESS', 'letterbox')
    assert eng.preprocess_mode() == 'letterbox'
    monkeypatch.setenv('ROOP_R50_GPU_PREPROCESS', 'nonsense')
    assert eng.preprocess_mode() == 'squash'


# ── pool, lease, providers, resolution ───────────────────────────────────────

def test_lease_returns_the_instance_even_when_detect_raises(monkeypatch):
    q = queue.Queue()
    inst = object()
    q.put(inst)
    monkeypatch.setattr(eng, '_ensure_pool', lambda: {'items': [inst], 'q': q})
    with pytest.raises(RuntimeError):
        with eng.lease() as got:
            assert got is inst and q.empty()
            raise RuntimeError('boom')
    assert q.qsize() == 1


def test_detect_passes_the_globals_nms_threshold_and_the_gate(monkeypatch):
    seen = {}

    class Inst:
        def detect(self, frame, thresh, nms, scales=None, estimated_face_height=None):
            seen.update(thresh=thresh, nms=nms)
            return dets()

    q = queue.Queue()
    q.put(Inst())
    monkeypatch.setattr(eng, '_ensure_pool', lambda: {'items': [], 'q': q})
    monkeypatch.setattr(roop.globals, 'face_detector_nms', 0.3, raising=False)
    eng.detect(FRAME, det_thresh=0.55)
    assert seen == {'thresh': 0.55, 'nms': 0.3}


def test_provider_order_follows_the_app_policy(monkeypatch):
    pytest.importorskip('face_engine', reason='needs the repo root on sys.path')
    eng._import_face_engine()
    from face_engine.core.config import Provider
    monkeypatch.setattr(roop.globals, 'CFG', types.SimpleNamespace(force_cpu=False), raising=False)
    monkeypatch.setattr(roop.globals, 'execution_providers',
                        ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], raising=False)
    assert eng._provider_order() == [Provider.TENSORRT, Provider.CUDA, Provider.CPU]
    monkeypatch.setattr(roop.globals, 'execution_providers',
                        ['CUDAExecutionProvider', 'CPUExecutionProvider'], raising=False)
    assert eng._provider_order() == [Provider.CUDA, Provider.CPU]           # TensorRT not admitted (the 3060 tier)
    monkeypatch.setattr(roop.globals, 'CFG', types.SimpleNamespace(force_cpu=True), raising=False)
    assert eng._provider_order() == [Provider.CPU]


def test_model_resolution_prefers_the_registry_copy_and_falls_back_loudly(monkeypatch, capsys):
    eng._import_face_engine()
    path, from_registry = eng._resolve_model()
    # the copy the compiled engine is stamped against (<repo>/.cache/models), not app/models
    assert from_registry and path.replace('\\', '/').endswith('.cache/models/retinaface_r50.onnx')
    import face_engine.models.zoo as zoo
    monkeypatch.setattr(zoo, 'build_default_registry', lambda: (_ for _ in ()).throw(OSError('offline')))
    path, from_registry = eng._resolve_model()
    assert not from_registry and path.replace('\\', '/').endswith('app/models/retinaface_r50.onnx')


def test_pool_refuses_without_cuda_instead_of_returning_no_faces(monkeypatch):
    monkeypatch.setattr(eng, '_pool', None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA device'):
        eng._ensure_pool()


def test_a_missing_face_engine_package_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(eng, '_repo_root', lambda: os.path.join(HERE, 'no_such_root'))
    for m in [m for m in sys.modules if m == 'face_engine' or m.startswith('face_engine.')]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(sys, 'path', [p for p in sys.path if os.path.basename(p) != 'roop-ultimate'
                                      and not os.path.exists(os.path.join(p, 'face_engine'))])
    with pytest.raises(RuntimeError, match='needs the face_engine package'):
        eng._import_face_engine()


def test_runner_logging_dedups_and_names_the_provider(capsys):
    bp.clear_sessions()
    for _ in range(2):
        bp.log_runner('detector:x', file='C:/m/retinaface_r50.onnx', provider='TensorRT AOT engine',
                      trt_fp16='on', input_shape='input:Bx3x640x640', requested=['Tensorrt'], instances=2)
    out = capsys.readouterr().out
    assert out.count('[Session] detector:x') == 1
    assert 'provider=TensorRT AOT engine' in out and 'instances=2' in out
    assert bp.sessions()[0]['instances'] == 4


# ── registration, and the other engines untouched ────────────────────────────

def test_the_engine_is_registered_everywhere_it_has_to_be_and_the_others_still_are():
    fu = pytest.importorskip('roop.face_util')
    assert 'retinaface_r50_gpu' in fu._HYBRID_ENGINES
    for old in ('yoloface', 'retinaface', 'retinaface_r50', 'yunet'):
        assert old in fu._HYBRID_ENGINES
    api_src = open(os.path.join(APP, 'api.py'), encoding='utf-8').read()
    assert '"retinaface_r50", "retinaface_r50_gpu", "yunet"' in api_src
    raw = open(os.path.join(APP, 'roop', 'face_util.py'), encoding='utf-8').read()
    for old_branch in ("engine == 'yoloface'", "engine == 'retinaface'", "engine == 'retinaface_r50'",
                       "engine == 'yunet'"):
        assert old_branch in raw
    assert "engine == 'retinaface_r50_gpu'" in raw


def test_dispatch_goes_through_the_hybrid_wrapper_and_the_aux_models(monkeypatch):
    fu = pytest.importorskip('roop.face_util')
    seen = {}
    monkeypatch.setattr(roop.globals, 'detector_engine', 'retinaface_r50_gpu', raising=False)
    monkeypatch.setattr(eng, 'detect', lambda frame, **kw: seen.setdefault('kw', kw) and dets(box(10, 10, 60, 70)))

    class Model:
        def get(self, img, face):
            seen['aux'] = seen.get('aux', 0) + 1

    fa = types.SimpleNamespace(models={'recognition': Model(), 'landmark_2d_106': Model()},
                               det_model=None)

    @contextlib.contextmanager
    def lease():
        yield fa

    monkeypatch.setattr(fu, 'lease_face_analyser', lease)
    out = fu._detect_faces_raw(np.zeros((120, 160, 3), np.uint8), det_thresh=0.55)
    assert len(out) == 1 and seen['aux'] == 2                          # aux models unchanged: one get() each
    assert seen['kw'] == {'det_thresh': 0.55}
    assert fu._hybrid_engine_active()


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
