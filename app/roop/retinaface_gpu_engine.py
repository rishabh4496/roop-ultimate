"""retinaface_r50_gpu: face_engine's on-device RetinaFace R50 as a hybrid detector engine.

Same hybrid pattern as roop/retinaface.py (this module detects bbox + 5 keypoints,
face_util pairs it with buffalo_l's aux models), same contract as
``retinaface.detect``:

    detect(frame_bgr_uint8, ...) -> (bboxes (N, 5) [x1 y1 x2 y2 score], kpss (N, 5, 2))

in the ORIGINAL frame's coordinates, highest score first.

What runs where
---------------
* The frame goes to the device ONCE (uint8). Context padding (reflect, which is
  exactly OpenCV's BORDER_REFLECT_101: asserted by test), the pyramid levels, the
  letterbox, the network (face_engine's AOT TensorRT engine, one execution context per
  pool instance), anchor decoding and the score threshold all stay on the GPU.
* Only the handful of anchors above threshold come back to the host, where the app's
  OWN rules finish the job: ``roop.nms.nms_keep`` (offset 1.0, the same concentricity
  rule every other engine answers to) per level, and ``face_detector.diou_nms`` to merge
  pyramid levels. ``face_engine``'s own ``torchvision.batched_nms`` is deliberately not used.
* The pyramid keeps ``face_detector.should_trigger_pyramid`` and its thresholds
  verbatim, and the single-pass reuse the CPU pyramid has: the adaptive single pass is
  the scale-1.0 level, so only the other levels run (batched into one network call,
  the engine's profile takes 2).

Preprocessing: squash by default, letterbox on request
------------------------------------------------------
``RetinaFaceR50Detector`` letterboxes (aspect kept, black bars). ``roop/retinaface.py``
documents the opposite for this export (a direct square resize is its calibrated input;
letterboxing "suppressed scores under TensorRT on 16:9"), and face_engine measured the
reverse against SCRFD. Both are true of different footage, so it was measured here on the
four baseline clips (docs/perf/gpu_engine_recall_2026-10-04*.json), against the EXISTING
r50 engine (same network) and against scrfd:

* ``squash``  -- reproduces the existing r50 engine to ~4 faces in 3,400 (IoU >= 0.5);
* ``letterbox`` -- loses 76 / 24 / 0 / 83 of the existing engine's faces on d1 / d4 / d6 /
  Love, mislocalises 72 faces on d1's contact stretch, and found a 0.92-score full-height
  box on an empty floor in d4. It does find more on Love (354 extra faces, many of them the
  second person).

So the default is the faithful port, ``squash`` (per-axis resize, bilinear without
antialiasing exactly as ``retinaface.py`` does it); ``ROOP_R50_GPU_PREPROCESS=letterbox``
selects face_engine's own geometry.

The pyramid note: with letterbox every level reaches the network at near-identical scale; with
squash it does too (the existing engine has the same property). The pyramid is kept because
the trigger rules are unchanged and it is the same behaviour as ``retinaface_r50``.

The model is resolved through face_engine's registry (the copy its engine was compiled
from): the AOT engine is keyed on the model file's size AND mtime, so the byte-identical
``app/models/retinaface_r50.onnx`` would be rejected and the detector would run on ONNX
Runtime without a word. The runner actually bound is logged as a ``[Session]`` line and a
missing engine is a loud warning.
"""
import os
import sys
import threading
from contextlib import contextmanager
from queue import Queue

import numpy as np

import roop.globals
from roop.degrade import swallowed as _swallowed
from roop import baseline_probe as _bp
from roop.nms import nms_keep
from roop.face_detector import (
    DEFAULT_PYRAMID_SCALES,
    MIN_BORDER_PADDING_PX,
    diou_nms,
    parse_scale_pyramid,
    remove_context_padding,
    rescale_detections,
    should_trigger_pyramid,
)

ENGINE_NAME = 'retinaface_r50_gpu'
# The faithful port. Measured 2026-10-04 (docs/perf/gpu_engine_*.md): with 'squash' this engine
# agrees with the existing retinaface_r50 engine to ~4 faces in 3,400 across d1/d4/d6/Love;
# with 'letterbox' (face_engine's own design) it loses 76/24/0/83 of that engine's faces and
# differs on d1's contact shot. 'letterbox' stays available: ROOP_R50_GPU_PREPROCESS=letterbox.
DEFAULT_PREPROCESS = 'squash'
_ZOO_MODEL = 'retinaface_r50'
_INPUT_SIZE = 640

_pool = None                        # {'items': [...], 'q': Queue}
_pool_lock = threading.Lock()       # guards pool CONSTRUCTION only


def preprocess_mode():
    """'letterbox' (face_engine's design: aspect kept, black bars) or 'squash' (a direct resize
    to 640x640 with independent axis scales -- what roop/retinaface.py documents as this
    export's calibrated input). Read per call so an A/B can flip it in one process."""
    mode = os.environ.get('ROOP_R50_GPU_PREPROCESS', DEFAULT_PREPROCESS).strip().lower()
    return mode if mode in ('letterbox', 'squash') else DEFAULT_PREPROCESS


def _repo_root():
    # app/roop/<this file> -> app/roop -> app -> repo root (where face_engine lives)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_face_engine():
    """face_engine is a sibling package at the repository root; the app's own import path
    does not include it, so add it deliberately and fail with the reason if it still is
    not there."""
    root = _repo_root()
    if root not in sys.path:
        sys.path.append(root)
    try:
        import face_engine  # noqa: F401
        from face_engine.boosted.retinaface_gpu import RetinaFaceR50Detector
        return RetinaFaceR50Detector
    except Exception as exc:
        raise RuntimeError(
            "detector_engine 'retinaface_r50_gpu' needs the face_engine package "
            "(%s) plus torch + TensorRT/onnxruntime: %s: %s" % (root, type(exc).__name__, exc))


def context_pad(h, w, min_padding=MIN_BORDER_PADDING_PX):
    """The context border `face_detector.apply_context_padding` adds (same formula; a test
    asserts they agree)."""
    return max(int(min_padding), int(min(h, w) * 0.05), MIN_BORDER_PADDING_PX)


def _provider_order():
    """The app's provider policy expressed in face_engine's vocabulary."""
    from face_engine.core.config import Provider
    names = [str(p[0] if isinstance(p, (tuple, list)) else p).lower()
             for p in (getattr(roop.globals, 'execution_providers', None) or [])]
    cfg = getattr(roop.globals, 'CFG', None)
    if cfg is not None and getattr(cfg, 'force_cpu', False):
        names = ['cpuexecutionprovider']
    order = []
    if any('tensorrt' in n for n in names):
        order.append(Provider.TENSORRT)
    if any('tensorrt' in n or 'cuda' in n for n in names):
        order.append(Provider.CUDA)
    order.append(Provider.CPU)
    return order


def _resolve_model():
    """(path, from_registry). The registry copy is the one the compiled engine is stamped
    against; the app's own copy is a byte-identical fallback that can only run on ORT."""
    try:
        from face_engine.models.zoo import build_default_registry
        return str(build_default_registry().ensure(_ZOO_MODEL)), True
    except Exception as exc:
        _swallowed("roop/retinaface_gpu_engine.py:resolve_model", exc,
                   "registry copy unavailable; using app/models (ONNX Runtime only)")
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models', 'retinaface_r50.onnx'), False


class _Instance:
    """One independent detector: its own execution context, its own priors."""

    def __init__(self, detector_cls, engine, model_path):
        self.det = detector_cls(engine=engine, model_path=model_path, score_threshold=0.0)
        self.runner = self.det.runner                       # builds the context (or the ORT session)
        self.tensorrt = bool(self.det.uses_tensorrt_engine)
        self.max_batch = int(getattr(self.runner, 'max_batch', 1) or 1)
        self.device = None

    def _squash_blob(self, im):
        """BGR minus (104, 117, 123) of the frame resized straight to 640x640, exactly the
        resize `retinaface.py` applies (cv2.resize default = bilinear, NO antialiasing). The
        returned info is the identity: decode lands on the 640 canvas, `_unsquash` maps it
        back with per-axis scales."""
        import torch
        import torch.nn.functional as F
        from face_engine.pipeline.detector import Letterbox
        from face_engine.boosted.retinaface_gpu import MEAN_BGR

        h, w = im.shape[-2:]
        x = im if im.is_floating_point() else im.float()
        x = F.interpolate(x, size=(_INPUT_SIZE, _INPUT_SIZE), mode='bilinear', align_corners=False,
                          antialias=False)
        mean = torch.tensor(MEAN_BGR, device=x.device).view(1, 3, 1, 1)
        info = Letterbox(scale=1.0, pad_x=0.0, pad_y=0.0, input_size=_INPUT_SIZE, frame_size=(h, w))
        return (x - mean).contiguous(), info

    # ── network ──────────────────────────────────────────────────────────────
    def _infer(self, images, thresh, audit=True):
        """One entry per image: (dets (N,5) float32 sorted by score, kps (N,5,2)), NOT yet
        NMS'd, in each image's own coordinates. `images`: (1,3,h,w) BGR tensors on the device."""
        import torch

        blobs, infos = [], []
        squash = preprocess_mode() == 'squash'
        for im in images:
            blob, info = self._squash_blob(im) if squash else self.det.blob_cuda(im)
            blobs.append(blob)
            infos.append(info)
        blob = torch.cat(blobs) if len(blobs) > 1 else blobs[0]
        runner = self.runner
        n = self.det._priors_cuda(blob.device).shape[0]
        names = runner.output_names
        loc, conf, landms = [], [], []
        for start in range(0, blob.shape[0], self.max_batch):
            part = blob[start:start + self.max_batch].contiguous()
            k = part.shape[0]
            out = runner.run_binding({runner.input_names[0]: part}, output_shapes={
                names[0]: (k, n, 4), names[1]: (k, n, 2), names[2]: (k, n, 10)})
            loc.append(out[names[0]])
            conf.append(out[names[1]])
            landms.append(out[names[2]])
        loc, conf, landms = torch.cat(loc), torch.cat(conf), torch.cat(landms)
        _bp.count('gpudet.network_images', len(images))
        _bp.count('gpudet.network_calls', (len(images) + self.max_batch - 1) // self.max_batch)

        results = []
        for i, info in enumerate(infos):
            boxes, kps, scores = self.det.decode_cuda(loc[i:i + 1], conf[i:i + 1],
                                                      landms[i:i + 1], info)
            if squash:                                   # canvas -> frame, per axis
                fh, fw = info.frame_size
                ax = torch.tensor([fw / _INPUT_SIZE, fh / _INPUT_SIZE], device=boxes.device)
                boxes = (boxes.reshape(*boxes.shape[:-1], 2, 2) * ax).flatten(-2)
                kps = kps * ax
            sc = scores[0]
            keep = (sc >= thresh).nonzero(as_tuple=True)[0]          # the one data-dependent sync
            if keep.numel() == 0:
                if not audit:
                    results.append((np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)))
                    continue
                try:
                    from roop.procmgr_runtime import audit_detect_best_rejected
                    audit_detect_best_rejected(float(sc.max()))
                except Exception as exc:
                    _swallowed("roop/retinaface_gpu_engine.py:best_rejected", exc, "audit skipped")
                results.append((np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)))
                continue
            det = torch.cat([boxes[0][keep], sc[keep][:, None]], 1).cpu().numpy()
            kp = kps[0][keep].cpu().numpy()
            ok = np.isfinite(det).all(1) & np.isfinite(kp.reshape(len(kp), -1)).all(1)
            det, kp = det[ok], kp[ok]
            order = det[:, 4].argsort()[::-1]
            results.append((det[order].astype(np.float32, copy=False),
                            kp[order].astype(np.float32, copy=False)))
        return results

    # ── one detect() call ────────────────────────────────────────────────────
    def detect(self, frame, det_thresh, nms_thresh, scales=None, estimated_face_height=None):
        import torch
        import torch.nn.functional as F

        h, w = frame.shape[:2]
        arr = np.ascontiguousarray(frame[..., :3]) if frame.ndim == 3 else None
        if arr is None or arr.size == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        t = torch.from_numpy(arr).cuda(non_blocking=False)           # the one host->device copy
        chw = t.permute(2, 0, 1).unsqueeze(0)
        pad = context_pad(h, w)
        # reflect == OpenCV BORDER_REFLECT_101; it needs pad < the side, replicate otherwise
        if pad < min(h, w):
            padded = F.pad(chw, (pad, pad, pad, pad), mode='reflect')
        else:
            padded = F.pad(chw.float(), (pad, pad, pad, pad), mode='replicate')
        offs = (pad, pad, pad, pad)
        hp, wp = padded.shape[-2:]

        def finish(dets, kps):
            keep = nms_keep(dets, nms_thresh, offset=1.0) if len(dets) else []
            return dets[keep], kps[keep]

        def unpad(dets, kps):
            if len(dets) == 0:
                return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
            return remove_context_padding(dets, kps, offs)

        _bp.count('gpudet.calls')
        parsed = parse_scale_pyramid(getattr(roop.globals, 'detector_scale_pyramid', None)
                                     if scales is None else scales)
        single = None
        if parsed is None:
            if not should_trigger_pyramid((h, w), estimated_face_height=estimated_face_height):
                b0, k0 = finish(*self._infer([padded], det_thresh)[0])
                ub, uk = unpad(b0, k0)
                if not should_trigger_pyramid((h, w), initial_dets=ub):
                    _bp.count('gpudet.path.single_scale')
                    return ub, uk
                single = (b0, k0)                                    # reused below as the 1.0 level
                _bp.count('gpudet.trigger.initial_closeup')
            else:
                _bp.count('gpudet.trigger.estimated_face_height')
        elif parsed == [1.0]:
            _bp.count('gpudet.path.explicit_single_scale')
            return unpad(*finish(*self._infer([padded], det_thresh)[0]))

        # ── pyramid: every level is built from the ONE padded tensor ──────────
        active = parsed if parsed is not None else list(DEFAULT_PYRAMID_SCALES)
        levels = []                           # (scale (sx, sy), image or None when reused)
        for s in active:
            if abs(s - 1.0) < 1e-4:
                levels.append(((1.0, 1.0), padded))
                continue
            nw, nh = max(16, int(round(wp * s))), max(16, int(round(hp * s)))
            img = F.interpolate(padded.float(), size=(nh, nw),
                                mode='area' if s < 1.0 else 'bilinear',
                                **({} if s < 1.0 else {'align_corners': False}))
            levels.append(((nw / wp, nh / hp), img))
        _bp.count('gpudet.path.pyramid_executed')
        _bp.count('gpudet.pyramid_levels', len(levels))

        results = [None] * len(levels)
        todo = []
        for i, (sc, img) in enumerate(levels):
            if single is not None and img is padded:
                results[i] = single
                _bp.count('gpudet.single_pass_reused')
            else:
                todo.append(i)
        if todo:
            ran = self._infer([levels[i][1] for i in todo], det_thresh)
            for i, (d, k) in zip(todo, ran):
                results[i] = finish(d, k)

        boxes, kpss = [], []
        for (sc, _), (d, k) in zip(levels, results):
            if len(d) == 0:
                continue
            d, k = rescale_detections(d, k, scale_factor=sc)
            boxes.append(d)
            kpss.append(k)
        if not boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        kept, kept_k, _ = diou_nms(np.vstack(boxes), kpss=np.vstack(kpss),
                                   iou_thresh=nms_thresh, offset=1.0)
        return unpad(kept, kept_k)


def _ensure_pool():
    """Lazily build the detector pool: N independent execution contexts, N from
    ``session_pool.detector_pool_size`` (the same knob every hybrid engine answers to)."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("detector_engine 'retinaface_r50_gpu' needs a CUDA device")
        detector_cls = _import_face_engine()
        from face_engine.core.config import EngineConfig, Provider
        from face_engine.core.execution import ExecutionEngine
        from face_engine.core.trt_compiler import aot_available

        order = _provider_order()
        config = EngineConfig(providers=order, register_gpu_dlls=False,
                              device_id=int(getattr(roop.globals, 'cuda_device_id', 0) or 0))
        model_path, from_registry = _resolve_model()
        try:
            from roop import session_pool
            n = session_pool.detector_pool_size(
                model_key='detector:retinaface_r50_gpu',
                input_shape=(1, 3, _INPUT_SIZE, _INPUT_SIZE))
        except Exception as exc:
            _swallowed("roop/retinaface_gpu_engine.py:pool_size", exc, "pool size 1")
            n = 1
        n = max(1, int(n))
        engine = ExecutionEngine(config)
        items = [_Instance(detector_cls, engine, model_path) for _ in range(n)]
        for it in items:
            it.device = torch.device('cuda', config.device_id)
            _warm(it)
        trt = items[0].tensorrt
        wanted = Provider.TENSORRT in order
        provider = ('TensorRT AOT engine' if trt else
                    str(getattr(items[0].runner, 'primary_provider', 'ONNX Runtime')))
        _bp.log_runner('detector:retinaface_r50_gpu', file=model_path, provider=provider,
                       trt_fp16='on' if trt else 'n/a',
                       input_shape='input:Bx3x%dx%d (letterbox)' % (_INPUT_SIZE, _INPUT_SIZE),
                       requested=[p.value for p in order], instances=n)
        if wanted and not trt:
            why = ('model not in the face_engine registry (stamp mismatch)' if not from_registry
                   else 'no verified engine for this GPU/TensorRT/model'
                   if not aot_available(model_path, 'fp16', config) else 'engine failed to load')
            print("[RetinaFaceGPU] WARNING: no compiled TensorRT engine (%s); running on %s, "
                  "slower. Build with `python tools/compile_engines.py --models retinaface_r50`."
                  % (why, provider), flush=True)
        q = Queue()
        for it in items:
            q.put(it)
        _pool = {'items': items, 'q': q}
        if n > 1:
            print('[RetinaFaceGPU] pool of %d instances (%s) -- detection runs %d-way concurrent.'
                  % (n, provider, n))
    return _pool


def _warm(inst):
    """Pay first-use costs (context setup, the in-graph-softmax probe) on a dummy frame, not
    on frame 0 of a render."""
    import torch
    x = torch.zeros((1, 3, _INPUT_SIZE, _INPUT_SIZE), dtype=torch.uint8, device=inst.device)
    inst._infer([x], 1.1, audit=False)     # nothing clears 1.1; and it must not enter the miss audit


@contextmanager
def lease():
    pool = _ensure_pool()
    inst = pool['q'].get()
    try:
        yield inst
    finally:
        pool['q'].put(inst)


def detect(frame, det_size=_INPUT_SIZE, det_thresh=0.5, scales=None, estimated_face_height=None):
    """The ``retinaface.detect`` contract. ``det_size`` is accepted for signature parity and
    ignored: the network is calibrated at 640 only (face_engine enforces it)."""
    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    with lease() as inst:
        return inst.detect(frame, float(det_thresh), float(nms_thresh), scales=scales,
                           estimated_face_height=estimated_face_height)


def release_detector():
    global _pool
    with _pool_lock:
        _pool = None


def shrink_detector(width=1):
    """Drop detector instances beyond ``width``; returns how many were dropped.

    The swap phase of a replayed render only calls the detector for verification and
    rescue paths, so the pre-pass's width is idle memory by then
    (face_util.shrink_analysis_pools)."""
    from roop import session_pool
    with _pool_lock:
        return len(session_pool.shrink_lease_pool(_pool, width)) if _pool else 0
