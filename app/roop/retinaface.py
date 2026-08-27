"""RetinaFace (retinaface_10g / retinaface_resnet50) — alternate face detector engines.

Same hybrid pattern as YOLOFace (roop/yoloface.py): this model only detects
(bbox + 5 keypoints); face_util pairs it with buffalo_l's aux models for
identity + landmarks, so the rest of the pipeline sees the exact same Face
objects. RetinaFace has higher recall than SCRFD on hard poses/lighting,
which reduces detection dropouts (swap blink) at the source.

Two distinct model families are supported, each with its own output/anchor
convention:
  - '10g': FaceFusion's retinaface_10g.onnx. Same anchor-free, distance-
    regression head as insightface's SCRFD, so it's routed straight through
    insightface's own ModelRouter/RetinaFace class, which already decodes it
    correctly.
  - 'r50': a biubug6/Pytorch_Retinaface ResNet50 export. This is a classic
    *anchor-based* detector (predefined box sizes per stride, decoded via
    prior boxes + variance), a completely different convention from
    '10g'/SCRFD, so it needs its own PriorBox + decode implementation
    (RetinaFace3Output below) and its own preprocessing (mean-subtract only,
    BGR, no std scaling — not the 127.5/128 SCRFD convention).
"""

import os
import threading
from contextlib import contextmanager
from queue import Queue
import cv2
import numpy as np
import onnxruntime
import roop.globals
from roop.utilities import resolve_relative_path, conditional_download
from roop.nms import nms_keep, bind_instance_nms

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


# biubug6/Pytorch_Retinaface PriorBox config (ResNet50 variant).
_R50_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_R50_STEPS = (8, 16, 32)
_R50_VARIANCE = (0.1, 0.2)
_R50_DEFAULT_INPUT_SIZE = (640, 640)

_prior_cache = {}


def _generate_priors(image_size, min_sizes_cfg=_R50_MIN_SIZES, steps=_R50_STEPS):
    """Anchor-based PriorBox generation, matching biubug6/Pytorch_Retinaface's
    box_utils.PriorBox exactly (grid + anchor order must match how the network
    was trained). Result is normalized (cx, cy, w, h) in [0, 1]; cached per
    image_size since it's static for a fixed input resolution."""
    cached = _prior_cache.get(image_size)
    if cached is not None:
        return cached

    h, w = image_size
    anchors = []
    for k, step in enumerate(steps):
        fh = int(np.ceil(h / step))
        fw = int(np.ceil(w / step))
        min_sizes = min_sizes_cfg[k]
        num_anchors = len(min_sizes)

        cy = (np.arange(fh, dtype=np.float32) + 0.5) * step / h
        cx = (np.arange(fw, dtype=np.float32) + 0.5) * step / w
        grid_cy, grid_cx = np.meshgrid(cy, cx, indexing='ij')
        grid_cy = grid_cy.reshape(-1)
        grid_cx = grid_cx.reshape(-1)

        sizes = np.array(min_sizes, dtype=np.float32)
        cx_rep = np.repeat(grid_cx, num_anchors)
        cy_rep = np.repeat(grid_cy, num_anchors)
        skx = np.tile(sizes / w, grid_cx.shape[0])
        sky = np.tile(sizes / h, grid_cy.shape[0])
        anchors.append(np.stack([cx_rep, cy_rep, skx, sky], axis=1))

    priors = np.concatenate(anchors, axis=0).astype(np.float32)
    _prior_cache[image_size] = priors
    return priors


def _decode_boxes(loc, priors, variances=_R50_VARIANCE):
    """box_utils.decode from biubug6/Pytorch_Retinaface. Returns normalized
    (x1, y1, x2, y2) in [0, 1]."""
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landmarks(pre, priors, variances=_R50_VARIANCE):
    """box_utils.decode_landm counterpart — 5 (x, y) points, normalized."""
    return np.concatenate([
        priors[:, :2] + pre[:, 2 * i:2 * i + 2] * variances[0] * priors[:, 2:]
        for i in range(5)
    ], axis=1)


class RetinaFace3Output:
    """biubug6/Pytorch_Retinaface ResNet50 decoder — anchor-based, NOT the
    insightface/SCRFD anchor-free convention. See module docstring."""

    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        # biubug6 preprocessing: BGR, mean-subtract only, no std scaling.
        self.input_mean = (104.0, 117.0, 123.0)

        if self.session is None:
            from roop.utilities import get_onnx_session_options
            self.session = onnxruntime.InferenceSession(self.model_file, get_onnx_session_options(), providers=['CPUExecutionProvider'])

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str) or input_shape[2] is None:
            # The r50 export uses dynamic h/w symbols, but its priors and
            # TensorRT optimization profile are trained/built at 640x640.
            # Leaving this as None made the mixed TensorRT path fail before
            # inference (model_ratio could not be computed), and face_util
            # swallowed that exception as a detector miss.
            self.input_size = _R50_DEFAULT_INPUT_SIZE
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        self.input_name = input_cfg.name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        if 'nms_thresh' in kwargs:
            self.nms_thresh = kwargs['nms_thresh']
        if 'det_thresh' in kwargs:
            self.det_thresh = kwargs['det_thresh']
        if 'input_size' in kwargs:
            self.input_size = kwargs['input_size']

    def detect(self, img, input_size=None, max_num=0, metric='default'):
        input_size = self.input_size if input_size is None else input_size
        if input_size is None:
            # Defensive fallback for callers constructing the object through a
            # legacy path that did not populate session input metadata.
            input_size = _R50_DEFAULT_INPUT_SIZE
        # This ResNet50 export is trained on a square, directly-resized image.
        # Letterboxing it (the correct SCRFD/10g geometry) suppresses r50 face
        # scores under TensorRT, especially on 16:9 footage. Keep independent
        # axis scales so decoded boxes map back to the source correctly.
        direct_square = self.model_file and os.path.basename(self.model_file).lower() in {
            'retinaface_r50.onnx', 'retinaface-resnet50.onnx',
            'retinaface_resnet50.onnx'}
        if direct_square:
            det_img = cv2.resize(img, (input_size[0], input_size[1]))
            scale_x = float(input_size[0]) / img.shape[1]
            scale_y = float(input_size[1]) / img.shape[0]
            det_scale = None
        else:
            im_ratio = float(img.shape[0]) / img.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio > model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)
            det_scale = float(new_height) / img.shape[0]
            scale_x = scale_y = det_scale
            resized_img = cv2.resize(img, (new_width, new_height))
            det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
            det_img[:new_height, :new_width, :] = resized_img

        blob = cv2.dnn.blobFromImage(det_img, 1.0, input_size, self.input_mean, swapRB=False)
        net_outs = self.session.run(self.output_names, {self.input_name : blob})

        loc = net_outs[0][0]
        conf = net_outs[1][0]
        landms = net_outs[2][0]

        # The nakamura196 r50 export already applies softmax inside the graph
        # and uses the standard [background, face] class order. Do NOT
        # re-softmax probabilities; only use softmax for a future export whose
        # rows are not already normalized.
        if abs(float(conf[0].sum()) - 1.0) < 1e-3:
            scores = conf[:, 1]
        else:
            scores = softmax(conf)[:, 1]
        priors = _generate_priors((input_size[1], input_size[0]))

        # Threshold BEFORE decoding. Decoding is elementwise per anchor — row i of
        # the output depends only on row i of loc/landms/priors — so decoding the
        # 16,800 anchors and then keeping the 1-3 above threshold produced exactly
        # the same numbers as decoding only those rows, at ~90x the cost: measured
        # 2.838 ms/call vs 0.031 ms, against a ~43 ms detect. That is CPU time
        # spent inside the detect call with the GPU idle, and the pre-pass runs
        # this once per frame per worker, so it is worth the reorder.
        pos_inds = np.where(scores >= self.det_thresh)[0]
        if pos_inds.size == 0 and scores.size:
            # Nothing cleared the gate — record how close the best anchor came,
            # so the audit can say whether the gate is what lost this frame.
            # Only reached on a frame that was already a miss, so it is free.
            try:
                from roop.procmgr_runtime import audit_detect_best_rejected
                audit_detect_best_rejected(scores.max())
            except Exception:
                pass
        pos_scores = scores[pos_inds]
        pos_priors = priors[pos_inds]

        pos_boxes = _decode_boxes(loc[pos_inds], pos_priors)
        pos_boxes[:, 0::2] *= input_size[0]
        pos_boxes[:, 1::2] *= input_size[1]
        if det_scale is None:
            pos_boxes[:, 0::2] /= scale_x
            pos_boxes[:, 1::2] /= scale_y
        else:
            pos_boxes /= det_scale

        pos_kpss = _decode_landmarks(landms[pos_inds], pos_priors)
        pos_kpss[:, 0::2] *= input_size[0]
        pos_kpss[:, 1::2] *= input_size[1]
        if det_scale is None:
            pos_kpss[:, 0::2] /= scale_x
            pos_kpss[:, 1::2] /= scale_y
        else:
            pos_kpss /= det_scale
        # (-1, 5, 2), not (n, -1, 2): numpy cannot infer -1 from a zero-sized
        # array, and a frame with no face above threshold reaches here with n=0.
        pos_kpss = pos_kpss.reshape((-1, 5, 2))

        order = pos_scores.argsort()[::-1]
        pos_scores = pos_scores[order]
        pos_boxes = pos_boxes[order]
        pos_kpss = pos_kpss[order]

        pre_det = np.hstack((pos_boxes, pos_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.nms(pre_det)

        det = pre_det[keep, :]
        kpss = pos_kpss[keep, :, :]

        return det, kpss

    def nms(self, dets):
        # Shared with SCRFD, yoloface and yunet so one face-vs-duplicate rule
        # governs every engine — see roop/nms.py. offset=1 keeps this lineage's
        # (x2 - x1 + 1) area convention, so the numbers are unchanged.
        return nms_keep(dets, self.nms_thresh, offset=1.0)


_MODEL_10G_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/retinaface_10g.onnx"
_MODEL_R50_URL = "https://huggingface.co/nakamura196/retinaface-r50-onnx/resolve/main/retinaface_r50.onnx"

# model_type -> {'items': [detector, ...], 'q': Queue}
_POOLS = {}
_detector_lock = threading.Lock()   # guards pool CONSTRUCTION only


def _pool_size():
    """One detector instance per concurrent detect worker.

    The rule now lives in session_pool.detector_pool_size() because yunet and
    yoloface need the identical thing — three copies of it would drift, and the
    three engines have to answer to the same ROOP_DETECTOR_POOL.
    """
    try:
        from roop import session_pool
        return session_pool.detector_pool_size()
    except Exception:
        return 1


def _build_one(model_type, model_path, providers, file):
    """Construct ONE independent detector (its own ORT session)."""
    if model_type == 'r50':
        from roop.utilities import get_onnx_session_options
        session = onnxruntime.InferenceSession(model_path, get_onnx_session_options(), providers=providers)
        det = RetinaFace3Output(model_path, session=session)
    else:
        from insightface.model_zoo import get_model
        det = get_model(model_path, providers=providers)
        if det is None or not hasattr(det, 'detect'):
            raise RuntimeError(f'insightface could not route {file} to a detector')
        # 10g decodes inside insightface, whose nms() would otherwise delete one
        # of two touching faces on a rule the other engines no longer use. Same
        # signature and contract; bound per instance, so site-packages is
        # untouched. See roop/nms.py.
        bind_instance_nms(det)
    return det


def _ensure_pool(model_type):
    """Lazily build the detector pool, honouring force_cpu and the current
    execution providers.

    Previously this module held ONE shared session and serialised every call
    behind a mutex, on the reasoning that detection is cheap next to the swap.
    That stopped being true for the temporal pre-pass, where detection is ~85%
    of the work: N detmask workers all queued on that one lock, so widening
    ROOP_DETMASK_POOL only added queue time (measured: 68.7ms/call at pool 4 and
    104.1ms/call at pool 6 — both exactly 57-58 f/s, with the GPU at ~51%).
    Each thread now leases its own instance, so nothing serialises.
    """
    pool = _POOLS.get(model_type)
    if pool is not None:
        return pool
    with _detector_lock:
        pool = _POOLS.get(model_type)
        if pool is not None:
            return pool

        if model_type == 'r50':
            url, file = _MODEL_R50_URL, "retinaface_r50.onnx"
        else:
            url, file = _MODEL_10G_URL, "retinaface_10g.onnx"

        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [url])
        model_path = os.path.join(model_dir, file)

        if roop.globals.CFG is not None and roop.globals.CFG.force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = roop.globals.execution_providers

        n = _pool_size()
        items = [_build_one(model_type, model_path, providers, file) for _ in range(n)]
        q = Queue()
        for det in items:
            q.put(det)
        pool = {'items': items, 'q': q}
        _POOLS[model_type] = pool
        if n > 1:
            print(f'[RetinaFace] pool of {n} {model_type} instances — '
                  f'detection runs {n}-way concurrent (lock-free).')
    return pool


@contextmanager
def lease_detector(model_type='10g'):
    """Lease one detector instance for a single detect call. The queue blocks
    once all N are out, capping concurrency at the pool size — so per-instance
    mutable state (det_thresh / nms_thresh / input_size) is owned by exactly
    one thread for the duration of the call, which is what the old global lock
    was really protecting."""
    pool = _ensure_pool(model_type)
    det = pool['q'].get()
    try:
        yield det
    finally:
        pool['q'].put(det)


def get_detector(model_type='10g'):
    """A detector instance, NOT leased — for callers that only read static
    attributes. Concurrent detect calls must go through detect()/lease_detector()."""
    return _ensure_pool(model_type)['items'][0]


def detect(frame, det_size=640, det_thresh=0.5, model_type='10g'):
    """Run detection; returns (bboxes (N,5) incl. score, kpss (N,5,2)) in
    original frame coordinates — the same contract as yoloface.detect."""
    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    with lease_detector(model_type) as det:
        det.det_thresh = det_thresh
        if hasattr(det, 'nms_thresh'):
            det.nms_thresh = nms_thresh
        return det.detect(frame, input_size=(int(det_size), int(det_size)), max_num=0)


def release_detector():
    with _detector_lock:
        _POOLS.clear()
