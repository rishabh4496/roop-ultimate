"""YOLOFace (yoloface_8n) detector — an alternate face detector engine.

insightface's default SCRFD detector is fast and accurate on frontal faces but
weaker on steep profiles / partially-occluded faces. YOLOFace tends to catch
those better. It only produces bbox + 5 keypoints (not identity/landmarks), so
face_util pairs it with the existing buffalo_l aux models (recognition + 106/68
landmarks) to build full Face objects — see face_util._hybrid_yolo_faces.

Output decoding mirrors FaceFusion's yoloface path: the model emits (1, 20, N)
= 4 bbox (cx,cy,w,h) + 1 score + 15 (5 kps × x,y,score) in a 640-canvas the
downscaled frame is pasted into top-left of; canvas coords scale back to the
original by the resize ratio.
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
from roop.nms import nms_keep

_YOLO_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.onnx"

_pool = None                        # {'items': [...], 'q': Queue}
_detector_lock = threading.Lock()   # guards pool CONSTRUCTION only


def _nms(boxes, scores, iou_thresh=0.4):
    """IoU non-max suppression. boxes: (N,4) x1y1x2y2.

    Delegates to the shared rule so this engine agrees with the others about
    when two overlapping boxes are two touching faces rather than one face
    detected twice — see roop/nms.py. offset=0 keeps this engine's own area
    convention (no +1), so the IoU numbers are unchanged.
    """
    if len(boxes) == 0:
        return []
    dets = np.concatenate([np.asarray(boxes, np.float32),
                           np.asarray(scores, np.float32).reshape(-1, 1)], axis=1)
    return nms_keep(dets, iou_thresh, offset=0.0)


class YoloFaceDetector:
    def __init__(self, providers):
        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [_YOLO_URL])
        model_path = os.path.join(model_dir, "yoloface_8n.onnx")
        so = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # No lock: each instance is leased to one thread for the length of a
        # call (see lease_detector). The old shared-session mutex assumed
        # detection was the cheap stage, which stopped being true for the
        # temporal pre-pass — there detection is ~85% of the work, and the
        # measured cost of the same pattern on retinaface was a constant 17.4ms
        # serial section at every pool width, with the GPU at ~51%.

    def detect(self, frame, det_size=640, det_thresh=0.5):
        h0, w0 = frame.shape[:2]
        scale = min(det_size / w0, det_size / h0)
        rw, rh = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((det_size, det_size, 3), dtype=frame.dtype)
        canvas[:rh, :rw, :] = resized
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]

        out = self.session.run(None, {self.input_name: blob})[0]

        det = np.squeeze(out).T                     # (N, 20)
        if det.ndim != 2 or det.shape[1] < 20:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        bbox_raw = det[:, 0:4]
        score_raw = det[:, 4]
        kps_raw = det[:, 5:20]
        keep = score_raw > det_thresh
        bbox_raw, score_raw, kps_raw = bbox_raw[keep], score_raw[keep], kps_raw[keep]
        if bbox_raw.shape[0] == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)

        # cx,cy,w,h (canvas coords) → x1,y1,x2,y2 (original coords)
        inv = 1.0 / scale
        cx, cy, w, h = bbox_raw[:, 0], bbox_raw[:, 1], bbox_raw[:, 2], bbox_raw[:, 3]
        x1 = (cx - w / 2) * inv
        y1 = (cy - h / 2) * inv
        x2 = (cx + w / 2) * inv
        y2 = (cy + h / 2) * inv
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        kps = kps_raw.reshape(-1, 5, 3)[:, :, :2] * inv

        nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
        idx = _nms(boxes, score_raw, iou_thresh=nms_thresh)
        boxes, score_raw, kps = boxes[idx], score_raw[idx], kps[idx]

        # Clamp to frame bounds.
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w0 - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h0 - 1)

        bboxes = np.concatenate([boxes, score_raw[:, None]], axis=1).astype(np.float32)
        return bboxes, kps.astype(np.float32)


def _ensure_pool():
    """Lazily build the detector pool, honouring force_cpu and the current
    execution providers. yoloface_8n.onnx is ~9MB, so N instances are cheap."""
    global _pool
    if _pool is not None:
        return _pool
    with _detector_lock:
        if _pool is not None:
            return _pool
        if roop.globals.CFG is not None and roop.globals.CFG.force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = roop.globals.execution_providers
        try:
            from roop import session_pool
            n = session_pool.detector_pool_size()
        except Exception:
            n = 1
        items = [YoloFaceDetector(providers) for _ in range(n)]
        q = Queue()
        for det in items:
            q.put(det)
        _pool = {'items': items, 'q': q}
        if n > 1:
            print(f'[YOLOFace] pool of {n} instances — detection runs '
                  f'{n}-way concurrent (lock-free).')
    return _pool


@contextmanager
def lease_detector():
    """Lease one detector for a single detect call. The queue blocks once all N
    are out, capping concurrency at the pool size — which is what makes each
    session's exclusive ownership true without a mutex."""
    pool = _ensure_pool()
    det = pool['q'].get()
    try:
        yield det
    finally:
        pool['q'].put(det)


def get_detector():
    """A detector instance, NOT leased — for callers that only read static
    attributes. Concurrent detect calls must go through detect()."""
    return _ensure_pool()['items'][0]


def detect(frame, det_size=640, det_thresh=0.5):
    """Run detection; returns (bboxes (N,5) incl. score, kpss (N,5,2)) in
    original frame coordinates — the same contract as retinaface.detect and
    yunet.detect. Module-level so all three hybrid engines are called alike."""
    with lease_detector() as det:
        return det.detect(frame, det_size=det_size, det_thresh=det_thresh)


def release_detector():
    global _pool
    with _detector_lock:
        _pool = None
