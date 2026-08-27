import os
import cv2
import numpy as np
import threading
from contextlib import contextmanager
from queue import Queue

import roop.globals
from roop.utilities import resolve_relative_path, conditional_download
from roop.nms import nms_keep, CENTER_FRAC

# What OpenCV's internal NMS is set to when the shared rule is doing the real
# suppression: high enough that it cannot pre-delete a pair the rule would have
# kept (the rule tops out around IoU 0.60), low enough that exact-duplicate
# boxes still collapse inside OpenCV rather than being carried out and back.
_RAW_NMS = 0.85

_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

_pool = None                        # {'items': [...], 'q': Queue}
_detector_lock = threading.Lock()   # guards pool CONSTRUCTION only


def _build_one(model_path):
    """One independent FaceDetectorYN. Size (320,320) is a placeholder — every
    detect() call sets the real input size for the frame it is given."""
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.6, 0.3, 5000)


def _ensure_pool():
    """Lazily build the detector pool, downloading weights if necessary.

    This module used to hold ONE detector and wrap every detect in a global
    mutex, because setInputSize/setScoreThreshold/setNMSThreshold mutate the
    detector and a concurrent call would corrupt it. That made yunet single-file
    no matter how wide ROOP_DETMASK_POOL was set — the same defect measured on
    retinaface, where the serial section stayed a constant 17.4ms per call at
    every pool width with the GPU at ~51%.

    Giving each worker its own instance is what the mutex was really protecting:
    per-instance settings owned by exactly one thread for the duration of a call.
    yunet's weights are ~350KB, so N instances are essentially free.
    """
    global _pool
    if _pool is not None:
        return _pool
    with _detector_lock:
        if _pool is not None:
            return _pool
        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [_MODEL_URL])
        model_path = os.path.join(model_dir, _MODEL_FILE)
        try:
            from roop import session_pool
            n = session_pool.detector_pool_size(
                model_key='detector:yunet', input_shape=(1, 3, 320, 320))
        except Exception:
            n = 1
        items = [_build_one(model_path) for _ in range(n)]
        q = Queue()
        for det in items:
            q.put(det)
        _pool = {'items': items, 'q': q}
        if n > 1:
            print(f'[YuNet] pool of {n} instances — detection runs '
                  f'{n}-way concurrent (lock-free).')
    return _pool


@contextmanager
def lease_detector():
    """Lease one detector for a single detect call. The queue blocks once all N
    are out, so concurrency is capped at the pool size and each instance's
    mutable settings belong to one thread for the length of the call."""
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
    original frame coordinates."""
    h, w = frame.shape[:2]

    # Scale frame dynamically so the longest side matches det_size. Done OUTSIDE
    # the lease: it is pure CPU on a private array and holding an instance
    # through it would shrink the pool's effective width for no reason.
    scale = float(det_size) / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    # YuNet suppresses inside OpenCV, where the rule cannot be replaced — so it
    # is asked not to decide. With the shared face-vs-duplicate rule active,
    # OpenCV runs permissively and the real suppression happens below, on boxes
    # this module owns; otherwise it keeps deciding exactly as before. Without
    # this, yunet would be the one engine still deleting a touching face. See
    # roop/nms.py.
    raw_nms = max(nms_thresh, _RAW_NMS) if CENTER_FRAC > 0 else nms_thresh
    with lease_detector() as det:
        det.setInputSize((new_w, new_h))
        det.setScoreThreshold(det_thresh)
        det.setNMSThreshold(raw_nms)
        _, faces = det.detect(resized)


    if faces is None or len(faces) == 0:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)
        
    bboxes = []
    kpss = []
    for face in faces:
        # face shape: bbox [x, y, w, h] (0:4), landmarks [5, 2] (4:14), score (14)
        x1, y1, width, height = face[0:4]
        score = face[14]
        
        # Scale back to original coordinates
        ox1 = x1 / scale
        oy1 = y1 / scale
        ox2 = (x1 + width) / scale
        oy2 = (y1 + height) / scale
        
        bboxes.append([ox1, oy1, ox2, oy2, score])
        
        # Convert landmarks to shape (5, 2)
        lm = face[4:14].reshape((5, 2)) / scale
        kpss.append(lm)

    bboxes = np.array(bboxes, dtype=np.float32)
    kpss = np.array(kpss, dtype=np.float32)

    if CENTER_FRAC > 0 and len(bboxes) > 1:
        # The suppression OpenCV was told to skip, at the configured threshold —
        # same greedy-by-score algorithm, plus the concentricity requirement, so
        # with the rule disabled this reduces to what OpenCV was doing.
        keep = nms_keep(bboxes, nms_thresh, offset=0.0)
        bboxes, kpss = bboxes[keep], kpss[keep]

    return bboxes, kpss


def release_detector():
    global _pool
    with _detector_lock:
        _pool = None
