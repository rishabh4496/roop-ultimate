import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool


# Segment Anything (FastSAM) as a face-mask engine. FastSAM is a YOLOv8-seg model
# that segments "everything" in one pass, then we pick the instance covering the
# centre of the aligned crop (the face/head). It is faster but COARSER than
# MobileSAM: it tends to return a whole-head region (hair included), closer in
# character to XSeg's blob than to MobileSAM's tight inner-face mask. Offered as a
# separate, fast engine; pick MobileSAM when you need precise hair/occluder edges.
#
# ONNX I/O (ultralytics export, imgsz 1024): input images[1,3,1024,1024] RGB/255
# NCHW; output0[1,37,21504] = 4 box (cxcywh, 1024 space) + 1 conf + 32 mask coeffs
# per anchor; output1[1,32,256,256] = mask prototypes. Mask = sigmoid(coeffs @
# protos), bounded to the detected box, kept to the centre component, then blurred.
_URL = 'https://github.com/rishabh4496/roop-sam-weights/releases/download/v1/fastsam_s.onnx'
_FILE = 'fastsam_s.onnx'
_SIZE = 1024
_CONF = 0.4          # anchor confidence threshold
_IOU = 0.6           # NMS IoU threshold


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Mask_FastSAM():
    plugin_options: dict = None

    model = None

    processorname = 'mask_fastsam'
    type = 'mask'

    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent sessions so the
        # mask runs concurrently across worker threads. None -> single shared
        # session serialised by the global lock (original safe default).
        self.pool = None

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_URL])
            model_path = os.path.join(model_dir, _FILE)
            onnxruntime.set_default_logger_severity(3)

            # Run on CUDA, not TensorRT: keep these per-crop SAM models off the TRT
            # EP (slow/fragile engine builds, shape-tensor issues) — see
            # session_pool.providers_without_tensorrt.
            providers = session_pool.providers_without_tensorrt(roop.globals.execution_providers)
            from roop.utilities import get_onnx_session_options
            _sess_opts = get_onnx_session_options()

            def _build(_i=0):
                return onnxruntime.InferenceSession(
                    model_path, _sess_opts, providers=providers)

            self.model = _build()
            self.input_name = self.model.get_inputs()[0].name

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size()
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model] + extras): _e[i], n)

    def Run(self, img1, keywords: str) -> Frame:
        # img1 is the aligned face crop (BGR uint8). Returned mask matches the
        # other engines' convention: 1.0 = keep ORIGINAL (exclude from swap),
        # 0.0 = use the swapped pixels.
        h0, w0 = img1.shape[:2]
        img = cv2.resize(img1, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)

        if self.pool is not None:
            with self.pool.lease() as sess:
                out0, out1 = sess.run(None, {self.input_name: blob})
        else:
            out0, out1 = self.model.run(None, {self.input_name: blob})

        face = self._decode(out0, out1)               # 1 inside face/head region
        face = cv2.resize(face, (w0, h0), interpolation=cv2.INTER_LINEAR)
        # Stronger blur than the other engines — softens FastSAM's coarse / box-
        # bounded edges into a gradient so the blend has no hard seam.
        face = cv2.GaussianBlur(face, (0, 0), sigmaX=5)
        face = np.clip(face, 0.0, 1.0)
        # invert: keep original everywhere outside the face region
        return (1.0 - face).astype(np.float32)

    def _decode(self, out0, out1):
        c = _SIZE // 2                                # crop centre in 1024 space
        preds = out0[0].T                             # (21504, 37)
        boxes, scores, coeffs = preds[:, :4], preds[:, 4], preds[:, 5:]
        keep = scores > _CONF
        boxes, scores, coeffs = boxes[keep], scores[keep], coeffs[keep]
        if len(boxes) == 0:
            return np.zeros((_SIZE, _SIZE), np.float32)

        xy = np.empty_like(boxes)                     # cxcywh -> xyxy
        xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        idxs = cv2.dnn.NMSBoxes(xy.tolist(), scores.tolist(), _CONF, _IOU)
        if idxs is None or len(idxs) == 0:
            return np.zeros((_SIZE, _SIZE), np.float32)
        idxs = np.array(idxs).flatten()

        # Prefer the LARGEST instance whose box contains the centre (the whole
        # head), ignoring a near-full-frame "background" instance.
        best, barea = None, -1.0
        for i in idxs:
            x0, y0, x1, y1 = xy[i]
            area = (x1 - x0) * (y1 - y0)
            if x0 <= c <= x1 and y0 <= c <= y1 and area < _SIZE * _SIZE * 0.97 and area > barea:
                best, barea = i, area
        if best is None:
            best = idxs[int(np.argmax(scores[idxs]))]

        proto = out1[0].reshape(out1.shape[1], -1)    # (32, 256*256)
        m = _sigmoid(coeffs[best] @ proto).reshape(out1.shape[2], out1.shape[3])
        m = cv2.resize(m, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
        mb = (m > 0.5).astype(np.uint8)

        # Bound to the detected box (drops loose prototype activations elsewhere)…
        x0, y0, x1, y1 = np.clip(xy[best], 0, _SIZE).astype(int)
        box = np.zeros_like(mb)
        box[y0:y1, x0:x1] = 1
        mb *= box
        # …then keep only the connected component under the centre.
        n_lbl, lab = cv2.connectedComponents(mb)
        cl = lab[c, c]
        if cl > 0:
            mb = (lab == cl).astype(np.uint8)
        return mb.astype(np.float32)

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model
        self.model = None
