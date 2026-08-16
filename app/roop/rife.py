"""RIFE 4.9 frame interpolation (ONNX) — motion-compensated in-between frames.

Used by the optional post-swap interpolation pass (api.py) to raise a video's
frame rate 2x/4x: for each consecutive frame pair the net synthesizes true
motion-interpolated frames at timestep t (verified: a square moving 20->60px
lands at ~40px for t=0.5, ~30px for t=0.25 — real motion, not a crossfade).

Model: rife49_ensemble_True_scale_1_sim.onnx (21 MB), inputs img0/img1
(1,3,H,W float RGB in [0,1]) + timestep (1,), dynamic H/W padded to a multiple
of 64. Runs on CUDA/CPU FP32 like the frame upscalers (TensorRT excluded — no
engine-build stall, and the fp16 overflow family of bugs can't apply).
"""

import os
import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.utilities import resolve_relative_path, conditional_download

MODEL_URL = "https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife49_ensemble_True_scale_1_sim.onnx"
MODEL_FILE = "rife49_ensemble_True_scale_1_sim.onnx"

_PAD = 64  # RIFE's pyramid needs dims divisible by this at scale=1


def _providers():
    provs = list(roop.globals.execution_providers or ["CPUExecutionProvider"])
    def _is_trt(p):
        name = p[0] if isinstance(p, (tuple, list)) else p
        return "tensorrt" in str(name).lower()
    filtered = [p for p in provs if not _is_trt(p)]
    return filtered or ["CUDAExecutionProvider", "CPUExecutionProvider"]


class RIFE:
    """One ONNX session; interpolate() is called sequentially by the pass."""

    def __init__(self):
        model_dir = resolve_relative_path("../models/Frame")
        conditional_download(model_dir, [MODEL_URL])
        model_path = os.path.join(model_dir, MODEL_FILE)
        self.session = onnxruntime.InferenceSession(model_path, None, providers=_providers())
        self._lock = threading.Lock()
        self._pair = None      # (id0-key, prepped img0) cache: reuse across timesteps

    @staticmethod
    def _prep(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        ph = (-h) % _PAD
        pw = (-w) % _PAD
        if ph or pw:
            rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        return rgb.transpose(2, 0, 1)[None]

    def interpolate(self, frame_a, frame_b, t: float):
        """Return the motion-interpolated BGR uint8 frame at fraction t of a→b."""
        h, w = frame_a.shape[:2]
        with self._lock:
            out = self.session.run(None, {
                "img0": self._prep(frame_a),
                "img1": self._prep(frame_b),
                "timestep": np.array([t], dtype=np.float32),
            })[0]
        img = out[0].transpose(1, 2, 0)[:h, :w]
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def release(self):
        self.session = None
