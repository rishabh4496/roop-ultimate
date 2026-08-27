import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool


# Segment Anything (MobileSAM) as a face-mask engine. MobileSAM = TinyViT image
# encoder + the original SAM prompt-encoder/mask-decoder. We run it on the aligned
# face crop with a fixed set of prompt points (the crop is canonically aligned, so
# the face sits at known fractional positions): foreground points on the inner
# face, background points at the corners. SAM returns the contiguous face object —
# so hair, background, and anything OCCLUDING the face (a hand in front) fall
# outside the mask, which is exactly what we want to keep from the original.
#
# Critical convention (samexporter ONNX): the encoder PADS sub-1024 inputs rather
# than upscaling them, so we pre-resize the crop to 1024x1024, give prompt points
# in 1024 space, set orig_im_size=(1024,1024), then resize the mask back. Feeding
# the native crop size produces a wrong, right-shifted half-mask.
_ENC_URL = 'https://github.com/rishabh4496/roop-sam-weights/releases/download/v1/mobilesam_encoder.onnx'
_DEC_URL = 'https://github.com/rishabh4496/roop-sam-weights/releases/download/v1/mobilesam_decoder.onnx'
_ENC_FILE = 'mobilesam_encoder.onnx'
_DEC_FILE = 'mobilesam_decoder.onnx'
_SAM_SIZE = 1024

# Foreground points on the inner face + background points at the corners, as
# fractions of the (canonically aligned) crop. Tuned on real aligned crops to
# yield a clean inner-face mask (hair/background excluded), IoU ~0.97.
_FG = np.array([[0.50, 0.55], [0.40, 0.46], [0.60, 0.46], [0.50, 0.40],
                [0.50, 0.70], [0.43, 0.66], [0.57, 0.66]], dtype=np.float32)
_BG = np.array([[0.02, 0.02], [0.98, 0.02], [0.02, 0.98], [0.98, 0.98]], dtype=np.float32)


class Mask_MobileSAM():
    plugin_options: dict = None

    encoder = None
    decoder = None

    processorname = 'mask_mobilesam'
    type = 'mask'

    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent sessions so the
        # mask runs concurrently across worker threads. None -> single shared
        # session serialised by the global lock (original safe default). Only the
        # heavy encoder is pooled; the decoder is cheap and stays shared.
        self.pool = None
        # Prebuilt decoder prompt tensors (point coords filled per call).
        self._labels = np.concatenate(
            [np.ones(len(_FG), np.float32), np.zeros(len(_BG), np.float32)])[None, :]
        self._coords_frac = np.concatenate([_FG, _BG], axis=0)   # (N,2) fractions

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.encoder is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_ENC_URL, _DEC_URL])
            enc_path = os.path.join(model_dir, _ENC_FILE)
            dec_path = os.path.join(model_dir, _DEC_FILE)
            onnxruntime.set_default_logger_severity(3)

            # Run on CUDA, not TensorRT: the decoder's float `orig_im_size` is a
            # shape tensor TRT rejects (Error Code 4, must be Int32/Int64).
            providers = session_pool.providers_without_tensorrt(roop.globals.execution_providers)
            from roop.utilities import get_onnx_session_options
            _sess_opts = get_onnx_session_options()

            def _build_enc(_i=0):
                return onnxruntime.InferenceSession(
                    enc_path, _sess_opts, providers=providers)

            self.encoder = _build_enc()
            self.decoder = onnxruntime.InferenceSession(
                dec_path, _sess_opts, providers=providers)
            self.enc_input = self.encoder.get_inputs()[0].name

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Optional multi-session pool over the encoder (the costly stage).
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size(
                    model_key='mask:mobilesam', input_shape=(1, 3, 1024, 1024))
                extras = [_build_enc(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.encoder] + extras): _e[i], n,
                    model_key='mask:mobilesam', input_shape=(1, 3, 1024, 1024))

    def Run(self, img1, keywords: str) -> Frame:
        # img1 is the aligned face crop (BGR uint8). Returned mask matches the
        # other engines' convention: 1.0 = keep ORIGINAL (exclude from swap),
        # 0.0 = use the swapped pixels. process_mask resizes it to the crop.
        h0, w0 = img1.shape[:2]
        crop = cv2.resize(img1, (_SAM_SIZE, _SAM_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0-255, encoder normalizes

        if self.pool is not None:
            with self.pool.lease() as sess:
                emb = sess.run(None, {self.enc_input: rgb})[0]
        else:
            emb = self.encoder.run(None, {self.enc_input: rgb})[0]

        coords = (self._coords_frac * _SAM_SIZE)[None].astype(np.float32)  # 1024 space
        masks = self.decoder.run(None, {
            'image_embeddings': emb,
            'point_coords': coords,
            'point_labels': self._labels,
            'mask_input': np.zeros((1, 1, 256, 256), np.float32),
            'has_mask_input': np.zeros(1, np.float32),
            'orig_im_size': np.array([_SAM_SIZE, _SAM_SIZE], np.float32),
        })[0]

        face = (masks[0, 0] > 0).astype(np.float32)          # 1 inside face object
        face = cv2.resize(face, (w0, h0), interpolation=cv2.INTER_LINEAR)
        # Soften edges so the blend matches the smooth XSeg/FaceParser output.
        face = cv2.GaussianBlur(face, (0, 0), sigmaX=3)
        face = np.clip(face, 0.0, 1.0)
        # invert: keep original everywhere outside the face region
        return (1.0 - face).astype(np.float32)

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.encoder
        self.encoder = None
        del self.decoder
        self.decoder = None
