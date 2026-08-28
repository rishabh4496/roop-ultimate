import os
import numpy as np
import cv2
import onnxruntime
import threading
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool
from roop.precision_policy import providers_for

THREAD_LOCK_OCCLUDER = threading.Lock()

# FaceFusion 2.x face_occluder.onnx. The original facefusion-assets release was
# taken down, so the weight is mirrored on the project's own release (same host
# as the MobileSAM/FastSAM weights).
_MODEL_URL = 'https://github.com/rishabh4496/roop-sam-weights/releases/download/v1/face_occluder.onnx'


class Mask_Occluder():
    """Dedicated face-occluder mask engine (FaceFusion `face_occluder.onnx`).

    Unlike generic XSeg (which segments the visible face skin) this model is
    trained specifically to find occluding objects in front of the face — hands,
    hair strands, microphones, etc. — and the hidden side of strong profiles. The
    raw model outputs HIGH on the *visible face* (the region to keep swapped), so
    we invert it to match this project's mask convention, where a high mask value
    means "restore the ORIGINAL pixels here" (see ProcessMgr.process_mask:
    result = (1 - mask) * swap + mask * original).

    The convention can be flipped at runtime with ROOP_OCCLUDER_RAW=1 in case a
    future model variant ships the opposite polarity — handy for visual A/B.
    """

    plugin_options: dict = None

    model_occluder = None

    processorname = 'mask_occluder'
    type = 'mask'

    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent sessions so the
        # mask runs concurrently across worker threads; None → single shared
        # session serialised by the global lock (original safe default).
        self.pool = None

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_occluder is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, 'face_occluder.onnx')
            from roop.utilities import get_onnx_session_options
            _sess_opts = get_onnx_session_options()
            providers, _precision = providers_for(
                'masking:occluder', roop.globals.execution_providers, model_path)

            def _build(_i=0):
                return onnxruntime.InferenceSession(model_path, _sess_opts, providers=providers)

            self.model_occluder = _build()
            self.model_inputs = self.model_occluder.get_inputs()
            self.model_outputs = self.model_occluder.get_outputs()

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Optional multi-session pool: primary + (N-1) extras → up to N threads
            # run the mask concurrently, each on its own TensorRT context.
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size(
                    model_key='mask:occluder', input_shape=(1, 3, 512, 512))
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_occluder] + extras): _e[i], n,
                    model_key='mask:occluder', input_shape=(1, 3, 512, 512))

    def _run_session(self, sess, temp_frame):
        io_binding = sess.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self.model_outputs[0].name, self.devicename)
        sess.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()

    def Run(self, img1, keywords: str) -> Frame:
        # Model input: (1, 256, 256, 3) NHWC, float32 in [0, 1].
        temp_frame = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = temp_frame[None, ...]
        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = self._run_session(sess, temp_frame)
        else:
            ort_outs = self._run_session(self.model_occluder, temp_frame)
        # Output: (1, 256, 256, 1) → drop batch + channel dims to a 2D mask.
        result = ort_outs[0][0]
        if result.ndim == 3:
            result = result[..., 0]
        result = np.clip(result, 0, 1.0)
        # Raw model is HIGH on the visible face. Invert so HIGH = occluder/hidden
        # region → restore original there. ROOP_OCCLUDER_RAW=1 skips the invert.
        if os.environ.get('ROOP_OCCLUDER_RAW', '0') != '1':
            result = 1.0 - result
        return result

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_occluder
        self.model_occluder = None
