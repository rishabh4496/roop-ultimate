import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool
from roop.precision_policy import providers_for


# FaceFusion's third-generation XSeg occluder (added in FF 3.2). Same family and
# I/O contract as the classic face_occluder / xseg models — NHWC (1,256,256,3)
# float [0,1] in, (1,256,256,1) [0,1] out, HIGH on the *visible face* — but a
# newer training run with better handling of hands/hair/objects crossing the
# face. Offered alongside "Face Occluder" and "DFL XSeg" for A/B; conventions
# are inverted to this project's mask polarity (HIGH = restore ORIGINAL pixels)
# exactly like the other engines. ROOP_XSEG3_RAW=1 skips the inversion in case
# a future variant flips polarity.
_MODEL_URL = 'https://huggingface.co/facefusion/models-3.2.0/resolve/main/xseg_3.onnx'
_MODEL_FILE = 'xseg_3.onnx'


class Mask_XSeg3():
    plugin_options: dict = None

    model_xseg3 = None

    processorname = 'mask_xseg3'
    type = 'mask'

    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent TensorRT sessions
        # so the mask runs concurrently across worker threads. None → single shared
        # session serialised by the global lock (original safe default).
        self.pool = None

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_xseg3 is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, _MODEL_FILE)
            from roop.utilities import get_onnx_session_options
            _sess_opts = get_onnx_session_options()
            providers, _precision = providers_for(
                'masking:xseg3', roop.globals.execution_providers, model_path)

            def _build(_i=0):
                return onnxruntime.InferenceSession(model_path, _sess_opts, providers=providers)

            self.model_xseg3 = _build()
            self.model_inputs = self.model_xseg3.get_inputs()
            self.model_outputs = self.model_xseg3.get_outputs()

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Optional multi-session pool: primary + (N-1) extras → up to N threads
            # run the mask concurrently, each on its own TensorRT context.
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size(
                    model_key='mask:xseg3', input_shape=(1, 3, 512, 512))
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_xseg3] + extras): _e[i], n,
                    model_key='mask:xseg3', input_shape=(1, 3, 512, 512))

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
            ort_outs = self._run_session(self.model_xseg3, temp_frame)
        # Output: (1, 256, 256, 1) → drop batch + channel dims to a 2D mask.
        result = ort_outs[0][0]
        if result.ndim == 3:
            result = result[..., 0]
        result = np.clip(result, 0, 1.0)
        # Raw model is HIGH on the visible face. Invert so HIGH = occluder/hidden
        # region → restore original there. ROOP_XSEG3_RAW=1 skips the invert.
        if os.environ.get('ROOP_XSEG3_RAW', '0') != '1':
            result = 1.0 - result
        return result

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_xseg3
        self.model_xseg3 = None
