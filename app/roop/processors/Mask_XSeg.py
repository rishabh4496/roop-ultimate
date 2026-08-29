import numpy as np
import cv2
import onnxruntime
import threading
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path
from roop import session_pool
from roop.precision_policy import providers_for

THREAD_LOCK_CLIP = threading.Lock()


class Mask_XSeg():
    plugin_options:dict = None

    model_xseg = None

    processorname = 'mask_xseg'
    type = 'mask'


    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent TensorRT sessions
        # so the mask runs concurrently across worker threads. None → single shared
        # session serialised by the global lock (original safe default).
        self.pool = None


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_xseg is None:
            model_path = resolve_relative_path('../models/xseg.onnx')
            from roop.utilities import (get_onnx_session_options,
                                        get_small_card_safe_providers)
            _sess_opts = get_onnx_session_options()
            providers = get_small_card_safe_providers(
                roop.globals.execution_providers,
                model_path=model_path,
                stage='mask:xseg')
            providers, _precision = providers_for('masking:xseg', providers, model_path)
            self._cpu_only = providers == ['CPUExecutionProvider']

            def _build(_i=0):
                return onnxruntime.InferenceSession(model_path, _sess_opts, providers=providers)

            self.model_xseg = _build()
            self.model_inputs = self.model_xseg.get_inputs()
            self.model_outputs = self.model_xseg.get_outputs()

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Optional multi-session pool: primary + (N-1) extras → up to N threads
            # run the mask concurrently, each on its own TensorRT context.
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size(
                    model_key='mask:xseg', input_shape=(1, 3, 512, 512))
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_xseg] + extras): _e[i], n,
                    model_key='mask:xseg', input_shape=(1, 3, 512, 512))


    def _run_session(self, sess, temp_frame):
        if getattr(self, '_cpu_only', False):
            return sess.run([o.name for o in self.model_outputs],
                            {self.model_inputs[0].name: temp_frame})
        io_binding = sess.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self.model_outputs[0].name, self.devicename)
        sess.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()


    def Run(self, img1, keywords:str) -> Frame:
        temp_frame = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = temp_frame[None, ...]
        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = self._run_session(sess, temp_frame)
        else:
            ort_outs = self._run_session(self.model_xseg, temp_frame)
        result = ort_outs[0][0]
        result = np.clip(result, 0, 1.0)
        result[result < 0.1] = 0
        # invert values to mask areas to keep
        result = 1.0 - result
        return result


    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_xseg
        self.model_xseg = None


