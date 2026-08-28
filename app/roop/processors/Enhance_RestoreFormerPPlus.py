from typing import Any, List, Callable
import threading
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.processors.enhance_common import is_usable, sized, exclusive
from roop.precision_policy import providers_for
from roop import session_pool

class Enhance_RestoreFormerPPlus():
    plugin_options:dict = None
    model_restoreformerpplus = None
    devicename = None
    name = None
    pool = None        # SessionPool of (session, io_binding) for TRT multi-context
    # Guards the SINGLE shared session/binding used when there is no pool.
    # Session state lives on the CLASS here, so the lock does too.
    _session_lock = threading.Lock()

    processorname = 'restoreformer++'
    # Every session call goes through `exclusive()`, so no context of
    # this processor's is ever entered twice at once -- the only guarantee
    # ProcessMgr's enhance-stage lock provides. Declaring it lets that
    # stage skip the lock, so this class's HOST work stops serialising
    # against every other worker thread. See enhance_common.exclusive.
    self_excluding = True
    type = 'enhance'
    # FFHQ-trained — see Enhance_CodeFormer.model_template.
    model_template = 'ffhq_512'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_restoreformerpplus is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            model_path = resolve_relative_path('../models/restoreformer_plus_plus.onnx')

            from roop.utilities import get_onnx_session_options
            opts = get_onnx_session_options()
            session_providers, _precision = providers_for(
                'restoreformer_pp', roop.globals.execution_providers, model_path)

            def _build(_i=0):
                sess = onnxruntime.InferenceSession(model_path, opts, providers=session_providers)
                outs = sess.get_outputs()
                iob = sess.io_binding()
                iob.bind_output(outs[0].name, self.devicename)
                return (sess, iob)

            self.model_restoreformerpplus, self.io_binding = _build()
            self.model_inputs = self.model_restoreformerpplus.get_inputs()

            # Optional TensorRT multi-context pool: primary (session, io_binding)
            # plus (N-1) independent extras so N workers can enhance concurrently.
            # Each copy keeps its own io_binding (binding state is not shareable
            # across threads).
            if session_pool.pooling_enabled():
                n = session_pool.pool_size(
                    model_key='enhancer:restoreformer', input_shape=(1, 3, 512, 512))
                extras = [_build(i) for i in range(n - 1)]
                primary = (self.model_restoreformerpplus, self.io_binding)
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([primary] + extras): _e[i], n,
                    model_key='enhancer:restoreformer', input_shape=(1, 3, 512, 512))

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        fallback_bgr = temp_frame   # resized input, kept for the non-finite guard
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)
        
        # An independent (session, io_binding) per worker when pooled, else
        # this class's own lock over the single shared pair -- either way
        # exclusive, and no wider than the model call.
        with exclusive(self.pool, self._session_lock,
                       (self.model_restoreformerpplus, self.io_binding)) as (sess, iob):
            iob.bind_cpu_input(self.model_inputs[0].name, temp_frame)
            sess.run_with_iobinding(iob)
            ort_outs = iob.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs

        # np.clip does not remove NaN and uint8(NaN) is 0 — see
        # enhance_common.is_usable. This one runs on a POOL of TensorRT
        # contexts, so it also covers a torn session, not just an overflow.
        if not is_usable(result):
            print("[RestoreFormer++] non-finite output — using unenhanced frame "
                  "(FP16 overflow? try an fp32 provider)")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return sized(result.astype(np.uint8), input_size)


    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_restoreformerpplus
        self.model_restoreformerpplus = None
        del self.io_binding
        self.io_binding = None

