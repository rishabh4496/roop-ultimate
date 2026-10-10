import numpy as np
import cv2
import onnxruntime
import threading
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path
from roop import session_pool, baseline_probe
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
        self._bound = {}     # id(ort session) -> BoundStaticSession (ROOP_TRT_BOUND only)


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

            # Opt-in A/B (ROOP_TRT_BOUND=xseg=bound|graph): persistent device I/O on the session's own
            # stream, optionally replayed as a CUDA graph. Unset = the path below, unchanged.
            from roop.trt_bound_session import bound_mode, BoundStaticSession
            _bmode = None if self._cpu_only else bound_mode('xseg')

            def _build(_i=0):
                if _bmode:
                    try:
                        _in = onnxruntime.InferenceSession(model_path, _sess_opts, providers=providers).get_inputs()[0].name
                        bound = BoundStaticSession(model_path, providers, input_shapes={_in: (1, 256, 256, 3)},
                                                   cuda_graph=(_bmode == 'graph'), session_options=_sess_opts)
                        if 'tensorrt' in bound.provider.lower():
                            baseline_probe.log_session('mask:xseg[%s]' % _bmode, bound.session, providers,
                                                       model_file=model_path)
                            self._bound[id(bound.session)] = bound
                            return bound.session
                        print('[XSeg] ROOP_TRT_BOUND=%s: provider is %s, not TensorRT -- using the shipped path'
                              % (_bmode, bound.provider))
                        bound.close()
                    except Exception as exc:                       # noqa: BLE001
                        print('[XSeg] ROOP_TRT_BOUND=%s failed (%s: %s) -- using the shipped path'
                              % (_bmode, type(exc).__name__, str(exc)[:160]))
                sess = onnxruntime.InferenceSession(model_path, _sess_opts, providers=providers)
                baseline_probe.log_session('mask:xseg', sess, providers, model_file=model_path)
                return sess

            self.model_xseg = _build()
            if not self._cpu_only and id(self.model_xseg) not in self._bound:
                # Startup canary (roop/aux_canary.py): compare the live engine with a CUDA FP32 reference, cached by engine
                # hash. On a failed verdict the session is rebuilt on TensorRT FP32, then CUDA/CPU; `providers` is rebound
                # so the pool extras below are built on whatever the primary now runs on.
                from roop import aux_canary
                self.model_xseg, providers, _verdict = aux_canary.guard(
                    'xseg', self.model_xseg, model_path, providers,
                    lambda chain: onnxruntime.InferenceSession(model_path, _sess_opts, providers=chain))
            self.model_inputs = self.model_xseg.get_inputs()
            self.model_outputs = self.model_xseg.get_outputs()

            dev = str(self.plugin_options["devicename"]).lower()
            self.devicename = 'cuda' if 'cuda' in dev else ('mps' if 'mps' in dev else 'cpu')

            # Optional multi-session pool: primary + (N-1) extras → up to N threads
            # run the mask concurrently, each on its own TensorRT context.
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size(
                    model_key='mask:xseg', input_shape=(1, 3, 512, 512))
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_xseg] + extras): _e[i], n,
                    model_key='mask:xseg', input_shape=(1, 3, 512, 512))

            try:
                from roop.model_lifecycle import register_model_lifecycle, format_shape_from_session
                act_p = self.model_xseg.get_providers()[0]
                in_shape = format_shape_from_session(self.model_xseg)
                register_model_lifecycle(
                    model=getattr(self, 'processorname', 'mask_xseg'),
                    device=self.devicename,
                    provider=act_p,
                    precision=_precision if '_precision' in locals() else "fp32",
                    input_shape=in_shape,
                    engine_cache="ENABLED" if "tensorrt" in act_p.lower() else f"N/A ({act_p})",
                    vram_cost="pooled" if self.pool is not None else "shared",
                    init_time="initialized",
                    session_id=id(self.model_xseg),
                )
            except Exception:
                pass


    def _get_io_binding(self, sess):
        iob = getattr(sess, '_cached_io_binding', None)
        if iob is None:
            iob = sess.io_binding()
            iob.bind_output(self.model_outputs[0].name, self.devicename)
            sess._cached_io_binding = iob
        return iob


    def _run_session(self, sess, temp_frame):
        bound = self._bound.get(id(sess))
        if bound is not None:
            return bound.run({self.model_inputs[0].name: temp_frame})
        if getattr(self, '_cpu_only', False):
            return sess.run([o.name for o in self.model_outputs],
                            {self.model_inputs[0].name: temp_frame})
        iob = self._get_io_binding(sess)
        iob.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        sess.run_with_iobinding(iob)
        return iob.copy_outputs_to_cpu()


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
        for _b in self._bound.values():
            _b.close()
        self._bound = {}
        if hasattr(self.model_xseg, '_cached_io_binding'):
            del self.model_xseg._cached_io_binding
        del self.model_xseg
        self.model_xseg = None


