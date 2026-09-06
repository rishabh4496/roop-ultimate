from typing import Any, List, Callable
import threading
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.processors.enhance_common import (is_usable, sized, exclusive,
                                            fp32_trt_providers,
                                            looks_collapsed)
from roop import session_pool


class Enhance_GFPGAN():
    plugin_options:dict = None

    model_gfpgan = None
    name = None
    devicename = None
    pool = None
    io_binding = None
    _lut = None

    processorname = 'gfpgan'
    # Every session call goes through `exclusive()`, so no context of
    # this processor's is ever entered twice at once -- the only guarantee
    # ProcessMgr's enhance-stage lock provides. Declaring it lets that
    # stage skip the lock, so this class's HOST work stops serialising
    # against every other worker thread. See enhance_common.exclusive.
    self_excluding = True
    # Guards the single shared session when there is no pool.
    _session_lock = threading.Lock()
    type = 'enhance'
    _warned_collapse = False
    # FFHQ-trained — see Enhance_CodeFormer.model_template for the measured
    # mismatch against each swapper's crop, and ProcessMgr for the re-warp.
    model_template = 'ffhq_512'


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_gfpgan is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.onnx')
            # FORCED FP32 UNDER TENSORRT, and this one was silent.
            #
            # GFPGAN v1.4 does not survive TensorRT's FP16 kernels, but it does
            # not overflow to NaN the way GPEN 1024/2048 does — it COLLAPSES.
            # Measured 2026-08-24 on an RTX 4070, same input, same pre/post:
            #
            #   TRT fp16   raw range [-0.47, -0.14]  pixel std 16.0  detail 0.08
            #   TRT fp32   raw range [-1.00,  1.00]  pixel std 65.2  detail 4.35
            #   CUDA       raw range [-1.00,  1.00]  pixel std 65.2  detail 4.35
            #
            # fp32 matches the CUDA reference to 0.03/255; fp16 differs from it
            # by 59/255. Every fp16 value is finite, so `is_usable` never fired
            # and the enhancer shipped returning a uniform grey face that still
            # looked like an image. It cost 65 s to build the FP32 engine once
            # (cached thereafter) and runs at 93 ms against CUDA's 568 ms.
            providers = fp32_trt_providers(roop.globals.execution_providers,
                                           'gfpgan')
            from roop.utilities import get_onnx_session_options
            opts = get_onnx_session_options()
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            def _build(_i=0):
                sess = onnxruntime.InferenceSession(model_path, opts, providers=providers)
                iob = sess.io_binding()
                iob.bind_output(sess.get_outputs()[0].name, self.devicename)
                return (sess, iob)

            self.model_gfpgan, self.io_binding = _build()
            self.name = self.model_gfpgan.get_inputs()[0].name
            self.output_name = self.model_gfpgan.get_outputs()[0].name
            self._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)

            if session_pool.pooling_enabled():
                n = session_pool.pool_size(
                    model_key='enhancer:gfpgan', input_shape=(1, 3, 512, 512))
                cap = plugin_options.get('pool_size')
                if cap:
                    n = max(1, min(int(n), int(cap)))
                gb = session_pool._detect_vram_gb()
                if 0 < gb < 11.5:
                    n = 1
                elif 11.5 <= gb < 15.5:
                    n = min(n, 2)
                if n > 1:
                    extras = []
                    try:
                        extras = [_build(i + 1) for i in range(n - 1)]
                        primary = (self.model_gfpgan, self.io_binding)
                        self.pool = session_pool.SessionPool(
                            lambda i, _e=([primary] + extras): _e[i], n,
                            model_key='enhancer:gfpgan', input_shape=(1, 3, 512, 512))
                    except Exception as e:
                        extras.clear()
                        self.pool = None
                        print(f"[GFPGAN] multi-context pool unavailable ({e}); "
                              f"falling back to one session behind the lock")

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1
        input_size = temp_frame.shape[1]
        if temp_frame.shape[0] != 512 or temp_frame.shape[1] != 512:
            src = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            src = temp_frame
        fallback_bgr = src

        # One gather: uint8 BGR HWC -> float32 RGB CHW in [-1, 1].
        x = self._lut[src.transpose(2, 0, 1)[::-1]][None]

        # Exclusive use of session and io_binding.
        with exclusive(self.pool, self._session_lock,
                       (self.model_gfpgan, self.io_binding)) as (sess, iob):
            iob.bind_cpu_input(self.name, x)
            sess.run_with_iobinding(iob)
            ort_outs = iob.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs

        # np.clip does not remove NaN and uint8(NaN) is 0, so a single
        # overflowed value paints black and a saturated graph paints a black
        # FACE — silently. See enhance_common.is_usable.
        if not is_usable(result):
            print("[GFPGAN] non-finite output — using unenhanced frame "
                  "(FP16 overflow? try an fp32 provider)")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        # post-process
        hwc = np.ascontiguousarray(result[::-1].transpose(1, 2, 0), dtype=np.float32)
        np.maximum(hwc, -1.0, out=hwc)
        res = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        # The guard `is_usable` cannot provide: a finite but COLLAPSED output.
        # The FP32 provider above is the real fix; this is the net that would
        # have caught the failure in the first place instead of letting a flat
        # grey face render for months.
        if looks_collapsed(res, fallback_bgr):
            if not Enhance_GFPGAN._warned_collapse:
                Enhance_GFPGAN._warned_collapse = True
                print("[GFPGAN] output has collapsed to a near-uniform image "
                      "(finite, but no dynamic range) — using the unenhanced "
                      "frame. This is the TensorRT FP16 failure; unset "
                      "ROOP_GFPGAN_FP16 to get the forced-FP32 engine back.")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        return sized(res, input_size)


    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_gfpgan
        self.model_gfpgan = None
        self.io_binding = None
        self._lut = None











