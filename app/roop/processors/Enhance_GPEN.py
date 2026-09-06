from typing import Any, List, Callable
import threading
import os
import cv2
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path, conditional_download
from roop.processors.enhance_common import is_usable, sized, fp32_trt_providers, exclusive
from roop.precision_policy import providers_for
from roop import session_pool


def _fp32_trt_providers(providers):
    """GPEN 1024/2048 overflow in FP16 under TensorRT and paint a solid black
    face. Delegates to the shared helper — see enhance_common.fp32_trt_providers
    for the two distinct FP16 failure modes and the measurements behind them.
    ROOP_GPEN_FP16=1 opts back in (not recommended at >=1024)."""
    import os
    if os.environ.get('ROOP_GPEN_FP16', '0') == '1':
        return providers
    return fp32_trt_providers(providers, 'gpen')


# GPEN blind face restoration at four native resolutions. 512 is the classic
# roop weight; 1024/2048 are the FaceFusion exports for large/close-up faces
# (better detail ceiling, proportionally more VRAM/time). 256 goes the other
# way: a quarter of 512's pixels through the net, for distant faces, comparison
# grids and preview scrubbing where 512's detail is thrown away by the paste
# downscale anyway.
GPEN_MODELS = {
    256: {
        "file": "gpen_bfr_256.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_256.onnx",
    },
    512: {
        "file": "GPEN-BFR-512.onnx",
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
    },
    1024: {
        "file": "gpen_bfr_1024.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_1024.onnx",
    },
    2048: {
        "file": "gpen_bfr_2048.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_2048.onnx",
    },
}


class Enhance_GPEN():
    plugin_options:dict = None

    model_gpen = None
    name = None
    devicename = None

    processorname = 'gpen'
    # Every session call goes through `exclusive()`, so no context of
    # this processor's is ever entered twice at once -- the only guarantee
    # ProcessMgr's enhance-stage lock provides. Declaring it lets that
    # stage skip the lock, so this class's HOST work stops serialising
    # against every other worker thread. See enhance_common.exclusive.
    self_excluding = True
    # Guards the single shared session. These two build a fresh io_binding
    # per call, but the SESSION (and so its TensorRT context) is shared,
    # and that is what must not be entered twice at once.
    _session_lock = threading.Lock()
    type = 'enhance'
    # FFHQ-trained — see Enhance_CodeFormer.model_template.
    model_template = 'ffhq_512'

    def __init__(self):
        self.model_size = 512
        self.sessions = {}
        self.pools = {}
        self.io_bindings = {}
        self.pool = None
        self._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options

        size = int(plugin_options.get("size", 512))
        if size not in GPEN_MODELS:
            size = 512

        devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

        if size not in self.sessions:
            spec = GPEN_MODELS[size]
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [spec["url"]])
            model_path = f"{model_dir}/{spec['file']}"
            providers = roop.globals.execution_providers
            # 1024/2048 overflow in FP16 → black face; force FP32 on TensorRT.
            # 512 (classic weight) is stable in FP16, so leave it fast.
            if size >= 1024:
                providers = _fp32_trt_providers(providers)
            else:
                providers, _precision = providers_for(
                    f'gpen_{size}', providers, model_path)
            from roop.utilities import get_onnx_session_options
            opts = get_onnx_session_options()

            def _build(_i=0):
                sess = onnxruntime.InferenceSession(model_path, opts, providers=providers)
                iob = sess.io_binding()
                iob.bind_output(sess.get_outputs()[0].name, devicename)
                return (sess, iob)

            session, iob = _build()
            from roop import predictor
            predictor.verify_and_warmup(session, providers,
                                        f'enhancer:gpen_{size}',
                                        default_hw=(size,))
            self.sessions[size] = session
            self.io_bindings[size] = iob

            if session_pool.pooling_enabled():
                n = session_pool.pool_size(
                    model_key=f'enhancer:gpen_{size}',
                    input_shape=(1, 3, size, size))
                cap = plugin_options.get('pool_size')
                if cap:
                    n = max(1, min(int(n), int(cap)))
                gb = session_pool._detect_vram_gb()
                if 0 < gb < 11.5 or size >= 1024:
                    n = 1
                elif 11.5 <= gb < 15.5:
                    n = min(n, 2 if size >= 512 else 4)
                if n > 1:
                    extras = []
                    try:
                        extras = [_build(i + 1) for i in range(n - 1)]
                        primary = (session, iob)
                        self.pools[size] = session_pool.SessionPool(
                            lambda i, _e=([primary] + extras): _e[i], n,
                            model_key=f'enhancer:gpen_{size}',
                            input_shape=(1, 3, size, size))
                    except Exception as e:
                        extras.clear()
                        self.pools[size] = None
                        print(f"[GPEN] multi-context pool unavailable ({e}); "
                              f"falling back to one session behind the lock")

        self.devicename = devicename
        self.model_size = size
        self.model_gpen = self.sessions[size]
        self.pool = self.pools.get(size)
        self.name = self.model_gpen.get_inputs()[0].name
        self.output_name = self.model_gpen.get_outputs()[0].name

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1
        input_size = temp_frame.shape[1]
        sz = self.model_size
        if temp_frame.shape[0] != sz or temp_frame.shape[1] != sz:
            src = cv2.resize(temp_frame, (sz, sz), interpolation=cv2.INTER_CUBIC)
        else:
            src = temp_frame
        fallback_bgr = src

        # One gather: uint8 BGR HWC -> float32 RGB CHW in [-1, 1].
        x = self._lut[src.transpose(2, 0, 1)[::-1]][None]

        with exclusive(self.pools.get(sz), self._session_lock,
                       (self.model_gpen, self.io_bindings.get(sz))) as (sess, iob):
            iob.bind_cpu_input(self.name, x)
            sess.run_with_iobinding(iob)
            ort_outs = iob.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs

        # Defense-in-depth: FP16 overflow or a torn session can yield non-finite
        # output; np.clip would keep the NaN and uint8(NaN)=0 paints a solid black
        # face. Fall back to the unenhanced (resized) input so a black frame can
        # never reach the screen. The FP32 provider above is the real fix; this is
        # the safety net for any residual/other cause.
        if not is_usable(result):
            print("[GPEN] non-finite output — using unenhanced frame "
                  "(FP16 overflow? set trt precision to fp32 or ROOP_GPEN_FP16=0)")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        # post-process
        hwc = np.ascontiguousarray(result[::-1].transpose(1, 2, 0), dtype=np.float32)
        np.maximum(hwc, -1.0, out=hwc)
        res = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)
        return sized(res, input_size)


    def Release(self):
        for pool in self.pools.values():
            if pool is not None:
                pool.release()
        self.pools.clear()
        self.io_bindings.clear()
        self.sessions.clear()
        self.pool = None
        self.model_gpen = None
