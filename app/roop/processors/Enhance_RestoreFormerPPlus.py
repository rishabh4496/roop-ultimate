from typing import Any, List, Callable
import os
import threading
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.processors.enhance_common import (is_usable, looks_collapsed, sized,
                                            exclusive)
from roop.precision_policy import providers_for
from roop import session_pool, baseline_probe

try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
except (ImportError, OSError):                      # CPU-only / no torch: the host path runs
    torch = None
    _TORCH_CUDA = False

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

            # Experimental, opt-in: run a natively built TensorRT engine (per-layer FP32 islands; see
            # tools/restorer_native_engine.py and docs/perf/restorer_fp16_islands_2026-10-10.md). The ORT TensorRT
            # provider cannot express a per-layer precision, so this is the only way an island reaches a render.
            native_engine = os.environ.get('ROOP_RESTORER_NATIVE_ENGINE', '').strip()

            def _bind(sess):
                iob = sess.io_binding()
                iob.bind_output(sess.get_outputs()[0].name, self.devicename)
                return (sess, iob)

            def _build_ort(_i=0):
                from roop.strict_session import create_strictly_accelerated_session
                sess = create_strictly_accelerated_session(
                    model_path, providers=session_providers, sess_options=opts,
                    tag='enhancer:restoreformer++')
                baseline_probe.log_session('enhancer:restoreformer++', sess,
                                           session_providers, model_file=model_path)
                return _bind(sess)

            def _build_native(_i=0):    # the opt-in native engine replaces the ORT builder, pool extras included
                from roop.trt_native_runner import NativeSession
                sess = NativeSession(native_engine)
                print('[Session] enhancer:restoreformer++ file=%s provider=TensorrtNativeEngine trt_fp16=islands '
                      'input=input:1x3x512x512' % os.path.basename(native_engine), flush=True)
                return (sess, sess.io_binding())

            _build = _build_native if native_engine else _build_ort
            self.model_restoreformerpplus, self.io_binding = _build()
            if native_engine:
                # The native engine used to skip the canary entirely: it is the one RF++ path with per-layer precision the
                # perceptual check could not see. Same two inputs and floors as the ORT engine, verdict cached by the engine
                # file's content. A failed engine is dropped for the ORT engine (FP32 by default), which is then canary-checked
                # below like any other start; the downgrade is recorded for the runtime report.
                from roop import aux_canary
                _verdict = aux_canary.check_session('restoreformer_plus_plus', self.model_restoreformerpplus, model_path)
                if _verdict.passed is False:
                    print('[AuxCanary] restoreformer_plus_plus: the native engine FAILED - %s; using the ONNX Runtime engine'
                          % _verdict.reason, flush=True)
                    aux_canary._record_downgrade('restoreformer_plus_plus', 'tensorrt-native', 'tensorrt-ort', _verdict.reason)
                    native_engine = ''
                    _build = _build_ort
                    self.model_restoreformerpplus, self.io_binding = _build()
            if not native_engine:
                # Startup canary (roop/aux_canary.py): two fixed inputs through the live engine and a one-off FP32 reference,
                # SSIM >= 0.96 and cosine >= 0.995, verdict cached by engine hash. A TensorRT engine that builds, runs at speed and smears the
                # face passes every other check. On failure the restorer is rebuilt on TensorRT FP32, then CUDA/CPU;
                # `session_providers` is rebound so the pool extras below are built on whatever the primary now runs on.
                from roop import aux_canary
                _guarded, session_providers, _verdict = aux_canary.guard(
                    'restoreformer_plus_plus', self.model_restoreformerpplus, model_path, session_providers,
                    lambda chain: onnxruntime.InferenceSession(model_path, opts, providers=chain))
                if _guarded is not self.model_restoreformerpplus:
                    self.model_restoreformerpplus, self.io_binding = _bind(_guarded)
            self.model_inputs = self.model_restoreformerpplus.get_inputs()
            self._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)

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

            try:
                from roop.model_lifecycle import register_model_lifecycle, format_shape_from_session
                act_p = self.model_restoreformerpplus.get_providers()[0]
                in_shape = format_shape_from_session(self.model_restoreformerpplus)
                register_model_lifecycle(
                    model=getattr(self, 'processorname', 'restoreformer++'),
                    device=self.devicename,
                    provider=act_p,
                    precision=_precision if '_precision' in locals() else "mixed",
                    input_shape=in_shape,
                    engine_cache="ENABLED" if "tensorrt" in act_p.lower() else f"N/A ({act_p})",
                    vram_cost="pooled" if self.pool is not None else "shared",
                    init_time="initialized",
                    session_id=id(self.model_restoreformerpplus),
                )
            except Exception as _e_reg:
                from roop.degrade import swallowed as _swallowed
                _swallowed("roop/processors/Enhance_RestoreFormerPPlus.py:model_lifecycle", _e_reg, "model lifecycle register fallback")

    # ── zero-copy path ───────────────────────────────────────────────────────
    _zc_ok = None          # None = undecided; decided once, on the first face
    _zc_faces = 0
    _lut_t = None
    _zc_lock = threading.Lock()

    def _zero_copy_ready(self):
        """An ONNX Runtime CUDA/TensorRT session with a static float32 (1,3,512,512) first
        output. The opt-in native TensorRT engine has its own binding API and keeps the host path."""
        if self._zc_ok is not None:
            return self._zc_ok
        ok = False
        try:
            sess = self.model_restoreformerpplus
            if (_TORCH_CUDA and self.devicename == 'cuda' and isinstance(sess, onnxruntime.InferenceSession)
                    and sess.get_providers()[0] in ('CUDAExecutionProvider', 'TensorrtExecutionProvider')):
                i, o = sess.get_inputs()[0], sess.get_outputs()[0]
                want = [1, 3, 512, 512]
                ok = (i.type == o.type == 'tensor(float)'
                      and list(i.shape) == want and list(o.shape) == want)
        except Exception as error:
            from roop.degrade import swallowed
            swallowed("roop/processors/Enhance_RestoreFormerPPlus.py:zero_copy_ready", error, "host path used")
        self._zc_ok = ok
        return ok

    def _run_zero_copy(self, src, input_size):
        """Normalise on the GPU into a ring slot, let ORT read/write the slot by pointer,
        finish on the GPU (finite check, uint8 cast, collapse test) and download the face once.
        The host path's bytes (tests/test_enhancer_restoreformer_zero_copy.py). None = host path."""
        from roop.optimized_processor import get_zero_copy_pool, zero_copy_enabled
        if not zero_copy_enabled() or not self._zero_copy_ready():
            return None
        from roop.processors.enhance_common import looks_collapsed_tensor
        dev = int(getattr(roop.globals, 'cuda_device_id', 0) or 0)
        try:
            pool = get_zero_copy_pool(dev)
            if pool is None:
                return None
            if self._lut_t is None or self._lut_t.device.index != dev:
                from roop.restore_ultra_optimizer import RestoreUltraBufferPool
                self._lut_t = torch.as_tensor(RestoreUltraBufferPool._LUT_NORM, device=f'cuda:{dev}')
            # Lock order everywhere: ring slot first, then the session lease.
            with pool.lease(512) as slot:
                src_t = torch.from_numpy(np.ascontiguousarray(src)).to(f'cuda:{dev}')
                slot.load(self._lut_t[src_t.long()].permute(2, 0, 1).flip(0).unsqueeze(0))
                with exclusive(self.pool, self._session_lock,
                               (self.model_restoreformerpplus, self.io_binding)) as (sess, iob):
                    out = slot.run(sess, iob, self.model_inputs[0].name, sess.get_outputs()[0].name,
                                   dev, output_is_static=True)
                finite = torch.isfinite(out).all()
                # postprocess_model_output: RGB CHW -> BGR HWC, (x + 1) * 127.5, clip, truncating cast.
                res = ((out[0].flip(0).permute(1, 2, 0) + 1.0) * 127.5).clamp_(0.0, 255.0).to(torch.uint8)
                collapsed = looks_collapsed_tensor(res, src_t)
                flags = torch.stack([finite, collapsed]).cpu()
                if not bool(flags[0]):
                    print("[RestoreFormer++] non-finite output — using unenhanced frame "
                          "(FP16 overflow? try an fp32 provider)")
                    return sized(src.astype(np.uint8), input_size)
                if bool(flags[1]):
                    print("[RestoreFormer++] collapsed output — using unenhanced frame")
                    return sized(src.astype(np.uint8), input_size)
                result = res.cpu().numpy()
        except Exception as error:
            from roop.degrade import swallowed
            swallowed("roop/processors/Enhance_RestoreFormerPPlus.py:zero_copy", error, "host path used")
            self._zc_ok = False
            print(f"[RestoreFormer++] zero-copy path disabled ({type(error).__name__}: {error}); "
                  f"host path used", flush=True)
            return None
        with self._zc_lock:
            Enhance_RestoreFormerPPlus._zc_faces += 1
        return sized(result, input_size)

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1
        from roop.smart_restorer_bypass import should_bypass_restorer, apply_dog_unsharp_mask
        _cfg = getattr(roop.globals, 'CFG', None)
        _bypass_toggle = getattr(_cfg, 'auto_bypass_restorer_native_swap', False) if _cfg else False
        _swap_m = getattr(roop.globals, 'swap_model', None)
        if should_bypass_restorer(target_face, _swap_m, self, enabled=_bypass_toggle, max_height_px=320.0):
            return apply_dog_unsharp_mask(temp_frame), 1
        input_size = temp_frame.shape[1]
        if temp_frame.shape[0] != 512 or temp_frame.shape[1] != 512:
            src = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            src = temp_frame
        fallback_bgr = src

        zero_copy = self._run_zero_copy(src, input_size)
        if zero_copy is not None:
            return zero_copy

        # Preallocated buffer gather: uint8 BGR HWC -> float32 RGB CHW in [-1, 1].
        from roop.restore_ultra_optimizer import BUFFER_POOL
        x = BUFFER_POOL.prepare_model_input(src)

        # An independent (session, io_binding) per worker when pooled, else
        # this class's own lock over the single shared pair -- either way
        # exclusive, and no wider than the model call.
        with exclusive(self.pool, self._session_lock,
                       (self.model_restoreformerpplus, self.io_binding)) as (sess, iob):
            # The zero-copy path clears this binding after each run; bind the
            # output here instead of relying on what `_bind` did at construction.
            if hasattr(iob, 'clear_binding_outputs'):
                iob.clear_binding_outputs()
            iob.bind_cpu_input(self.model_inputs[0].name, x)
            iob.bind_output(sess.get_outputs()[0].name, self.devicename)
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

        res = BUFFER_POOL.postprocess_model_output(result)
        if looks_collapsed(res, fallback_bgr):
            print("[RestoreFormer++] collapsed output — using unenhanced frame")
            return sized(fallback_bgr.astype(np.uint8), input_size)
        return sized(res, input_size)


    def Release(self):
        if Enhance_RestoreFormerPPlus._zc_faces:
            print(f"[RestoreFormer++] {Enhance_RestoreFormerPPlus._zc_faces} faces on the zero-copy path",
                  flush=True)
            Enhance_RestoreFormerPPlus._zc_faces = 0
        self._zc_ok = None
        self._lut_t = None
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_restoreformerpplus
        self.model_restoreformerpplus = None
        del self.io_binding
        self.io_binding = None
        self._lut = None
