import os
import threading
from typing import Any, Optional, Union, Dict, Tuple
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool
from roop.precision_policy import providers_for
from roop.degrade import swallowed as _swallowed
from roop.xseg3_optimizer import BUFFER_POOL, MASK_CACHE


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
    _session_lock = threading.Lock()

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

            self._cpu_only = providers == ['CPUExecutionProvider']

            def _build(_i=0):
                return onnxruntime.InferenceSession(model_path, _sess_opts, providers=providers)

            self.model_xseg3 = _build()
            self.model_inputs = self.model_xseg3.get_inputs()
            self.model_outputs = self.model_xseg3.get_outputs()

            dev = str(self.plugin_options["devicename"]).lower()
            self.devicename = 'cuda' if 'cuda' in dev else ('mps' if 'mps' in dev else 'cpu')

            # Optional multi-session pool: primary + (N-1) extras → up to N threads
            # run the mask concurrently, each on its own TensorRT context.
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size(
                    model_key='mask:xseg3', input_shape=(1, 256, 256, 3))
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_xseg3] + extras): _e[i], n,
                    model_key='mask:xseg3', input_shape=(1, 256, 256, 3))

            try:
                from roop.model_lifecycle import register_model_lifecycle, format_shape_from_session
                act_p = self.model_xseg3.get_providers()[0]
                in_shape = format_shape_from_session(self.model_xseg3)
                register_model_lifecycle(
                    model=getattr(self, 'processorname', 'mask_xseg3'),
                    device=self.devicename,
                    provider=act_p,
                    precision=_precision if '_precision' in locals() else "fp32",
                    input_shape=in_shape,
                    engine_cache="ENABLED" if "tensorrt" in act_p.lower() else f"N/A ({act_p})",
                    vram_cost="pooled" if self.pool is not None else "shared",
                    init_time="initialized",
                    session_id=id(self.model_xseg3),
                )
            except Exception as _e_reg:
                _swallowed("roop/processors/Mask_XSeg3.py:model_lifecycle", _e_reg, "model lifecycle register fallback")

    def _get_io_binding(self, sess):
        iob = getattr(sess, '_cached_io_binding', None)
        if iob is None:
            iob = sess.io_binding()
            iob.bind_output(self.model_outputs[0].name, self.devicename)
            sess._cached_io_binding = iob
        return iob

    def _run_session(self, sess, temp_frame):
        if getattr(self, '_cpu_only', False):
            return sess.run([o.name for o in self.model_outputs],
                            {self.model_inputs[0].name: temp_frame})
        iob = self._get_io_binding(sess)
        iob.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        sess.run_with_iobinding(iob)
        return iob.copy_outputs_to_cpu()

    def Run(self, img1, keywords: str = "", target_face: Optional[Any] = None,
            frame_idx: Optional[int] = None, track_id: Optional[Any] = None) -> Frame:
        if img1 is None or getattr(img1, 'size', 0) == 0:
            return img1

        # Check geometry-aware cache for conditional mask reuse
        kps = (target_face.get('kps') if isinstance(target_face, dict)
               else getattr(target_face, 'kps', None)) if target_face is not None else None
        f_idx = frame_idx if frame_idx is not None else 0
        t_id = track_id if track_id is not None else (
            target_face.get('_track_id', target_face.get('track_id')) if isinstance(target_face, dict)
            else (getattr(target_face, '_track_id', None) or getattr(target_face, 'track_id', None))
        ) if target_face is not None else None

        can_reuse, cached_mask, _reason = MASK_CACHE.evaluate_reuse(
            track_id=t_id,
            current_kps=kps,
            target_face=target_face,
            crop_bgr=img1,
            frame_idx=f_idx
        )
        if can_reuse and cached_mask is not None:
            return cached_mask

        # Model input: (1, 256, 256, 3) NHWC, float32 in [0, 1] via preallocated buffer pool
        temp_frame = BUFFER_POOL.prepare_model_input(img1)

        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = self._run_session(sess, temp_frame)
        else:
            with self._session_lock:
                ort_outs = self._run_session(self.model_xseg3, temp_frame)

        # Output: (1, 256, 256, 1) → drop batch + channel dims to a 2D mask.
        result = ort_outs[0][0]
        if result.ndim == 3:
            result = result[..., 0]
        result = np.clip(result, 0.0, 1.0)

        # Raw model is HIGH on the visible face. Invert so HIGH = occluder/hidden
        # region → restore original there. ROOP_XSEG3_RAW=1 skips the invert.
        if os.environ.get('ROOP_XSEG3_RAW', '0') != '1':
            result = 1.0 - result

        # Record fresh observation in geometry cache
        MASK_CACHE.update(
            track_id=t_id,
            mask_256=result,
            current_kps=kps,
            target_face=target_face,
            crop_bgr=img1,
            frame_idx=f_idx
        )

        return result

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        if hasattr(self.model_xseg3, '_cached_io_binding'):
            del self.model_xseg3._cached_io_binding
        del self.model_xseg3
        self.model_xseg3 = None
