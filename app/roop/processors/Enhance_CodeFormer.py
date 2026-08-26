from typing import Any, List, Callable
import threading
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.processors.enhance_common import is_usable, sized, exclusive
from roop import session_pool


# THREAD_LOCK = threading.Lock()


class Enhance_CodeFormer():
    model_codeformer = None

    plugin_options:dict = None
    fp16 = False
    in_dtype = np.float32
    pool = None        # SessionPool of independent sessions for TRT multi-context
    # Guards the SINGLE shared session used when there is no pool. This
    # class keeps its session on the CLASS, not the instance (the fidelity
    # weight is bound per call on a fresh io_binding), so the lock lives
    # there too.
    _session_lock = threading.Lock()

    processorname = 'codeformer'
    # Every session call goes through `exclusive()`, so no context of
    # this processor's is ever entered twice at once -- the only guarantee
    # ProcessMgr's enhance-stage lock provides. Declaring it lets that
    # stage skip the lock, so this class's HOST work stops serialising
    # against every other worker thread. See enhance_common.exclusive.
    self_excluding = True
    type = 'enhance'
    # The 5-point alignment this model was TRAINED on. CodeFormer, like every
    # other restorer here that declares one, learned its prior from FFHQ-
    # aligned 512 faces — but the crop it is handed comes from the SWAPPER's
    # template, because the enhancer reuses the swap crop. Measured against
    # ffhq_512 at 512px, the mismatch is not small:
    #
    #   arcface (inswapper family)  scale 0.856   mean landmark error 22.1 px
    #   arcface_112_v1 (ghost…)     scale 0.828                       16.3 px
    #   arcface_112_v2 (blendswap)  scale 0.744                       30.7 px
    #   mtcnn_512 (hififace)        scale 0.875                       13.2 px
    #   ffhq_512 (uniface…)         scale 1.000                        0.0 px
    #
    # With the default swapper the face arrives ~17% larger than the prior
    # expects and the eyes sit 31 px too high. That matters more here than for
    # a plain CNN restorer: CodeFormer's codebook lookup is a discrete nearest
    # neighbour into a learned dictionary, so an off-distribution input
    # retrieves the wrong entries. ProcessMgr re-warps to this template when
    # `enhancer_align` is on. None = "leave the crop alone".
    model_template = 'ffhq_512'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        # A precision switch, not a second processor: same graph, same `w`
        # input, same pre/post — only the weights differ. Measured on an
        # RTX 4070 / CUDA at 512: fp32 162.9 ms/call, fp16 102.0 ms/call
        # (1.60x), with a mean output difference of about 3/255.
        want_fp16 = bool(plugin_options.get('fp16', False))
        if self.plugin_options is not None and self.fp16 != want_fp16:
            self.Release()

        self.plugin_options = plugin_options
        self.fp16 = want_fp16
        if self.model_codeformer is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            # conditional_download saves under the URL's basename, so this has
            # to be the name as published, dots and all.
            name = 'codeformer.fp16.onnx' if want_fp16 else 'CodeFormerv0.1.onnx'
            model_path = resolve_relative_path(f'../models/CodeFormer/{name}')
            from roop.utilities import get_onnx_session_options
            opt_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if want_fp16 else None
            opts = get_onnx_session_options(optimization_level=opt_level)
            def _build(_i=0):
                return onnxruntime.InferenceSession(
                    model_path, opts, providers=roop.globals.execution_providers)

            self.model_codeformer = _build()
            self.model_inputs = self.model_codeformer.get_inputs()
            self.model_outputs = self.model_codeformer.get_outputs()
            self.in_dtype = np.float16 if 'float16' in self.model_inputs[0].type else np.float32

            # Optional TensorRT multi-context pool: the primary session plus
            # (N-1) independent extras, so N workers can enhance concurrently.
            #
            # Without this, `_gpu_guard` hands the enhance stage the GLOBAL GPU
            # lock (it only exempts a processor that owns a pool), which makes
            # the single most expensive stage in a render — ~36% of wall clock —
            # a one-thread-at-a-time queue while every other worker waits. That
            # is a hard ceiling no thread count can lift: with the enhancer at
            # ~22 ms/face and the rest of the per-face work at ~57 ms, the
            # pipeline saturates at ~4 threads and then simply stops scaling.
            #
            # Only SESSIONS are pooled here, not (session, io_binding) pairs as
            # in Enhance_RestoreFormerPPlus. This Run() already builds a fresh
            # io_binding per call — it has to, because the fidelity weight `w`
            # is a second input that must be bound per call — so binding state
            # was never shared and only the TensorRT context needs isolating.
            #
            # Sized from VRAM by session_pool, so a card that cannot afford the
            # extra engines gets 0 and the original single-session + global-lock
            # behaviour back, byte for byte.
            #
            # MEASURED cost of this pool, RTX 4070, trt_precision 'mixed',
            # 4 contexts, each one actually run (see the trap below):
            #
            #   RestoreFormer++   3345 MB   (683 MB per extra context)
            #   CodeFormer fp16   2763 MB   (530 MB per extra context)
            #   CodeFormer fp32   2700 MB   (560 MB per extra context)
            #
            # So BOTH tiers here are cheaper than RestoreFormer++, which has
            # always pooled 4. Pooling this one is not a new risk; it is less
            # than the risk already shipped as the recommended enhancer.
            #
            # Note fp32 is not the expensive tier — it is marginally cheaper.
            # The ONNX file sizes (359 MB fp32 vs 180 MB fp16) do NOT predict
            # VRAM, because 'mixed' has TensorRT build FP16 kernels from either
            # set of weights: what lives on the card is the engine, not the
            # file. Do not re-derive this budget from model file sizes.
            #
            # The trap, if these are ever remeasured: creating a session is not
            # what allocates the execution context. Measured without running an
            # inference, every extra context looks like 10 MB and the whole
            # question looks free.
            #
            # trt_precision 'fp32' is NOT covered by the above — that builds
            # genuinely FP32 engines and was not measured. The fallback below is
            # what carries that case.
            #
            # Building the extras can still run the card out of memory on a
            # config the VRAM tier did not anticipate — a big swapper, an
            # expression pool and four restorer contexts all want the same
            # 12GB. An OOM here must not take the render down, because the
            # single-session path is right there and works: on failure, drop
            # whatever was built and carry on with the global lock. Slower,
            # never broken.
            if session_pool.pooling_enabled():
                n = session_pool.pool_size()
                # A caller that is loading this net ALONGSIDE other heavy models
                # can cap the pool. Measured 2026-08-22: with realswap's two
                # nets, RealityUX and the detector pools already resident, this
                # pool's extra contexts push a 12GB card to 96% and the render
                # collapses to 0.2 fps -- 100% "utilisation" at a THIRD of the
                # power limit, which is thrashing, not compute. The comment
                # above anticipated exactly that ("a big swapper, an expression
                # pool and four restorer contexts all want the same 12GB").
                # Absent, nothing changes for any existing caller.
                cap = plugin_options.get('pool_size')
                if cap:
                    n = max(1, min(int(n), int(cap)))
                extras = []
                try:
                    extras = [_build(i) for i in range(n - 1)]
                    self.pool = session_pool.SessionPool(
                        lambda i, _e=([self.model_codeformer] + extras): _e[i], n)
                except Exception as e:
                    extras.clear()
                    self.pool = None
                    print(f"[CodeFormer] multi-context pool unavailable ({e}); "
                          f"falling back to one session behind the GPU lock")


    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[1]
        # preprocess
        temp_frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        fallback_bgr = temp_frame   # resized input, kept for the non-finite guard
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        # Fresh io_binding per call: a shared binding is not thread-safe, and
        # this graph needs one anyway because the fidelity weight `w` is a
        # second input bound per call (float64 for this ONNX export). Each
        # input is bound exactly once per binding, with a consistent dtype.
        cf_fidelity = getattr(roop.globals, 'codeformer_fidelity', 0.5)

        def _infer(sess):
            iob = sess.io_binding()
            iob.bind_cpu_input(self.model_inputs[0].name,
                               temp_frame.astype(self.in_dtype))
            iob.bind_cpu_input(self.model_inputs[1].name,
                               np.array([cf_fidelity], dtype=np.float64))
            iob.bind_output(self.model_outputs[0].name, self.devicename)
            sess.run_with_iobinding(iob)
            return iob.copy_outputs_to_cpu()

        # An independent session per worker when pooled, else this class's own
        # lock over the single shared one -- either way exclusive, and no wider
        # than the model call. Input and output NAMES are identical across
        # sessions built from one ONNX, so the cached self.model_inputs/outputs
        # stay valid for any lease.
        with exclusive(self.pool, self._session_lock,
                       self.model_codeformer) as sess:
            ort_outs = _infer(sess)
        # float32 regardless of the model's precision — every step below
        # (clip, rescale, cvtColor) is written for it, and cv2 rejects float16.
        result = np.asarray(ort_outs[0][0], dtype=np.float32)
        del ort_outs

        # np.clip does not remove NaN and uint8(NaN) is 0, so a single
        # overflowed value paints black and a saturated graph paints a black
        # FACE — silently. See enhance_common.is_usable. Especially worth
        # having on the fp16 tier, which is a half-precision graph.
        if not is_usable(result):
            print("[CodeFormer] non-finite output — using unenhanced frame "
                  "(FP16 overflow? try the fp32 tier)")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        # post-process
        result = result.transpose((1, 2, 0))

        un_min = -1.0
        un_max = 1.0
        result = np.clip(result, un_min, un_max)
        result = (result - un_min) / (un_max - un_min)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = (result * 255.0).round()
        return sized(result.astype(np.uint8), input_size)


    def Release(self):
        # Pool first: it holds the extra sessions AND a reference to the
        # primary one, so dropping the primary while the pool still owns it
        # would leave live TensorRT contexts with no owner to free them.
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_codeformer
        self.model_codeformer = None

