"""GPEN Realistic — GPEN-512's detail without GPEN's colour, pooled and lean.

TWO FINDINGS, and the second corrects the first build of this file.

1. GPEN's problem is COLOUR, not detail. It gets called "plastic" or
   "cartoonish", which sounds like softness; it is not. GPEN pushes the whole
   face pink, paints magenta onto the eyelids and flushes the cheeks. Measured
   against the crop the restorer was handed, chroma drift (LAB a/b, mean abs)
   runs 2.7-3.0 while the input is 0. Keeping GPEN's LUMINANCE and taking
   chrominance from the swapper's own crop removes it — 2.72 -> 0.36 at 512 —
   with detail completely unchanged, for 0.27 ms.

2. But the size that matters is 512, not 256, and the reason is the PASTE.
   `realswap` emits a 256 crop, so a 256 restorer returns 256 and pastes at
   scale 1, while a 512 restorer returns 512 and pastes at scale 2 — twice the
   resolution reaches the frame. Detail carried through to the paste, measured
   on a real crop as high-frequency std at 512:

       swap input 2.67 | GPEN-256 2.82 | CodeFormer-512 4.11 | GPEN-512 5.14

   GPEN-256 sits 1.8x below GPEN-512 and barely above the UNENHANCED input. The
   first version of this processor used 256 plus the colour fix and was reported
   as indistinguishable from plain GPEN-256 — correctly. A post-filter cannot
   recover detail the network never synthesised, and that was the mistake.

So this runs GPEN-512, which measures SHARPER THAN CODEFORMER (5.14 vs 4.11)
and faster than it (30.9 ms vs 37.9 before optimisation). Its bad reputation is
finding 1: the cast, which finding 1 removes.

The colour fix is a signed grey delta added to all three BGR channels rather
than a LAB channel swap, because both remove the cast (0.30 vs 0.11 residual
against a ~2.9 problem), they are visually indistinguishable, and the grey delta
costs 0.27 ms against 0.60. `_LAB_EXACT` switches forms.

FREE SPEED ON TOP. `Enhance_GPEN` has no SessionPool, so it runs behind
ProcessMgr's GLOBAL GPU lock — the most expensive stage in a render, serialised
to one thread while every other worker waits. This processor owns a pool, so the
stage is lock-free and scales with the worker count. Its host path is the lean
one too (a 256-entry LUT gather for pre, one saturating C++ scale for post, an
io_binding held per pool slot) instead of the five-pass float32 spelling.

WHAT THIS IS NOT: it is not GPEN-256 speed. 256-net speed and 512-net sharpness
are not available together — the detail is synthesised by the network, and the
smaller one does not synthesise it. ROOP_GPENR_SIZE=256 gets the fast, much
softer tier back for preview scrubbing.

Knobs, for re-measuring rather than for shipping a different default:
    ROOP_GPENR_CHROMA   0 = the swapper's colour (default), 1 = GPEN's own
    ROOP_GPENR_SIZE     512 (default) or 256 for the fast, much softer tier
"""

import os
import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.typing import Face, FaceSet, Frame
from roop.processors.enhance_common import (looks_collapsed, sized, exclusive,
                                            luma_only_recolour)
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool


class Enhance_GPENRealistic:
    processorname = 'gpen_realistic'
    # Every session call goes through `exclusive()`, so no context of
    # this processor's is ever entered twice at once -- the only guarantee
    # ProcessMgr's enhance-stage lock provides. Declaring it lets that
    # stage skip the lock, so this class's HOST work stops serialising
    # against every other worker thread. See enhance_common.exclusive.
    self_excluding = True
    type = 'enhance'
    # FFHQ-trained, like every restorer here — see
    # Enhance_CodeFormer.model_template for what that costs against the swap
    # crop and what `enhancer_align` does about it.
    model_template = 'ffhq_512'

    # 512, NOT 256, and the reason is the paste rather than the network.
    # `realswap` emits a 256 crop, so a 256 restorer returns 256 and pastes at
    # scale 1, while a 512 restorer returns 512 and pastes at scale 2 — twice the
    # resolution reaches the frame. Measured on a real crop, detail carried
    # through to the paste (high-frequency std at 512):
    #
    #     swap input 2.67 | GPEN-256 2.82 | CodeFormer-512 4.11 | GPEN-512 5.14
    #
    # GPEN-256 is 1.8x short of GPEN-512 and barely above the unenhanced input;
    # no post-filter recovers that, because the detail was never synthesised. The
    # first build of this processor used 256 and was correctly reported as
    # indistinguishable from plain GPEN-256.
    #
    # GPEN-512 is also SHARPER THAN CODEFORMER (5.14 vs 4.11) and faster than it
    # (30.9 ms vs 37.9 unoptimised). Its reputation for looking bad is the colour
    # cast this class removes: raw chroma drift 2.72, after the fix 0.36, with
    # detail unchanged at 5.14.
    #
    # ROOP_GPENR_SIZE=256 gets the fast/soft tier back for preview scrubbing.
    _SIZES = {
        256: ('gpen_bfr_256.onnx',
              'https://huggingface.co/facefusion/models-3.0.0/resolve/main/'
              'gpen_bfr_256.onnx'),
        512: ('GPEN-BFR-512.onnx',
              'https://huggingface.co/countfloyd/deepfake/resolve/main/'
              'GPEN-BFR-512.onnx'),
    }
    _SIZE = 512

    # The LAB channel swap is more exact (0.11 residual drift vs 0.30) and more
    # than twice the cost. Off, because this processor exists to be fast and the
    # two are indistinguishable on footage.
    _LAB_EXACT = False

    _warned_colour = False

    def __init__(self):
        self.plugin_options = None
        self.devicename = None
        self.session = None
        self.io_binding = None
        self.size = self._SIZE
        self.pool = None          # SessionPool of (session, io_binding) slots
        self.in_name = None
        self.out_name = None
        self._lut = None
        self._faces = 0
        self._lock = threading.Lock()
        # Guards the SINGLE shared (session, io_binding) used when there is no
        # pool -- binding state is not thread-safe. Distinct from _lock, which
        # only counts faces.
        self._session_lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options.get("devicename") != plugin_options.get("devicename"):
                self.Release()
        self.plugin_options = plugin_options
        self.devicename = plugin_options["devicename"].replace('mps', 'cpu')

        if self.session is not None:
            return

        try:
            want = int(os.environ.get('ROOP_GPENR_SIZE', '') or self._SIZE)
        except ValueError:
            want = self._SIZE
        self.size = want if want in self._SIZES else self._SIZE
        fname, url = self._SIZES[self.size]
        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [url])
        model_path = os.path.join(model_dir, fname)
        from roop.utilities import get_onnx_session_options
        opts = get_onnx_session_options()

        def _build(_i=0):
            # No FP32 forcing here. That guard exists for GPEN 1024/2048, whose
            # activations overflow in FP16 and paint a black face. 512 is the
            # classic weight and is FP16-stable — Enhance_GPEN says so itself and
            # only forces FP32 at >=1024 — and 256 more so.
            sess = onnxruntime.InferenceSession(
                model_path, opts, providers=roop.globals.execution_providers)
            iob = sess.io_binding()
            iob.bind_output(sess.get_outputs()[0].name, self.devicename)
            return (sess, iob)

        self.session, self.io_binding = _build()
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        # uint8 -> float32 normalised to [-1, 1], in one gather.
        self._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)

        # THE POINT OF THE POOL: `_gpu_guard` only exempts a processor that owns
        # one. Without it this stage takes the global GPU lock and the enhancer —
        # ~36% of a render — becomes a one-thread queue no thread count can
        # widen. GPEN-256 is small (a quarter of 512's pixels), so extra
        # contexts are cheap; `pool_size` still caps it for a caller loading
        # this alongside other heavy nets.
        # VRAM CAP, measured rather than assumed. A GPEN-512 pool of 2 costs
        # 3123 MiB -- 1.8x CodeFormer-fp16's -- and on a 12GB card, alongside
        # realswap's two nets, RealityUX and the detector/detmask pools, that
        # tips the card into paging. Measured end to end on s1.mp4: UltraMax
        # 10.50 fps against this processor's 6.60 with an uncapped pool, even
        # though it is FASTER per face in isolation (27.5 ms vs 30.6). 100%
        # "utilisation" at a fraction of the power limit is thrashing, not work,
        # and it is exactly what Enhance_CodeFormer's pool comment predicted.
        #
        # So the 512 tier caps its pool by free VRAM. 256 is a quarter of the
        # pixels and is left alone. ROOP_GPENR_POOL overrides for re-measuring.
        if session_pool.pooling_enabled():
            n = session_pool.pool_size()
            cap = plugin_options.get('pool_size')
            if cap:
                n = max(1, min(int(n), int(cap)))
            try:
                forced = int(os.environ.get('ROOP_GPENR_POOL', '') or 0)
            except ValueError:
                forced = 0
            if forced:
                n = max(1, forced)
            elif self.size >= 512:
                gb = session_pool._detect_vram_gb()
                if 0 < gb < 15.5:
                    n = min(n, 1)
            extras = []
            try:
                extras = [_build(i + 1) for i in range(n - 1)]
                primary = (self.session, self.io_binding)
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([primary] + extras): _e[i], n)
            except Exception as e:
                extras.clear()
                self.pool = None
                print(f"[GPEN Realistic] multi-context pool unavailable ({e}); "
                      f"falling back to one session behind the GPU lock")

    def Release(self):
        line = self.cost_summary()
        if line:
            print(line, flush=True)
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        self.io_binding = None
        self.session = None
        self._lut = None

    # ── colour ───────────────────────────────────────────────────────────────
    @classmethod
    def _keep_source_colour(cls, restored, source, chroma=0.0):
        """GPEN's luminance, the swapper's chrominance.

        A luminance-only edit is the same signed offset on every channel, so
        adding `grey(restored) - grey(source)` to the source moves brightness
        and leaves hue and saturation where the swapper put them. That is the
        whole colour fix, and it is two C++ passes.

        `chroma` lerps back toward GPEN's own colour for anyone who wants to
        re-measure; 0 is the default and the reason this processor exists.

        The body lives in `enhance_common.luma_only_recolour` because UltraMax
        needs the identical operator — CodeFormer drifts the OTHER way, pale
        rather than pink, and the same two passes fix both. This stays as the
        named entry point: it carries this class's `_LAB_EXACT` and its
        warn-once, and the GPEN Realistic tests drive it directly.
        """
        try:
            return luma_only_recolour(restored, source, chroma, cls._LAB_EXACT)
        except cv2.error as e:
            # Say so once. A colour fix that silently no-ops leaves a plausible
            # image with the exact pink cast this class exists to remove, and
            # nothing anywhere would show it.
            if not cls._warned_colour:
                cls._warned_colour = True
                print(f"[GPEN Realistic] colour fix skipped: {e}", flush=True)
            return restored

    # ── run ──────────────────────────────────────────────────────────────────
    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1

        input_size = temp_frame.shape[1]
        S = self.size
        if temp_frame.shape[0] != S or temp_frame.shape[1] != S:
            src = cv2.resize(temp_frame, (S, S), interpolation=cv2.INTER_CUBIC)
        else:
            src = temp_frame

        # One gather: uint8 BGR HWC -> float32 RGB CHW in [-1, 1]. Fancy indexing
        # returns a fresh C-contiguous array, so [None] after it is a free view.
        x = self._lut[src.transpose(2, 0, 1)[::-1]][None]

        def _infer(sess, iob):
            iob.bind_cpu_input(self.in_name, x)
            sess.run_with_iobinding(iob)
            return iob.copy_outputs_to_cpu()

        with exclusive(self.pool, self._session_lock,
                       (self.session, self.io_binding)) as (sess, iob):
            ort_outs = _infer(sess, iob)

        hwc = np.ascontiguousarray(ort_outs[0][0][::-1].transpose(1, 2, 0),
                                   dtype=np.float32)
        del ort_outs

        # np.clip does NOT remove NaN and uint8(NaN) is 0, so one overflowed
        # value paints a pixel black and a saturated graph paints the whole FACE
        # black, silently. A single sum propagates NaN and +-inf for a fraction
        # of isfinite().all()'s cost. See enhance_common.is_usable.
        if not np.isfinite(hwc.sum()):
            print("[GPEN Realistic] non-finite output — using unenhanced frame")
            return sized(temp_frame, input_size)

        # maximum() first so convertScaleAbs' abs() is a no-op on the low end;
        # its saturate_cast handles the high end. Two passes, both in C++.
        np.maximum(hwc, -1.0, out=hwc)
        restored = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        # NOT `is_usable` here: this array is uint8 and np.isfinite is always
        # True on an integer dtype, so that call could never fire. The non-finite
        # case is already caught by the sum() above, on the float. What CAN still
        # go wrong after the cast is a precision COLLAPSE — every value finite,
        # dynamic range gone, a flat grey face — which is how GFPGAN's FP16
        # engine failed. This is a 512 FP16 graph, so it gets that check.
        if looks_collapsed(restored, src):
            print("[GPEN Realistic] output collapsed (flat) — using unenhanced frame")
            return sized(temp_frame, input_size)

        try:
            chroma = float(os.environ.get('ROOP_GPENR_CHROMA', '') or 0.0)
        except ValueError:
            chroma = 0.0
        out = self._keep_source_colour(restored, src, chroma)

        with self._lock:
            self._faces += 1
        return sized(out, input_size)

    # ── reporting ────────────────────────────────────────────────────────────
    def cost_summary(self):
        with self._lock:
            f = self._faces
        if not f:
            return None
        return (f"[GPEN Realistic] {f} faces at {self.size} "
                f"(GPEN luminance, swapper chrominance"
                f"{', pooled' if self.pool is not None else ''})")
