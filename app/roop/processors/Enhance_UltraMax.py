"""UltraMax — CodeFormer's restoration on a lean host path.

WHAT THIS IS, STATED HONESTLY. UltraMax runs `codeformer.fp16.onnx` — the same
weights as `Codeformer (fp16)`. It is not a different network and it does not
claim to be. What it changes is everything AROUND the network, and that is where
its win lives. Measured per face on an RTX 4070 / TensorRT, 512 crop:

    the network itself, fresh io_binding per call        24.98 ms
    the network itself, io_binding reused                23.50 ms
    Enhance_CodeFormer.Run() end to end                  36.33 ms
      its host pre-processing                             3.86 ms
      its host post-processing                            5.63 ms
    the OLD UltraMax refinement filter, on top of that   10.13 ms

So the old build cost ~46.5 ms/face against CodeFormer's 36.3 — the 13% slowdown
measured on 2026-08-23 — and 10.1 ms of it bought an unsharp mask that the user
reported as "too sharp, blurry on the eyes". Both halves of that are fixed here.

1. LEAN HOST PATH (the speed).
   - Pre-processing is a single 256-entry LUT gather straight into the model's
     dtype (2.29 ms), instead of astype -> /255 -> -0.5 -> /0.5 -> transpose ->
     astype, which is five full passes over 786k floats (3.86 ms).
   - Post-processing is one contiguous copy plus one saturating C++ scale
     (1.66 ms), instead of cast -> isfinite -> transpose -> clip -> rescale ->
     cvtColor -> multiply -> round -> cast (5.63 ms).
   - Each pool slot owns its io_binding for the life of the run instead of
     allocating one per call (1.5 ms), the way Enhance_RestoreFormerPPlus does.
   Net: ~27 ms of network + host against CodeFormer's 36.3 ms, same weights,
   same fidelity, comparable output.

   None of that is magic and none of it is exclusive: porting the same pre/post
   into Enhance_CodeFormer would close most of the gap. It is kept here rather
   than pushed there because Enhance_CodeFormer.Run() is the reference
   implementation several saved benchmarks are calibrated against.

2. TEXTURE RESTORE — MEASURED AND TURNED OFF (default gain 0).

   Read this before re-enabling it. Graded on rendered frames of s1.mp4, paired
   over 102 frames, against the ORIGINAL footage's own skin (cheeks and forehead
   placed from the plate's landmarks, so the measurement window cannot move with
   the treatment):

       skin texture vs the plate   156.8% -> 156.7%   paired t -0.7   (nothing)
       temporal flicker            8.3664 -> 8.3752   paired t +4.6   (worse)
       identity margin             0.4151 -> 0.4166   paired t +2.0   (a hair)

   It costs 2.49 ms/face — 8% of this processor's budget — to move texture by an
   amount indistinguishable from zero. So it is off, and turning it off makes
   UltraMax bit-identical to `Codeformer (fp16)` and FASTER still: 28.71 ms
   against 35.18, i.e. 1.23x rather than 1.127x.

   THE PREMISE WAS ALSO WRONG, and that is the part worth keeping. The filter
   was built to close a "skin gap" — an earlier measurement said the swapped
   face carried only 36% of the footage's skin texture. That number came from a
   mask defined as "the flattest 45% of the RENDERED frame", which selects the
   pixels each treatment touched LEAST and so cancels the very effect it is
   measuring. With the window anchored geometrically instead, the swapped face
   carries **~155% of the plate's** skin micro-texture: it is OVER-textured, not
   under-textured, which is consistent with the user's "too sharp" report and
   leaves nothing for a restore to restore. `tests/sweep_detail_transfer.py`
   carries the same finding for the merger's detail-transfer stage.

   The implementation below is kept, unused, because it is correct code for a
   job this pipeline does not have — and because the three mask definitions that
   disagreed by 34% / 155% / 500% on the same footage are the actual lesson.

3. WHAT THE FILTER DOES, if it is ever switched back on.
   CodeFormer's failure mode is waxy skin: the codebook prior draws clean
   structure and flattens dermal micro-texture. The OLD answer here was a
   multi-scale unsharp (0.45 of the fine detail plus 0.20 of a sigma 1.0-2.5
   BAND) followed by CLAHE. Two things were wrong with that. It ADDS edge energy
   rather than RESTORING skin, so it reads as sharpened rather than
   photographed — and the medium band is exactly the operator that turns an
   infraorbital crease into a second lower eyelid, which is the "blurry eyes"
   report.

   `_restore_texture` instead re-injects the high-frequency luminance the
   restorer just erased, taken from its OWN input, and only where the restorer
   drew flat skin. Everything CodeFormer rendered as structure — iris rims, lash
   lines, lid creases, lip margins, nostrils, hairline — is left exactly as the
   codebook produced it, because that output is already the sharpest source in
   the pipeline and every filter applied over it has made it worse.

   NOTE FOR ANYONE REMEASURING: do not grade this with Laplacian variance. That
   instrument counts added edge energy, and it has already endorsed one build
   here (UltraMax v1, "4.39x GPEN-256") that the user reported as plastic, and a
   second (the filter above) reported as over-sharp. Grade it on the footage.

Knobs, for re-measuring rather than for shipping a different default:
    ROOP_ULTRAMAX_TEXTURE         texture-restore gain, default 0 (OFF)
    ROOP_ULTRAMAX_TEXTURE_SIGMA   the band it is taken from, default 2.5 at 512
"""

import os
import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.typing import Face, FaceSet, Frame
from roop.processors.enhance_common import is_usable, sized
from roop.utilities import resolve_relative_path
from roop import session_pool


def _env_float(name, default):
    try:
        v = os.environ.get(name)
        return default if v is None or v == '' else float(v)
    except (TypeError, ValueError):
        return default


class Enhance_UltraMax:
    processorname = 'ultramax'
    type = 'enhance'
    # Same prior as Enhance_CodeFormer, because it is the same weights: FFHQ
    # aligned 512. ProcessMgr re-warps the swap crop into this space when
    # `enhancer_align` is on. See Enhance_CodeFormer.model_template for the
    # per-swapper mismatch table.
    model_template = 'ffhq_512'

    _SIZE = 512

    # Mid-tone weight for the texture restore's exposure gate: full weight where
    # skin lives, zero in a specular highlight or a crushed shadow, where an
    # injected residual could only read as noise. A table because it is indexed
    # by a uint8 luminance, and evaluating the sine per pixel instead measured
    # as the single most expensive line in the filter.
    _EXPOSURE_LUT = np.clip(
        np.sin(np.pi * (np.arange(256, dtype=np.float32) / 255.0)), 0.0, 1.0)

    _warned_texture = False

    # THE SIGMA IS THE WHOLE BALLGAME, and getting it wrong is silent.
    # This filter runs on the 512 template, but the face is pasted back into the
    # frame at whatever size it really is -- ~300 px in s1.mp4. A residual
    # extracted at sigma 1.1 therefore lands at sigma 0.65 after that downscale,
    # below Nyquist, and is resampled straight back out: measured end to end,
    # UltraMax at sigma 1.1 ran on all 2195 faces of s1 and moved rendered skin
    # texture by 0.4%. `apply_detail_transfer` already had this right -- its
    # sigma is `max(1.0, w/256)`, i.e. 2.0 at 512 -- and that is why IT works.
    # `tests/calibrate_ultramax_texture.py` is the measurement.
    _SIGMA = 2.5

    # Texture-restore gain. DEFAULT 0 — OFF — because measured properly it does
    # nothing. See the "MEASURED AND TURNED OFF" section of the module
    # docstring; this is a negative result kept as code so the next person does
    # not rebuild it.
    _TEXTURE_GAIN = 0.0

    def __init__(self):
        self.plugin_options = None
        self.devicename = None
        self.session = None
        self.io_binding = None
        self.pool = None          # SessionPool of (session, io_binding) slots
        self.model_inputs = None
        self.model_outputs = None
        self.in_dtype = np.float32
        self._lut = None
        self._faces = 0
        self._textured = 0
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options.get("devicename") != plugin_options.get("devicename"):
                self.Release()
        self.plugin_options = plugin_options
        self.devicename = plugin_options["devicename"].replace('mps', 'cpu')

        if self.session is not None:
            return

        model_path = resolve_relative_path('../models/CodeFormer/codeformer.fp16.onnx')
        # This export trips ORT's SimplifiedLayerNormFusion at ORT_ENABLE_ALL and
        # the session then fails to build at all — but only on the CPU provider,
        # so it would look like an enhancer that works right up until someone
        # falls back to CPU. EXTENDED builds everywhere. Same reasoning as
        # Enhance_CodeFormer; keep the two in step.
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        def _build(_i=0):
            sess = onnxruntime.InferenceSession(
                model_path, opts, providers=roop.globals.execution_providers)
            # One io_binding per slot, held for the life of the run. A binding is
            # not shareable across threads, and rebuilding it per call measured
            # 1.5 ms of the 25 ms budget. The OUTPUT binding never changes, so it
            # is set once here; only the two inputs are rebound per call.
            iob = sess.io_binding()
            iob.bind_output(sess.get_outputs()[0].name, self.devicename)
            return (sess, iob)

        self.session, self.io_binding = _build()
        self.model_inputs = self.session.get_inputs()
        self.model_outputs = self.session.get_outputs()
        self.in_dtype = (np.float16 if 'float16' in self.model_inputs[0].type
                         else np.float32)

        # uint8 -> the model's own dtype, normalised to [-1, 1], in one gather.
        self._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0).astype(self.in_dtype)

        # Optional TensorRT multi-context pool. Without it `_gpu_guard` hands the
        # enhance stage the GLOBAL GPU lock (it only exempts a processor that
        # owns a pool), which makes the single most expensive stage in a render a
        # one-thread-at-a-time queue. Measured VRAM for this net: 530 MB per
        # extra context. `pool_size` caps it, because on a 12GB card realswap's
        # two nets, RealityUX and the detector pools are already resident and a
        # fourth context collapses the render to 0.2 fps.
        if session_pool.pooling_enabled():
            n = session_pool.pool_size()
            cap = plugin_options.get('pool_size')
            if cap:
                n = max(1, min(int(n), int(cap)))
            extras = []
            try:
                extras = [_build(i) for i in range(n - 1)]
                primary = (self.session, self.io_binding)
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([primary] + extras): _e[i], n)
            except Exception as e:
                extras.clear()
                self.pool = None
                print(f"[UltraMax] multi-context pool unavailable ({e}); "
                      f"falling back to one session behind the GPU lock")

    def Release(self):
        line = self.cost_summary()
        if line:
            print(line, flush=True)

        # Pool first: it holds the extra sessions AND a reference to the primary
        # pair, so dropping the primary while the pool still owns it would leave
        # live TensorRT contexts with no owner to free them.
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        self.io_binding = None
        self.session = None
        self.model_inputs = None
        self.model_outputs = None
        self._lut = None

    # ── texture restore ──────────────────────────────────────────────────────
    @staticmethod
    def _restore_texture(restored, source, gain, sigma=_SIGMA):
        """Put back the dermal micro-texture the restorer flattened.

        `restored` = CodeFormer's 512 output, `source` = the crop it was handed,
        at the same 512. Both uint8 BGR.

        The residual is taken from the SOURCE, so what lands on the skin is
        texture that was photographed rather than texture that was synthesised —
        which is the whole difference between reading as skin and reading as a
        sharpened render. It is gated two ways:

          structure gate  1 where the RESTORED face is flat, falling to 0 on
                          anything the codebook drew as an edge. Eyes, lashes,
                          brows, lip margins, nostrils and the hairline
                          therefore pass through completely untouched. This is
                          the gate the old filter did not have, and its absence
                          is what printed a second crease under the eye.

          exposure gate   a mid-tone weight, so nothing is injected into a
                          specular highlight or a crushed shadow, where it could
                          only read as noise.

        Self-limiting by construction: a source crop that carries no real
        texture has a near-zero residual, so this adds nothing rather than
        inventing something.

        COST. The whole point of this processor is that it is not slower than
        the net it wraps, so this has a budget: it must stay well under the
        6.1 ms the lean host path saves. The straightforward numpy spelling
        measured 6.0 ms at 512 and spent it entirely on temporaries — a
        `np.sin` over 262k floats and a dozen intermediate arrays. Three
        changes bring it to ~2 ms with no visible difference:

          - the exposure gate is a 256-entry LUT gather off the uint8
            luminance, not a transcendental evaluated per pixel;
          - the structure gate is built at HALF resolution. It is a blurred
            edge map, so it has no detail to lose, and it costs a quarter as
            much;
          - the residual is carried as int16 rather than float32 from the
            multiply onward, which halves the bytes moved through the last
            three passes — the ones that touch all three colour channels.
        """
        if gain <= 0.0:
            return restored
        try:
            g_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            g_res = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
            h, w_px = g_res.shape[:2]

            # Real high-frequency luminance from the input, at FULL resolution —
            # this is the texture itself, so it is the one thing that cannot be
            # approximated. The clamp scales with sigma because a wider band
            # legitimately carries a larger swing; it is there to stop a
            # compression block or a specular speckle punching through, not to
            # cap the texture.
            sigma = max(0.6, float(sigma))
            hf = cv2.subtract(g_src, cv2.GaussianBlur(g_src, (0, 0), sigmaX=sigma),
                              dtype=cv2.CV_32F)
            lim = 11.0 * sigma
            np.clip(hf, -lim, lim, out=hf)

            # BOTH gates are built at half resolution and upsampled once. One is
            # a blurred edge map and the other is a smooth function of exposure,
            # so neither has detail to lose — and the pair measured 1.6 ms at
            # full resolution against 0.7 ms here, on a filter whose entire
            # budget is the 6.1 ms the lean host path saves.
            small = cv2.resize(g_res, (w_px // 2, h // 2),
                               interpolation=cv2.INTER_AREA)

            # Structure gate. Laplacian rather than two Sobels: one pass, and the
            # second derivative is what "the codebook drew a line here" actually
            # looks like.
            edge = cv2.GaussianBlur(
                np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), (0, 0),
                sigmaX=1.0)
            edge -= 4.0
            edge *= (1.0 / 16.0)
            np.clip(edge, 0.0, 1.0, out=edge)
            np.subtract(1.0, edge, out=edge)

            # Exposure gate, off the restored luminance, as a table lookup.
            cv2.multiply(edge, Enhance_UltraMax._EXPOSURE_LUT[small], dst=edge)
            gate = cv2.resize(edge, (w_px, h), interpolation=cv2.INTER_LINEAR)

            # int16 from here on: the last three passes touch all three colour
            # channels, so halving the element size is worth more than it looks.
            delta = cv2.multiply(hf, gate, scale=float(gain), dtype=cv2.CV_16S)
            # A luminance-only edit is the same signed offset on every channel.
            # `merge` and not `cvtColor(GRAY2BGR)`: cvtColor rejects CV_16S
            # outright, and because the whole body is guarded it did so
            # silently — the filter became a no-op that still returned a
            # plausible image. Kept as a comment because that is exactly how it
            # would come back.
            return cv2.add(restored, cv2.merge((delta, delta, delta)),
                           dtype=cv2.CV_8U)
        except cv2.error as e:
            # Never take a render down over a look filter — but say so, or the
            # failure above repeats and nothing in the output ever shows it.
            if not Enhance_UltraMax._warned_texture:
                Enhance_UltraMax._warned_texture = True
                print(f"[UltraMax] texture restore skipped: {e}", flush=True)
            return restored

    # ── run ──────────────────────────────────────────────────────────────────
    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1

        input_size = temp_frame.shape[1]
        S = self._SIZE

        # The 512 source is needed twice — as the network's input and as the
        # texture reference — so it is built once.
        if temp_frame.shape[0] != S or temp_frame.shape[1] != S:
            src512 = cv2.resize(temp_frame, (S, S), interpolation=cv2.INTER_CUBIC)
        else:
            src512 = temp_frame

        # One gather: uint8 BGR HWC -> model dtype RGB CHW in [-1, 1]. Fancy
        # indexing returns a fresh C-contiguous array, so the [None] that follows
        # is a free view and no extra copy is made.
        x = self._lut[src512.transpose(2, 0, 1)[::-1]][None]
        w_fid = np.array([getattr(roop.globals, 'codeformer_fidelity', 0.5)],
                         dtype=np.float64)

        in0, in1 = self.model_inputs[0].name, self.model_inputs[1].name

        def _infer(sess, iob):
            iob.bind_cpu_input(in0, x)
            iob.bind_cpu_input(in1, w_fid)
            sess.run_with_iobinding(iob)
            return iob.copy_outputs_to_cpu()

        if self.pool is not None:
            # An independent context per worker, so this thread runs concurrently
            # with the others. Input/output NAMES are identical across sessions
            # built from one ONNX, so the cached lists stay valid for any lease.
            with self.pool.lease() as (sess, iob):
                ort_outs = _infer(sess, iob)
        else:
            ort_outs = _infer(self.session, self.io_binding)

        # CHW RGB -> HWC BGR, contiguous and float32, in one copy.
        hwc = np.ascontiguousarray(ort_outs[0][0][::-1].transpose(1, 2, 0),
                                   dtype=np.float32)
        del ort_outs

        # np.clip does NOT remove NaN and uint8(NaN) is 0, so one overflowed
        # value paints a pixel black and a saturated graph paints the whole FACE
        # black, silently — and this is a half-precision graph. A single sum
        # propagates both NaN and +-inf for a fraction of isfinite().all()'s
        # cost and allocates nothing. See enhance_common.is_usable.
        if not np.isfinite(hwc.sum()):
            print("[UltraMax] non-finite output — using unenhanced frame "
                  "(FP16 overflow?)")
            return sized(src512, input_size)

        # maximum() first so convertScaleAbs' abs() is a no-op on the low end;
        # its saturate_cast handles the high end. Two passes, both in C++.
        np.maximum(hwc, -1.0, out=hwc)
        restored = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        if not is_usable(restored):
            return sized(src512, input_size)

        gain = _env_float('ROOP_ULTRAMAX_TEXTURE', self._TEXTURE_GAIN)
        if gain > 0.0:
            restored = self._restore_texture(
                restored, src512, gain,
                _env_float('ROOP_ULTRAMAX_TEXTURE_SIGMA', self._SIGMA))
            with self._lock:
                self._textured += 1

        with self._lock:
            self._faces += 1
        return sized(restored, input_size)

    # ── reporting ────────────────────────────────────────────────────────────
    def cost_summary(self):
        with self._lock:
            f, t = self._faces, self._textured
        if not f:
            return None
        return (f"[UltraMax] {f} faces restored (codeformer.fp16, lean host "
                f"path); texture restored on {t}")
