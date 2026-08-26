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

4. PALE SKIN — fixed 2026-08-24 by borrowing GPEN Realistic's colour path.

   Reported by the user as UltraMax's skin looking pale where GPEN Realistic
   and GPEN 256 Pro do not. It is real, it is entirely colour, and the reason
   those two do not have it is that they already drop the network's chrominance
   and keep the swapper's. Measured against the crop the restorer was handed,
   5 frames of s1.mp4, face-centre window (tests/diag_ultramax_cost_and_colour.py):

       enhancer          chroma drift   dLAB-a   dLAB-b     dL   saturation
       UltraMax (was)            2.51    -0.96    +0.35   +2.22       x0.958
       GPEN Realistic            0.31    -0.11    -0.08   +4.06       x0.966
       GPEN 256 Pro              0.38    -0.07    -0.00   +1.54       x1.011

   Less red, brighter, less saturated — that IS pale, and it is CodeFormer's
   own bias rather than anything this file added. `luma_only_recolour` keeps
   the network's luminance and puts the crop's chrominance back under it, for
   0.27 ms and with no effect on detail whatsoever, because a luminance-only
   edit preserves every high-frequency variation the network drew.

   NOTE this makes UltraMax no longer bit-identical to `Codeformer (fp16)`.
   That equality was a property worth stating while the only difference was a
   filter measured to do nothing; it is not worth keeping a reported defect for.
   `ROOP_ULTRAMAX_CHROMA=1` restores CodeFormer's own colour exactly, and
   `tests/bench_ultramax_vs_codeformer.py` sets it to assert the equality still
   holds through the lean host path.

5. SPEED — where it actually goes, and what does NOT help.

   Measured per face, 256 crop in, RTX 4070 / TensorRT, idle GPU
   (tests/diag_ultramax_cost_and_colour.py):

       resize 256->512 (CUBIC)     0.31 ms    0.9%
       pre, LUT gather             2.10 ms    6.3%
       INFER (the network)        25.46 ms   76.0%
       post, CHW->HWC f32          1.54 ms    4.6%
       finite + convertScaleAbs    1.11 ms    3.3%
       looks_collapsed             3.80 ms   11.3%   <- fixed, see below
       sized() back to 256         0.07 ms    0.2%
       Run() total                33.51 ms

   `looks_collapsed` cost more than the rest of the host path put together, on
   a guard that fires approximately never: it made two float32 copies of a
   512x512x3 image per face. `cv2.meanStdDev` plus the parallel-axis theorem is
   EXACT and 11.5x cheaper (3.75 -> 0.33 ms, verified to agree on 100 pairs
   including degenerate ones). It applies equally to GPEN Realistic, GPEN 256
   Pro and GFPGAN, which share it. After:

       looks_collapsed             0.33 ms    1.1%   (was 3.80 / 11.3%)
       Run() total                30.03 ms           (was 33.51, -10.4%)

   AND THE STAGE IS NOW AT THE NETWORK'S OWN FLOOR. Under the production shape
   -- 10 worker threads, pool 2 -- the same harness that measured the pool
   curve reads 24.95 and 24.91 ms/call wall across two runs, against 38.4
   faces/s (26.02 ms) before. 40.1 faces/s is 24.9 ms/face, and the network
   alone is 25.5: there is no host time left to find, only the network.

   NOT CLAIMED: a render-clock number. This is a STAGE measurement, and this
   pipeline has produced three stage-level wins in a row that measured neutral
   end to end (stabilizer rounds, temporal detection, det_size). Removing host
   time from a GPU-bound pipeline usually does not move the clock. Anyone
   quoting a wall-clock figure for this needs a counterbalanced A/B first.

   TWO THINGS MEASURED AND REJECTED, so they are not re-attempted:

     - A DEEPER TENSORRT POOL DOES NOT HELP. The live ROOP_PROFILE reads
       enhance = 61.94 ms/call against this 33.5, and the obvious reading is
       that 10 worker threads are queueing on a 2-deep pool. They are, and
       widening it makes things WORSE, because the queue is not the constraint —
       one context already saturates the card. 10 threads, 240 faces
       (tests/diag_ultramax_pool_scaling.py):

           contexts 1: 40.0 faces/s   +1136 MB VRAM
           contexts 2: 38.4 faces/s   +1519 MB
           contexts 3: 37.4 faces/s   +1927 MB
           contexts 4: 36.6 faces/s   +2704 MB
           contexts 6: 36.0 faces/s   +3748 MB

       Monotonically down. 40 faces/s is 25 ms/face, i.e. exactly the network's
       own time — the GPU is the floor and extra contexts buy only VRAM and
       scheduling overhead. This is the same lesson as the stabilizer rounds and
       temporal detection: the pipeline is GPU-BOUND, and a stage's share of
       summed thread time is not a speedup budget.

     - The pre and post host paths have no meaningful slack left. Four
       alternative spellings of the pre gather (gather-then-transpose,
       cv2.split + gather, cv2.split + cv2.LUT) all measured within 5% of the
       current one or worse, and the fastest post variant (flat f32 cast plus
       per-plane convertScaleAbs, bit-identical) saves 0.17 ms of 1.88. Neither
       is worth the churn. NOTE the post path must be benched on FLOAT16 — this
       graph's output is fp16, and an f32 test array makes cv2.merge look like a
       win it is not.

   So the enhance stage's 25.5 ms/face of GPU work is the network, and there is
   no padding here to reclaim the way there was in the detector's 640 canvas.

Knobs, for re-measuring rather than for shipping a different default:
    ROOP_ULTRAMAX_CHROMA          0 = the swapper's colour (default),
                                  1 = CodeFormer's own (the old pale output)
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
from roop.processors.enhance_common import (looks_collapsed, sized, exclusive,
                                            luma_only_recolour)
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
    # Every session call goes through `exclusive()`, so no context of
    # this processor's is ever entered twice at once -- the only guarantee
    # ProcessMgr's enhance-stage lock provides. Declaring it lets that
    # stage skip the lock, so this class's HOST work stops serialising
    # against every other worker thread. See enhance_common.exclusive.
    self_excluding = True
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

    # How much of CODEFORMER'S OWN colour to keep. 0 = none of it: the network
    # supplies luminance and the swapper's crop supplies chrominance, which is
    # what both GPEN processors already do and what stops the skin reading pale.
    # See the "PALE SKIN" section of the module docstring for the measurement.
    # A knob defaulting to the old behaviour is the old behaviour for everyone,
    # so this defaults to the fix and exists for re-measuring.
    _CHROMA = 0.0

    _warned_colour = False

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

        model_path = resolve_relative_path('../models/CodeFormer/codeformer.fp16.onnx')
        # This export trips ORT's SimplifiedLayerNormFusion at ORT_ENABLE_ALL and
        # the session then fails to build at all — but only on the CPU provider,
        # so it would look like an enhancer that works right up until someone
        # falls back to CPU. EXTENDED builds everywhere. Same reasoning as
        # Enhance_CodeFormer; keep the two in step.
        from roop.utilities import get_onnx_session_options
        opts = get_onnx_session_options(optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)

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
                extras = [_build(i + 1) for i in range(n - 1)]
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

        # An independent context per worker when pooled, else this processor's
        # own lock over its single session -- either way exclusive, and no wider
        # than the model call. Input/output NAMES are identical across sessions
        # built from one ONNX, so the cached lists stay valid for any lease.
        with exclusive(self.pool, self._session_lock,
                       (self.session, self.io_binding)) as (sess, iob):
            ort_outs = _infer(sess, iob)

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
            return sized(temp_frame, input_size)

        # maximum() first so convertScaleAbs' abs() is a no-op on the low end;
        # its saturate_cast handles the high end. Two passes, both in C++.
        np.maximum(hwc, -1.0, out=hwc)
        restored = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        # NOT `is_usable` here: this array is uint8 and np.isfinite is always
        # True on an integer dtype, so that call could never fire. Non-finite is
        # already caught by the sum() above, on the float. The failure that
        # survives the cast is a precision COLLAPSE — finite values, dynamic
        # range gone — which is how GFPGAN's FP16 engine failed, and this is a
        # half-precision graph too.
        if looks_collapsed(restored, src512):
            print("[UltraMax] output collapsed (flat) — using unenhanced frame")
            return sized(temp_frame, input_size)

        # ── PALE SKIN ───────────────────────────────────────────────────
        # CodeFormer's own chrominance is dropped in favour of the crop's. This
        # is the same operator GPEN Realistic and GPEN 256 Pro have always run,
        # and its absence here is the whole of the reported difference: those
        # two drift 0.31 and 0.38 against the input while this drifted 2.51,
        # less red (LAB a -0.96), brighter (L +2.22) and desaturated (x0.958).
        # It cannot cost detail — a luminance-only edit keeps every
        # high-frequency variation the network drew — and it is measured at
        # 0.27 ms against the 25.5 ms the network itself costs.
        try:
            restored = luma_only_recolour(
                restored, src512,
                _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA))
        except cv2.error as e:
            # Say so once. A colour fix that silently no-ops leaves a plausible
            # image carrying the exact pale cast it exists to remove.
            if not Enhance_UltraMax._warned_colour:
                Enhance_UltraMax._warned_colour = True
                print(f"[UltraMax] colour fix skipped: {e}", flush=True)

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
        c = _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA)
        colour = ("swapper chrominance" if c <= 0.0
                  else f"chrominance {c:g} toward CodeFormer's own")
        return (f"[UltraMax] {f} faces restored (codeformer.fp16, lean host "
                f"path, {colour}); texture restored on {t}")
