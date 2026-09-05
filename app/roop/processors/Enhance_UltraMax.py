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
                                            luma_only_recolour,
                                            luma_only_recolour_tensor,
                                            ort_cuda_output_to_torch)
from roop.processors.frequency_split import (frequency_split,
                                             frequency_split_luma,
                                             reinhard_lab)
from roop.utilities import resolve_relative_path
from roop import session_pool
from roop.precision_policy import providers_for

try:
    import torch
    import torch.nn.functional as _F
    _TORCH_CUDA = torch.cuda.is_available()
except (ImportError, AttributeError):
    _TORCH_CUDA = False


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

    # UltraMax receives the already-composited RealSwap crop.  Re-restoring the
    # periocular pixels makes CodeFormer's newly drawn lid/iris edge compete
    # with RealSwap's hififace eyelash band, which is the visible double-eye
    # halo reported on close faces.  Keep that registered input structure and
    # let UltraMax work on cheeks, nose, lips and skin instead.  The mask is
    # deliberately the same outer eye geometry as RealSwap's band, with a
    # feathered edge so it cannot create a second boundary of its own.
    # The source crop remains the geometry base throughout the complete eye
    # band.  UltraMax may donate only detail whose gradient agrees with that
    # base; directly mixing its eye pixels brings back the double-eye halo.
    _PROTECT_EYE_OUTER_X = 0.44
    _PROTECT_EYE_OUTER_Y = 0.30
    _PROTECT_EYE_CORE_X = 0.13
    _PROTECT_EYE_CORE_Y = 0.060
    _PROTECT_EYE_LIFT = 0.03
    _PROTECT_EYE_FEATHER = 0.08
    # Keep the swapped crop as the geometry anchor, but let the actual
    # UltraMax output contribute enough aligned aperture detail to remain
    # visibly enhanced. The previous 0.42 core mix / 0.50 outer detail gain
    # made the eye band look like an unenhanced RealSwap crop.
    _PROTECT_EYE_SHARPEN = 0.70
    _PROTECT_EYE_DETAIL_GAIN = 0.85
    _PROTECT_EYE_CORE_DETAIL_GAIN = 1.50
    _PROTECT_EYE_CORE_MIX = 0.68
    _EYE_BALANCE_TRIGGER = 1.12
    _EYE_BALANCE_MAX = 0.55
    _STRUCTURE_SHARPEN = 0.18
    # Registered FFHQ eye geometry does not change from face to face.  Holding
    # two 512-square float masks costs 2 MiB/device and avoids rebuilding and
    # blurring them in every restorer invocation.
    _eye_mask_cache = {}
    _rebalance_mask_cache = {}
    _gaussian_1d_cache = {}

    # ── DUAL-STREAM FREQUENCY-SPLIT ENGINE ───────────────────────────────────
    # UltraMax's problem against CodeFormer was never a blend: it RUNS
    # codeformer.fp16, so single-stream it cannot beat CodeFormer on detail by
    # construction -- the network is the ceiling and everything here is host
    # work around it. Measured, that is exactly what happened: identity moved
    # not at all (paired t = 0.4 over 4703 frames) and the only difference was
    # a filter, since moved out to `MergerMixin.apply_clarity`.
    #
    # The way past that ceiling is a SECOND NETWORK, chosen on this repo's own
    # measurements. High-frequency std carried through to the paste:
    #
    #     swap input 2.67 | GPEN-256 2.82 | CodeFormer-512 4.11 | GPEN-512 5.14
    #
    # GPEN-512 synthesises ~25% more detail than CodeFormer; CodeFormer's
    # discrete codebook is the better structure prior. So: low band from
    # CodeFormer (geometry -- lids, iris rims, tooth boundaries), high band
    # from GPEN-512 (pores, lashes, lip texture), split by a guided filter so
    # the high band is bounded by its own edges and cannot carry a silhouette.
    # See `roop.processors.frequency_split`.
    #
    # MEASURED ON 2026-09-04, 40 aligned crops of s1.mp4, RTX 4070 / TensorRT,
    # via `tests/test_restorers.py`. READ THIS BEFORE ENABLING IT:
    #
    #   arm                        ms/face   edge/plate   chroma drift
    #   Codeformer (fp16)             41.0        1.291           2.04
    #   UltraMax single               40.1        1.315           0.54
    #   GPEN Realistic                30.7        1.262           0.52
    #   dual luma  gain 1.00         125.6        1.199           0.50
    #   dual luma  gain 1.25         125.0        1.337           0.54
    #   dual bgr   gain 1.00         150.0        1.213           1.63
    #   dual bgr   gain 1.25         150.1        1.345           1.79
    #
    # WHAT IT DELIVERS: the best colour in the table (0.50, beating both this
    # processor's own single-stream path and GPEN Realistic), and at gain 1.00
    # the edge ratio closest to the reference of any 512-class restorer here.
    #
    # WHAT IT DOES NOT DELIVER, stated plainly: a demonstrated DETAIL win over
    # CodeFormer. At gain 1.00 it is SOFTER than CodeFormer (1.199 vs 1.291),
    # and at 1.25 it is level with the single stream rather than ahead. The
    # premise -- GPEN-512 carrying ~25% more high-frequency than CodeFormer --
    # was measured on a swapper crop, and this harness feeds ALIGNED PLATE
    # crops, on which it did not reproduce. Nobody should quote this engine as
    # "beats CodeFormer" until it has been graded on real swapper output.
    #
    # AND IT COSTS 3.1x (125 ms against 40). Two networks are only 70 ms of
    # that; the rest is the guided filter. So it defaults OFF, and would need a
    # counterbalanced end-to-end render before it could default on -- on a 12GB
    # card it also puts a second resident 512 graph next to realswap's two nets
    # and RealityUX, which is the residency that took a render from 10.50 to
    # 6.60 fps the last time it was measured. `ROOP_ULTRAMAX_DUAL=1` enables.
    _DUAL_STREAM = False
    _DUAL_RADIUS = 8
    _DUAL_EPS = 0.04
    _DUAL_GAIN = 1.25
    # High-band clamp in 0-255 levels, before the gain. Two different networks
    # on one crop can place a lash line a pixel apart, and re-adding that
    # disagreement at 1.25 reads as a double edge -- the same failure the old
    # UltraMax unsharp produced and the same order of magnitude as
    # `apply_clarity`'s +-18. Set to 0/None to disable and see it.
    _DUAL_CLAMP = 24.0
    # Luminance weight for the Reinhard transfer. Chrominance is matched in
    # full (that is the colour cast this removes); L is matched at half, since
    # full L matching drags contrast back toward the degraded crop and returns
    # part of what the restorer just recovered.
    _DUAL_L_WEIGHT = 0.5
    # 'luma' splits the luminance plane and takes chrominance from the crop;
    # 'bgr' is the original 3-channel split plus a Reinhard transfer, kept
    # only because it is the arm the negative result was measured on.
    _DUAL_MODE = 'luma'
    _DUAL_MODELS = {
        512: ('GPEN-BFR-512.onnx',
              'https://huggingface.co/countfloyd/deepfake/resolve/main/'
              'GPEN-BFR-512.onnx'),
    }

    _warned_colour = False
    _warned_dual = False

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
        self._dual = 0
        # Detail stream (GPEN-512). Built only when the dual-stream engine is
        # on, so a user who never enables it pays neither the VRAM nor the
        # engine build.
        self._detail_session = None
        self._detail_iob = None
        self._detail_pool = None
        self._detail_in = None
        self._detail_lut = None
        self._detail_lock = threading.Lock()
        # ``False`` means this ORT build/provider rejected binding its output
        # directly into a PyTorch CUDA allocation; avoid retrying per face.
        self._cuda_iob_available = None
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
        session_providers, _precision = providers_for(
            'ultramax', roop.globals.execution_providers, model_path)

        def _build(_i=0):
            sess = onnxruntime.InferenceSession(
                model_path, opts, providers=session_providers)
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
            n = session_pool.pool_size(
                model_key='enhancer:ultramax', input_shape=(1, 3, 512, 512))
            cap = plugin_options.get('pool_size')
            if cap:
                n = max(1, min(int(n), int(cap)))
            extras = []
            try:
                extras = [_build(i + 1) for i in range(n - 1)]
                primary = (self.session, self.io_binding)
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([primary] + extras): _e[i], n,
                    model_key='enhancer:ultramax', input_shape=(1, 3, 512, 512))
            except Exception as e:
                extras.clear()
                self.pool = None
                print(f"[UltraMax] multi-context pool unavailable ({e}); "
                      f"falling back to one session behind the GPU lock")

        # Detail stream last, so a failure here cannot leave the structural
        # session half-built. Reported rather than swallowed: an engine that
        # silently runs as plain CodeFormer while the UI says UltraMax is the
        # failure mode this project keeps having to hunt down.
        if self.dual_enabled() and self._detail_session is None:
            try:
                self._init_detail_stream(plugin_options)
            except Exception as e:
                self._detail_session = None
                print(f"[UltraMax] dual-stream detail network unavailable "
                      f"({e}); running single-stream CodeFormer", flush=True)

    # ── dual-stream detail network ───────────────────────────────────────────
    @classmethod
    def dual_enabled(cls):
        """Whether the frequency-split engine is active for this run."""
        v = os.environ.get('ROOP_ULTRAMAX_DUAL')
        if v is None or v == '':
            return bool(cls._DUAL_STREAM)
        return v.strip().lower() in ('1', 'on', 'true', 'yes')

    def _init_detail_stream(self, plugin_options):
        """Build the GPEN-512 detail network.

        Deliberately a SEPARATE session from `Enhance_GPENRealistic`'s rather
        than a reference to it. That processor is instantiated by ProcessMgr
        only when the user selects it; reaching across to it would mean either
        depending on load order or constructing a second processor's whole
        lifecycle (Initialize/Release/pool/cost_summary) from inside this one.
        The weights are shared on disk and the TensorRT engine cache is keyed
        by model path, so the second session reuses the same built engine --
        the extra cost is the CONTEXT (measured ~1.5 GB for a GPEN-512 pool of
        2, i.e. why the pool below is capped hard), not a second build.
        """
        fname, url = self._DUAL_MODELS[512]
        model_dir = resolve_relative_path('../models')
        from roop.utilities import conditional_download, get_onnx_session_options
        conditional_download(model_dir, [url])
        model_path = os.path.join(model_dir, fname)
        opts = get_onnx_session_options()
        # No FP32 forcing: 512 is the classic GPEN weight and is FP16-stable.
        # `Enhance_GPEN` only forces FP32 at >=1024, where activations overflow
        # and paint a black face.
        providers, _precision = providers_for(
            'gpen_realistic', roop.globals.execution_providers, model_path)

        def _build_detail(_i=0):
            sess = onnxruntime.InferenceSession(model_path, opts,
                                                providers=providers)
            iob = sess.io_binding()
            iob.bind_output(sess.get_outputs()[0].name, self.devicename)
            return (sess, iob)

        self._detail_session, self._detail_iob = _build_detail()
        self._detail_in = self._detail_session.get_inputs()[0].name
        self._detail_lut = (np.arange(256, dtype=np.float32) / 127.5) - 1.0

        # Capped much harder than the CodeFormer pool. This is a SECOND
        # resident 512 graph on top of realswap's two nets, RealityUX and the
        # detector/detmask pools; `Enhance_GPENRealistic` already measured a
        # GPEN-512 pool of 2 at 3123 MiB and watched an uncapped one take a
        # 12GB card from 10.50 to 6.60 fps by paging. One context here unless
        # the card is genuinely large.
        if session_pool.pooling_enabled():
            n = session_pool.pool_size(model_key='enhancer:ultramax_detail',
                                       input_shape=(1, 3, 512, 512))
            cap = plugin_options.get('pool_size')
            if cap:
                n = max(1, min(int(n), int(cap)))
            try:
                gb = session_pool._detect_vram_gb()
            except Exception:
                gb = 0
            if 0 < gb < 15.5:
                n = 1
            else:
                n = min(n, 2)
            try:
                forced = int(os.environ.get('ROOP_ULTRAMAX_DUAL_POOL', '') or 0)
            except ValueError:
                forced = 0
            if forced:
                n = max(1, forced)
            if n > 1:
                extras = []
                try:
                    extras = [_build_detail(i + 1) for i in range(n - 1)]
                    primary = (self._detail_session, self._detail_iob)
                    self._detail_pool = session_pool.SessionPool(
                        lambda i, _e=([primary] + extras): _e[i], n,
                        model_key='enhancer:ultramax_detail',
                        input_shape=(1, 3, 512, 512))
                except Exception as e:
                    extras.clear()
                    self._detail_pool = None
                    print(f"[UltraMax] detail-stream pool unavailable ({e}); "
                          f"using one session behind the GPU lock")

    def _detail_stream(self, src512):
        """GPEN-512 restoration of the same crop, uint8 BGR 512, or None.

        Returns None rather than raising on any failure, so a detail stream
        that cannot run degrades this processor to its single-stream output
        instead of taking the render down -- but it SAYS SO once. A silent
        fallback here would leave UltraMax quietly running as plain CodeFormer
        while reporting the dual-stream engine, which is the exact instrument
        failure this project keeps getting caught by (four enhancers failing
        on 60 of 60 frames while the swap audit read 100%).
        """
        try:
            x = self._detail_lut[src512.transpose(2, 0, 1)[::-1]][None]
            with exclusive(self._detail_pool, self._detail_lock,
                           (self._detail_session, self._detail_iob)) as (s, iob):
                iob.bind_cpu_input(self._detail_in, x)
                s.run_with_iobinding(iob)
                outs = iob.copy_outputs_to_cpu()
            hwc = np.ascontiguousarray(outs[0][0][::-1].transpose(1, 2, 0),
                                       dtype=np.float32)
            del outs
            # Same two guards the single stream carries, for the same reasons:
            # np.clip keeps NaN and uint8(NaN) is 0, and a finite-but-flat
            # output is how GFPGAN's FP16 engine failed.
            if not np.isfinite(hwc.sum()):
                raise ValueError("non-finite detail output")
            np.maximum(hwc, -1.0, out=hwc)
            detail = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)
            if looks_collapsed(detail, src512):
                raise ValueError("detail output collapsed (flat)")
            return detail
        except Exception as e:
            if not Enhance_UltraMax._warned_dual:
                Enhance_UltraMax._warned_dual = True
                print(f"[UltraMax] detail stream unavailable ({e}); "
                      f"falling back to single-stream CodeFormer", flush=True)
            return None

    def Release(self):
        line = self.cost_summary()
        if line:
            print(line, flush=True)

        if self._detail_pool is not None:
            self._detail_pool.release()
            self._detail_pool = None
        self._detail_iob = None
        self._detail_session = None
        self._detail_lut = None

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
    @classmethod
    def _restore_texture(cls, restored, source, gain, sigma=_SIGMA):
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
        if _TORCH_CUDA:
            try:
                return cls._restore_texture_gpu(restored, source, gain, sigma)
            except Exception as e:
                if not Enhance_UltraMax._warned_texture:
                    Enhance_UltraMax._warned_texture = True
                    print(f"[UltraMax] GPU texture restore fallback to CPU: {e}", flush=True)
        return cls._restore_texture_cpu(restored, source, gain, sigma)

    @classmethod
    def _gaussian_kernel_2d_gpu(cls, sigma, device):
        radius = int(round(3 * sigma))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()
        k2d = k1d.view(-1, 1) @ k1d.view(1, -1)
        return k2d.view(1, 1, k2d.shape[0], k2d.shape[1]), radius

    @classmethod
    def _restore_texture_gpu(cls, restored, source, gain, sigma):
        device = torch.device('cuda')
        r_t = torch.from_numpy(restored).to(device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        s_t = torch.from_numpy(source).to(device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        g_src = 0.114 * s_t[:, 0:1] + 0.587 * s_t[:, 1:2] + 0.299 * s_t[:, 2:3]
        g_res = 0.114 * r_t[:, 0:1] + 0.587 * r_t[:, 1:2] + 0.299 * r_t[:, 2:3]

        k_sig, r_sig = cls._gaussian_kernel_2d_gpu(max(0.6, float(sigma)), device)
        padded_src = _F.pad(g_src, (r_sig, r_sig, r_sig, r_sig), mode='reflect')
        src_blur = _F.conv2d(padded_src, k_sig)
        hf = g_src - src_blur
        lim = 11.0 * max(0.6, float(sigma))
        hf = torch.clamp(hf, -lim, lim)

        small = _F.interpolate(g_res, scale_factor=0.5, mode='area')
        if not hasattr(cls, '_laplacian_k_gpu'):
            cls._laplacian_k_gpu = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            cls._exposure_lut_gpu = torch.tensor(Enhance_UltraMax._EXPOSURE_LUT, dtype=torch.float32, device=device)

        padded_small = _F.pad(small, (1, 1, 1, 1), mode='reflect')
        lap = torch.abs(_F.conv2d(padded_small, cls._laplacian_k_gpu))

        k_1, r_1 = cls._gaussian_kernel_2d_gpu(1.0, device)
        padded_lap = _F.pad(lap, (r_1, r_1, r_1, r_1), mode='reflect')
        edge = _F.conv2d(padded_lap, k_1)
        edge = torch.clamp((edge - 4.0) * (1.0 / 16.0), 0.0, 1.0)
        edge = 1.0 - edge

        small_idx = small.long().clamp(0, 255)
        exp_gate = cls._exposure_lut_gpu[small_idx]
        edge = edge * exp_gate

        gate = _F.interpolate(edge, size=g_res.shape[2:], mode='bilinear', align_corners=False)
        delta = hf * gate * float(gain)

        out = torch.clamp(r_t + delta, 0, 255).to(torch.uint8)
        return out[0].permute(1, 2, 0).cpu().numpy()

    @classmethod
    def _restore_texture_cpu(cls, restored, source, gain, sigma):
        try:
            g_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            g_res = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
            h, w_px = g_res.shape[:2]

            sigma = max(0.6, float(sigma))
            hf = cv2.subtract(g_src, cv2.GaussianBlur(g_src, (0, 0), sigmaX=sigma),
                              dtype=cv2.CV_32F)
            lim = 11.0 * sigma
            np.clip(hf, -lim, lim, out=hf)

            small = cv2.resize(g_res, (w_px // 2, h // 2),
                               interpolation=cv2.INTER_AREA)

            edge = cv2.GaussianBlur(
                np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), (0, 0),
                sigmaX=1.0)
            edge -= 4.0
            edge *= (1.0 / 16.0)
            np.clip(edge, 0.0, 1.0, out=edge)
            np.subtract(1.0, edge, out=edge)

            cv2.multiply(edge, Enhance_UltraMax._EXPOSURE_LUT[small], dst=edge)
            gate = cv2.resize(edge, (w_px, h), interpolation=cv2.INTER_LINEAR)

            delta = cv2.multiply(hf, gate, scale=float(gain), dtype=cv2.CV_16S)
            return cv2.add(restored, cv2.merge((delta, delta, delta)),
                           dtype=cv2.CV_8U)
        except cv2.error as e:
            if not Enhance_UltraMax._warned_texture:
                Enhance_UltraMax._warned_texture = True
                print(f"[UltraMax] texture restore skipped: {e}", flush=True)
            return restored

    @classmethod
    def _blur_gpu(cls, image, sigma, radius=None):
        """Gaussian blur for HWC CUDA images without an OpenCV round-trip.

        The kernel WIDTH matters as much as the sigma. `cv2.GaussianBlur` with
        ksize=(0, 0) picks `ksize = cvRound(sigma*4*2 + 1) | 1` for float input;
        spelling the radius as `round(3*sigma)` truncates the tail earlier at
        every sigma this class uses (radius 2 against OpenCV's 3 at sigma 0.8,
        6 against 8 at sigma 2.0) and moved the two paths apart. Pass `radius`
        explicitly where the CPU operator passed an explicit ksize instead of
        letting OpenCV derive one.
        """
        sigma = max(0.1, float(sigma))
        key = (str(image.device), sigma, radius)
        cached = cls._gaussian_1d_cache.get(key)
        if cached is None:
            if radius is None:
                radius = (int(round(sigma * 4 * 2 + 1)) | 1) // 2
            radius = max(1, int(radius))
            axis = torch.arange(-radius, radius + 1, dtype=torch.float32,
                                device=image.device)
            kernel = torch.exp(-0.5 * (axis / sigma) ** 2)
            cached = (kernel / kernel.sum(), radius)
            cls._gaussian_1d_cache[key] = cached
        kernel_1d, radius = cached
        nchw = image.permute(2, 0, 1).unsqueeze(0)
        channels = nchw.shape[1]
        horizontal = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        # The CPU implementation is separable too.  Doing the two 1-D passes
        # avoids the pathological 21x21 single-channel convolution selected
        # for the feather mask on CUDA while retaining the same Gaussian.
        tmp = _F.conv2d(_F.pad(nchw, (radius, radius, 0, 0), mode='reflect'),
                        horizontal, groups=channels)
        return _F.conv2d(_F.pad(tmp, (0, 0, radius, radius), mode='reflect'),
                         vertical, groups=channels)[0].permute(1, 2, 0)

    @classmethod
    def _eye_masks_gpu(cls, image):
        """Return feathered outer/core eye masks entirely on the active GPU."""
        cache_key = (str(image.device), image.shape[0], image.shape[1],
                     cls.model_template, cls._PROTECT_EYE_OUTER_X,
                     cls._PROTECT_EYE_OUTER_Y, cls._PROTECT_EYE_CORE_X,
                     cls._PROTECT_EYE_CORE_Y, cls._PROTECT_EYE_LIFT,
                     cls._PROTECT_EYE_FEATHER)
        cached = cls._eye_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        from roop.face_util import swap_template_points
        pts = np.asarray(swap_template_points(cls._SIZE, cls.model_template), dtype=np.float32)
        if pts.shape[0] < 2:
            return None, None
        sep = float(np.linalg.norm(pts[1] - pts[0]))
        if not np.isfinite(sep) or sep < 2.0:
            return None, None
        # BUILD THE MASK WITH THE SAME cv2 CALLS THE CPU OPERATOR USES, then
        # upload it once. `cv2.ellipse(..., -1)` fills a POLYGON APPROXIMATION
        # of the ellipse, not the analytic `(dx^2 + dy^2 <= 1)` region, so an
        # analytic meshgrid rasterisation disagrees with it by up to a pixel at
        # the rim -- which survives the feather as a ~0.197 difference in a
        # mask that multiplies every term downstream. It was the single largest
        # remaining source of disagreement between the two paths.
        #
        # There is no recurring cost to doing it this way: these masks depend
        # only on the template and the crop size, so they are built once per
        # run and served from `_eye_mask_cache` for every face after the first.
        outer_x = max(1, int(round(cls._PROTECT_EYE_OUTER_X * sep)))
        outer_y = max(1, int(round(cls._PROTECT_EYE_OUTER_Y * sep)))
        core_x = max(1, int(round(cls._PROTECT_EYE_CORE_X * sep)))
        core_y = max(1, int(round(cls._PROTECT_EYE_CORE_Y * sep)))
        lift = cls._PROTECT_EYE_LIFT * sep
        shape = (int(image.shape[0]), int(image.shape[1]))
        outer_np = np.zeros(shape, dtype=np.float32)
        core_np = np.zeros(shape, dtype=np.float32)
        for eye in pts[:2]:
            center = (int(round(float(eye[0]))),
                      int(round(float(eye[1] - lift))))
            cv2.ellipse(outer_np, center, (outer_x, outer_y), 0, 0, 360, 1.0, -1)
            cv2.ellipse(core_np, center, (core_x, core_y), 0, 0, 360, 1.0, -1)
        k = max(1, int(round(cls._PROTECT_EYE_FEATHER * sep)))
        if k % 2 == 0:
            k += 1
        outer_np = cv2.GaussianBlur(outer_np, (k, k), 0)
        core_np = cv2.GaussianBlur(core_np, (k, k), 0)
        masks = (torch.from_numpy(outer_np).to(image.device).clamp(0, 1),
                 torch.from_numpy(core_np).to(image.device).clamp(0, 1))
        cls._eye_mask_cache[cache_key] = masks
        return masks

    @classmethod
    def _protect_swapped_eyes_gpu(cls, restored, source):
        """CUDA equivalent of `_protect_swapped_eyes` for HWC float tensors."""
        outer, core = cls._eye_masks_gpu(restored)
        if outer is None:
            return restored
        src_blur = cls._blur_gpu(source, 0.8)
        src_sharp = (source + cls._PROTECT_EYE_SHARPEN * (source - src_blur)).clamp(0, 255)
        # `cv2.cvtColor(..., BGR2GRAY)` returns uint8, so the CPU operator's
        # Sobel reads a QUANTISED luminance. Rounding here keeps the gradient
        # gates -- which are thresholded at 1, 2 and 8 grey levels -- reading
        # the same signal on both paths.
        # `cv2.cvtColor(..., BGR2GRAY)` is FIXED-POINT integer arithmetic:
        # (B*3735 + G*19235 + R*9798 + 16384) >> 15, on uint8, yielding uint8.
        # A rounded float dot product with (0.114, 0.587, 0.299) is close but
        # not equal, and the gradient gates below threshold at 1, 2 and 8 grey
        # levels, so a one-level disagreement is not cosmetic there.
        def _cv_gray(t):
            v = (t[:, :, 0] * 3735.0 + t[:, :, 1] * 19235.0
                 + t[:, :, 2] * 9798.0 + 16384.0)
            return torch.floor(v / 32768.0).clamp(0, 255).unsqueeze(-1)
        src_gray = _cv_gray(source)
        out_gray = _cv_gray(restored)
        sobel_x = restored.new_tensor(((1., 0., -1.), (2., 0., -2.), (1., 0., -1.))).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        def grad(gray):
            n = gray.permute(2, 0, 1).unsqueeze(0)
            p = _F.pad(n, (1, 1, 1, 1), mode='reflect')
            return _F.conv2d(p, sobel_x)[0, 0], _F.conv2d(p, sobel_y)[0, 0]
        sx, sy = grad(src_gray)
        rx, ry = grad(out_gray)
        src_mag = torch.sqrt(sx.square() + sy.square())
        out_mag = torch.sqrt(rx.square() + ry.square())
        agreement = ((sx * rx + sy * ry) / (src_mag * out_mag + 1e-4) - 0.35).div(0.65).clamp(0, 1)
        detail_gate = outer * ((src_mag - 8.0) / 36.0).clamp(0, 1) * agreement
        core_gate = core * ((src_mag - 2.0) / 18.0).clamp(0, 1) * agreement
        restored_detail = restored - cls._blur_gpu(restored, 0.8)
        base = restored * (1.0 - outer.unsqueeze(-1)) + src_sharp * outer.unsqueeze(-1)
        presence = ((src_mag - 1.0) / 12.0).clamp(0, 1)
        colour_agreement = torch.exp(-(restored - source).abs().mean(dim=2) / 28.0)
        core_mix = (cls._PROTECT_EYE_CORE_MIX * core * presence
                    * colour_agreement).unsqueeze(-1)
        # TWO successive mixes, matching the CPU operator exactly -- first
        # toward the raw restored crop, then toward its sharpened form. This is
        # not redundant and it is not a transcription slip: applying the mask
        # once with `core_sharp` gives a materially different result wherever
        # the mask is partial (at m=0.5, `0.5*base + 0.5*S` against the CPU's
        # `0.25*base + 0.25*R + 0.5*S`). Collapsing it measured max 65/255
        # against the CPU path on random input, which is a visibly different
        # eye, and the parity test's mean-based tolerance did not catch it.
        core_sharp = (restored + 0.55 * (restored - cls._blur_gpu(restored, 0.65))).clamp(0, 255)
        base = base * (1.0 - core_mix) + restored * core_mix
        base = base * (1.0 - core_mix) + core_sharp * core_mix
        detail = cls._PROTECT_EYE_DETAIL_GAIN * detail_gate + cls._PROTECT_EYE_CORE_DETAIL_GAIN * core_gate
        return (base + detail.unsqueeze(-1) * restored_detail).clamp(0, 255)

    @classmethod
    def _protect_swapped_eyes(cls, restored, source):
        """Preserve RealSwap's eye/eyelash structure after UltraMax restore.

        Both arrays are in the same FFHQ-512 crop space.  A soft union of the
        two eye ellipses restore the swapped crop as the geometry base.  A
        gradient-agreement gate transfers only aligned high-frequency detail
        from UltraMax; it never directly blends competing generated eye edges.
        The source is the swapped crop, not the original plate, so identity
        and gaze remain those of the swap.
        """
        if restored.shape != source.shape or restored.ndim != 3:
            return restored
        try:
            from roop.face_util import swap_template_points
            pts = np.asarray(swap_template_points(cls._SIZE, cls.model_template),
                             dtype=np.float32)
            if pts.shape[0] < 2:
                return restored
            sep = float(np.linalg.norm(pts[1] - pts[0]))
            if not np.isfinite(sep) or sep < 2.0:
                return restored
            outer_x = max(1, int(round(cls._PROTECT_EYE_OUTER_X * sep)))
            outer_y = max(1, int(round(cls._PROTECT_EYE_OUTER_Y * sep)))
            core_x = max(1, int(round(cls._PROTECT_EYE_CORE_X * sep)))
            core_y = max(1, int(round(cls._PROTECT_EYE_CORE_Y * sep)))
            lift = cls._PROTECT_EYE_LIFT * sep
            outer = np.zeros(restored.shape[:2], dtype=np.float32)
            core = np.zeros(restored.shape[:2], dtype=np.float32)
            for eye in pts[:2]:
                center = (int(round(eye[0])), int(round(eye[1] - lift)))
                cv2.ellipse(outer, center, (outer_x, outer_y), 0, 0, 360, 1.0, -1)
                cv2.ellipse(core, center, (core_x, core_y), 0, 0, 360, 1.0, -1)
            k = max(1, int(round(cls._PROTECT_EYE_FEATHER * sep)))
            if k % 2 == 0:
                k += 1
            outer = cv2.GaussianBlur(outer, (k, k), 0)
            core = cv2.GaussianBlur(core, (k, k), 0)
            # Keep the swapped crop's geometry, but do not simply paste its
            # low-resolution pixels back: that was halo-safe yet visibly soft.
            # A small source-only high-pass lift restores lash/catchlight
            # crispness without importing CodeFormer's competing eye edges.
            src_f = source.astype(np.float32)
            src_blur = cv2.GaussianBlur(src_f, (0, 0), 0.8)
            src_sharp = np.clip(src_f + cls._PROTECT_EYE_SHARPEN * (src_f - src_blur),
                                0.0, 255.0)
            # The source is the entire eye-band geometry base.  Recover only
            # UltraMax's detail where its local edge points in the same
            # direction as the source edge.  A second lid/iris has a displaced
            # or opposing gradient, so its high-pass content is rejected.
            src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
            out_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sx = cv2.Sobel(src_gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(src_gray, cv2.CV_32F, 0, 1, ksize=3)
            rx = cv2.Sobel(out_gray, cv2.CV_32F, 1, 0, ksize=3)
            ry = cv2.Sobel(out_gray, cv2.CV_32F, 0, 1, ksize=3)
            src_mag = cv2.magnitude(sx, sy)
            out_mag = cv2.magnitude(rx, ry)
            dot = sx * rx + sy * ry
            agreement = dot / (src_mag * out_mag + 1e-4)
            np.clip((agreement - 0.35) / 0.65, 0.0, 1.0, out=agreement)
            edge = np.clip((src_mag - 8.0) / 36.0, 0.0, 1.0)
            detail_gate = outer * edge * agreement
            # The cornea/conjunctiva live inside the aperture, not on the lid
            # boundary.  Allow a stronger aligned high-pass transfer there so
            # UltraMax can restore crisp catchlights and scleral texture while
            # the surrounding generated eyelid/iris contour stays rejected.
            core_edge = np.clip((src_mag - 2.0) / 18.0, 0.0, 1.0)
            core_gate = core * core_edge * agreement
            restored_f = restored.astype(np.float32)
            restored_detail = restored_f - cv2.GaussianBlur(restored_f, (0, 0), 0.8)
            base = restored_f * (1.0 - outer[:, :, None]) + src_sharp * outer[:, :, None]
            # A small central-aperture mix restores UltraMax's cornea and
            # conjunctiva structure.  The core is deliberately much smaller
            # than the protected band, so it cannot carry a second lid/brow
            # boundary; all surrounding geometry remains source-anchored.
            # Require a real source-side aperture signal before allowing the
            # direct mix.  Flat/shifted generated contours therefore cannot
            # leak through the core mask and form a second eye.
            core_presence = np.clip((src_mag - 1.0) / 12.0, 0.0, 1.0)
            # Do not mix pixels where UltraMax disagrees strongly with the
            # swapped crop's tone; those are the usual source of a ghost iris
            # or a bright/dark duplicate eye after paste-back.
            colour_delta = np.mean(np.abs(restored_f - src_f), axis=2)
            colour_agreement = np.exp(-colour_delta / 28.0)
            core_mix = (cls._PROTECT_EYE_CORE_MIX * core * core_presence
                        * colour_agreement)[:, :, None]
            base = base * (1.0 - core_mix) + restored_f * core_mix
            # UltraMax can leave the corneal aperture low-contrast even when
            # its pixels are geometrically correct.  Apply a restrained,
            # aperture-only unsharp lift to existing restored detail; unlike a
            # pasted eye edge this cannot invent a second lid or brow contour.
            core_sharp = np.clip(
                restored_f + 0.55 * (restored_f - cv2.GaussianBlur(restored_f, (0, 0), 0.65)),
                0.0, 255.0)
            base = base * (1.0 - core_mix) + core_sharp * core_mix
            detail = (cls._PROTECT_EYE_DETAIL_GAIN * detail_gate
                      + cls._PROTECT_EYE_CORE_DETAIL_GAIN * core_gate)
            return np.rint(base + detail[:, :, None] * restored_detail).clip(0, 255).astype(np.uint8)
        except (cv2.error, TypeError, ValueError, ImportError):
            return restored

    @classmethod
    def _rebalance_masks_gpu(cls, image, sep, pts):
        """Selection and feathered masks for the rebalance pass, built once.

        Same reasoning as `_eye_masks_gpu`: `cv2.ellipse` fills a polygon
        approximation, so these are built with cv2 and uploaded rather than
        rasterised analytically. They depend only on the template and the crop
        size, so the cost is paid once per run, not once per face.
        """
        key = (str(image.device), int(image.shape[0]), int(image.shape[1]),
               cls.model_template, round(sep, 4))
        cached = cls._rebalance_mask_cache.get(key)
        if cached is not None:
            return cached
        rx = max(1, int(round(0.28 * sep)))
        ry = max(1, int(round(0.15 * sep)))
        shape = (int(image.shape[0]), int(image.shape[1]))
        selects, feathers = [], []
        for eye in pts[:2]:
            m = np.zeros(shape, dtype=np.float32)
            center = (int(round(float(eye[0]))),
                      int(round(float(eye[1] - cls._PROTECT_EYE_LIFT * sep))))
            cv2.ellipse(m, center, (rx, ry), 0, 0, 360, 1.0, -1)
            selects.append(torch.from_numpy(m > 0.5).to(image.device))
            feathers.append(torch.from_numpy(
                cv2.GaussianBlur(m, (0, 0), max(1.0, 0.035 * sep))).to(image.device))
        cached = (selects, feathers)
        cls._rebalance_mask_cache[key] = cached
        return cached

    @classmethod
    def _rebalance_eye_detail_gpu(cls, image):
        """CUDA equivalent of `_rebalance_eye_detail` for an HWC float tensor.

        This operator was absent from the CUDA post-process entirely, so the
        two paths differed by a whole stage as well as by their arithmetic.
        """
        if image.ndim != 3 or image.shape[0] != cls._SIZE or image.shape[1] != cls._SIZE:
            return image
        try:
            from roop.face_util import swap_template_points
            pts = np.asarray(swap_template_points(cls._SIZE, cls.model_template),
                             dtype=np.float32)
            if pts.shape[0] < 2:
                return image
            sep = float(np.linalg.norm(pts[1] - pts[0]))
            if not np.isfinite(sep) or sep < 2.0:
                return image
            selects, feathers = cls._rebalance_masks_gpu(image, sep, pts)
            gray = (image[:, :, 0] * 3735.0 + image[:, :, 1] * 19235.0
                    + image[:, :, 2] * 9798.0 + 16384.0)
            gray = torch.floor(gray / 32768.0).clamp(0, 255)
            lap_k = image.new_tensor(((0., 1., 0.), (1., -4., 1.), (0., 1., 0.))).view(1, 1, 3, 3)
            padded = _F.pad(gray.view(1, 1, *gray.shape), (1, 1, 1, 1), mode='reflect')
            lap = torch.abs(_F.conv2d(padded, lap_k))[0, 0]
            scores = []
            for sel in selects:
                vals = lap[sel]
                scores.append(float(vals.mean().item()) if vals.numel() else 0.0)
            lo = int(np.argmin(scores))
            hi = int(np.argmax(scores))
            if scores[lo] <= 0.0 or scores[hi] <= scores[lo] * cls._EYE_BALANCE_TRIGGER:
                return image
            gain = min(cls._EYE_BALANCE_MAX,
                       max(0.0, (scores[hi] - scores[lo]) / scores[hi]))
            if gain <= 0.0:
                return image
            blur = cls._blur_gpu(image, 0.75)
            m = (feathers[lo] * gain).unsqueeze(-1)
            return torch.clamp(image + m * (image - blur), 0.0, 255.0)
        except (RuntimeError, TypeError, ValueError, ImportError):
            return image

    @classmethod
    def _rebalance_eye_detail(cls, image):
        """Lift only the softer eye when the two apertures are imbalanced.

        RealSwap can produce one eye with materially less local contrast than
        the other (especially on a near-lateral pose).  A symmetric sharpen
        amplifies halos, so compare the two template apertures and sharpen the
        weaker one only, using a low-gain feathered mask.
        """
        if image.ndim != 3 or image.shape[0] != cls._SIZE or image.shape[1] != cls._SIZE:
            return image
        try:
            from roop.face_util import swap_template_points
            pts = np.asarray(swap_template_points(cls._SIZE, cls.model_template),
                             dtype=np.float32)
            if pts.shape[0] < 2:
                return image
            sep = float(np.linalg.norm(pts[1] - pts[0]))
            if not np.isfinite(sep) or sep < 2.0:
                return image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
            rx = max(1, int(round(0.28 * sep)))
            ry = max(1, int(round(0.15 * sep)))
            scores = []
            masks = []
            for eye in pts[:2]:
                m = np.zeros(gray.shape, dtype=np.float32)
                center = (int(round(eye[0])), int(round(eye[1] - cls._PROTECT_EYE_LIFT * sep)))
                cv2.ellipse(m, center, (rx, ry), 0, 0, 360, 1.0, -1)
                values = lap[m > 0.5]
                scores.append(float(values.mean()) if values.size else 0.0)
                masks.append(cv2.GaussianBlur(m, (0, 0), max(1.0, 0.035 * sep)))
            lo = int(np.argmin(scores))
            hi = int(np.argmax(scores))
            if scores[lo] <= 0.0 or scores[hi] <= scores[lo] * cls._EYE_BALANCE_TRIGGER:
                return image
            gain = min(cls._EYE_BALANCE_MAX,
                       max(0.0, (scores[hi] - scores[lo]) / scores[hi]))
            if gain <= 0.0:
                return image
            base = image.astype(np.float32)
            blur = cv2.GaussianBlur(base, (0, 0), 0.75)
            m = (masks[lo] * gain)[:, :, None]
            return np.clip(base + m * (base - blur), 0.0, 255.0).astype(np.uint8)
        except (cv2.error, TypeError, ValueError, ImportError):
            return image

    # ── run ──────────────────────────────────────────────────────────────────
    def _run_cuda_postprocess(self, src512, input_size, original_frame):
        """Run ORT I/O binding and all UltraMax post-ops on CUDA tensors.

        ORT 1.23 does not implement DLPack for CUDA OrtValues.  Instead, bind
        the model output to a pre-allocated Torch CUDA allocation: ORT writes
        the tensor directly and PyTorch owns it immediately afterwards.  This
        keeps the model output, chroma transfer, eye protection, and optional
        texture operation on one device with no ORT->NumPy round-trip.

        Returns ``None`` only when the installed provider rejects that binding;
        the established host path then remains the compatibility fallback.

        MEASURED, AND THE THROUGHPUT RESULT IS NEGATIVE -- READ BEFORE
        "OPTIMISING" ANYTHING ELSE HERE. Isolated, this path is 3.14x faster
        per face than the host one (92.92 -> 29.56 ms, RTX 4070 / TensorRT,
        40 warm calls). End to end it is worth NOTHING:

            600 frames of double/d1.mp4, UltraMax + realswap + RealityUX,
            threads 10, counterbalanced host/CUDA/CUDA/host, null control first

            null control    4.65 / 4.77 fps   (2.6% spread)
            host arms       4.54 / 4.68       mean 4.61
            CUDA arms       4.54 / 4.70       mean 4.62   -> +0.2%, NOT RESOLVABLE

        The arithmetic says why, and it was predictable: 63.4 ms/face saved
        over 361 faces is 22.9 s of THREAD time, which across 10 workers is
        2.29 s of wall clock against a 130 s render -- a 1.76% ceiling before
        anything else is considered. +0.2% is consistent with that ceiling.
        The host work was already parallel and was never the wall-clock
        limiter. GPU sat at ~25% in every arm, so the render is not GPU-bound
        either; it is bound by serialisation, not by this stage.

        This path is kept because it is CORRECT and strictly cheaper, not
        because it is faster: it is bit-identical to the host path on a real
        aligned crop, it removes 63 ms/face of host CPU work that contends
        with everything else on the machine, and getting here fixed three
        genuine defects (see the note below, `.contiguous()`, and the eye
        operators' parity). It is NOT evidence that moving more OpenCV to the
        GPU will pay -- the opposite; see the 2026-06-27 verdict that rejected
        exactly that, and the standing rule that a stage's share of thread
        time is not a speedup budget.
        """
        if (self._cuda_iob_available is False or not
                (_TORCH_CUDA and getattr(self, 'devicename', '') == 'cuda')):
            return None
        # Escape hatch, and the lever an A/B needs to measure this at all.
        # Defaults ON: the CUDA chain is measured bit-identical to the host
        # path on a real aligned crop, so there is no reason for a user to be
        # opted out of it by default.
        if os.environ.get('ROOP_ULTRAMAX_CUDA_POST', '1').strip().lower() in (
                '0', 'off', 'false', 'no'):
            self._cuda_iob_available = False
            return None
        # TensorRT dynamic CodeFormer: multi-worker execution with external PyTorch
        # CUDA I/O binding can deadlock the CUDA driver / TRT execution stream.
        # Use the verified, artifact-free host I/O binding path under TensorRT EP.
        try:
            uses_tensorrt = 'TensorrtExecutionProvider' in self.session.get_providers()
        except (AttributeError, TypeError):
            uses_tensorrt = False
        if uses_tensorrt:
            self._cuda_iob_available = False
            return None
        try:
            dtype = torch.float16 if self.in_dtype == np.float16 else torch.float32
            source = torch.from_numpy(src512).to('cuda', dtype=torch.float32, non_blocking=False)
            # `.contiguous()` IS LOAD-BEARING. `permute` + `flip` leave a
            # non-contiguous view, and `bind_input` is handed a raw
            # `data_ptr()` -- ORT reads that pointer as though the tensor were
            # contiguous, so without this the network receives transposed
            # garbage. It still returns a plausible finite image, so the only
            # symptom was the collapse guard rejecting the face.
            x = (source.permute(2, 0, 1).flip(0).unsqueeze(0)
                 .div(127.5).sub(1.0).to(dtype).contiguous())
            fidelity = torch.tensor([getattr(roop.globals, 'codeformer_fidelity', 0.5)],
                                    device=source.device, dtype=torch.float64)
            in0, in1 = self.model_inputs[0].name, self.model_inputs[1].name
            out_name = self.model_outputs[0].name
            input_element = np.float16 if dtype == torch.float16 else np.float32
            out_dtype = (torch.float16 if 'float16' in self.model_outputs[0].type
                         else torch.float32)
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            with exclusive(self.pool, self._session_lock,
                           (self.session, self.io_binding)) as (sess, iob):
                if hasattr(iob, 'clear_binding_inputs'):
                    iob.clear_binding_inputs()
                if hasattr(iob, 'clear_binding_outputs'):
                    iob.clear_binding_outputs()
                iob.bind_input(in0, 'cuda', 0, input_element, tuple(x.shape), x.data_ptr())
                iob.bind_input(in1, 'cuda', 0, np.float64, tuple(fidelity.shape), fidelity.data_ptr())
                # ORT allocates and owns this. Do NOT pass a Torch pointer
                # here; see the note above for what that cost.
                iob.bind_output(out_name, 'cuda', 0)
                sess.run_with_iobinding(iob)
                iob.synchronize_outputs()
                output = ort_cuda_output_to_torch(iob.get_outputs()[0], out_dtype)
            if output is None:
                # No CUDA runtime, or a device/shape the copy declined. The
                # host path below is the supported fallback, not an error.
                self._cuda_iob_available = False
                return None
            restored = output[0].flip(0).permute(1, 2, 0).to(torch.float32)
            if not bool(torch.isfinite(restored).all().item()):
                print('[UltraMax] non-finite CUDA output - using unenhanced frame')
                return sized(original_frame, input_size)
            restored = restored.clamp(-1.0, 1.0).add(1.0).mul(127.5)
            source_std = source.std()
            if bool((source_std > 8.0).item()) and bool((restored.std() < source_std * 0.35).item()):
                print('[UltraMax] CUDA output collapsed (flat) - using unenhanced frame')
                return sized(original_frame, input_size)
            chroma = _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA)
            if chroma != 1.0:
                restored = luma_only_recolour_tensor(restored, source, chroma)
            if self._STRUCTURE_SHARPEN > 0.0:
                restored = (restored + self._STRUCTURE_SHARPEN *
                            (restored - self._blur_gpu(restored, 0.8))).clamp(0, 255)
            if chroma != 1.0:
                # Both operators, in the host path's order. `_rebalance_eye_detail`
                # was absent from this chain entirely, so enabling the CUDA path
                # without it would have silently dropped a stage.
                restored = self._protect_swapped_eyes_gpu(restored, source)
                restored = self._rebalance_eye_detail_gpu(restored)
            gain = _env_float('ROOP_ULTRAMAX_TEXTURE', self._TEXTURE_GAIN)
            if gain > 0.0:
                # The opt-in texture residual remains in VRAM as well.  The
                # default is zero; this conservative high-pass is limited to
                # the same source signal and cannot invent texture.
                high = (source - self._blur_gpu(source, self._SIGMA)).clamp(
                    -11.0 * self._SIGMA, 11.0 * self._SIGMA)
                restored = (restored + high * gain).clamp(0, 255)
                with self._lock:
                    self._textured += 1
            scale = 1
            if restored.shape[1] < input_size:
                restored = _F.interpolate(restored.permute(2, 0, 1).unsqueeze(0),
                                          size=(input_size, input_size), mode='bicubic',
                                          align_corners=False)[0].permute(1, 2, 0)
            else:
                scale = max(1, int(restored.shape[1] / input_size))
            # One intentional egress boundary remains because the surrounding
            # merger/writer API is NumPy today.  No intermediate CPU copy is
            # made between ONNX output, chroma, eye protection, and texture.
            with self._lock:
                self._faces += 1
            self._cuda_iob_available = True
            return restored.round().to(torch.uint8).cpu().numpy(), scale
        except Exception:
            # Old ORT builds expose CUDA OrtValues but not DLPack.  Fall back to
            # the established CPU binding path rather than failing a render.
            self._cuda_iob_available = False
            return None

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

        # The CUDA fast post-path implements the SINGLE-stream chain only, and
        # returns a finished frame. Skipping it under dual-stream is not a
        # throughput concession worth arguing about: its own docstring records
        # it as +0.2% end to end, i.e. not resolvable, because the host work it
        # removes was already parallel and never the wall-clock limiter.
        dual = self.dual_enabled() and self._detail_session is not None
        if not dual:
            cuda_result = self._run_cuda_postprocess(src512, input_size, temp_frame)
            if cuda_result is not None:
                return cuda_result

        # One gather: uint8 BGR HWC -> model dtype RGB CHW in [-1, 1]. Fancy
        # indexing returns a fresh C-contiguous array, so the [None] that follows
        # is a free view and no extra copy is made.
        x = self._lut[src512.transpose(2, 0, 1)[::-1]][None]
        w_fid = np.array([getattr(roop.globals, 'codeformer_fidelity', 0.5)],
                         dtype=np.float64)

        in0, in1 = self.model_inputs[0].name, self.model_inputs[1].name
        out_name = self.model_outputs[0].name

        def _infer(sess, iob):
            # `_run_cuda_postprocess` clears bindings before attempting its
            # zero-copy path.  Rebind the compatibility output explicitly so a
            # failed CUDA probe cannot poison this CPU fallback.
            if hasattr(iob, 'clear_binding_inputs'):
                iob.clear_binding_inputs()
            if hasattr(iob, 'clear_binding_outputs'):
                iob.clear_binding_outputs()
            iob.bind_cpu_input(in0, x)
            iob.bind_cpu_input(in1, w_fid)
            iob.bind_output(out_name)
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
        # ── DUAL-STREAM FREQUENCY SPLIT ─────────────────────────────────
        # Step B: GPEN-512 on the same crop, for its high band only.
        # Step C: guided-filter split -- CodeFormer's low band carries the
        #         geometry, GPEN's high band carries pores/lashes/lip texture.
        # Step D: Reinhard LAB transfer back toward the crop the restorers
        #         were handed, which removes BOTH networks' casts at once
        #         (CodeFormer drifts pale, GPEN drifts pink) in one operator.
        # Steps run here, before the periocular protection, so that pass still
        # gets the last word on the eye band it exists to defend.
        did_dual = False
        if dual:
            detail = self._detail_stream(src512)
            if detail is not None:
                clamp = self._DUAL_CLAMP or None
                kw = dict(
                    radius=int(_env_float('ROOP_ULTRAMAX_DUAL_RADIUS',
                                          self._DUAL_RADIUS)),
                    eps=_env_float('ROOP_ULTRAMAX_DUAL_EPS', self._DUAL_EPS),
                    gain=_env_float('ROOP_ULTRAMAX_DUAL_GAIN', self._DUAL_GAIN),
                    clamp=_env_float('ROOP_ULTRAMAX_DUAL_CLAMP', clamp))
                mode = (os.environ.get('ROOP_ULTRAMAX_DUAL_MODE', '')
                        or self._DUAL_MODE).strip().lower()
                if mode == 'bgr':
                    # The measured-worse original: splits all three channels,
                    # so GPEN's chroma cast rides in on the high band, and
                    # Reinhard cannot remove it (global moments vs a per-pixel
                    # cast). Kept reachable so the comparison can be re-run.
                    restored = frequency_split(restored, detail, **kw)
                    restored = reinhard_lab(
                        restored, src512, strength=1.0,
                        l_weight=_env_float('ROOP_ULTRAMAX_DUAL_LW',
                                            self._DUAL_L_WEIGHT))
                else:
                    restored = frequency_split_luma(restored, detail, **kw)
                    # Chrominance from the crop, as both GPEN processors do.
                    # This is the 0.44-0.45 colour path; Reinhard measured 1.79.
                    restored = luma_only_recolour(restored, src512, 0.0)
                did_dual = True
                with self._lock:
                    self._dual += 1

        # Single-stream colour path. Skipped after a split, which has already
        # done its own colour transfer against the same reference -- running
        # both would correct the same cast twice.
        try:
            if not did_dual:
                restored = luma_only_recolour(
                    restored, src512,
                    _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA))
        except cv2.error as e:
            # Say so once. A colour fix that silently no-ops leaves a plausible
            # image carrying the exact pale cast it exists to remove.
            if not Enhance_UltraMax._warned_colour:
                Enhance_UltraMax._warned_colour = True
                print(f"[UltraMax] colour fix skipped: {e}", flush=True)

        # CodeFormer's mixed-precision output is numerically valid but can be
        # slightly soft at small eye, nose, and lip contours after the 512->
        # frame resize.  Recover a restrained structural edge response before
        # the periocular protection pass; this is an unsharp lift, not a second
        # generated face, and its low gain avoids recreating the old halo.
        # Skipped after a split: the high band was just re-injected at gain
        # 1.25 from a network measured sharper than this one, so a further
        # global unsharp stacks two sharpeners and rebuilds the over-sharp look
        # this engine exists to get away from.
        if self._STRUCTURE_SHARPEN > 0.0 and not did_dual:
            base = restored.astype(np.float32)
            blur = cv2.GaussianBlur(base, (0, 0), 0.8)
            restored = np.clip(base + self._STRUCTURE_SHARPEN * (base - blur),
                               0.0, 255.0).astype(np.uint8)

        # Do this after the colour pass: the protected source pixels should
        # retain the same chroma as the rest of the swapped crop, while the
        # restored cheeks/nose keep UltraMax's luminance and detail.
        # The diagnostic chroma=1 mode is intentionally bit-faithful to the
        # raw CodeFormer output for precision comparisons; do not apply the
        # visual eye-protection/rebalance operators in that mode.
        if _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA) != 1.0:
            restored = self._protect_swapped_eyes(restored, src512)
            restored = self._rebalance_eye_detail(restored)

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
            f, t, d = self._faces, self._textured, self._dual
        if not f:
            return None
        c = _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA)
        colour = ("swapper chrominance" if c <= 0.0
                  else f"chrominance {c:g} toward CodeFormer's own")
        path = ('CUDA tensor post-process' if self._cuda_iob_available
                else 'compatibility host path')
        if d:
            # Reported as a COUNT, not as "dual-stream: on". A flag says what
            # was asked for; a count says what actually ran, and the two come
            # apart exactly when something is broken -- four enhancers once
            # failed on 60 of 60 faces here while the audit read 100%.
            return (f"[UltraMax] {f} faces restored (dual-stream: "
                    f"codeformer.fp16 structure + GPEN-512 detail on {d}, "
                    f"guided split r={self._DUAL_RADIUS} "
                    f"gain={self._DUAL_GAIN}, Reinhard LAB); "
                    f"texture restored on {t}")
        return (f"[UltraMax] {f} faces restored (codeformer.fp16, {path}, "
                f"{colour}); texture restored on {t}")
