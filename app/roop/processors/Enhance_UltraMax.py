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

   REMOVED 2026-10-13 (`_restore_texture`, its GPU twin, `ROOP_ULTRAMAX_TEXTURE`
   and `ROOP_ULTRAMAX_TEXTURE_SIGMA`; last present in commit d995920). The three
   mask definitions that disagreed by 34% / 155% / 500% on the same footage are
   the lesson worth keeping, which is why this section stays.

   What it did: re-injected the high-frequency luminance the restorer erased,
   taken from the restorer's OWN input, only where the restorer drew flat skin
   (a structure gate left iris rims, lash lines, lid creases and lip margins
   exactly as the codebook drew them). It replaced an older multi-scale unsharp
   + CLAHE whose medium band turned an infraorbital crease into a second lower
   eyelid — the "blurry eyes" report.

   NOTE FOR ANYONE REMEASURING: do not grade a texture operator with Laplacian
   variance. That instrument counts added edge energy, and it has already
   endorsed one build here (UltraMax v1, "4.39x GPEN-256") that the user
   reported as plastic, and a second (the filter above) reported as over-sharp.
   Grade it on the footage.

3. DUAL-STREAM FREQUENCY SPLIT — MEASURED AND REMOVED (2026-10-13; last present
   in commit d995920, `ROOP_ULTRAMAX_DUAL*`, `_init_detail_stream`).
   The idea: CodeFormer's low band for geometry, GPEN-512's high band for pores
   and lashes, split by a guided filter (`roop.processors.frequency_split`,
   which other modules still use). Measured 2026-09-04, 40 aligned crops of
   s1.mp4, RTX 4070 / TensorRT:

       arm                        ms/face   edge/plate   chroma drift
       Codeformer (fp16)             41.0        1.291           2.04
       UltraMax single               40.1        1.315           0.54
       GPEN Realistic                30.7        1.262           0.52
       dual luma  gain 1.00         125.6        1.199           0.50
       dual luma  gain 1.25         125.0        1.337           0.54
       dual bgr   gain 1.00         150.0        1.213           1.63

   3.1x the cost, SOFTER than CodeFormer at gain 1.00 and level with it at 1.25:
   the premise (GPEN-512 carries ~25% more high frequency) was measured on a
   swapper crop and did not reproduce on aligned plate crops. It also put a
   second resident 512 graph beside realswap's two nets and RealityUX, the
   residency that took a render from 10.50 to 6.60 fps by paging. It defaulted
   OFF and nothing reached it, so it went.

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

     - A NATIVE TENSORRT ENGINE IS NOT THE ESCAPE EITHER (2026-09-28). ORT's
       TensorRT EP already runs these graphs whole: GPU-resident, warm,
       GPEN-256 3.47 ms ORT vs 6.81 native, RestoreFormer++ 20.44 vs 21.11,
       CodeFormer fp16 22.37 ms through ORT (its native build needs the scalar
       `w` input given a shape; not pursued after the other two). End to end on
       s7.mp4 (600 frames, hyperswap + DFL XSeg, threads 20) UltraMax costs
       +25.7 ms/frame against no enhancer -- about the network's own time.

Knobs, for re-measuring rather than for shipping a different default:
    ROOP_ULTRAMAX_CHROMA          0 = the swapper's colour (default),
                                  1 = CodeFormer's own (the old pale output)
"""
from roop.degrade import swallowed as _swallowed

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


# Cards below this hold ONE context -- this is a resident 512 graph sharing the
# card with realswap's two nets, RealityUX and the detector/detmask pools.
_POOL_SINGLE_CONTEXT_BELOW_GB = 15.5


def _structural_pool_size(requested):
    """How many CodeFormer-512 contexts to build for the structural stream.

    `tests/diag_ultramax_pool_scaling.py`, RTX 4070, 10 threads, 240 real
    crops, TensorRT -- a deeper pool is monotonically WORSE:

        contexts   faces/s   VRAM
        1          40.0      +1136 MB
        2          38.4      +1519 MB
        4          36.6      +2704 MB

    40 faces/s is 25 ms/face, which is the network's own time. The card is
    already the floor, so extra contexts buy nothing but memory pressure and
    scheduling overhead -- and this stage shares a 12GB card with realswap's
    two nets, RealityUX and the detector/detmask pools, where over-committing
    does not degrade gracefully: the driver pages contexts over PCIe and the
    render wedges while reporting ~100% GPU utilisation.

    Returns at least 1, never 0: a size-1 SessionPool is still a pool, and
    `_gpu_guard` waives the global TensorRT lock only for a processor that owns
    one. Dropping to no pool would hand the most expensive stage in a render
    back to a one-thread-at-a-time lock -- a much larger regression than the
    one being avoided here.

    ROOP_ULTRAMAX_POOL overrides, for re-measuring the table above.
    """
    n = max(1, int(requested or 1))
    try:
        gb = session_pool._detect_vram_gb()
    except Exception as _degrade_error:
        _swallowed("roop/processors/Enhance_UltraMax.py:262", _degrade_error, "fallback continued")
        gb = 0
    if 0 < gb < _POOL_SINGLE_CONTEXT_BELOW_GB:
        n = 1
    try:
        forced = int(os.environ.get('ROOP_ULTRAMAX_POOL', '') or 0)
    except ValueError:
        forced = 0
    return max(1, forced) if forced else n


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
    # TensorRT + the zero-copy ring. The pre-ring external binding deadlocked the driver
    # with several workers, so TensorRT stayed on the host path; the ring serialises
    # each slot and leases sessions in one fixed order. Opt-in until a multi-worker
    # stress run on the live engine clears it.
    @staticmethod
    def _trt_ring_enabled():
        return os.environ.get('ROOP_ULTRAMAX_TRT_RING', '0').strip().lower() in ('1', 'on', 'true', 'yes')

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
    _eye_mask_cpu_cache = {}
    _rebalance_mask_cpu_cache = {}
    _gaussian_1d_cache = {}

    _warned_colour = False
    _warned_lowband = False

    @classmethod
    def _structure_sharpen(cls):
        """Unsharp gain applied to CodeFormer's output. `ROOP_ULTRAMAX_STRUCTURE_SHARPEN` re-measures it (default 0.18)."""
        return _env_float('ROOP_ULTRAMAX_STRUCTURE_SHARPEN', cls._STRUCTURE_SHARPEN)

    # ── LOW BAND FROM THE SWAP ───────────────────────────────────────────────
    # The restorer's output used to replace the swap's low band (shading, tone, the face's
    # large-scale shape) as well as its fine detail. A codebook prior re-draws those low
    # frequencies every frame, so the face shape and shading wobbled between frames and drifted
    # from the swapped identity. `roop.enhance_blend` had measured the cure on Restore Ultra
    # (2026-09-25, 72 real calls: swap low band + w x restorer detail raised identity by +0.039
    # and brought skin texture from 1.58x the plate's to 1.09x); UltraMax never got it.
    #
    # MEASURED 2026-10-12 on Taras.mp4 (420 frames of the lead, live config, realswap +
    # RealityUX, TensorRT, pinned capture, sequential-path A/B). Temporal acceleration of the
    # face in excess of the plate's own (lower = steadier), face-core texture vs the plate:
    #
    #     no enhancer                         1.52   1.3x
    #     UltraMax, full restorer (old)       4.57   6.4x
    #     detail weight 1.0                   2.99   4.9x
    #     detail weight 0.75  (default)       2.09   3.2x
    #     detail weight 0.50                  1.36   1.9x
    #
    # i.e. two thirds of the shimmer the lead showed was the restorer's re-drawn low band, and
    # this removes ~54% of the total at 0.75 for ~0.5 ms/face of host work, in ANY frame order.
    # Rejected on the same footage: CodeFormer fidelity w 0.55 -> 0.8 (4.57 -> 4.68: the control
    # works but moves the output ~0.005), a whole-face gradient-agreement gate on the detail band
    # (4.57 -> 3.98 at full strength), and the HF-texture carry (-8% at best, and under
    # round-robin workers it only runs on ~7% of faces).  Switching the eye operators OFF made it
    # worse (5.59): they anchor the eyes to the swapped crop for the same reason.
    #
    # Settings `ultramax_detail_weight` (0..1) and `ultramax_swap_lowband` (False = the old
    # full-restorer paste); `ROOP_ULTRAMAX_DETAIL_WEIGHT` overrides both for an A/B
    # ('off' = old behaviour, or a number).
    _DETAIL_WEIGHT = 0.75

    @classmethod
    def _detail_weight(cls):
        """Weight of the restorer's high band over the swap's low band, or None for the legacy paste."""
        raw = os.environ.get('ROOP_ULTRAMAX_DETAIL_WEIGHT')
        if raw is not None and raw.strip() != '':
            if raw.strip().lower() in ('off', 'legacy', 'none', 'false', 'no'):
                return None
            try:
                return min(1.5, max(0.0, float(raw)))
            except ValueError:
                pass
        if not bool(getattr(roop.globals, 'ultramax_swap_lowband', True)):
            return None
        try:
            return min(1.5, max(0.0, float(getattr(roop.globals, 'ultramax_detail_weight',
                                                    cls._DETAIL_WEIGHT))))
        except (TypeError, ValueError):
            return cls._DETAIL_WEIGHT

    @classmethod
    def _lowband_blend_gpu(cls, restored, source, weight):
        """`enhance_blend.frequency_blend` on HWC float tensors: blur(swap) + w * (restored - blur(restored))."""
        if weight is None or restored.shape != source.shape or restored.ndim != 3:
            return restored
        try:
            from roop import enhance_blend
            sigma = enhance_blend.SPLIT_SIGMA_512 * (restored.shape[1] / 512.0)
            return (cls._blur_gpu(source, sigma) + weight
                    * (restored - cls._blur_gpu(restored, sigma))).clamp(0, 255)
        except (RuntimeError, TypeError, ValueError) as exc:
            # Say so once: a recombination that silently no-ops leaves a plausible image and a setting that
            # reads as applied.
            if not Enhance_UltraMax._warned_lowband:
                Enhance_UltraMax._warned_lowband = True
                print(f"[UltraMax] low-band recombination failed ({type(exc).__name__}: {exc}); "
                      f"full restorer output used", flush=True)
            return restored

    @staticmethod
    def _eye_ops_enabled():
        """Eye protection + rebalance. `ROOP_ULTRAMAX_EYE_OPS=0` turns both off, for A/B only (default on)."""
        from roop.env import env_bool
        return env_bool('ROOP_ULTRAMAX_EYE_OPS', True)

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
        self._ring_faces = 0
        self._lowband = 0
        # ``False`` means this ORT build/provider rejected binding its output
        # directly into a PyTorch CUDA allocation; avoid retrying per face.
        self._cuda_iob_available = None
        self._lock = threading.Lock()
        # Serialises Initialize / Release (engine compilation + context creation).
        self._init_lock = threading.RLock()
        # Guards the SINGLE shared (session, io_binding) used when there is no
        # pool -- binding state is not thread-safe. Distinct from _lock, which
        # only counts faces.
        self._session_lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        # TensorRT compiles the engine inside the first session build and the canary's first inference, which takes minutes on a
        # cold cache: two threads must not both build it, so initialise and release serialise on one re-entrant lock (Release is
        # called from here on a device change).
        with self._init_lock:
            self._initialize(plugin_options)

    def _initialize(self, plugin_options: dict):
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

        def _session(chain):
            from roop.strict_session import create_strictly_accelerated_session
            return create_strictly_accelerated_session(
                model_path, providers=chain, sess_options=opts,
                tag='enhancer:ultramax')

        def _bind(sess):
            # One io_binding per slot, held for the life of the run. A binding is
            # not shareable across threads, and rebuilding it per call measured
            # 1.5 ms of the 25 ms budget. The OUTPUT binding never changes, so it
            # is set once here; only the two inputs are rebound per call.
            iob = sess.io_binding()
            iob.bind_output(sess.get_outputs()[0].name, self.devicename)
            return (sess, iob)

        from roop.backend_manager import build_session_with_fallback
        def _build_raw(chain):
            return _bind(_session(chain))

        (self.session, self.io_binding), session_providers = build_session_with_fallback(
            _build_raw, session_providers, tag='enhancer:ultramax')
        def _build(_i=0):
            return _bind(_session(session_providers))
        # Startup canary (roop/aux_canary.py): two fixed noisy-skin inputs through the live engine and a one-off reference
        # (the same graph on the CPU EP), SSIM floor from docs/perf/aux_canary_enhancer_calibration.json, verdict cached by
        # engine hash. Non-finite output fails it outright. On failure the enhancer is rebuilt on TensorRT FP32, then CUDA/CPU.
        # Skipped (verdict None) off TensorRT, i.e. on the 3060. Nothing here is a speed change: a passing verdict is one
        # dict lookup after the first start.
        from roop import aux_canary
        _guarded, session_providers, _verdict = aux_canary.guard(
            'codeformer.fp16', self.session, model_path, session_providers, _session)
        if _guarded is not self.session:
            self.session, self.io_binding = _bind(_guarded)
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
            n = _structural_pool_size(n)
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

    def Release(self):
        with self._init_lock:
            self._release()

    def _release(self):
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
    def _eye_masks_cpu(cls, shape):
        """Return feathered outer/core eye masks on CPU, cached by shape and template."""
        cache_key = (shape[0], shape[1], cls.model_template, cls._SIZE)
        cached = cls._eye_mask_cpu_cache.get(cache_key)
        if cached is not None:
            return cached
        from roop.face_util import swap_template_points
        pts = np.asarray(swap_template_points(cls._SIZE, cls.model_template), dtype=np.float32)
        if pts.shape[0] < 2:
            return None, None
        sep = float(np.linalg.norm(pts[1] - pts[0]))
        if not np.isfinite(sep) or sep < 2.0:
            return None, None
        outer_x = max(1, int(round(cls._PROTECT_EYE_OUTER_X * sep)))
        outer_y = max(1, int(round(cls._PROTECT_EYE_OUTER_Y * sep)))
        core_x = max(1, int(round(cls._PROTECT_EYE_CORE_X * sep)))
        core_y = max(1, int(round(cls._PROTECT_EYE_CORE_Y * sep)))
        lift = cls._PROTECT_EYE_LIFT * sep
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
        cached = (outer_np, core_np)
        cls._eye_mask_cpu_cache[cache_key] = cached
        return cached

    @classmethod
    def _rebalance_masks_cpu(cls, shape):
        """Return selection and feathered masks for eye rebalancing on CPU, cached."""
        cache_key = (shape[0], shape[1], cls.model_template, cls._SIZE)
        cached = cls._rebalance_mask_cpu_cache.get(cache_key)
        if cached is not None:
            return cached
        from roop.face_util import swap_template_points
        pts = np.asarray(swap_template_points(cls._SIZE, cls.model_template), dtype=np.float32)
        if pts.shape[0] < 2:
            return None, None
        sep = float(np.linalg.norm(pts[1] - pts[0]))
        if not np.isfinite(sep) or sep < 2.0:
            return None, None
        rx = max(1, int(round(0.28 * sep)))
        ry = max(1, int(round(0.15 * sep)))
        selects, masks = [], []
        for eye in pts[:2]:
            m = np.zeros(shape, dtype=np.float32)
            center = (int(round(float(eye[0]))),
                      int(round(float(eye[1] - cls._PROTECT_EYE_LIFT * sep))))
            cv2.ellipse(m, center, (rx, ry), 0, 0, 360, 1.0, -1)
            selects.append(m > 0.5)
            masks.append(cv2.GaussianBlur(m, (0, 0), max(1.0, 0.035 * sep)))
        cached = (selects, masks)
        cls._rebalance_mask_cpu_cache[cache_key] = cached
        return cached

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
        outer_np, core_np = cls._eye_masks_cpu((int(image.shape[0]), int(image.shape[1])))
        if outer_np is None:
            return None, None
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
            outer, core = cls._eye_masks_cpu(restored.shape[:2])
            if outer is None:
                return restored
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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            selects, masks = cls._rebalance_masks_cpu(gray.shape)
            if selects is None:
                return image
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
            scores = []
            for sel in selects:
                values = lap[sel]
                scores.append(float(values.mean()) if values.size else 0.0)
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
        keeps the model output, chroma transfer, low-band recombination and eye
        protection on one device with no ORT->NumPy round-trip.

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
        from roop.optimized_processor import get_zero_copy_pool, zero_copy_enabled
        ring = get_zero_copy_pool(0) if zero_copy_enabled() else None
        if uses_tensorrt and not (ring is not None and self._trt_ring_enabled()):
            self._cuda_iob_available = False
            return None
        try:
            dtype = torch.float16 if self.in_dtype == np.float16 else torch.float32
            source = torch.from_numpy(src512).to('cuda', dtype=torch.float32, non_blocking=False)
            fidelity = torch.tensor(float(getattr(roop.globals, 'codeformer_fidelity', 0.5)),
                                    device=source.device, dtype=torch.float64)
            in0, in1 = self.model_inputs[0].name, self.model_inputs[1].name
            out_name = self.model_outputs[0].name
            input_element = np.float16 if dtype == torch.float16 else np.float32
            out_dtype = (torch.float16 if 'float16' in self.model_outputs[0].type
                         else torch.float32)
            if ring is not None:
                # Lock order everywhere: ring slot first, then the session lease.
                # The slot's input is a dense tensor `slot.load` gathers into (the
                # contiguity trap below is closed once, there), and the output stays
                # ORT-OWNED -- `output_is_static=False` -- because a Torch pointer
                # for this dynamic output is what striped the face. The returned
                # tensor is a fresh Torch copy, so the lease ends before the post-ops.
                with ring.lease(self._SIZE, dtype=dtype, out_dtype=out_dtype) as slot:
                    slot.load(source.permute(2, 0, 1).flip(0).unsqueeze(0)
                              .div(127.5).sub(1.0))
                    with exclusive(self.pool, self._session_lock,
                                   (self.session, self.io_binding)) as (sess, iob):
                        output = slot.run(sess, iob, in0, out_name, 0,
                                          output_is_static=False,
                                          extra_inputs={in1: fidelity})
                with self._lock:
                    self._ring_faces += 1
            else:
                # `.contiguous()` IS LOAD-BEARING. `permute` + `flip` leave a
                # non-contiguous view, and `bind_input` is handed a raw
                # `data_ptr()` -- ORT reads that pointer as though the tensor were
                # contiguous, so without this the network receives transposed
                # garbage. It still returns a plausible finite image, so the only
                # symptom was the collapse guard rejecting the face.
                x = (source.permute(2, 0, 1).flip(0).unsqueeze(0)
                     .div(127.5).sub(1.0).to(dtype).contiguous())
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
            _ss = self._structure_sharpen()
            if _ss > 0.0:
                restored = (restored + _ss *
                            (restored - self._blur_gpu(restored, 0.8))).clamp(0, 255)
            _dw = self._detail_weight()
            if _dw is not None and chroma != 1.0:
                _lb = self._lowband_blend_gpu(restored, source, _dw)
                if _lb is not restored:
                    with self._lock:
                        self._lowband += 1
                restored = _lb
            if chroma != 1.0 and self._eye_ops_enabled():
                # Both operators, in the host path's order. `_rebalance_eye_detail`
                # was absent from this chain entirely, so enabling the CUDA path
                # without it would have silently dropped a stage.
                restored = self._protect_swapped_eyes_gpu(restored, source)
                restored = self._rebalance_eye_detail_gpu(restored)
            scale = 1
            if restored.shape[1] < input_size:
                restored = _F.interpolate(restored.permute(2, 0, 1).unsqueeze(0),
                                          size=(input_size, input_size), mode='bicubic',
                                          align_corners=False)[0].permute(1, 2, 0)
            else:
                scale = max(1, int(restored.shape[1] / input_size))
            # One intentional egress boundary remains because the surrounding
            # merger/writer API is NumPy today.  No intermediate CPU copy is
            # made between ONNX output, chroma, low band, and eye protection.
            with self._lock:
                self._faces += 1
            self._cuda_iob_available = True
            return restored.round().to(torch.uint8).cpu().numpy(), scale
        except Exception as _degrade_error:
            # Old ORT builds expose CUDA OrtValues but not DLPack.  Fall back to
            # the established CPU binding path rather than failing a render.
            _swallowed("roop/processors/Enhance_UltraMax.py:1352", _degrade_error, "fallback continued")
            self._cuda_iob_available = False
            return None

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1

        input_size = temp_frame.shape[1]
        S = self._SIZE

        # The 512 source is needed twice — as the network's input and as the
        # colour / eye-protection / low-band reference — so it is built once.
        if temp_frame.shape[0] != S or temp_frame.shape[1] != S:
            src512 = cv2.resize(temp_frame, (S, S), interpolation=cv2.INTER_CUBIC)
        else:
            src512 = temp_frame

        cuda_result = self._run_cuda_postprocess(src512, input_size, temp_frame)
        if cuda_result is not None:
            res_frame, scale = cuda_result
            try:
                from roop.pipeline_manager import PipelineManager
                orig_crop = getattr(target_face, 'aligned_img', None)
                if orig_crop is None:
                    orig_crop = src512
                res_frame, _ = PipelineManager.get_instance().process_enhancer_crop(
                    enhanced_crop=res_frame,
                    original_crop=orig_crop,
                    target_face=target_face,
                )
            except Exception as _st7_err:
                _swallowed("roop/processors/Enhance_UltraMax.py:stage7_cuda", _st7_err, "stage7 fallback")
            return res_frame, scale

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

        # CodeFormer's mixed-precision output is numerically valid but can be
        # slightly soft at small eye, nose, and lip contours after the 512->
        # frame resize.  Recover a restrained structural edge response before
        # the periocular protection pass; this is an unsharp lift, not a second
        # generated face, and its low gain avoids recreating the old halo.
        _ss = self._structure_sharpen()
        if _ss > 0.0:
            base = restored.astype(np.float32)
            blur = cv2.GaussianBlur(base, (0, 0), 0.8)
            restored = np.clip(base + _ss * (base - blur),
                               0.0, 255.0).astype(np.uint8)

        # Do this after the colour pass: the protected source pixels should
        # retain the same chroma as the rest of the swapped crop, while the
        # restored cheeks/nose keep UltraMax's luminance and detail.
        # The diagnostic chroma=1 mode is intentionally bit-faithful to the
        # raw CodeFormer output for precision comparisons; do not apply the
        # visual eye-protection/rebalance operators in that mode.
        did_gpu_post = False
        if _TORCH_CUDA and getattr(self, 'devicename', '') == 'cuda':
            try:
                res_t = torch.from_numpy(restored).to('cuda', dtype=torch.float32, non_blocking=False)
                src_t = torch.from_numpy(src512).to('cuda', dtype=torch.float32, non_blocking=False)
                _dw = self._detail_weight()
                if _dw is not None and _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA) != 1.0:
                    _lb = self._lowband_blend_gpu(res_t, src_t, _dw)
                    if _lb is not res_t:
                        with self._lock:
                            self._lowband += 1
                    res_t = _lb
                if (_env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA) != 1.0
                        and self._eye_ops_enabled()):
                    res_t = self._protect_swapped_eyes_gpu(res_t, src_t)
                    res_t = self._rebalance_eye_detail_gpu(res_t)
                restored = res_t.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
                did_gpu_post = True
            except Exception as _gpu_err:
                did_gpu_post = False

        if not did_gpu_post:
            _dw = self._detail_weight()
            if _dw is not None and _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA) != 1.0:
                try:
                    from roop import enhance_blend
                    restored = enhance_blend.frequency_blend(src512, restored, weight=_dw)
                    with self._lock:
                        self._lowband += 1
                except Exception as exc:
                    if not Enhance_UltraMax._warned_lowband:
                        Enhance_UltraMax._warned_lowband = True
                        print(f"[UltraMax] low-band recombination failed ({type(exc).__name__}: {exc}); "
                              f"full restorer output used", flush=True)
            if (_env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA) != 1.0
                    and self._eye_ops_enabled()):
                restored = self._protect_swapped_eyes(restored, src512)
                restored = self._rebalance_eye_detail(restored)

        with self._lock:
            self._faces += 1
        try:
            from roop.pipeline_manager import PipelineManager
            orig_crop = getattr(target_face, 'aligned_img', None)
            if orig_crop is None:
                orig_crop = src512
            restored, _ = PipelineManager.get_instance().process_enhancer_crop(
                enhanced_crop=restored,
                original_crop=orig_crop,
                target_face=target_face,
            )
        except Exception as _st7_err:
            _swallowed("roop/processors/Enhance_UltraMax.py:stage7_cpu", _st7_err, "stage7 fallback")
        return sized(restored, input_size)

    # ── reporting ────────────────────────────────────────────────────────────
    def cost_summary(self):
        with self._lock:
            f, lowband = self._faces, self._lowband
        if not f:
            return None
        c = _env_float('ROOP_ULTRAMAX_CHROMA', self._CHROMA)
        colour = ("swapper chrominance" if c <= 0.0
                  else f"chrominance {c:g} toward CodeFormer's own")
        path = ('CUDA tensor post-process' if self._cuda_iob_available
                else 'compatibility host path')
        _dw = self._detail_weight()
        gate = (f"; swap low band + {_dw:g} x restorer detail on {lowband}" if _dw is not None
                else "; full restorer output (legacy, low band not anchored to the swap)")
        ring = f", {self._ring_faces} through the zero-copy ring" if self._ring_faces else ""
        return (f"[UltraMax] {f} faces restored (codeformer.fp16, {path}{ring}, "
                f"{colour}){gate}")
