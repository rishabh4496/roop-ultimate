"""Boundary and high-frequency stability for the paste-back path.

Three separate mechanisms live here.  They are grouped because they attack one
symptom -- a swapped face that shimmers frame to frame and shows a hard rim at
the hairline/jaw -- from the three places the pipeline can actually cause it.

    1. AdaptiveLandmarkSmoother   the geometry that defines the crop and the hull
    2. soft_distance_matte        the shape of the alpha ramp at the boundary
    3. HighFrequencyFlowStabilizer the restorer's per-frame texture hallucination

WHAT WAS ALREADY HERE, AND WHY THIS IS NOT A DUPLICATE
------------------------------------------------------
`roop/one_euro.py` already smooths the **5-point** kps (`KpsStabilizer`), the
enhanced crop as a whole (`EnhancerStabilizer`) and the mask
(`MaskStabilizer`); `roop/occlusion_mask.py` already flow-warps the OCCLUSION
mask.  None of them covers what is here:

* THE 5-POINT KPS AND THE DENSE LANDMARKS ARE FILTERED SEPARATELY, and they
  must not be.  The 5 points drive the alignment matrix (where the swap is
  SAMPLED); the 106 points drive `procmgr_masking.landmark_hull` (where the
  matte is DRAWN).  Two code paths, two different defects:

    - `temporal_detection` ON (the shipped default): the tracking pre-pass
      `procmgr_tracking._build_temporal_faces` smooths BOTH -- but with two
      independent OneEuro filters holding independent state.  OneEuro's cutoff
      is `min_cutoff + beta*|dx_hat|` PER ELEMENT, and a 5-point array and a
      106-point array do not yield the same |dx_hat| on the same head motion,
      so the crop and its own outline settle at different rates.

    - `temporal_detection` OFF: `ProcessMgr._apply_stab` replaces `face.kps`
      alone and `landmark_2d_106` is genuinely never filtered.

  Either way the paste boundary drifts against the crop by a pixel or two
  during motion, opening and closing a sliver of untouched plate at the jaw.
  This smooths both under ONE beta derived from the 5-point velocity, so they
  cannot disagree.  CORRECTION TO AN EARLIER DRAFT OF THIS FILE: it claimed the
  dense landmarks were unsmoothed outright.  That is true only on the second
  path; on the shipped default they were already filtered, just not coupled.

* `blur_area` builds the ramp as erode + Gaussian.  A Gaussian of a binary
  region is not a distance ramp: its width collapses wherever the matte is
  locally thin or convex, so the seam is soft on the cheek and hard at the
  temple in the same frame.  A distance transform gives a ramp whose width is
  the same everywhere by construction.

* `EnhancerStabilizer` blends the WHOLE crop with a scalar alpha and no motion
  compensation.  That is the right model for a still face and the wrong one for
  a moving one -- it must back off exactly when the restorer flickers most.
  Warping the previous HIGH-FREQUENCY layer along dense flow removes the
  head-motion term, so the texture can be blended while the face moves without
  ghosting the face itself.

THE CONTIGUITY GUARD IS NOT OPTIONAL
------------------------------------
`ProcessMgr.read_frames_thread` dispatches frames strict round-robin
(`_thr = num_frame % num_threads`), so at N workers NO worker sees two adjacent
frames.  Every "previous frame" here is therefore checked to be
`frame_index - 1` for that track before it is used.  Without that check a
"temporal" filter blends content from up to N frames away and calls it
anti-flicker -- the trap that made UltraMax's anchor cache warp a face from
0.67 s away over the current one (Session Log 2026-08-22->08-23 section 3).

Under plain round-robin dispatch these filters therefore DECLINE on most faces
and say so; they apply in the parallel-stabilization path, whose blocks are
contiguous per worker, and in the sequential path.  `summary_line()` prints
applied/declined at the end of every run, because a silent no-op and a working
filter are indistinguishable in the output -- the single most expensive
confusion in this project's history.
"""

import math
import os
import threading

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# environment helpers (same contract as roop/occlusion_mask.py)
# ---------------------------------------------------------------------------

def _env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ('0', 'off', 'false', 'no')


def _env_float(name, default, lo, hi):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, value))


class _Counters:
    """Tallies shared by an engine and every per-block clone of it.

    `clone_for_block` is a shallow copy, so a clone that kept its counters as
    plain attributes would tally into its own instance and be thrown away with
    the block -- leaving `summary_line()` on the shared engine reporting zeros
    for a filter that ran on every face. That is indistinguishable from the
    filter never having run, which is precisely the reporting failure these
    counters exist to make impossible.

    So the counters live in one object that `copy.copy` passes by REFERENCE,
    with its own lock. The clone replaces `_lock` and `_states` (which must be
    per-block) and deliberately does not replace this.
    """

    __slots__ = ('_lock', '_values')

    def __init__(self):
        self._lock = threading.Lock()
        self._values = {}

    def bump(self, name, amount=1):
        with self._lock:
            self._values[name] = self._values.get(name, 0) + amount

    def get(self, name, default=0):
        with self._lock:
            return self._values.get(name, default)

    def snapshot(self):
        with self._lock:
            return dict(self._values)

    def clear(self):
        with self._lock:
            self._values.clear()


# ===========================================================================
# 1. Inter-frame landmark smoothing
# ===========================================================================

class AdaptiveLandmarkSmoother:
    """Velocity-adaptive EMA over a face's 5-point kps AND its dense landmarks.

        L_smooth_t = beta * L_t + (1 - beta) * L_smooth_{t-1}

    with beta rising toward `beta_fast` on rapid head motion (react now, do not
    lag) and falling toward `beta_slow` when the face is nearly still (kill
    detector micro-jitter).

    THREE THINGS THE PLAIN SPECIFICATION GETS WRONG ON REAL FOOTAGE, and what
    is done about each.  All three are corrections in the same direction: the
    naive form reintroduces the jitter it was written to remove.

    (a) A PIXEL THRESHOLD IS NOT A MOTION THRESHOLD.  4 px/frame is a violent
        turn on a face 60 px wide in a wide shot and is imperceptible on a
        500 px close-up.  Velocity here is divided by the face's own extent, so
        `velocity_threshold` is a fraction of face size and means the same
        thing at every resolution and every shot size.  This is the same
        normalisation `KpsStabilizer.match_scale` already uses to associate
        tracks, for the same reason.

    (b) A HARD SWITCH AT THE THRESHOLD IS ITSELF A VISIBLE ARTEFACT.  Stepping
        beta 0.35 -> 0.9 the instant a scalar crosses a line changes the
        filter's lag discontinuously, so the face lurches on the crossing
        frame; and a face hovering AT the threshold alternates regimes frame to
        frame, which is a new flicker with a new period.  This project has
        already paid for that shape of bug once -- see
        `memory/roll-is-not-yaw.md`, "a latch over a wrong metric still
        flickers".  So beta ramps smoothly (smoothstep) across a band centred
        on the threshold.  Well below the band beta is exactly `beta_slow` and
        well above it exactly `beta_fast`, so the specified endpoints are
        reproduced; only the crossing is continuous.

    (c) THE 5-POINT AND DENSE SETS MUST SHARE ONE BETA.  The 5 points drive the
        alignment matrix (where the swap is sampled) and the 106/68 points
        drive `landmark_hull` (where the matte is drawn).  Filtering them
        independently lets them settle at different rates, so the crop and its
        own outline disagree during motion -- which opens and closes a sliver
        of untouched plate at the jaw, i.e. the exact seam this is meant to
        remove.  Beta is computed ONCE from the 5-point velocity and applied to
        both.

    State is keyed by track id (the tracker's answer when there is one) and
    guarded on frame contiguity; see the module docstring.
    """

    # Beta at the two ends of the ramp.
    BETA_SLOW = 0.35
    BETA_FAST = 0.90
    # Velocity, as a fraction of face extent per frame, at the centre of the
    # ramp. 0.02 is ~10 px/frame on a 500 px face: a deliberate head turn, not
    # detector noise (measured jitter on a still face is well under 0.005).
    VELOCITY_THRESHOLD = 0.02
    # Half-width of the ramp, in the same units. The ramp therefore spans
    # roughly 0.01 .. 0.03; outside it the two constant regimes hold exactly.
    VELOCITY_BAND = 0.01
    # Drop a track unseen for this many frames.
    MAX_MISSING = 8
    MAX_TRACKS = 64

    def __init__(self, beta_slow=None, beta_fast=None, velocity_threshold=None,
                 velocity_band=None, enabled=True, max_tracks=None):
        self.enabled = bool(enabled)
        self.beta_slow = float(self.BETA_SLOW if beta_slow is None else beta_slow)
        self.beta_fast = float(self.BETA_FAST if beta_fast is None else beta_fast)
        self.velocity_threshold = float(self.VELOCITY_THRESHOLD
                                        if velocity_threshold is None
                                        else velocity_threshold)
        self.velocity_band = max(1e-6, float(self.VELOCITY_BAND
                                             if velocity_band is None
                                             else velocity_band))
        self.max_tracks = int(max_tracks or self.MAX_TRACKS)
        self._states = {}
        self._order = []
        self._lock = threading.RLock()
        self._counts = _Counters()

    @classmethod
    def from_env(cls, enabled=None):
        if enabled is None:
            enabled = _env_flag('ROOP_LANDMARK_SMOOTH', True)
        return cls(
            beta_slow=_env_float('ROOP_LANDMARK_BETA_SLOW', cls.BETA_SLOW, 0.01, 1.0),
            beta_fast=_env_float('ROOP_LANDMARK_BETA_FAST', cls.BETA_FAST, 0.01, 1.0),
            velocity_threshold=_env_float('ROOP_LANDMARK_VELOCITY',
                                          cls.VELOCITY_THRESHOLD, 0.0005, 1.0),
            enabled=bool(enabled),
        )

    # -- beta ---------------------------------------------------------------
    def beta_for(self, velocity):
        """Smoothstep from `beta_slow` to `beta_fast` across the ramp band.

        `velocity` is already normalised by face extent. Returns exactly
        `beta_slow` below the band and exactly `beta_fast` above it.
        """
        lo = self.velocity_threshold - self.velocity_band
        hi = self.velocity_threshold + self.velocity_band
        if velocity <= lo:
            return self.beta_slow
        if velocity >= hi:
            return self.beta_fast
        t = (velocity - lo) / (hi - lo)
        t = t * t * (3.0 - 2.0 * t)          # smoothstep: C1 at both ends
        return self.beta_slow + (self.beta_fast - self.beta_slow) * t

    # -- block-parallel lifecycle -------------------------------------------
    def warmup_frames(self, eps=0.01):
        """Frames a parallel block must discard before this filter's seed is gone.

        The filter is an EMA, so the seed's weight after W frames is
        (1 - beta)^W and the SLOWEST case sets the boundary -- that is
        `beta_slow`, the still-face regime, because `beta_for` only ever raises
        beta above it. Solving (1 - beta_slow)^W <= eps at the default 0.35
        gives 11 frames.
        """
        from roop.one_euro import ema_warmup_frames
        return ema_warmup_frames(self.beta_slow, eps)

    def clone_for_block(self):
        """An instance for one contiguous parallel block: same configuration,
        empty state, its own lock.

        Every writer of `_states` runs in the swap phase, so a block starts
        from nothing and re-primes from its own warm-up frames. Two blocks can
        then never advance one track's landmark history out of order.

        NOTE this engine deliberately does NOT join `_want_temporal_ordered` in
        ProcessMgr. The ordered engines produce WRONG output when frames arrive
        out of order, so they force a sequential fallback; this one declines
        (returns the raw landmarks) and counts it. Forcing `threads = 1` for a
        filter that degrades safely would buy nothing and cost 3x throughput --
        the exact regression Session Log 2026-08-25 Part 3 documents.
        """
        import copy as _copy_module
        clone = _copy_module.copy(self)
        clone._lock = threading.RLock()
        clone._states = {}
        clone._order = []
        return clone

    # -- bookkeeping --------------------------------------------------------
    def _touch(self, key):
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        while len(self._order) > self.max_tracks:
            self._states.pop(self._order.pop(0), None)

    def reset(self):
        with self._lock:
            self._states.clear()
            self._order.clear()
        self._counts.clear()

    @staticmethod
    def _extent(kps):
        """The face's own scale, used to normalise velocity.

        Interocular distance is the natural unit but collapses toward zero on a
        full profile, which would make the normalised velocity explode and pin
        beta at `beta_fast` for the whole shot -- i.e. no smoothing precisely
        where landmarks are noisiest. The kps bounding extent does not vanish
        on a profile, so it is the floor.
        """
        eye = float(np.linalg.norm(kps[0] - kps[1]))
        span = max(float(np.ptp(kps[:, 0])), float(np.ptp(kps[:, 1])))
        return max(eye, span, 1.0)

    # -- the filter ---------------------------------------------------------
    def smooth(self, kps, dense=None, track_id=None, frame_index=None):
        """Return `(kps_smoothed, dense_smoothed, beta)`.

        Never raises. Any failure returns the inputs unchanged and is counted:
        this is a quality layer, and losing a frame to it would be a far worse
        trade than leaving one face unsmoothed.
        """
        if not self.enabled or kps is None:
            return kps, dense, 1.0
        try:
            cur = np.asarray(kps, dtype=np.float32)
            if cur.shape != (5, 2) or not np.all(np.isfinite(cur)):
                return kps, dense, 1.0
            if track_id is None or frame_index is None:
                self._counts.bump('skipped_no_key')
                return kps, dense, 1.0

            cur_dense = None
            if dense is not None:
                d = np.asarray(dense, dtype=np.float32)
                if d.ndim == 2 and d.shape[0] >= 5 and np.all(np.isfinite(d)):
                    cur_dense = d

            key = ('track', track_id)
            out_kps, out_dense, beta = cur, cur_dense, 1.0

            with self._lock:
                state = self._states.get(key)
                contiguous = (state is not None
                              and state['frame_index'] == frame_index - 1)
                if contiguous:
                    prev = state['kps']
                    extent = self._extent(cur)
                    # Mean per-point displacement, normalised by face scale.
                    velocity = float(np.mean(
                        np.linalg.norm(cur - prev, axis=1))) / extent
                    beta = self.beta_for(velocity)
                    out_kps = (beta * cur + (1.0 - beta) * prev).astype(np.float32)
                    self._counts.bump('applied')
                    self._counts.bump('beta_sum', beta)

                    prev_dense = state.get('dense')
                    if (cur_dense is not None and prev_dense is not None
                            and prev_dense.shape == cur_dense.shape):
                        # SAME beta -- see (c) in the class docstring.
                        out_dense = (beta * cur_dense
                                     + (1.0 - beta) * prev_dense).astype(np.float32)
                        self._counts.bump('dense_applied')
                elif state is None:
                    self._counts.bump('seeded')
                else:
                    # Normal under round-robin dispatch, not an error.
                    self._counts.bump('skipped_noncontiguous')

                self._states[key] = {
                    'kps': np.array(out_kps, dtype=np.float32, copy=True),
                    'dense': (None if out_dense is None else
                              np.array(out_dense, dtype=np.float32, copy=True)),
                    'frame_index': int(frame_index),
                }
                self._touch(key)
            return out_kps, out_dense, beta
        except Exception:
            self._counts.bump('errors')
            return kps, dense, 1.0

    # -- reporting ----------------------------------------------------------
    def stats(self):
        c = self._counts.snapshot()
        applied = c.get('applied', 0)
        return {'applied': applied,
                'dense_applied': c.get('dense_applied', 0),
                'seeded': c.get('seeded', 0),
                'skipped_noncontiguous': c.get('skipped_noncontiguous', 0),
                'skipped_no_key': c.get('skipped_no_key', 0),
                'errors': c.get('errors', 0),
                'mean_beta': (c.get('beta_sum', 0.0) / applied
                              if applied else 0.0)}

    def summary_line(self):
        s = self.stats()
        total = (s['applied'] + s['seeded'] + s['skipped_noncontiguous']
                 + s['skipped_no_key'])
        if total == 0:
            return None
        return ('[LandmarkSmooth] adaptive EMA beta %.2f..%.2f: applied %d/%d '
                '(%.1f%%), dense %d, mean beta %.3f, seeded %d, '
                'non-contiguous %d, no-key %d, errors %d'
                % (self.beta_slow, self.beta_fast, s['applied'], total,
                   100.0 * s['applied'] / total, s['dense_applied'],
                   s['mean_beta'], s['seeded'], s['skipped_noncontiguous'],
                   s['skipped_no_key'], s['errors']))


# ===========================================================================
# 2. Soft distance-transform matte
# ===========================================================================

# Boundary ramp width, in frame pixels, at the two ends of the face-scale range
# it is interpolated over. A fixed margin reads as a hard cut on a close-up and
# swallows the jaw on a wide shot.
MARGIN_MIN_PX = 8.0
MARGIN_MAX_PX = 18.0
# Face extents (sqrt of the matte's bounding area) those two margins belong to.
MARGIN_SCALE_LO = 128.0
MARGIN_SCALE_HI = 512.0
# Sigmoid steepness over the normalised ramp. 8 puts ~80% of the transition in
# the middle half of the margin, which is the visible difference from a linear
# ramp: no crease where the ramp meets solid matte, no visible onset where it
# meets zero.
SIGMOID_K = 8.0


def adaptive_margin_px(face_extent_px,
                       margin_min=MARGIN_MIN_PX, margin_max=MARGIN_MAX_PX):
    """Boundary ramp width for a face of this on-screen size, in pixels."""
    try:
        extent = float(face_extent_px)
    except (TypeError, ValueError):
        return margin_min
    if not math.isfinite(extent):
        return margin_min
    t = (extent - MARGIN_SCALE_LO) / (MARGIN_SCALE_HI - MARGIN_SCALE_LO)
    t = min(1.0, max(0.0, t))
    return float(margin_min + (margin_max - margin_min) * t)


def _normalised_sigmoid(t, k=SIGMOID_K):
    """Sigmoid on t in [0,1], rescaled so f(0)==0 and f(1)==1 exactly.

    The rescale is the load-bearing part. A raw logistic evaluated on [0,1]
    lands at ~0.018 and ~0.982, so it would leave a 2%-of-alpha STEP where the
    ramp meets the outside and another where it meets the interior -- two new
    hard edges either side of the soft one, which is worse than the Gaussian it
    replaces. Rescaling removes both by construction.
    """
    # float32 throughout, INCLUDING the two endpoint constants. Computing them
    # in float64 while `raw` is float32 leaves f(1) at 0.99999994 rather than
    # 1.0 -- a 6e-8 alpha deficit over the whole face interior, which is
    # harmless numerically but means the composite is no longer bit-identical
    # to a plain paste where alpha should be saturated. That is exactly the
    # kind of "changed nothing, but not provably nothing" that costs a session
    # to re-verify later.
    kf = np.float32(k)
    s0 = np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(0.5) * kf))
    s1 = np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-0.5) * kf))
    t = np.asarray(t, dtype=np.float32)
    raw = np.float32(1.0) / (np.float32(1.0) + np.exp(-kf * (t - np.float32(0.5))))
    return np.clip((raw - s0) / (s1 - s0), 0.0, 1.0)


def soft_distance_matte(binary_matte, face_extent_px=None, margin_px=None,
                        k=SIGMOID_K, as_uint8=False):
    """Continuous 1.0-at-the-centre -> 0.0-at-the-rim alpha from a hard matte.

    `binary_matte` is the uint8 hull/ellipse product built in
    `procmgr_masking.paste_upscale`; the return is float32 in [0, 1], or
    0-255 uint8 when `as_uint8` (which is what `blur_area` needs, and which
    saves a full-frame float32 -> uint8 conversion).

    WHY A DISTANCE TRANSFORM AND NOT A BIGGER BLUR. `blur_area`'s Gaussian is
    applied to the binary region, so the ramp it produces is only as wide as
    the local geometry allows: at a convex corner (temple, chin tip) the blur
    of a half-plane is halved, and in a locally thin part of the matte the two
    sides' ramps overlap and the alpha never reaches 1. That is why the seam is
    soft along the cheek and hard at the temple in the SAME frame -- one
    setting, two visible widths. `cv2.distanceTransform` measures the true
    distance to the boundary, so the ramp is `margin_px` wide everywhere
    regardless of curvature, and it is exactly 1.0 in the interior rather than
    asymptotically near it.

    EVERYTHING BELOW RUNS IN THE FACE'S OWN BOX, and that is a measurement
    result rather than tidiness. Profiled at 4K, the transform itself costs
    2.9 ms while the FULL-FRAME numpy around it cost far more than the operator
    it was wrapping:

        np.where over the frame        14.6 ms      -> cv2.boundingRect  0.5 ms
        (mask > 127).astype            4.1 ms       -> cv2.threshold     ~0.3 ms
        clip(a * 255).astype(uint8)    20.7 ms      -> done in the ROI

    The first draft was 102 ms per paste at 4K and 11 ms at 720p, against the
    Gaussian's 7.3 and 1.7 -- a quality layer costing more than three enhancer
    calls. None of that was the distance transform.
    """
    mask = np.asarray(binary_matte)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    out_dtype = np.uint8 if as_uint8 else np.float32

    # cv2.threshold rather than a numpy comparison + astype: same result, and
    # it is one C pass instead of a bool temporary the size of the frame.
    _, hard = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    x, y, bw, bh = cv2.boundingRect(hard)
    if bw == 0 or bh == 0:
        return np.zeros(mask.shape, dtype=out_dtype)

    if margin_px is None:
        if face_extent_px is None:
            face_extent_px = math.sqrt(float(bw) * float(bh))
        margin_px = adaptive_margin_px(face_extent_px)
    margin = max(1.0, float(margin_px))

    # EXACTNESS. Alpha saturates at 1.0 for every pixel further than `margin`
    # from the boundary, so only distances up to `margin` need to be right. A
    # box expanded by margin + 2 therefore contains, for every boundary pixel,
    # every zero that could be its nearest -- the answer inside the box is the
    # same number a full-frame transform produces, not an approximation of it.
    # Beyond the box the matte is empty, so alpha is 0 there by construction.
    pad = int(math.ceil(margin)) + 2
    h, w = hard.shape
    ry0, ry1 = max(0, y - pad), min(h, y + bh + pad)
    rx0, rx1 = max(0, x - pad), min(w, x + bw + pad)

    # DIST_L2 with mask size 5 is the accurate variant; size 3 uses the cheaper
    # 3x3 chamfer approximation, whose ~2% error is a visible wobble in the ramp
    # position at these widths.
    dist = cv2.distanceTransform(np.ascontiguousarray(hard[ry0:ry1, rx0:rx1]),
                                 cv2.DIST_L2, 5)
    ramp = _normalised_sigmoid(np.clip(dist / margin, 0.0, 1.0), k)
    out = np.zeros(mask.shape, dtype=out_dtype)
    out[ry0:ry1, rx0:rx1] = (np.rint(ramp * 255.0).astype(np.uint8)
                             if as_uint8 else ramp)
    return out


def boundary_illumination_match(paste, target, alpha, strength=1.0,
                                sigma_color=32.0, sigma_space=9.0,
                                downscale=4):
    """Grade the paste's LOW frequencies toward the plate, at the rim only.

    A residual seam that survives a perfect alpha ramp is an ILLUMINATION step:
    the restorer regrades the face, so the swapped skin meets the untouched
    skin at a slightly different brightness and the eye reads the ramp as an
    edge however soft it is.

    Two properties make this safe to run on identity-bearing pixels:

      * only the LOW-frequency difference is transferred, from a bilateral
        filter, so pores and lashes are untouched and no target texture is
        copied onto the face (the mistake `procmgr_color.apply_detail_transfer`
        has an edge-stop gate for);
      * it is weighted by `1 - alpha_interior`, so it is full strength at the
        rim and ZERO in the face centre. The identity region is bit-unchanged.

    The bilateral rather than a Gaussian is what keeps the correction from
    bleeding the background's brightness across the jaw into the face: a
    Gaussian at this radius averages across the very edge being fixed, which
    reintroduces a halo one ramp-width inside the old one.

    Related but not the same as `temporal_compositing.composite_multiband`,
    which adapts the low band across the WHOLE matte at a fixed
    `color_strength`. This is rim-only and derived from the alpha ramp, so the
    two do not double-apply on the interior.
    """
    if strength <= 0.0:
        return paste
    p = np.asarray(paste, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if p.shape != t.shape or p.ndim != 3 or a.shape != p.shape[:2]:
        return paste

    h, w = p.shape[:2]
    # The low band is being ESTIMATED, so the grid it is estimated on only has
    # to resolve illumination, not detail. A fixed downscale prices a 4K face
    # sixteen times a 720p one for no extra information; targeting a ~96 px
    # working grid instead makes the bilateral's cost flat in ROI size.
    ds = max(int(downscale), int(round(max(h, w) / 96.0)))
    sh, sw = max(8, h // ds), max(8, w // ds)
    # uint8: cv2.bilateralFilter has a dedicated 8-bit path, and the inputs are
    # 8-bit images anyway. sigma_space is expressed in the DOWNSCALED grid,
    # which is what bounds the cost; scaling it with `ds` keeps the filter's
    # reach constant in ROI terms.
    small_p = cv2.resize(p, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.uint8)
    small_t = cv2.resize(t, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.uint8)
    low_p = cv2.bilateralFilter(small_p, -1, sigma_color, sigma_space / ds)
    low_t = cv2.bilateralFilter(small_t, -1, sigma_color, sigma_space / ds)

    delta = low_t.astype(np.float32) - low_p.astype(np.float32)
    # Bound the correction: an occluder or a blown highlight on the rim would
    # otherwise drag a large step onto the face.
    np.clip(delta, -32.0, 32.0, out=delta)
    delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)

    # THE RIM WEIGHT IS COMPUTED AT FULL RESOLUTION, from the true alpha, and
    # that is not an oversight to optimise away later. Weighting on the small
    # grid and upsampling is ~2 ms cheaper and was tried: downsampling alpha
    # averages rim pixels into the saturated interior, and the bilinear
    # upsample spreads them further, so the correction becomes non-zero inside
    # the face. That breaks the one property that makes it safe to run this on
    # identity-bearing pixels at all -- `test_identity_region_is_bit_unchanged`
    # is what caught it. Exactness here is the feature; the 2 ms is not.
    #
    # Rim weight: inside the matte (a > 0) but not yet at full alpha, so it is
    # EXACTLY zero wherever alpha saturates.
    rim = np.clip(a * (1.0 - a) * 4.0, 0.0, 1.0) * float(strength)
    return np.clip(p + delta * rim[..., None], 0, 255).astype(paste.dtype)


# ===========================================================================
# 3. High-frequency temporal stabilizer
# ===========================================================================

class HighFrequencyFlowStabilizer:
    """Flow-compensated EMA over the restorer's high-frequency layer.

    GPEN / CodeFormer / UltraMax synthesise pores, lashes and stubble from a
    generative prior. The prior is evaluated independently every frame, so on a
    face that is barely moving the SAME skin is given a slightly different pore
    field 25 times a second -- the boiling that reads as "the skin is alive".

    `EnhancerStabilizer` (roop/one_euro.py) damps that by blending whole crops
    with a motion-adaptive scalar, which works when the head is still and must
    switch itself off when it is not, because blending unregistered crops
    ghosts. This registers first:

        hf_t   = crop_t - blur(crop_t)
        hf_out = (1 - w) * hf_t + w * warp(hf_{t-1}, flow_{t-1 -> t})
        out    = blur(crop_t) + hf_out

    Only the HIGH band is blended, so illumination, colour and expression come
    entirely from the current frame and cannot ghost. The flow removes the
    head-motion term, so `w` does not have to be reduced on motion the way a
    whole-crop blend does.

    `w` defaults to 0.15: enough to break the frame-to-frame independence of
    the prior's noise (a 15% carry gives the pore field a ~6-frame memory)
    while leaving 85% of the current frame's detail, so genuinely new texture
    -- a turn revealing a new cheek -- is not held back.

    Guards, in order of how much they matter:

      * FRAME CONTIGUITY. See the module docstring. Without it this warps
        texture from N frames away onto the current face.
      * FLOW RESIDUAL. If the warped previous crop does not predict the
        current one, the flow is wrong (occlusion, cut, a fast turn) and the
        carry is dropped for that face rather than smeared.
      * SHAPE. A crop that changed size between frames cannot share a state.
    """

    FLOW_SIZE = 64          # flow is solved on a 64x64 grid; see _dense_flow
    WEIGHT = 0.15
    HF_SIGMA = 1.0          # pore scale; larger starts carrying structure
    RESET_RESIDUAL = 18.0   # mean abs prediction error, 0-255, above which we bail
    MAX_TRACKS = 64

    def __init__(self, weight=None, flow_size=None, hf_sigma=None,
                 reset_residual=None, enabled=True, max_tracks=None):
        self.enabled = bool(enabled)
        self.weight = min(1.0, max(0.0, float(self.WEIGHT if weight is None
                                              else weight)))
        self.flow_size = int(flow_size or self.FLOW_SIZE)
        self.hf_sigma = float(self.HF_SIGMA if hf_sigma is None else hf_sigma)
        self.reset_residual = float(self.RESET_RESIDUAL if reset_residual is None
                                    else reset_residual)
        self.max_tracks = int(max_tracks or self.MAX_TRACKS)
        self._states = {}
        self._order = []
        self._lock = threading.RLock()
        self._tls = threading.local()
        self._counts = _Counters()

    @classmethod
    def from_env(cls, enabled=None):
        if enabled is None:
            enabled = _env_flag('ROOP_HF_FLOW_STABILIZE', True)
        return cls(
            weight=_env_float('ROOP_HF_FLOW_WEIGHT', cls.WEIGHT, 0.0, 0.9),
            flow_size=int(_env_float('ROOP_HF_FLOW_SIZE', cls.FLOW_SIZE, 32, 256)),
            reset_residual=_env_float('ROOP_HF_FLOW_RESIDUAL',
                                      cls.RESET_RESIDUAL, 1.0, 255.0),
            enabled=bool(enabled),
        )

    # -- flow ---------------------------------------------------------------
    def _flow_engine(self):
        """One DIS instance per thread.

        cv2's DISOpticalFlow holds internal scratch buffers and is NOT
        thread-safe; sharing one across the worker pool corrupts the field
        rather than raising, which presents as texture that jitters WORSE with
        the filter on than off. Same reasoning, same fix, as
        `occlusion_mask.TemporalMaskSmoother._flow_engine`.
        """
        engine = getattr(self._tls, 'dis', None)
        if engine is None:
            try:
                engine = cv2.DISOpticalFlow_create(
                    cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            except (AttributeError, cv2.error):
                engine = False   # False, not None: probe cv2 once per thread
            self._tls.dis = engine
        return engine or None

    def _dense_flow(self, cur_small, prev_small):
        """Backward flow: for each pixel of the CURRENT crop, where it was.

        Argument order is load-bearing. `calc(a, b)` gives, per pixel of `a`,
        the displacement to its match in `b`; `remap` samples the SOURCE at a
        position given per DESTINATION pixel, and the destination is the
        current frame. Passing these the other way round yields a field that
        looks entirely plausible and drags the texture the wrong way.
        """
        engine = self._flow_engine()
        if engine is not None:
            return engine.calc(cur_small, prev_small, None)
        return cv2.calcOpticalFlowFarneback(
            cur_small, prev_small, None, 0.5, 2, 13, 2, 5, 1.1, 0)

    @staticmethod
    def _warp(image, flow, shape):
        h, w = int(shape[0]), int(shape[1])
        fh, fw = flow.shape[:2]
        if (fh, fw) != (h, w):
            # Scale the field to image resolution AND scale the vectors with
            # it: one small-grid pixel of displacement is w/fw image pixels.
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow = flow.copy()
            flow[..., 0] *= float(w) / float(fw)
            flow[..., 1] *= float(h) / float(fh)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        return cv2.remap(image, grid_x + flow[..., 0], grid_y + flow[..., 1],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _observation(self, crop):
        image = np.asarray(crop)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (self.flow_size, self.flow_size),
                          interpolation=cv2.INTER_AREA)

    def _touch(self, key):
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        while len(self._order) > self.max_tracks:
            self._states.pop(self._order.pop(0), None)

    def reset(self):
        with self._lock:
            self._states.clear()
            self._order.clear()
        self._counts.clear()

    # -- block-parallel lifecycle -------------------------------------------
    def warmup_frames(self, eps=0.01):
        """The carried fraction is `weight`, so the current frame enters with
        (1 - weight) and the seed decays as weight^W. At the default 0.15 that
        is 3 frames -- this filter is nearly free to prime, which is why it can
        be defaulted on without widening the parallel blocks."""
        from roop.one_euro import ema_warmup_frames
        return ema_warmup_frames(1.0 - self.weight, eps)

    def clone_for_block(self):
        """Fresh state per contiguous block; see AdaptiveLandmarkSmoother.

        `_tls` is rebuilt rather than shared: the copy would otherwise carry the
        parent's thread-local DIS handles, and a cv2 DISOpticalFlow used from
        two threads corrupts its scratch buffers silently.
        """
        import copy as _copy_module
        clone = _copy_module.copy(self)
        clone._lock = threading.RLock()
        clone._tls = threading.local()
        clone._states = {}
        clone._order = []
        return clone

    # -- the filter ---------------------------------------------------------
    def stabilize(self, crop, track_id=None, frame_index=None):
        """Return the crop with its high-frequency layer temporally carried.

        Never raises; a crop that failed to stabilize is returned unchanged and
        counted.
        """
        if not self.enabled or crop is None or self.weight <= 0.0:
            return crop
        try:
            cur = np.asarray(crop)
            if cur.ndim != 3 or cur.size == 0:
                return crop
            if track_id is None or frame_index is None:
                self._counts.bump('skipped_no_key')
                return crop

            key = ('track', track_id)
            cur_f = cur.astype(np.float32)
            low = cv2.GaussianBlur(cur_f, (0, 0), sigmaX=self.hf_sigma,
                                   sigmaY=self.hf_sigma)
            hf = cur_f - low
            small = self._observation(cur)
            out = cur
            out_hf = hf

            with self._lock:
                state = self._states.get(key)
                contiguous = (state is not None
                              and state['frame_index'] == frame_index - 1
                              and state['hf'].shape == hf.shape)
                if contiguous:
                    flow = self._dense_flow(small, state['gray'])
                    # VALIDATE THE FIELD BEFORE TRUSTING IT, on the same small
                    # grid the flow was solved on. Warping the full-resolution
                    # low band instead would answer the identical question --
                    # can this field explain the change? -- for a second
                    # full-crop remap and a retained 3 MB array per track, and
                    # a flow that is wrong is wrong at 64x64 too.
                    pred = self._warp(state['gray'].astype(np.float32), flow,
                                      small.shape[:2])
                    residual = float(np.mean(np.abs(pred - small.astype(np.float32))))
                    if residual > self.reset_residual:
                        self._counts.bump('reset_residual')
                    else:
                        warped_hf = self._warp(state['hf'], flow, hf.shape[:2])
                        out_hf = ((1.0 - self.weight) * hf
                                  + self.weight * warped_hf)
                        out = np.clip(low + out_hf, 0, 255).astype(cur.dtype)
                        self._counts.bump('applied')
                elif state is None:
                    self._counts.bump('seeded')
                else:
                    self._counts.bump('skipped_noncontiguous')

                self._states[key] = {
                    'gray': small,
                    'hf': np.array(out_hf, dtype=np.float32, copy=True),
                    'frame_index': int(frame_index),
                }
                self._touch(key)
            return out
        except Exception:
            self._counts.bump('errors')
            return crop

    # -- reporting ----------------------------------------------------------
    def stats(self):
        c = self._counts.snapshot()
        return {'applied': c.get('applied', 0),
                'seeded': c.get('seeded', 0),
                'skipped_noncontiguous': c.get('skipped_noncontiguous', 0),
                'skipped_no_key': c.get('skipped_no_key', 0),
                'reset_residual': c.get('reset_residual', 0),
                'errors': c.get('errors', 0),
                'weight': self.weight}

    def summary_line(self):
        s = self.stats()
        total = (s['applied'] + s['seeded'] + s['skipped_noncontiguous']
                 + s['skipped_no_key'] + s['reset_residual'])
        if total == 0:
            return None
        return ('[HFStabilize] flow-warped HF carry w=%.2f: applied %d/%d '
                '(%.1f%%), seeded %d, non-contiguous %d, no-key %d, '
                'flow-reset %d, errors %d'
                % (s['weight'], s['applied'], total,
                   100.0 * s['applied'] / total, s['seeded'],
                   s['skipped_noncontiguous'], s['skipped_no_key'],
                   s['reset_residual'], s['errors']))
