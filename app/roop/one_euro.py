"""One Euro filter for temporal smoothing of face keypoints across video frames.

The One Euro filter (Casiez et al., 2012) is an adaptive low-pass filter: it
smooths heavily when the signal is slow/still (kills detector jitter) and lightly
when the signal moves fast (avoids lag), so it beats a fixed-alpha EMA's single
jitter-vs-lag tradeoff.

Time `t` here is a monotonically increasing per-frame index (dt = 1 frame), so
`min_cutoff` / `beta` are expressed in per-frame units.
"""
import math
import os
from collections import defaultdict, deque
import numpy as np


def _env_float(name, default):
    """Numeric environment override that can never break startup.

    A malformed value falls back to the calibrated default rather than raising
    out of a constructor that runs on the render path.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _alpha(t_e, cutoff):
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class StreamingStabilizationHistory:
    """Bounded, ordered history shared by the streaming stabilization path.

    The filters retain the full-resolution previous value they need to smooth;
    this object records the compact causal context that defines that state:
    landmarks, crop affine matrices, and decimated mask weights.  It also owns
    hard-cut detection so none of those values can bleed into a new shot.
    """

    # ── Why the cut rule is adaptive and not a single number ──────────────
    # The original gate was a bare `mean|diff| >= 0.32` on a 64px luma
    # thumbnail. 0.32 is an enormous difference for that statistic: it is
    # roughly a third of full swing averaged over the WHOLE frame, which in
    # practice only a cut between near-black and near-white produces. That is
    # exactly what the unit test fed it (a black frame then a white one), so
    # the constant passed its test and then never fired again.
    #
    # Measured on the clip this was found on (a 428s, 12840-frame music video
    # that is visibly a montage, tools/cut_threshold_probe.py):
    #
    #     p50 0.0042   p95 0.0482   p99 0.0793   p99.9 0.1616   max 0.2002
    #
    # The LARGEST adjacent-frame difference in the entire clip is 0.2002, so
    # the shipped gate fired on 0 of 12839 pairs. Both consumers of the signal
    # were therefore dead code on this footage: the swap phase never reset its
    # kps/enhancer/mask stabilisers at a shot change, and the tracking pre-pass
    # never learned a shot had changed at all.
    #
    # No fixed number fixes this properly, because the statistic is not
    # scale-free: a high-contrast action clip and a soft-graded interior have
    # different baselines, and a cut is defined relative to how much the
    # picture normally moves. So the rule is BOTH:
    #
    #   * an absolute floor, so grain/exposure ripple in a locked-off shot can
    #     never be called a cut however quiet that shot is, and
    #   * a multiple of the RECENT MEDIAN difference, which is what makes it
    #     mean the same thing on any content.
    #
    # Calibrated against the same probe. floor=0.045 sits just under the clip's
    # p95 and well above its p75; k=6 against a rolling median found 124 cuts
    # (17.4/min, mean shot 3.4s) which matches the actual edit rate of the
    # footage. Loosening to floor=0.030/k=4 finds 272 (38/min) and starts
    # splitting on camera shake; tightening to 0.060/k=8 finds 67 and misses
    # cuts between similar-looking shots. The median is taken over a short
    # trailing window so the reference tracks the current shot rather than the
    # whole clip.
    #
    # ROOP_CUT_FLOOR / ROOP_CUT_RATIO override both; ROOP_CUT_FLOOR=1.0
    # restores the effectively-never-fires behaviour if a regression is ever
    # traced here.
    CUT_FLOOR = 0.045
    CUT_RATIO = 6.0
    CUT_WINDOW = 61

    # Frames after a cut during which another cut is not reported. A shot
    # change is one event; its settling frames are not more of them.
    CUT_REFRACTORY = 3

    # The bar before any baseline exists (the first frames of a clip). Set at
    # the reference clip's p99.9 (0.1616): high enough that ordinary motion
    # cannot reach it, low enough that a genuine opening cut still does.
    CUT_COLD_START = 0.16

    # Differences needed before the ratio test is trusted. Short enough that a
    # new shot is judged on its own terms within a few frames, long enough that
    # the median is not one or two unsettled readings.
    CUT_MIN_BASELINE = 8

    def __init__(self, capacity=32, cut_threshold=None,
                 cut_floor=None, cut_ratio=None):
        self.capacity = max(4, int(capacity))
        # Kept for callers (and tests) that pin an explicit absolute gate. When
        # given it is used as the floor AND disables the adaptive term, which is
        # the old single-number behaviour exactly.
        self.cut_threshold = None if cut_threshold is None else float(cut_threshold)
        self.cut_floor = float(
            _env_float('ROOP_CUT_FLOOR',
                       self.CUT_FLOOR if cut_floor is None else cut_floor))
        self.cut_ratio = float(
            _env_float('ROOP_CUT_RATIO',
                       self.CUT_RATIO if cut_ratio is None else cut_ratio))
        self.frames = deque(maxlen=self.capacity)
        self.landmarks = defaultdict(lambda: deque(maxlen=self.capacity))
        self.affines = defaultdict(lambda: deque(maxlen=self.capacity))
        self.mask_weights = defaultdict(lambda: deque(maxlen=self.capacity))
        self._scene_signature = None
        # Trailing window of recent frame-to-frame differences. Only the
        # differences are kept (a float each), never the frames.
        self._diffs = deque(maxlen=self.CUT_WINDOW)
        # Frames observed since the last cut, for the refractory guard. Starts
        # large so the first frames of a clip are not treated as post-cut.
        self._since_cut = 1 << 30
        self.scene_cuts = 0

    @staticmethod
    def _signature(frame):
        """Small luma thumbnail; no full frame is retained by the FIFO."""
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        stride = max(1, int(max(arr.shape[:2]) / 64))
        sample = arr[::stride, ::stride, :3].astype(np.float32)
        return (0.114 * sample[:, :, 0] + 0.587 * sample[:, :, 1] +
                0.299 * sample[:, :, 2]) / 255.0

    def observe_frame(self, frame, t):
        """Append the linear frame observation and return ``True`` on a cut."""
        signature = self._signature(frame)
        if signature is None:
            return False
        previous = self._scene_signature
        cut = False
        diff = None
        if previous is not None:
            # A deterministic resize-free comparison: adjacent signatures have
            # the same source dimensions for one stream. Resolution changes are
            # a cut by definition.
            if previous.shape != signature.shape:
                cut = True
            else:
                diff = float(np.mean(np.abs(signature - previous)))
                cut = self._is_cut(diff)
        if cut:
            self.frames.clear()
            self.landmarks.clear()
            self.affines.clear()
            self.mask_weights.clear()
            self.scene_cuts += 1
            # A new shot measures its OWN baseline. Keeping the previous
            # shot's differences here is what made a lively shot following a
            # quiet one report a second cut a few frames in: the median was
            # still describing the quiet shot, so ordinary movement in the new
            # one sat far above `ratio * baseline`. The window is therefore
            # emptied, and `_is_cut` covers the interval where it is refilling
            # with an absolute bar instead of a ratio (see CUT_COLD_START) —
            # which is also what stops the cascade that simply clearing it
            # caused on its own.
            self._diffs.clear()
            self._since_cut = 0
        elif diff is not None:
            self._since_cut += 1
            self._diffs.append(diff)
        self._scene_signature = signature
        self.frames.append((int(t), float(signature.mean()), float(signature.std())))
        return cut

    def _is_cut(self, diff):
        """Absolute floor AND (unless pinned) a multiple of the recent median.

        Below the floor is never a cut, whatever the local baseline: that is
        what stops a perfectly still locked-off shot, whose median difference is
        near zero, from calling its own sensor noise a scene change.
        """
        if diff < self.cut_floor:
            return False
        if self.cut_threshold is not None:
            # Explicit absolute gate requested by the caller.
            return diff >= self.cut_threshold
        if self.cut_ratio <= 0:
            return True
        # A real cut is followed by a settling frame or two (the new shot's
        # first inter-frame difference is often large as well: motion blur
        # resolving, a fade completing, rolling shutter). Reporting those as
        # further cuts would reset the stabilisers again mid-shot, which is the
        # flicker this whole mechanism exists to prevent. One shot change is
        # enough; suppress immediate repeats.
        if self._since_cut < self.CUT_REFRACTORY:
            return False
        if len(self._diffs) < self.CUT_MIN_BASELINE:
            # No baseline for THIS shot yet — the clip has just started, or a
            # cut has just emptied the window. A ratio test is meaningless
            # here, and the floor alone is a motion threshold rather than a cut
            # threshold, so neither can be used. Fall back to a deliberately
            # high absolute bar: high enough that ordinary movement in an
            # unsettled new shot cannot reach it, low enough that a genuine
            # second cut arriving before the baseline is rebuilt is still seen.
            return diff >= self.CUT_COLD_START
        baseline = float(np.median(np.fromiter(self._diffs, dtype=np.float64)))
        return diff >= self.cut_ratio * max(baseline, 1e-4)

    def record_landmarks(self, track_id, kps, t):
        if kps is not None:
            self.landmarks[track_id].append((int(t), np.asarray(kps, np.float32).copy()))

    def record_affine(self, track_id, matrix, t):
        if matrix is not None:
            self.affines[track_id].append((int(t), np.asarray(matrix, np.float32).reshape(2, 3).copy()))

    def record_mask(self, track_id, mask, t):
        if mask is None:
            return
        value = np.asarray(mask, np.float32)
        stride = max(1, int(max(value.shape[:2]) / 32))
        self.mask_weights[track_id].append((int(t), value[::stride, ::stride].copy()))


# Warm-up frames needed before a filter's seed stops showing in its output.
#
# Parallel stabilization splits a clip into contiguous blocks and gives each
# block its own filter, seeded from the first frame it sees rather than from the
# true history. Every one of these filters is an EMA
# (`out = a*x + (1-a)*prev`), so that wrong seed decays geometrically: after W
# frames its weight is exactly (1-a)^W. Priming each block with W frames it then
# discards is what makes a block boundary seam-free — and W is not a taste
# parameter, it is fixed by `a` and the error you are willing to leave behind.
#
# A single hardcoded W cannot be right for every setting: `a` is derived from
# the smoothing strength the user chose, and it varies enormously.
#
#     strength   a       (1-a)^4    W for <=1%
#       0.00     0.725     0.57%        4
#       0.50     0.580     3.10%        6
#       0.75     0.430    10.57%        9
#       1.00     0.112    62.28%       39
#
# So the old fixed WU=4 was right only at the weak end and left 62% of the seed
# error at the strong end — a visible step at every block boundary, in the
# configuration that asked for the MOST smoothing.
_MAX_WARMUP = 240


def ema_warmup_frames(alpha, eps=0.01):
    """Frames for an EMA with factor `alpha` to forget its seed to <= `eps`.

    Solves (1-alpha)^W <= eps. Capped: alpha near 0 is a filter that effectively
    never forgets, and no finite warm-up fixes it — the caller must fall back to
    sequential rather than pretend.
    """
    a = float(alpha)
    if a >= 1.0:
        return 0            # no memory: the seed is gone after one frame
    if a <= 0.0:
        return _MAX_WARMUP  # never forgets
    return min(_MAX_WARMUP, max(1, math.ceil(math.log(eps) / math.log(1.0 - a))))


class OneEuroFilter:
    """Adaptive low-pass filter over an arbitrary-shaped numpy signal (elementwise)."""

    def __init__(self, min_cutoff=0.05, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1.0
        a_d = _alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(t_e, cutoff)          # per-element adaptive smoothing factor
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class KpsStabilizer:
    """Smooths face 5-point keypoints across frames, with nearest-centroid
    tracking so each face in a multi-face scene keeps its own filter."""

    def __init__(self, min_cutoff=0.05, beta=0.02, max_missing=8, match_scale=0.6):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.max_missing = int(max_missing)   # drop a track unseen for this many frames
        self.match_scale = float(match_scale)  # match radius as a fraction of face size
        self.tracks = []                       # [{filter, centroid, last_t}]

    def warmup_frames(self, eps=0.01):
        """Warm-up for parallel stabilization. `beta` only ever RAISES the cutoff
        on motion, which speeds convergence, so the still case (cutoff =
        min_cutoff) is the worst case and the one to size for."""
        return ema_warmup_frames(_alpha(1.0, self.min_cutoff), eps)

    def reset(self):
        self.tracks = []

    def apply(self, kps, t):
        """Return temporally-smoothed (5,2) keypoints for the face at frame `t`."""
        kps = np.asarray(kps, dtype=np.float64)
        if kps.shape != (5, 2):
            return kps.astype(np.float32)
        centroid = kps.mean(axis=0)
        size = max(float(np.ptp(kps[:, 0])), float(np.ptp(kps[:, 1])), 1.0)

        best, best_d = None, float('inf')
        for tr in self.tracks:
            d = float(np.linalg.norm(tr['centroid'] - centroid))
            if d < best_d:
                best_d, best = d, tr

        if best is not None and best_d <= self.match_scale * size and (t - best['last_t']) <= self.max_missing:
            tr = best
        else:
            tr = {'filter': OneEuroFilter(self.min_cutoff, self.beta), 'centroid': centroid, 'last_t': t}
            self.tracks.append(tr)

        smoothed = tr['filter'](kps, t)
        tr['centroid'] = smoothed.mean(axis=0)
        tr['last_t'] = t
        # prune stale tracks
        self.tracks = [x for x in self.tracks if (t - x['last_t']) <= self.max_missing]
        return smoothed.astype(np.float32)


class EmaKpsStabilizer:
    """Smooths face 5-point keypoints across frames using an Exponential Moving Average (EMA)."""

    def __init__(self, alpha=0.3, max_missing=8, match_scale=0.6):
        self.alpha = float(alpha)
        self.max_missing = int(max_missing)
        self.match_scale = float(match_scale)
        self.tracks = []

    def warmup_frames(self, eps=0.01):
        """Warm-up for parallel stabilization — a plain fixed-alpha EMA."""
        return ema_warmup_frames(self.alpha, eps)

    def reset(self):
        self.tracks = []

    def apply(self, kps, t):
        kps = np.asarray(kps, dtype=np.float64)
        if kps.shape != (5, 2):
            return kps.astype(np.float32)
        centroid = kps.mean(axis=0)
        size = max(float(np.ptp(kps[:, 0])), float(np.ptp(kps[:, 1])), 1.0)

        best, best_d = None, float('inf')
        for tr in self.tracks:
            d = float(np.linalg.norm(tr['centroid'] - centroid))
            if d < best_d:
                best_d, best = d, tr

        if best is not None and best_d <= self.match_scale * size and (t - best['last_t']) <= self.max_missing:
            tr = best
            smoothed = self.alpha * kps + (1.0 - self.alpha) * tr['prev']
            tr['prev'] = smoothed
        else:
            tr = {'prev': kps, 'centroid': centroid, 'last_t': t}
            self.tracks.append(tr)
            smoothed = kps

        tr['centroid'] = smoothed.mean(axis=0)
        tr['last_t'] = t
        self.tracks = [x for x in self.tracks if (t - x['last_t']) <= self.max_missing]
        return smoothed.astype(np.float32)


class EnhancerStabilizer:
    """Reduces per-frame enhancer texture flicker (GFPGAN/GPEN/Codeformer shimmer)
    by temporally blending the *aligned* enhanced crop with a motion-adaptive
    blend factor (One Euro logic, but the speed is a single scalar — head motion —
    applied uniformly to the whole crop, NOT per-pixel: a per-pixel derivative
    would treat the flicker itself as "motion" and refuse to smooth it).

    Blend hard when the face is still → kills flicker; pass the current frame
    through on fast motion → avoids ghosting. Per-face tracking by kps centroid.
    """

    def __init__(self, strength=0.5, max_missing=8, match_scale=0.6, motion_beta=8.0):
        self.strength = float(min(max(strength, 0.0), 1.0))
        # strength 0 → light (base_cutoff 0.42), strength 1 → heavy (0.02)
        self.base_cutoff = 0.4 * (1.0 - self.strength) + 0.02
        self.motion_beta = float(motion_beta)
        self.max_missing = int(max_missing)
        self.match_scale = float(match_scale)
        self.tracks = []   # [{prev(float32 crop), centroid, last_t}]

    def warmup_frames(self, eps=0.01):
        """Warm-up for parallel stabilization. `motion_beta` only ever RAISES the
        cutoff (faster convergence), so a still face — cutoff = base_cutoff — is
        the worst case. This is the filter that scales hardest with the user's
        strength setting: 4 frames at strength 0, 39 at strength 1."""
        return ema_warmup_frames(_alpha(1.0, self.base_cutoff), eps)

    def reset(self):
        self.tracks = []

    def apply(self, crop, kps, t):
        if crop is None:
            return crop
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape != (5, 2):
            return crop
        centroid = kps.mean(axis=0)
        size = max(float(np.ptp(kps[:, 0])), float(np.ptp(kps[:, 1])), 1.0)

        best, best_d = None, float('inf')
        for tr in self.tracks:
            d = float(np.linalg.norm(tr['centroid'] - centroid))
            if d < best_d:
                best_d, best = d, tr

        matched = (best is not None and best_d <= self.match_scale * size
                   and (t - best['last_t']) <= self.max_missing
                   and best['prev'].shape == crop.shape)
        if matched:
            tr = best
            t_e = max(t - tr['last_t'], 1)
            motion = (best_d / t_e) / size            # head motion as a fraction of face size
            cutoff = self.base_cutoff + self.motion_beta * motion
            a = _alpha(t_e, cutoff)                   # → 1 (current) when fast, small when still
            out = a * crop.astype(np.float32) + (1.0 - a) * tr['prev']
            tr['prev'] = out
            tr['centroid'] = centroid
            tr['last_t'] = t
            self.tracks = [x for x in self.tracks if (t - x['last_t']) <= self.max_missing]
            return np.clip(out, 0, 255).astype(np.uint8)
        # new / unmatched track — pass through and seed.
        self.tracks.append({'prev': crop.astype(np.float32), 'centroid': centroid, 'last_t': t})
        return crop


class MaskStabilizer(EnhancerStabilizer):
    """Same motion-adaptive per-track blending as EnhancerStabilizer, applied to
    a mask array instead of a BGR crop.

    Masks in this codebase are float32 in [0, 1] (0 = swap, 1 = restore
    original — see procmgr_masking._composite_mask), not uint8 [0, 255] pixels,
    so the parent's final `np.clip(out, 0, 255).astype(np.uint8)` would corrupt
    every value below 1.0 down to 0. Overriding just that line keeps everything
    else — track matching by kps centroid, motion-adaptive cutoff, warm-up
    derivation for the chunked-parallel path — identical.

    Why the mask needs this at all: XSeg/FaceParser (and their RealityUX
    fusion) recompute the occlusion mask from scratch every frame with no
    memory of the previous one. On a close-up where hair grazes an eye
    slightly differently frame to frame, the boundary jitters — most visible
    as a swapped face's makeup/eyeshadow region flickering in and out along
    that edge. Smoothing the mask temporally, the same way the enhancer output
    already is, damps that without touching identity or swap correctness (the
    mask only controls the blend edge, never which source face gets used).

    `fast_restore_alpha` is an opt-in occlusion policy. In this mask convention
    1.0 means "restore the original plate", so a positive transition is an
    occluder entering the face. That transition is allowed to use a larger EMA
    factor while the reverse transition keeps the normal smoothing factor. This
    prevents a hand/hair edge from being painted over for several frames, while
    still fading the mask out gently when the occluder leaves. It is deliberately
    disabled by default: an aggressive response can expose a one-frame false
    positive from a noisy segmenter, and existing renders must remain unchanged.
    """

    def __init__(self, *args, fast_restore_alpha=0.0, flow_warp=False,
                 flow_stats=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.fast_restore_alpha = min(1.0, max(0.0, float(fast_restore_alpha)))
        except (TypeError, ValueError):
            self.fast_restore_alpha = 0.0
        # Motion compensation (setting `mask_flow_warp`). Without it the EMA
        # below blends the previous mask IN PLACE: the aligned crop cancels
        # the head's rigid motion, but not a hand crossing the face, hair
        # swinging, or an expression, so anything that moves relative to the
        # face leaves a trail -- the previous boundary fades out over several
        # frames where it WAS, instead of following it. With it, the previous
        # mask is first warped into the current crop along dense optical flow
        # between the two aligned crops:
        #
        #     M_t = a * M_t + (1 - a) * Warp(M_{t-1}, flow)
        #
        # The flow machinery (DIS ultrafast at 128px, backward-flow argument
        # order, per-thread engine) is occlusion_mask.TemporalMaskSmoother's,
        # reused rather than copied. That class itself is not constructed by
        # the render path; this is where its warp actually runs.
        #
        # Warps only across ADJACENT frames (t - last_t == 1). A gap means
        # the stored crop is from further back than a flow field can bridge,
        # and the plain EMA (the pre-existing behaviour) is used instead.
        self.flow_warp = bool(flow_warp)
        self._flow = None
        if self.flow_warp:
            from roop.occlusion_mask import TemporalMaskSmoother
            self._flow = TemporalMaskSmoother(enabled=True)
        # Shared across the per-chunk instances the parallel path builds, so
        # the end-of-run line counts the whole render, not one chunk.
        self.flow_stats = (flow_stats if flow_stats is not None
                           else {'applied': 0, 'declined': 0, 'reset': 0})

    def flow_summary_line(self):
        """Counts, printed at the end of a run. A warp that declined on every
        face renders identically to one that is switched off."""
        st = self.flow_stats
        total = st['applied'] + st['declined'] + st['reset']
        if not self.flow_warp:
            return None
        if total == 0:
            return ('[Stabilize] mask flow-warp ENABLED BUT NEVER INVOKED: no '
                    'matched face reached it; it had no effect on this run')
        return ('[Stabilize] mask flow-warp: applied %d/%d (%.1f%%), '
                'declined %d (gap or no guide), flow-reset %d'
                % (st['applied'], total, 100.0 * st['applied'] / total,
                   st['declined'], st['reset']))

    def _flow_gray(self, guide):
        if self._flow is None or guide is None:
            return None
        try:
            return self._flow._observation(guide, self._flow.flow_size)
        except Exception as exc:
            from roop.degrade import swallowed
            swallowed("roop/one_euro.py:MaskStabilizer._flow_gray", exc,
                      "flow warp declined for this face")
            return None

    def apply(self, mask, kps, t, guide=None):
        """`guide` is the aligned crop the mask was computed from (BGR, any
        size). Only read when `flow_warp` is on."""
        if mask is None:
            return mask
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape != (5, 2):
            return mask
        centroid = kps.mean(axis=0)
        size = max(float(np.ptp(kps[:, 0])), float(np.ptp(kps[:, 1])), 1.0)

        best, best_d = None, float('inf')
        for tr in self.tracks:
            d = float(np.linalg.norm(tr['centroid'] - centroid))
            if d < best_d:
                best_d, best = d, tr

        matched = (best is not None and best_d <= self.match_scale * size
                   and (t - best['last_t']) <= self.max_missing
                   and best['prev'].shape == mask.shape)
        if matched:
            tr = best
            t_e = max(t - tr['last_t'], 1)
            motion = (best_d / t_e) / size
            cutoff = self.base_cutoff + self.motion_beta * motion
            a = _alpha(t_e, cutoff)
            current = mask.astype(np.float32)
            previous = tr['prev']
            gray = self._flow_gray(guide)
            if self.flow_warp:
                prev_gray = tr.get('gray')
                if gray is not None and prev_gray is not None and t - tr['last_t'] == 1:
                    flow = self._flow._dense_flow(gray, prev_gray)
                    # XSeg hands back (H, W, 1) and cv2.remap drops a
                    # trailing unit channel; without the reshape the blend
                    # below broadcasts to (H, W, H) and the track never
                    # matches again.
                    warped = self._flow._warp(
                        np.ascontiguousarray(previous.reshape(previous.shape[:2])),
                        flow, current.shape[:2]).reshape(previous.shape)
                    if float(np.mean(np.abs(warped - current))) > self._flow.reset_residual:
                        # The flow did not explain the change (a cut, a
                        # re-detect jump): keep the unwarped EMA rather than
                        # drag a mask the field could not follow.
                        self.flow_stats['reset'] += 1
                    else:
                        previous = warped
                        self.flow_stats['applied'] += 1
                elif t != tr['last_t']:
                    # t == last_t is the same frame's mask handed back in a
                    # second time (ProcessMgr re-applies the stabilizer after
                    # process_mask already did); blending it with itself is a
                    # no-op, and counting it would read as half the faces
                    # declined.
                    self.flow_stats['declined'] += 1
            if self.fast_restore_alpha > 0.0:
                # Positive delta means more of the original should be restored.
                # Apply the larger factor only to those pixels. The reverse
                # transition remains the ordinary low-pass path, which avoids
                # reintroducing boundary flicker when an occluder disappears.
                delta = current - previous
                out = previous + a * delta
                if float(delta.max()) > 0.0:
                    entering = delta > 0.0
                    restore_a = max(a, self.fast_restore_alpha)
                    out[entering] = (restore_a * current[entering]
                                     + (1.0 - restore_a) * previous[entering])
            else:
                # Keep the legacy arithmetic untouched when the experiment is
                # disabled; this is the default and the compatibility path.
                out = a * current + (1.0 - a) * previous
            tr['prev'] = out
            tr['gray'] = gray
            tr['centroid'] = centroid
            tr['last_t'] = t
            self.tracks = [x for x in self.tracks if (t - x['last_t']) <= self.max_missing]
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        self.tracks.append({'prev': mask.astype(np.float32), 'centroid': centroid,
                            'last_t': t, 'gray': self._flow_gray(guide)})
        return mask
