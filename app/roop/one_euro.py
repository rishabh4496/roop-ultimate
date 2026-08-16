"""One Euro filter for temporal smoothing of face keypoints across video frames.

The One Euro filter (Casiez et al., 2012) is an adaptive low-pass filter: it
smooths heavily when the signal is slow/still (kills detector jitter) and lightly
when the signal moves fast (avoids lag), so it beats a fixed-alpha EMA's single
jitter-vs-lag tradeoff.

Time `t` here is a monotonically increasing per-frame index (dt = 1 frame), so
`min_cutoff` / `beta` are expressed in per-frame units.
"""
import math
import numpy as np


def _alpha(t_e, cutoff):
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


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
    """

    def apply(self, mask, kps, t):
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
            out = a * mask.astype(np.float32) + (1.0 - a) * tr['prev']
            tr['prev'] = out
            tr['centroid'] = centroid
            tr['last_t'] = t
            self.tracks = [x for x in self.tracks if (t - x['last_t']) <= self.max_missing]
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        self.tracks.append({'prev': mask.astype(np.float32), 'centroid': centroid, 'last_t': t})
        return mask
