"""Vectorized, zero-phase One-Euro (1 euro) landmark filter for a whole track at once.

WHY THIS IS A SEPARATE MODULE FROM `one_euro.py`
------------------------------------------------
`one_euro.OneEuroFilter` is the textbook causal filter: one Python object per scalar, stepped
once per frame. The tracking pre-pass does not need to be causal - it holds every frame of a
track before the swap pass starts - and a causal filter's lag is exactly what it cannot avoid
(`tools/gen_jitter_fixture.py`: One Euro 0.1/0.1 reads 0.5 frame behind on a slow sway). This
module runs the SAME adaptive-cutoff idea over the complete sequence, forward and then
backward, so it has no phase delay, and does it as array arithmetic over every landmark
coordinate and every run at once.

THE FILTER
----------
    speed_t   per-point speed of the 5 raw keypoints (central difference), divided by the
              keypoint extent: "face widths per second". ONE speed per frame, shared by the
              5 keypoints, the 106 landmarks, the 68 3-D landmarks (and anything else handed
              in), so the alignment crop and its own outline cannot disagree - the same
              coupling `AdaptiveLandmarkSmoother` documents, and the reason a per-element
              1 euro (cutoff from each coordinate's own derivative) was NOT used.
    dspeed    `speed` low-passed at `d_cutoff` (forward + backward) - the 1 euro derivative
              filter.
    cutoff_t  min_cutoff + beta * dspeed_t                      (Hz)
    alpha_t   1 / (1 + fps / (2 pi cutoff_t))
    y         x -> EMA(alpha) forward -> EMA(alpha) backward    (zero phase)

WHY IT IS A PARALLEL SCAN, NOT A LOOP
-------------------------------------
The textbook filter derives its derivative from its OWN previous output, so each step depends on
the last and the work is a loop. Here the speed is read from the RAW keypoints, so every alpha_t is
known before any smoothing starts, and `y_t = (1 - alpha_t) y_(t-1) + alpha_t x_t` is a LINEAR
time-varying recurrence. Linear recurrences compose associatively, so all T steps are done in
ceil(log2 T) vectorized passes (Hillis-Steele doubling) instead of T sequential ones, on numpy or a
torch tensor on the GPU, across every coordinate, run and track together. Doubling multiplies
numbers in [0, 1] and never divides, so it is stable however long the track is (a cumulative
product / division formulation underflows after a few hundred frames).

BOUNDED, LIKE THE SMOOTHER IT SITS BESIDE
-----------------------------------------
A run ends at a shot cut, at a missing frame and at a frame whose points are not finite: alpha is 1
at its first frame (the recurrence restarts) and nothing is averaged across it, so one side's
coordinates cannot bleed into the other. Runs shorter than 3 frames are returned unchanged.

WHAT IT DOES NOT CLAIM. On real conversational footage no filter removes much of the > 4 Hz band
because that band is partly real motion (see `tools/gen_jitter_fixture.py`); the figures for this
filter on the stored fixture are asserted in `tests/test_one_euro_vec.py`.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

#: Defaults. Chosen on two kinds of evidence, because each alone misleads: the stored synthetic fixture
#: (`app/tests/data/synthetic_face_jitter.npz`, a known head path through the real detector) says beta 50 is best
#: (82/89% jitter removed, 1.6 px at a 60 px head turn), but real tracks (Weeds, Love, Monica Bellucci; 900 frames
#: each, `tests/probe_phase14_prepass.py --section filter`) say that is too open - detector noise in the speed
#: estimate lets the cutoff through, and still-frame jitter is then no better than the shipped filter. 0.6 / 25 / 3
#: removes 91/95% on the fixture and, on the real clips, matches or beats the shipped filter on still jitter, on
#: 106-landmark noise and on scale, while following fast motion to 1-3% of the eye distance where the shipped causal
#: filter is 1.6 / 11.3 / 4.5% behind it. A lower beta smooths a still face harder and tracks fast motion less well
#: (0.6 / 12 / 3: still jitter -20%, deviation in fast motion x1.7); docs/perf/phase14/README.md has the table.
MIN_CUTOFF_HZ = 0.6
BETA = 25.0           # Hz per "face width per second" of head speed
D_CUTOFF_HZ = 3.0
DEFAULT_FPS = 24.0

#: Below this many scalars (T x D) numpy is faster than a kernel-launch-bound GPU call. Measured by
#: `tests/probe_phase14_prepass.py --section filter`; the `auto` device only leaves the CPU above it.
GPU_MIN_ELEMENTS = 140_000


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, value)) if math.isfinite(value) else default


# ── array backend ───────────────────────────────────────────────────────────────────────────────────
def _torch_cuda():
    """The torch module when a CUDA device is usable, else None. Never imports torch just to ask."""
    import sys
    torch = sys.modules.get('torch')
    try:
        if torch is not None and torch.cuda.is_available():
            return torch
    except Exception as exc:                                    # a broken driver is "no GPU"
        _swallowed("roop/one_euro_vec.py:_torch_cuda", exc, "numpy backend used")
    return None


def resolve_device(elements: int, device: Optional[str] = None) -> str:
    """'cpu' or 'cuda'. `device` (or ROOP_ONE_EURO_DEVICE) is auto | cpu | cuda; auto = cuda only for big jobs."""
    want = str(device or os.environ.get('ROOP_ONE_EURO_DEVICE', 'auto')).strip().lower()
    if want == 'cpu':
        return 'cpu'
    if _torch_cuda() is None:
        return 'cpu'
    if want == 'cuda':
        return 'cuda'
    return 'cuda' if int(elements) >= GPU_MIN_ELEMENTS else 'cpu'


def _scan(a, b):
    """y_t = a_t * y_(t-1) + b_t along axis 0 (time), y_(-1) = 0. numpy or torch, same code.

    Hillis-Steele doubling over the associative pair (a, b): composing t with t-s gives
    (a_t * a_(t-s), b_t + a_t * b_(t-s)). Positions before the start are the identity (1, 0).
    """
    t = a.shape[0]
    s = 1
    while s < t:
        a_prev = a[:-s]
        b_prev = b[:-s]
        b_new = b.clone() if hasattr(b, 'clone') else b.copy()
        a_new = a.clone() if hasattr(a, 'clone') else a.copy()
        b_new[s:] = b[s:] + a[s:] * b_prev
        a_new[s:] = a[s:] * a_prev
        a, b = a_new, b_new
        s *= 2
    return b


def _ema(x, alpha):
    """Time-varying EMA along axis 0: y_t = alpha_t x_t + (1 - alpha_t) y_(t-1). alpha_0 (and every run
    start) must be 1, which restarts the recurrence at x_t."""
    return _scan(1.0 - alpha, alpha * x)


def _alpha(cutoff_hz, fps):
    """EMA weight of a first-order low-pass with this cutoff at this frame rate."""
    return 1.0 / (1.0 + fps / (2.0 * math.pi) / cutoff_hz)


# ── run bookkeeping ─────────────────────────────────────────────────────────────────────────────────
def run_bounds(frames: Sequence[int], valid: Sequence[bool], cuts=()) -> List[Tuple[int, int]]:
    """Index ranges [i, j) of consecutive valid frames that do not cross a cut (a cut frame STARTS a run).

    The same contract as `temporal_smoother.ZeroPhaseTrackSmoother.runs`."""
    cuts = {int(c) for c in cuts}
    out, start = [], None
    for k, f in enumerate(frames):
        if not valid[k]:
            if start is not None:
                out.append((start, k))
                start = None
            continue
        if start is not None and (int(f) != int(frames[k - 1]) + 1 or int(f) in cuts):
            out.append((start, k))
            start = None
        if start is None:
            start = k
    if start is not None:
        out.append((start, len(frames)))
    return out


def _neighbours(run_start: np.ndarray, run_end: np.ndarray):
    """For every index, the previous/next index inside ITS run (itself at an edge)."""
    n = len(run_start)
    idx = np.arange(n)
    prev_i = np.maximum(idx - 1, run_start)
    next_i = np.minimum(idx + 1, run_end - 1)
    return prev_i, next_i


def smooth_runs(kps: np.ndarray, dense: np.ndarray, bounds: Sequence[Tuple[int, int]], *,
                fps: float = DEFAULT_FPS, min_cutoff: float = MIN_CUTOFF_HZ, beta: float = BETA,
                d_cutoff: float = D_CUTOFF_HZ, device: Optional[str] = None
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Filter every run of one track in ONE call.

    kps    (T, 5, 2) float64 raw keypoints (rows outside `bounds` are passed through untouched)
    dense  (T, D) float64: every other coordinate to be smoothed with the SAME alphas, flattened per frame
           (may have D == 0)
    bounds [(i, j)) runs, each at least 3 frames long, non-overlapping, ascending

    Returns (kps_smoothed, dense_smoothed, info) with the input's shapes; info has the device that ran it,
    the mean cutoff and the mean alpha over the filtered frames.
    """
    t = len(kps)
    out_k = np.array(kps, dtype=np.float64, copy=True)
    out_d = np.array(dense, dtype=np.float64, copy=True).reshape(t, np.shape(dense)[1] if t else 0)
    info: Dict[str, Any] = {'device': 'none', 'frames': 0, 'runs': 0, 'mean_cutoff_hz': 0.0, 'mean_alpha': 0.0}
    bounds = [(i, j) for (i, j) in bounds if j - i >= 3]
    if t == 0 or not bounds:
        return out_k, out_d, info

    run_start = np.arange(t)
    run_end = np.arange(t) + 1
    in_run = np.zeros(t, dtype=bool)
    for (i, j) in bounds:
        run_start[i:j] = i
        run_end[i:j] = j
        in_run[i:j] = True
    prev_i, next_i = _neighbours(run_start, run_end)

    # Speed of the RAW keypoints, extent-normalised, "face widths per second". Central difference inside a
    # run, one-sided at its ends.
    k_flat = out_k.reshape(t, 5, 2)
    span = np.maximum(next_i - prev_i, 1).astype(np.float64)
    disp = np.linalg.norm(k_flat[next_i] - k_flat[prev_i], axis=2).mean(axis=1) / span
    ext = np.maximum(np.ptp(k_flat, axis=1).max(axis=1), 1.0)
    speed = disp * float(fps) / ext

    # alpha restarts at every run start (and so does the derivative filter).
    is_start = np.zeros(t, dtype=bool)
    is_start[[i for (i, _) in bounds]] = True
    is_end = np.zeros(t, dtype=bool)
    is_end[[j - 1 for (_, j) in bounds]] = True

    flat = np.concatenate([k_flat.reshape(t, 10), out_d], axis=1)
    elements = flat.shape[0] * flat.shape[1]
    dev = resolve_device(elements, device)
    torch = _torch_cuda() if dev == 'cuda' else None
    if torch is not None:
        dtype = torch.float32
        cast = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device='cuda')   # noqa: E731
        back = lambda a: a.detach().double().cpu().numpy()                                       # noqa: E731
        flip = lambda a: torch.flip(a, dims=(0,))                                                # noqa: E731
    else:
        dev = 'cpu'
        cast = lambda a: np.ascontiguousarray(a, dtype=np.float64)                              # noqa: E731
        back = lambda a: a                                                                       # noqa: E731
        flip = lambda a: a[::-1].copy()                                                          # noqa: E731

    def _bidirectional(x, fc, fwd_start, bwd_start):
        """Forward EMA then backward EMA with per-frame cutoff `fc`; restarts where told."""
        alpha = _alpha(fc, float(fps))          # a fresh array each call: safe to edit in place
        a_f = alpha * 1.0
        a_f[fwd_start] = 1.0
        y = _ema(x, a_f)
        a_b = flip(alpha)
        a_b[bwd_start] = 1.0
        return flip(_ema(flip(y), a_b))

    xs = cast(flat)
    sp = cast(speed)[:, None]
    # In the reversed sequence the first element of a run is its LAST frame.
    fwd_start, bwd_start = is_start, is_end[::-1].copy()
    if torch is not None:
        fwd_start = torch.as_tensor(fwd_start, device='cuda')
        bwd_start = torch.as_tensor(bwd_start, device='cuda')
    # derivative filter (fixed d_cutoff), zero phase; then the adaptive cutoff.
    dspeed = _bidirectional(sp, sp * 0.0 + float(d_cutoff), fwd_start, bwd_start)
    positive = dspeed.clamp(min=0.0) if torch is not None else np.maximum(dspeed, 0.0)
    cutoff = float(min_cutoff) + float(beta) * positive
    smoothed = back(_bidirectional(xs, cutoff, fwd_start, bwd_start))

    keep = in_run
    flat_out = flat.copy()
    flat_out[keep] = smoothed[keep]
    out_k = flat_out[:, :10].reshape(t, 5, 2)
    out_d = flat_out[:, 10:].reshape(out_d.shape)
    cut_np = back(cutoff)[:, 0]
    info.update(device=dev, frames=int(keep.sum()), runs=len(bounds),
                mean_cutoff_hz=float(cut_np[keep].mean()),
                mean_alpha=float(_alpha(cut_np[keep], float(fps)).mean()))
    return out_k, out_d, info


class OneEuroTrackSmoother:
    """Drop-in beside `ZeroPhaseTrackSmoother`: the same `smooth_track(frames, kps, dense, cuts)` contract.

    frames: ascending frame numbers; kps: list of (5, 2) arrays or None; dense: {name: list of arrays or None}.
    Returns ({'kps': [...]} plus each dense name, as lists aligned with `frames`) and a stats dict. Entries are
    replaced where a run was smoothed and passed through untouched elsewhere.
    """

    def __init__(self, min_cutoff: float = MIN_CUTOFF_HZ, beta: float = BETA, d_cutoff: float = D_CUTOFF_HZ,
                 fps: float = DEFAULT_FPS, device: Optional[str] = None):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.fps = float(fps) if fps and float(fps) > 0 else DEFAULT_FPS
        self.device = device

    @classmethod
    def from_env(cls, fps: Optional[float] = None):
        return cls(min_cutoff=_env_float('ROOP_ONE_EURO_MIN_CUTOFF', MIN_CUTOFF_HZ, 0.05, 30.0),
                   beta=_env_float('ROOP_ONE_EURO_BETA', BETA, 0.0, 200.0),
                   d_cutoff=_env_float('ROOP_ONE_EURO_D_CUTOFF', D_CUTOFF_HZ, 0.05, 30.0),
                   fps=fps or DEFAULT_FPS)

    def smooth_track(self, frames, kps, dense=None, cuts=()):
        dense = dense or {}
        n = len(frames)
        valid = [k is not None and np.shape(k) == (5, 2) and bool(np.all(np.isfinite(k))) for k in kps]
        result = {'kps': list(kps)}
        for name in dense:
            result[name] = list(dense[name])
        stats = {'runs': 0, 'frames': 0, 'device': 'none', 'mean_cutoff_hz': 0.0}
        bounds = [b for b in run_bounds(frames, valid, cuts) if b[1] - b[0] >= 3]
        if not bounds:
            return result, stats

        k_arr = np.zeros((n, 5, 2))
        for m in range(n):
            if valid[m]:
                k_arr[m] = np.asarray(kps[m], np.float64)
        # A dense array rides along only if EVERY frame of every run has it, finite and in one shape;
        # otherwise that name is left alone (a partial smooth would put a seam in the outline).
        names, columns, shapes = [], [], {}
        for name, seq in dense.items():
            parts = [seq[m] for (i, j) in bounds for m in range(i, j)]
            if not parts or any(p is None for p in parts) or len({np.shape(p) for p in parts}) != 1:
                continue
            block = np.zeros((n,) + np.shape(parts[0]))
            ok = True
            for (i, j) in bounds:
                for m in range(i, j):
                    v = np.asarray(seq[m], np.float64)
                    if not np.all(np.isfinite(v)):
                        ok = False
                        break
                    block[m] = v
                if not ok:
                    break
            if ok:
                names.append(name)
                shapes[name] = np.shape(parts[0])
                columns.append(block.reshape(n, -1))
        d_arr = np.concatenate(columns, axis=1) if columns else np.zeros((n, 0))
        k_s, d_s, info = smooth_runs(k_arr, d_arr, bounds, fps=self.fps, min_cutoff=self.min_cutoff,
                                     beta=self.beta, d_cutoff=self.d_cutoff, device=self.device)
        offset = 0
        for (i, j) in bounds:
            for m in range(i, j):
                result['kps'][m] = k_s[m].astype(np.float32)
        for name in names:
            width = int(np.prod(shapes[name]))
            part = d_s[:, offset:offset + width].reshape((n,) + shapes[name])
            offset += width
            for (i, j) in bounds:
                for m in range(i, j):
                    result[name][m] = part[m].astype(np.float32)
        stats.update(runs=info['runs'], frames=info['frames'], device=info['device'],
                     mean_cutoff_hz=info['mean_cutoff_hz'])
        return result, stats

    def summary_line(self, stats):
        if not stats.get('runs'):
            return None
        return ('1-euro landmark smoothing: %d run(s), %d frame(s), min_cutoff %.2f Hz beta %.1f d_cutoff %.2f Hz '
                '(zero-phase, mean cutoff %.2f Hz at %.1f fps, %s; bounded by cuts and gaps)'
                % (stats['runs'], stats['frames'], self.min_cutoff, self.beta, self.d_cutoff,
                   stats['mean_cutoff_hz'], self.fps, stats['device']))
