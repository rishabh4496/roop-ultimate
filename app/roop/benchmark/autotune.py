"""Auto-tune: measure provider x swap-batch on the user's own render, keep a
winner only if it survives a counterbalanced 600-frame confirmation.

WHY NOT THE EXISTING /api/benchmark RUNNER. `BenchmarkRunner` calls
`ProcessMgr.process_frame` one frame at a time, on one thread, in preview mode.
The cross-frame swap batcher, the worker pool and the video writer never run
there -- so a batch-size axis measured through it would post a number for code
that did not execute (AGENTS.md: "prove the code path executes"). Every arm
here is a REAL trimmed render through `api._run_swap`, the Start button's path,
replaying the last render's normalized request.

THE PROTOCOL (all of it is AGENTS.md's benchmarking rules, not taste):

  screen   every (provider, batch) arm at SCREEN_FRAMES, twice, in grouped
           A..Z Z..A order so position drift cancels in each arm's mean.
           Screening only ranks; nothing is saved from it.
  guard    an arm that swapped fewer faces than the baseline, or whose
           effective batch/provider differs from what it asked for (the VRAM
           governor can clamp a batch), is disqualified under its label.
  confirm  the top finalists against the baseline at CONFIRM_FRAMES, A/B/B/A.
           A finalist is kept only if it beats the baseline in BOTH pairs AND
           by more than the baseline's own replicate spread (the null control)
           AND by at least MIN_EFFECT. Otherwise the baseline stays -- on the
           4070 ~5% effects are not resolvable, and saying so is the result.
  encoder  NVENC p1..p7 (-tune hq, VBR at the user's -cq) encode rate and
           achieved bitrate, measured on the target itself. The preset kept is
           the highest-quality one that encodes at >= 2x the confirmed render
           rate -- the writer runs CONCURRENTLY with the render, so a faster
           preset than that buys no wall clock and costs bitrate
           (roop/bench.py::_recommend_encoder, same reasoning).

The session takes its measurement and encoder functions by injection, so the
whole decision path is unit-tested without a GPU (tests/test_autotune.py).
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from roop.degrade import swallowed as _swallowed

SCREEN_FRAMES = 100
CONFIRM_FRAMES = 600
BATCHES = (1, 2, 4, 8)
PROVIDERS = ('cuda', 'tensorrt')
NVENC_PRESETS = tuple(f'p{i}' for i in range(1, 8))
MIN_EFFECT = 0.03
SWAP_GUARD = 0.98
FINALISTS = 2
ENCODER_HEADROOM = 2.0

RESULT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'autotune_result.json')


@dataclass(frozen=True)
class Arm:
    provider: str
    batch: int          # 0 = the current 'auto' setting, left untouched

    @property
    def key(self) -> str:
        return f"{self.provider}/b{self.batch or 'auto'}"


@dataclass
class ArmRun:
    arm: Arm
    phase: str
    frames: int
    fps: float = 0.0
    swaps: int = 0
    effective_batch: Optional[int] = None
    effective_provider: Optional[str] = None
    error: Optional[str] = None
    order: int = 0

    @property
    def honest(self) -> bool:
        """The arm ran what its label says."""
        if self.error or self.fps <= 0:
            return False
        if self.effective_provider and self.effective_provider != self.arm.provider:
            return False
        if (self.arm.batch and self.effective_batch is not None
                and self.effective_batch != self.arm.batch):
            return False
        return True

    def as_dict(self) -> dict:
        row = asdict(self)
        row['arm'] = self.arm.key
        row['honest'] = self.honest
        return row


# ── pure decision functions ────────────────────────────────────────────────

def candidate_arms(providers: Sequence[str], threads: int,
                   batches: Sequence[int] = BATCHES) -> List[Arm]:
    """Every (provider, batch) the machine can run. A batch above the worker
    count can never fill (the batcher coalesces one crop per worker), so it is
    not an arm -- it would be a second copy of the largest reachable batch."""
    reachable = [b for b in batches if b == 1 or b <= max(1, int(threads))]
    return [Arm(p, b) for p in providers for b in reachable]


def screening_order(arms: Sequence[Arm]) -> List[Arm]:
    """A..Z then Z..A, grouped by provider so a provider switch (a full model
    rebuild) happens three times, not once per arm."""
    groups: Dict[str, List[Arm]] = {}
    for arm in arms:
        groups.setdefault(arm.provider, []).append(arm)
    forward = [a for p in groups for a in groups[p]]
    return forward + list(reversed(forward))


def summarize(runs: Sequence[ArmRun]) -> Dict[str, dict]:
    """Per-arm mean fps over its honest runs, and its worst swap count."""
    out: Dict[str, dict] = {}
    for run in runs:
        row = out.setdefault(run.arm.key, {'arm': run.arm, 'fps': [], 'swaps': [],
                                           'dishonest': 0, 'errors': []})
        if run.honest:
            row['fps'].append(run.fps)
            row['swaps'].append(run.swaps)
        else:
            row['dishonest'] += 1
            if run.error:
                row['errors'].append(run.error)
    for row in out.values():
        row['mean_fps'] = statistics.mean(row['fps']) if row['fps'] else 0.0
        row['min_swaps'] = min(row['swaps']) if row['swaps'] else 0
    return out


def pick_finalists(summary: Dict[str, dict], baseline: Arm,
                   k: int = FINALISTS) -> Tuple[List[Arm], Dict[str, str]]:
    """Up to k arms faster than the baseline that pass the guard, and a reason
    for every arm that was excluded."""
    base = summary.get(baseline.key)
    reasons: Dict[str, str] = {}
    if not base or not base['fps']:
        return [], {baseline.key: 'baseline did not produce a measurement'}
    if base['min_swaps'] <= 0:
        return [], {baseline.key: 'baseline swapped no faces on these frames'}
    ranked = []
    for key, row in summary.items():
        if key == baseline.key:
            continue
        if row['dishonest'] or not row['fps']:
            reasons[key] = ('did not run as labelled (clamped, fell back or failed)'
                            if not row['errors'] else row['errors'][0])
            continue
        if row['min_swaps'] < SWAP_GUARD * base['min_swaps']:
            reasons[key] = (f"swapped {row['min_swaps']} faces vs baseline "
                            f"{base['min_swaps']} -- faster by doing less")
            continue
        if row['mean_fps'] <= base['mean_fps']:
            reasons[key] = 'not faster than the baseline in screening'
            continue
        ranked.append(row)
    ranked.sort(key=lambda r: r['mean_fps'], reverse=True)
    for row in ranked[k:]:
        reasons[row['arm'].key] = 'faster, but not in the top %d' % k
    return [r['arm'] for r in ranked[:k]], reasons


def confirm_order(baseline: Arm, candidate: Arm) -> List[Arm]:
    return [baseline, candidate, candidate, baseline]


def verdict(base_runs: Sequence[ArmRun], cand_runs: Sequence[ArmRun]) -> dict:
    """ABBA verdict. Pairs are (B1, C1) and (C2, B2) in run order."""
    b = [r.fps for r in base_runs if r.honest]
    c = [r.fps for r in cand_runs if r.honest]
    if len(b) < 2 or len(c) < 2:
        return {'accepted': False, 'reason': 'an arm of the confirmation failed',
                'improvement_pct': None, 'noise_pct': None}
    base_mean, cand_mean = statistics.mean(b), statistics.mean(c)
    improvement = cand_mean / base_mean - 1.0
    noise = abs(b[0] - b[1]) / base_mean
    both = c[0] > b[0] and c[1] > b[1]
    swap_ok = min(r.swaps for r in cand_runs) >= SWAP_GUARD * min(r.swaps for r in base_runs)
    accepted = both and swap_ok and improvement > max(noise, MIN_EFFECT)
    if accepted:
        reason = 'beat the baseline in both pairs, above the noise floor'
    elif not swap_ok:
        reason = 'swapped fewer faces than the baseline'
    elif not both:
        reason = 'lost one of the two counterbalanced pairs'
    else:
        reason = (f'+{improvement * 100:.1f}% is inside the noise '
                  f'({max(noise, MIN_EFFECT) * 100:.1f}%)')
    return {'accepted': accepted, 'reason': reason,
            'improvement_pct': round(improvement * 100, 2),
            'noise_pct': round(noise * 100, 2),
            'baseline_fps': round(base_mean, 3), 'candidate_fps': round(cand_mean, 3)}


def pick_nvenc_preset(rows: Sequence[dict], render_fps: float,
                      headroom: float = ENCODER_HEADROOM) -> Optional[dict]:
    """The highest-quality preset that keeps up with the render with headroom.

    `rows` are {'preset': 'pN', 'fps': float, ...}. Higher N is slower and
    better at the same -cq. If none keeps up, the fastest one measured."""
    ok = [r for r in rows if r.get('fps')]
    if not ok:
        return None
    need = max(0.0, float(render_fps)) * headroom
    keeping_up = [r for r in ok if r['fps'] >= need]
    if keeping_up:
        return max(keeping_up, key=lambda r: int(str(r['preset'])[1:]))
    return max(ok, key=lambda r: r['fps'])


# ── the session ────────────────────────────────────────────────────────────

class Cancelled(Exception):
    pass


MeasureFn = Callable[[Arm, int, str], ArmRun]
EncodeFn = Callable[[Sequence[str]], dict]


@dataclass
class TuneProgress:
    running: bool = False
    phase: str = 'idle'
    status: str = ''
    arm: Optional[str] = None
    arms_done: int = 0
    arms_total: int = 0
    last_fps: Optional[float] = None
    log: List[str] = field(default_factory=list)
    error: Optional[str] = None


class AutoTuneSession:
    def __init__(self, measure: MeasureFn, encode: Optional[EncodeFn],
                 providers: Sequence[str], threads: int, baseline: Arm,
                 screen_frames: int = SCREEN_FRAMES, confirm_frames: int = CONFIRM_FRAMES,
                 available_frames: Optional[int] = None,
                 provider_notes: Optional[Dict[str, str]] = None):
        self.measure = measure
        self.encode = encode
        self.providers = list(providers)
        self.threads = int(threads)
        self.baseline = baseline
        cap = int(available_frames) if available_frames else None
        self.screen_frames = min(screen_frames, cap) if cap else screen_frames
        self.confirm_frames = min(confirm_frames, cap) if cap else confirm_frames
        self.provider_notes = dict(provider_notes or {})
        self.progress = TuneProgress()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.runs: List[ArmRun] = []

    # -- bookkeeping ------------------------------------------------------
    def cancel(self):
        self._cancel.set()

    def _log(self, message: str):
        print(f'[AutoTune] {message}', flush=True)
        with self._lock:
            self.progress.log.append(message)
            del self.progress.log[:-200]

    def snapshot(self) -> dict:
        with self._lock:
            snap = asdict(self.progress)
        snap['log'] = snap['log'][-60:]
        return snap

    def _run(self, arm: Arm, frames: int, phase: str) -> ArmRun:
        if self._cancel.is_set():
            raise Cancelled()
        with self._lock:
            self.progress.arm = arm.key
            self.progress.status = f'{phase}: {arm.key} ({frames} frames)'
        started = time.time()
        try:
            run = self.measure(arm, frames, phase)
        except Cancelled:
            raise
        except Exception as exc:
            # A crashing arm (an OOM at batch 8, say) is a RESULT: it is
            # recorded, logged below, and disqualified -- not fatal to the run.
            _swallowed("autotune.py:arm", exc, "arm recorded as failed")
            run = ArmRun(arm=arm, phase=phase, frames=frames,
                         error=f'{type(exc).__name__}: {exc}')
        run.order = len(self.runs)
        self.runs.append(run)
        with self._lock:
            self.progress.arms_done += 1
            self.progress.last_fps = run.fps or None
        self._log(f'{phase} {arm.key}: '
                  + (f'{run.fps:.2f} fps, {run.swaps} swaps'
                     + ('' if run.honest else
                        f' -- NOT AS LABELLED (batch {run.effective_batch}, '
                        f'provider {run.effective_provider})')
                     if not run.error else f'FAILED {run.error}')
                  + f' [{time.time() - started:.0f}s]')
        if self._cancel.is_set():
            raise Cancelled()
        return run

    # -- the protocol -----------------------------------------------------
    def run(self) -> dict:
        arms = candidate_arms(self.providers, self.threads)
        if self.baseline not in arms:
            arms.insert(0, self.baseline)
        screen = screening_order(arms)
        with self._lock:
            self.progress = TuneProgress(
                running=True, phase='screen',
                arms_total=len(screen) + FINALISTS * 4,
                status='starting')
        result: Dict[str, Any] = {
            'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'baseline': self.baseline.key, 'screen_frames': self.screen_frames,
            'confirm_frames': self.confirm_frames,
            'confirm_meets_600_rule': self.confirm_frames >= CONFIRM_FRAMES,
            'provider_notes': self.provider_notes,
        }
        try:
            self._log(f'screening {len(arms)} arms x2 at {self.screen_frames} frames '
                      f'(baseline {self.baseline.key})')
            screen_runs = [self._run(a, self.screen_frames, 'screen') for a in screen]
            summary = summarize(screen_runs)
            finalists, excluded = pick_finalists(summary, self.baseline)
            result['screen'] = [
                {'arm': k, 'mean_fps': round(v['mean_fps'], 3), 'min_swaps': v['min_swaps'],
                 'runs': len(v['fps']), 'excluded': excluded.get(k)}
                for k, v in sorted(summary.items(), key=lambda kv: -kv[1]['mean_fps'])]
            result['finalists'] = [a.key for a in finalists]
            with self._lock:
                self.progress.phase = 'confirm'
                self.progress.arms_total = len(screen) + 4 * len(finalists)
            confirmations = []
            winner, winner_fps = self.baseline, summary.get(self.baseline.key, {}).get('mean_fps', 0.0)
            for cand in finalists:
                order = confirm_order(self.baseline, cand)
                runs = [self._run(a, self.confirm_frames, 'confirm') for a in order]
                v = verdict([r for r in runs if r.arm == self.baseline],
                            [r for r in runs if r.arm == cand])
                v['arm'] = cand.key
                confirmations.append(v)
                self._log(f'confirm {cand.key}: {v["reason"]} '
                          f'({v.get("improvement_pct")}% vs noise {v.get("noise_pct")}%)')
                if v['accepted'] and (winner == self.baseline
                                      or v['candidate_fps'] > winner_fps):
                    winner, winner_fps = cand, v['candidate_fps']
                elif winner == self.baseline and v.get('baseline_fps'):
                    winner_fps = v['baseline_fps']
            result['confirm'] = confirmations
            result['winner'] = {'arm': winner.key, 'provider': winner.provider,
                                'batch': winner.batch, 'fps': round(winner_fps, 3),
                                'changed': winner != self.baseline}
            if self.encode is not None:
                with self._lock:
                    self.progress.phase = 'encoder'
                    self.progress.status = 'NVENC presets'
                enc = self.encode(NVENC_PRESETS)
                pick = pick_nvenc_preset(enc.get('rows', []), winner_fps)
                enc['picked'] = pick['preset'] if pick else None
                enc['need_fps'] = round(winner_fps * ENCODER_HEADROOM, 2)
                result['encoder'] = enc
                if pick:
                    self._log(f"NVENC: {pick['preset']} ({pick['fps']:.0f} fps encode vs "
                              f"{enc['need_fps']:.1f} needed)")
            result['status'] = 'complete'
        except Cancelled:
            result['status'] = 'cancelled'
            self._log('cancelled')
        except Exception as exc:
            _swallowed("autotune.py:run", exc, "auto-tune stopped, nothing applied")
            result['status'] = 'failed'
            result['error'] = f'{type(exc).__name__}: {exc}'
            self._log('FAILED ' + result['error'])
            with self._lock:
                self.progress.error = result['error']
        result['runs'] = [r.as_dict() for r in self.runs]
        result['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with self._lock:
            self.progress.running = False
            self.progress.phase = result['status']
            self.progress.status = result['status']
        return result


def save_result(result: dict, path: str = RESULT_FILE) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, default=str)
    os.replace(tmp, path)


def load_result(path: str = RESULT_FILE) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
