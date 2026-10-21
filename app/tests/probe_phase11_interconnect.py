#!/usr/bin/env python3
"""Phase 11 baseline: what the five "interconnectivity" items would actually save, measured on a real render.

The brief asks for (1) one probe/demuxer context across the stages, (2) precomputed per-face affines consumed by
Stage 3 instead of CPU `cv2.estimateAffinePartial2D`, (3) the Stage 1/2 detector + recogniser unloaded after
Stage 2, (4) a bounded-queue async encoder pipe with the audio mapped in the same pass, (5) a 10 Hz progress
throttle. AGENTS.md: a stage is not "built" or "not built" on a reading, so this renders the Phase 5 ground-truth
clip (600 frames, CFR, audio) through the production path (`core.batch_process_with_options`, the live
config.yaml models, `sparse_auto_range` ON) and counts, per phase (open / stage 1 / stage 2 / stage 3 / finalize):

  * every `cv2.VideoCapture(...)` construction and every ffprobe/ffmpeg process spawn, with its wall time
  * every CPU affine call (`estimateAffinePartial2D`, `invertAffineTransform`, ...) and `warpAffine`, with its time
  * time inside the writer's `write_frame` (the encoder pipe) and inside `finalize_render` (the audio mux)
  * time inside `ProcessMgr.update_progress` (what a progress throttle could save)
  * device VRAM before/after the Stage 2 -> 3 analysis release, and which analyser models stay resident

    app/env/Scripts/python.exe app/tests/probe_phase11_interconnect.py [--workdir DIR] [--out report.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                       # noqa: E402

import fixtures                                  # noqa: E402
import verify_three_stage_pipeline as V          # noqa: E402

PHASE = {'name': 'init'}
_LOCK = threading.Lock()
STATS = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))     # kind -> phase -> [n, ns]
EVENTS = []                                                                          # phase transitions, VRAM marks
DETAIL = collections.defaultdict(list)                                               # kind -> [(phase, what, ms)]
_T0 = time.perf_counter()


def add(kind: str, ns: int, what: str = None) -> None:
    with _LOCK:
        cell = STATS[kind][PHASE['name']]
        cell[0] += 1
        cell[1] += ns
        if what is not None and len(DETAIL[kind]) < 400:
            DETAIL[kind].append((PHASE['name'], what, round(ns / 1e6, 2)))


def set_phase(name: str) -> None:
    with _LOCK:
        PHASE['name'] = name
        EVENTS.append({'t': round(time.perf_counter() - _T0, 3), 'phase': name})


def timed(kind: str, fn, what=None):
    def wrapper(*a, **k):
        t = time.perf_counter_ns()
        try:
            return fn(*a, **k)
        finally:
            add(kind, time.perf_counter_ns() - t, what)
    wrapper.__wrapped__ = fn
    return wrapper


def vram_used_mib() -> float:
    import torch
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 2 ** 20, 1)


def install() -> None:
    # --- cv2: container opens + CPU affine math -------------------------------------------------------------
    orig_vc = cv2.VideoCapture

    # A FUNCTION returning real VideoCapture objects, not a subclass: a Python subclass of the cv2 binding
    # segfaulted inside gc.collect() (faulthandler: "Garbage-collecting" in the Stage 2 -> 3 release hook).
    def VC(*a, **k):                             # noqa: N802 - mirrors the cv2 name
        t = time.perf_counter_ns()
        cap = orig_vc(*a, **k)
        add('cv2.VideoCapture', time.perf_counter_ns() - t, os.path.basename(str(a[0])) if a else '?')
        return cap

    cv2.VideoCapture = VC
    for name in ('estimateAffinePartial2D', 'estimateAffine2D', 'invertAffineTransform', 'getAffineTransform',
                 'warpAffine', 'getPerspectiveTransform', 'warpPerspective'):
        fn = getattr(cv2, name, None)
        if fn is not None:
            setattr(cv2, name, timed('cv2.' + name, fn))

    # --- python-level alignment helpers (modules import them by name, so patch every reference) ----------------
    import roop.face_util as FU
    for name in ('estimate_norm', 'align_crop'):
        orig = getattr(FU, name, None)
        if orig is None:
            continue
        new = timed('face_util.' + name, orig)
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.startswith('roop') and getattr(mod, name, None) is orig:
                setattr(mod, name, new)

    # --- processes: ffprobe / ffmpeg spawns, spawn cost AND lifetime ----------------------------------------------
    orig_init, orig_comm, orig_wait = subprocess.Popen.__init__, subprocess.Popen.communicate, subprocess.Popen.wait

    def p_init(self, args, *a, **k):
        t = time.perf_counter_ns()
        orig_init(self, args, *a, **k)
        argv0 = args if isinstance(args, str) else (args[0] if args else '')
        base = os.path.basename(str(argv0).split(' ')[0]).lower().replace('.exe', '')
        self._p11 = (PHASE['name'], base, t)
        add('spawn.' + base, time.perf_counter_ns() - t, ' '.join(str(x) for x in (args if not isinstance(args, str) else [args])[1:6])[:90])

    def _finish(self):
        meta = getattr(self, '_p11', None)
        if meta and not getattr(self, '_p11_done', False) and self.returncode is not None:
            self._p11_done = True
            with _LOCK:
                cell = STATS['life.' + meta[1]][meta[0]]
                cell[0] += 1
                cell[1] += time.perf_counter_ns() - meta[2]

    def p_comm(self, *a, **k):
        try:
            return orig_comm(self, *a, **k)
        finally:
            _finish(self)

    def p_wait(self, *a, **k):
        try:
            return orig_wait(self, *a, **k)
        finally:
            _finish(self)

    subprocess.Popen.__init__, subprocess.Popen.communicate, subprocess.Popen.wait = p_init, p_comm, p_wait

    # --- writers / progress / finalize ------------------------------------------------------------------------------
    from roop.ffmpeg_writer import FFMPEG_VideoWriter
    from roop.segment_writer import SegmentedVideoWriter
    depth = threading.local()

    def guarded(kind, fn):
        def w(self, *a, **k):
            if getattr(depth, 'in_write', False):
                return fn(self, *a, **k)
            depth.in_write = True
            t = time.perf_counter_ns()
            try:
                return fn(self, *a, **k)
            finally:
                depth.in_write = False
                add(kind, time.perf_counter_ns() - t)
        return w

    SegmentedVideoWriter.write_frame = guarded('writer.write_frame', SegmentedVideoWriter.write_frame)
    FFMPEG_VideoWriter.write_frame = guarded('writer.write_frame', FFMPEG_VideoWriter.write_frame)
    SegmentedVideoWriter.close = guarded('writer.close', SegmentedVideoWriter.close)

    from roop.ProcessMgr import ProcessMgr
    ProcessMgr.update_progress = timed('progress.update_progress', ProcessMgr.update_progress)

    from roop import finalize_output
    finalize_output.finalize_render = timed('finalize_render', finalize_output.finalize_render)

    # --- phase markers ------------------------------------------------------------------------------------------------
    orig_run = ProcessMgr.run_batch_inmem

    def run_inmem(self, *a, **k):
        set_phase('open')
        try:
            return orig_run(self, *a, **k)
        finally:
            set_phase('finalize')

    ProcessMgr.run_batch_inmem = run_inmem

    import roop.sparse_range_scan as srs
    orig_scan = srs.scan_active_ranges

    def scan(*a, **k):
        set_phase('stage1')
        try:
            return orig_scan(*a, **k)
        finally:
            set_phase('gap')

    srs.scan_active_ranges = scan

    orig_pre = ProcessMgr._precompute_temporal

    def pre(self, *a, **k):
        set_phase('stage2')
        try:
            return orig_pre(self, *a, **k)
        finally:
            set_phase('seal')

    ProcessMgr._precompute_temporal = pre

    orig_rel = ProcessMgr._release_replayed_analysis

    def rel(self, frame_count):
        import roop.face_util as fu
        import gc
        gc.collect()
        before = vram_used_mib()
        resident_before = _analyser_models(fu)
        r = orig_rel(self, frame_count)
        gc.collect()
        after = vram_used_mib()
        EVENTS.append({'release_replayed_analysis': {
            'returned': r, 'vram_used_mib_before': before, 'vram_used_mib_after': after,
            'freed_mib': round(before - after, 1), 'analyser_models_before': resident_before,
            'analyser_models_after': _analyser_models(fu)}})
        set_phase('stage3')
        return r

    ProcessMgr._release_replayed_analysis = rel


def _analyser_models(fu) -> dict:
    out = {}
    for i, fa in enumerate(list(getattr(fu, 'FACE_ANALYSER_POOL', []) or [])):
        models = getattr(fa, 'models', {}) or {}
        out[f'pool{i}'] = sorted(models)
    return out


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak = collections.defaultdict(float)

    def run(self):
        while not self.stop.is_set():
            try:
                v = vram_used_mib()
                p = PHASE['name']
                self.peak[p] = max(self.peak[p], v)
            except Exception:
                pass
            time.sleep(0.25)


def summarize(frames: int, swapped_faces: int, wall: float) -> dict:
    out = {}
    for kind, phases in sorted(STATS.items()):
        out[kind] = {ph: {'n': n, 'ms': round(ns / 1e6, 2)} for ph, (n, ns) in sorted(phases.items())}
        out[kind]['_total'] = {'n': sum(n for n, _ in phases.values()), 'ms': round(sum(ns for _, ns in phases.values()) / 1e6, 2)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--enhancer', default=None)
    ap.add_argument('--runs', type=int, default=2, help='measured 600-frame renders after the warm-up')
    args = ap.parse_args()
    workdir = os.path.abspath(args.workdir or tempfile.mkdtemp(prefix='phase11_'))
    os.makedirs(workdir, exist_ok=True)
    report_path = os.path.abspath(args.out or os.path.join(workdir, 'probe_phase11_interconnect.json'))
    angle = fixtures.clip('angle.mp4')
    if not os.path.isfile(angle):
        print(f'PENDING: {angle} not found')
        return 3

    frames, truth, sprites = V.build_scene(angle)
    cfr = V.write_clip(frames, os.path.join(workdir, 'phase11_cfr.mp4'), vfr=False, audio=True)
    P = V.Probe()
    ctx = V.prepare_pipeline(args, workdir, frames, sprites)
    V.install_hooks(P)
    install()

    report = {'script': 'app/tests/probe_phase11_interconnect.py', 'models': ctx.config_view,
              'git_head': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
              'clip': V.probe_clip(cfr), 'runs': []}
    out_root = os.path.join(workdir, 'out')
    three = {'sparse_auto_range': True}

    set_phase('warmup')
    warm = V.render(ctx, P, cfr, 'warmup_3stage_first90', out_root, three, rng=(0, 90))
    report['warmup'] = {k: warm[k] for k in ('seconds', 'error', 'failure_signatures')}

    for i in range(args.runs):
        STATS.clear()
        DETAIL.clear()
        EVENTS.clear()
        set_phase('pre')
        sampler = VramSampler()
        sampler.start()
        t = time.perf_counter()
        run = V.render(ctx, P, cfr, f'measured_{i}', out_root, three)
        wall = time.perf_counter() - t
        sampler.stop.set()
        sampler.join(timeout=2)
        marks = [e for e in EVENTS if 'phase' in e]
        spans = {}
        for a, b in zip(marks, marks[1:] + [{'t': round(time.perf_counter() - _T0, 3), 'phase': 'end'}]):
            spans[a['phase']] = round(spans.get(a['phase'], 0.0) + b['t'] - a['t'], 3)
        report['runs'].append({
            'tag': f'measured_{i}', 'seconds': run['seconds'], 'fps': run['fps'], 'error': run['error'],
            'output_frames': run['output_frames'], 'faces_swapped': run['faces_swapped'],
            'swapped_frames': len(run['swapped_frames']), 'detector_calls': run['detector_calls'],
            'phase_wall_s': spans, 'peak_vram_mib_by_phase': {k: round(v, 1) for k, v in sampler.peak.items()},
            'release_event': next((e['release_replayed_analysis'] for e in EVENTS if 'release_replayed_analysis' in e), None),
            'stats': summarize(600, run['faces_swapped'], wall),
            'detail': {k: v for k, v in DETAIL.items() if k.startswith(('spawn.', 'cv2.VideoCapture'))},
            'failure_signatures': run['failure_signatures'], 'log': run['log'],
        })
        print(f"[probe_phase11] run {i}: {run['seconds']} s, {run['fps']} fps, phases {spans}", flush=True)
    json.dump(report, open(report_path, 'w'), indent=1, default=str)
    print(f'[probe_phase11] report {report_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
