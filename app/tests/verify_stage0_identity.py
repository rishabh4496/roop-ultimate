#!/usr/bin/env python3
"""Phase 6, Stage 0 end to end: the frozen source-identity table hands the swapper the SAME vector the
live per-face selection would have, and the swap pass stops re-deriving it.

Same harness as the other Phase 5/6 verifiers (`verify_three_stage_pipeline.py`: the synthesized
600-frame ground-truth clip, the production render with the user's live config.yaml models). The
source is a V2 faceset built from the library's `ashna` with real detection (`write_faceset_v2`), because
EVERY shipped `.fsz` is a V1 archive (34 of 34): the V2 pose-cell path, and so the table, is a no-op on
V1 sets without an identity recipe. The V2 archive is written into the work dir, never into the library.

FOUR RENDERS (live = `ROOP_STAGE0=0`, the original per-face selection):

    L0  live, no recipe      T0  table, no recipe
    L1  live, recipe         T1  table, recipe  (Identity Blender: jawline dial 0.6)

A TAP sits at the swapper boundary (`FaceSwapInsightFace.Run` / `RunBatch`, `SwapBatcher.submit`): for
every face-frame it records the identity vector actually handed over AND recomputes what the live path
would hand over for THAT target face (the same two functions, called outside the counters). Comparing
the two per face-frame removes the one confound a cross-run comparison has: two renders run their own
pre-pass, and GPU noise can move a keypoint across a pose-cell edge in one and not the other.

INVARIANTS (all must hold; exit 0 only then):

  S1  exact        every face-frame of T0 and T1: the vector handed to the swapper == the live path's, bit for
                   bit (embedding AND the unit `normed_embedding`), and the tap saw > 0 face-frames
  S2  armed        T0/T1 printed "[Stage0] source 0: 9 of 9 pose cells" and the table was HIT for (nearly)
                   every swapped face; L0/L1 printed the decline "ROOP_STAGE0=0" and never hit it
  S3  not re-derived   `pose_embedding_for_target` and `ResolvedBlend.transform` were called 0 times AFTER
                   Stage 3 began in T0/T1 (the live arms: > 0, or the counter proves nothing). Counted at
                   the functions, split at the moment the pre-pass hands over
  S4  same swaps   T0 swapped the same frames as L0, T1 the same as L1, with the same face counts
  S5  stage 1      Stage 1 never read a source vector: 0 calls of either function before the pre-pass in the
                   table arms' Stage 1 window (it matches the captured TARGET faces)

Also reported (not gated): per-face selection cost, live vs table, measured on the real set; PSNR of the
table renders against the live ones on sampled frames (lossy encoder, so a bound).
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                           # noqa: E402

import fixtures                              # noqa: E402
import verify_three_stage_pipeline as V      # noqa: E402


class Tap:
    def __init__(self):
        self.lock = threading.Lock()
        self.tls = threading.local()
        self.reset()

    def reset(self):
        # 'pre' = Stage 0 and everything before the sparse scan, 'stage1' = inside `scan_active_ranges`,
        # 'prepass' = after it until the pre-pass hands over, 'stage3' = from there to the end.
        self.phase = 'pre'
        self.pose_calls = collections.Counter()
        self.blend_calls = collections.Counter()
        self.face_frames = 0
        self.mismatches = []
        self.fs = None
        self.blend = None
        self.stage0 = None

    def count(self, table):
        if getattr(self.tls, 'quiet', False):
            return
        with self.lock:
            table[self.phase] += 1


def install(T: Tap, P):
    from roop.ProcessMgr import ProcessMgr
    from roop.processors import FaceSwapInsightFace as FSI
    from roop.identity_algebra import ResolvedBlend
    from roop import stage0_preflight as s0
    from roop.swap_batcher import SwapBatcher

    orig_pose = FSI.pose_embedding_for_target

    def pose(*a, **k):
        T.count(T.pose_calls)
        return orig_pose(*a, **k)

    FSI.pose_embedding_for_target = pose

    orig_tr = ResolvedBlend.transform

    def tr(self, *a, **k):
        T.count(T.blend_calls)
        return orig_tr(self, *a, **k)

    ResolvedBlend.transform = tr

    orig_pre = ProcessMgr._release_replayed_analysis

    def seal(self, frame_count):
        T.phase = 'stage3'
        T.stage0 = getattr(self, '_stage0', None)
        T.blend = getattr(self, '_identity_blend', None)
        return orig_pre(self, frame_count)

    ProcessMgr._release_replayed_analysis = seal

    from roop import sparse_range_scan
    orig_scan = sparse_range_scan.scan_active_ranges

    def scan(*a, **k):
        T.phase = 'stage1'
        try:
            return orig_scan(*a, **k)
        finally:
            T.phase = 'prepass'

    sparse_range_scan.scan_active_ranges = scan

    orig_swap = ProcessMgr.swap_faces

    def swap_faces(self, frame, *a, **k):
        T.tls.idx = k.get('frame_idx')
        return orig_swap(self, frame, *a, **k)

    ProcessMgr.swap_faces = swap_faces

    def live_vector(inputface, target_face):
        """What the live path hands the swapper for this target, computed OUTSIDE the counters."""
        T.tls.quiet = True
        try:
            fs = T.fs
            base = fs.faces[fs.canonical_face_index()]
            face = s0.pose_conditioned_input(base, fs, target_face)
            if T.blend is not None:
                face = s0.blended_input(face, T.blend, fs, target_face)
            return face
        finally:
            T.tls.quiet = False

    def check(src, tgt):
        if getattr(T.tls, 'quiet', False) or src is None:
            return
        try:
            want = live_vector(src, tgt)
            got_e, want_e = np.asarray(src['embedding']), np.asarray(want['embedding'])
            ok = got_e.dtype == want_e.dtype and np.array_equal(got_e, want_e)
            got_n, want_n = src.get('normed_embedding') if hasattr(src, 'get') else None, want.get('normed_embedding')
            if ok and (got_n is None) != (want_n is None):
                ok = False
            elif ok and got_n is not None:
                ok = np.array_equal(np.asarray(got_n), np.asarray(want_n))
            with T.lock:
                T.face_frames += 1
                if not ok:
                    T.mismatches.append({'frame': getattr(T.tls, 'idx', None),
                                         'max_abs_diff': float(np.max(np.abs(got_e.astype(np.float64) - want_e)))})
        except Exception as exc:                                   # the tap failing is a finding, not a pass
            with T.lock:
                T.mismatches.append({'tap_error': f'{type(exc).__name__}: {exc}'})

    orig_run, orig_batch = FSI.FaceSwapInsightFace.Run, FSI.FaceSwapInsightFace.RunBatch

    def run(self, source_face, target_face, temp_frame):
        check(source_face, target_face)
        return orig_run(self, source_face, target_face, temp_frame)

    def run_batch(self, source_face, target_face, temp_frames):
        check(source_face, target_face)
        return orig_batch(self, source_face, target_face, temp_frames)

    FSI.FaceSwapInsightFace.Run, FSI.FaceSwapInsightFace.RunBatch = run, run_batch
    orig_submit = SwapBatcher.submit

    def submit(self, src, tgt, blob):
        check(src, tgt)
        return orig_submit(self, src, tgt, blob)

    SwapBatcher.submit = submit


def psnr_samples(a_path, b_path, n=10):
    import cv2
    ca, cb = cv2.VideoCapture(a_path), cv2.VideoCapture(b_path)
    total = int(ca.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, total - 1, n).astype(int):
        ca.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        cb.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok_a, fa = ca.read()
        ok_b, fb = cb.read()
        if ok_a and ok_b and fa.shape == fb.shape:
            mse = float(np.mean((fa.astype(np.float32) - fb.astype(np.float32)) ** 2))
            out.append(99.0 if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 2))
    return out


def microbench(fs, blend, table, n=20000):
    """Per-face cost of choosing the vector: the live two steps vs the table lookup (CPU, one thread)."""
    from insightface.app.common import Face
    from roop import stage0_preflight as s0
    rng = np.random.default_rng(0)
    targets = [Face(_adaptive_yaw=float(rng.uniform(-80, 80)), _adaptive_pitch=float(rng.uniform(-35, 35)),
                    _adaptive_roll=float(rng.uniform(-40, 40))) for _ in range(256)]
    base = fs.faces[fs.canonical_face_index()]

    def timed(fn):
        t0 = time.perf_counter()
        for i in range(n):
            fn(targets[i & 255])
        return (time.perf_counter() - t0) / n * 1e6

    def live(t):
        f = s0.pose_conditioned_input(base, fs, t)
        if blend is not None:
            f = s0.blended_input(f, blend, fs, t)
        return f

    return {'live_us_per_face': round(timed(live), 2),
            'table_us_per_face': round(timed(lambda t: table.lookup(0, base, t)), 2), 'iterations': n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--enhancer', default=None)
    ap.add_argument('--angle-clip', default=None)
    args = ap.parse_args()
    workdir = os.path.abspath(args.workdir or tempfile.mkdtemp(prefix='verify_stage0_'))
    os.makedirs(workdir, exist_ok=True)
    report_path = os.path.abspath(args.out or os.path.join(workdir, 'verify_stage0_identity.json'))
    angle = args.angle_clip or fixtures.clip('angle.mp4')
    if not os.path.isfile(angle):
        print(f'[verify_stage0] PENDING: real profile footage not found ({angle})')
        return 3
    t_start = time.time()
    report = {'script': 'app/tests/verify_stage0_identity.py', 'workdir': workdir,
              'git_head': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip()}
    os.environ['ROOP_STAB_CHUNK_MB'] = '300'          # pins the stabilizer chunk geometry (pixels depend on free RAM)

    frames, truth, sprites = V.build_scene(angle)
    clip = V.write_clip(frames, os.path.join(workdir, 'stage0_cfr.mp4'), vfr=False, audio=False)
    P = V.Probe()
    ctx = V.prepare_pipeline(args, workdir, frames, sprites)
    V.install_hooks(P)
    report['models'] = ctx.config_view

    # the V2 source
    import compare_enhancers_video as C
    import roop.globals as g
    from roop.faceset_v2 import write_faceset_v2
    from two_face_video import load_library_faceset
    fs = load_library_faceset(V.SOURCE_FACESET)
    v2_path = os.path.join(workdir, V.SOURCE_FACESET + '_v2.fsz')
    meta = write_faceset_v2(v2_path, fs, fs.ref_images, source_name=V.SOURCE_FACESET)
    fs.attach_v2_metadata(meta)
    fs._source_id = V.SOURCE_FACESET
    report['source'] = {'faces': len(fs.faces), 'format_version': fs.format_version,
                        'pose_cells_filled': sorted(k for k in (meta.get('pose_bins') or {}))}
    C.load_library_faceset = lambda name: fs
    print(f"[verify_stage0] V2 source: {report['source']}", flush=True)

    T = Tap()
    install(T, P)
    recipe = {'enabled': True, 'components': [], 'dials': {'jawline': 0.6}}
    out_root = os.path.join(workdir, 'out')
    three = {'sparse_auto_range': True}

    def arm(tag, stage0_on, with_recipe):
        os.environ['ROOP_STAGE0'] = '1' if stage0_on else '0'
        g.identity_blend = dict(recipe) if with_recipe else None
        T.reset()
        T.fs = fs
        run = V.render(ctx, P, clip, tag, out_root, three)
        text = open(run['log'], encoding='utf-8', errors='replace').read()
        stage0_lines = [l.strip() for l in text.splitlines() if l.startswith('[Stage0]')]
        stats = T.stage0.stats if T.stage0 is not None else None
        run.update({
            'stage0_lines': stage0_lines[:4],
            'tap_face_frames': T.face_frames, 'tap_mismatches': T.mismatches[:5], 'tap_mismatch_count': len(T.mismatches),
            'pose_calls': collections.Counter(T.pose_calls), 'blend_calls': collections.Counter(T.blend_calls),
            'table_hits': None if stats is None else stats.hits, 'table_misses': None if stats is None else stats.misses,
            'table_armed': T.stage0 is not None, 'blend_active': T.blend is not None})
        print(f"[verify_stage0] {tag}: {run['seconds']} s, {len(run['swapped_frames'])} swapped frames, "
              f"tap {T.face_frames} face-frames, mismatches {len(T.mismatches)}, pose calls {T.pose_calls}, "
              f"blend calls {T.blend_calls}, table hits {run['table_hits']}", flush=True)
        return run

    # warm-up (engine builds must not land in an arm)
    os.environ['ROOP_STAGE0'] = '1'
    V.render(ctx, P, clip, 'warmup', os.path.join(workdir, 'out_warm'), three, rng=(0, 90))
    L0 = arm('L0_live', False, False)
    T0 = arm('T0_table', True, False)
    L1 = arm('L1_live_recipe', False, True)
    T1 = arm('T1_table_recipe', True, True)
    renders = [L0, T0, L1, T1]

    inv = {}
    inv['S1_exact'] = {
        'pass': bool(all(r['tap_face_frames'] > 0 and r['tap_mismatch_count'] == 0 for r in renders)),
        'face_frames': {r['tag']: r['tap_face_frames'] for r in renders},
        'mismatches': {r['tag']: r['tap_mismatches'] for r in renders if r['tap_mismatch_count']}}
    inv['S2_armed'] = {
        'pass': bool(all(r['table_armed'] and r['table_hits'] > 0 and r['table_misses'] == 0
                         and any('9 of 9 pose cells' in l for l in r['stage0_lines']) for r in (T0, T1))
                     and all(not r['table_armed'] and any('ROOP_STAGE0=0' in l for l in r['stage0_lines'])
                             for r in (L0, L1))),
        'hits': {r['tag']: r['table_hits'] for r in (T0, T1)}, 'misses': {r['tag']: r['table_misses'] for r in (T0, T1)},
        'stage0_lines': {r['tag']: r['stage0_lines'][:2] for r in renders}}
    inv['S3_not_rederived'] = {
        'pass': bool(T0['pose_calls']['stage3'] == 0 and T1['pose_calls']['stage3'] == 0
                     and T1['blend_calls']['stage3'] == 0
                     and L0['pose_calls']['stage3'] > 0 and L1['pose_calls']['stage3'] > 0
                     and L1['blend_calls']['stage3'] > 0),
        'pose_calls_stage3': {r['tag']: r['pose_calls']['stage3'] for r in renders},
        'blend_calls_stage3': {r['tag']: r['blend_calls']['stage3'] for r in renders},
        'pose_calls_before_stage3': {r['tag']: r['pose_calls']['pre'] + r['pose_calls']['prepass'] for r in renders}}
    inv['S4_same_swaps'] = {
        'pass': bool(sorted(L0['swapped_frames']) == sorted(T0['swapped_frames'])
                     and sorted(L1['swapped_frames']) == sorted(T1['swapped_frames'])
                     and L0['faces_swapped'] == T0['faces_swapped'] and L1['faces_swapped'] == T1['faces_swapped']),
        'faces_swapped': {r['tag']: r['faces_swapped'] for r in renders}}
    inv['S5_stage1_reads_no_source_vector'] = {
        'pass': bool(all(r['pose_calls']['stage1'] == 0 and r['blend_calls']['stage1'] == 0 for r in renders)
                     and all(r['pose_calls']['stage3'] > 0 for r in (L0, L1))),     # the counter works: live Stage 3 calls it
        'note': '`sparse_range_scan.scan_active_ranges` takes the captured TARGET faces only. Calls per phase '
                '(pre = Stage 0 build and before; prepass; stage3) are in the report.',
        'pose_calls_by_phase': {r['tag']: dict(r['pose_calls']) for r in renders},
        'blend_calls_by_phase': {r['tag']: dict(r['blend_calls']) for r in renders}}
    bad = [{'tag': r['tag'], 'error': r['error'], 'signatures': {k: v for k, v in r['failure_signatures'].items() if v}}
           for r in renders if r['error'] or any(r['failure_signatures'].values())]
    inv['0_zero_runtime_exceptions'] = {'pass': not bad, 'offending_runs': bad}

    # cost
    from roop import stage0_preflight as s0
    from roop.identity_algebra import BlendRecipe, resolve_recipe, load_directions
    table_plain = s0.build([fs], swap_mode='normed', use_source_bank=False, quality_engine=False, blend=None, log=lambda *_: None)
    blend = resolve_recipe(BlendRecipe.from_payload(recipe), [fs], load_directions())
    table_blend = s0.build([fs], swap_mode='normed', use_source_bank=False, quality_engine=False, blend=blend, log=lambda *_: None)
    report['selection_cost_per_face'] = {'no_recipe': microbench(fs, None, table_plain),
                                         'with_recipe': microbench(fs, blend, table_blend)}
    report['frame_time_context_ms'] = round(1000.0 * min(r['seconds'] for r in renders) / V.N_FRAMES, 1)
    report['psnr_table_vs_live'] = {'T0_vs_L0': psnr_samples(L0['output'], T0['output']) if L0['output'] and T0['output'] else None,
                                    'T1_vs_L1': psnr_samples(L1['output'], T1['output']) if L1['output'] and T1['output'] else None}
    report['seconds'] = {r['tag']: r['seconds'] for r in renders}
    report['invariants'] = inv
    ok = all(v['pass'] for v in inv.values())
    report['all_invariants_hold'] = ok
    report['wall_seconds'] = round(time.time() - t_start, 1)
    report['exit'] = 0 if ok else 1
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, default=str)
    print('\n== invariants ==')
    for k, v in sorted(inv.items()):
        brief = {a: b for a, b in v.items() if a not in ('offending_runs',)}
        print(('  PASS  ' if v['pass'] else '  FAIL  ') + k + '  ' + json.dumps(brief, default=str)[:420])
    print('\n== selection cost per face (us) ==  ' + json.dumps(report['selection_cost_per_face']))
    print(f"   one frame of this pipeline takes ~{report['frame_time_context_ms']} ms")
    print(f"\n[verify_stage0] {'ALL INVARIANTS HOLD' if ok else 'INVARIANT(S) VIOLATED'}  report: {report_path}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
