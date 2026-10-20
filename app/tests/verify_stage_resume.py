#!/usr/bin/env python3
"""Phase 6 end-to-end: an interrupted job resumes at Stage 3 without re-running Stage 1 or Stage 2.

Built on the Phase 5 harness (`verify_three_stage_pipeline.py`: the same synthesized 600-frame
ground-truth clip, the same in-process production render with the user's live config.yaml models and
`sparse_auto_range` ON). What this adds is a real PROJECT record (`project_checkpoint`) and the stage
cache (`stage_checkpoint.ProjectStageCache`) the API installs for a project's run.

THE SEQUENCE (one process, one pipeline init):

    A    reference    the whole clip, no project, no cache                       what an uninterrupted run does
    B1   interrupted  the project's first run, STOPPED part-way through Stage 3  (ROOP_RESUME_KEEP=1 keeps the
                      committed segments, ROOP_RESUME_CHUNK=100 makes them small)
    B2   resumed      the same project again                                     must skip Stages 1 and 2
    B3   tampered     B1's state restored, the Stage 2 sidecar edited by 1 byte  must REFUSE the cache and recompute
    B4   re-keyed     B1's state restored, a ROOP_* knob changed                 must REFUSE the cache and recompute

INVARIANTS (all must hold; exit 0 only then):

  R1  recorded      right after B1 stopped, the project holds a Stage 1 AND a Stage 2 record under one key,
                    written while B1 was still running (before its first segment commit would matter)
  R2  restored      B2 prints "Stage 1/3 restored" and "Stage 2 restored", /ws/telemetry's stage block reads
                    RESTORED for both, and it resumed at the committed frame (not frame 0)
  R3  no detector   B2 made 0 detector calls IN TOTAL (`face_util._detect_faces_raw`, the funnel under every
                    engine branch) - the pre-pass cost is gone, not moved. The reference makes N > 0, or the
                    counter proves nothing.
  R4  same swaps    B1's swapped frames + B2's (shifted by the resume point) == A's swapped frames
  R5  same plan     B2's Stage 2 plan == A's plan restricted to the resumed window
  R6  complete      B2's output has 600 frames
  R7  refuses       B3 and B4 each print why the cache was not used, make detector calls > 0 (the stages RAN),
                    and still produce 600 frames with the same swapped frames as A (a refused cache is a slow
                    resume, never a wrong one)

The numbers worth reading beside the verdict: B2's wall time against a resume that re-runs the pre-passes
(B3), and the cache's size and write/read time.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                           # noqa: E402

import fixtures                              # noqa: E402
import verify_three_stage_pipeline as V      # noqa: E402
import project_checkpoint as pc              # noqa: E402
import stage_checkpoint as sck               # noqa: E402

STOP_AT = 435                                # B1 is stopped once this many frames have been swapped-through
CHUNK = '100'                                # ROOP_RESUME_CHUNK: committed segment size
STAB_MB = '150'                              # ROOP_STAB_CHUNK_MB: ~200-frame stabilizer chunks (the default is ~480
                                             # on this clip, so a stop at frame 430 had written NOTHING yet) - also
                                             # the pin the repo requires for any A/B of rendered pixels


def hooks_for_plan(P):
    """Snapshot the plan at the moment Stage 3 starts, on BOTH paths (computed or restored)."""
    from roop.ProcessMgr import ProcessMgr
    orig = ProcessMgr._release_replayed_analysis

    def snap(self, frame_count):
        P.plan_at_stage3 = sorted(int(k) for k in (getattr(self, 'stage2_frame_plan', None) or {}))
        P.stage3_window = int(frame_count or 0)
        P.sealed = True              # from here to the end of the render is Stage 3, whichever path got us here
        return orig(self, frame_count)

    ProcessMgr._release_replayed_analysis = snap

    # Where the pipeline's own work begins. Everything the harness does BEFORE this - notably
    # `load_library_faceset`, which runs the detector over the source faceset's reference photos - is not a
    # stage and happens in every run, resumed or not (5 calls for `ashna`).
    orig_freeze = ProcessMgr._freeze_source_identity

    def at_stage0(self):
        with P._lock:
            P.calls_before_pipeline = int(P.raw['pre']) + int(P.raw['stage3'])
        return orig_freeze(self)

    ProcessMgr._freeze_source_identity = at_stage0


class Stopper(threading.Thread):
    """What /api/stop does: processing -> False, once enough frames have gone through the swap pass."""

    def __init__(self, P, at):
        super().__init__(daemon=True)
        self.P, self.at, self.done, self.fired_at = P, at, threading.Event(), None

    def run(self):
        import roop.globals as g
        while not self.done.is_set():
            if self.P.swapped and max(self.P.swapped) >= self.at:
                self.fired_at = max(self.P.swapped)
                g.processing = False
                return
            time.sleep(0.05)


def rend(ctx, P, clip, tag, out_dir, overrides, **kw):
    """V.render with the Stage-3 plan snapshot cleared first, so a run that never reaches the hook cannot
    inherit the previous run's plan."""
    P.plan_at_stage3 = None
    P.calls_before_pipeline = 0
    run = V.render(ctx, P, clip, tag, out_dir, overrides, **kw)
    run['calls_before_pipeline'] = int(P.calls_before_pipeline)
    return run


def total_detector_calls(run):
    """Detector calls made by the PIPELINE (Stage 0 onward): the harness's own ingest of the source photos
    is subtracted, because it runs before any stage in every run and is not what a resume skips."""
    total = int(run['detector_calls']['pre']) + int(run['detector_calls']['stage3'])
    return total - int(run.get('calls_before_pipeline') or 0)


def log_text(run):
    return open(run['log'], encoding='utf-8', errors='replace').read()


def first_int(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def sample_psnr(a_path, b_path, start, n=12):
    """PSNR between two renders on n frames from `start` on. Lossy encoders, so this is a bound, not 0."""
    import cv2
    cap_a, cap_b = cv2.VideoCapture(a_path), cv2.VideoCapture(b_path)
    total = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(start, total - 1, n).astype(int)
    out = []
    for i in picks:
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if ok_a and ok_b and fa.shape == fb.shape:
            mse = float(np.mean((fa.astype(np.float32) - fb.astype(np.float32)) ** 2))
            out.append(99.0 if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--enhancer', default=None)
    ap.add_argument('--angle-clip', default=None)
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir or tempfile.mkdtemp(prefix='verify_resume_'))
    os.makedirs(workdir, exist_ok=True)
    report_path = os.path.abspath(args.out or os.path.join(workdir, 'verify_stage_resume.json'))
    angle = args.angle_clip or fixtures.clip('angle.mp4')
    if not os.path.isfile(angle):
        print(f'[verify_stage_resume] PENDING: real profile footage not found ({angle})')
        return 3
    t_start = time.time()
    report = {'script': 'app/tests/verify_stage_resume.py', 'workdir': workdir,
              'git_head': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(), 'stop_at': STOP_AT, 'chunk': CHUNK, 'stab_chunk_mb': STAB_MB}

    frames, truth, sprites = V.build_scene(angle)
    clip = V.write_clip(frames, os.path.join(workdir, 'resume_cfr.mp4'), vfr=False, audio=True)
    P = V.Probe()
    ctx = V.prepare_pipeline(args, workdir, frames, sprites)
    V.install_hooks(P)
    hooks_for_plan(P)
    report['models'] = ctx.config_view

    import roop.globals as g
    from two_face_video import LIB
    projects = os.path.join(workdir, 'projects')
    os.makedirs(projects, exist_ok=True)
    pc.PROJECTS_DIR = projects
    source_path = os.path.join(LIB, V.SOURCE_FACESET + '.fsz')
    src = pc.file_identity(source_path)
    src['source_id'] = ''
    cfg = SimpleNamespace(provider='tensorrt', trt_precision='mixed', swap_model=ctx.swapper, output_path=workdir,
                          hardware={}, hardware_signature='', output_video_format='mp4', output_video_codec='',
                          video_quality=14, output_method='File', output_template='')
    record = pc.new_project(job_id=None, name='resume-verify', payload={'swap_model': ctx.swapper},
                            sources=[src], target=pc.file_identity(clip), frame_start=0, frame_end=V.N_FRAMES,
                            output={'directory': workdir}, cfg=cfg, target_faces=[])
    pid = record['id']
    report['project_id'] = pid

    out_ref = os.path.join(workdir, 'out_ref')
    out_run = os.path.join(workdir, 'out_run')
    three = {'sparse_auto_range': True}
    os.environ['ROOP_RESUME_CHUNK'] = CHUNK
    os.environ['ROOP_STAB_CHUNK_MB'] = STAB_MB

    def install_cache():
        g._stage_cache = sck.ProjectStageCache(pid, app_version='verify-stage-resume')

    # warm-up (engine builds and first-use loads must not land in a timed arm)
    g._stage_cache = None
    rend(ctx, P, clip, 'warmup', os.path.join(workdir, 'out_warm'), three, rng=(0, 90))

    # ── A: the uninterrupted reference ──────────────────────────────────────────────────────────────
    os.environ.pop('ROOP_RESUME_KEEP', None)
    g._stage_cache = None
    A = rend(ctx, P, clip, 'A_reference', out_ref, three)
    A['plan_stage3'] = list(getattr(P, 'plan_at_stage3', []) or [])
    print(f"[verify_stage_resume] A reference: {A['seconds']} s, {len(A['swapped_frames'])} swapped frames, "
          f"{total_detector_calls(A)} detector calls", flush=True)

    # ── B1: the project's first run, stopped mid-Stage 3 ────────────────────────────────────────────
    os.environ['ROOP_RESUME_KEEP'] = '1'
    install_cache()
    stopper_holder = {}

    def run_b1():
        P.reset()
        stopper = Stopper(P, STOP_AT)
        stopper_holder['s'] = stopper
        stopper.start()
        try:
            return rend(ctx, P, clip, 'B1_interrupted', out_run, three)
        finally:
            stopper.done.set()

    # render() resets the probe itself; the stopper must watch the SAME probe object after that reset
    B1 = run_b1()
    g.processing = True
    report['b1_stopped_after_swapped_frame'] = stopper_holder['s'].fired_at
    after_b1 = pc.usable_stage_cache(pid, sck.live_identity('verify-stage-resume'))
    manifest = [n for n in os.listdir(out_run) if n.endswith('.resume.json')]
    print(f"[verify_stage_resume] B1 interrupted at frame {stopper_holder['s'].fired_at}: "
          f"stage1={'yes' if after_b1['stage1'] else 'NO'} stage2={'yes' if after_b1['stage2'] else 'NO'} "
          f"manifest={manifest}", flush=True)
    snapshot = os.path.join(workdir, 'snapshot_after_b1')
    shutil.rmtree(snapshot, ignore_errors=True)
    shutil.copytree(out_run, snapshot)
    record_after = pc.load(pid)
    shutil.copy(pc.project_path(pid), os.path.join(workdir, 'record_after_b1.json'))
    sidecar_path = pc.stage_sidecar_path(pid, 'stage2')
    side_backup = os.path.join(workdir, 'stage2_sidecar_backup.npz')
    if os.path.isfile(sidecar_path):
        shutil.copy(sidecar_path, side_backup)

    def restore_state():
        shutil.rmtree(out_run, ignore_errors=True)
        shutil.copytree(snapshot, out_run)
        shutil.copy(os.path.join(workdir, 'record_after_b1.json'), pc.project_path(pid))
        if os.path.isfile(side_backup):
            shutil.copy(side_backup, sidecar_path)

    inv = {}
    stage_cache_entry = (record_after.get('stage_cache') or {})
    inv['R1_recorded'] = {
        'pass': bool(after_b1['stage1'] and after_b1['stage2'] and not after_b1['reasons']
                     and stage_cache_entry.get('key')),
        'stage1_ranges': (after_b1['stage1'] or {}).get('active_ranges'),
        'stage2': {k: v for k, v in (after_b1['stage2'] or {}).items() if k in ('faces', 'frames', 'plan_frames', 'sidecar')},
        'reasons': after_b1['reasons'], 'b1_error': B1['error']}

    # ── B2: the resume ─────────────────────────────────────────────────────────────────────────────
    install_cache()
    B2 = rend(ctx, P, clip, 'B2_resumed', out_run, three)
    b2_log = log_text(B2)
    skip = first_int(r'found (\d+) already-encoded frames', b2_log, 0)
    stage_states = (B2['stage_trace'][-1]['states'] if B2['stage_trace'] else {})
    all_states = [t['states'] for t in B2['stage_trace']]
    saw_restored = lambda k: any(s.get(k) == 'RESTORED' for s in all_states)           # noqa: E731
    print(f"[verify_stage_resume] B2 resumed at frame {skip}: {B2['seconds']} s, "
          f"{total_detector_calls(B2)} detector calls", flush=True)
    inv['R2_restored'] = {
        'pass': bool('Stage 1/3 restored from the project checkpoint' in b2_log
                     and 'Stage 2 restored from the checkpoint' in b2_log
                     and saw_restored('1') and saw_restored('2') and skip > 0),
        'resumed_at_frame': skip, 'telemetry_states_seen': all_states[-4:],
        'log_lines': [l.strip() for l in b2_log.splitlines() if '[StageCache]' in l or 'restored from' in l][:6]}
    inv['R3_no_detector_calls'] = {
        'pass': bool(total_detector_calls(B2) == 0 and total_detector_calls(A) > 0),
        'resumed_run_pipeline_calls': total_detector_calls(B2), 'reference_pipeline_calls': total_detector_calls(A),
        'resumed_run_raw': B2['detector_calls'], 'reference_raw': A['detector_calls'],
        'harness_ingest_calls_before_stage0': {'reference': A['calls_before_pipeline'], 'resumed': B2['calls_before_pipeline']}}
    swapped_a = set(A['swapped_frames'])
    merged = set(B1['swapped_frames']) | {f + skip for f in B2['swapped_frames']}
    # B1 swapped some frames past its last COMMITTED segment (it was stopped mid-segment): those frames are
    # re-done by B2, so compare each side only inside its own window.
    merged = {f for f in B1['swapped_frames'] if f < skip} | {f + skip for f in B2['swapped_frames']}
    inv['R4_same_swaps'] = {'pass': merged == swapped_a, 'reference': len(swapped_a), 'b1_plus_b2': len(merged),
                            'only_in_reference': sorted(swapped_a - merged)[:20], 'only_in_resumed': sorted(merged - swapped_a)[:20]}
    plan_b2 = list(getattr(P, 'plan_at_stage3', []) or [])
    expect = sorted(f - skip for f in A['plan_stage3'] if f >= skip)
    inv['R5_same_plan'] = {'pass': plan_b2 == expect, 'resumed': len(plan_b2), 'expected': len(expect),
                           'only_resumed': sorted(set(plan_b2) - set(expect))[:10], 'only_expected': sorted(set(expect) - set(plan_b2))[:10]}
    inv['R6_complete'] = {'pass': B2['output_frames'] == V.N_FRAMES and not B2['error'],
                          'frames': B2['output_frames'], 'error': B2['error']}
    if B2.get('output') and A.get('output') and os.path.exists(B2['output']) and os.path.exists(A['output']):
        report['psnr_b2_vs_reference_after_resume_point'] = sample_psnr(A['output'], B2['output'], skip)

    # ── B3 / B4: the cache must be REFUSED ──────────────────────────────────────────────────────────
    refusals = {}
    for tag, mutate in (('B3_tampered', 'tamper'), ('B4_rekeyed', 'rekey')):
        restore_state()
        if mutate == 'tamper':
            data = bytearray(open(sidecar_path, 'rb').read())
            data[len(data) // 2] ^= 0x01
            open(sidecar_path, 'wb').write(bytes(data))
        else:
            os.environ['ROOP_TEMPORAL_GAP'] = '11'                    # a real Stage 2 knob (default 10): any such change re-keys
        install_cache()
        R = rend(ctx, P, clip, tag, out_run, three)
        text = log_text(R)
        reasons = [l.strip() for l in text.splitlines() if '[StageCache] not used' in l]
        rskip = first_int(r'found (\d+) already-encoded frames', text, 0)
        swapped = {f for f in R['swapped_frames']}
        refusals[tag] = {
            'resumed_at_frame': rskip, 'seconds': R['seconds'], 'detector_calls': R['detector_calls'], 'pipeline_calls': total_detector_calls(R),
            'reasons_printed': reasons[:3], 'output_frames': R['output_frames'], 'error': R['error'],
            'same_swaps_as_reference': ({f + rskip for f in swapped} | {f for f in B1['swapped_frames'] if f < rskip}) == swapped_a}
        if mutate == 'rekey':
            os.environ.pop('ROOP_TEMPORAL_GAP', None)
    inv['R7_refuses'] = {
        'pass': all(r['reasons_printed'] and r['pipeline_calls'] > 0
                    and r['output_frames'] == V.N_FRAMES and not r['error'] and r['same_swaps_as_reference']
                    for r in refusals.values()),
        'arms': refusals}
    report['timing_seconds'] = {'reference_uninterrupted': A['seconds'], 'b1_until_stopped': B1['seconds'],
                                'b2_resumed_with_cache': B2['seconds'],
                                'b3_resumed_cache_refused': refusals.get('B3_tampered', {}).get('seconds'),
                                'b4_resumed_cache_refused': refusals.get('B4_rekeyed', {}).get('seconds')}
    report['cache'] = inv['R1_recorded']['stage2']
    report['invariants'] = inv
    bad = [{'tag': r['tag'], 'error': r['error'], 'signatures': {k: v for k, v in r['failure_signatures'].items() if v}}
           for r in (A, B1, B2) if r['error'] or any(r['failure_signatures'].values())]
    inv['0_zero_runtime_exceptions'] = {'pass': not bad, 'offending_runs': bad}
    ok = all(v['pass'] for v in inv.values())
    report['all_invariants_hold'] = ok
    report['wall_seconds'] = round(time.time() - t_start, 1)
    report['exit'] = 0 if ok else 1
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, default=str)
    print('\n== invariants ==')
    for k, v in sorted(inv.items()):
        brief = {a: b for a, b in v.items() if a not in ('arms', 'offending_runs')}
        print(('  PASS  ' if v['pass'] else '  FAIL  ') + k + '  ' + json.dumps(brief, default=str)[:420])
    print('\n== timing (s) ==  ' + json.dumps(report['timing_seconds']))
    print(f"\n[verify_stage_resume] {'ALL INVARIANTS HOLD' if ok else 'INVARIANT(S) VIOLATED'}  report: {report_path}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
