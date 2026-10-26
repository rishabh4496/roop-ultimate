#!/usr/bin/env python3
"""Phase 9 end-to-end: Stage 4 finalization and the per-job report, on real renders of the Phase 5 clip.

Reuses `verify_three_stage_pipeline.py`'s ground-truth clip (600 frames, known empty / distractor / target
regions) and its production render path (`core.batch_process_with_options`, the user's live config.yaml
models, `sparse_auto_range` ON), with two additions the brief needs and the Phase 5 clip lacks:

  * the audio is deterministic WHITE NOISE (the Phase 5 clip carries a 440 Hz sine, which is periodic and so
    cannot show a lip-sync offset), which lets the delivered file's audio be cross-correlated against the
    SAME seconds of the source;
  * a trimmed render (frames 150-510), because trimmed audio is where the offset defects lived.

After each render this does what `api._run_swap` does at the end of a job (`stage_telemetry.set_finalize` for
every Stage 4 result, then `end_run`) and reads the report back.

INVARIANTS (exit 0 only when all hold):

  F1  Stage 4 ok: `roop.globals.last_finalize` holds a result with ok=True, the intermediate deleted, no orphans
      (no `__temp`, `.seg`, `__audio_cut`, `.resume.json` file left in the output folder)
  F2  the delivered file has exactly the frames of the SOURCE'S OWN packet count (ffprobe -count_packets of
      the clip as built here: independent of the pipeline's belief and of the header count Stage 4 read), and a
      non-zero duration. (The VFR clip is written with `-fps_mode vfr`, which DROPS frames: it holds 591 of the
      600 generated, so the generator's 600 is never used as ground truth.)
  F3  lip-sync: the delivered audio, cross-correlated with the source's same seconds, is within ONE FRAME
  F4  colour tags on the delivered file equal the intermediate's, and are the writer's bt709
  R1  passthrough_frames + gpu_frames == the trim's frame count, and passthrough_frames == the frames of the
      trim outside the Stage 1 active ranges (the counters are exact, not estimates)
  R2  Stage 1/2/3 wall clocks are measured, in order, and their sum is within the render's own wall clock
  R3  peak VRAM sampled (> 0, <= device total); Stage 1 worker utilization in (0, 100]
  R4  cap.set / cap.grab counts equal the `[Track] positioned ... N seek(s) ... over W frame(s)` lines the
      seekers print themselves
  R5  the report is JSON-safe and equals what `/api/runtime/state` would carry (PERFORMANCE.pipeline_report)
  R6  Stage 3's `frames_retired` (frames the output writer accepted) equals the delivered file's own packet count,
      `effective_output_fps` == retired / Stage 3 wall, the swap-inference counter is non-zero when faces were
      swapped, and the closing `[Stage 3] retired N/M frames | X Effective Output FPS` line is in the render log

INFORMATIONAL: the speedup estimate against the measured dense (1-stage) arms, ABBA-counterbalanced after a
warm-up, with the estimate's error reported (nothing gates on it: its tolerance has no distribution yet).

    app/env/Scripts/python.exe app/tests/verify_stage4_report.py [--workdir DIR] [--out report.json]
        [--skip-benchmark] [--skip-vfr]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                               # noqa: E402

import fixtures                                  # noqa: E402
import verify_three_stage_pipeline as V          # noqa: E402
from roop import finalize_output as fo           # noqa: E402
from roop import stage_telemetry as st           # noqa: E402

FPS, N = V.FPS, V.N_FRAMES
TRIM = (150, 510)
SR = 48000
ORPHAN_MARKERS = ('__temp', '.seg', '__audio_cut', '.resume.json')


def _noise_audio_clip(video_only: str, out: str, seconds: float) -> str:
    subprocess.run([V._ffmpeg(), '-y', '-v', 'error', '-i', video_only, '-f', 'lavfi', '-i',
                    f'anoisesrc=d={seconds:.3f}:c=white:r={SR}:a=0.5:seed=11', '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', f'{seconds:.3f}', out], check=True)
    return out


def _pcm(path, start, seconds):
    out = subprocess.run([V._ffmpeg(), '-v', 'error', '-ss', f'{start:.6f}', '-i', path, '-t', f'{seconds:.6f}',
                          '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True,
                         check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _lag_s(ref, got, max_s=0.25):
    n = min(len(ref), len(got))
    ref, got = ref[:n].astype(np.float64), got[:n].astype(np.float64)
    size = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(got, size) * np.conj(np.fft.rfft(ref, size)), size)
    w = int(max_s * SR)
    lags = np.concatenate([np.arange(0, w), np.arange(-w, 0)])
    vals = np.concatenate([corr[:w], corr[size - w:]])
    return float(lags[int(np.argmax(vals))]) / SR


def _leftovers(folder: str) -> list:
    return sorted(n for n in os.listdir(folder) if any(m in n for m in ORPHAN_MARKERS))


def _seek_lines(log_text: str) -> dict:
    seeks = walked = 0
    for m in re.finditer(r'\[Track\] positioned \d+ active range\(s\): (\d+) seek\(s\), \d+ walk\(s\) over (\d+) frame\(s\)',
                         log_text):
        seeks += int(m.group(1))
        walked += int(m.group(2))
    return {'seeks': seeks, 'walked': walked}


def run_one(ctx, P, clip, tag, out_root, overrides, rng, source_for_audio, mp3=False) -> dict:
    """One render + the end-of-job calls api._run_swap makes; returns the measurements."""
    import roop.globals as g
    st.reset()                                           # api._run_swap does this at job start
    g.last_finalize = []
    t0 = time.time()
    r = V.render(ctx, P, clip, tag, out_root, overrides, rng=rng)
    wall = time.time() - t0
    fins = list(getattr(g, 'last_finalize', None) or [])
    for f in fins:
        st.set_finalize(f)
    rep = st.end_run(ok=r['error'] is None)
    log_text = open(r['log'], encoding='utf-8', errors='replace').read()
    trim_start, trim_end = (rng if rng != (0, 0) else (0, V.count_frames(clip)))      # the SOURCE's own packets
    expected = trim_end - trim_start
    out = r['output']
    res = {'tag': tag, 'overrides': overrides, 'range': [trim_start, trim_end], 'expected_frames': expected,
           'render_seconds': round(wall, 2), 'error': r['error'], 'output': out, 'report': rep,
           'finalize_results': fins, 'leftovers': _leftovers(out_root), 'log': r['log']}
    if out and os.path.isfile(out):
        res['output_packets'] = V.count_frames(out)
        probe = fo.probe_container(out)
        res['output_color'] = fo.color_tags(probe)
        res['output_video_duration_s'] = (probe or {}).get('video', {}).get('duration_s')
        if source_for_audio:
            ref = _pcm(source_for_audio, trim_start / FPS, 1.5)
            got = _pcm(out, 0.0, 1.5)
            res['audio_lag_ms'] = round(_lag_s(ref, got) * 1000.0, 2)
    res['seek_lines'] = _seek_lines(log_text)
    return res


def check_run(res: dict, strict_audio: bool = True) -> dict:
    """The F/R invariants for one 3-stage render. Returns {name: {ok, detail}}."""
    rep = res['report']
    s1, s2, s3, s4 = (rep['stages'][k] for k in (st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS,
                                                 st.STAGE_3_MAIN_SWAP, st.STAGE_4_FINALIZE))
    exp = res['expected_frames']
    fin = (res['finalize_results'] or [{}])[-1]
    frame_s = 1000.0 / FPS
    inv = {}
    inv['F1_stage4_ok_no_orphans'] = {
        'ok': bool(fin.get('ok')) and bool(fin.get('temp_deleted')) and not fin.get('orphans') and not res['leftovers'],
        'detail': {'ok': fin.get('ok'), 'temp_deleted': fin.get('temp_deleted'), 'orphans': fin.get('orphans'),
                   'leftovers': res['leftovers'], 'problems': fin.get('problems')}}
    pk = res.get('output_packets')
    inv['F2_frames_and_duration'] = {
        'ok': pk == exp and bool(res.get('output_video_duration_s')),
        'detail': {'packets': pk, 'expected_by_the_source': exp, 'duration_s': res.get('output_video_duration_s')}}
    lag = res.get('audio_lag_ms')
    inv['F3_lipsync_under_one_frame'] = {'ok': lag is not None and abs(lag) < frame_s,
                                         'detail': {'audio_lag_ms': lag, 'one_frame_ms': round(frame_s, 2)}}
    ref_tags = (fin.get('verify') or {}).get('color')
    inv['F4_colour_tags'] = {
        'ok': res.get('output_color') == ref_tags and res.get('output_color', {}).get('color_primaries') == 'bt709'
        and bool((fin.get('verify') or {}).get('color_preserved')),
        'detail': {'delivered': res.get('output_color'), 'stage4_saw': ref_tags,
                   'preserved': (fin.get('verify') or {}).get('color_preserved')}}
    # R1: exact counters
    rng_lo, rng_hi = res['range']
    active = (res.get('active_mask_in_trim'))
    pt, gpu = s3.get('passthrough_frames'), s3.get('gpu_frames')
    ran = s3.get('frames')      # the frames the PIPELINE ran (its own belief); F2 covers a belief that was short
    inv['R1_frame_accounting'] = {
        'ok': pt is not None and gpu is not None and ran is not None and pt + gpu == ran
        and (active is None or pt == ran - active),
        'detail': {'passthrough': pt, 'gpu': gpu, 'sum': None if pt is None or gpu is None else pt + gpu,
                   'frames_the_pipeline_ran': ran, 'active_in_trim_by_stage1': active}}
    w1, w2, w3 = s1.get('wall_s'), s2.get('wall_s'), s3.get('wall_s')
    walls_ok = all(w is not None and w > 0 for w in (w1, w2, w3)) and (w1 + w2 + w3) <= res['render_seconds'] + 0.5
    inv['R2_wall_clocks'] = {'ok': bool(walls_ok), 'detail': {'s1': w1, 's2': w2, 's3': w3,
                                                               'sum': None if None in (w1, w2, w3) else round(w1 + w2 + w3, 2),
                                                               'render_seconds': res['render_seconds'],
                                                               'stage4_s': s4.get('wall_s') if s4 else None}}
    vr = rep['vram']
    util = s1.get('worker_utilization_pct')
    inv['R3_vram_and_worker_utilization'] = {
        'ok': bool(vr.get('peak_gb')) and vr['peak_gb'] <= (vr.get('total_gb') or 1e9) and util is not None
        and 0 < util <= 100,
        'detail': {'vram': vr, 'stage3_peak_gb': s3.get('peak_vram_gb'), 'worker_utilization_pct': util,
                   'workers': s1.get('workers'), 'sampled_frames': s1.get('sampled_frames')}}
    # R6: Stage 3's fps divides frames the WRITER accepted. Ground truth is the delivered file's own packet count.
    ret, eff, w3 = s3.get('frames_retired'), s3.get('effective_output_fps'), s3.get('wall_s')
    neural, skipped = s3.get('neural_inferences'), s3.get('smart_skip_faces')
    log_text = open(res['log'], encoding='utf-8', errors='replace').read()
    line = re.search(r'\[Stage 3\] retired (\d+)/(\d+) frames \| ([\d.]+) Effective Output FPS', log_text)
    inv['R6_retired_frames_and_rates'] = {
        'ok': ret is not None and ret == pk == exp and w3 and eff == round(ret / w3, 2)
        and (not s3.get('faces_swapped') or (neural or 0) + (skipped or 0) > 0)
        and line is not None and int(line.group(1)) == ret and float(line.group(3)) == eff
        and rep.get('effective_fps_basis', '').startswith('retired'),
        'detail': {'frames_retired': ret, 'delivered_packets': pk, 'expected': exp, 'effective_output_fps': eff,
                   'stage3_wall_s': w3, 'neural_inferences': neural, 'smart_skip_faces': skipped,
                   'smart_skip_ratio_pct': s3.get('smart_skip_ratio_pct'), 'faces_swapped': s3.get('faces_swapped'),
                   'terminal_line': line.group(0) if line else None, 'run_effective_fps': rep.get('effective_fps')}}
    sl = res['seek_lines']
    inv['R4_seek_counts_match_the_seekers'] = {
        'ok': s2.get('set_calls') == sl['seeks'] and s2.get('grab_calls') == sl['walked'],
        'detail': {'report': {'set_calls': s2.get('set_calls'), 'grab_calls': s2.get('grab_calls')}, 'seeker_log': sl}}
    try:
        json.dumps(rep)
        json_ok = True
    except TypeError:
        json_ok = False
    from roop import runtime_state
    perf = runtime_state.snapshot()['sections']['PERFORMANCE']['values']['pipeline_report']
    inv['R5_json_and_runtime_state'] = {'ok': json_ok and perf == rep, 'detail': {'json_safe': json_ok,
                                                                                   'runtime_state_equal': perf == rep}}
    return inv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--enhancer', default=None)
    ap.add_argument('--angle-clip', default=None)
    ap.add_argument('--skip-benchmark', action='store_true')
    ap.add_argument('--skip-vfr', action='store_true')
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir or tempfile.mkdtemp(prefix='verify_stage4_'))
    os.makedirs(workdir, exist_ok=True)
    report_path = os.path.abspath(args.out or os.path.join(workdir, 'verify_stage4_report.json'))
    angle = args.angle_clip or fixtures.clip('angle.mp4')
    if not os.path.isfile(angle):
        print(f'[verify_stage4] PENDING: real profile footage not found ({angle}); not substituting another clip')
        return 3
    report = {'script': 'app/tests/verify_stage4_report.py', 'workdir': workdir,
              'git_head': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(), 'trim': list(TRIM)}

    frames, truth, sprites = V.build_scene(angle)
    seconds = N / FPS
    cfr_v = V.write_clip(frames, os.path.join(workdir, 'clip_cfr_video.mp4'), vfr=False, audio=False)
    cfr = _noise_audio_clip(cfr_v, os.path.join(workdir, 'clip_cfr.mp4'), seconds)
    clips = {'cfr': cfr}
    if not args.skip_vfr:
        vfr_v = V.write_clip(frames, os.path.join(workdir, 'clip_vfr_video.mp4'), vfr=True, audio=False)
        clips['vfr'] = _noise_audio_clip(vfr_v, os.path.join(workdir, 'clip_vfr.mp4'), seconds)

    P = V.Probe()
    ctx = V.prepare_pipeline(args, workdir, frames, sprites)
    V.install_hooks(P)
    report['models'] = ctx.config_view
    out_root = os.path.join(workdir, 'out')
    three = {'sparse_auto_range': True}
    one = {'sparse_auto_range': False, 'temporal_detection': False, 'track_identities': False}

    # warm-up of BOTH paths (engine builds and first loads must not land in a timed arm)
    report['warmup'] = {}
    for tag, ov in (('warmup_3stage', three), ('warmup_1stage', one)):
        w = V.render(ctx, P, cfr, tag, out_root, ov, rng=(0, 90))
        report['warmup'][tag] = {'seconds': w['seconds'], 'error': w['error']}
        print(f"[verify_stage4] {tag}: {w['seconds']} s", flush=True)

    runs = report['runs'] = []
    invariants = report['invariants'] = {}

    def do(tag, clip, ov, rng):
        res = run_one(ctx, P, clip, tag, out_root, ov, rng, clip)
        # Stage 1's ranges are INCLUSIVE and relative to the trim (0 = its first frame), so the frames the
        # active-range gate lets through are their lengths summed; every other frame of the trim must pass through.
        act = sum(b - a + 1 for a, b in (P.active_ranges or [])) if ov.get('sparse_auto_range') else None
        res['active_mask_in_trim'] = act
        runs.append(res)
        print(f"[verify_stage4] {tag}: {res['render_seconds']} s  finalize ok={((res['finalize_results'] or [{}])[-1]).get('ok')}"
              f"  frames {res.get('output_packets')}/{res['expected_frames']}  audio lag {res.get('audio_lag_ms')} ms", flush=True)
        return res

    primary = do('cfr_3stage_full', cfr, three, (0, 0))
    invariants['cfr_full'] = check_run(primary)
    trimmed = do('cfr_3stage_trim', cfr, three, TRIM)
    invariants['cfr_trim'] = check_run(trimmed)
    if not args.skip_vfr:
        vfr = do('vfr_3stage_full', clips['vfr'], three, (0, 0))
        invariants['vfr_full'] = check_run(vfr)

    if not args.skip_benchmark:
        dense_a = do('cfr_1stage_A', cfr, one, (0, 0))
        dense_b = do('cfr_1stage_B', cfr, one, (0, 0))
        three_b = do('cfr_3stage_B', cfr, three, (0, 0))
        def pipeline_s(run):
            stg = run['report']['stages']
            return sum((stg[k]['wall_s'] or 0.0) for k in (st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS,
                                                           st.STAGE_3_MAIN_SWAP)) + ((stg[st.STAGE_4_FINALIZE] or {}).get('wall_s') or 0.0)
        d_pipe = [pipeline_s(dense_a), pipeline_s(dense_b)]
        t_pipe = [pipeline_s(primary), pipeline_s(three_b)]
        d_wall = [dense_a['render_seconds'], dense_b['render_seconds']]
        t_wall = [primary['render_seconds'], three_b['render_seconds']]
        mean = lambda xs: sum(xs) / len(xs)
        est = [primary['report']['speedup']['multiplier'], three_b['report']['speedup']['multiplier']]
        per_arm = [round(d / mean(t_pipe), 2) for d in d_pipe]
        report['speedup'] = {
            'estimates': est,
            'measured_pipeline_only': {'dense_s': [round(x, 1) for x in d_pipe], 'three_stage_s': [round(x, 1) for x in t_pipe],
                                       'ratio_of_means': round(mean(d_pipe) / mean(t_pipe), 2),
                                       'ratio_per_dense_arm': per_arm},
            'measured_end_to_end_render_call': {'dense_s': d_wall, 'three_stage_s': t_wall,
                                                'ratio_of_means': round(mean(d_wall) / mean(t_wall), 2),
                                                'fixed_overhead_s_per_render': [round(w - p, 1) for w, p in
                                                                                zip(d_wall + t_wall, d_pipe + t_pipe)]},
            'estimate_within_dense_arm_range': (min(per_arm) <= min(e for e in est if e) and
                                                max(e for e in est if e) <= max(per_arm)),
            'note': 'the estimate is defined on STAGE walls (Stage 1+2+3+4), so the like-for-like measurement is '
                    '`measured_pipeline_only`; `render()` also carries ~30 s of model load / init that precedes the '
                    'first stage and is in no stage wall. Two dense arms differ by up to ~25% (first-arm effects), so '
                    'the estimate is compared with the RANGE they span. Informational, not gated.'}
        print('[verify_stage4] speedup ' + json.dumps(report['speedup']), flush=True)

    failed = []
    for arm, inv in invariants.items():
        for name, v in inv.items():
            tag = 'PASS' if v['ok'] else 'FAIL'
            print(f'[verify_stage4] {tag} {arm}:{name} {json.dumps(v["detail"])}', flush=True)
            if not v['ok']:
                failed.append(f'{arm}:{name}')
    report['failed'] = failed
    report['exit'] = 1 if failed else 0
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f'[verify_stage4] report: {report_path}  exit {report["exit"]}', flush=True)
    return report['exit']


if __name__ == '__main__':
    sys.exit(main())
