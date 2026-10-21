#!/usr/bin/env python3
"""Phase 14 probe: `prepass_detect_interval` (detect every K frames, optical flow between) and the
zero-phase 1-euro landmark filter, measured on a REAL pre-pass of a real clip.

Four sections.

    interval   one real pre-pass per K (the pipeline's own `_precompute_temporal`, stopped as soon as it
               returns, so no swap pass is rendered). The first run is a warm-up (TensorRT engine build);
               the dense run (K=1) after it is the reference. For every K it reports

                 time            wall seconds of the pre-pass
                 detector calls  full + ROI calls the tracker actually scheduled, and calls per frame
                 coverage        frames holding a face, against the reference
                 landmark error  on the frames the K run did NOT detect (flagged `_interpolated`), the 5-point
                                 keypoints against the reference's detection of the same frame, in % of the
                                 inter-ocular distance (IOD): median / p95 / max. Next to it, the SAME frames
                                 filled by straight linear interpolation between the K run's own detections
                                 (what `temporal_step` ships), computed offline from the reference.
                 flow share      how many of those frames were placed by optical flow (`_flow_bridged`)

               A K=1 run repeated is the null control: its error must read ~0 and its time is the noise floor.

    filter     offline arms on tracks captured by section `interval --save` (or probe_stage25_capture):
               RAW, SHIPPED (adaptive EMA), ZERO-PHASE EMA, and the vectorized 1-euro, on still jitter
               (RMS second difference of the alignment), fast-motion lag and mouth-opening retention.

    report     the markdown tables of docs/perf/phase14/README.md, rendered from the JSON the other sections wrote
               (`report --dir docs/perf/phase14`), so a number in the write-up is never typed in by hand.

    flow       the point-tracking back ends (`roop/flow_backend.py`) timed on the flow chain's own workload - a
               crop of at most ~192 px and ~70 points per call - and on larger batches, cv (CPU) against torch
               on the CPU and on the GPU, plus how far each lands from a known translation. The NVOF rung is
               reported as unavailable with its reason when this OpenCV has no CUDA.

    app\\env\\Scripts\\python.exe app\\tests\\probe_phase14_prepass.py interval --clip <clip.mp4> --ks 1,1,2,4,8 \
        --frames 900 --out docs/perf/phase14/interval_<clip>.json
    app\\env\\Scripts\\python.exe app\\tests\\probe_phase14_prepass.py filter tracks_a.npz tracks_b.npz

The app must be stopped (one render at a time; a render holds 12-15 GB).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                     # noqa: E402


class Done(BaseException):
    """Raised by the capture hook to stop the run right after the pre-pass (the pipeline's own
    `except Exception` cannot swallow a BaseException)."""


# ── section: interval ───────────────────────────────────────────────────────────────────────────
def capture_prepass(clip, source, k, frames, save=None):
    """One real pre-pass at interval `k`; returns a dict of arrays and counters."""
    import compare_enhancers_video as C
    import sample_bench as sb
    from roop.ProcessMgr import ProcessMgr
    from roop.temporal_smoother import AdaptiveLandmarkSmoother

    os.environ['ROOP_PREPASS_DETECT_INTERVAL'] = str(int(k))
    raw = {}
    orig_smooth = AdaptiveLandmarkSmoother.smooth

    def smooth(self, kps, lm, track_id=None, frame_index=None, *a, **kw):
        raw[(int(track_id), int(frame_index))] = (
            np.asarray(kps, np.float32).copy(), None if lm is None else np.asarray(lm, np.float32).copy())
        return orig_smooth(self, kps, lm, *a, track_id=track_id, frame_index=frame_index, **kw)

    result = {}
    orig_pre = ProcessMgr._precompute_temporal

    def pre(self, *a, **kw):
        self.options.stabilize_face = True          # production runs it; the bench options default it off
        t0 = time.perf_counter()
        orig_pre(self, *a, **kw)
        wall = time.perf_counter() - t0
        rows = []
        for f_idx, faces in self._temporal_faces.items():
            for face in faces:
                tid = face.get('_track_id')
                if tid is None or face.get('kps') is None:
                    continue
                r = raw.get((int(tid), int(f_idx)))
                lm68 = face.get('landmark_3d_68')
                rows.append((int(tid), int(f_idx), r[0] if r else None, r[1] if r else None,
                             np.asarray(face['kps'], np.float32),
                             None if face.get('landmark_2d_106') is None
                             else np.asarray(face['landmark_2d_106'], np.float32),
                             None if lm68 is None else np.asarray(lm68, np.float32),
                             bool(face.get('_interpolated')), bool(face.get('_flow_bridged')),
                             bool(face.get('_coasted'))))
        rows.sort(key=lambda x: (x[0], x[1]))
        nan5 = np.full((5, 2), np.nan, np.float32)
        nan106 = np.full((106, 2), np.nan, np.float32)
        nan68 = np.full((68, 3), np.nan, np.float32)
        result.update(
            wall=wall, schedule=dict(getattr(self, '_prepass_schedule', {}) or {}),
            tracker=dict(getattr(getattr(self, '_temporal_tracker', None), 'stats', {}) or {}),
            flow=dict(getattr(self, '_flow_stats', {}) or {}),
            tracks_n=len({r[0] for r in rows}),
            plan_frames=len(getattr(self, 'stage2_frame_plan', {}) or {}),
            plan_set=sorted(int(f) for f in (getattr(self, 'stage2_frame_plan', {}) or {})),
            source_map={int(k): (None if v is None or v[0] is None else int(v[0]))
                        for k, v in (getattr(self, '_track_source_map', {}) or {}).items()},
            arrays=dict(
                track=np.array([r[0] for r in rows], np.int32), frame=np.array([r[1] for r in rows], np.int32),
                kps_raw=np.stack([r[2] if r[2] is not None else nan5 for r in rows]) if rows else np.zeros((0, 5, 2)),
                lm106_raw=np.stack([r[3] if r[3] is not None else nan106 for r in rows]) if rows else np.zeros((0, 106, 2)),
                kps_cur=np.stack([r[4] for r in rows]) if rows else np.zeros((0, 5, 2)),
                lm106_cur=np.stack([r[5] if r[5] is not None else nan106 for r in rows]) if rows else np.zeros((0, 106, 2)),
                lm68=np.stack([r[6] if r[6] is not None else nan68 for r in rows]) if rows else np.zeros((0, 68, 3)),
                interp=np.array([r[7] for r in rows], bool), flow=np.array([r[8] for r in rows], bool),
                coasted=np.array([r[9] for r in rows], bool),
                cuts=np.array(sorted(int(c) for c in (getattr(self, '_shot_boundaries', None) or [])), np.int32),
                n_frames=np.int32(getattr(self, '_temporal_covered', 0))))
        raise Done()

    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    orig_run = sb.run_swap
    AdaptiveLandmarkSmoother.smooth = smooth
    ProcessMgr._precompute_temporal = pre
    sb.run_swap = lambda *a, **kw: orig_run(*a, frames=int(frames), **kw)
    try:
        C.render(clip, source, 'None', os.path.join(APP, 'output', 'phase14_probe'),
                 cfg.get('swap_model', 'realswap'), mask, int(cfg.get('max_threads', 16)),
                 overrides={'sparse_auto_range': False})
    except Done:
        pass
    finally:
        AdaptiveLandmarkSmoother.smooth = orig_smooth
        ProcessMgr._precompute_temporal = orig_pre
        sb.run_swap = orig_run
    if not result:
        raise SystemExit('the pre-pass never ran (nothing captured)')
    if save:
        np.savez(save, **result['arrays'])
    return result


def _iod(kps):
    return float(np.linalg.norm(kps[1] - kps[0]))


def _by_frame(arrays):
    """{frame: [row index]}"""
    out = {}
    for i, f in enumerate(arrays['frame'].tolist()):
        out.setdefault(f, []).append(i)
    return out


def _nme(a, b):
    return 100.0 * float(np.mean(np.linalg.norm(a - b, axis=1))) / max(_iod(b), 1e-6)


def compare(ref, run):
    """Error of `run`'s non-detected frames against `ref`'s detection of the same frame."""
    ra, rb = ref['arrays'], run['arrays']
    ref_rows = _by_frame(ra)
    errs_flow, errs_other, errs_lin = [], [], []
    detected_frames = {}                       # run track -> sorted frames it DETECTED
    for i in range(len(rb['frame'])):
        if not rb['interp'][i]:
            detected_frames.setdefault(int(rb['track'][i]), []).append(int(rb['frame'][i]))
    by_track_row = {(int(rb['track'][i]), int(rb['frame'][i])): i for i in range(len(rb['frame']))}
    matched = missing = 0
    for i in range(len(rb['frame'])):
        if not rb['interp'][i] or np.isnan(rb['kps_raw'][i]).any():
            continue
        f = int(rb['frame'][i])
        cands = [j for j in ref_rows.get(f, []) if not ra['interp'][j] and not np.isnan(ra['kps_raw'][j]).any()]
        if not cands:
            missing += 1
            continue
        mine = rb['kps_raw'][i]
        j = min(cands, key=lambda c: float(np.linalg.norm(ra['kps_raw'][c].mean(0) - mine.mean(0))))
        truth = ra['kps_raw'][j]
        if float(np.linalg.norm(truth.mean(0) - mine.mean(0))) > _iod(truth):
            missing += 1                         # a different person: not comparable
            continue
        matched += 1
        e = _nme(mine, truth)
        (errs_flow if rb['flow'][i] else errs_other).append(e)
        # the same frame by linear interpolation between this run's own neighbouring detections
        det = detected_frames.get(int(rb['track'][i]), [])
        lo = [d for d in det if d < f]
        hi = [d for d in det if d > f]
        if lo and hi:
            a, b = lo[-1], hi[0]
            ia, ib = by_track_row.get((int(rb['track'][i]), a)), by_track_row.get((int(rb['track'][i]), b))
            ta = [c for c in ref_rows.get(a, []) if not ra['interp'][c]]
            tb = [c for c in ref_rows.get(b, []) if not ra['interp'][c]]
            if ta and tb:
                ka = min(ta, key=lambda c: float(np.linalg.norm(ra['kps_raw'][c].mean(0) - rb['kps_raw'][ia].mean(0))))
                kb = min(tb, key=lambda c: float(np.linalg.norm(ra['kps_raw'][c].mean(0) - rb['kps_raw'][ib].mean(0))))
                w = (f - a) / float(b - a)
                lin = (1 - w) * ra['kps_raw'][ka] + w * ra['kps_raw'][kb]
                errs_lin.append(_nme(lin, truth))

    def pct(x):
        if not x:
            return None
        x = np.asarray(x)
        return dict(n=len(x), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()),
                    mean=float(x.mean()))
    allx = errs_flow + errs_other
    ref_frames = len(set(ra['frame'].tolist()))
    run_frames = len(set(rb['frame'].tolist()))
    return dict(matched=matched, no_reference=missing, bridged=pct(allx), by_flow=pct(errs_flow),
                not_flow=pct(errs_other), linear_baseline=pct(errs_lin),
                ref_frames=ref_frames, run_frames=run_frames,
                faces=int(len(rb['frame'])), ref_faces=int(len(ra['frame'])),
                interpolated=int(rb['interp'].sum()), flow_bridged=int(rb['flow'].sum()))


def section_interval(args):
    ks = [int(x) for x in args.ks.split(',')]
    runs = []
    for n, k in enumerate(ks):
        t0 = time.time()
        r = capture_prepass(args.clip, args.source, k, args.frames,
                            save=(args.save_prefix + '_k%d_run%d.npz' % (k, n)) if args.save_prefix else None)
        runs.append((k, r))
        s = r['schedule']
        print('[probe] run %d K=%d: pre-pass %.1f s (wall incl. load %.1f s), %s detector calls / %s frames, '
              '%d tracks, plan frames %d' % (n, k, r['wall'], time.time() - t0, s.get('detector_calls'),
                                             s.get('frames'), r['tracks_n'], r['plan_frames']), flush=True)
    # reference = the SECOND dense run when there is one (the first paid the engine build)
    dense = [r for k, r in runs if k == 1]
    ref = dense[1] if len(dense) > 1 else dense[0]
    out = dict(clip=args.clip, frames=args.frames, runs=[])
    for n, (k, r) in enumerate(runs):
        c = compare(ref, r)
        row = dict(run=n, k=k, wall_s=r['wall'], schedule=r['schedule'], tracker=r['tracker'], flow=r['flow'],
                   tracks=r['tracks_n'], plan_frames=r['plan_frames'], compare=c)
        out['runs'].append(row)
        b = c['bridged']
        print('[probe] K=%d  time %.1f s  calls/frame %.2f  frames %d/%d  faces %d/%d  bridged n=%s '
              'NME median %s p95 %s max %s  | linear median %s p95 %s | flow-placed %d' % (
                  k, r['wall'], (r['schedule'].get('detector_calls', 0) / max(1, r['schedule'].get('frames', 1))),
                  c['run_frames'], c['ref_frames'], c['faces'], c['ref_faces'],
                  b['n'] if b else 0, '%.2f' % b['median'] if b else '-', '%.2f' % b['p95'] if b else '-',
                  '%.2f' % b['max'] if b else '-',
                  '%.2f' % c['linear_baseline']['median'] if c['linear_baseline'] else '-',
                  '%.2f' % c['linear_baseline']['p95'] if c['linear_baseline'] else '-', c['flow_bridged']),
              flush=True)
    explain_plan_loss(ref, runs)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        print('[probe] wrote', args.out)
    return 0


def _ranges(frames):
    out, start, prev = [], None, None
    for f in sorted(frames):
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            out.append((start, prev))
            start = prev = f
    if start is not None:
        out.append((start, prev))
    return out


def explain_plan_loss(ref, runs):
    """Which frames held a face matched to a source in the reference and do not in a K run, and what the K run
    made of that stretch (which tracks cover it, and whether those tracks were given a source)."""
    ref_plan = set(ref['plan_set'])
    ra = ref['arrays']
    for k, r in runs:
        if k == 1 or r is ref:
            continue
        lost = sorted(ref_plan - set(r['plan_set']))
        gained = sorted(set(r['plan_set']) - ref_plan)
        print('[probe] K=%d: plan frames lost %d, gained %d (reference %d, run %d)' % (
            k, len(lost), len(gained), len(ref_plan), len(r['plan_set'])), flush=True)
        rb = r['arrays']
        for a, b in _ranges(lost)[:12]:
            ref_tr = sorted({int(t) for t, f in zip(ra['track'], ra['frame']) if a <= f <= b})
            run_rows = [(int(t), bool(i)) for t, f, i in zip(rb['track'], rb['frame'], rb['interp']) if a <= f <= b]
            run_tr = sorted({t for t, _ in run_rows})
            print('[probe]   lost frames %d-%d (%d): reference tracks %s sources %s | run tracks %s sources %s, '
                  '%d/%d of the run rows there are not detections' % (
                      a, b, b - a + 1, ref_tr, [ref['source_map'].get(t) for t in ref_tr], run_tr,
                      [r['source_map'].get(t) for t in run_tr], sum(1 for _, i in run_rows if i), len(run_rows)),
                  flush=True)


# ── section: filter ─────────────────────────────────────────────────────────────────────────────
def section_filter(args):
    import probe_stage25_filters as P
    from roop.one_euro_vec import smooth_runs
    from roop.temporal_smoother import ZeroPhaseTrackSmoother
    summary = {}
    for path in args.npz:
        d = np.load(path)
        track, frame, cuts = d['track'], d['frame'], d['cuts']
        arms = ['RAW', 'SHIPPED', 'ZERO_PHASE_EMA', 'ONE_EURO']
        res = {a: {'still': [], 'fast': []} for a in arms}
        l106 = {a: [] for a in arms}
        keep = {a: [] for a in arms if a != 'RAW'}
        zp = ZeroPhaseTrackSmoother.from_env()
        for t in sorted(set(track.tolist())):
            idx = np.nonzero(track == t)[0]
            for (a, b) in P.runs_of(frame[idx], cuts):
                sel = idx[a:b]
                key = 'has_raw' if 'has_raw' in d.files else None
                if key and not d[key][sel].all():
                    continue
                raw = d['kps_raw'][sel].astype(np.float64)
                if np.isnan(raw).any():
                    continue
                cur = d['kps_cur'][sel].astype(np.float64)
                interp = d['interp'][sel] if 'interp' in d.files else np.zeros(len(sel), bool)
                coasted = d['coasted'][sel] if 'coasted' in d.files else np.zeros(len(sel), bool)
                real = ~(interp | coasted)
                rs = P.speed(P.bidirectional(raw, P.betas(raw, 0.12)))
                lm_raw = d['lm106_raw'][sel].astype(np.float64)
                n = len(sel)
                lm_cur = d['lm106_cur'][sel].astype(np.float64)
                k_zp, (lm_zp,) = zp.smooth_run(raw, [lm_raw])
                dense = lm_raw.reshape(n, -1)
                k_eu, d_eu, _ = smooth_runs(raw, dense, [(0, n)], fps=args.fps)
                lm_eu = d_eu.reshape(lm_raw.shape)
                outs = {'RAW': (raw, lm_raw), 'SHIPPED': (cur, lm_cur), 'ZERO_PHASE_EMA': (k_zp, lm_zp),
                        'ONE_EURO': (k_eu, lm_eu)}
                rel_raw = P._rel_std(lm_raw)
                for name, (k, lm) in outs.items():
                    j = P.still_jitter(k, rs, real)
                    f = P.fast_lag(k, raw, rs, real) if name != 'RAW' else None
                    if j:
                        res[name]['still'].append(j)
                    if f:
                        res[name]['fast'].append(f)
                    l106[name].append(P._pts_jitter(lm, rs, real))
                    if name != 'RAW':
                        keep[name].append(dict(n=n, ratio=P._rel_std(lm) / max(rel_raw, 1e-9)))
        print('\n== %s' % os.path.basename(path))
        print('%-16s %8s | %8s %8s %8s %8s | %8s %10s | %9s | %6s' % (
            'arm', 'still n', 'tx px', 'ty px', 'deg', 'scale%', 'fast n', 'lag %iod', 'lm106 px', 'mouth'))
        summary[os.path.basename(path)] = {}
        for arm in arms:
            r = res[arm]
            row = dict(still_n=sum(i['n'] for i in r['still']), tx=P.wmean(r['still'], 'tx'),
                       ty=P.wmean(r['still'], 'ty'), deg=P.wmean(r['still'], 'deg'),
                       scale_pct=P.wmean(r['still'], 'scale_pct'), fast_n=sum(i['n'] for i in r['fast']),
                       lag_pct_iod=P.wmean(r['fast'], 'lag_pct_iod'),
                       rms_dev_pct_iod=P.wmean(r['fast'], 'rms_dev_pct_iod'),
                       lm106=P.wmean(l106[arm], 'rms'),
                       mouth=(P.wmean(keep[arm], 'ratio') if arm in keep else 1.0))
            summary[os.path.basename(path)][arm] = row
            print('%-16s %8d | %8.4f %8.4f %8.4f %8.4f | %8d %10.3f | %9.4f | %6.3f' % (
                arm, row['still_n'], row['tx'], row['ty'], row['deg'], row['scale_pct'], row['fast_n'],
                row['lag_pct_iod'], row['lm106'], row['mouth']))
    print('\nJSON ' + json.dumps(summary))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=1)
    return 0


# ── section: flow ───────────────────────────────────────────────────────────────────────────────
def section_flow(args):
    import cv2
    from roop import flow_backend as fb

    def texture(h, w, seed=0):
        rng = np.random.default_rng(seed)
        img = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 2.2)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    trackers = {'cv': fb.CvLK()}
    try:
        import torch
        trackers['torch-cpu'] = fb.TorchLK(device='cpu')
        if torch.cuda.is_available():
            trackers['torch-cuda'] = fb.TorchLK(device='cuda')
    except Exception as exc:
        print('[probe] torch back ends unavailable:', exc)
    reason = fb.nvof_unavailable_reason()
    print('[probe] nvof:', 'AVAILABLE (unverified here)' if reason is None else 'unavailable - ' + reason)
    out = {'nvof_unavailable_reason': reason, 'rows': []}
    shift = (3.0, -2.0)
    for side, n_pts in [(96, 40), (192, 70), (192, 300), (384, 1000), (768, 3000)]:
        img = texture(side, side)
        m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        nxt = cv2.warpAffine(img, m, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rng = np.random.default_rng(1)
        pts = rng.uniform(0.25 * side, 0.75 * side, (n_pts, 1, 2)).astype(np.float32)
        truth = pts + np.float32(shift)
        for name, tr in trackers.items():
            for _ in range(3):                                  # warm-up (context, kernels)
                tr.track(img, nxt, pts)
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                new, st, _err = tr.track(img, nxt, pts)
                times.append((time.perf_counter() - t0) * 1000.0)
            ok = st.reshape(-1) == 1
            miss = np.linalg.norm((new - truth).reshape(-1, 2)[ok], axis=1) if ok.any() else np.array([np.nan])
            row = dict(crop=side, points=n_pts, backend=name, ms_median=float(np.median(times)),
                       ms_p95=float(np.percentile(times, 95)), tracked=float(ok.mean()),
                       miss_median_px=float(np.median(miss)), miss_p95_px=float(np.percentile(miss, 95)))
            out['rows'].append(row)
            print('[probe] crop %4d  points %4d  %-10s %8.3f ms (p95 %8.3f)  tracked %5.1f%%  miss median %.3f p95 %.3f px'
                  % (side, n_pts, name, row['ms_median'], row['ms_p95'], 100 * row['tracked'],
                     row['miss_median_px'], row['miss_p95_px']), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1)
        print('[probe] wrote', args.out)
    return 0


# ── section: report ─────────────────────────────────────────────────────────────────────────────
def section_report(args):
    import glob
    lines = []
    for path in sorted(glob.glob(os.path.join(args.dir, 'interval_final_*.json'))):
        d = json.load(open(path, encoding='utf-8'))
        clip = os.path.basename(path)[len('interval_final_'):-len('.json')]
        runs = d['runs']
        dense = [r for r in runs if r['k'] == 1]
        ref = dense[1] if len(dense) > 1 else dense[0]
        lines += ['', '**%s** (%d frames, reference = the second dense run; the two dense runs are bit-identical)' % (clip, d['frames']), '',
                  '| K | pre-pass | detector calls / frame | faces found | frames with a face | frames matched to a source | '
                  'flow-placed error median / p95 (% eye dist.) | linear interpolation median | placed without flow: n, median |',
                  '|---|---|---|---|---|---|---|---|---|']
        for r in runs[1:] if len(dense) > 1 else runs:
            c, s_ = r['compare'], r['schedule']
            # A dense (K = 1) run has no bridged frames to compare: what the comparison sees there is one frame
            # that was not a detection in the reference either, which says nothing.
            bf, nf, lb = ((None, None, None) if r['k'] == 1
                          else (c['by_flow'], c['not_flow'], c['linear_baseline']))
            pct = lambda a, b: '%+.0f%%' % (100.0 * (a - b) / b) if b else '-'
            lines.append('| %d | %.1f s (%s) | %.2f | %d (%s) | %d / %d | %d (%s) | %s | %s | %s |' % (
                r['k'], r['wall_s'], pct(r['wall_s'], ref['wall_s']), s_['detector_calls'] / max(1, s_['frames']),
                c['faces'], pct(c['faces'], c['ref_faces']), c['run_frames'], c['ref_frames'], r['plan_frames'],
                pct(r['plan_frames'], ref['plan_frames']),
                ('%.2f / %.1f' % (bf['median'], bf['p95'])) if bf else '-', ('%.2f' % lb['median']) if lb else '-',
                ('%d, %.1f' % (nf['n'], nf['median'])) if nf else '-'))
    path = os.path.join(args.dir, 'flow_backends.json')
    if os.path.isfile(path):
        d = json.load(open(path, encoding='utf-8'))
        lines += ['', '**Point-tracking back ends** (ms per call, median of %s; NVOF: %s)' % (
            'the probe repetitions', d.get('nvof_unavailable_reason') or 'available'), '',
            '| crop px | points | cv (CPU) | torch (CPU) | torch (CUDA) |', '|---|---|---|---|---|']
        by = {}
        for row in d['rows']:
            by.setdefault((row['crop'], row['points']), {})[row['backend']] = row['ms_median']
        for (crop, pts), v in sorted(by.items()):
            lines.append('| %d | %d | %.2f | %s | %s |' % (crop, pts, v.get('cv', float('nan')),
                         ('%.1f' % v['torch-cpu']) if 'torch-cpu' in v else '-', ('%.1f' % v['torch-cuda']) if 'torch-cuda' in v else '-'))
    text = chr(10).join(lines)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(text + chr(10))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='section', required=True)
    a = sub.add_parser('interval')
    a.add_argument('--clip', required=True)
    a.add_argument('--source', default='akansha')
    a.add_argument('--ks', default='1,1,2,4,8')
    a.add_argument('--frames', type=int, default=900)
    a.add_argument('--out', default=None)
    a.add_argument('--save-prefix', default=None, help='also save every run as <prefix>_k<K>_run<n>.npz')
    f = sub.add_parser('filter')
    f.add_argument('npz', nargs='+')
    f.add_argument('--fps', type=float, default=24.0)
    f.add_argument('--out', default=None)
    r = sub.add_parser('report')
    r.add_argument('--dir', default=os.path.join(os.path.dirname(APP), 'docs', 'perf', 'phase14'))
    r.add_argument('--out', default=None)
    w = sub.add_parser('flow')
    w.add_argument('--reps', type=int, default=40)
    w.add_argument('--out', default=None)
    args = ap.parse_args()
    # The pipeline changes its working directory while it loads, so every path is made absolute NOW.
    for name in ('clip', 'out', 'save_prefix', 'dir'):
        if getattr(args, name, None):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    if args.section == 'filter':
        args.npz = [os.path.abspath(p) for p in args.npz]
    return {'interval': section_interval, 'filter': section_filter, 'flow': section_flow,
            'report': section_report}[args.section](args)


if __name__ == '__main__':
    sys.exit(main())
