"""Stage 17: does the REAL pre-pass keep the swap crop steady behind an occluder?

Builds two clips from one source window -- a reference with nothing pasted, and a
copy with a skin-coloured occluder that slides over an eye / the mouth / half the
face inside known windows -- through the same lossless-ish codec, then runs roop's
own `_precompute_temporal` on each. The reference's faces are the truth; the
occluded clip is run with flow alignment OFF (`_FLOW_ALIGN=False`, what shipped
before) and ON.

Reported, in 256-px-crop pixels (alignment error = mean distance between where the
arm's `estimate_norm` matrix and the reference's put the reference's 106
landmarks):
  inside the windows   -- what the feature is for
  outside the windows  -- what it must not touch (regression guard)

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_align_e2e_stage17.py --video <clip> --frames 300
"""

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from probe_alignment_stage17 import occluder, paste, CROP        # noqa: E402
from probe_alignment_chain_stage17 import shifted                # noqa: E402


def writer(path, size, fps):
    for fourcc in ('FFV1', 'MJPG'):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            if fourcc == 'MJPG':
                try:
                    w.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
                except Exception:
                    pass
            return w, fourcc
    raise SystemExit('no usable video codec')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--lengths', default='8,12')
    ap.add_argument('--every', type=int, default=24)
    ap.add_argument('--max-side', type=int, default=960)
    ap.add_argument('--provider', default=None)
    ap.add_argument('--workdir', default=None)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]

    import two_face_video as tfv
    import angle_bench as ab
    tfv._apply_startup_runtime_environment()
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
    ab.init_pipeline(args.provider or cfg.get('provider') or 'cuda',
                     cfg.get('swap_model') or 'inswapper', 'None', 'None', sync_config=True)
    import roop.globals as G
    import roop.procmgr_tracking as pt
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions
    from roop.face_util import estimate_norm, get_all_faces
    from roop.visibility_alignment import apply

    G.processing = True
    work = args.workdir or tempfile.mkdtemp(prefix='s17_')
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frames = []
    for _ in range(args.frames):
        ok, fr = cap.read()
        if not ok:
            break
        s = args.max_side / float(max(fr.shape[:2]))
        if s < 1.0:
            fr = cv2.resize(fr, (int(fr.shape[1] * s), int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    size = (frames[0].shape[1], frames[0].shape[0])

    # Per-frame faces on the CLEAN frames decide where windows can go and give the
    # occluder its geometry; the pre-pass truth is taken from its own reference run.
    det = []
    for fr in frames:
        fs = get_all_faces(fr) or []
        good = [f for f in fs if getattr(f, 'landmark_2d_106', None) is not None]
        det.append(good[0] if len(fs) == 1 and len(good) == 1 else None)

    windows = []
    kinds = ('eye', 'mouth', 'half')
    n = 0
    for L in lengths:
        k = 12
        while k + L + 8 < len(frames):
            if all(det[g] is not None for g in range(k - 4, k + L + 4)):
                windows.append((L, k, kinds[n % 3]))
                n += 1
                k += args.every + L
            else:
                k += 1
    print(f'[e2e17] {os.path.basename(args.video)}: {len(frames)} frames, {len(windows)} occlusion windows')
    if not windows:
        raise SystemExit('no usable windows')

    rng = np.random.default_rng(11)
    in_window = set()
    ref_path, occ_path = os.path.join(work, 'ref.avi'), os.path.join(work, 'occ.avi')
    wr, codec = writer(ref_path, size, fps)
    wo, _ = writer(occ_path, size, fps)
    win_at = {}
    for L, k, kind in windows:
        for g in range(k, k + L):
            win_at[g] = (L, k, kind)
            in_window.add(g)
    for i, fr in enumerate(frames):
        wr.write(fr)
        if i in win_at and det[i] is not None:
            L, k, kind = win_at[i]
            final = occluder(fr.shape, det[k + L // 2] if det[k + L // 2] is not None else det[i], kind, rng)
            ramp = min(1.0, (i - k + 1) / 3.0) * min(1.0, (k + L - i) / 3.0)
            if kind == 'half':
                m = final * ramp
            else:
                iod = float(np.linalg.norm(np.asarray(det[i].kps)[0] - np.asarray(det[i].kps)[1]))
                m = shifted(final, int(round((1.0 - ramp) * 0.9 * iod))) if ramp < 1.0 else final
            fr = paste(fr, det[i], np.clip(m, 0, 1), rng)
        wo.write(fr)
    wr.release()
    wo.release()
    print(f'[e2e17] wrote {ref_path} and {occ_path} ({codec})')

    def run(path, align):
        pt._FLOW_ALIGN = align
        mgr = ProcessMgr(None)
        opts = ProcessOptions.__new__(ProcessOptions)
        for name, value in (('stabilize_face', False), ('stabilize_method', 'one_euro'),
                            ('stabilize_min_cutoff', 0.1), ('stabilize_beta', 0.1),
                            ('stabilize_landmarks', False), ('swap_mode', 'selected'),
                            ('selected_index', 0), ('face_distance_threshold', 0.75)):
            setattr(opts, name, value)
        mgr.options = opts
        mgr.target_face_datas, mgr.target_face_groups = [], []
        mgr.progress_gradio = None
        import time
        t0 = time.time()
        mgr._precompute_temporal(path, None, 0, len(frames), len(frames))
        return mgr, time.time() - t0

    ref_mgr, _ = run(ref_path, False)
    truth = {i: fs[0] for i, fs in ref_mgr._temporal_faces.items()
             if len(fs) == 1 and not fs[0].get('_interpolated')
             and fs[0].get('landmark_2d_106') is not None and fs[0].get('kps') is not None}
    print(f'[e2e17] reference: {len(truth)} real single-face frames')

    out = {}
    for name, align in (('align off', False), ('align on', True)):
        mgr, secs = run(occ_path, align)
        out[name] = (mgr._temporal_faces, secs, dict(getattr(mgr, '_flow_stats', {})))

    def err(tf, g):
        fs = tf.get(g) or []
        if len(fs) != 1 or g not in truth:
            return None
        t = truth[g]
        M_ref = estimate_norm(np.asarray(t.kps, np.float32), CROP, 'arcface')
        M = estimate_norm(np.asarray(fs[0].kps, np.float32), CROP, 'arcface')
        lm = np.asarray(t.landmark_2d_106, np.float32)[:, :2]
        return float(np.linalg.norm(apply(M, lm) - apply(M_ref, lm), axis=1).mean())

    print('\n   region          | n frames |  align off  mean (p95) |  align on  mean (p95) | replaced by flow')
    for label, sel in (('inside windows', lambda g: g in in_window),
                       ('outside windows', lambda g: g not in in_window)):
        gs = [g for g in sorted(truth) if sel(g)]
        cells = []
        for name in ('align off', 'align on'):
            e = [x for x in (err(out[name][0], g) for g in gs) if x is not None]
            cells.append((np.mean(e) if e else float('nan'), np.percentile(e, 95) if e else float('nan'), len(e)))
        rep = sum(1 for g in gs for f in (out['align on'][0].get(g) or []) if f.get('_flow_aligned'))
        print(f'   {label:15s} | {cells[0][2]:8d} | {cells[0][0]:8.2f} ({cells[0][1]:6.2f})  | '
              f'{cells[1][0]:8.2f} ({cells[1][1]:6.2f})  | {rep}')
    for kind in kinds:
        gs = [g for g in sorted(truth) if g in win_at and win_at[g][2] == kind]
        cells = []
        for name in ('align off', 'align on'):
            e = [x for x in (err(out[name][0], g) for g in gs) if x is not None]
            cells.append((np.mean(e) if e else float('nan'), len(e)))
        print(f'   window {kind:8s} | {cells[0][1]:8d} | {cells[0][0]:8.2f}              | {cells[1][0]:8.2f}')
    print(f'\n[e2e17] pre-pass seconds  off {out["align off"][1]:.1f}  on {out["align on"][1]:.1f}   '
          f'scan stats (on) {out["align on"][2]}')


if __name__ == '__main__':
    main()
