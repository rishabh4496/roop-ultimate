"""Stage 16: does the REAL pre-pass place a dropped face better with flow on?

`probe_flow_stage16.py` measures the flow module in isolation. This drives
roop's own `_precompute_temporal` (scan loop -> flow chains -> gap fill), so it
tests the WIRING: the gray ring, the anchor/step/close sequence, the stitch
merge and `_build_temporal_faces`'s bridge.

Method. One clean run is the reference: its non-interpolated faces are what the
detector really saw. Then a set of 6-10 frame dropouts is chosen where the
reference has a real face on both sides and throughout, and the pre-pass is run
twice more with the detector forced to return NOTHING on those frames -- once
with `ROOP_FLOW_COAST=0`, once with `=1`. Landmark error (NME, % of the
inter-ocular distance) at the dropped frames, against the reference.

Frames are matched to indices by a pixel fingerprint, because the detector
wrapper only sees the array. Fingerprints must be unique in the window or the
run refuses.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_flow_e2e_stage16.py --video <clip> --frames 400
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def fingerprint(fr):
    return (fr.shape, int(fr[::31, ::47].astype(np.int64).sum()),
            int(fr[7::53, 11::61].astype(np.int64).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--frames', type=int, default=400)
    ap.add_argument('--lengths', default='6,8,10')
    ap.add_argument('--every', type=int, default=29)
    ap.add_argument('--provider', default=None)
    ap.add_argument('--detail', action='store_true', help='per-frame rows for every window')
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]

    import two_face_video as tfv          # noqa: E402
    import angle_bench as ab              # noqa: E402
    tfv._apply_startup_runtime_environment()
    import yaml
    with open(os.path.join(APP, 'config.yaml')) as fh:
        cfg = yaml.safe_load(fh) or {}
    ab.init_pipeline(args.provider or (cfg.get('provider') or 'cuda'),
                     cfg.get('swap_model') or 'inswapper', 'None', 'None', sync_config=True)
    import roop.globals as G
    import roop.face_util as fu
    import roop.procmgr_tracking as pt
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions
    from roop.nvdec_reader import wrap_capture

    G.processing = True
    path = args.video
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = min(total, args.frames)
    cap = wrap_capture(cap, path, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS),
                       tag='probe')
    index = {}
    for i in range(frames):
        ok, fr = cap.read()
        if not ok:
            break
        fp = fingerprint(fr)
        if fp in index:
            raise SystemExit(f'fingerprint collision at frame {i} (static footage) - pick another clip')
        index[fp] = i
    cap.release()
    print(f'[e2e] indexed {len(index)} frames of {os.path.basename(path)}')

    drop = set()
    real_all, real_roi = fu.get_all_faces, fu.get_all_faces_in_roi

    def _all(fr, *a, **k):
        if index.get(fingerprint(fr)) in drop:
            return []
        return real_all(fr, *a, **k)

    def _roi(fr, *a, **k):
        if index.get(fingerprint(fr)) in drop:
            return []
        return real_roi(fr, *a, **k)

    def run(flow):
        pt._FLOW_COAST = flow
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
        mgr._precompute_temporal(path, None, 0, frames, frames)
        return mgr

    fu.get_all_faces, fu.get_all_faces_in_roi = real_all, real_roi
    ref = run(False)._temporal_faces
    real = {i: fs[0] for i, fs in ref.items()
            if len(fs) == 1 and not fs[0].get('_interpolated')
            and fs[0].get('landmark_2d_106') is not None and fs[0].get('kps') is not None}
    print(f'[e2e] reference: {len(real)} real single-face frames of {frames}')

    windows = []
    for L in lengths:
        k = 15
        while k + L + 12 < frames:
            if all(g in real for g in range(k - 6, k + L + 6)):
                windows.append((L, k))
                k += args.every + L
            else:
                k += 1
    drop.update(g for L, k in windows for g in range(k, k + L))
    print(f'[e2e] {len(windows)} dropout windows, {len(drop)} dropped frames')
    if not windows:
        raise SystemExit('no usable windows')

    fu.get_all_faces, fu.get_all_faces_in_roi = _all, _roi
    out = {}
    for name, flow in (('flow off', False), ('flow on', True)):
        mgr = run(flow)
        out[name] = (mgr._temporal_faces, getattr(mgr, '_flow_bridged', 0),
                     getattr(mgr, '_flow_refused', 0), dict(getattr(mgr, '_flow_stats', {})))
    fu.get_all_faces, fu.get_all_faces_in_roi = real_all, real_roi

    if args.detail:
        print('\n[e2e] per-frame NME% (flow off | flow on) and how each face was made')
        for L, k in windows:
            for g in range(k - 1, k + L + 1):
                row = []
                for name in ('flow off', 'flow on'):
                    fs = out[name][0].get(g) or []
                    if len(fs) != 1:
                        row.append(f'n={len(fs)}')
                        continue
                    f = fs[0]
                    t = np.asarray(real[g]['landmark_2d_106'], np.float32)[:, :2]
                    p = np.asarray(f['landmark_2d_106'], np.float32)[:, :2]
                    iod = float(np.linalg.norm(np.asarray(real[g].kps)[0] - np.asarray(real[g].kps)[1]))
                    tag = ('flow' if f.get('_flow_bridged') else 'coast' if f.get('_coasted')
                           else 'interp' if f.get('_interpolated') else 'real')
                    row.append(f'{np.mean(np.linalg.norm(p - t, axis=1)) / max(iod, 1e-6) * 100:7.2f} {tag:6s}')
                print(f'  L{L:2d} k={k:3d} g={g:3d} {"DROP" if g in drop else "    "}  ' + ' | '.join(row))
    print('\n   L | n  | missing: off on | NME% mean (p95)  flow off      flow on     | bridged faces')
    for L in lengths:
        ws = [(l, k) for l, k in windows if l == L]
        if not ws:
            continue
        cells = {}
        for name in out:
            tf = out[name][0]
            errs, miss = [], 0
            for _, k in ws:
                for g in range(k, k + L):
                    fs = tf.get(g) or []
                    if len(fs) != 1:
                        miss += 1
                        continue
                    t = np.asarray(real[g]['landmark_2d_106'], np.float32)[:, :2]
                    p = np.asarray(fs[0]['landmark_2d_106'], np.float32)[:, :2]
                    iod = float(np.linalg.norm(np.asarray(real[g].kps)[0] - np.asarray(real[g].kps)[1]))
                    errs.append(float(np.mean(np.linalg.norm(p - t, axis=1)) / max(iod, 1e-6) * 100))
            cells[name] = (np.mean(errs) if errs else float('nan'),
                           np.percentile(errs, 95) if errs else float('nan'), miss)
        a, b = cells['flow off'], cells['flow on']
        print(f'  {L:2d} | {len(ws):2d} |   {a[2]:4d} {b[2]:4d}      | '
              f'{a[0]:6.2f} ({a[1]:6.2f})   {b[0]:6.2f} ({b[1]:6.2f})')
    print(f'\n[e2e] flow on: {out["flow on"][1]} faces placed by flow, '
          f'{out["flow on"][2]} frames refused (missed the closing detection); '
          f'scan stats {out["flow on"][3]}')


if __name__ == '__main__':
    main()
