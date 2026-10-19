"""Stage 16: does the bystander veto ever fire on real footage, and what does it cost?

`test_bystander_veto.py` proves the rule on synthetic embeddings. This asks the
only question that matters for shipping it: on real multi-person clips, with ONE
person captured as the target (so everybody else is a bystander), how many
(track, person) bindings does the veto prevent, and does it take any frames away
from the target?

Both arms run roop's own `_precompute_temporal`, differing only in
`procmgr_tracking._BYSTANDER_VETO`. Reported per arm: tracks bound to the target,
frames those tracks cover, and the veto's own count. If the veto never fires and
the two arms bind identical frames, it is neutral on that clip; if it removes
target frames, that is a regression and this is where it shows.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_bystander_veto_stage16.py --video <clip> [--frames N]
"""

import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--frames', type=int, default=0, help='0 = whole clip')
    ap.add_argument('--provider', default=None)
    args = ap.parse_args()

    import two_face_video as tfv          # noqa: E402
    import angle_bench as ab              # noqa: E402
    tfv._apply_startup_runtime_environment()
    import yaml
    with open(os.path.join(APP, 'config.yaml')) as fh:
        cfg = yaml.safe_load(fh) or {}
    ab.init_pipeline(args.provider or (cfg.get('provider') or 'cuda'),
                     cfg.get('swap_model') or 'inswapper', 'None', 'None', sync_config=True)
    import roop.globals as G
    import roop.procmgr_tracking as pt
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions

    G.processing = True
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    frames = min(total, args.frames) if args.frames else total

    targets, groups = tfv.auto_capture_targets(args.video, expect=1)
    print(f'[veto] {os.path.basename(args.video)}: {frames} frames, '
          f'{len(targets)} captured angle(s), {len(set(groups))} target person(s)')
    G.TARGET_FACES = list(targets)
    G.TARGET_FACE_GROUP = list(groups)

    def run(veto):
        pt._BYSTANDER_VETO = veto
        mgr = ProcessMgr(None)
        opts = ProcessOptions.__new__(ProcessOptions)
        for name, value in (('stabilize_face', False), ('stabilize_method', 'one_euro'),
                            ('stabilize_min_cutoff', 0.1), ('stabilize_beta', 0.1),
                            ('stabilize_landmarks', False), ('swap_mode', 'selected'),
                            ('selected_index', 0),
                            ('face_distance_threshold', float(cfg.get('face_distance_threshold', 0.75)))):
            setattr(opts, name, value)
        mgr.options = opts
        mgr.target_face_datas = list(targets)
        mgr.target_face_groups = list(groups)
        mgr.progress_gradio = None
        mgr._precompute_temporal(args.video, None, 0, frames, frames)
        bound_tracks = sum(1 for v in (mgr._track_source_map or {}).values()
                           if v[0] is not None)
        bound_frames = sum(1 for lst in (mgr._track_assignments or {}).values()
                           for (_c, src, _e) in lst if src is not None)
        return (len(mgr._track_source_map or {}), bound_tracks, bound_frames,
                int(getattr(mgr, '_bystander_refused', 0) or 0))

    off = run(False)
    on = run(True)
    print('\n   arm      | tracks | bound tracks | bound face-frames | veto count')
    print(f'   veto off | {off[0]:6d} | {off[1]:12d} | {off[2]:17d} | {off[3]:5d}')
    print(f'   veto on  | {on[0]:6d} | {on[1]:12d} | {on[2]:17d} | {on[3]:5d}')
    print(f'   target face-frames lost to the veto: {off[2] - on[2]}')


if __name__ == '__main__':
    main()
