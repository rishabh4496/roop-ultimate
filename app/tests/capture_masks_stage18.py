"""Stage 18 helper: run `two_face_video.py` and record the production masks.

For every swapped face this keeps, in call order (run it with --threads 1 so the
order is frame order):

  raw    the mask `process_mask` returned (XSeg + occluder + parser fusion, before
         the mask stabilizer)                                  1 = keep the original
  final  the mask `_composite_mask` actually composites (after `_ms.apply`)
  crop   the aligned target crop the mask belongs to

Masks live in the aligned crop, where the alignment has already removed the face's
motion, so frame-to-frame change in them IS mask instability and needs no flow
compensation to read.

Usage (cwd = app), exactly the two_face_video.py arguments plus --capture-out:
    env/Scripts/python.exe tests/capture_masks_stage18.py --capture-out masks.npz -- \
        --tag x --video ... --sources ... [two_face_video args]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    argv = sys.argv[1:]
    out = None
    if '--capture-out' in argv:
        i = argv.index('--capture-out')
        out = argv[i + 1]
        del argv[i:i + 2]
    if argv and argv[0] == '--':
        argv = argv[1:]
    sys.argv = [sys.argv[0]] + argv

    import two_face_video as tfv
    tfv._apply_startup_runtime_environment()
    from roop.ProcessMgr import ProcessMgr

    import pickle
    items = []          # one dict per process_mask call, in call order
    orig_pm = ProcessMgr.process_mask
    orig_cm = ProcessMgr._composite_mask

    def pm(self, *a, **k):
        res = orig_pm(self, *a, **k)
        try:
            if k.get('reuse_mask') is None and isinstance(res, tuple) and res[1] is not None:
                tf = k.get('target_face')
                aligned = a[1] if len(a) > 1 else None
                items.append({
                    'raw': np.asarray(res[1], np.float32).copy(),
                    'crop': None if aligned is None else np.asarray(aligned).copy(),
                    'bbox': None if tf is None else np.asarray(tf.bbox, np.float32).copy(),
                    'tid': None if tf is None else (tf.get('_track_id') if isinstance(tf, dict) else None),
                    'final': None})
                pending[0] = items[-1]
        except Exception as e:                         # capture must never break the render
            print('[capture] process_mask:', e)
        return res

    last = {'id': None}
    pending = [None]

    def cm(self, mask, aligned_img, target):
        if mask is not None and id(mask) != last['id']:
            if pending[0] is not None and pending[0]['final'] is None:
                pending[0]['final'] = np.asarray(mask, np.float32).copy()
            last['id'] = id(mask)
        return orig_cm(self, mask, aligned_img, target)

    ProcessMgr.process_mask = pm
    ProcessMgr._composite_mask = cm
    try:
        tfv.main()
    finally:
        print(f'[capture] {len(items)} process_mask calls, '
              f'{sum(1 for i in items if i["final"] is not None)} with a final mask')
        if out and items:
            with open(out, 'wb') as fh:
                pickle.dump(items, fh)
            print('[capture] wrote', out)


if __name__ == '__main__':
    main()
