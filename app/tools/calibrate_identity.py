"""Measure the same-person / different-person distance gap on YOUR footage.

The whole wrong-face problem is a threshold sitting in the wrong place, and the
right place depends on the recogniser AND on the material. This measures it
instead of guessing, for both recognisers side by side:

  w600k    — buffalo_l's ArcFace embedding, what matching uses today
  AdaFace  — the quality-adaptive recogniser, on its own distance scale

Ground truth without hand labelling, from two facts that are free:
  DIFFERENT people — any two faces detected in the SAME frame are, necessarily,
                     two different people.
  SAME person      — a frame where exactly ONE face is detected, followed by
                     another such frame within a few frames, is overwhelmingly
                     the same person continuing.

What matters is not the absolute numbers but the GAP: how far the top of the
same-person distribution sits below the bottom of the different-person one. A
wider gap means a threshold can be strict without dropping hard frames — which
is the entire reason to consider a different recogniser.

Usage (from the app directory, inside the venv):
    python tools/calibrate_identity.py <video> [--frames 400] [--step 15]

Run it when nothing else is using the GPU.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                    # noqa: E402
from roop.utilities import compute_cosine_distance     # noqa: E402


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float('nan')


def _report(name, same, diff):
    print(f'\n=== {name} ===')
    if not same or not diff:
        print('  not enough pairs — try more frames, or footage with both '
              'solo shots and multi-person shots')
        return None

    same, diff = np.array(same), np.array(diff)
    s95, d05 = _pct(same, 95), _pct(diff, 5)
    print(f'  same person      n={len(same):5d}  '
          f'median {np.median(same):.3f}  p95 {s95:.3f}  max {same.max():.3f}')
    print(f'  different people n={len(diff):5d}  '
          f'min {diff.min():.3f}  p5 {d05:.3f}  median {np.median(diff):.3f}')

    gap = d05 - s95
    if gap <= 0:
        print(f'  GAP: NONE ({gap:+.3f}) — the distributions overlap, so no '
              f'threshold separates them cleanly on this footage.')
    else:
        print(f'  GAP: {gap:.3f}   suggested threshold {(s95 + d05) / 2:.2f}  '
              f'(midpoint of p95-same and p5-different)')
    overlap = float((same > d05).mean() * 100)
    print(f'  {overlap:.1f}% of same-person pairs sit above the different-people p5')
    return gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--frames', type=int, default=400, help='frames to sample')
    ap.add_argument('--step', type=int, default=15, help='sample every Nth frame')
    args = ap.parse_args()

    os.environ.setdefault('ROOP_ADAFACE', '1')       # so the module will load
    from roop.face_util import get_all_faces
    from roop.capturer import get_video_frame, get_video_frame_total
    from roop import recognizer_adaface as ada

    total = get_video_frame_total(args.video)
    print(f'{os.path.basename(args.video)}: {total} frames, '
          f'sampling every {args.step} up to {args.frames}')

    have_ada = False
    try:
        ada.download()
        ada.embed_crop(np.zeros((112, 112, 3), np.uint8))
        have_ada = True
    except Exception as e:
        print(f'AdaFace unavailable ({e}); reporting w600k only')

    solo_prev = None                    # (frame_idx, w600k emb, adaface emb)
    same_w, diff_w, same_a, diff_a = [], [], [], []

    for n in range(args.frames):
        f_idx = n * args.step
        if total and f_idx >= total:
            break
        frame = get_video_frame(args.video, f_idx)
        if frame is None:
            continue
        try:
            faces = get_all_faces(frame) or []
        except Exception:
            continue

        embs = []
        for f in faces:
            w = getattr(f, 'embedding', None)
            a = ada.face_embedding(f, frame) if have_ada else None
            embs.append((w, a))

        # DIFFERENT: every pair inside one frame.
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                if embs[i][0] is not None and embs[j][0] is not None:
                    diff_w.append(compute_cosine_distance(embs[i][0], embs[j][0]))
                if embs[i][1] is not None and embs[j][1] is not None:
                    diff_a.append(compute_cosine_distance(embs[i][1], embs[j][1]))

        # SAME: consecutive solo frames, close enough together to be continuous.
        if len(faces) == 1:
            cur = (f_idx, embs[0][0], embs[0][1])
            if solo_prev is not None and f_idx - solo_prev[0] <= args.step * 3:
                if solo_prev[1] is not None and cur[1] is not None:
                    same_w.append(compute_cosine_distance(solo_prev[1], cur[1]))
                if solo_prev[2] is not None and cur[2] is not None:
                    same_a.append(compute_cosine_distance(solo_prev[2], cur[2]))
            solo_prev = cur
        elif len(faces) != 1:
            solo_prev = None

        if n % 50 == 0:
            print(f'  ...{n}/{args.frames} sampled', flush=True)

    gap_w = _report('w600k (buffalo_l) — drives matching today', same_w, diff_w)
    gap_a = _report('AdaFace ir101 — matching-only candidate', same_a, diff_a) if have_ada else None

    print('\n=== verdict ===')
    if gap_w is None or gap_a is None:
        print('  Not enough data from both recognisers to compare.')
    elif gap_a > gap_w:
        print(f'  AdaFace separates this footage BETTER (gap {gap_a:.3f} vs '
              f'{gap_w:.3f}). Set ROOP_ADAFACE=1 and ROOP_ADAFACE_DIST to the '
              f'suggested AdaFace threshold above.')
    else:
        print(f'  AdaFace is NOT better here (gap {gap_a:.3f} vs {gap_w:.3f}). '
              f'Stay on w600k and use its suggested threshold as '
              f'max_face_distance.')


if __name__ == '__main__':
    main()
