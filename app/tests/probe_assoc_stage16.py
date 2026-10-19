"""Stage 16: replay the pre-pass association on recorded detections.

`probe_bystander_stage16.py collect` records, per detection, the box, keypoints,
pose, crop contamination and the unit w600k embedding. This replays
`procmgr_tracking._consume`'s association on those records -- Hungarian over
(detections x live tracks), appearance-only Re-ID for the unmatched, EMA
identity with the dirty-seed rule -- using `roop.association_cost`, once with
the legacy cost and once with the composite, and asks the one question that
matters: **how many detections were handed to the OTHER person's track?**

Truth is `probe_bystander_stage16.label_people` (identity-clustered on clean
frames, carried across a contact stretch by x-order inside a run); each track's
owner is the majority truth id of its CLEAN detections. A labelled detection
assigned to a track owned by the other person is a wrong-person assignment.
That is what paints the bystander's face with the target's swap.

numpy only (no GPU): re-runs from the saved .npz files.

    env/Scripts/python.exe tests/probe_assoc_stage16.py --npz Love2.npz [--npz ...]
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop.association_cost import (CostWeights, composite_cost, cosine_distance,  # noqa: E402
                                   iou, legacy_cost, propagate_kps)
import probe_bystander_stage16 as pb                                               # noqa: E402

IOU_MIN, EMB_MAX, REID_MAX, STALE = 0.2, 0.7, 0.5, 15
CONTAM_MAX = 0.35           # face_contact.CONTAM_MAX


def _unpack(rows):
    f = rows[:, 0].astype(int)
    box = rows[:, 1:5]
    contam = rows[:, 5]
    n = rows[:, 7].astype(int)
    if rows.shape[1] >= 8 + 13 + 512:
        kps = rows[:, 8:18].reshape(-1, 5, 2)
        pose = rows[:, 18:21]
    else:
        kps = np.full((len(rows), 5, 2), np.nan, np.float32)
        pose = np.full((len(rows), 3), np.nan, np.float32)
    emb = rows[:, -512:]
    return f, box, contam, n, kps, pose, emb


def replay(rows, mode, weights, step=1):
    f, box, contam, n, kps, pose, emb = _unpack(rows)
    frames = sorted(set(f))
    by_frame = {fr: list(np.where(f == fr)[0]) for fr in frames}
    active, retired = [], []
    next_id = 0
    assign = np.full(len(rows), -1, int)

    def predict(t, fr):
        pb_ = t['bbox']
        dt = fr - t['last_seen']
        if 0 < dt <= 6 and t['vel'] is not None and np.any(t['vel']):
            proj = t['bbox'] + t['vel'] * dt
            if proj[2] > proj[0] and proj[3] > proj[1]:
                pb_ = proj
        return pb_

    for fr in frames:
        ix = by_frame[fr]
        fresh = []
        for t in active:
            (fresh if t['last_seen'] >= fr - STALE * step else retired).append(t)
        active = fresh
        best = {}
        if active and ix:
            from scipy.optimize import linear_sum_assignment
            cm = np.full((len(ix), len(active)), 1e5, np.float32)
            for a, i in enumerate(ix):
                dirty = contam[i] >= CONTAM_MAX
                for b, t in enumerate(active):
                    pbx = predict(t, fr)
                    cd = cosine_distance(t['emb'], emb[i])
                    if mode == 'legacy':
                        c = legacy_cost(iou(box[i], pbx), cd, dirty, IOU_MIN, EMB_MAX)
                    else:
                        c = composite_cost(
                            box[i], kps[i], pose[i], cd, float(contam[i]), dirty,
                            pbx, propagate_kps(t['kps'], t['bbox'], pbx), t['pose'],
                            IOU_MIN, EMB_MAX, weights)
                    if c is not None:
                        cm[a, b] = c
            r, c_ = linear_sum_assignment(cm)
            for a, b in zip(r, c_):
                if cm[a, b] < 1e4:
                    best[a] = active[b]
        used = {t['id'] for t in best.values()}
        for a, i in enumerate(ix):
            dirty = contam[i] >= CONTAM_MAX
            t = best.get(a)
            reid = False
            if t is None and not dirty:
                cand, cd_best = None, 9.0
                for u in active:
                    if u['id'] in used:
                        continue
                    d = cosine_distance(u['emb'], emb[i])
                    if d <= EMB_MAX and d < cd_best:
                        cand, cd_best = u, d
                if retired:
                    ds = [cosine_distance(u['emb'], emb[i]) for u in retired[-500:]]
                    k = int(np.argmin(ds))
                    if ds[k] <= REID_MAX and ds[k] < cd_best:
                        cand, cd_best = retired[-500:][k], ds[k]
                        retired.remove(cand)
                        active.append(cand)
                if cand is not None:
                    t, reid = cand, True
            if t is None:
                t = {'id': next_id, 'bbox': box[i].copy(), 'vel': np.zeros(4, np.float32),
                     'emb': emb[i].copy(), 'emb_sum': emb[i].astype(np.float64).copy(),
                     'dirty_seed': bool(dirty), 'last_seen': fr,
                     'kps': kps[i].copy(), 'pose': pose[i].copy()}
                next_id += 1
                active.append(t)
            else:
                dt = fr - t['last_seen']
                if reid:
                    t['vel'] = np.zeros(4, np.float32)
                elif dt > 0:
                    t['vel'] = (box[i] - t['bbox']) / dt
                t['bbox'] = box[i].copy()
                t['last_seen'] = fr
                t['kps'] = kps[i].copy()
                t['pose'] = pose[i].copy()
                if not dirty:
                    if t['dirty_seed']:
                        t['emb'], t['dirty_seed'] = emb[i].copy(), False
                    elif cosine_distance(t['emb'], emb[i]) <= 0.5:
                        t['emb'] = (0.75 * t['emb'] + 0.25 * emb[i]).astype(np.float32)
            used.add(t['id'])
            assign[i] = t['id']
    return assign


def score(rows, assign, pid):
    f, box, contam, n, kps, pose, emb = _unpack(rows)
    tids = sorted(set(assign.tolist()))
    owner = {}
    for t in tids:
        # Owner = majority truth over EVERY labelled detection of the track. (An
        # owner taken from clean frames only leaves a track born inside a contact
        # "unowned", and its errors uncounted.) A detection that disagrees with
        # its track's majority is a wrong-person assignment.
        votes = Counter(int(pid[i]) for i in np.where((assign == t) & (pid >= 0))[0])
        if sum(votes.values()) >= 3:
            owner[t] = votes.most_common(1)[0][0]
    out = {}
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01)):
        m = (pid >= 0) & (contam >= lo) & (contam < hi)
        idx = np.where(m)[0]
        wrong = sum(1 for i in idx if assign[i] in owner and owner[assign[i]] != pid[i])
        unowned = sum(1 for i in idx if assign[i] not in owner)
        out[(lo, hi)] = (len(idx), wrong, unowned)
    return out, len(tids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', action='append', required=True)
    ap.add_argument('--clean', type=float, default=0.1)
    ap.add_argument('--identity', type=float, default=CostWeights.identity)
    ap.add_argument('--iou', type=float, default=CostWeights.iou)
    ap.add_argument('--kpt', type=float, default=CostWeights.kpt)
    ap.add_argument('--pose', type=float, default=CostWeights.pose)
    ap.add_argument('--contam-full', type=float, default=CostWeights.contam_full)
    ap.add_argument('--contam-zero', type=float, default=CostWeights.contam_zero)
    ap.add_argument('--contam-floor', type=float, default=CostWeights.contam_floor)
    args = ap.parse_args()
    w = CostWeights(args.identity, args.iou, args.kpt, args.pose,
                    args.contam_full, args.contam_zero, args.contam_floor)
    print(f'composite weights: {w}')
    tot = {m: {} for m in ('legacy', 'composite')}
    for path in args.npz:
        rows = np.load(path, allow_pickle=True)['rows']
        pid, _ = pb.label_people(rows, args.clean)
        if pid is None:
            print(f'[{os.path.basename(path)}] cannot label')
            continue
        print(f'\n=== {os.path.basename(path)}  detections {len(rows)}  labelled {int((pid >= 0).sum())}')
        print('  contam band   |     legacy: n  wrong  unowned |  composite: n  wrong  unowned')
        res = {}
        for mode in ('legacy', 'composite'):
            a = replay(rows, mode, w)
            res[mode] = score(rows, a, pid)
        for band in res['legacy'][0]:
            n_, wl, ul = res['legacy'][0][band]
            if n_ == 0:
                continue
            _, wc, uc = res['composite'][0][band]
            print(f'  [{band[0]:.2f},{band[1]:.2f})  | {n_:14d} {wl:6d} {ul:8d} | {n_:15d} {wc:6d} {uc:8d}')
            for m, (w_, u_) in (('legacy', (wl, ul)), ('composite', (wc, uc))):
                t = tot[m].setdefault(band, [0, 0, 0])
                t[0] += n_; t[1] += w_; t[2] += u_
        print(f'  tracks: legacy {res["legacy"][1]}  composite {res["composite"][1]}')
        a_l, a_c = replay(rows, 'legacy', w), replay(rows, 'composite', w)
        # How many detections do the two costs partition differently? Map each
        # legacy track to the composite track most of its detections landed in.
        major = {t: Counter(a_c[a_l == t].tolist()).most_common(1)[0][0] for t in set(a_l.tolist())}
        diff = np.array([major[t] != c for t, c in zip(a_l, a_c)])
        f_, _, contam_, *_ = _unpack(rows)
        print(f'  detections assigned differently: {int(diff.sum())} of {len(diff)}'
              f'  (of which contam >= 0.2: {int((diff & (contam_ >= 0.2)).sum())})')
    print('\n--- pooled ---')
    print('  contam band   |  legacy wrong%  unowned% |  composite wrong%  unowned%')
    for band in tot['legacy']:
        n_, wl, ul = tot['legacy'][band]
        _, wc, uc = tot['composite'][band]
        if n_:
            print(f'  [{band[0]:.2f},{band[1]:.2f})  | n={n_:5d} {100*wl/n_:6.2f}%  {100*ul/n_:6.2f}% | '
                  f'{100*wc/n_:6.2f}%  {100*uc/n_:6.2f}%')


if __name__ == '__main__':
    main()
