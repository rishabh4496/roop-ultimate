#!/usr/bin/env python3
"""Offline comparison of landmark filters on CAPTURED real tracks (`probe_stage25_capture.py`).

Arms, all on the same tracks:
    RAW      what the detector said
    SHIPPED  what the cache holds after `AdaptiveLandmarkSmoother` (causal, beta 0.35 -> 0.90)
    BI(bs)   zero-phase: the same velocity-ramped beta, run forward AND backward and averaged, one beta per
             frame taken from the 5-point velocity (central difference) and shared by kps / lm106 / lm68

Two numbers matter to the brief ("zero sub-pixel jitter when still, zero lag on fast turns"):
    still jitter   RMS second difference of the alignment (translation px, eye-line angle deg, scale %) on
                   frames whose RAW speed is below 0.004 face-widths/frame. White detector noise sigma shows up
                   as sigma*sqrt(6), so this reads the NOISE that reaches the warp.
    fast lag       on frames whose RAW speed is above 0.02 face-widths/frame, the mean of (arm - raw) projected
                   on the direction of motion, in % of the inter-ocular distance. A causal filter is NEGATIVE
                   (it trails); zero-phase should sit near 0. Also the RMS deviation from raw there.

    app/env/Scripts/python.exe app/tests/probe_stage25_filters.py a.npz b.npz ...
"""
import json
import sys

import numpy as np

BETA_FAST, THR, BAND = 0.90, 0.02, 0.01
STILL, FAST = 0.004, 0.02


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def runs_of(frames, cuts, min_len=12):
    """Index ranges [i, j) of consecutive frames that do not cross a cut."""
    out, start = [], 0
    cuts = set(int(c) for c in cuts)
    for k in range(1, len(frames) + 1):
        if k == len(frames) or frames[k] != frames[k - 1] + 1 or int(frames[k]) in cuts:
            if k - start >= min_len:
                out.append((start, k))
            start = k
    return out


def face_scale(k):
    return np.linalg.norm(k[:, 1] - k[:, 0], axis=1)


def speed(k):
    """Centroid speed in face-widths per frame, central difference (symmetric)."""
    c = k.mean(axis=1)
    v = np.zeros(len(c))
    if len(c) > 2:
        v[1:-1] = np.linalg.norm(c[2:] - c[:-2], axis=1) / 2.0
        v[0], v[-1] = v[1], v[-2]
    return v / np.maximum(face_scale(k), 1e-6)


def betas(k, beta_slow):
    return beta_slow + (BETA_FAST - beta_slow) * smoothstep((speed(k) - (THR - BAND)) / (2 * BAND))



def ema(x, beta):
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        b = beta[t]
        out[t] = b * x[t] + (1 - b) * out[t - 1]
    return out


def bidirectional(x, beta):
    """Zero-phase adaptive EMA. x: (T, ...) ; beta: (T,). Forward pass + backward pass, averaged."""
    fwd = ema(x, beta)
    bwd = ema(x[::-1], beta[::-1])[::-1]
    return 0.5 * (fwd + bwd)


def alignment(k):
    """Translation (px), eye-line angle (deg) and scale (px) of each frame's 5 points."""
    c = k.mean(axis=1)
    e = k[:, 1] - k[:, 0]
    return c[:, 0], c[:, 1], np.degrees(np.arctan2(e[:, 1], e[:, 0])), np.linalg.norm(e, axis=1)


def rms(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float('nan')


def still_jitter(k, ref_speed, real):
    """RMS second difference of the alignment parameters over still, real frames. (x px, y px, deg, scale %)"""
    tx, ty, ang, sc = alignment(k)
    ok = real[1:-1] & real[:-2] & real[2:] & (ref_speed[1:-1] < STILL)
    if ok.sum() < 6:
        return None
    d2 = lambda a: a[2:] - 2 * a[1:-1] + a[:-2]                      # noqa: E731
    sc_pct = 100.0 * sc / np.maximum(sc.mean(), 1e-6)
    return dict(n=int(ok.sum()), tx=rms(d2(tx)[ok]), ty=rms(d2(ty)[ok]), deg=rms(d2(ang)[ok]), scale_pct=rms(d2(sc_pct)[ok]))


def fast_lag(k, raw, ref_speed, real):
    c, cr = k.mean(axis=1), raw.mean(axis=1)
    v = np.zeros_like(cr)
    v[1:-1] = (cr[2:] - cr[:-2]) / 2.0
    n = np.linalg.norm(v, axis=1)
    ok = real & (ref_speed > FAST) & (n > 1e-6)
    ok[0] = ok[-1] = False
    if ok.sum() < 3:
        return None
    unit = v[ok] / n[ok, None]
    iod = face_scale(raw)[ok]
    along = ((c[ok] - cr[ok]) * unit).sum(axis=1) / iod * 100.0
    dev = np.linalg.norm(c[ok] - cr[ok], axis=1) / iod * 100.0
    return dict(n=int(ok.sum()), lag_pct_iod=float(along.mean()), rms_dev_pct_iod=rms(dev))


def analyse(path):
    d = np.load(path)
    track, frame, cuts = d['track'], d['frame'], d['cuts']
    arms = {'RAW': [], 'SHIPPED': []}
    sweep = [0.35, 0.2, 0.12]
    for bs in sweep:
        arms['BI(%.2f)' % bs] = []
    lm68 = {'RAW': [], 'BI(%.2f)' % sweep[1]: []}
    results = {a: {'still': [], 'fast': []} for a in arms}
    l106 = {a: [] for a in ('RAW', 'SHIPPED', 'BI(%.2f)' % sweep[1])}
    l68 = {a: [] for a in lm68}
    keep = {a: [] for a in ('SHIPPED',) + tuple('BI(%.2f)' % b for b in sweep)}
    for t in sorted(set(track.tolist())):
        idx = np.nonzero(track == t)[0]
        for (a, b) in runs_of(frame[idx], cuts):
            sel = idx[a:b]
            raw = d['kps_raw'][sel].astype(np.float64)
            if not d['has_raw'][sel].all():
                continue
            cur = d['kps_cur'][sel].astype(np.float64)
            real = ~(d['interp'][sel] | d['coasted'][sel])
            # stillness / fast-motion are classified from a LOW-NOISE reference (the heaviest zero-phase smooth of
            # the raw track), the same for every arm: the raw track's own central-difference speed is dominated by
            # detector noise (sigma ~0.3 px on a 60 px face reads as 0.005 widths/frame, above the still threshold)
            rs = speed(bidirectional(raw, betas(raw, 0.12)))
            outs = {'RAW': raw, 'SHIPPED': cur}
            lm106_in = d['lm106_raw'][sel].astype(np.float64)
            lm68_in = d['lm68'][sel].astype(np.float64)
            for bs in sweep:
                be = betas(raw, bs)
                outs['BI(%.2f)' % bs] = bidirectional(raw, be)
                if abs(bs - sweep[1]) < 1e-9:
                    f106 = bidirectional(lm106_in, be)
                    f68 = bidirectional(lm68_in, be)
                    sv = lambda lm: (lm[..., :2] if lm.shape[-1] > 2 else lm)           # noqa: E731
                    for name, arr in (('RAW', lm106_in), ('SHIPPED', d['lm106_cur'][sel].astype(np.float64)), ('BI(%.2f)' % bs, f106)):
                        l106[name].append(_pts_jitter(arr, rs, real))
                    for name, arr in (('RAW', lm68_in), ('BI(%.2f)' % bs, f68)):
                        l68[name].append(_pts_jitter(arr[..., :2], rs, real))
            # EXPRESSION RETENTION: std over time of the mouth's vertical extent (speech), as a ratio to the RAW
            # track's. 1.0 = every bit of real mouth motion survives; a stronger smoother flattens speech.
            rel_raw = _rel_std(lm106_in)
            outs106 = {'SHIPPED': d['lm106_cur'][sel].astype(np.float64)}
            for bs in sweep:
                outs106['BI(%.2f)' % bs] = bidirectional(lm106_in, betas(raw, bs))
            for name, arr in outs106.items():
                keep[name].append(dict(n=b - a, ratio=_rel_std(arr) / max(rel_raw, 1e-9)))
            for name, k in outs.items():
                j = still_jitter(k, rs, real)
                f = fast_lag(k, raw, rs, real) if name != 'RAW' else None
                if j:
                    results[name]['still'].append(j)
                if f:
                    results[name]['fast'].append(f)
    return results, l106, l68, keep


MOUTH = slice(52, 72)          # insightface 2d106: the 20 lip points (outer + inner contour)


def _rel_std(arr):
    """Std over time of the mouth's vertical extent, in eye-distance units: how much speech survives."""
    m = arr[:, MOUTH, 1]
    extent = m.max(axis=1) - m.min(axis=1)
    return float(np.std(extent))


def _pts_jitter(arr, ref_speed, real):
    ok = real[1:-1] & real[:-2] & real[2:] & (ref_speed[1:-1] < STILL)
    if ok.sum() < 6:
        return None
    d2 = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    return dict(n=int(ok.sum()), rms=rms(np.linalg.norm(d2[ok], axis=-1)))


def wmean(items, key):
    items = [i for i in items if i]
    if not items:
        return float('nan')
    w = np.array([i['n'] for i in items], float)
    return float(np.average([i[key] for i in items], weights=w))


def main(paths):
    summary = {}
    for p in paths:
        res, l106, l68, keep = analyse(p)
        name = p.replace('\\', '/').split('/')[-1]
        summary[name] = {}
        print('\n== %s' % name)
        print('%-10s %8s  | %8s %8s %8s %8s | %8s %10s' % ('arm', 'still n', 'tx px', 'ty px', 'deg', 'scale%', 'fast n', 'lag %iod'))
        for arm, r in res.items():
            sn = sum(i['n'] for i in r['still'])
            fn = sum(i['n'] for i in r['fast'])
            row = dict(still_n=sn, tx=wmean(r['still'], 'tx'), ty=wmean(r['still'], 'ty'), deg=wmean(r['still'], 'deg'),
                       scale_pct=wmean(r['still'], 'scale_pct'), fast_n=fn,
                       lag_pct_iod=wmean(r['fast'], 'lag_pct_iod'), rms_dev_pct_iod=wmean(r['fast'], 'rms_dev_pct_iod'))
            summary[name][arm] = row
            print('%-10s %8d  | %8.4f %8.4f %8.4f %8.4f | %8d %10.3f' % (arm, sn, row['tx'], row['ty'], row['deg'], row['scale_pct'], fn, row['lag_pct_iod']))
        print('dense landmarks, still-frame jitter (RMS second difference of a point, px):')
        for arm, v in l106.items():
            print('   lm106 %-9s %.4f' % (arm, wmean(v, 'rms')))
            summary[name]['lm106_' + arm] = wmean(v, 'rms')
        print('mouth-opening retention (std over time of the lip extent, ratio to RAW; 1.0 = speech intact):')
        for arm, v in keep.items():
            print('   %-9s %.3f' % (arm, wmean(v, 'ratio')))
            summary[name]['keep_' + arm] = wmean(v, 'ratio')
        for arm, v in l68.items():
            print('   lm68  %-9s %.4f' % (arm, wmean(v, 'rms')))
            summary[name]['lm68_' + arm] = wmean(v, 'rms')
    print('\nJSON ' + json.dumps(summary))


if __name__ == '__main__':
    main(sys.argv[1:])
