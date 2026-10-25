"""Per-frame 'was a face replaced' flag for each render, judged against the ORIGINAL clip's frame.

A frame is flagged when more than AREA_PCT % of its pixels differ from the original by more than LEVEL (max over
channels). Encoder noise sits far below that; a replaced face (hundreds of px square) sits far above. The threshold
is calibrated by printing the distribution so the gap between the two populations is visible.
"""
import json
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(errors='replace')
res = json.load(open(sys.argv[1], encoding='utf-8'))['results']
by_tag = {r['tag']: r for r in res}
clip = sys.argv[2]
LEVEL = 18
tags = ['base_5', 'base_6', 'k4_1', 'k4_4', 'k4r_2', 'k4r_3']
caps = {t: cv2.VideoCapture(by_tag[t]['path']) for t in tags}
orig = cv2.VideoCapture(clip)
area = {t: [] for t in tags}
n = 0
while n < 600:
    ok, fo = orig.read()
    if not ok:
        break
    for t in tags:
        okt, ft = caps[t].read()
        if not okt:
            area[t].append(np.nan)
            continue
        if ft.shape != fo.shape:
            ft = cv2.resize(ft, (fo.shape[1], fo.shape[0]))
        area[t].append(float((np.abs(fo.astype(np.int16) - ft).max(axis=2) > LEVEL).mean() * 100))
    n += 1
A = {t: np.array(v) for t, v in area.items()}
b = A['base_5']
print('frames', n, 'size', fo.shape)
print('base changed-area % percentiles (10/25/50/75/90/99):', np.round(np.percentile(b[~np.isnan(b)], [10, 25, 50, 75, 90, 99]), 3))
hist, edges = np.histogram(b[~np.isnan(b)], bins=[0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 100])
print('hist', list(zip([round(e, 2) for e in edges[:-1]], hist)))
AREA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
flag = {t: (A[t] > AREA) for t in tags}
for t in tags:
    print(t, 'frames with a replaced face:', int(flag[t].sum()))
for x, y in (('base_5', 'base_6'), ('base_5', 'k4_1'), ('k4_1', 'k4_4'), ('base_5', 'k4r_2'), ('k4_1', 'k4r_2')):
    only_x = np.where(flag[x] & ~flag[y])[0]
    only_y = np.where(~flag[x] & flag[y])[0]
    print('%s vs %s: only in %s %d, only in %s %d' % (x, y, x, len(only_x), y, len(only_y)))
    if len(only_x) or len(only_y):
        def runs(ix):
            out = []
            for f in ix:
                if out and f == out[-1][1] + 1:
                    out[-1][1] = int(f)
                else:
                    out.append([int(f), int(f)])
            return out
        print('   only %s runs: %s' % (x, runs(only_x)[:24]))
        print('   only %s runs: %s' % (y, runs(only_y)[:24]))
