"""Why does roop/retinaface.py say letterboxing r50 "suppresses scores under TensorRT on 16:9"?

    env/Scripts/python.exe tests/r50_letterbox_explain.py [--per-clip 40] [--out f.json]

The claim has two parts: (1) the GEOMETRY hurts r50, (2) it hurts UNDER TENSORRT. They are separable.
Each sampled frame is turned into one 640x640 blob per geometry and the SAME blob goes through two
sessions of the same model: the production TensorRT-mixed session (the pooled r50 instance) and a CUDA
FP32 session. Geometries (all BGR minus (104, 117, 123), pad = black before the mean is removed):

    squash          cv2.resize straight to 640x640 (what the app feeds r50)
    lb_centred_aa   aspect kept, centred, area-style antialiasing (face_engine's letterbox_cuda)
    lb_centred      aspect kept, centred, cv2 bilinear, no antialiasing (isolates the antialias term)
    lb_topleft      aspect kept, pasted at the top-left, bars right/bottom (roop/retinaface.py's
                    own non-`direct_square` branch -- what the comment was probably written against)

Reported per geometry and provider: anchors >= 0.5, faces after the app's NMS, faces found that the
FP32-squash reference also finds (IoU >= 0.5) / misses / extra, and, on the anchors FP32 scores >= 0.5, the
TensorRT/FP32 score ratio -- the direct test of "suppressed under TensorRT".
"""
import argparse
import json
import os
import statistics as st
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import fixtures                                   # noqa: E402

CLIPS = [('d1', 'double/d1.mp4', 0, 418), ('d4', 'double/d4.mp4', 0, 600),
         ('d6', 'double/d6.mp4', 0, 488), ('Love', 'Love.mp4', 1100, 1700)]
S = 640
MEAN = (104.0, 117.0, 123.0)


def build(frame, geom):
    """-> (640x640 BGR uint8 image, (sx, sy, ox, oy)) with canvas = frame * s + o."""
    h, w = frame.shape[:2]
    if geom == 'squash':
        return cv2.resize(frame, (S, S)), (S / w, S / h, 0.0, 0.0)
    r = min(S / w, S / h)
    rw, rh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    interp = cv2.INTER_AREA if geom == 'lb_centred_aa' else cv2.INTER_LINEAR
    small = cv2.resize(frame, (rw, rh), interpolation=interp)
    canvas = np.zeros((S, S, 3), np.uint8)
    ox, oy = ((S - rw) // 2, (S - rh) // 2) if geom != 'lb_topleft' else (0, 0)
    canvas[oy:oy + rh, ox:ox + rw] = small
    return canvas, (rw / w, rh / h, float(ox), float(oy))


def decode(net_outs, geom_info, thresh):
    from roop.retinaface import _decode_boxes, _decode_landmarks, _generate_priors
    from roop.nms import nms_keep
    loc, conf, landms = net_outs[0][0], net_outs[1][0], net_outs[2][0]
    scores = conf[:, 1] if abs(float(conf[0].sum()) - 1.0) < 1e-3 else None
    assert scores is not None, 'export without in-graph softmax'
    priors = _generate_priors((S, S))
    pos = np.where(scores >= thresh)[0]
    sx, sy, ox, oy = geom_info
    boxes = _decode_boxes(loc[pos], priors[pos]) * S
    boxes[:, 0::2] = (boxes[:, 0::2] - ox) / sx
    boxes[:, 1::2] = (boxes[:, 1::2] - oy) / sy
    det = np.hstack([boxes, scores[pos][:, None]]).astype(np.float32)
    det = det[det[:, 4].argsort()[::-1]]
    keep = nms_keep(det, 0.3, offset=1.0) if len(det) else []
    return scores, det[keep]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-clip', type=int, default=40)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    import onnxruntime
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), 'None', 'None', sync_config=True)
    g.detector_engine = 'retinaface_r50'
    from roop import retinaface
    trt = retinaface._ensure_pool('r50')['items'][0].session
    assert trt.get_providers()[0] == 'TensorrtExecutionProvider', trt.get_providers()
    model = os.path.join(APP, 'models', 'retinaface_r50.onnx')
    fp32 = onnxruntime.InferenceSession(
        model, providers=[('CUDAExecutionProvider', {'use_tf32': '0'}), 'CPUExecutionProvider'])
    assert fp32.get_providers()[0] == 'CUDAExecutionProvider'
    name = trt.get_inputs()[0].name
    outs = [o.name for o in trt.get_outputs()]
    geoms = ['squash', 'lb_centred_aa', 'lb_centred', 'lb_topleft']
    providers = {'trt_mixed': trt, 'cuda_fp32': fp32}

    report = {}
    for clip, rel, lo, hi in CLIPS:
        cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
        idxs = np.linspace(lo, hi - 1, args.per_clip).astype(int)
        acc = {(p, gm): {'anchors': 0, 'faces': 0, 'found': 0, 'missed': 0, 'extra': 0, 'ratios': [], 'max_scores': []}
               for p in providers for gm in geoms}
        ref_total = 0
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            res = {}
            for gm in geoms:
                img, info = build(frame, gm)
                blob = cv2.dnn.blobFromImage(img, 1.0, (S, S), MEAN, swapRB=False)
                for pn, sess in providers.items():
                    res[(pn, gm)] = decode(sess.run(outs, {name: blob}), info, 0.5)
            ref = res[('cuda_fp32', 'squash')][1]
            ref_total += len(ref)
            for (pn, gm), (scores, det) in res.items():
                a = acc[(pn, gm)]
                a['anchors'] += int((scores >= 0.5).sum())
                a['faces'] += len(det)
                a['max_scores'].append(float(scores.max()))
                a['found'] += sum(1 for r in ref if any(iou(r, d) >= 0.5 for d in det))
                a['missed'] += sum(1 for r in ref if not any(iou(r, d) >= 0.5 for d in det))
                a['extra'] += sum(1 for d in det if not any(iou(r, d) >= 0.5 for r in ref))
            for gm in geoms:                                       # TRT/FP32 on the anchors FP32 calls a face
                s_trt, s_f32 = res[('trt_mixed', gm)][0], res[('cuda_fp32', gm)][0]
                m = s_f32 >= 0.5
                if m.any():
                    acc[('trt_mixed', gm)]['ratios'].extend((s_trt[m] / s_f32[m]).tolist())
        cap.release()
        rep = {'frames': len(idxs), 'reference_faces(cuda_fp32 squash)': ref_total}
        for (pn, gm), a in acc.items():
            r = a.pop('ratios')
            ms = a.pop('max_scores')
            a['median_max_score'] = round(st.median(ms), 4)
            if pn == 'trt_mixed':
                a['trt_over_fp32_score_median'] = round(st.median(r), 4) if r else None
                a['trt_over_fp32_score_p5'] = round(float(np.percentile(r, 5)), 4) if r else None
            rep['%s | %s' % (pn, gm)] = a
        report[clip] = rep
        print('== %s (%d frames, %d reference faces)' % (clip, len(idxs), ref_total), flush=True)
        for k, v in rep.items():
            if '|' in k:
                print('  %-28s %s' % (k, json.dumps(v)), flush=True)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
