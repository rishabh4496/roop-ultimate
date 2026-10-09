"""retinaface_r50 (ONNX Runtime + TensorRT) shape-profile probe: what profile is it built under, and what
does ONE pooled instance cost in device memory around its first inference?

    env/Scripts/python.exe tests/r50_profile_probe.py [--det-size 512] [--clip double/d4.mp4] [--out f.json]
    ROOP_TRT_STATIC_PROFILE=1 env/Scripts/python.exe tests/r50_profile_probe.py --det-size 640

Prints `trt_shape_profile.describe()` for the detector model, the provider options the pool really builds
its sessions with (profile shapes + the engine cache namespace), then for every pooled instance the
device-wide used VRAM (`torch.cuda.mem_get_info`, the quantity nvidia-smi reports; nvidia-smi is logged
beside it) just BEFORE and just AFTER that instance's first inference. TensorRT allocates its execution
context on the first inference, so a load-time number under-reports the model by the part that grows
(AGENTS.md: measure through `angle_bench.init_pipeline`, never a bare process).
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

import cv2                                  # noqa: E402
import fixtures                             # noqa: E402


def used_mib():
    import torch
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1048576.0


def smi_mib():
    from roop import baseline_probe
    return baseline_probe.nvidia_smi_used_mib()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-size', type=int, default=0, help='0 = the live config (face_detector_size)')
    ap.add_argument('--clip', default='double/d4.mp4')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--steady', type=int, default=100)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), 'None', 'None', sync_config=True)
    g.detector_engine = 'retinaface_r50'
    det_size = args.det_size or int(str(cfg.face_detector_size))

    from roop import retinaface, trt_shape_profile
    from roop.precision_policy import providers_for
    model = os.path.join(APP, 'models', 'retinaface_r50.onnx')
    report = {'det_size': det_size, 'clip': args.clip,
              'env': {k: os.environ[k] for k in os.environ if k.startswith('ROOP_TRT') or k.startswith('ROOP_DETECTOR')}}

    desc = trt_shape_profile.describe('face_detection:r50', model)
    print('[probe] trt_shape_profile.describe(face_detection:r50) =')
    print(json.dumps(desc, indent=2))
    report['describe'] = desc

    provs, decision = providers_for('face_detection:r50', g.execution_providers, model)
    trt = [p for p in provs if isinstance(p, (tuple, list)) and 'tensorrt' in str(p[0]).lower()]
    opts = dict(trt[0][1]) if trt else {}
    shown = {k: v for k, v in opts.items() if k.startswith('trt_profile') or k in (
        'trt_engine_cache_path', 'trt_fp16_enable', 'trt_layer_norm_fp32_fallback')}
    print('[probe] precision decision: requested=%s effective=%s backend=%s' % (
        decision.requested, decision.effective, decision.backend))
    print('[probe] TensorRT provider options the session is built with:')
    print(json.dumps(shown, indent=2))
    report['decision'] = {'requested': decision.requested, 'effective': decision.effective,
                          'backend': decision.backend}
    report['provider_options'] = shown

    cap = cv2.VideoCapture(fixtures.clip(args.clip, required=True))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    assert ok, 'could not read a frame'
    report['frame_shape'] = list(frame.shape)

    rows = []
    rows.append({'label': 'after init_pipeline (before the r50 pool)', 'torch_used_mib': round(used_mib(), 1),
                 'smi_used_mib': smi_mib()})
    t0 = time.perf_counter()
    pool = retinaface._ensure_pool('r50')
    build_s = time.perf_counter() - t0
    rows.append({'label': 'after pool build (%d session(s), no inference yet)' % len(pool['items']),
                 'torch_used_mib': round(used_mib(), 1), 'smi_used_mib': smi_mib(), 'seconds': round(build_s, 2)})
    report['pool_width'] = len(pool['items'])

    for i, det in enumerate(pool['items']):
        det.det_thresh = 0.5
        before, smi_before = used_mib(), smi_mib()
        t0 = time.perf_counter()
        boxes, kps = det.detect(frame, input_size=(det_size, det_size), max_num=0)
        first_ms = (time.perf_counter() - t0) * 1000.0
        after, smi_after = used_mib(), smi_mib()
        rows.append({'label': 'instance %d first inference' % i, 'torch_used_mib': round(after, 1),
                     'smi_used_mib': smi_after, 'delta_torch_mib': round(after - before, 1),
                     'delta_smi_mib': None if smi_before is None or smi_after is None else smi_after - smi_before,
                     'first_call_ms': round(first_ms, 1), 'faces': int(len(boxes))})

    det = pool['items'][0]
    for _ in range(10):
        det.detect(frame, input_size=(det_size, det_size), max_num=0)
    ts = []
    for _ in range(args.steady):
        t0 = time.perf_counter()
        det.detect(frame, input_size=(det_size, det_size), max_num=0)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    report['steady_ms'] = {'median': round(ts[len(ts) // 2], 3), 'p10': round(ts[len(ts) // 10], 3),
                           'p90': round(ts[(len(ts) * 9) // 10], 3), 'n': len(ts)}
    rows.append({'label': 'after %d steady calls' % args.steady, 'torch_used_mib': round(used_mib(), 1),
                 'smi_used_mib': smi_mib()})
    report['vram'] = rows

    print('[probe] VRAM around the first inference (device-wide, MiB):')
    for r in rows:
        print('  %-52s torch %8s  smi %8s  %s' % (
            r['label'], r['torch_used_mib'], r['smi_used_mib'],
            '  '.join('%s=%s' % (k, v) for k, v in r.items()
                      if k in ('delta_torch_mib', 'delta_smi_mib', 'first_call_ms', 'faces', 'seconds'))))
    print('[probe] steady-state detect (single caller, one frame, no pyramid): %s ms' % report['steady_ms'])
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1)
        print('[probe] wrote', args.out)


if __name__ == '__main__':
    main()
