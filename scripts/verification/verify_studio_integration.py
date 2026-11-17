"""Offline functional probes of Studio's actual model/pre-pass entry points.

No throughput acceptance or physical 4K/3060 memory claim is made here.
Run parts in separate processes so sessions from one probe cannot mask OOM.
"""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'app'
sys.path[:0] = [str(APP), str(APP / 'tests')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['prescan', 'batch', 'compile-cpu'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    os.environ['ROOP_OFFLINE'] = '1'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ROOP_STAB_CHUNK_MB'] = '256'
    os.chdir(APP)
    from settings import Settings
    import angle_bench
    cfg = Settings('config.yaml')
    runtime = angle_bench.init_pipeline(cfg.provider, cfg.swap_model,
        cfg.selected_enhancer, cfg.mask_engine, sync_config=True)
    runtime.execution_threads = 20
    report = {'part': args.part, 'offline': True, 'performance_acceptance': False}
    if args.part == 'compile-cpu':
        from roop.ultramax_compile import compile_session
        report.update(compile_session('strict_fp32', 'cpu'))
    elif args.part == 'prescan':
        from roop.benchmark.regression import default_clip
        from roop.studio_prescan import scan
        from roop import trajectory_cache
        clip = default_clip(600, print)
        runtime.processing = True
        runtime.detector_engine = 'gnkps'
        runtime.g_desired_face_analysis = ['detection', 'recognition', 'landmark_2d_106', 'landmark_3d_68']
        try:
            report.update(scan(clip, {'stabilize_face': True, 'stabilization_strength': .5}))
            report['summary'].pop('thumbnails', None)
            assert trajectory_cache.active() is not None
        finally:
            runtime.processing = False
    else:
        import numpy as np
        from types import SimpleNamespace
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax
        from roop.hardware_streamer import VramSessionManager
        from roop import studio_parameters
        from roop.utilities import get_device
        processor = Enhance_UltraMax()
        processor.Initialize({'devicename': get_device()})
        import cv2
        plate = cv2.imread(str(APP / 'roop/assets/benchmark/source_reference.png'))
        if plate is None:
            raise RuntimeError('The offline reference face is unavailable')
        plate = cv2.resize(plate[55:245, 55:200], (512, 512))
        frames = [cv2.GaussianBlur(plate, (0, 0), sigma) for sigma in (2., 3., 4., 5.)]
        faces = [SimpleNamespace(bbox=np.array([0, 0, 400, 400]), aligned_img=None) for _ in frames]
        runtime.ultramax_batch_size = 1
        # This probe measures the restorer, independently of Stage 7's plate
        # replacement mask. Full render coverage belongs to the regression.
        runtime.enable_xseg_occlusion = False
        report['stage7_occlusion'] = False
        try:
            solo = [processor.Run(None, face, frame)[0] for face, frame in zip(faces, frames)]
            manager = VramSessionManager.for_device(runtime.cuda_device_id)
            original = manager.run_batches
            batches = []
            def observed(inputs, run, requested=8, frame_shape=None):
                def infer(items):
                    batches.append(len(items))
                    return run(items)
                return original(inputs, infer, requested, frame_shape)
            manager.run_batches = observed
            try:
                with studio_parameters.frame_scope(0, (2160, 3840, 3)):
                    batched = processor.RunBatch([None] * 4, faces, frames, 4)
            finally:
                manager.run_batches = original
            delta = np.abs(np.stack(solo).astype(np.int16) - np.stack([out[0] for out in batched]).astype(np.int16))
            report.update(neural_batches=batches, faces=len(batched), max_levels=int(delta.max()), mean_levels=float(delta.mean()))
            report['restored_calls'] = processor._faces
            report['changed_levels'] = [float(np.abs(out[0].astype(float) - frame).mean())
                                        for out, frame in zip(batched, frames)]
            print(json.dumps(report), flush=True)
            assert len(batched) == 4 and sum(batches) == 4
            assert processor._faces == 8 and min(report['changed_levels']) > 1.0
            assert report['max_levels'] <= 24 and report['mean_levels'] <= 1.0
        finally:
            processor.Release()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps(report, default=str), flush=True)


if __name__ == '__main__':
    main()
