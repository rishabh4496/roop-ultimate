"""Diagnostic: does hyperswap's batch>1 TRT failure permanently disable
TensorRT for single-frame (batch=1) Run() calls too?

Theory being tested: FaceSwapInsightFace._infer() catches ANY exception
(including a batch-shape-specific one) and calls _rebuild_without_trt(),
which rebuilds the session WITHOUT TensorRT and sets self._trt_disabled=True
permanently. If the very first call that ever fails happens to be a batch
call (RunBatch/RunBatchMulti), single-frame Run() calls made AFTER that
never get a chance to use TensorRT again, even though batch=1 may never
have actually been broken under TRT.

Isolates JUST the swap processor (no detection/masking/tracking pipeline)
so this runs in seconds, not minutes. Real source (harjot faceset) and a
real aligned target crop (from s1.mp4's first frame) are used so the timing
reflects real inference cost, not synthetic-data shortcuts.

Usage: env/Scripts/python.exe tests/diag_trt_fallback.py
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import angle_bench as ab                      # noqa: E402
from two_face_video import load_library_faceset  # noqa: E402
from roop.procmgr_tiling import PixelBoostMixin  # noqa: E402
from roop.face_util import get_all_faces, align_crop  # noqa: E402


def _timed_run(p, source_face, target_face, blob, n=8, label=""):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        p.Run(source_face, target_face, blob.copy())
        times.append(time.perf_counter() - t0)
    times_ms = [t * 1000 for t in times]
    print(f"[diag] {label}: n={n} mean={sum(times_ms)/n:.1f}ms "
          f"min={min(times_ms):.1f}ms max={max(times_ms):.1f}ms "
          f"all={[round(t,1) for t in times_ms]}", flush=True)
    return times_ms


def main():
    from settings import Settings
    cfg = Settings("config.yaml")
    swap_model = cfg.swap_model
    print(f"[diag] swap_model={swap_model} provider={cfg.provider}", flush=True)

    ab.init_pipeline(cfg.provider, swap_model, "None", "None")

    from roop.utilities import get_device
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
    p = FaceSwapInsightFace()
    p.Initialize({"devicename": get_device(), "swap_model": swap_model})
    print(f"[diag] loaded_model_key={p.loaded_model_key} "
          f"trt_disabled_initially={p._trt_disabled} "
          f"batch_unsupported_initially={p._batch_unsupported}", flush=True)

    harjot = load_library_faceset("harjot")
    source_face = harjot.faces[0]

    video = os.path.join("G:/pinokio/roop-keep/single", "s1.mp4")
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read a frame from {video}"
    faces = get_all_faces(frame) or []
    assert faces, "no face detected in s1.mp4 frame 0"
    target_face = faces[0]

    size = p.model_output_size
    crop, _M = align_crop(frame, target_face.kps, size, p.model_template)
    blob = PixelBoostMixin.prepare_crop_frame(None, crop, p)
    print(f"[diag] crop shape={crop.shape} blob shape={blob.shape}", flush=True)

    print("\n[diag] === PHASE 1: fresh session, single-frame Run() only ===", flush=True)
    before = _timed_run(p, source_face, target_face, blob, n=8, label="BEFORE batch failure")
    print(f"[diag] after phase 1: trt_disabled={p._trt_disabled}", flush=True)

    print("\n[diag] === PHASE 2: trigger a batch>1 call (expected to fail for hyperswap) ===", flush=True)
    try:
        p.RunBatch(source_face, target_face, [blob.copy(), blob.copy()])
        print("[diag] RunBatch(2) did NOT raise — batch>1 apparently works for this model; "
              "theory does not apply here.", flush=True)
    except Exception as e:
        print(f"[diag] RunBatch(2) raised directly (unexpected — should have been caught "
              f"internally and fallen back): {e!r}", flush=True)
    print(f"[diag] after phase 2: trt_disabled={p._trt_disabled} "
          f"batch_unsupported={p._batch_unsupported}", flush=True)

    print("\n[diag] === PHASE 3: single-frame Run() again, AFTER the batch failure ===", flush=True)
    after = _timed_run(p, source_face, target_face, blob, n=8, label="AFTER batch failure")

    mean_before = sum(before) / len(before)
    mean_after = sum(after) / len(after)
    ratio = mean_after / mean_before if mean_before else float('inf')
    print(f"\n[diag] === VERDICT ===", flush=True)
    print(f"[diag] mean before: {mean_before:.1f}ms   mean after: {mean_after:.1f}ms   "
          f"ratio: {ratio:.2f}x", flush=True)
    print(f"[diag] trt_disabled after batch failure: {p._trt_disabled}", flush=True)
    if p._trt_disabled and ratio > 1.3:
        print("[diag] THEORY CONFIRMED: single-frame Run() lost TensorRT acceleration "
              "as a side effect of an unrelated batch>1 shape failure.", flush=True)
    elif p._trt_disabled:
        print("[diag] trt_disabled flipped True, but single-frame timing did not slow down "
              "meaningfully — CUDA fallback may be cheap enough here after all.", flush=True)
    else:
        print("[diag] trt_disabled never flipped — theory does not apply to this run.", flush=True)


if __name__ == "__main__":
    main()
