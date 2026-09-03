"""Per-call cost of the three roop/temporal_smoother.py mechanisms.

WHAT THIS DOES AND DOES NOT ESTABLISH. This is an ISOLATED microbenchmark: it
prices each operator at the sizes the pipeline actually hands it. It is not an
end-to-end result and must not be quoted as one. This project has repeatedly
measured a stage-level win that came back NEUTRAL in a render, because the
pipeline is GPU-bound and host work that fits in the shadow of a GPU stage
costs nothing at the wall clock (Session Log 2026-08-24 Part 8 section 0).

The useful direction is the negative one: a per-face cost far below the ~30 ms
an enhancer already spends on the GPU cannot produce a visible regression, and
one above it certainly can. Anything in between needs a counterbalanced
600-frame A/B, which lives in tests/ab_temporal_detection.py.

    env/Scripts/python.exe -m tests.bench_temporal_smoother
"""

import time

import cv2
import numpy as np

from roop.temporal_smoother import (AdaptiveLandmarkSmoother,
                                    HighFrequencyFlowStabilizer,
                                    boundary_illumination_match,
                                    soft_distance_matte)


def _time(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def _frame_matte(w, h):
    """A face-sized hull in a full frame -- the shape blur_area is handed."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(h * 0.18), int(h * 0.26)),
                0, 0, 360, 255, -1)
    return m


def main():
    rs = np.random.RandomState(0)
    print("roop/temporal_smoother.py -- isolated per-call cost")
    print("(one call = one FACE, except the matte rows which are one PASTE)\n")

    # 1. landmark smoothing -------------------------------------------------
    sm = AdaptiveLandmarkSmoother()
    kps = np.array([[300., 400.], [500., 400.], [400., 500.],
                    [330., 600.], [470., 600.]], dtype=np.float32)
    dense = rs.rand(106, 2).astype(np.float32) * 400.0
    counter = {'i': 0}

    def landmark_call():
        counter['i'] += 1
        sm.smooth(kps, dense, track_id=1, frame_index=counter['i'])

    print("1. AdaptiveLandmarkSmoother.smooth   5-pt + 106-pt")
    print("     %8.4f ms/face" % _time(landmark_call, n=2000, warmup=200))

    # 2. the boundary matte -------------------------------------------------
    print("\n2. soft_distance_matte   (replaces blur_area's erode+Gaussian)")
    for w, h in ((1280, 720), (1920, 1080), (3840, 2160)):
        matte = _frame_matte(w, h)
        new = _time(lambda: soft_distance_matte(matte, margin_px=12.0,
                                                as_uint8=True), n=60)

        def gaussian_ref(m=matte):
            """What blur_area does today, for the honest comparison: this is a
            REPLACEMENT, so its cost is the DIFFERENCE, not the total."""
            out = cv2.GaussianBlur(m, (3, 3), 0)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            out = cv2.erode(out, k, iterations=1)
            return cv2.GaussianBlur(out, (25, 25), 0)

        old = _time(gaussian_ref, n=60)
        print("     %4dx%-5d  distance %7.3f ms   gaussian %7.3f ms   "
              "delta %+7.3f ms" % (w, h, new, old, new - old))

    # 3. rim illumination ---------------------------------------------------
    print("\n3. boundary_illumination_match   (ROI-sized, opt-in, 0 = no-op)")
    for n in (192, 384, 768):
        paste = (rs.rand(n, n, 3) * 255).astype(np.uint8)
        target = (rs.rand(n, n, 3) * 255).astype(np.uint8)
        alpha = np.zeros((n, n), np.float32)
        cv2.circle(alpha, (n // 2, n // 2), n // 3, 1.0, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), n / 24.0)
        ms = _time(lambda: boundary_illumination_match(paste, target, alpha, 1.0),
                   n=60)
        print("     ROI %3dx%-4d  %7.3f ms" % (n, n, ms))

    # 4. HF carry -----------------------------------------------------------
    print("\n4. HighFrequencyFlowStabilizer.stabilize   (opt-in)")
    # The fixture has to be a PLAUSIBLE FACE SEQUENCE, not independent noise.
    # Eight unrelated random crops trip the flow-residual guard on every call,
    # so `applied` comes back 0 and the timing is the DECLINED path -- which
    # skips the warp that is most of the work. Measuring the branch that does
    # not run is the same error as pricing a config production never renders.
    # A slowly drifting blurred field is correlated frame to frame, so the
    # guard passes and the active path is what gets timed. `applied` is printed
    # for exactly that reason: it is the proof the number means anything.
    for n in (256, 512):
        hf = HighFrequencyFlowStabilizer()
        base = cv2.GaussianBlur((rs.rand(n, n, 3) * 255).astype(np.uint8),
                                (0, 0), 5.0)
        crops = []
        for i in range(16):
            M = np.float32([[1, 0, i * 0.7], [0, 1, i * 0.4]])   # gentle drift
            moved = cv2.warpAffine(base, M, (n, n),
                                   borderMode=cv2.BORDER_REPLICATE)
            grain = np.random.RandomState(100 + i).randn(n, n, 3) * 6.0
            crops.append(np.clip(moved.astype(np.float32) + grain, 0, 255)
                         .astype(np.uint8))
        state = {'i': 0}

        def hf_call():
            state['i'] += 1
            hf.stabilize(crops[state['i'] % 16], track_id=1,
                         frame_index=state['i'])

        ms = _time(hf_call, n=200, warmup=20)
        s = hf.stats()
        active = 100.0 * s['applied'] / max(1, s['applied'] + s['reset_residual']
                                            + s['skipped_noncontiguous'])
        print("     crop %3dx%-4d  %7.3f ms/face   (%.0f%% of calls took the "
              "ACTIVE path -- below ~90%% the number is not the cost of the "
              "filter)" % (n, n, ms, active))

    print("\nReference: the cheapest restorer in this repo (GPEN 256) is "
          "5.3 ms/face\n           and the configured default is ~30 ms/face, "
          "both on the GPU.")


if __name__ == '__main__':
    main()
