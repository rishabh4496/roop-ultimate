"""Which cv2 blur / erode / dilate calls does a real render make, on what, at what cost?

Wraps ``cv2.GaussianBlur`` / ``erode`` / ``dilate`` around one production render and prints
call counts, image size, kernel and time per call. It is how the full-frame matte passes were
found (2026-10-03, 1080p, one face, shipped config, 90 frames)::

    GaussianBlur k105-113  1920x1080 uint8   ~77 ms/call   (per-call time under 20 workers)
    dilate 39-43           1920x1080 uint8   ~43 ms/call
    erode  27-29           1920x1080 uint8   ~19 ms/call
    total ~224 CPU-ms per frame

``roop.mask_roi`` now runs those on the matte's support only; compare with ``ROOP_MASK_ROI=0``.

Run (app stopped; ~3 min)::

    app\\env\\Scripts\\python.exe tools/profile_cv2_mask_ops.py [--frames 90] [--out ops.json]

Needs the media folder of tests/test_performance_regression.py (``ROOP_KEEP_DIR``).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "tests"), str(REPO / "app"), str(REPO / "app" / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import cv2

    import test_performance_regression as T

    tmp = Path(tempfile.mkdtemp(prefix="cv2ops_"))
    rig = T.Rig(tmp)             # pins ROOP_STAB_CHUNK_MB (render output depends on free RAM)
    rg = rig.rg
    clip = T.cut_with_audio(T.media_dir() / "single" / "s3.mp4", tmp / "c.mp4", args.frames)
    target, _ = rg.capture_target(str(clip))
    rig.render(T.cut_with_audio(clip, tmp / "w.mp4", 12), target, tmp / "warm")   # engines + init

    lock = threading.Lock()
    acc = collections.defaultdict(lambda: [0, 0.0])

    def wrap(name, describe):
        orig = getattr(cv2, name)

        def timed(*a, **k):
            t = time.perf_counter()
            result = orig(*a, **k)
            dt = time.perf_counter() - t
            key = (name,) + describe(a, k)
            with lock:
                acc[key][0] += 1
                acc[key][1] += dt
            return result
        setattr(cv2, name, timed)
        return orig

    def blur_key(a, k):
        size = a[1] if len(a) > 1 else k.get("ksize")
        img = a[0]
        return f"k{size[0]}", f"{img.shape[1]}x{img.shape[0]}", str(img.dtype)

    def morph_key(a, k):
        kernel = a[1] if len(a) > 1 else k.get("kernel")
        img = a[0]
        return f"k{kernel.shape[0]}x{kernel.shape[1]}", f"{img.shape[1]}x{img.shape[0]}", str(img.dtype)

    originals = {"GaussianBlur": wrap("GaussianBlur", blur_key),
                 "erode": wrap("erode", morph_key), "dilate": wrap("dilate", morph_key)}
    try:
        t = time.perf_counter()
        rig.render(clip, target, tmp / "r")
        wall = time.perf_counter() - t
    finally:
        for name, orig in originals.items():
            setattr(cv2, name, orig)
        rig.close()

    rows = sorted(({"op": k[0], "kernel": k[1], "image": k[2], "dtype": k[3], "calls": n,
                    "total_ms": t * 1e3, "ms_per_call": t * 1e3 / n}
                   for k, (n, t) in acc.items()), key=lambda r: -r["total_ms"])
    total = sum(r["total_ms"] for r in rows)
    print(f"render wall {wall:.1f} s, {args.frames} frames; cv2 blur/morphology "
          f"{total:.0f} CPU-ms = {total / args.frames:.1f} per frame")
    for r in rows[:14]:
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
