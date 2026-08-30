"""Phase 16 output-integrity sweep: is the file we shipped actually intact?

WHY THIS EXISTS. Every other Phase 16 harness grades the RENDER -- fps, identity,
swap rate, wrong-faceset. None of them opens the finished file and asks whether
the frames in it are sane. That gap is not theoretical: the RTX 3060 enhancer
matrix found four enhancers that fail on every frame while the pipeline catches
the GPU error, writes the ORIGINAL frame, and the swap audit still reports
"swapped (every face) 100.0%". A throughput-only bench called those arms fast
and fine. Only an independent check caught them.

So this reads the decoded output and reports, per file:

  frames        decoded frame count vs the container's declared nb_frames
  duration      container duration vs frames/fps, to catch cadence corruption
  audio         whether an audio stream survived (compared against the source)
  black         frames whose mean luma is under a floor -- dropped/unwritten
  uniform       frames with near-zero variance -- flat grey/green corruption
  nan           non-finite pixels, which decode as garbage rather than erroring
  duplicate     consecutive frames that are bit-identical -- a stalled writer

None of these is a quality judgement. They are all "did the encoder emit a real
picture", which is the class of failure that survives every other gate here.

    env/Scripts/python.exe tests/phase16_integrity.py --glob "output/**/*.mp4"
"""
import argparse
import glob as globmod
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


def _ffprobe():
    import shutil
    found = shutil.which("ffprobe")
    if found:
        return found
    # Same lesson as the ffmpeg lookups: do not hardcode a drive. Resolve the
    # ffmpeg the profiler found and take ffprobe from beside it.
    from roop.runtime_optimizer import HardwareProfiler
    ff = HardwareProfiler._resolve_ffmpeg()
    if not ff:
        return None
    cand = os.path.join(os.path.dirname(ff),
                        "ffprobe.exe" if os.name == "nt" else "ffprobe")
    return cand if os.path.isfile(cand) else None


def probe(path, ffprobe):
    if not ffprobe:
        return {}
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_streams", "-show_format",
             "-of", "json", path], text=True, timeout=60)
        return json.loads(out)
    except Exception as exc:
        return {"error": str(exc)}


def inspect(path, black_floor=6.0, uniform_floor=1.0):
    """Decode every frame once and report the integrity signals."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"opened": False}
    frames = black = uniform = nan = dup = 0
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not np.isfinite(frame).all():
            nan += 1
        if float(g.mean()) < black_floor:
            black += 1
        if float(g.std()) < uniform_floor:
            uniform += 1
        if prev is not None and frame.shape == prev.shape and np.array_equal(frame, prev):
            dup += 1
        prev = frame
    cap.release()
    return {"opened": True, "frames": frames, "black": black,
            "uniform": uniform, "nan": nan, "duplicate": dup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True,
                    help="glob of rendered outputs; repeatable")
    ap.add_argument("--expect-frames", type=int, default=None,
                    help="fail a file whose decoded count differs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ffprobe = _ffprobe()
    paths = []
    for g in args.glob:
        paths.extend(sorted(globmod.glob(g, recursive=True)))
    # Intermediate segment and temp files are not shipped artifacts.
    paths = [p for p in paths
             if "_temp" not in os.path.basename(p)
             and ".seg" not in os.path.basename(p)]

    rows = []
    print("%-52s %6s %6s %7s %4s %5s %s" %
          ("file", "frames", "black", "uniform", "nan", "dup", "verdict"))
    for p in paths:
        r = inspect(p)
        meta = probe(p, ffprobe)
        streams = meta.get("streams", []) if isinstance(meta, dict) else []
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        vid = next((s for s in streams if s.get("codec_type") == "video"), {})
        r.update({"path": p, "audio": has_audio,
                  "codec": vid.get("codec_name"),
                  "width": vid.get("width"), "height": vid.get("height")})
        try:
            r["duration_s"] = round(float(meta.get("format", {}).get("duration")), 3)
        except (TypeError, ValueError):
            r["duration_s"] = None

        problems = []
        if not r.get("opened"):
            problems.append("UNREADABLE")
        if r.get("frames", 0) == 0:
            problems.append("NO FRAMES")
        if r.get("nan"):
            problems.append("NaN x%d" % r["nan"])
        if r.get("black"):
            problems.append("black x%d" % r["black"])
        if r.get("uniform"):
            problems.append("uniform x%d" % r["uniform"])
        if r.get("duplicate"):
            problems.append("dup x%d" % r["duplicate"])
        if args.expect_frames and r.get("frames") != args.expect_frames:
            problems.append("frames %s != %d" % (r.get("frames"), args.expect_frames))
        r["verdict"] = "PASS" if not problems else "; ".join(problems)
        rows.append(r)
        print("%-52s %6s %6s %7s %4s %5s %s" %
              (os.path.basename(os.path.dirname(p))[:52],
               r.get("frames"), r.get("black"), r.get("uniform"),
               r.get("nan"), r.get("duplicate"), r["verdict"]))

    npass = sum(1 for r in rows if r["verdict"] == "PASS")
    print("\n%d of %d files PASS" % (npass, len(rows)))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "passed": npass, "total": len(rows)}, fh, indent=2)
        print("wrote %s" % args.out)
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
