"""Phase 4 probe: can a GOP slice with no verified target frame be STREAM-COPIED instead of rendered?

Two measurements, both needed before such a bypass is written:

1. COVERAGE. Does "this GOP holds no `stage2_frame_plan` frame" imply "this GOP holds no swap"?
   Reads the per-frame plan/swap sets that `probe_stage3_plan_coverage.py --out` writes
   (`plan_frame_list`, `swapped_frame_list`) and scores them per GOP of the same clip (keyframes read
   from the packets). A swap inside a zero-plan GOP is a swap the bypass would delete.

2. SPLICE. Can a stream-copied source GOP be joined (`concat -c copy`) with a re-encoded GOP at all?
   Copies the first GOP of the clip, re-encodes the next one with each encoder in `--encoders`, joins
   them, decodes the join with `framemd5` and compares each half with its standalone decode. The
   decoder-visible outcome is what is scored: the exit code and stderr of the concat are not (see the
   hevc row of docs/perf/stage4_gop).

    env/Scripts/python.exe tests/probe_gop_bypass_stage4.py --clip clip.mp4 --probe plan_cov.json \
        --encoders h264_nvenc hevc_nvenc libx264 --out result.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary  # noqa: E402


def _run(*cmd):
    return subprocess.run(list(cmd), capture_output=True, text=True)


def keyframes(clip):
    flags = _run(ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "packet=flags", "-of", "csv=p=0", clip).stdout.split()
    return [i for i, f in enumerate(flags) if "K" in f], len(flags)


def coverage(clip, probe_json):
    keys, n = keyframes(clip)
    data = json.load(open(probe_json, encoding="utf-8"))
    plan, swapped = set(data["plan_frame_list"]), set(data["swapped_frame_list"])
    rows = []
    for a, b in zip(keys, keys[1:] + [n]):
        frames = set(range(a, b))
        rows.append({"start": a, "frames": b - a, "plan": len(frames & plan), "swapped": len(frames & swapped)})
    zero = [r for r in rows if r["plan"] == 0]
    return {
        "frames": n, "gops": len(rows), "plan_frames": len(plan), "swap_frames": len(swapped),
        "swap_frames_not_in_plan": len(swapped - plan),
        "zero_plan_gops": len(zero), "zero_plan_frames": sum(r["frames"] for r in zero),
        "swaps_inside_zero_plan_gops": sum(r["swapped"] for r in zero), "per_gop": rows,
    }


def _md5(path):
    r = _run(ffmpeg_binary(), "-v", "error", "-i", path, "-map", "0:v:0", "-f", "framemd5", "-")
    return [l.split(",")[-1].strip() for l in r.stdout.splitlines() if l and not l.startswith("#")], r.stderr.strip()


def splice(clip, encoders):
    keys, n = keyframes(clip)
    if len(keys) < 2:
        return {"error": "clip has a single GOP"}
    first, second_end = keys[1], (keys[2] if len(keys) > 2 else n)
    rate = _run(ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=avg_frame_rate,codec_name", "-of", "csv=p=0", clip).stdout.strip().split(",")
    num, den = rate[1].split("/") if "/" in rate[1] else (rate[1], "1")
    fps = int(num) / int(den)
    result = {"source_codec": rate[0], "copied_frames": first, "reencoded_frames": second_end - first, "rows": []}
    work = tempfile.mkdtemp(prefix="gop_splice_")
    try:
        a = os.path.join(work, "a_copy.mp4")
        src = os.path.join(work, "b_src.mp4")
        _run(ffmpeg_binary(), "-v", "error", "-y", "-i", clip, "-map", "0:v:0", "-an", "-frames:v", str(first),
             "-c:v", "copy", "-avoid_negative_ts", "make_zero", a)
        _run(ffmpeg_binary(), "-v", "error", "-y", "-ss", str(first / fps), "-i", clip, "-map", "0:v:0", "-an",
             "-frames:v", str(second_end - first), "-c:v", "copy", "-avoid_negative_ts", "make_zero", src)
        md5_a, _ = _md5(a)
        for enc in encoders:
            b = os.path.join(work, f"b_{enc}.mp4")
            r = _run(ffmpeg_binary(), "-v", "error", "-y", "-i", src, "-c:v", enc, "-b:v", "4M", b)
            if r.returncode:
                result["rows"].append({"encoder": enc, "error": r.stderr.strip()[-160:]})
                continue
            listing = os.path.join(work, f"l_{enc}.txt")
            with open(listing, "w", encoding="utf-8") as fh:
                for p in (a, b):
                    fh.write("file '%s'\n" % p.replace("\\", "/"))
            joined = os.path.join(work, f"j_{enc}.mp4")
            cat = _run(ffmpeg_binary(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
                       "-map", "0:v:0", "-c:v", "copy", joined)
            md5_b, _ = _md5(b)
            md5_j, decode_err = _md5(joined)
            result["rows"].append({
                "encoder": enc, "concat_rc": cat.returncode, "concat_stderr": cat.stderr.strip()[:120],
                "joined_frames": len(md5_j), "wanted_frames": len(md5_a) + len(md5_b),
                "copied_half_identical": md5_j[:len(md5_a)] == md5_a,
                "reencoded_half_identical": md5_j[len(md5_a):] == md5_b,
                "decode_stderr": decode_err[:120],
            })
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--probe", help="probe_stage3_plan_coverage.py --out JSON for this clip")
    ap.add_argument("--encoders", nargs="*", default=["h264_nvenc", "hevc_nvenc", "libx264"])
    ap.add_argument("--out")
    args = ap.parse_args()
    out = {"clip": os.path.basename(args.clip)}
    if args.probe:
        out["coverage"] = coverage(args.clip, args.probe)
    out["splice"] = splice(args.clip, args.encoders)
    for key in ("coverage", "splice"):
        if key in out:
            view = {k: v for k, v in out[key].items() if k != "per_gop"}
            print("PROBE", key, json.dumps(view))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
