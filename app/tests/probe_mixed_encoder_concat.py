"""Can segments from TWO different encoders be joined without loss? (the premise of "split-GOP NVENC + QSV")

    env/Scripts/python.exe tests/probe_mixed_encoder_concat.py [out.json]

Three segments of the same 60-frame synthetic clip are encoded with identical colour tags: encoder X, encoder Y,
encoder X again. They are joined four ways and each result is read back with ffprobe (-count_frames) and a full
decode (-v error). The control joins X + X (same encoder). Y is libx265 / libx264, standing in for QuickSync:
this machine's ffmpeg has no ``*_qsv`` and no Intel GPU is enabled, so a real NVENC + QSV pair is UNMEASURED.

Joins: ``mp4-concat-copy`` (the concat demuxer SegmentedVideoWriter and distributed_render use), ``annexb->mp4 hvc1``,
``annexb->mp4 hev1`` (re-wrap through a raw elementary stream), ``raw-annexb`` (the elementary stream itself).
The re-wrap arms lose 2 frames even in the same-encoder control (a raw stream carries no timestamps, so ffmpeg
re-derives them), so they are not a usable route in either direction; the rows that decide are ``mp4-concat-copy``
mixed vs control.
Result 2026-10-26 (ffmpeg 8.1.2): HEVC mixed concat-copy = 120/180 frames, exit 0, 71 decoder-error lines; its
control = 120/120, 0 errors. H.264 mixed (h264_nvenc + libx264) = 180/180, 0 errors.
Writes docs/perf/qsv_encode/mixed_encoder_concat.json by default.
"""
import json
import os
import subprocess
import sys
import tempfile

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary  # noqa: E402

FF, FP = ffmpeg_binary(), ffprobe_binary()
FRAMES = 60
SRC = ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2", "-frames:v", str(FRAMES), "-pix_fmt", "yuv420p",
       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
ENC = {
    "hevc_nvenc": ["-c:v", "hevc_nvenc", "-rc", "vbr", "-cq", "23", "-preset", "p5", "-tune", "hq"],
    "libx265": ["-c:v", "libx265", "-crf", "23", "-preset", "faster"],
    "h264_nvenc": ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "23", "-preset", "p5", "-tune", "hq"],
    "libx264": ["-c:v", "libx264", "-crf", "23", "-preset", "faster"],
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, creationflags=0x08000000 if os.name == "nt" else 0)


def has_encoder(name):
    return f" {name} " in (run([FF, "-hide_banner", "-encoders"]).stdout or b"").decode("utf-8", "replace")


def readback(path, fmt=None):
    pre = ["-f", fmt] if fmt else []
    probe = run([FP, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", *pre, path])
    dec = run([FF, "-v", "error", *pre, "-i", path, "-f", "null", "-"])
    try:
        frames = int((probe.stdout or b"").decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        frames = None
    errs = [ln for ln in (dec.stderr or b"").decode("utf-8", "replace").splitlines() if ln.strip()]
    return {"frames": frames, "decoder_error_lines": len(errs), "first_error": errs[0][:120] if errs else ""}


def joins(d, seg, ext, bsf, tag_name):
    out = {}
    lst = os.path.join(d, "list.txt")
    with open(lst, "w") as fh:
        for s in seg:
            fh.write("file '%s'\n" % s.replace("\\", "/"))
    p = os.path.join(d, "j_concat.mp4")
    r = run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", p])
    out["mp4-concat-copy"] = {"exit": r.returncode, **readback(p)}
    raws = []
    for i, s in enumerate(seg):
        raw = os.path.join(d, f"r{i}.{ext}")
        run([FF, "-v", "error", "-y", "-i", s, "-map", "0:v:0", "-c:v", "copy", "-bsf:v", bsf, "-f", ext, raw])
        raws.append(raw)
    joined = os.path.join(d, f"joined.{ext}")
    with open(joined, "wb") as fh:
        for raw in raws:
            fh.write(open(raw, "rb").read())
    out["raw-annexb"] = {**readback(joined, ext)}
    for tag in tag_name:
        p = os.path.join(d, f"j_{tag}.mp4")
        r = run([FF, "-v", "error", "-y", "-r", "30", "-f", ext, "-i", joined, "-c:v", "copy", "-tag:v", tag, p])
        out[f"annexb->mp4 {tag}"] = {"exit": r.returncode, **readback(p)}
    return out


def family(d, label, x, y, ext, bsf, tags):
    if not (has_encoder(x) and has_encoder(y)):
        return {"skipped": f"{x} or {y} missing from this ffmpeg"}
    seg = {}
    for name, enc in (("X1", x), ("Y", y), ("X2", x)):
        p = os.path.join(d, f"{label}_{name}.mp4")
        r = run([FF, "-v", "error", "-y", *SRC, *ENC[enc], p])
        if r.returncode:
            return {"skipped": f"{enc} failed to encode: {(r.stderr or b'').decode()[:120]}"}
        seg[name] = p
    return {"encoders": [x, y, x], "expected_frames": 3 * FRAMES,
            "control_same_encoder (X+X)": joins(d, [seg["X1"], seg["X2"]], ext, bsf, tags),
            "mixed (X+Y+X)": joins(d, [seg["X1"], seg["Y"], seg["X2"]], ext, bsf, tags)}


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(APP), "docs", "perf", "qsv_encode",
                                                                   "mixed_encoder_concat.json")
    result = {"ffmpeg": (run([FF, "-version"]).stdout or b"").decode().splitlines()[0], "frames_per_segment": FRAMES}
    with tempfile.TemporaryDirectory(prefix="mixenc_") as d:
        result["hevc"] = family(d, "hevc", "hevc_nvenc", "libx265", "hevc", "hevc_mp4toannexb", ["hvc1", "hev1"])
        d2 = os.path.join(d, "h264")
        os.makedirs(d2)
        result["h264"] = family(d2, "h264", "h264_nvenc", "libx264", "h264", "h264_mp4toannexb", ["avc1"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
