"""Measure the BGR -> encoder -> decoder -> BGR colour round-trip error.

The pipeline processes frames as bgr24 in host memory and hands them to an
ffmpeg rawvideo pipe.  Whatever matrix the encoder uses to go BGR->YUV must be
the matrix a player uses to go back, or every rendered pixel shifts.  Two
things decide that:

* the conversion applied on the way in (a filter, or swscale's implicit
  default), and
* the colour tags written into the container, which is what tells the player
  which matrix to invert.

This harness feeds a known chart through real ffmpeg encode/decode and reports
the error, so the arms can be compared as measurements rather than as
reasoning about filter graphs.  It is deliberately standalone: it builds the
argv itself instead of importing FFMPEG_VideoWriter, so it can measure the
"before" arm after the source has already been changed.

    python tests/measure_color_roundtrip.py

A perfect round trip is not reachable -- yuv420p is lossy in chroma and 8-bit
limited-range quantisation costs about 1/255 -- so read the arms against each
other, not against zero.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.ffmpeg_path import ffmpeg_binary  # noqa: E402


def chart(width: int = 1280, height: int = 720) -> np.ndarray:
    """A BGR chart weighted toward the colours this project actually cares about.

    Flat saturated primaries expose a matrix error most loudly, but skin tones
    are what the defect is reported as ("washed out / green skin"), so the
    lower half is a sweep through plausible skin BGR values.
    """
    frame = np.zeros((height, width, 3), np.uint8)
    bars = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
            (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0)]
    bar_w = width // len(bars)
    for i, colour in enumerate(bars):
        frame[:height // 2, i * bar_w:(i + 1) * bar_w] = colour
    # Skin sweep: BGR, light to deep, roughly along the human skin locus.
    skin = [(172, 196, 229), (150, 176, 214), (124, 150, 190), (98, 124, 165),
            (76, 100, 138), (58, 78, 112), (44, 60, 88), (32, 44, 66)]
    for i, colour in enumerate(skin):
        frame[height // 2:, i * bar_w:(i + 1) * bar_w] = colour
    return frame


def _run(cmd, stdin_data=None):
    proc = subprocess.run(cmd, input=stdin_data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("%s\n%s" % (" ".join(cmd),
                                       proc.stderr.decode(errors="replace")[-2000:]))
    return proc.stdout


# The arms.  `vf` is the filter chain applied to the incoming bgr24 rawvideo;
# `tags` are the stream colour tags written into the container.
ARMS = {
    # What ffmpeg_writer emitted before this change: a colourspace filter that
    # declares the input to be BT.601-625 and converts it to BT.709, on data
    # that was never YUV at all, and no output tags.
    "legacy": {
        "vf": "colorspace=bt709:iall=bt601-6-625:fast=1",
        "tags": [],
    },
    # No conversion, no tags: swscale's implicit bgr24->yuv420p default.
    "untagged": {
        "vf": None,
        "tags": [],
    },
    # The odd-dimension branch as it was: -vf scale REPLACED the colourspace
    # filter, so an odd-sized render shipped with every colour tag unknown.
    "odd_legacy": {
        "vf": "scale=1278:718",
        "tags": [],
    },
    # The fix: chain the scale AHEAD of the colourspace conversion so an odd
    # size keeps the same conversion and tagging as an even one, and state the
    # tags explicitly instead of relying on filter metadata propagation.
    "odd_fixed": {
        "vf": "scale=1278:718,colorspace=bt709:iall=bt601-6-625:fast=1",
        "tags": ["-colorspace", "bt709", "-color_primaries", "bt709",
                 "-color_trc", "bt709", "-color_range", "tv"],
    },
    # Legacy conversion plus explicit tags: does stating them change anything?
    "legacy_tagged": {
        "vf": "colorspace=bt709:iall=bt601-6-625:fast=1",
        "tags": ["-colorspace", "bt709", "-color_primaries", "bt709",
                 "-color_trc", "bt709", "-color_range", "tv"],
    },
    # The fix: one explicit BGR->YUV conversion with a named matrix and range,
    # and container tags that name the same matrix back to the player.
    "bt709": {
        "vf": "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "tags": ["-colorspace", "bt709", "-color_primaries", "bt709",
                 "-color_trc", "bt709", "-color_range", "tv"],
    },
}


def roundtrip(frame: np.ndarray, arm: str, codec: str, crf: int,
              tmpdir: str) -> np.ndarray:
    h, w = frame.shape[:2]
    spec = ARMS[arm]
    out = os.path.join(tmpdir, "arm_%s.mp4" % arm)
    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", "%dx%d" % (w, h), "-pix_fmt", "bgr24", "-r", "25",
           "-an", "-i", "-"]
    cmd += ["-vcodec", codec, "-crf", str(crf)]
    if spec["vf"]:
        cmd += ["-vf", spec["vf"]]
    cmd += spec["tags"]
    cmd += ["-pix_fmt", "yuv420p", out]
    # 8 identical frames: a single-frame file gives some decoders nothing to
    # settle on, and the measurement is of colour, not of temporal coding.
    _run(cmd, stdin_data=frame.tobytes() * 8)

    # Decode the way a player would: no matrix override, so the container tags
    # (or their absence) decide how the YUV is inverted.  That is the whole
    # question -- overriding it here would measure the encoder in isolation and
    # hide the tagging half of the defect.
    raw = _run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
                "-i", out, "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    resized = scale_size(spec["vf"])
    if resized:
        w, h = resized
    return np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3)


def resampled_reference(frame: np.ndarray, vf: str) -> np.ndarray:
    """The chart through this arm's GEOMETRY only, with no encode and no colour op.

    An arm that rescales cannot be compared against the original chart -- the
    resample alone moves pixels.  Running the same scale with no encoder and no
    colour conversion isolates the colour question from the geometry question,
    so two arms that share a scale can be read against each other.
    """
    h, w = frame.shape[:2]
    geometry = ",".join(part for part in vf.split(",") if part.startswith("scale="))
    raw = _run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", "%dx%d" % (w, h), "-pix_fmt", "bgr24", "-r", "25",
                "-an", "-i", "-", "-frames:v", "1", "-vf", geometry,
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
               stdin_data=frame.tobytes())
    tw, th = scale_size(vf)
    return np.frombuffer(raw[:tw * th * 3], np.uint8).reshape(th, tw, 3)


def scale_size(vf):
    """(w, h) if this filter chain resizes to a fixed size, else None.

    `scale=` also carries non-geometric options in these arms
    (`scale=out_color_matrix=bt709`), which do NOT resize -- so the test is a
    numeric w:h, not the presence of the filter name.
    """
    for part in (vf or "").split(","):
        if not part.startswith("scale="):
            continue
        head = part.split("=", 1)[1].split(":")
        if len(head) >= 2 and head[0].isdigit() and head[1].isdigit():
            return int(head[0]), int(head[1])
    return None


def probe_tags(path: str) -> str:
    try:
        probe = os.path.join(os.path.dirname(ffmpeg_binary()), "ffprobe")
        if not (os.path.isfile(probe) or os.path.isfile(probe + ".exe")):
            probe = "ffprobe"
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_space,color_primaries,color_transfer,color_range",
             "-of", "default=nw=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        return out.decode(errors="replace").strip().replace("\n", " ")
    except Exception:
        return "(ffprobe unavailable)"


def compare_readers(path: str) -> int:
    """Which matrix does each READER invert a real file with?

    The writer half above is only half the round trip.  This asks the other
    half of the question against a real clip: cv2.VideoCapture and the ffmpeg
    pipe are the two readers this pipeline can use, and if they disagree the
    rendered colour depends on which one a given file happened to get.

    Measured on a bt709-tagged 1280x720 clip (b1.mp4), first frame, against
    cv2 as the reference:

        ffmpeg pipe, tag-honouring        mean 0.5113  max 10
        ffmpeg pipe, in_color_matrix=601  mean 0.0000  max  0   <- cv2 IS this
        ffmpeg pipe, in_color_matrix=709  mean 0.5113  max 10

    i.e. cv2 inverts a BT.709-tagged stream with the BT.601 matrix.  The pipe
    already reads the stream's own tag, so `scale=in_color_matrix=auto` on the
    pipe is a no-op -- it was measured byte-identical to the shipping command.
    """
    import cv2  # local: the writer half of this harness needs no OpenCV

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, cv_frame = cap.read()
    cap.release()
    if not ok:
        print("cv2 could not read %s" % path)
        return 1
    print("clip %s  %dx%d" % (os.path.basename(path), w, h))
    print("source tags: %s" % probe_tags(path))

    def pipe(vf=None, hwaccel=None):
        cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]
        cmd += ["-noautorotate", "-i", path, "-fps_mode", "passthrough"]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-an", "-sn", "pipe:1"]
        raw = _run(cmd)
        return np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3)

    ref = cv_frame.astype(np.int16)
    arms = [
        ("pipe, software, tag-honouring", None, None),
        ("pipe, software, forced bt601", "scale=in_color_matrix=bt601,format=bgr24", None),
        ("pipe, software, forced bt709", "scale=in_color_matrix=bt709,format=bgr24", None),
        ("pipe, NVDEC, tag-honouring", None, "cuda"),
        ("pipe, NVDEC, in_matrix=auto", "scale=in_color_matrix=auto,format=bgr24", "cuda"),
    ]
    print("%-32s %14s %6s" % ("reader arm", "mean|d| vs cv2", "max"))
    for label, vf, hw in arms:
        try:
            frame = pipe(vf, hw)
        except Exception as exc:
            print("%-32s  FAILED (%s)" % (label, str(exc).splitlines()[0][:60]))
            continue
        delta = np.abs(frame.astype(np.int16) - ref)
        print("%-32s %14.4f %6d" % (label, delta.mean(), delta.max()))
    print("\nAn arm reading 0.0000 is the matrix cv2 itself used.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--arms", default="legacy,untagged,bt709,legacy_tagged,odd_legacy,odd_fixed")
    ap.add_argument("--readers", metavar="VIDEO",
                    help="instead of the writer arms, compare what each READER "
                         "decodes this real clip to (see compare_readers)")
    args = ap.parse_args()

    if args.readers:
        return compare_readers(args.readers)

    src = chart()
    ref = src.astype(np.int16)
    # Interior-only comparison: a bar boundary lands on a chroma-subsampled
    # edge, so including it measures 4:2:0 ringing rather than the matrix.
    h, w = src.shape[:2]
    bar_w = w // 8
    cols = np.concatenate([np.arange(i * bar_w + 8, (i + 1) * bar_w - 8)
                           for i in range(8)])
    rows = np.concatenate([np.arange(8, h // 2 - 8),
                           np.arange(h // 2 + 8, h - 8)])
    sel = np.ix_(rows, cols)

    print("chart %dx%d, codec %s crf %d" % (w, h, args.codec, args.crf))
    print("%-10s %10s %10s %10s   %s"
          % ("arm", "mean|d|", "max|d|", "skin|d|", "container tags"))
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm in args.arms.split(","):
            arm = arm.strip()
            if not arm:
                continue
            vf = ARMS[arm]["vf"] or ""
            if scale_size(vf):
                # Geometry-matched reference: this arm rescales, so the original
                # chart is the wrong yardstick (see resampled_reference).
                arm_ref = resampled_reference(src, vf).astype(np.int16)
                ah, aw = arm_ref.shape[:2]
                abar = aw // 8
                acols = np.concatenate([np.arange(i * abar + 8, (i + 1) * abar - 8)
                                        for i in range(8)])
                arows = np.concatenate([np.arange(8, ah // 2 - 8),
                                        np.arange(ah // 2 + 8, ah - 8)])
                arm_sel, skin_lo, skin_hi = np.ix_(arows, acols), ah // 2 + 8, ah - 8
            else:
                arm_ref, arm_sel, skin_lo, skin_hi = ref, sel, h // 2 + 8, h - 8
            back = roundtrip(src, arm, args.codec, args.crf, tmp).astype(np.int16)
            delta = np.abs(back - arm_ref)
            skin = np.abs(back[skin_lo:skin_hi] - arm_ref[skin_lo:skin_hi])
            results[arm] = (float(delta[arm_sel].mean()), int(delta[arm_sel].max()),
                            float(skin.mean()))
            print("%-10s %10.3f %10d %10.3f   %s"
                  % (arm, results[arm][0], results[arm][1], results[arm][2],
                     probe_tags(os.path.join(tmp, "arm_%s.mp4" % arm))))

    if "legacy" in results and "bt709" in results:
        a, b = results["legacy"][0], results["bt709"][0]
        print("\nbt709 vs legacy: mean |delta| %.3f -> %.3f (%+.1f%%)"
              % (a, b, (b - a) / a * 100.0 if a else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
