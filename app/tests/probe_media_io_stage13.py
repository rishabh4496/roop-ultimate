"""Does the video decode / encode side wait on the GPU, and would pinned host memory change it? Measured on the app's own reader/writer.

    env/Scripts/python.exe tests/probe_media_io_stage13.py --section decode|encode|pinned [--frames 300]

Stage 13 asked for pinned (page-locked) frame buffers read with `readinto`, a dedicated H2D/D2H copy stream, tuned NVDEC/NVENC
flags, and "decode and encode wait below 1.5 ms per frame at 1080p and 4K". This measures the thing it names.

  decode   `video_stream.open_video_capture` (the production NVDEC reader: pooled pre-sized buffer, `readinto`, prefetch
           thread) read() latency per frame: consumer IDLE (back-to-back = decode throughput) and PACED at the rate a real
           render consumes (default 70 ms/frame = the 13.6 fps stabilised render) = the time a worker actually WAITS.
  encode   `ffmpeg_writer.FFMPEG_VideoWriter.write_frame` latency per frame, idle and paced, with the configured codec.
  pinned   one ffmpeg NVDEC pipe read into (a) a pageable NumPy buffer and (b) a pinned torch tensor's memoryview, then a
           host->device copy of that frame from each: what page-locking buys per frame at this frame size.

Everything is reported as ms per frame (mean / p50 / p99). Arms are interleaved over 3 rounds, round 1 discarded.
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KEEP = os.environ.get("ROOP_MEDIA_DIR") or os.path.normpath(os.path.join(APP, "..", "..", "..", "roop-keep"))
CLIPS = [
    ("1080p", os.path.join(APP, "roop", "assets", "benchmark", "regression_solo_600f_1080p.mp4")),
    ("4K", os.path.join(KEEP, "final", "9155070-uhd_4096_2160_25fps.mp4")),
]


def stats(ms):
    ms = sorted(ms)
    return "%.2f / %.2f / %.2f" % (statistics.fmean(ms), ms[len(ms) // 2], ms[min(len(ms) - 1, int(len(ms) * 0.99))])


def geometry(path):
    c = cv2.VideoCapture(path)
    w, h, fps = int(c.get(3)), int(c.get(4)), c.get(5) or 30.0
    c.release()
    return w, h, fps


def read_times(path, frames, pace_s):
    from roop.video_stream import open_video_capture
    w, h, fps = geometry(path)
    cap = open_video_capture(path, w, h, fps, tag="stage13 probe")
    times = []
    try:
        for _ in range(frames):
            t0 = time.perf_counter()
            ok, fr = cap.read()
            times.append((time.perf_counter() - t0) * 1000.0)
            if not ok:
                break
            if pace_s:
                time.sleep(pace_s)
    finally:
        cap.release()
    return times[1:] if len(times) > 1 else times, type(cap).__name__


def section_decode(frames, pace_ms):
    for label, path in CLIPS:
        for rnd in range(3):
            idle, kind = read_times(path, frames, 0.0)
            paced, _ = read_times(path, min(frames, 120), pace_ms / 1000.0)
            if rnd:
                print("[decode] %-5s %-22s idle ms mean/p50/p99 %-24s  (%.0f fps)   paced %d ms: wait ms mean/p50/p99 %s" % (
                    label, kind, stats(idle), 1000.0 / statistics.fmean(idle), pace_ms, stats(paced)), flush=True)


def section_encode(frames, pace_ms):
    from roop.ffmpeg_writer import FFMPEG_VideoWriter
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    codec = str(getattr(cfg, "video_encoder", None) or "hevc_nvenc")
    quality = int(getattr(cfg, "video_quality", 14) or 14)
    for label, path in CLIPS:
        w, h, _ = geometry(path)
        cap = cv2.VideoCapture(path)
        src = []
        for _ in range(8):
            ok, fr = cap.read()
            if not ok:
                break
            src.append(fr)
        cap.release()
        for rnd in range(3):
            for mode, pace in (("idle", 0.0), ("paced", pace_ms / 1000.0)):
                n = frames if not pace else min(frames, 120)
                out = os.path.join(tempfile.gettempdir(), "stage13_encode_probe.mp4")
                wr = FFMPEG_VideoWriter(out, (w, h), 30.0, codec=codec, crf=quality)
                ms = []
                for i in range(n):
                    t0 = time.perf_counter()
                    wr.write_frame(src[i % len(src)])
                    ms.append((time.perf_counter() - t0) * 1000.0)
                    if pace:
                        time.sleep(pace)
                wr.close()
                if rnd:
                    print("[encode] %-5s codec=%-11s %-5s write_frame ms mean/p50/p99 %s" % (
                        label, wr.codec, mode, stats(ms[1:])), flush=True)
                try:
                    os.remove(out)
                except OSError:
                    pass


def ffmpeg_bin():
    from roop.ffmpeg_writer import FFMPEG_BINARY
    return FFMPEG_BINARY


def pipe_read(path, w, h, frames, sink):
    """Read `frames` bgr24 frames from an NVDEC ffmpeg pipe with readinto into sink(); ms per frame."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-hwaccel", "cuda", "-i", path,
           "-frames:v", str(frames), "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
    need, ms = w * h * 3, []
    try:
        while True:
            mv = sink()
            t0 = time.perf_counter()
            n = 0
            while n < need:
                k = proc.stdout.readinto(mv[n:])
                if not k:
                    return ms[1:]
                n += k
            ms.append((time.perf_counter() - t0) * 1000.0)
    finally:
        proc.kill()
        proc.wait()


def section_pinned(frames):
    import torch
    assert torch.cuda.is_available()
    for label, path in CLIPS:
        w, h, _ = geometry(path)
        nbytes = w * h * 3
        pageable = np.empty((h, w, 3), np.uint8)
        pinned = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
        pv, pm = memoryview(pageable.reshape(-1)), memoryview(pinned.numpy().reshape(-1))
        dev = torch.empty((h, w, 3), dtype=torch.uint8, device="cuda")
        pageable_t = torch.from_numpy(pageable)

        def h2d(src, reps=60):
            torch.cuda.synchronize()
            out = []
            for _ in range(reps):
                t0 = time.perf_counter()
                dev.copy_(src, non_blocking=True)
                torch.cuda.synchronize()
                out.append((time.perf_counter() - t0) * 1000.0)
            return out[5:]
        for rnd in range(3):
            a = pipe_read(path, w, h, frames, lambda: pv)
            b = pipe_read(path, w, h, frames, lambda: pm)
            ha, hb = h2d(pageable_t), h2d(pinned)
            if rnd:
                print("[pinned] %-5s %.1f MB/frame | pipe readinto ms mean/p50/p99: pageable %s  pinned %s | H2D copy ms: pageable %s  pinned %s" % (
                    label, nbytes / 1e6, stats(a), stats(b), stats(ha), stats(hb)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=("decode", "encode", "pinned"), required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--pace-ms", type=float, default=70.0)
    args = ap.parse_args()
    if args.section == "decode":
        section_decode(args.frames, args.pace_ms)
    elif args.section == "encode":
        section_encode(args.frames, args.pace_ms)
    else:
        section_pinned(args.frames)


if __name__ == "__main__":
    main()
