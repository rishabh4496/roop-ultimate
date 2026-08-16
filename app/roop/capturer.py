from typing import Optional
from collections import OrderedDict
import json
import os
import subprocess
import threading
import cv2
import numpy as np

from roop.typing import Frame

# ── HEVC / long-GOP fallback ─────────────────────────────────────────────────
# cv2.VideoCapture is the fast path and stays the default, but on some long
# H.265 files its seek is unreliable: CAP_PROP_POS_FRAMES lands nowhere and
# read() returns nothing, or CAP_PROP_FRAME_COUNT comes back as 0. Both fail
# SILENTLY — a zero frame count clamps every request to frame 0, so the preview
# and the whole timeline show the first frame of the clip forever, and a failed
# read just yields a blank preview. The batch pipeline already routes around
# this through the ffmpeg pipe reader (nvdec_reader); preview/timeline did not.
#
# So: ask ffprobe for the real geometry when cv2's is unusable, and read through
# the same ffmpeg pipe when cv2 cannot produce a frame. The switch is per file
# and only latches once ffmpeg has PROVEN it can read a frame cv2 could not —
# a read failing at the end of a clip is normal and must not condemn the file.
_pipe_reader = None      # FFmpegVideoReader currently open, or None
_pipe_path = None        # the file it is reading
_pipe_next = None        # frame index its next read() will return
_pipe_needed = set()     # files where cv2 proved unreliable (sticky)
_probe_cache = {}        # path -> {"w","h","fps","frames"} | None

current_video_path = None
current_frame_total = 0
current_capture = None
# Frame index the next read() on current_capture will return, or None if unknown.
# Lets sequential requests (playback, forward scrubbing) skip the expensive
# CAP_PROP_POS_FRAMES seek — for long-GOP codecs (H.264/HEVC) a random seek
# re-decodes from the nearest keyframe and can cost seconds per frame, whereas
# reading the next frame in order is near real-time.
current_next_pos = None

# Serialises all access to the shared VideoCapture and animated-WebP cache.
# cv2.VideoCapture is NOT thread-safe: concurrent reads from multiple Gradio
# worker threads corrupt FFmpeg's internal codec context, triggering:
#   "Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:173"
_capture_lock = threading.Lock()

# PIL-based cache for animated webp (PIL is already in the env via clip)
_awebp_path = None
_awebp_frames = None   # list of BGR numpy arrays, populated lazily


# ── Decoded-frame cache ──────────────────────────────────────────────────────
# A random seek costs 40-200 ms on long-GOP video (measured: median 80 ms on a
# 720p H.264 clip, worse at 4K) while a sequential read costs ~2 ms. The preview
# UI asks for the SAME frame several times over: /api/target/preview draws the
# raw frame, /api/preview decodes it again for the swap, and the timeline's hover
# thumbnail asks for it a third time. Each of those arrives after the decoder has
# already moved past the frame, so every one of them paid a full seek — three
# seeks to show one frame, and scrubbing backwards paid a seek per frame even
# over ground just covered.
#
# So keep the decoded frames themselves, keyed by (path, mtime, requested index),
# under a byte budget. Lookups take their own lock and happen BEFORE the capture
# lock: a hit must not queue behind another thread's 200 ms seek.
_frame_cache = OrderedDict()     # key -> BGR array
_frame_cache_bytes = 0
_frame_cache_lock = threading.Lock()


def _cache_budget() -> int:
    """Bytes of decoded frames to retain. ~192 MB holds ~30 frames of 1080p or
    ~8 of 4K, which is the working set of a scrub."""
    try:
        mb = int(os.environ.get("ROOP_FRAME_CACHE_MB", "192"))
    except ValueError:
        mb = 192
    return max(0, mb) * 1024 * 1024


def _cache_key(video_path: str, frame_number: int):
    """Keyed on mtime as well as path so a file rewritten in place (a re-encode,
    a re-download) can never serve frames from the old one. None means "don't
    cache" — the file vanished, and the decode below will fail anyway."""
    try:
        return (video_path, os.path.getmtime(video_path), int(frame_number))
    except OSError:
        return None


def _cache_get(key):
    """The cached frame, or None. Returns a COPY — see _cache_put."""
    if key is None:
        return None
    with _frame_cache_lock:
        frame = _frame_cache.get(key)
        if frame is None:
            return None
        _frame_cache.move_to_end(key)
        return frame.copy()


def _cache_put(key, frame):
    """Store a frame. The cache keeps its OWN copy and hands out copies, so a
    caller and the cache never share an array.

    That costs a memcpy per decode (~1 ms at 1080p, against the 40-200 ms seek
    this whole cache exists to avoid) and it is not optional: roop's preview path
    draws mask outlines and face boxes onto the frame it is handed, so a shared
    array would let one preview burn its overlay into every later reader's copy
    of that frame — corruption that surfaces long after, on a frame nobody was
    touching at the time."""
    global _frame_cache_bytes
    if key is None or frame is None:
        return
    budget = _cache_budget()
    size = int(frame.nbytes)
    if budget <= 0 or size > budget:
        return
    frame = frame.copy()
    with _frame_cache_lock:
        old = _frame_cache.pop(key, None)
        if old is not None:
            _frame_cache_bytes -= int(old.nbytes)
        _frame_cache[key] = frame
        _frame_cache_bytes += size
        while _frame_cache_bytes > budget and len(_frame_cache) > 1:
            _, evicted = _frame_cache.popitem(last=False)
            _frame_cache_bytes -= int(evicted.nbytes)


def clear_frame_cache():
    """Drop every cached frame. Nothing in the normal flow needs this — the byte
    budget bounds the cache and the key covers file identity — so it exists for
    tests and for anywhere that wants the memory back deliberately."""
    global _frame_cache_bytes
    with _frame_cache_lock:
        _frame_cache.clear()
        _frame_cache_bytes = 0


def frame_cache_stats():
    """(entries, bytes) currently held."""
    with _frame_cache_lock:
        return len(_frame_cache), _frame_cache_bytes


def _walk_max() -> int:
    """How many frames to decode-and-discard forward rather than seeking.

    Measured on a 720p H.264 clip: cv2's CAP_PROP_POS_FRAMES seek costs a flat
    ~125-180 ms whatever the distance, while grab() (decode, skip conversion) is
    0.67 ms/frame — so walking wins out to roughly 190 frames. Scrub drags and
    frame-stepping move a handful of frames at a time and were paying the full
    seek for every one of them. 90 keeps a wide margin under break-even for
    files whose seek is cheaper than this one's."""
    try:
        return max(0, int(os.environ.get("ROOP_SEEK_WALK", "90")))
    except ValueError:
        return 90


def _pipe_walk_max() -> int:
    """Same idea for the ffmpeg-pipe reader, but its skip is a FULL decode plus a
    pipe transfer (no cheap grab()), and the alternative — respawning ffmpeg with
    an accurate input seek — costs ~110 ms rather than cv2's ~150 ms. So the
    break-even sits much lower."""
    try:
        return max(0, int(os.environ.get("ROOP_PIPE_WALK", "24")))
    except ValueError:
        return 24


def _load_animated_webp(video_path: str):
    """Load all frames of an animated WebP into memory as BGR arrays.

    Must be called while holding _capture_lock."""
    global _awebp_path, _awebp_frames
    if _awebp_path == video_path and _awebp_frames is not None:
        return
    from PIL import Image
    frames = []
    with Image.open(video_path) as img:
        for i in range(getattr(img, "n_frames", 1)):
            img.seek(i)
            frame_rgb = np.array(img.convert("RGB"))
            frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    _awebp_path = video_path
    _awebp_frames = frames


def _popen_kwargs():
    return {"creationflags": 0x08000000} if os.name == "nt" else {}


def _ffprobe_binary() -> str:
    """ffprobe sits next to ffmpeg in every build we ship with, so derive it from
    whatever FFMPEG_BINARY resolved to (bare name or absolute path)."""
    try:
        from roop.ffmpeg_writer import FFMPEG_BINARY
    except Exception:
        return "ffprobe"
    d = os.path.dirname(FFMPEG_BINARY)
    return os.path.join(d, "ffprobe") if d else "ffprobe"


def _parse_rate(text) -> float:
    """ffprobe rates are rationals like '30000/1001'."""
    try:
        s = str(text or "").strip()
        if "/" in s:
            num, den = s.split("/", 1)
            den = float(den)
            return float(num) / den if den else 0.0
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _probe_video(video_path: str):
    """Real width/height/fps/frame-count from ffprobe, cached per path.

    Used only when cv2's own numbers are missing or nonsensical. nb_frames is
    absent in plenty of containers, so fall back to duration x fps — an estimate,
    but a usable timeline beats a frame count of 0, which pins every request to
    frame 0."""
    if video_path in _probe_cache:
        return _probe_cache[video_path]
    info = None
    try:
        cmd = [_ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
               "-show_entries",
               "stream=width,height,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt"
               ":format=duration",
               "-of", "json", video_path]
        proc = subprocess.run(cmd, capture_output=True, timeout=30, **_popen_kwargs())
        blob = json.loads((proc.stdout or b"{}").decode("utf-8", "replace"))
        st = (blob.get("streams") or [{}])[0]
        w = int(st.get("width") or 0)
        h = int(st.get("height") or 0)
        codec = str(st.get("codec_name") or "").lower()
        pix_fmt = str(st.get("pix_fmt") or "").lower()
        fps = _parse_rate(st.get("avg_frame_rate"))
        frames = int(st.get("nb_frames") or 0)
        if frames <= 0:
            dur = 0.0
            for src in (st.get("duration"), (blob.get("format") or {}).get("duration")):
                try:
                    dur = float(src)
                except (TypeError, ValueError):
                    dur = 0.0
                if dur > 0:
                    break
            if dur > 0 and fps > 0:
                frames = int(round(dur * fps))
        if w > 0 and h > 0:
            info = {"w": w, "h": h, "fps": fps or 25.0, "frames": max(0, frames),
                    "codec": codec, "pix_fmt": pix_fmt}
    except Exception:
        info = None
    _probe_cache[video_path] = info
    return info


def probe_media_dimensions(path: str):
    """(width, height) of an image or video file, or None if it can't be read.

    Videos go through _probe_video's ffprobe path (cached per path) rather than
    opening a cv2.VideoCapture just to read two numbers. Images are decoded
    directly — there's no cheaper way to get a raster's dimensions than reading
    it."""
    import roop.utilities as util
    if util.is_video(path):
        info = _probe_video(path)
        return (info["w"], info["h"]) if info else None
    img = get_image_frame(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)


# cv2's HEVC seek was measured against self-labelling clips (each frame a known
# flat value, so the frame identifies itself). Requesting frame N and checking
# which frame actually came back:
#
#   codec  pix_fmt        cv2 error   ffmpeg-pipe error
#   h264   yuv420p / 444p     exact       exact
#   hevc   yuv420p  (8-bit)   exact       exact
#   hevc   yuv420p10le       16 frames    exact     <- ordinary 10-bit footage
#   hevc   yuv422p           16 frames    exact
#   hevc   yuv444p           16 frames    exact
#
# and it does not fail loudly: asked for frame 200 it returns frame 214 while
# CAP_PROP_POS_FRAMES still reports 201. So there is nothing to detect at read
# time — a wrong frame looks exactly like a right one — and the only safe
# trigger is the format itself. 8-bit 4:2:0 HEVC (the common case) keeps cv2's
# ~3 ms seek; the formats where it demonstrably lies take the pipe's ~110 ms,
# which is invisible next to a preview render. Sequential reads are unaffected
# either way (~0.05 ms) so playback does not slow down.
_CV2_SAFE_HEVC_PIX = {"yuv420p", "yuvj420p", "nv12", ""}
_BAD_SEEK_CODECS = {"hevc", "h265"}


def _needs_pipe(video_path: str) -> bool:
    """Should this file bypass cv2 entirely? ROOP_CAPTURE_PIPE=1 forces yes,
    0 forces no, anything else (default) decides on the measured table above."""
    mode = os.environ.get("ROOP_CAPTURE_PIPE", "auto").strip().lower()
    if mode == "1":
        return True
    if mode == "0":
        return False
    info = _probe_video(video_path)
    if not info:
        return False
    return (info.get("codec") in _BAD_SEEK_CODECS
            and info.get("pix_fmt") not in _CV2_SAFE_HEVC_PIX)


def _release_pipe():
    """Close the ffmpeg fallback reader. Caller holds _capture_lock."""
    global _pipe_reader, _pipe_path, _pipe_next
    if _pipe_reader is not None:
        try:
            _pipe_reader.release()
        except Exception:
            pass
    _pipe_reader = None
    _pipe_path = None
    _pipe_next = None


def _read_via_pipe(video_path: str, target: int):
    """Decode frame `target` (0-based) through the ffmpeg pipe reader.

    Keeps the pipe alive across calls: a request for the NEXT frame just reads
    on, which is what makes playback and forward scrubbing usable; a frame a
    little further ahead is reached by reading through the ones between, and
    anything else re-spawns ffmpeg with an accurate input seek. Caller holds
    _capture_lock, so the single shared reader needs no lock of its own."""
    global _pipe_reader, _pipe_path, _pipe_next
    info = _probe_video(video_path)
    if not info:
        return None
    # A short hop forward — the shape of every scrub drag and frame-step — is
    # cheaper to read through than to re-seek, so skip the frames between rather
    # than tearing the pipe down (see _pipe_walk_max).
    if (_pipe_reader is not None and _pipe_path == video_path
            and _pipe_next is not None and 0 < target - _pipe_next <= _pipe_walk_max()):
        while _pipe_next < target:
            ok, _ = _pipe_reader.read()
            if not ok:
                _release_pipe()
                break
            _pipe_next += 1
    if _pipe_reader is not None and (_pipe_path != video_path or target != _pipe_next):
        _release_pipe()
    if _pipe_reader is None:
        try:
            from roop.nvdec_reader import FFmpegVideoReader, _probe, nvdec_wanted
            hw = "cuda" if (nvdec_wanted() and _probe(video_path)) else None
            reader = FFmpegVideoReader(video_path, info["w"], info["h"], info["fps"],
                                       hwaccel=hw)
            # set() is only honoured before the first read, which is exactly the
            # state a freshly constructed reader is in.
            reader.set(cv2.CAP_PROP_POS_FRAMES, target)
            _pipe_reader, _pipe_path, _pipe_next = reader, video_path, target
        except Exception:
            _release_pipe()
            return None
    ok, frame = _pipe_reader.read()
    if not ok or frame is None:
        _release_pipe()
        return None
    _pipe_next = target + 1
    return frame


def get_image_frame(filename: str):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        print(f"Exception reading {filename}")
    return None


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    global current_video_path, current_capture, current_frame_total, current_next_pos

    # Cache lookup deliberately OUTSIDE _capture_lock: a hit is the common case
    # while scrubbing (the same frame is asked for by the raw preview, the swap
    # preview and the hover thumbnail) and must not wait behind another thread's
    # seek.
    key = _cache_key(video_path, frame_number)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    with _capture_lock:
        # Another thread may have decoded this exact frame while we queued for
        # the lock — which is precisely what a burst of scrub requests looks
        # like. Check again rather than seeking for a frame we now have.
        cached = _cache_get(key)
        if cached is not None:
            return cached

        # Animated WebP — use PIL-based reader (no FFmpeg involved)
        if video_path.lower().endswith(".webp"):
            _load_animated_webp(video_path)
            if _awebp_frames:
                idx = max(0, min(len(_awebp_frames) - 1, frame_number - 1))
                # A COPY, for the same reason _cache_get returns one: _awebp_frames
                # is a module-level list retained for the whole session, so handing
                # out its own array lets one caller's overlay (face boxes, mask
                # outlines) burn permanently into every later read of that frame.
                return _awebp_frames[idx].copy()
            return None

        # `or current_capture is None` matters: release_video() can be called
        # directly (shutdown, between runs) and leaves the path set, so without
        # this a second request for the SAME file would skip re-opening and then
        # dereference a None capture.
        if video_path != current_video_path or current_capture is None:
            release_video()
            current_capture = cv2.VideoCapture(video_path)
            current_video_path = video_path
            current_frame_total = current_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            # A frame count of 0 (cv2 could not index the file) would clamp every
            # request to frame 0 and show the same still for the whole clip. Ask
            # ffprobe instead so the timeline gets a real length.
            if int(current_frame_total or 0) <= 0:
                info = _probe_video(video_path)
                if info and info["frames"] > 0:
                    current_frame_total = info["frames"]
                    print(f"[Capturer] OpenCV reported no frame count for "
                          f"{os.path.basename(video_path)}; using ffprobe's "
                          f"{info['frames']} frames.", flush=True)
            current_next_pos = 0  # a fresh capture reads frame 0 on the next read()

        total = int(current_frame_total or 0)
        if total <= 0:
            return None
        target = max(0, min(total - 1, frame_number - 1))

        # Known-bad file: go straight to the pipe, either because cv2 already
        # failed to read here or because this codec/pixel-format combination is
        # one where cv2 silently returns the WRONG frame (see _needs_pipe).
        if video_path not in _pipe_needed and _needs_pipe(video_path):
            _pipe_needed.add(video_path)
            info = _probe_video(video_path) or {}
            print(f"[Capturer] {os.path.basename(video_path)} is "
                  f"{info.get('codec', '?')}/{info.get('pix_fmt', '?')} — OpenCV mis-seeks "
                  f"this format silently, so preview/timeline will decode it through "
                  f"ffmpeg instead.", flush=True)
        if video_path in _pipe_needed:
            frame = _read_via_pipe(video_path, target)
            _cache_put(key, frame)
            return frame

        # Fast path: the requested frame is exactly the next one in sequence, so
        # decode it in order without a seek (see current_next_pos note above).
        #
        # Near-fast path: it is a short hop FORWARD, so walk there with grab()
        # (decode without the colour conversion, 0.67 ms/frame) instead of
        # seeking. A seek costs a flat ~125-180 ms however short the hop, so
        # stepping three frames forward used to cost forty times what walking
        # does — and a scrub drag is nothing but short hops (see _walk_max).
        if target != current_next_pos:
            gap = target - current_next_pos if current_next_pos is not None else None
            if gap is not None and 0 < gap <= _walk_max():
                for _ in range(gap):
                    if not current_capture.grab():
                        # Ran off the end of what the decoder will give us; the
                        # position is now unknown, so fall back to a real seek.
                        current_next_pos = None
                        break
                else:
                    current_next_pos = target
            if target != current_next_pos:
                current_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        has_frame, frame = current_capture.read()
        if has_frame:
            current_next_pos = target + 1
            _cache_put(key, frame)
            return frame
        current_next_pos = None  # position unknown after a failed read

        # cv2 came up empty. Only blame cv2 if ffmpeg can actually produce this
        # frame — a read failing past the end of a clip is normal and must not
        # condemn the file to the slower path for the rest of the session.
        frame = _read_via_pipe(video_path, target)
        if frame is not None and video_path not in _pipe_needed:
            _pipe_needed.add(video_path)
            print(f"[Capturer] OpenCV could not decode frame {target} of "
                  f"{os.path.basename(video_path)} but ffmpeg could — using the "
                  f"ffmpeg pipe for this file (common with long H.265).", flush=True)
        _cache_put(key, frame)
        return frame


def release_video():
    global current_capture, current_next_pos, current_video_path, current_frame_total

    # Caller must hold _capture_lock when called from get_video_frame;
    # direct callers (shutdown, etc.) should also acquire it.
    if current_capture is not None:
        current_capture.release()
        current_capture = None
    _release_pipe()
    current_next_pos = None
    # Clear the identity too, so the next request re-opens instead of matching a
    # path whose capture is gone (and re-reads the frame count, which was stale).
    current_video_path = None
    current_frame_total = 0
    # NOT clear_frame_cache(): this runs on every target switch, and the cache is
    # keyed by path, so cached frames of the file being switched away from are
    # still valid — and switching back is the common move. The byte budget, not
    # this, is what bounds the memory.


def get_video_frame_total(video_path: str) -> int:
    # Animated WebP — use PIL
    if video_path.lower().endswith(".webp"):
        try:
            from PIL import Image
            with Image.open(video_path) as img:
                return getattr(img, "n_frames", 1)
        except Exception:
            return 1
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if video_frame_total > 0:
        return video_frame_total
    # cv2 could not index the file (seen on long H.265). Everything downstream —
    # the timeline length, the trim range, the run's frame budget — is derived
    # from this number, and 0 makes all of them collapse, so fall back to
    # ffprobe rather than returning a length the clip does not have.
    info = _probe_video(video_path)
    if info and info["frames"] > 0:
        return info["frames"]
    return video_frame_total
