"""Add frame is not None check and temporary scrub keyframe extraction fallback to capturer.py."""
import os

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
path = os.path.join(APP_DIR, "roop", "capturer.py")
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

fallback_fn = '''def _extract_fallback_frame(video_path: str, target: int) -> Optional[Frame]:
    \"\"\"Extract and cache a single frame via accurate ffmpeg seek.

    Acts as the final fallback for tricky, corrupt, or non-indexable video formats,
    caching extracted frames in a temporary directory for responsive scrubbing.
    \"\"\"
    try:
        from roop.ffmpeg_path import ffmpeg_binary
        info = _probe_video(video_path)
        fps = float(info.get("fps") or 25.0) if info else 25.0
        ts = max(0.0, target / fps)

        import tempfile
        import hashlib
        temp_dir = os.path.join(tempfile.gettempdir(), "roop_scrub_cache")
        os.makedirs(temp_dir, exist_ok=True)
        h = hashlib.sha256(f"{video_path}_{target}".encode("utf-8")).hexdigest()[:16]
        out_jpg = os.path.join(temp_dir, f"frame_{h}_{target}.jpg")

        if os.path.isfile(out_jpg) and os.path.getsize(out_jpg) > 0:
            frame = cv2.imread(out_jpg)
            if frame is not None and getattr(frame, "size", 0) > 0:
                return frame

        cmd = [
            ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
            "-ss", f"{ts:.4f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            out_jpg
        ]
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kwargs)
        if res.returncode == 0 and os.path.isfile(out_jpg) and os.path.getsize(out_jpg) > 0:
            frame = cv2.imread(out_jpg)
            if frame is not None and getattr(frame, "size", 0) > 0:
                return frame
    except Exception as _degrade_error:
        _swallowed("roop/capturer.py:_extract_fallback_frame", _degrade_error, "fallback continued")
    return None

'''

if "def _extract_fallback_frame(" not in content:
    idx = content.find("def get_image_frame(filename: str):")
    assert idx != -1, "get_image_frame not found"
    content = content[:idx] + fallback_fn + "\n" + content[idx:]

old_read_block = """        has_frame, frame = current_capture.read()
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
        return frame"""

new_read_block = """        has_frame, frame = current_capture.read()
        if has_frame and frame is not None and getattr(frame, "size", 0) > 0:
            current_next_pos = target + 1
            _cache_put(key, frame)
            return frame
        current_next_pos = None  # position unknown after a failed read

        # Fallback 1: ffmpeg pipe reader
        frame = _read_via_pipe(video_path, target)
        if frame is not None and getattr(frame, "size", 0) > 0:
            if video_path not in _pipe_needed:
                _pipe_needed.add(video_path)
                print(f"[Capturer] OpenCV could not decode frame {target} of "
                      f"{os.path.basename(video_path)} but ffmpeg could — using the "
                      f"ffmpeg pipe for this file (common with long H.265).", flush=True)
            _cache_put(key, frame)
            return frame

        # Fallback 2: Direct ffmpeg extraction into temporary scrub cache
        frame = _extract_fallback_frame(video_path, target)
        if frame is not None and getattr(frame, "size", 0) > 0:
            _cache_put(key, frame)
            return frame

        return None"""

assert old_read_block in content, "old_read_block not found in capturer.py"
content = content.replace(old_read_block, new_read_block, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("[OK] Updated app/roop/capturer.py with frame None checks and fallback cache extraction")
