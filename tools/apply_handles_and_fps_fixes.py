"""Apply OpenCV lifecycle hardening (try/finally release) and fractional FPS preservation."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")

def update_utilities():
    path = os.path.join(APP, "roop", "utilities.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add detect_fps_fractional and update detect_fps
    old_detect_fps = """def detect_fps(target_path: str) -> float:
    # Animated WebP: OpenCV returns 0 FPS — derive from PIL frame durations instead
    if target_path and target_path.lower().endswith('.webp'):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                n = getattr(img, 'n_frames', 1)
                if n > 1:
                    durations = []
                    for i in range(n):
                        img.seek(i)
                        d = img.info.get('duration', None)
                        durations.append(d)
                    print(f"[detect_fps] WebP '{os.path.basename(target_path)}': "
                          f"{n} frames, raw durations (ms) = {durations}")
                    # Treat None or 0 as 100 ms (browsers use ~100 ms as the
                    # effective minimum for animated WebP, similar to GIF).
                    cleaned = [(d if d and d > 0 else 100) for d in durations]
                    avg_ms = sum(cleaned) / len(cleaned)
                    fps = round(1000.0 / avg_ms, 2)
                    print(f"[detect_fps] avg_ms={avg_ms:.1f} → fps={fps}")
                    return fps
        except Exception as exc:
            print(f"[detect_fps] WebP duration read failed: {exc}")
        return 10.0  # safe fallback: 100 ms per frame
    fps = 24.0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return constant_frame_rate(fps)"""

    new_detect_fps = """def detect_fps_fractional(target_path: str) -> str:
    \"\"\"Extract exact fractional r_frame_rate string (e.g. '24000/1001', '30/1') using ffprobe.

    Falls back to stringified detect_fps() if ffprobe is unavailable or fails.
    \"\"\"
    if target_path and target_path.lower().endswith('.webp'):
        return str(detect_fps(target_path))
    try:
        from roop.ffmpeg_path import ffprobe_binary
        cmd = [
            ffprobe_binary(), "-v", "0", "-of", "csv=p=0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            target_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (res.stdout or "").strip()
        if out and "/" in out:
            num, den = out.split("/", 1)
            num_i, den_i = int(num), int(den)
            if den_i > 0 and num_i > 0:
                return out
    except Exception as _degrade_error:
        _swallowed("utilities.py:detect_fps_fractional", _degrade_error, "fallback continued")
    return str(detect_fps(target_path))


def detect_fps(target_path: str) -> float:
    # Animated WebP: OpenCV returns 0 FPS — derive from PIL frame durations instead
    if target_path and target_path.lower().endswith('.webp'):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                n = getattr(img, 'n_frames', 1)
                if n > 1:
                    durations = []
                    for i in range(n):
                        img.seek(i)
                        d = img.info.get('duration', None)
                        durations.append(d)
                    print(f"[detect_fps] WebP '{os.path.basename(target_path)}': "
                          f"{n} frames, raw durations (ms) = {durations}")
                    cleaned = [(d if d and d > 0 else 100) for d in durations]
                    avg_ms = sum(cleaned) / len(cleaned)
                    fps = round(1000.0 / avg_ms, 2)
                    print(f"[detect_fps] avg_ms={avg_ms:.1f} → fps={fps}")
                    return fps
        except Exception as exc:
            print(f"[detect_fps] WebP duration read failed: {exc}")
        return 10.0

    # Primary: Dynamic fractional FPS extraction using ffprobe
    try:
        from roop.ffmpeg_path import ffprobe_binary
        cmd = [
            ffprobe_binary(), "-v", "0", "-of", "csv=p=0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            target_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (res.stdout or "").strip()
        if out and "/" in out:
            num, den = out.split("/", 1)
            num_i, den_i = int(num), int(den)
            if den_i > 0 and num_i > 0:
                val = num_i / den_i
                if np.isfinite(val) and 0.0 < val <= 240.0:
                    return constant_frame_rate(val)
    except Exception as _degrade_error:
        _swallowed("utilities.py:detect_fps", _degrade_error, "fallback continued")

    # Fallback: OpenCV VideoCapture with robust try/finally release
    fps = 24.0
    cap = None
    try:
        cap = cv2.VideoCapture(target_path)
        if cap.isOpened():
            val = cap.get(cv2.CAP_PROP_FPS)
            if val and val > 0:
                fps = val
    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()
    return constant_frame_rate(fps)"""

    if old_detect_fps in content:
        content = content.replace(old_detect_fps, new_detect_fps, 1)

    # 2. Fix detect_dimensions leak
    old_dims = """    if is_image(target_path):
        img = cv2.imread(target_path)
        if img is not None:
            return img.shape[1], img.shape[0]
        return 0, 0
    # Animated WebP: OpenCV VideoCapture returns 0x0 — use PIL instead
    if target_path and target_path.lower().endswith('.webp') and is_animated_webp(target_path):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                return img.width, img.height
        except Exception as _degrade_error:
            _swallowed("roop/utilities.py:522", _degrade_error, "fallback continued")
            pass
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return 0, 0"""

    new_dims = """    if is_image(target_path):
        img = cv2.imread(target_path)
        if img is not None:
            return img.shape[1], img.shape[0]
        return 0, 0
    # Animated WebP: OpenCV VideoCapture returns 0x0 — use PIL instead
    if target_path and target_path.lower().endswith('.webp') and is_animated_webp(target_path):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                return img.width, img.height
        except Exception as _degrade_error:
            _swallowed("roop/utilities.py:522", _degrade_error, "fallback continued")
            pass
    cap = None
    try:
        cap = cv2.VideoCapture(target_path)
        if cap.isOpened():
            return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        if cap is not None:
            cap.release()
    return 0, 0"""

    if old_dims in content:
        content = content.replace(old_dims, new_dims, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/utilities.py")


def update_util_ffmpeg():
    path = os.path.join(APP, "roop", "util_ffmpeg.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pass dynamic fractional framerate into create_video
    old_create_vid = """def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    # scale=trunc(iw/2)*2:trunc(ih/2)*2 rounds odd dimensions down to even, which is
    # required by yuv420p / libx264. Without this, frames with odd width or height
    # cause ffmpeg to fail silently and produce an empty (corrupt) output file.
    fps = util.constant_frame_rate(fps)
    vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,pad=ceil(iw/2)*2:ceil(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'
    run_ffmpeg([
        '-framerate', str(fps),
        '-i', os.path.join(temp_directory_path, f'%06d.{roop.globals.CFG.output_image_format}'),
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-vf', f'{vf},pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-r', str(fps),
        '-vsync', 'cfr',
        '-fps_mode', 'cfr',
        '-an',
        '-y', dest_filename
    ])
    return dest_filename"""

    new_create_vid = """def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)

    # Fractional FPS extraction: pass exact fractional r_frame_rate directly into -framerate
    framerate_str = str(fps)
    try:
        if target_path and os.path.isfile(target_path):
            fractional = util.detect_fps_fractional(target_path)
            if fractional:
                framerate_str = fractional
    except Exception:
        pass

    fps_num = util.constant_frame_rate(fps)
    vf = util.cfr_video_filter(fps_num) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,pad=ceil(iw/2)*2:ceil(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'
    run_ffmpeg([
        '-framerate', framerate_str,
        '-i', os.path.join(temp_directory_path, f'%06d.{roop.globals.CFG.output_image_format}'),
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-vf', f'{vf},pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-r', framerate_str,
        '-vsync', 'cfr',
        '-fps_mode', 'cfr',
        '-an',
        '-y', dest_filename
    ])
    return dest_filename"""

    if old_create_vid in content:
        content = content.replace(old_create_vid, new_create_vid, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/util_ffmpeg.py")


def update_routes_extras():
    path = os.path.join(APP, "routes_extras.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Wrap VideoWriter in extras_apply with try/finally
    old_apply_writer = """    writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    for i in range(1, total + 1):
        fr = get_video_frame(path, i)
        if fr is None:
            continue
        writer.write(_process_frame(fr))
    writer.release()"""

    new_apply_writer = """    writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    try:
        for i in range(1, total + 1):
            fr = get_video_frame(path, i)
            if fr is None:
                continue
            writer.write(_process_frame(fr))
    finally:
        writer.release()"""

    if old_apply_writer in content:
        content = content.replace(old_apply_writer, new_apply_writer, 1)

    # Wrap VideoWriter in extras_enhance with try/finally
    old_enh_writer = """        raw_tmp = os.path.join(out_dir, f".raw_{operation}_{subtype}_{stem}.mp4")
        writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        writer.write(out_first)
        for i in range(2, total + 1):
            fr = get_video_frame(path, i)
            if fr is None:
                continue
            res = proc.Run(fr)
            if res.shape[:2] != (oh, ow):
                res = cv2.resize(res, (ow, oh))
            writer.write(res)
        writer.release()"""

    new_enh_writer = """        raw_tmp = os.path.join(out_dir, f".raw_{operation}_{subtype}_{stem}.mp4")
        writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        try:
            writer.write(out_first)
            for i in range(2, total + 1):
                fr = get_video_frame(path, i)
                if fr is None:
                    continue
                res = proc.Run(fr)
                if res.shape[:2] != (oh, ow):
                    res = cv2.resize(res, (ow, oh))
                writer.write(res)
        finally:
            writer.release()"""

    if old_enh_writer in content:
        content = content.replace(old_enh_writer, new_enh_writer, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/routes_extras.py")


def update_post_swap():
    path = os.path.join(APP, "post_swap.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_classical_cap = """    cap = cv2.VideoCapture(path)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()"""

    new_classical_cap = """    cap = None
    try:
        cap = cv2.VideoCapture(path)
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        if cap is not None:
            cap.release()"""

    if old_classical_cap in content:
        content = content.replace(old_classical_cap, new_classical_cap, 1)

    old_minterp_cap = """    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()"""

    new_minterp_cap = """    cap = None
    try:
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        if cap is not None:
            cap.release()"""

    if old_minterp_cap in content:
        content = content.replace(old_minterp_cap, new_minterp_cap, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/post_swap.py")


def update_capturer():
    path = os.path.join(APP, "roop", "capturer.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_cap_total = """    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()"""

    new_cap_total = """    capture = None
    try:
        capture = cv2.VideoCapture(video_path)
        video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        if capture is not None:
            capture.release()"""

    if old_cap_total in content:
        content = content.replace(old_cap_total, new_cap_total, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/capturer.py")


if __name__ == "__main__":
    update_utilities()
    update_util_ffmpeg()
    update_routes_extras()
    update_post_swap()
    update_capturer()
    print("\nAll file handles and FPS updates applied successfully!")
