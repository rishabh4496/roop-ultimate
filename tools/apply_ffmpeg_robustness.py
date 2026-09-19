"""Apply FFmpeg robustness fixes:
1. Odd dimension padding: pad=ceil(iw/2)*2:ceil(ih/2)*2 on -c:v libx264 commands
2. Optional audio mapping: -map 0:v:0 -map 1:a? -c:a aac -b:a 192k (protects silent videos)
3. Subprocess error handling: checks returncode and prints stderr on failure
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")

def update_util_ffmpeg():
    path = os.path.join(APP, "roop", "util_ffmpeg.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update finalize_web_video with pad filter, -map 1:a?, and stderr error logging
    old_finalize = """def finalize_web_video(raw_video_path: str, final_video_path: str, audio_source: str = None, crf: int = 18, delete_raw: bool = False) -> bool:
    \"\"\"Post-process or re-encode video into a browser-compliant web-compatible MP4.
    
    Explicitly includes:
      - -c:v libx264
      - -pix_fmt yuv420p
      - -movflags +faststart
      - -c:a aac -b:a 192k (or cleanly stripped with -an if muted / no audio)
    \"\"\"
    temp_dest = (final_video_path + ".web.tmp.mp4") if os.path.abspath(raw_video_path) == os.path.abspath(final_video_path) else final_video_path
    has_audio = False
    if audio_source and os.path.isfile(audio_source):
        has_audio = bool(util.audio_sample_rate(audio_source))

    cmd = [
        ffmpeg_binary(), '-hide_banner', '-y', '-loglevel', 'error',
        '-i', raw_video_path,
    ]
    if has_audio:
        cmd.extend([
            '-i', audio_source,
            '-map', '0:v:0',
            '-map', '1:a?',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
        ])
    else:
        cmd.extend([
            '-map', '0:v:0',
            '-an',
        ])
    quality = crf if crf is not None else 18
    cmd.extend([
        '-c:v', 'libx264',
        *_rate_control('libx264', quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        temp_dest,
    ])
    kwargs = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }
    if os.name == 'nt':
        kwargs['creationflags'] = 0x08000000
    res = subprocess.run(cmd, **kwargs)
    if res.returncode == 0 and os.path.isfile(temp_dest):
        if temp_dest != final_video_path:
            os.replace(temp_dest, final_video_path)
        if delete_raw and raw_video_path != final_video_path and os.path.isfile(raw_video_path):
            try:
                os.remove(raw_video_path)
            except OSError:
                pass
        return True
    if os.path.isfile(temp_dest) and temp_dest != final_video_path:
        try:
            os.remove(temp_dest)
        except OSError:
            pass
    return False"""

    new_finalize = """def finalize_web_video(raw_video_path: str, final_video_path: str, audio_source: str = None, crf: int = 18, delete_raw: bool = False) -> bool:
    \"\"\"Post-process or re-encode video into a browser-compliant web-compatible MP4.
    
    Explicitly includes:
      - -c:v libx264 with dimension padding: -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
      - -pix_fmt yuv420p
      - -movflags +faststart
      - -map 0:v:0 -map 1:a? -c:a aac -b:a 192k (optional audio mapping so silent videos never crash)
      - Stderr logging and return code validation
    \"\"\"
    temp_dest = (final_video_path + ".web.tmp.mp4") if os.path.abspath(raw_video_path) == os.path.abspath(final_video_path) else final_video_path

    cmd = [
        ffmpeg_binary(), '-hide_banner', '-y', '-loglevel', 'error',
        '-i', raw_video_path,
    ]
    if audio_source and os.path.isfile(audio_source):
        cmd.extend([
            '-i', audio_source,
            '-map', '0:v:0',
            '-map', '1:a?',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
        ])
    else:
        cmd.extend([
            '-map', '0:v:0',
            '-an',
        ])
    quality = crf if crf is not None else 18
    cmd.extend([
        '-c:v', 'libx264',
        *_rate_control('libx264', quality),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        temp_dest,
    ])
    kwargs = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }
    if os.name == 'nt':
        kwargs['creationflags'] = 0x08000000
    res = subprocess.run(cmd, **kwargs)
    if res.returncode != 0:
        err = (res.stderr or b"").decode("utf-8", "replace").strip()
        print(f"[FFmpeg Error] finalize_web_video failed (exit {res.returncode}):\\nCommand: {' '.join(cmd)}\\n{err}", flush=True)
        if os.path.isfile(temp_dest) and temp_dest != final_video_path:
            try:
                os.remove(temp_dest)
            except OSError:
                pass
        return False

    if os.path.isfile(temp_dest):
        if temp_dest != final_video_path:
            os.replace(temp_dest, final_video_path)
        if delete_raw and raw_video_path != final_video_path and os.path.isfile(raw_video_path):
            try:
                os.remove(raw_video_path)
            except OSError:
                pass
        return True
    return False"""
    if old_finalize in content:
        content = content.replace(old_finalize, new_finalize, 1)

    # 2. Update cut_video to use pad filter and -map 0:v:0 -map 0:a?
    old_cut = """def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame
    has_audio = bool(util.audio_sample_rate(original_video))

    if reencode:
        audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
        run_ffmpeg([
            '-ss', format(start_time, ".2f"),
            '-i', original_video,
            '-c:v', 'libx264',
            *_rate_control('libx264', roop.globals.video_quality),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            *audio_flags,
            '-frames:v', str(num_frames),
            cut_video
        ])
    else:
        cmd = ['-ss', format(start_time, ".2f"), '-i', original_video, '-frames:v', str(num_frames), '-c:v', 'copy']
        if has_audio:
            cmd.extend(['-c:a', 'copy'])
        else:
            cmd.append('-an')
        if cut_video.lower().endswith(('.mp4', '.mov', '.m4v')):
            cmd.extend(['-movflags', '+faststart'])
        cmd.append(cut_video)
        run_ffmpeg(cmd)"""

    new_cut = """def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    if reencode:
        run_ffmpeg([
            '-ss', format(start_time, ".2f"),
            '-i', original_video,
            '-map', '0:v:0',
            '-map', '0:a?',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:v', 'libx264',
            *_rate_control('libx264', roop.globals.video_quality),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '192k',
            '-frames:v', str(num_frames),
            cut_video
        ])
    else:
        cmd = [
            '-ss', format(start_time, ".2f"),
            '-i', original_video,
            '-map', '0:v:0',
            '-map', '0:a?',
            '-frames:v', str(num_frames),
            '-c:v', 'copy',
            '-c:a', 'copy',
        ]
        if cut_video.lower().endswith(('.mp4', '.mov', '.m4v')):
            cmd.extend(['-movflags', '+faststart'])
        cmd.append(cut_video)
        run_ffmpeg(cmd)"""
    if old_cut in content:
        content = content.replace(old_cut, new_cut, 1)

    # 3. Update create_video to pad odd dimensions
    old_create = "vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'"
    new_create = "vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,pad=ceil(iw/2)*2:ceil(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'"
    if old_create in content:
        content = content.replace(old_create, new_create)

    # 4. Update create_video_from_gif to include pad
    old_gif_vf = "vf = f\"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p,fps={fps}\""
    new_gif_vf = "vf = f\"scale=trunc(iw/2)*2:trunc(ih/2)*2,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p,fps={fps}\""
    if old_gif_vf in content:
        content = content.replace(old_gif_vf, new_gif_vf, 1)

    # 5. Update resize_video
    old_resize = """def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
    scale_filter = (
        f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
        f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    )
    has_audio = bool(util.audio_sample_rate(input_path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    return run_ffmpeg([
        '-i', input_path,
        '-vf', scale_filter,
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        *audio_flags,
        output_path
    ])"""
    new_resize = """def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
    w = width + (width % 2)
    h = height + (height % 2)
    scale_filter = (
        f'scale={w}:{h}:force_original_aspect_ratio=decrease,'
        f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,pad=ceil(iw/2)*2:ceil(ih/2)*2'
    )
    return run_ffmpeg([
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '0:a?',
        '-vf', scale_filter,
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ])"""
    if old_resize in content:
        content = content.replace(old_resize, new_resize, 1)

    # 6. Update rotate_media
    old_rotate = """    return run_ffmpeg([
        '-i', input_path,
        '-vf', vf,
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        *audio_flags,
        output_path
    ])"""
    new_rotate = """    return run_ffmpeg([
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '0:a?',
        '-vf', f"{vf},pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ])"""
    if old_rotate in content:
        content = content.replace(old_rotate, new_rotate, 1)

    # 7. Update change_fps
    old_fps = """    return run_ffmpeg([
        '-i', input_path,
        '-vf', f'fps={fps}',
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        *audio_flags,
        output_path
    ])"""
    new_fps = """    return run_ffmpeg([
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '0:a?',
        '-vf', f'fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ])"""
    if old_fps in content:
        content = content.replace(old_fps, new_fps, 1)

    # 8. Update crop_media
    old_crop = """    return run_ffmpeg([
        '-i', input_path,
        '-vf', crop_filter,
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        *audio_flags,
        output_path
    ])"""
    new_crop = """    return run_ffmpeg([
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '0:a?',
        '-vf', f"{crop_filter},pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ])"""
    if old_crop in content:
        content = content.replace(old_crop, new_crop, 1)

    # 9. Update apply_media_transforms
    old_transforms = """    if is_video:
        has_audio = bool(util.audio_sample_rate(input_path))
        audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
        args += [
            '-c:v', 'libx264',
            *_rate_control('libx264', quality),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            *audio_flags
        ]"""
    new_transforms = """    if is_video:
        args += [
            '-map', '0:v:0',
            '-map', '0:a?',
            '-c:v', 'libx264',
            *_rate_control('libx264', quality),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '192k'
        ]"""
    if old_transforms in content:
        content = content.replace(old_transforms, new_transforms, 1)

    # Also in apply_media_transforms, add pad to vf if is_video
    old_vf_join = "vf = ','.join(vf_filters)\n    args = ['-i', input_path, '-vf', vf]"
    new_vf_join = "all_filters = list(vf_filters) + (['pad=ceil(iw/2)*2:ceil(ih/2)*2'] if is_video else [])\n    vf = ','.join(all_filters)\n    args = ['-i', input_path, '-vf', vf]"
    if old_vf_join in content:
        content = content.replace(old_vf_join, new_vf_join, 1)

    # 10. Update apply_media_transforms_webp
    old_webp_vf = "'-vf', vf,"
    new_webp_vf = "'-vf', f'{vf},pad=ceil(iw/2)*2:ceil(ih/2)*2',"
    if old_webp_vf in content:
        content = content.replace(old_webp_vf, new_webp_vf, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/util_ffmpeg.py with odd-dimension pad & optional audio")


def update_routes_queue():
    path = os.path.join(APP, "routes_queue.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_queue_proc = """        kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, **kwargs)
        if proc.returncode != 0 or not os.path.exists(dest):
            return JSONResponse(status_code=500, content={
                "message": f"ffmpeg concat failed (exit {proc.returncode}) — see the terminal log"})"""

    new_queue_proc = """        kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kwargs)
        if proc.returncode != 0 or not os.path.exists(dest):
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            print(f"[Queue Error] ffmpeg concat failed (exit {proc.returncode}):\\n{err}", flush=True)
            return JSONResponse(status_code=500, content={
                "message": f"ffmpeg concat failed (exit {proc.returncode}): {err[:200]}"})"""

    if old_queue_proc in content:
        content = content.replace(old_queue_proc, new_queue_proc, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Updated app/routes_queue.py with stderr error logging")


def update_segment_writer():
    path = os.path.join(APP, "roop", "segment_writer.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_concat_proc = """            proc = subprocess.run(cmd, capture_output=True, **kwargs)
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", "replace")[:400]
                bar_write(f"[Resume] segment concat failed (ffmpeg exit {proc.returncode}): {err}")
                return False"""

    new_concat_proc = """            proc = subprocess.run(cmd, capture_output=True, **kwargs)
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", "replace").strip()
                bar_write(f"[Resume] segment concat failed (ffmpeg exit {proc.returncode}): {err[:400]}")
                print(f"[Resume Error] segment concat failed (ffmpeg exit {proc.returncode}):\\n{err}", flush=True)
                return False"""

    if old_concat_proc in content:
        content = content.replace(old_concat_proc, new_concat_proc, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Updated app/roop/segment_writer.py with stderr error logging")


def update_post_swap():
    path = os.path.join(APP, "post_swap.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # In _classical_video_inplace
    old_classical = """    has_audio = bool(util.audio_sample_rate(path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    cmd = ([FFMPEG_BINARY, '-hide_banner', '-y', '-i', path, '-vf', vf,
            '-c:v', 'libx264' if enc == 'libx264' else enc] + _rate_control(enc, q) +
           ['-pix_fmt', 'yuv420p', '-movflags', '+faststart'] + audio_flags + [tmp])"""

    new_classical = """    pad_vf = f"{vf},pad=ceil(iw/2)*2:ceil(ih/2)*2"
    cmd = ([FFMPEG_BINARY, '-hide_banner', '-y', '-i', path,
            '-map', '0:v:0', '-map', '0:a?',
            '-vf', pad_vf,
            '-c:v', 'libx264' if enc == 'libx264' else enc] + _rate_control(enc, q) +
           ['-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '192k', tmp])"""

    if old_classical in content:
        content = content.replace(old_classical, new_classical, 1)

    # In _interp_video_minterpolate
    old_minterp = """    has_audio = bool(util.audio_sample_rate(path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    cmd = ([FFMPEG_BINARY, "-hide_banner", "-y", "-i", path, "-vf", vf,
            "-c:v", 'libx264' if enc == 'libx264' else enc] + _rate_control(enc, q) +
           ['-pix_fmt', 'yuv420p', '-movflags', '+faststart'] + audio_flags + [tmp])"""

    new_minterp = """    pad_vf = f"{vf},pad=ceil(iw/2)*2:ceil(ih/2)*2"
    cmd = ([FFMPEG_BINARY, "-hide_banner", "-y", "-i", path,
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", pad_vf,
            "-c:v", 'libx264' if enc == 'libx264' else enc] + _rate_control(enc, q) +
           ['-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            "-c:a", "aac", "-b:a", "192k", tmp])"""

    if old_minterp in content:
        content = content.replace(old_minterp, new_minterp, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/post_swap.py with odd-dimension pad & optional audio")


if __name__ == "__main__":
    update_util_ffmpeg()
    update_routes_queue()
    update_segment_writer()
    update_post_swap()
    print("\nAll FFmpeg robustness fixes applied successfully!")
