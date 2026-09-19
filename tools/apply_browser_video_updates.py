"""Apply browser-compliant video rendering flags across roop-ultimate.

Flags enforced:
- -c:v libx264
- -pix_fmt yuv420p
- -movflags +faststart
- -c:a aac -b:a 192k (cleanly stripped with -an if processing muted video)
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")

def update_util_ffmpeg():
    path = os.path.join(APP, "roop", "util_ffmpeg.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add finalize_web_video helper function
    helper_code = '''

def finalize_web_video(raw_video_path: str, final_video_path: str, audio_source: str = None, crf: int = 18, delete_raw: bool = False) -> bool:
    """Post-process or re-encode video into a browser-compliant web-compatible MP4.
    
    Explicitly includes:
      - -c:v libx264
      - -pix_fmt yuv420p
      - -movflags +faststart
      - -c:a aac -b:a 192k (or cleanly stripped with -an if muted / no audio)
    """
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
            '-map', '1:a:0?',
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
    return False

'''
    if "def finalize_web_video(" not in content:
        idx = content.find("def quality_max(codec: str) -> int:")
        assert idx != -1, "Could not find insertion point in util_ffmpeg.py"
        content = content[:idx] + helper_code + "\n" + content[idx:]

    old_cut = """def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    if reencode:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video, '-c:v', roop.globals.video_encoder, '-c:a', 'aac', '-frames:v', str(num_frames), cut_video])
    else:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video,  '-frames:v', str(num_frames), '-c:v' ,'copy','-c:a' ,'copy', cut_video])"""
    new_cut = """def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
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
    if old_cut in content:
        content = content.replace(old_cut, new_cut, 1)

    old_create = """def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    # scale=trunc(iw/2)*2:trunc(ih/2)*2 rounds odd dimensions down to even, which is
    # required by yuv420p / libx264. Without this, frames with odd width or height
    # cause ffmpeg to fail silently and produce an empty (corrupt) output file.
    fps = util.constant_frame_rate(fps)
    vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'
    run_ffmpeg(['-framerate', str(fps), '-i', os.path.join(temp_directory_path, f'%06d.{roop.globals.CFG.output_image_format}'), '-c:v', roop.globals.video_encoder] + _rate_control(roop.globals.video_encoder, roop.globals.video_quality) + ['-pix_fmt', 'yuv420p', '-vf', vf, '-r', str(fps), '-vsync', 'cfr', '-fps_mode', 'cfr', '-y', dest_filename])
    return dest_filename"""
    new_create = """def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    # scale=trunc(iw/2)*2:trunc(ih/2)*2 rounds odd dimensions down to even, which is
    # required by yuv420p / libx264. Without this, frames with odd width or height
    # cause ffmpeg to fail silently and produce an empty (corrupt) output file.
    fps = util.constant_frame_rate(fps)
    vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'
    run_ffmpeg([
        '-framerate', str(fps),
        '-i', os.path.join(temp_directory_path, f'%06d.{roop.globals.CFG.output_image_format}'),
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-vf', vf,
        '-r', str(fps),
        '-vsync', 'cfr',
        '-fps_mode', 'cfr',
        '-an',
        '-y', dest_filename
    ])
    return dest_filename"""
    if old_create in content:
        content = content.replace(old_create, new_create, 1)

    old_gif = """def create_video_from_gif(gif_path: str, output_path):
    fps = util.detect_fps(gif_path)
    filter = \"\"\"scale='trunc(in_w/2)*2':'trunc(in_h/2)*2',format=yuv420p,fps=10\"\"\"
    run_ffmpeg(['-i', gif_path, '-vf', f'"{filter}"', '-movflags', '+faststart', '-shortest', output_path])"""
    new_gif = """def create_video_from_gif(gif_path: str, output_path):
    fps = util.detect_fps(gif_path)
    vf = f"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p,fps={fps}"
    run_ffmpeg([
        '-i', gif_path,
        '-vf', vf,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-an',
        '-shortest',
        output_path
    ])"""
    if old_gif in content:
        content = content.replace(old_gif, new_gif, 1)

    old_resize = """def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
    scale_filter = (
        f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
        f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    )
    return run_ffmpeg(['-i', input_path, '-vf', scale_filter,
                       '-c:v', roop.globals.video_encoder,
                       *_rate_control(roop.globals.video_encoder, roop.globals.video_quality),
                       '-c:a', 'copy', output_path])"""
    new_resize = """def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
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
    if old_resize in content:
        content = content.replace(old_resize, new_resize, 1)

    old_rotate = """def rotate_media(input_path: str, output_path: str, transform: str) -> bool:
    transform_map = {
        "90° Clockwise":        "transpose=1",
        "90° Counter-clockwise": "transpose=2",
        "180°":                  "transpose=1,transpose=1",
        "Flip Horizontal":       "hflip",
        "Flip Vertical":         "vflip",
    }
    vf = transform_map.get(transform, "transpose=1")
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-c:a', 'copy', output_path])"""
    new_rotate = """def rotate_media(input_path: str, output_path: str, transform: str) -> bool:
    transform_map = {
        "90° Clockwise":        "transpose=1",
        "90° Counter-clockwise": "transpose=2",
        "180°":                  "transpose=1,transpose=1",
        "Flip Horizontal":       "hflip",
        "Flip Vertical":         "vflip",
    }
    vf = transform_map.get(transform, "transpose=1")
    has_audio = bool(util.audio_sample_rate(input_path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    return run_ffmpeg([
        '-i', input_path,
        '-vf', vf,
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        *audio_flags,
        output_path
    ])"""
    if old_rotate in content:
        content = content.replace(old_rotate, new_rotate, 1)

    old_fps = """def change_fps(input_path: str, output_path: str, fps: float) -> bool:
    return run_ffmpeg(['-i', input_path, '-vf', f'fps={fps}',
                       '-c:v', roop.globals.video_encoder,
                       *_rate_control(roop.globals.video_encoder, roop.globals.video_quality),
                       '-c:a', 'copy', output_path])"""
    new_fps = """def change_fps(input_path: str, output_path: str, fps: float) -> bool:
    has_audio = bool(util.audio_sample_rate(input_path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    return run_ffmpeg([
        '-i', input_path,
        '-vf', f'fps={fps}',
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        *audio_flags,
        output_path
    ])"""
    if old_fps in content:
        content = content.replace(old_fps, new_fps, 1)

    old_crop = """def crop_media(input_path: str, output_path: str,
               left_pct: float, right_pct: float,
               top_pct: float,  bottom_pct: float) -> bool:
    l, r, t, b = left_pct / 100, right_pct / 100, top_pct / 100, bottom_pct / 100
    crop_filter = (
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f})"
        f":in_w*{l:.4f}:in_h*{t:.4f}"
    )
    return run_ffmpeg(['-i', input_path, '-vf', crop_filter, '-c:a', 'copy', output_path])"""
    new_crop = """def crop_media(input_path: str, output_path: str,
               left_pct: float, right_pct: float,
               top_pct: float,  bottom_pct: float) -> bool:
    l, r, t, b = left_pct / 100, right_pct / 100, top_pct / 100, bottom_pct / 100
    crop_filter = (
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f})"
        f":in_w*{l:.4f}:in_h*{t:.4f}"
    )
    has_audio = bool(util.audio_sample_rate(input_path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    return run_ffmpeg([
        '-i', input_path,
        '-vf', crop_filter,
        '-c:v', 'libx264',
        *_rate_control('libx264', roop.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        *audio_flags,
        output_path
    ])"""
    if old_crop in content:
        content = content.replace(old_crop, new_crop, 1)

    old_transforms = """def apply_media_transforms(input_path: str, output_path: str,
                           vf_filters: list, is_video: bool) -> bool:
    \"\"\"Apply a list of -vf filters in a single ffmpeg pass.\"\"\"
    if not vf_filters:
        return False
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    vf = ','.join(vf_filters)
    args = ['-i', input_path, '-vf', vf]
    if is_video:
        args += ['-c:v', codec, *_rate_control(codec, quality), '-c:a', 'copy']
    args.append(output_path)
    return run_ffmpeg(args)"""
    new_transforms = """def apply_media_transforms(input_path: str, output_path: str,
                           vf_filters: list, is_video: bool) -> bool:
    \"\"\"Apply a list of -vf filters in a single ffmpeg pass.\"\"\"
    if not vf_filters:
        return False
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    vf = ','.join(vf_filters)
    args = ['-i', input_path, '-vf', vf]
    if is_video:
        has_audio = bool(util.audio_sample_rate(input_path))
        audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
        args += [
            '-c:v', 'libx264',
            *_rate_control('libx264', quality),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            *audio_flags
        ]
    args.append(output_path)
    return run_ffmpeg(args)"""
    if old_transforms in content:
        content = content.replace(old_transforms, new_transforms, 1)

    old_webp_cmd = """    cmd = [
        ffmpeg_binary(), '-hide_banner', '-hwaccel', 'auto', '-y',
        '-loglevel', roop.globals.log_level,
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-an', '-i', '-',
        '-vf', vf,
        '-c:v', codec,
        *_rate_control(codec, quality),
        '-pix_fmt', 'yuv420p',
        output_path,
    ]"""
    new_webp_cmd = """    cmd = [
        ffmpeg_binary(), '-hide_banner', '-hwaccel', 'auto', '-y',
        '-loglevel', roop.globals.log_level,
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-an', '-i', '-',
        '-vf', vf,
        '-c:v', 'libx264',
        *_rate_control('libx264', quality),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path,
    ]"""
    if old_webp_cmd in content:
        content = content.replace(old_webp_cmd, new_webp_cmd, 1)

    old_frames_dir = """def create_video_from_frames_dir(frames_dir: str, output_path: str, fps: float,
                                  image_format: str = 'png') -> bool:
    \"\"\"Re-assemble a video from a directory of sequentially named frame images.

    Frames must follow the %06d.<image_format> naming convention that
    extract_frames() produces (e.g. 000001.png, 000002.png …).
    \"\"\"
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    # scale=trunc(iw/2)*2:trunc(ih/2)*2 rounds odd dimensions down to even, required by yuv420p.
    fps = util.constant_frame_rate(fps)
    vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'
    return run_ffmpeg([
        '-framerate', str(fps),
        '-i',    os.path.join(frames_dir, f'%06d.{image_format}'),
        '-c:v',  codec,
    ] + _rate_control(codec, quality) + [
        '-pix_fmt', 'yuv420p',
        '-vf',   vf,
        '-r', str(fps), '-vsync', 'cfr', '-fps_mode', 'cfr',
        '-y',    output_path,
    ])"""
    new_frames_dir = """def create_video_from_frames_dir(frames_dir: str, output_path: str, fps: float,
                                  image_format: str = 'png') -> bool:
    \"\"\"Re-assemble a video from a directory of sequentially named frame images.

    Frames must follow the %06d.<image_format> naming convention that
    extract_frames() produces (e.g. 000001.png, 000002.png …).
    \"\"\"
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    # scale=trunc(iw/2)*2:trunc(ih/2)*2 rounds odd dimensions down to even, required by yuv420p.
    fps = util.constant_frame_rate(fps)
    vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'
    return run_ffmpeg([
        '-framerate', str(fps),
        '-i',    os.path.join(frames_dir, f'%06d.{image_format}'),
        '-c:v',  'libx264',
    ] + _rate_control('libx264', quality) + [
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-vf',   vf,
        '-r', str(fps), '-vsync', 'cfr', '-fps_mode', 'cfr',
        '-an',
        '-y',    output_path,
    ])"""
    if old_frames_dir in content:
        content = content.replace(old_frames_dir, new_frames_dir, 1)

    old_restore_end = """    if duration is not None:
        commands += ['-t', format(duration, '.6f')]
    commands += [final_video]"""
    new_restore_end = """    if duration is not None:
        commands += ['-t', format(duration, '.6f')]
    if final_video.lower().endswith(('.mp4', '.mov', '.m4v')):
        commands += ['-movflags', '+faststart']
    commands += [final_video]"""
    if old_restore_end in content:
        content = content.replace(old_restore_end, new_restore_end, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/util_ffmpeg.py")


def update_ffmpeg_writer():
    path = os.path.join(APP, "roop", "ffmpeg_writer.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update audio encoding in _build_cmd
    old_audio = """        if audiofile is not None:
            cmd.extend([
                '-i', audiofile,
                '-acodec', 'copy'
            ])"""
    new_audio = """        if audiofile is not None:
            cmd.extend([
                '-i', audiofile,
                '-c:a', 'aac',
                '-b:a', '192k'
            ])"""
    if old_audio in content:
        content = content.replace(old_audio, new_audio, 1)

    # 2. Add -movflags +faststart
    old_tail = """        cmd.extend([
            '-pix_fmt', 'yuv420p',

        ])
        cmd.extend([
            self.filename
        ])"""
    new_tail = """        cmd.extend([
            '-pix_fmt', 'yuv420p',
        ])
        if self.filename.lower().endswith(('.mp4', '.mov', '.m4v')):
            cmd.extend(['-movflags', '+faststart'])
        cmd.extend([
            self.filename
        ])"""
    if old_tail in content:
        content = content.replace(old_tail, new_tail, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/ffmpeg_writer.py")


def update_video_stream():
    path = os.path.join(APP, "roop", "video_stream.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Audio in NVHardwareVideoWriter
    old_audio = """        if self.audio_source and os.path.exists(self.audio_source):
            cmd.extend([
                "-i",
                os.path.abspath(self.audio_source),
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
            ])
        else:
            cmd.extend(["-map", "0:v:0"])"""
    new_audio = """        if self.audio_source and os.path.exists(self.audio_source):
            cmd.extend([
                "-i",
                os.path.abspath(self.audio_source),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
            ])
        else:
            cmd.extend(["-map", "0:v:0", "-an"])"""
    if old_audio in content:
        content = content.replace(old_audio, new_audio, 1)

    # Faststart
    old_cmd_end = 'cmd.extend(["-pix_fmt", "yuv420p", self.output_path])'
    new_cmd_end = '''cmd.extend(["-pix_fmt", "yuv420p"])
        if self.output_path.lower().endswith((".mp4", ".mov", ".m4v")):
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(self.output_path)'''
    if old_cmd_end in content:
        content = content.replace(old_cmd_end, new_cmd_end, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/video_stream.py")


def update_optimized_processor():
    path = os.path.join(APP, "roop", "optimized_processor.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_audio_in = """        if self.audio_source:
            command.extend(["-i", self.audio_source, "-map", "0:v:0", "-map", "1:a:0?"])
        command.extend(["-c:v", codec])"""
    new_audio_in = """        if self.audio_source:
            command.extend(["-i", self.audio_source, "-map", "0:v:0", "-map", "1:a:0?"])
        else:
            command.extend(["-an"])
        command.extend(["-c:v", codec])"""
    if old_audio_in in content:
        content = content.replace(old_audio_in, new_audio_in, 1)

    old_audio_codec = """        if self.audio_source:
            command.extend(["-c:a", "copy"])
        command.append(self.path)"""
    new_audio_codec = """        if self.audio_source:
            command.extend(["-c:a", "aac", "-b:a", "192k"])
        if self.path.lower().endswith((".mp4", ".mov", ".m4v")):
            command.extend(["-movflags", "+faststart"])
        command.append(self.path)"""
    if old_audio_codec in content:
        content = content.replace(old_audio_codec, new_audio_codec, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/optimized_processor.py")


def update_segment_writer():
    path = os.path.join(APP, "roop", "segment_writer.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_concat = """            cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c", "copy", self.target_video]"""
    new_concat = """            cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c", "copy"]
            if self.target_video.lower().endswith((".mp4", ".mov", ".m4v")):
                cmd.extend(["-movflags", "+faststart"])
            cmd.append(self.target_video)"""
    if old_concat in content:
        content = content.replace(old_concat, new_concat, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/roop/segment_writer.py")


def update_routes_queue():
    path = os.path.join(APP, "routes_queue.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_cmd = """        cmd = [FFMPEG_BINARY, "-hide_banner", "-y", "-f", "concat", "-safe", "0",
               "-i", listing, "-c", "copy", dest]"""
    new_cmd = """        cmd = [FFMPEG_BINARY, "-hide_banner", "-y", "-f", "concat", "-safe", "0",
               "-i", listing, "-c", "copy"]
        if dest.lower().endswith((".mp4", ".mov", ".m4v")):
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(dest)"""
    if old_cmd in content:
        content = content.replace(old_cmd, new_cmd, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/routes_queue.py")


def update_post_swap():
    path = os.path.join(APP, "post_swap.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_classical = """    cmd = ([FFMPEG_BINARY, '-hide_banner', '-y', '-i', path, '-vf', vf,
            '-c:v', enc] + _rate_control(enc, q) + ['-c:a', 'copy', tmp])"""
    new_classical = """    has_audio = bool(util.audio_sample_rate(path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    cmd = ([FFMPEG_BINARY, '-hide_banner', '-y', '-i', path, '-vf', vf,
            '-c:v', 'libx264' if enc == 'libx264' else enc] + _rate_control(enc, q) +
           ['-pix_fmt', 'yuv420p', '-movflags', '+faststart'] + audio_flags + [tmp])"""
    if old_classical in content:
        content = content.replace(old_classical, new_classical, 1)

    old_minterpolate = """    cmd = ([FFMPEG_BINARY, "-hide_banner", "-y", "-i", path, "-vf", vf,
            "-c:v", enc] + _rate_control(enc, q) + ["-c:a", "copy", tmp])"""
    new_minterpolate = """    has_audio = bool(util.audio_sample_rate(path))
    audio_flags = ['-c:a', 'aac', '-b:a', '192k'] if has_audio else ['-an']
    cmd = ([FFMPEG_BINARY, "-hide_banner", "-y", "-i", path, "-vf", vf,
            "-c:v", 'libx264' if enc == 'libx264' else enc] + _rate_control(enc, q) +
           ['-pix_fmt', 'yuv420p', '-movflags', '+faststart'] + audio_flags + [tmp])"""
    if old_minterpolate in content:
        content = content.replace(old_minterpolate, new_minterpolate, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/post_swap.py")


def update_routes_extras():
    path = os.path.join(APP, "routes_extras.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update extras_apply to supplement cv2.VideoWriter with FFmpeg re-encode
    old_apply_block = """    first = _process_frame(first_frame)
    oh, ow = first.shape[:2]
    outpath = os.path.join(out_dir, "edited_" + os.path.splitext(os.path.basename(path))[0] + ".mp4")
    writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    for i in range(1, total + 1):
        fr = get_video_frame(path, i)
        if fr is None:
            continue
        writer.write(_process_frame(fr))
    writer.release()
    return {"path": outpath, "kind": "video"}"""

    new_apply_block = """    first = _process_frame(first_frame)
    oh, ow = first.shape[:2]
    outpath = os.path.join(out_dir, "edited_" + os.path.splitext(os.path.basename(path))[0] + ".mp4")
    raw_tmp = os.path.join(out_dir, f".raw_edited_{os.path.splitext(os.path.basename(path))[0]}_{ow}x{oh}.mp4")
    writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    for i in range(1, total + 1):
        fr = get_video_frame(path, i)
        if fr is None:
            continue
        writer.write(_process_frame(fr))
    writer.release()
    from roop.util_ffmpeg import finalize_web_video
    if not finalize_web_video(raw_tmp, outpath, audio_source=path, delete_raw=True):
        if os.path.isfile(raw_tmp):
            os.replace(raw_tmp, outpath)
    return {"path": outpath, "kind": "video"}"""

    if old_apply_block in content:
        content = content.replace(old_apply_block, new_apply_block, 1)

    # 2. Update extras_enhance to supplement cv2.VideoWriter with FFmpeg re-encode
    old_enhance_block = """        writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        writer.write(out_first)
        for i in range(2, total + 1):
            fr = get_video_frame(path, i)
            if fr is None:
                continue
            res = proc.Run(fr)
            if res.shape[:2] != (oh, ow):
                res = cv2.resize(res, (ow, oh))
            writer.write(res)
        writer.release()
        return {"path": outpath, "kind": "video"}"""

    new_enhance_block = """        raw_tmp = os.path.join(out_dir, f".raw_{operation}_{subtype}_{stem}.mp4")
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
        writer.release()
        from roop.util_ffmpeg import finalize_web_video
        if not finalize_web_video(raw_tmp, outpath, audio_source=path, delete_raw=True):
            if os.path.isfile(raw_tmp):
                os.replace(raw_tmp, outpath)
        return {"path": outpath, "kind": "video"}"""

    if old_enhance_block in content:
        content = content.replace(old_enhance_block, new_enhance_block, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/routes_extras.py")


if __name__ == "__main__":
    update_util_ffmpeg()
    update_ffmpeg_writer()
    update_video_stream()
    update_optimized_processor()
    update_segment_writer()
    update_routes_queue()
    update_post_swap()
    update_routes_extras()
    print("\nAll files successfully updated!")
