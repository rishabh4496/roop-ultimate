
import os
import subprocess
import roop.globals
import roop.utilities as util
from roop.ffmpeg_path import ffmpeg_binary

from typing import List, Any

def run_ffmpeg(args: List[str]) -> bool:
    commands = [ffmpeg_binary(), '-hide_banner', '-hwaccel', 'auto', '-y', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    print("Running ffmpeg")
    try:
        kwargs: dict = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
        }
        # CREATE_NO_WINDOW prevents the asyncio ProactorEventLoop on Windows from
        # raising ConnectionResetError (WinError 10054) when the subprocess pipe closes.
        if os.name == 'nt':
            kwargs['creationflags'] = 0x08000000
        result = subprocess.run(commands, **kwargs)
        if result.returncode != 0:
            print("Running ffmpeg failed! Commandline:")
            print(" ".join(commands))
            if result.stdout:
                print("FFmpeg output:")
                print(result.stdout.decode(errors='replace'))
            return False
        return True
    except Exception as e:
        print("Running ffmpeg failed! Commandline:")
        print(" ".join(commands))
        print(f"Error: {e}")
    return False



# Highest quality value each encoder family accepts. x264/x265 and NVENC's -cq
# top out at 51; the VP9/AV1 family goes to 63. The settings slider used to offer
# 0-100 with nothing clamping it.
#
# Measured, because the encoders do NOT agree on what happens past the limit:
#   libx264   -crf 80 -> exits 0, output byte-identical to -crf 51 (clamps itself)
#   libx265   -crf 80 -> FAILS, no output          <- the default encoder
#   h264_nvenc/hevc_nvenc -cq 80 -> FAILS, no output
# So on the default (libx265) and on both GPU encoders an out-of-range slider
# value killed the render outright; on libx264 it silently did nothing. Clamping
# makes all of them behave like libx264.
_QUALITY_MAX = 51
_QUALITY_MAX_WIDE = 63
_WIDE_RANGE_CODECS = ('libvpx', 'vp9', 'aom', 'av1', 'svtav1')


def quality_max(codec: str) -> int:
    """Largest quality value *codec* accepts (0 = best quality for all of them)."""
    c = (codec or '').lower()
    return _QUALITY_MAX_WIDE if any(k in c for k in _WIDE_RANGE_CODECS) else _QUALITY_MAX


def clamp_quality(codec: str, quality) -> int:
    """Coerce *quality* into the range *codec* actually accepts."""
    try:
        q = int(round(float(quality)))
    except (TypeError, ValueError):
        q = 14
    return max(0, min(q, quality_max(codec)))


def _rate_control(codec: str, quality) -> List[str]:
    """Return the encoder rate-control args for *codec* at *quality*.

    libx264/libx265 (and most CPU encoders) use -crf; NVENC has no -crf and uses
    -cq under VBR with p1..p7 presets instead. Keeping this in one place means
    every re-encode path supports h264_nvenc/hevc_nvenc (GPU encode) correctly.
    """
    q = clamp_quality(codec, quality)
    if codec in ('h264_nvenc', 'hevc_nvenc'):
        preset = os.environ.get('ROOP_NVENC_PRESET', 'p5').strip().lower()
        if preset not in {f'p{i}' for i in range(1, 8)}:
            preset = 'p5'
        return ['-rc', 'vbr', '-cq', str(q), '-preset', preset, '-tune', 'hq']
    return ['-crf', str(q)]


def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    if reencode:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video, '-c:v', roop.globals.video_encoder, '-c:a', 'aac', '-frames:v', str(num_frames), cut_video])
    else:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video,  '-frames:v', str(num_frames), '-c:v' ,'copy','-c:a' ,'copy', cut_video])

def join_videos(videos: List[str], dest_filename: str, simple: bool):
    if simple:
        txtfilename = util.resolve_relative_path('../temp')
        txtfilename = os.path.join(txtfilename, 'joinvids.txt')
        with open(txtfilename, "w", encoding="utf-8") as f:
            for v in videos:
                 v = v.replace('\\', '/')
                 f.write(f"file {v}\n")
        commands = ['-f', 'concat', '-safe', '0', '-i', f'{txtfilename}', '-vcodec', 'copy', f'{dest_filename}']
        run_ffmpeg(commands)

    else:
        inputs = []
        filter = ''
        for i,v in enumerate(videos):
            inputs.append('-i')
            inputs.append(v)
            filter += f'[{i}:v:0][{i}:a:0]'
        run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])    

        #     filter += f'[{i}:v:0][{i}:a:0]'
        # run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])    



def _extract_frames_from_animated_webp(target_path: str, trim_frame_start, trim_frame_end, temp_directory_path: str) -> bool:
    """Extract frames from animated WebP using PIL/Pillow.

    FFmpeg's native webp_pipe demuxer skips ANIM/ANMF chunks and cannot decode
    animated WebP files, producing zero frames.  Pillow handles them correctly.
    Frames are written as the configured output_image_format (typically png).
    """
    import numpy as np
    import cv2
    from PIL import Image

    try:
        with Image.open(target_path) as img:
            n_frames = getattr(img, 'n_frames', 1)
            start = int(trim_frame_start) if trim_frame_start is not None else 0
            end   = int(trim_frame_end)   if trim_frame_end   is not None else n_frames
            end   = min(end, n_frames)

            frame_num = 1
            for i in range(start, end):
                img.seek(i)
                frame_rgb = np.array(img.convert('RGB'))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(
                    temp_directory_path,
                    f'{frame_num:06d}.{roop.globals.CFG.output_image_format}',
                )
                cv2.imwrite(out_path, frame_bgr)
                frame_num += 1

        extracted = frame_num - 1
        print(f'Extracted {extracted} frames from animated WebP via PIL')
        return extracted > 0
    except Exception as e:
        print(f'PIL animated WebP frame extraction failed: {e}')
        return False


def extract_frames(target_path : str, trim_frame_start, trim_frame_end, fps : float) -> bool:
    util.create_temp(target_path)
    temp_directory_path = util.get_temp_directory_path(target_path)

    # FFmpeg's native webp_pipe demuxer cannot decode animated WebP (ANIM/ANMF chunks).
    # Detect animated webp and fall back to PIL-based extraction.
    if target_path.lower().endswith('.webp') and util.is_animated_webp(target_path):
        return _extract_frames_from_animated_webp(target_path, trim_frame_start, trim_frame_end, temp_directory_path)

    fps = util.constant_frame_rate(fps)
    commands = ['-i', target_path, '-q:v', '1', '-pix_fmt', 'rgb24', ]
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',' + util.cfr_video_filter(fps) ])
    else:
        commands.extend(['-vf', util.cfr_video_filter(fps)])
    # A numbered image sequence has no reliable input timestamps. Force CFR
    # here, rather than passing through VFR PTS and losing/duplicating frames.
    commands.extend(['-vsync', 'cfr', '-fps_mode', 'cfr',
                     os.path.join(temp_directory_path, '%06d.' + roop.globals.CFG.output_image_format)])
    return run_ffmpeg(commands)


def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    # scale=trunc(iw/2)*2:trunc(ih/2)*2 rounds odd dimensions down to even, which is
    # required by yuv420p / libx264. Without this, frames with odd width or height
    # cause ffmpeg to fail silently and produce an empty (corrupt) output file.
    fps = util.constant_frame_rate(fps)
    vf = util.cfr_video_filter(fps) + ',scale=trunc(iw/2)*2:trunc(ih/2)*2,colorspace=bt709:iall=bt601-6-625:fast=1'
    run_ffmpeg(['-framerate', str(fps), '-i', os.path.join(temp_directory_path, f'%06d.{roop.globals.CFG.output_image_format}'), '-c:v', roop.globals.video_encoder] + _rate_control(roop.globals.video_encoder, roop.globals.video_quality) + ['-pix_fmt', 'yuv420p', '-vf', vf, '-r', str(fps), '-vsync', 'cfr', '-fps_mode', 'cfr', '-y', dest_filename])
    return dest_filename


def create_gif_from_video(video_path: str, gif_path: str, target_fps: float = None):
    """Convert a video file to an optimised animated GIF.

    target_fps — if provided, use this frame rate instead of detecting it from
    the file.  Pass the known fps when converting from an intermediate temp MP4
    so we don't lose the original source timing through a second detect_fps call.
    """
    fps = target_fps if target_fps is not None else util.detect_fps(video_path)
    width, height = util.detect_dimensions(video_path)

    # Keep the larger dimension at its original size; auto-scale the other.
    if width >= height:
        scale = f'{width}:-1'
    else:
        scale = f'-1:{height}'

    run_ffmpeg(['-i', video_path, '-vf', f'fps={fps},scale={scale}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', '-loop', '0', gif_path])


def apply_media_transforms_gif(input_path: str, output_path: str,
                                vf_filters: list, target_fps=None) -> bool:
    """Re-encode an animated GIF with correct palette generation.

    FFmpeg's default GIF encoder uses a poor global palette that introduces
    colour artifacts on grayscale content.  This function uses the two-pass
    palettegen+paletteuse pipeline so the output palette is optimised for the
    actual frame content — exactly the same approach used by create_gif_from_video.

    vf_filters  - list of video filters to apply BEFORE palette generation
                  (e.g. crop, scale, transpose).  May be empty.
    target_fps  - if not None, a fps= filter is prepended so the frame-rate
                  of the output GIF matches the requested value.
    """
    all_filters = list(vf_filters)
    if target_fps is not None:
        all_filters.insert(0, f'fps={target_fps}')

    if all_filters:
        user_chain = ','.join(all_filters) + ','
    else:
        user_chain = ''

    # Two-pass palette approach in a single ffmpeg invocation using filtergraph.
    # The filter chain is: [user filters] → split → palettegen / paletteuse
    vf = f'{user_chain}split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse'
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-loop', '0', output_path])



def create_video_from_gif(gif_path: str, output_path):
    fps = util.detect_fps(gif_path)
    filter = """scale='trunc(in_w/2)*2':'trunc(in_h/2)*2',format=yuv420p,fps=10"""
    run_ffmpeg(['-i', gif_path, '-vf', f'"{filter}"', '-movflags', '+faststart', '-shortest', output_path])



def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
    scale_filter = (
        f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
        f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    )
    return run_ffmpeg(['-i', input_path, '-vf', scale_filter,
                       '-c:v', roop.globals.video_encoder,
                       *_rate_control(roop.globals.video_encoder, roop.globals.video_quality),
                       '-c:a', 'copy', output_path])


def rotate_media(input_path: str, output_path: str, transform: str) -> bool:
    transform_map = {
        "90° Clockwise":        "transpose=1",
        "90° Counter-clockwise": "transpose=2",
        "180°":                  "transpose=1,transpose=1",
        "Flip Horizontal":       "hflip",
        "Flip Vertical":         "vflip",
    }
    vf = transform_map.get(transform, "transpose=1")
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-c:a', 'copy', output_path])


def change_fps(input_path: str, output_path: str, fps: float) -> bool:
    return run_ffmpeg(['-i', input_path, '-vf', f'fps={fps}',
                       '-c:v', roop.globals.video_encoder,
                       *_rate_control(roop.globals.video_encoder, roop.globals.video_quality),
                       '-c:a', 'copy', output_path])


def crop_media(input_path: str, output_path: str,
               left_pct: float, right_pct: float,
               top_pct: float,  bottom_pct: float) -> bool:
    l, r, t, b = left_pct / 100, right_pct / 100, top_pct / 100, bottom_pct / 100
    crop_filter = (
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f})"
        f":in_w*{l:.4f}:in_h*{t:.4f}"
    )
    return run_ffmpeg(['-i', input_path, '-vf', crop_filter, '-c:a', 'copy', output_path])


def apply_media_transforms(input_path: str, output_path: str,
                           vf_filters: list, is_video: bool) -> bool:
    """Apply a list of -vf filters in a single ffmpeg pass."""
    if not vf_filters:
        return False
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    vf = ','.join(vf_filters)
    args = ['-i', input_path, '-vf', vf]
    if is_video:
        args += ['-c:v', codec, *_rate_control(codec, quality), '-c:a', 'copy']
    args.append(output_path)
    return run_ffmpeg(args)


def extract_audio_wav(input_path: str, output_wav_path: str, sample_rate: int = 16000) -> bool:
    """Mono PCM WAV at *sample_rate* Hz — the driving audio for lip-sync's
    feature-extraction pass. 16kHz is Whisper's native input rate.

    The only audio-decode entry point in this codebase: every other audio
    operation here is a stream-copy mux (restore_audio), never a real decode.
    Staying ffmpeg-subprocess-based keeps that the same for extraction too,
    rather than adding a librosa/soundfile dependency for one caller.
    """
    return run_ffmpeg(['-i', input_path, '-vn', '-ac', '1', '-ar', str(sample_rate),
                       '-f', 'wav', output_wav_path])


def crop_to_fill(source_w: int, source_h: int,
                 target_ratio_w: float, target_ratio_h: float) -> tuple:
    """Centered crop percentages (left, right, top, bottom) that bring
    source_w x source_h to the target_ratio_w:target_ratio_h aspect ratio,
    trimming whichever axis is oversized relative to that ratio — the
    "crop-to-fill" behaviour social platforms use, as opposed to letterboxing.
    All-zero when the source already matches the ratio, or on degenerate input.
    """
    if source_w <= 0 or source_h <= 0 or target_ratio_w <= 0 or target_ratio_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    source_ar = source_w / source_h
    target_ar = target_ratio_w / target_ratio_h
    if source_ar > target_ar:
        # Wider than the target ratio: trim width, keep full height.
        keep_w = source_h * target_ar
        trim_pct = max(0.0, (source_w - keep_w) / source_w * 100.0) / 2
        return (trim_pct, trim_pct, 0.0, 0.0)
    if source_ar < target_ar:
        # Taller than the target ratio: trim height, keep full width.
        keep_h = source_w / target_ar
        trim_pct = max(0.0, (source_h - keep_h) / source_h * 100.0) / 2
        return (0.0, 0.0, trim_pct, trim_pct)
    return (0.0, 0.0, 0.0, 0.0)


def export_preset(input_path: str, output_path: str,
                  target_w: int, target_h: int) -> bool:
    """Crop *input_path* to fill the target_w:target_h aspect ratio (centered,
    no letterboxing), scale to that exact resolution, and re-encode in one
    ffmpeg pass using the app's own encoder/quality settings."""
    from roop.capturer import probe_media_dimensions
    dims = probe_media_dimensions(input_path)
    if not dims:
        return False
    source_w, source_h = dims
    l, r, t, b = crop_to_fill(source_w, source_h, target_w, target_h)
    l, r, t, b = l / 100, r / 100, t / 100, b / 100
    filters = [
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f}):in_w*{l:.4f}:in_h*{t:.4f}",
        f"scale={target_w}:{target_h}",
    ]
    return apply_media_transforms(input_path, output_path, filters, is_video=True)


def apply_media_transforms_webp(input_path: str, output_path: str,
                                vf_filters: list, fps: float) -> bool:
    """Process animated webp: decode frames via PIL, pipe through ffmpeg with vf filters.

    FFmpeg cannot reliably decode animated webp files with malformed Exif headers.
    This function bypasses that by loading frames with Pillow and feeding raw BGR
    video into ffmpeg via stdin, applying any vf filters in a single pass.
    Output is always an mp4 (caller must ensure output_path has .mp4 extension).
    """
    import subprocess
    import numpy as np
    import cv2
    from PIL import Image

    try:
        frames = []
        width = height = 0
        with Image.open(input_path) as img:
            width, height = img.width, img.height
            for i in range(getattr(img, 'n_frames', 1)):
                img.seek(i)
                frame_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
    except Exception as e:
        print(f"apply_media_transforms_webp: failed to load frames: {e}")
        return False

    if not frames or width == 0 or height == 0:
        print("apply_media_transforms_webp: no frames or zero dimensions")
        return False

    # video_encoder/quality may be None if faceswap tab hasn't run yet — use safe defaults
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14

    # yuv420p requires even dimensions — round odd width/height down before encoding.
    even_scale = 'scale=trunc(iw/2)*2:trunc(ih/2)*2'
    user_vf = ','.join(vf_filters)
    vf = f'{user_vf},{even_scale}' if user_vf else even_scale
    cmd = [
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
    ]
    print(f"apply_media_transforms_webp: piping {len(frames)} frames @ {fps} fps")
    print(' '.join(cmd))

    # Concatenate all raw frame bytes up front so we can pass them via
    # communicate(input=...).  communicate() uses internal threads to write
    # stdin and drain stderr simultaneously, preventing the deadlock that occurs
    # when -loglevel debug fills the stderr pipe while we're still writing stdin.
    raw_data = b''.join(f.tobytes() for f in frames)

    try:
        popen_params = {
            'stdin':  subprocess.PIPE,
            'stdout': subprocess.DEVNULL,   # we don't read stdout
            'stderr': subprocess.PIPE,
        }
        if os.name == 'nt':
            popen_params['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, **popen_params)
        _, stderr = proc.communicate(input=raw_data)
        if proc.returncode != 0:
            print(f"apply_media_transforms_webp ffmpeg error:\n{stderr.decode(errors='replace')}")
        return proc.returncode == 0
    except Exception as e:
        print(f"apply_media_transforms_webp: subprocess failed: {e}")
        return False


def create_video_from_frames_dir(frames_dir: str, output_path: str, fps: float,
                                  image_format: str = 'png') -> bool:
    """Re-assemble a video from a directory of sequentially named frame images.

    Frames must follow the %06d.<image_format> naming convention that
    extract_frames() produces (e.g. 000001.png, 000002.png …).
    """
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
    ])


def create_gif_from_frames_dir(frames_dir: str, output_path: str, fps: float,
                                width: int, height: int,
                                image_format: str = 'png') -> bool:
    """Re-assemble an animated GIF from a directory of sequentially named frame images.

    Uses the two-pass palettegen+paletteuse pipeline for accurate colour reproduction.
    Frames must follow the %06d.<image_format> naming convention.
    """
    if width and height:
        scale = f'{width}:-1' if width >= height else f'-1:{height}'
    else:
        scale = 'iw:ih'   # no-op scale if dimensions are unknown
    vf = (
        f'scale={scale}:flags=lanczos,'
        f'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse'
    )
    return run_ffmpeg([
        '-r',   str(fps),
        '-i',   os.path.join(frames_dir, f'%06d.{image_format}'),
        '-vf',  vf,
        '-loop', '0',
        output_path,
    ])


def restore_audio(intermediate_video: str, original_video: str, trim_frame_start, trim_frame_end, final_video: str) -> bool:
    """Mux audio from *original_video* into *intermediate_video*, writing *final_video*.

    Uses -map 0:v:0 (video from the processed clip) and -map 1:a:0? (audio from
    the original source, optional so it silently succeeds on source-less files).
    trim_frame_start / trim_frame_end are used to seek the audio source to the
    correct position when the original was trimmed before processing.
    Also carries the original container's GLOBAL metadata (-map_metadata 1);
    stream-level rotation side data is deliberately left alone, see below.
    Returns True on success, False on failure.
    """
    fps = util.constant_frame_rate(util.detect_fps(original_video))

    # Seek the audio source to match any trim that was applied before processing.
    audio_seek = []
    if trim_frame_start is not None:
        audio_seek += ['-ss', format(trim_frame_start / fps, ".2f")]
    else:
        audio_seek += ['-ss', '0']
    duration = None
    if trim_frame_end is not None:
        duration = max(0.0, (trim_frame_end - (trim_frame_start or 0)) / fps)

    rate = util.audio_sample_rate(original_video)
    if rate is not None:
        print(f"[A/V] stream-copying {rate} Hz source audio with CFR video PTS")

    commands = (
        ['-i', intermediate_video]
        + audio_seek
        + ['-i', original_video,
           '-map', '0:v:0',
           '-map', '1:a:0?',
           # Carry the ORIGINAL container's global metadata (creation_time,
           # title, artist, ...) onto the render. Input 1 is the original, so
           # this is `1` and not `0` -- input 0 is the freshly encoded video,
           # whose only tags are the ones ffmpeg just invented for it, and
           # ffmpeg's default (map from input 0) is what silently dropped
           # creation_time on every render before this.
           #
           # GLOBAL metadata only, deliberately. Stream-level side data --
           # in particular the display matrix that encodes rotation -- is NOT
           # touched by this flag. That matters: the decode pipe runs
           # `-noautorotate` (see nvdec_reader) so frames are processed in
           # RAW orientation, and stamping the source's rotation onto them
           # would make a player rotate footage that was never un-rotated.
           # Do not widen this to `-map_metadata:s:v`.
           '-map_metadata', '1',
           '-c:v', 'copy', '-c:a', 'copy',
           '-vsync', 'cfr', '-fps_mode', 'cfr',
           '-avoid_negative_ts', 'make_zero',
           '-shortest',
          ]
    )
    if duration is not None:
        commands += ['-t', format(duration, '.6f')]
    commands += [final_video]
    return run_ffmpeg(commands)
