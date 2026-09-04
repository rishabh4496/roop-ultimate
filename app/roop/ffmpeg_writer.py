"""
FFMPEG_Writer - write set of frames to video file

original from
https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_writer.py

removed unnecessary dependencies

The MIT License (MIT)

Copyright (c) 2015 Zulko
Copyright (c) 2023 Janvarev Vladislav
"""

import os
import subprocess as sp

PIPE = -1
STDOUT = -2
DEVNULL = -3

from roop.ffmpeg_path import ffmpeg_binary

# Resolved rather than assumed: a bare "ffmpeg" only works inside a
# Pinokio-managed shell, and outside one the encoder pre-flight aborted
# every video render with "video encoder unavailable" on a machine whose
# ffmpeg was installed and working.  See roop/ffmpeg_path.py.
FFMPEG_BINARY = ffmpeg_binary()


def probe_encoder(codec="libx265", crf=14, timeout=30):
    """Pre-flight check that the ffmpeg encoder can actually launch and encode.

    Runs a tiny synthetic encode (a 3-frame lavfi source) with the *same* codec
    that the real render will use, capturing exit code and stderr with a hard
    timeout. This catches failures in seconds — BEFORE the long analysis pass —
    so a broken encoder aborts the run early instead of silently hanging the
    frame pipe mid-render.

    The classic Windows trigger: Smart App Control blocks an unsigned ffmpeg DLL
    (e.g. avdevice-62.dll) at process startup, killing the encoder. Because that
    block happens at launch/DLL-load time, this lavfi probe reproduces it without
    needing the real stdin pipe.

    Returns (ok: bool, message: str). message is empty on success.
    """
    import tempfile
    from roop.util_ffmpeg import clamp_quality
    crf = clamp_quality(codec, crf)
    tmp = os.path.join(tempfile.gettempdir(), f"roop_encoder_probe_{os.getpid()}.mp4")
    cmd = [
        FFMPEG_BINARY, '-hide_banner', '-loglevel', 'error', '-y',
        # 256x256 stays above NVENC's minimum supported frame dimensions
        # (smaller sizes fail hevc_nvenc's encoder init and false-negative here).
        '-f', 'lavfi', '-i', 'testsrc=size=256x256:rate=25:duration=1',
        '-frames:v', '3', '-vcodec', codec,
    ]
    if codec in ('h264_nvenc', 'hevc_nvenc'):
        cmd.extend(['-rc', 'vbr', '-cq', str(crf), '-preset', 'p5', '-tune', 'hq'])
    else:
        cmd.extend(['-crf', str(crf)])
    cmd.extend(['-pix_fmt', 'yuv420p', tmp])

    popen_params = {"stdout": sp.PIPE, "stderr": sp.PIPE, "stdin": DEVNULL}
    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    try:
        proc = sp.Popen(cmd, **popen_params)
    except FileNotFoundError:
        return (False, f"ffmpeg binary '{FFMPEG_BINARY}' was not found on PATH.")
    except Exception as e:
        return (False, f"could not launch ffmpeg: {e}")

    try:
        _, err = proc.communicate(timeout=timeout)
    except sp.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate()
        except Exception:
            pass
        return (False, "the ffmpeg encoder timed out during warm-up — it launched "
                       "but never made progress. On Windows this is usually Smart "
                       "App Control blocking an unsigned ffmpeg DLL.")
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    if proc.returncode != 0:
        detail = (err or b"").decode('utf-8', 'replace').strip()
        return (False, detail or f"the ffmpeg encoder exited with code {proc.returncode}.")
    return (True, "")


class FFMPEG_VideoWriter:
    """ A class for FFMPEG-based video writing.

    A class to write videos using ffmpeg. ffmpeg will write in a large
    choice of formats.

    Parameters
    -----------

    filename
      Any filename like 'video.mp4' etc. but if you want to avoid
      complications it is recommended to use the generic extension
      '.avi' for all your videos.

    size
      Size (width,height) of the output video in pixels.

    fps
      Frames per second in the output video file.

    codec
      FFMPEG codec. It seems that in terms of quality the hierarchy is
      'rawvideo' = 'png' > 'mpeg4' > 'libx264'
      'png' manages the same lossless quality as 'rawvideo' but yields
      smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
      of accepted codecs.

      Note for default 'libx264': by default the pixel format yuv420p
      is used. If the video dimensions are not both even (e.g. 720x405)
      another pixel format is used, and this can cause problem in some
      video readers.

    audiofile
      Optional: The name of an audio file that will be incorporated
      to the video.

    preset
      Sets the time that FFMPEG will take to compress the video. The slower,
      the better the compression rate. Possibilities are: ultrafast,superfast,
      veryfast, faster, fast, medium (default), slow, slower, veryslow,
      placebo.

    bitrate
      Only relevant for codecs which accept a bitrate. "5000k" offers
      nice results in general.

    """

    # A hardware encoder fails as a CLASS on a machine that is out of memory:
    # NVENC allocates its buffers when the encoder OPENS, so on a loaded box the
    # open itself fails ("CreateInputBuffer failed: out of memory") and the
    # process exits before a single packet is written. probe_encoder() cannot
    # catch that — it runs once at frame 0 when memory is still free, while
    # SegmentedVideoWriter opens a NEW encoder every ROOP_RESUME_CHUNK frames and
    # any one of them can be the one that fails. Measured: a 40934-frame render
    # on a 16 GB / RTX 3060 box died at frame 5001, the sixth segment boundary.
    _SW_FALLBACK = {
        'hevc_nvenc': 'libx265', 'h264_nvenc': 'libx264', 'av1_nvenc': 'libx265',
        'hevc_qsv': 'libx265', 'h264_qsv': 'libx264',
        'hevc_amf': 'libx265', 'h264_amf': 'libx264',
    }

    def __init__(self, filename, size, fps, codec="libx265", crf=14, audiofile=None,
                 preset="faster", bitrate=None,
                 logfile=None, threads=None, ffmpeg_params=None,
                 colorspace=None):

        if logfile is None:
            logfile = sp.PIPE

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]
        # Held so _build_cmd can be re-run for a different codec: the fallback
        # cannot just swap the codec name in the finished argv, because the
        # rate-control flags differ (-rc/-cq/-tune for NVENC vs -crf/-preset).
        self._size = size
        self._fps = fps
        self._crf = crf
        self._audiofile = audiofile
        self._preset = preset
        self._bitrate = bitrate
        self._logfile = logfile
        # bt709 preserves the established output matrix. "off" is an explicit
        # escape hatch for callers whose frames already have the desired
        # display-space conversion and need no extra full-frame pass.
        self._colorspace = colorspace
        if threads is None:
            # Explicit constructor input and explicit process environment win;
            # otherwise consume the bounded workload hint.  This keeps FFmpeg
            # from creating an unbounded CPU pool beside Python workers, ORT,
            # and OpenCV.
            threads = os.environ.get('ROOP_FFMPEG_THREADS')
            if threads is None or str(threads).strip().lower() in ('', 'auto', 'default'):
                threads = os.environ.get('ROOP_RUNTIME_FFMPEG_THREADS')
            try:
                threads = max(1, min(4, int(threads)))
            except (TypeError, ValueError):
                threads = None
        self._threads = threads
        self._ffmpeg_params = ffmpeg_params
        self._frames_written = 0
        self._fell_back = False
        self._spawn(codec)

    def _build_cmd(self, codec):
        size, fps = self._size, self._fps
        audiofile, bitrate = self._audiofile, self._bitrate
        threads, ffmpeg_params = self._threads, self._ffmpeg_params
        logfile = self._logfile
        w = size[0] - 1 if size[0] % 2 != 0 else size[0]
        h = size[1] - 1 if size[1] % 2 != 0 else size[1]

        # order is important
        cmd = [
            FFMPEG_BINARY,
            '-hide_banner',
            '-hwaccel', 'auto',
            '-y',
            '-loglevel', 'error' if logfile == sp.PIPE else 'info',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '%dx%d' % (size[0], size[1]),
            #'-pix_fmt', 'rgba' if withmask else 'rgb24',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-an', '-i', '-'
        ]

        if audiofile is not None:
            cmd.extend([
                '-i', audiofile,
                '-acodec', 'copy'
            ])

        cmd.extend(['-vcodec', codec])
        # Out-of-range quality makes ffmpeg exit before a single frame is written,
        # so clamp to what this codec accepts rather than failing the whole render.
        from roop.util_ffmpeg import clamp_quality
        crf = clamp_quality(codec, self._crf)
        is_nvenc = codec in ('h264_nvenc', 'hevc_nvenc')
        if is_nvenc:
            # NVENC has no -crf; -cq is the constant-quality equivalent (same 0-51
            # scale), so reuse the configured quality directly. Preset p1(fastest)
            # ..p7(slowest/best); p5 + "-tune hq" under VBR is a balanced default.
            # Encoding runs on the GPU's dedicated NVENC engine — off the CPU and
            # separate from CUDA inference. Override the preset with ROOP_NVENC_PRESET.
            nvenc_preset = os.environ.get('ROOP_NVENC_PRESET', 'p5').strip().lower()
            if nvenc_preset not in {f'p{i}' for i in range(1, 8)}:
                nvenc_preset = 'p5'
            cmd.extend(['-rc', 'vbr', '-cq', str(crf), '-preset', nvenc_preset, '-tune', 'hq'])
        else:
            cmd.extend(['-crf', str(crf)])

        # For libx264 / libx265 the preset trades encode SPEED for FILE SIZE at a
        # fixed CRF — the rate control holds perceptual quality constant, so a
        # faster preset speeds up encoding with no visible quality loss (just
        # slightly larger files). Only these encoders use x264-style preset
        # names; vp9 (-deadline) and nvenc (p1-p7) are left untouched. Override
        # the default with env ROOP_ENCODER_PRESET.
        if codec in ('libx264', 'libx265'):
            _valid = {'ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                      'medium', 'slow', 'slower', 'veryslow', 'placebo'}
            _preset = os.environ.get(
                'ROOP_ENCODER_PRESET',
                os.environ.get('ROOP_RUNTIME_ENCODER_PRESET', self._preset or 'faster')
            ).strip().lower()
            if _preset not in _valid:
                _preset = 'faster'
            cmd.extend(['-preset', _preset])
        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)
        if bitrate is not None:
            cmd.extend([
                '-b', bitrate
            ])

        # Scale odd dimensions when required, and convert + TAG the output
        # colour space.
        #
        # These were an if/elif, which meant an ODD-SIZED render took the scale
        # branch and skipped the colour conversion entirely -- shipping a file
        # with color_space / color_primaries / color_transfer / color_range all
        # `unknown`, so the player picks the matrix by guesswork. Measured on a
        # 1280x720 chart through real encode/decode
        # (tests/measure_color_roundtrip.py):
        #
        #   arm              mean|d|  max|d|  skin|d|  container tags
        #   even, converted    1.708       5    2.083  bt709 on all four
        #   even, no filter    1.792       5    2.250  all unknown
        #   ODD, scale only    1.139       5    1.734  all unknown   <- shipped
        #   ODD, scale+convert 1.028       4    1.474  bt709 on all four
        #
        # So the two are chained now rather than alternated: the scale runs
        # first and the conversion always follows it. The even-dimension path is
        # unchanged (1.708 both ways, same tags) -- this only reaches the odd
        # case, where it is better on all three axes.
        colorspace = self._colorspace
        if colorspace is None:
            colorspace = os.environ.get('ROOP_FFMPEG_COLORSPACE', 'bt709')
        colorspace = str(colorspace).strip().lower()
        convert = colorspace not in ('off', 'none', 'passthrough', '0', 'false')
        filters = []
        if w != size[0] or h != size[1]:
            filters.append(f'scale={w}:{h}')
        if convert:
            filters.append('colorspace=bt709:iall=bt601-6-625:fast=1')
        if filters:
            cmd.extend(['-vf', ','.join(filters)])
        if convert:
            # State the tags rather than relying on the filter to propagate
            # frame metadata into the muxer. Measured free: with the filter
            # present the round-trip error is identical to three decimals
            # (1.708 either way) and the tags come out the same, so this only
            # removes the dependency on that propagation -- which is exactly
            # what the odd-dimension branch lost.
            #
            # `tv` is limited range, which is what yuv420p delivery and the
            # legacy filter both already produce; a full-range ('pc') file
            # would need the decode side to agree or every level shifts.
            cmd.extend(['-colorspace', 'bt709', '-color_primaries', 'bt709',
                        '-color_trc', 'bt709', '-color_range', 'tv'])

        if threads is not None:
            cmd.extend(["-threads", str(threads)])

        cmd.extend([
            '-pix_fmt', 'yuv420p',

        ])
        cmd.extend([
            self.filename
        ])
        return cmd

    def _spawn(self, codec):
        self.codec = codec
        cmd = self._build_cmd(codec)

        test = str(cmd)
        print(test)

        popen_params = {"stdout": DEVNULL,
                        "stderr": self._logfile,
                        "stdin": sp.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        self.proc = sp.Popen(cmd, **popen_params)

    def _retry_as_software(self):
        """Respawn a dead HARDWARE encoder as its software equivalent.

        Only while nothing has been written, and only once. Both conditions are
        load-bearing: swapping codec mid-file would corrupt the segment, and a
        software encoder that also dies is a real failure that must surface
        rather than loop. Returns True if the caller should retry the write.
        """
        if self._fell_back or self._frames_written:
            return False
        sw = self._SW_FALLBACK.get(self.codec)
        if not sw:
            return False
        err = b""
        try:
            if self.proc is not None and self.proc.stderr is not None:
                err = self.proc.stderr.read() or b""
        except Exception:
            pass
        try:
            from roop.procmgr_runtime import bar_write as _say
        except Exception:
            _say = print
        failed = self.codec
        self._fell_back = True
        try:
            self._spawn(sw)
        except Exception as exc:
            _say(f"[Encoder] {failed} died and the {sw} fallback could not "
                 f"launch either: {exc}")
            return False
        _say(f"[Encoder] {failed} failed to start — encoding this segment with "
             f"{sw} on the CPU instead. This is almost always the machine being "
             f"out of memory rather than a bad file; if it repeats, lower "
             f"ROOP_STAB_CHUNK_MB or turn the stabilize_* options off.")
        detail = err.decode('utf-8', 'replace').strip()
        if detail:
            _say(f"[Encoder] {failed} said: {detail.splitlines()[-1][:300]}")
        return True


    def write_frame(self, img_array):
        """ Writes one frame in the file."""
        # Fail fast if the encoder process has already died (e.g. Windows Smart
        # App Control blocked an unsigned ffmpeg DLL at launch, or the codec
        # failed to initialise). Without this, writing into the dead pipe can
        # block forever and the whole render hangs silently — losing the entire
        # analysis pass with no error.
        if self.proc is None or self.proc.poll() is not None:
            if self._retry_as_software():
                return self.write_frame(img_array)
            rc = None if self.proc is None else self.proc.returncode
            ffmpeg_error = b""
            try:
                if self.proc is not None and self.proc.stderr is not None:
                    ffmpeg_error = self.proc.stderr.read() or b""
            except Exception:
                pass
            raise IOError(
                "Roop Ultimate error: the ffmpeg encoder process exited "
                f"unexpectedly (code {rc}) before the video was finished.\n\n"
                "On Windows this is usually Smart App Control blocking an "
                "unsigned ffmpeg DLL (e.g. avdevice-62.dll). Re-run the job, or "
                "turn off Smart App Control in Windows Security → App & "
                "browser control.\n\nffmpeg said:\n"
                + ffmpeg_error.decode('utf-8', 'replace'))
        try:
            # A contiguous numpy frame can be handed to the pipe as a buffer
            # view, avoiding the per-frame bytes allocation/copy.  Rotated or
            # otherwise strided frames are deliberately kept on the safe
            # fallback: BufferedWriter requires a contiguous buffer, and the
            # explicit tobytes() preserves the old output ordering/format.
            frame_view = memoryview(img_array)
            self.proc.stdin.write(
                frame_view if frame_view.c_contiguous else img_array.tobytes())
            self._frames_written += 1
        except IOError as err:
            # Same reasoning as the poll() check above: a hardware encoder that
            # never accepted a frame is a resource failure, not a bad file.
            if self._retry_as_software():
                return self.write_frame(img_array)
            _, ffmpeg_error = self.proc.communicate()
            error = (str(err) + ("\n\nRoop Ultimate error: FFMPEG encountered "
                                 "the following error while writing file %s:"
                                 "\n\n %s" % (self.filename, str(ffmpeg_error))))

            if b"Unknown encoder" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed because FFMPEG didn't find the specified "
                  "codec for video encoding (%s). Please install "
                  "this codec or change the codec when calling "
                  "write_videofile. For instance:\n"
                  "  >>> clip.write_videofile('myvid.webm', codec='libvpx')")%(self.codec)

            elif b"incorrect codec parameters ?" in ffmpeg_error:

                 error = error+("\n\nThe video export "
                  "failed, possibly because the codec specified for "
                  "the video (%s) is not compatible with the given "
                  "extension (%s). Please specify a valid 'codec' "
                  "argument in write_videofile. This would be 'libx264' "
                  "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx for webm. "
                  "Another possible reason is that the audio codec was not "
                  "compatible with the video codec. For instance the video "
                  "extensions 'ogv' and 'webm' only allow 'libvorbis' (default) as a"
                  "video codec."
                  )%(self.codec, self.ext)

            elif  b"encoder setup failed" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed, possibly because the bitrate you specified "
                  "was too high or too low for the video codec.")

            elif b"Invalid encoder type" in ffmpeg_error:

                error = error + ("\n\nThe video export failed because the codec "
                  "or file extension you provided is not a video")


            raise IOError(error)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()

        self.proc = None

    # Support the Context Manager protocol, to ensure that resources are cleaned up.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()



    
