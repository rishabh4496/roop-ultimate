"""What happens when a hardware encoder fails to OPEN at segment 2 or later of a segmented render?

    env/Scripts/python.exe tests/probe_segment_encoder_failure.py

Real ffmpeg throughout. Segment 1 encodes with the real ``h264_nvenc``. For segment 2 the spawned process is a codec
this ffmpeg does not have (``h264_qsv``): it dies before writing a packet, exactly as an NVENC open that hits the
session limit / out-of-memory does, while the writer still reports ``codec == 'h264_nvenc'`` so
``FFMPEG_VideoWriter._retry_as_software`` runs for real.

Result 2026-10-26 (RTX 4070, ffmpeg 8.1.2): the render DIES at frame 51 with ``OSError: [Errno 32] Broken pipe``
and the software fallback never spawns (writers made: nvenc, nvenc). The retry is refused because
``_frames_written`` is already 1: the first ``write`` into the dead pipe succeeds into the OS buffer, and the death is
only seen on the next write. Even if it did retry, ``SegmentedVideoWriter.write_frame`` would then raise "segmented
video encoder changed from h264_nvenc to libx264; refusing to concat mixed-codec segments" (see
probe_mixed_encoder_concat.py for why that guard is right for HEVC). Committed segments survive for a resume.
NOT fixed here; see docs/CHANGELOG.md 2026-10-26.
"""
import os
import sys
import tempfile

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ["ROOP_RESUME_CHUNK"] = "50"

import roop.globals  # noqa: E402
roop.globals.processing = True
from roop import ffmpeg_writer, segment_writer  # noqa: E402

made = []


class FlakyWriter(ffmpeg_writer.FFMPEG_VideoWriter):
    def _spawn(self, codec):
        made.append(codec)
        if len(made) == 2 and codec == "h264_nvenc":
            self.codec = codec                          # what the writer believes it is running
            cmd = self._build_cmd("h264_qsv")           # what actually starts: an encoder that is not there
            self.proc = ffmpeg_writer.sp.Popen(cmd, stdout=ffmpeg_writer.DEVNULL, stderr=ffmpeg_writer.sp.PIPE,
                                               stdin=ffmpeg_writer.sp.PIPE,
                                               creationflags=0x08000000 if os.name == "nt" else 0)
            return
        super()._spawn(codec)


segment_writer.FFMPEG_VideoWriter = FlakyWriter
segment_writer.hardware_stream_enabled = lambda: False

d = tempfile.mkdtemp(prefix="segfail_")
w = segment_writer.SegmentedVideoWriter(os.path.join(d, "out.mp4"), (320, 240), 25, codec="h264_nvenc", crf=23)
frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
i = 0
try:
    for i in range(120):
        w.write_frame(frame)
    w.close()
    print("RESULT: render survived; writers made:", made)
except Exception as exc:                                 # noqa: BLE001
    print("RESULT: render DIED at frame", i, "->", type(exc).__name__, str(exc)[:90].replace("\n", " "))
    print("writers made:", made, "| committed segments:", len(w.segments))
