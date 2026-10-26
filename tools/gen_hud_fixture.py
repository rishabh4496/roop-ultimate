"""Regenerate react-ui/.render-check/pipeline-hud.fixture.json from the REAL telemetry writers.

Drives `stage_telemetry` (and `procmgr_runtime._prof`, the writer behind the Stage 3 ops line) through a small
fake run and records both the LIVE frame (`snapshot()` while Stage 3 runs) and the finished report
(`build_summary()`), exactly what /ws/telemetry carries. The render check feeds these to the React HUD, so the
UI is checked against the backend's real field names rather than a hand-typed shape.

    python tools/gen_hud_fixture.py            # writes the fixture
"""
import json
import os
import sys
import time
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop import procmgr_runtime, stage_telemetry as st  # noqa: E402
import roop.globals as g  # noqa: E402

S1, S2, S3 = st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS, st.STAGE_3_MAIN_SWAP
clock = [1000.0]
mono = [50.0]                                       # `retire`'s window clock (time.monotonic)


def _spin(seconds):
    """Busy-wait: time.sleep rounds to ~15 ms on Windows, which would make every op read the same."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


FIXTURE = os.path.join(ROOT, "react-ui", ".render-check", "pipeline-hud.fixture.json")


def build():
    """Drive the real writers and return ``{"live": snapshot, "summary": report}`` (no file written)."""
    with mock.patch.object(st.time, "time", lambda: clock[0]), \
            mock.patch.object(st.time, "monotonic", lambda: mono[0]), \
            mock.patch.object(st, "_smart_skip_enabled", return_value=True), \
            mock.patch.object(g, "execution_providers", [("TensorrtExecutionProvider", {}), "CPUExecutionProvider"],
                              create=True), \
            mock.patch.object(g, "video_encoder", "h264_nvenc", create=True), \
            mock.patch.object(st._vram, "start", lambda: st._vram.reset()), \
            mock.patch.object(st, "_video_io_snapshot", return_value=None), \
            mock.patch.object(st, "_aux_offload", return_value=None):
        # video I/O and the OpenVINO block depend on which optional modules the process happened to load; the
        # fixture pins them off so its shape does not depend on what ran before it.
        st._vram.stop()
        st._vram._probe = lambda: None
        st.reset()
        st.begin_run(600)
        st.enter(S1)
        st.put(S1, sampled_frames=40, samples_total=40, scene_cuts=3, scan_mode="hardware-keyframe")
        st.progress(S1, 40, 40)
        clock[0] += 4.0
        st.leave(S1)
        st.enter(S2)
        st.put(S2, detector_calls=25, coast_frames=75, scan_frames=100, detect_interval=4)
        st.progress(S2, 100, 100)
        clock[0] += 5.0
        st.leave(S2)
        st.enter(S3, 600)
        for stage, ms, n in (("decode", 2, 5), ("alignment", 3, 5), ("swap", 11, 5), ("blend", 4, 5), ("encode", 6, 5)):
            for _ in range(n):
                with procmgr_runtime._prof(stage):
                    _spin(ms / 1000.0)
        st.put(S3, swapped_frames=120, faces_swapped=130)
        st.add(S3, "neural_inferences", 90)             # swap-model runs; 40 more faces reused a warped patch
        st.add(S3, "smart_skip_faces", 40)
        st.progress(S3, 300, 600)
        # The output writer accepts 30 frames a second for 10 s: the retired rate, live and final, is 30 fps.
        for _ in range(10):
            clock[0] += 1.0
            mono[0] += 1.0
            st.retire(30)
        live = st.snapshot()
        st.leave(S3)
        summary = st.end_run(ok=True)
    return json.loads(json.dumps({"live": live, "summary": summary}, default=str))


def main():
    data = build()
    with open(FIXTURE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", FIXTURE)


if __name__ == "__main__":
    main()
