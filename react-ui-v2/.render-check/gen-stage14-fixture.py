"""Regenerates stage14.fixture.json from the REAL backend code (run from the repo root):

    app/env/Scripts/python.exe react-ui/.render-check/gen-stage14-fixture.py

Nothing in the fixture is typed by hand: the status snapshots come out of `predictor.warmup_session` driving a
stub ORT session through `pipeline_status`, the queue snapshot out of a real `PipelineMonitor` +
`terminal_dashboard`, and `console` is the text the backend itself painted from them. The JS check
(render-stage14.mjs) then asserts the React screen shows the same words.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "app"))

from roop import pipeline_monitor, pipeline_status as ps, predictor, terminal_dashboard as td  # noqa: E402


class Session:
    def __init__(self, cache, build):
        self.cache, self.build = cache, build
        self.live = None

    def get_provider_options(self):
        return {"TensorrtExecutionProvider": {"trt_fp16_enable": True, "trt_max_workspace_size": 2 * 1024 ** 3,
                                              "trt_engine_cache_enable": True, "trt_engine_cache_path": self.cache}}

    def get_inputs(self):
        return [type("M", (), {"name": "x", "shape": [1, 3, 8, 8], "type": "tensor(float)"})()]

    def run(self, _out, _feed):
        self.live = ps.snapshot(events=0)
        if self.build:
            with open(os.path.join(self.cache, "m.engine"), "wb") as f:
                f.write(b"plan")
        return []


def queues(state, frames=24):
    pipeline_monitor.clear_levels()
    mon = pipeline_monitor.PipelineMonitor(12, lambda: state, emit=lambda _l: None)
    mon.start()
    for _ in range(frames):
        mon.frame_done()
    view = td.TerminalProgressManager._queues(True)
    return view, td.render_queue_line(view, color=False)


out = {}
with tempfile.TemporaryDirectory() as cache:
    cache_shown = cache.replace("\\", "/")
    ps.reset()
    predictor.reset()
    miss = Session(cache, build=True)
    predictor.warmup_session(miss, tag="gpen_256", once=False)
    out["compiling"] = miss.live
    out["after_miss"] = ps.snapshot(events=0)
    out["miss_receipt"] = ps.format_receipt(out["after_miss"]["last_compile"])
    predictor.reset()
    hit = Session(cache, build=False)
    predictor.warmup_session(hit, tag="gpen_256", once=False)
    out["after_hit"] = ps.snapshot(events=0)
    out["hit_receipt"] = ps.format_receipt(out["after_hit"]["last_compile"])
    out["cache_path"] = cache

ps.reset()
with ps.stage(ps.STAGE_COMPILING_TRT, model="cuda_model", precision="n/a"):
    pass
out["after_cuda"] = ps.snapshot(events=0)

gpu_bound, gpu_line = queues({"input": (8, 8), "output": (2, 8), "gpu": (4, 4)})
decode_bound, decode_line = queues({"input": (0, 8), "output": (0, 8), "gpu": (1, 4)})
encode_bound, encode_line = queues({"input": (8, 8), "output": (8, 8), "gpu": (4, 4)})
chunk_bound, chunk_line = queues({"input": (1, 2), "output": (0, 2), "unit": "chunks"})
out["queues"] = {"gpu_bound": gpu_bound, "decode_bound": decode_bound, "encode_bound": encode_bound, "chunks": chunk_bound}
out["console"] = {"gpu_bound": gpu_line, "decode_bound": decode_line, "encode_bound": encode_line, "chunks": chunk_line}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage14.fixture.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print("wrote", path)
