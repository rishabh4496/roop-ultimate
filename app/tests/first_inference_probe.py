"""Run tests/two_face_video.py (same argv) and record, for EVERY ONNX Runtime session the render builds, the real
provider, whether TensorRT FP16 is on, and the device memory just BEFORE and just AFTER that session's first inference.

    ROOP_BENCH_LAUNCHER=tests/first_inference_probe.py ROOP_FIRST_INFERENCE_LOG=out.jsonl  (see tools/quality_harness.py)

WHY a wrapper and not the `[VRAM]` stage lines. Those are sampled at model INITIALISE and at coarse run phases, and
TensorRT allocates its execution-context memory on the first inference (docs / memory: "TRT allocates on first
inference"), so a load-time number under-reports a model by exactly the part that grows. This wraps
`InferenceSession.run` / `run_with_iobinding`, which every model in the pipeline goes through, so the detector, the
recogniser, the swapper, the restorer and the mask nets are all covered with no per-model code.

First calls are serialised through one lock so two models' windows cannot overlap each other; steady-state calls take
a lock-free fast path. Device memory is `torch.cuda.mem_get_info` (device-wide, the same quantity nvidia-smi reports),
so another process on the GPU shows up in the numbers - the harness checks the card is quiet before it starts.
"""
import json
import os
import runpy
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = os.environ.get("ROOP_FIRST_INFERENCE_LOG")
_lock = threading.Lock()
_seen = set()


def _used_mib():
    import torch
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1048576.0


def _record(rec):
    if not LOG:
        return
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _wrap(cls, name):
    orig = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        key = id(self)
        if key in _seen:
            return orig(self, *args, **kwargs)
        with _lock:
            if key in _seen:
                return orig(self, *args, **kwargs)
            _seen.add(key)
            try:
                before = _used_mib()
            except Exception:                                   # no CUDA device: record that, do not guess
                before = None
            t0 = time.perf_counter()
            result = orig(self, *args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
            try:
                after = _used_mib()
            except Exception:
                after = None
            try:
                from roop import baseline_probe
                fp16 = baseline_probe.trt_fp16_state(self)
            except Exception:
                fp16 = "unknown"
            try:
                providers = list(self.get_providers())
            except Exception:
                providers = []
            try:     # sessions built from in-memory bytes have no path: identify them by their inputs
                sig = ",".join(sorted("%s:%s" % (i.name, "x".join(str(d) if isinstance(d, int) else "N" for d in i.shape))
                                      for i in self.get_inputs()))
            except Exception:
                sig = ""
            _record({"file": os.path.basename(str(getattr(self, "_model_path", "") or "")) or "?",
                     "provider": providers[0] if providers else "none", "providers": providers,
                     "trt_fp16": fp16, "input_signature": sig, "method": name, "thread": threading.current_thread().name,
                     "vram_before_mib": None if before is None else round(before, 1),
                     "vram_after_mib": None if after is None else round(after, 1),
                     "vram_delta_mib": None if before is None or after is None else round(after - before, 1),
                     "first_call_ms": round(ms, 1)})
            return result

    wrapped.__name__ = name
    setattr(cls, name, wrapped)


def main():
    from onnxruntime.capi import onnxruntime_inference_collection as C
    for name in ("run", "run_with_iobinding"):
        _wrap(C.InferenceSession, name)
    sys.argv = [os.path.join(HERE, "two_face_video.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(HERE, "two_face_video.py"), run_name="__main__")


if __name__ == "__main__":
    main()
