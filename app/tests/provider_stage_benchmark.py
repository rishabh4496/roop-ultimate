"""Per-stage latency and peak VRAM for the face-swap pipeline, by execution provider.

Run it directly (it is not a pytest module, so a bulk pytest run never starts it):

    app/env/Scripts/python.exe app/tests/provider_stage_benchmark.py --md out.md

What it measures, and how
-------------------------
Stages (the brief's list):

  detector           ONNX session on the shipped detector (retinaface_10g.onnx), per frame
  stabilizer         roop.one_euro.KpsStabilizer.apply, per face (CPU, provider-independent)
  face_parser        ONNX session on the shipped BiSeNet (resnet18.onnx), per face
  swapper            ONNX session on the shipped swap net (inswapper_128.onnx), per face
  occlusion_masker   roop.occlusion_masker.OcclusionMasker.compute, per face (CPU)
  laplacian_blender  roop.blender.laplacian_blend, per face (CPU)

Model stages time one ORT ``session.run`` on the shipped weights with random
inputs of the declared shape. Pre- and post-processing of the production
path are not included, so these numbers are the cost of the model call, not
the cost of the whole stage. The three Python stages run the production
classes directly.

Providers: cpu, cuda, tensorrt. A provider row is only reported as measured if
the session really runs on that provider: ORT is asked for the provider chain
and ``session.get_providers()`` is checked afterwards. A silent fall-back to CPU
is reported as "fell back" and its times are not presented as TensorRT times.
The TensorRT DLLs are put on PATH with the app's own ``prepare_environment``.
The memory notes record that a bare process without this step times a 4 ms
model at 210 ms, so it is required.

Peak VRAM is the change in GPU memory used on the whole device, sampled every
50 ms with NVML during that provider's run, minus the reading taken before the
sessions were built. The GPU is shared with the desktop, so this is an upper
bound on what the pipeline added, not a clean per-process figure.

Output is a markdown table on stdout (and to ``--md``), plus JSON with ``--json``.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np  # noqa: E402

STAGES = ["detector", "stabilizer", "face_parser", "swapper",
          "occlusion_masker", "laplacian_blender"]
MODEL_FILES = {
    "detector": "retinaface_10g.onnx",
    "face_parser": "resnet18.onnx",
    "swapper": "inswapper_128.onnx",
}
PROVIDER_ORDER = ["cpu", "cuda", "tensorrt"]


def _provider_chain(name, trt_cache):
    if name == "cpu":
        return ["CPUExecutionProvider"]
    if name == "cuda":
        return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    if name == "tensorrt":
        return [("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_fp16_enable": False,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(trt_cache)}),
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider"]
    raise ValueError(name)


_EXPECTED_HEAD = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
                  "tensorrt": "TensorrtExecutionProvider"}


class VramSampler:
    """Background NVML sampler of device memory in use (MiB)."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0.0
        self.available = False
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception:
            self._nvml = None

    def used_mib(self):
        if not self.available:
            return None
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 * 1024)

    def start(self):
        if not self.available:
            return
        self.peak = self.used_mib()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used_mib())
            time.sleep(self.interval)

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None


def _dummy_for(inp, default_hw):
    shape = []
    for i, d in enumerate(inp.shape):
        if isinstance(d, int) and d > 0:
            shape.append(d)
        elif i == 0:
            shape.append(1)
        else:
            shape.append(default_hw)
    dtype = np.float16 if "float16" in inp.type else np.float32
    if "int64" in inp.type:
        dtype = np.int64
    return np.random.default_rng(0).random(shape, dtype=np.float32).astype(dtype)


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {"median_ms": statistics.median(samples),
            "p95_ms": samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))],
            "n": len(samples)}


def _model_session(path, providers):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=providers)
    return sess


def _measure_provider(name, args, models_dir, cpu_stage_cache):
    """Return {stage: timing dict or reason string} and the VRAM peak for one provider."""
    import roop.one_euro as one_euro
    from roop.occlusion_masker import OcclusionMasker
    from roop.blender import laplacian_blend

    result, notes = {}, {}
    sampler = VramSampler()
    sampler.start()
    base = sampler.used_mib()
    sessions = {}
    try:
        for stage, fname in MODEL_FILES.items():
            path = models_dir / fname
            if not path.exists():
                result[stage] = f"missing model {fname}"
                continue
            try:
                sess = _model_session(path, _provider_chain(name, args.trt_cache))
            except Exception as exc:  # build failure is a result, not a crash
                result[stage] = f"build failed: {type(exc).__name__}: {str(exc)[:120]}"
                continue
            head = sess.get_providers()[0]
            if head != _EXPECTED_HEAD[name]:
                result[stage] = f"fell back to {head} (not measured as {name})"
                continue
            sessions[stage] = sess
        # Stage timings (model stages on the session, Python stages on the classes).
        if "detector" in sessions:
            sess = sessions["detector"]
            inp = sess.get_inputs()[0]
            x = _dummy_for(inp, 640)
            result["detector"] = _time(lambda: sess.run(None, {inp.name: x}), args.warmup, args.iters)
        if "face_parser" in sessions:
            sess = sessions["face_parser"]
            inp = sess.get_inputs()[0]
            x = _dummy_for(inp, 512)
            result["face_parser"] = _time(lambda: sess.run(None, {inp.name: x}), args.warmup, args.iters)
        if "swapper" in sessions:
            sess = sessions["swapper"]
            feeds = {i.name: _dummy_for(i, 128) for i in sess.get_inputs()}
            result["swapper"] = _time(lambda: sess.run(None, feeds), args.warmup, args.iters)
        # Provider-independent CPU stages, timed once and reported under every provider.
        if "stabilizer" not in cpu_stage_cache:
            stab = one_euro.KpsStabilizer()
            kps = np.array([[100, 110], [140, 110], [120, 135], [105, 155], [135, 155]], np.float64)
            counter = {"t": 0}

            def step():
                counter["t"] += 1
                stab.apply(kps + np.random.default_rng(counter["t"]).normal(0, 0.5, kps.shape),
                           float(counter["t"]))
            cpu_stage_cache["stabilizer"] = _time(step, args.warmup, args.iters)
            masker = OcclusionMasker()
            clean = np.full((96, 96, 3), (120, 150, 190), np.uint8)
            target = clean.copy()
            target[45:85, 40:55] = 25
            labels = np.ones((96, 96), np.uint8)
            sil = np.ones((96, 96), np.float32)
            frame_idx = {"i": 0}

            def mstep():
                frame_idx["i"] += 1
                masker.compute(target, clean, sil, labels=labels, track_id=1,
                               frame_index=frame_idx["i"])
            cpu_stage_cache["occlusion_masker"] = _time(mstep, args.warmup, args.iters)
            paste = np.full((256, 256, 3), (60, 200, 90), np.uint8)
            plate = np.full((256, 256, 3), (100, 150, 200), np.uint8)
            alpha = np.zeros((256, 256), np.float32)
            alpha[32:224, 32:224] = 1.0
            cpu_stage_cache["laplacian_blender"] = _time(
                lambda: laplacian_blend(paste, plate, alpha), args.warmup, args.iters)
        result["stabilizer"] = cpu_stage_cache["stabilizer"]
        result["occlusion_masker"] = cpu_stage_cache["occlusion_masker"]
        result["laplacian_blender"] = cpu_stage_cache["laplacian_blender"]
        # Peak VRAM while the sessions are alive and every stage has run.
        sampler.stop()
        peak = sampler.peak
    finally:
        sampler.stop()
        sessions.clear()
    vram = None
    if sampler.available and base is not None and peak is not None:
        vram = max(0.0, peak - base)
    return result, vram, sampler.available


def _cell(value):
    if isinstance(value, dict):
        return f"{value['median_ms']:.2f} (p95 {value['p95_ms']:.2f})"
    return str(value)


def _markdown(results, vram, gpu_name):
    lines = ["| Stage | " + " | ".join(p.upper() for p in PROVIDER_ORDER) + " |",
             "|---|" + "---|" * len(PROVIDER_ORDER)]
    for stage in STAGES:
        row = [_cell(results.get(p, {}).get(stage, "not run")) for p in PROVIDER_ORDER]
        unit = "per frame" if stage == "detector" else "per face"
        lines.append(f"| {stage} ({unit}, ms) | " + " | ".join(row) + " |")
    total = []
    for p in PROVIDER_ORDER:
        vals = [results.get(p, {}).get(s) for s in STAGES]
        if all(isinstance(v, dict) for v in vals):
            total.append(f"{sum(v['median_ms'] for v in vals if True):.2f}")
        else:
            total.append("n/a")
    lines.append("| sum of stage medians (ms) | " + " | ".join(total) + " |")
    vram_cells = []
    for p in PROVIDER_ORDER:
        v = vram.get(p)
        vram_cells.append("n/a" if v is None else f"{v:.0f} MiB")
    lines.append("| peak VRAM added (MiB) | " + " | ".join(vram_cells) + " |")
    lines.append("")
    lines.append(f"GPU: {gpu_name}. Median and p95 over the timed iterations after warm-up. "
                 "'fell back' means the session did not run on the requested provider.")
    return "\n".join(lines)


def _gpu_name():
    try:
        import pynvml
        pynvml.nvmlInit()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return name.decode() if isinstance(name, bytes) else name
    except Exception:
        return "no NVML GPU"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--providers", default="cpu,cuda,tensorrt")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trt-cache", default=str(APP / "models" / "trt_cache"))
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    # The app's own environment setup: puts the TensorRT DLLs on PATH. Without it
    # the TensorRT provider silently falls back to CPU (see module docstring).
    os.chdir(APP)
    import roop.globals as g
    from settings import Settings
    g.CFG = Settings("config.yaml")  # prepare_environment reads the settings object
    from ui.main import prepare_environment
    prepare_environment()

    import onnxruntime as ort
    available = set(ort.get_available_providers())
    models_dir = APP / "models"
    Path(args.trt_cache).mkdir(parents=True, exist_ok=True)

    results, vram, sampled = {}, {}, {}
    cpu_stage_cache = {}
    for name in [p.strip() for p in args.providers.split(",") if p.strip()]:
        if _EXPECTED_HEAD[name] not in available:
            results[name] = {s: f"{_EXPECTED_HEAD[name]} not installed" for s in STAGES}
            vram[name] = None
            continue
        t0 = time.perf_counter()
        res, v, ok = _measure_provider(name, args, models_dir, cpu_stage_cache)
        results[name] = res
        vram[name] = v
        sampled[name] = ok
        print(f"[bench] {name} done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    gpu = _gpu_name()
    md = _markdown(results, vram, gpu)
    print(md)
    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
    if args.json:
        payload = {"gpu": gpu, "results": results, "vram_mib": vram, "warmup": args.warmup,
                   "iters": args.iters}
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
