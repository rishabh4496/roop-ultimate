"""Recognition pipeline: correctness checks and a benchmark harness.

Two ways to use this file.

  pytest tests/test_recognition_pipeline.py            correctness (needs the model files)
  python tests/test_recognition_pipeline.py --benchmark [--device auto] [--out table.md]

THE CORRECTNESS TESTS run only on models already on disk (default: app/models; override with
ROOP_RECOGNITION_MODELS_DIR). A suite run must not silently pull ~520 MB, so a missing model is a
SKIP that names the file; ROOP_RECOGNITION_DOWNLOAD=1 fetches them through the registry.
Embeddings are checked on the CPU provider for exact, deterministic properties, and against
the GPU providers for equivalence. Real aligned faces (app/facesets, detected with the
buffalo_l detector) check that each model separates same-person from different-person -- random
noise cannot show that.

THE BENCHMARK measures what a face costs end to end (preprocess + inference + normalise) and
what a batch costs, with three traps designed out, each one a number this project has already
been misled by:

  * VRAM is read DEVICE-WIDE (torch.cuda.mem_get_info). torch.cuda.memory_allocated() sees only
    torch's own allocator, and ONNX Runtime does not use it: it would print 0 MB for every model.
  * The GPU clock ramps. A 30-iteration warm-up of ~2 ms calls lasts 60 ms; the card is still at
    ~1/3 clock afterwards. Warm-up runs for max(N iterations, ramp seconds) and the SM clock
    before/after is printed.
  * A failed model is a FAILED ROW and a non-zero exit, never a missing row.

This is a per-call micro-benchmark. It is not an end-to-end render acceptance number (those need
600 frames and counterbalanced arms, see AGENTS.md), and it shares the GPU with anything else
running -- stop the app first; the header prints the GPU utilisation it started under.
"""

import argparse
import glob
import gc
import os
import subprocess
import sys
import time
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.recognition_engine import RecognitionInferenceEngine           # noqa: E402
from roop.recognition_registry import RECOGNITION_REGISTRY, get_model_spec, get_registered_models  # noqa: E402

MODELS_DIR = os.path.abspath(os.environ.get("ROOP_RECOGNITION_MODELS_DIR") or os.path.join(APP, "models"))
ALLOW_DOWNLOAD = os.environ.get("ROOP_RECOGNITION_DOWNLOAD") == "1"
FACESET_DIR = os.path.join(APP, "facesets")
DETECTOR = os.path.join(APP, "models", "buffalo_l", "det_10g.onnx")

# Measured 2026-10-02 over 36 people x 3 perturbations (blur, JPEG q30, darkening): the narrowest
# same-vs-different gap among the registry models was 0.195 (SFace), the widest 0.433. Half the
# worst gap is the margin asserted.
SEPARATION_MARGIN = 0.10


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, get_model_spec(name).filename)


def model_available(name: str) -> bool:
    return ALLOW_DOWNLOAD or os.path.isfile(model_path(name))


def cuda_available() -> bool:
    return "CUDAExecutionProvider" in ort.get_available_providers()


def noise_face(seed: int = 42) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, (112, 112, 3), dtype=np.uint8)


# ============================================================================ correctness

class TestStructure(unittest.TestCase):
    """Every registered model: shape, unit norm, finiteness, determinism (CPU provider)."""

    def test_at_least_one_model_is_installed(self):
        """Guards the guards: if every model skipped, every test below would pass vacuously."""
        self.assertTrue(any(model_available(n) for n in get_registered_models()),
                        "no recognition model on disk in %s" % MODELS_DIR)

    def test_every_registered_model_honours_its_spec(self):
        face = noise_face()
        for name, spec in RECOGNITION_REGISTRY.items():
            with self.subTest(model=name):
                if not model_available(name):
                    self.skipTest("%s not on disk (set ROOP_RECOGNITION_DOWNLOAD=1 to fetch)" % spec.filename)
                engine = RecognitionInferenceEngine(name, MODELS_DIR, "cpu")
                vec, quality = engine.compute_embedding(face)
                self.assertEqual(vec.shape, (spec.output_dim,))
                self.assertEqual(vec.dtype, np.float32)
                self.assertTrue(np.isfinite(vec).all())
                self.assertLess(abs(float(np.linalg.norm(vec)) - 1.0), 1e-5)
                self.assertGreater(quality, 0.0)
                again, _ = engine.compute_embedding(face.copy())
                np.testing.assert_array_equal(vec, again)                 # CPU is deterministic
                self.assertEqual(engine.invalid_outputs, 0)

    def test_distinct_noise_inputs_do_not_collapse_to_one_vector(self):
        """A model returning a constant would pass every shape/norm check above."""
        engine = RecognitionInferenceEngine("default", MODELS_DIR, "cpu") if model_available("default") \
            else self.skipTest("default model not on disk")
        a, _ = engine.compute_embedding(noise_face(1))
        b, _ = engine.compute_embedding(noise_face(2))
        self.assertLess(float(a @ b), 0.99)


class TestGpuEquivalence(unittest.TestCase):
    """The provider must change speed, not the answer."""

    def _cross(self, device: str, floor: float):
        if device == "cuda" and not cuda_available():
            self.skipTest("no CUDA provider in this onnxruntime")
        face = noise_face()
        for name in get_registered_models():
            with self.subTest(model=name, device=device):
                if not model_available(name):
                    self.skipTest("%s not on disk" % name)
                cpu, _ = RecognitionInferenceEngine(name, MODELS_DIR, "cpu").compute_embedding(face)
                engine = RecognitionInferenceEngine(name, MODELS_DIR, device)
                if engine.degraded:
                    self.fail("%s asked for %s but ran on %s: %s" % (
                        name, device, engine.active_providers, engine.fallback_log))
                gpu, _ = engine.compute_embedding(face)
                self.assertGreater(float(cpu @ gpu), floor, engine.active_providers[:1])

    def test_cuda_matches_cpu(self):
        self._cross("cuda", 0.9999)

    def test_tensorrt_fp16_matches_cpu(self):
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            self.skipTest("no TensorRT provider")
        self._cross("tensorrt", 0.999)          # measured 0.99996+ on an RTX 4070


def _load_crops(limit: int = 8) -> List[Tuple[str, np.ndarray]]:
    """Aligned 112x112 crops of real faces: buffalo_l SCRFD detection + the standard 5-point warp."""
    from insightface.model_zoo import get_model
    from insightface.utils import face_align
    detector = get_model(DETECTOR, providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(640, 640))
    crops = []
    for path in sorted(glob.glob(os.path.join(FACESET_DIR, "*.png"))):
        image = cv2.imread(path)
        if image is None:
            continue
        _boxes, kps = detector.detect(image, input_size=(640, 640), max_num=1)
        if kps is not None and len(kps):
            crops.append((os.path.basename(path), face_align.norm_crop(image, kps[0], 112)))
        if len(crops) == limit:
            break
    return crops


def _perturbations(crop: np.ndarray) -> List[np.ndarray]:
    """Same person, degraded the way video degrades them."""
    jpeg = cv2.imdecode(cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 30])[1], 1)
    return [cv2.GaussianBlur(crop, (0, 0), 1.5), jpeg, (crop.astype(np.float32) * 0.75).astype(np.uint8)]


class TestIdentitySeparation(unittest.TestCase):
    """Real faces: the same person stays together, different people stay apart."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(DETECTOR):
            raise unittest.SkipTest("buffalo_l detector missing: %s" % DETECTOR)
        cls.crops = _load_crops()
        if len(cls.crops) < 4:
            raise unittest.SkipTest("need >= 4 detectable faces in %s" % FACESET_DIR)

    def test_every_model_separates_same_from_different(self):
        for name in get_registered_models():
            with self.subTest(model=name):
                if not model_available(name):
                    self.skipTest("%s not on disk" % name)
                engine = RecognitionInferenceEngine(name, MODELS_DIR, "cpu")
                base = [engine.compute_embedding(c)[0] for _, c in self.crops]
                same = [float(base[i] @ engine.compute_embedding(p)[0])
                        for i, (_, c) in enumerate(self.crops) for p in _perturbations(c)]
                diff = [float(base[i] @ base[j]) for i in range(len(base)) for j in range(i + 1, len(base))]
                self.assertAlmostEqual(float(base[0] @ engine.compute_embedding(self.crops[0][1])[0]), 1.0, places=5)
                gap = min(same) - max(diff)
                self.assertGreater(gap, SEPARATION_MARGIN,
                                   "%s: same-person min %.3f vs different-person max %.3f" % (name, min(same), max(diff)))


class TestFallback(unittest.TestCase):
    """A requested provider that cannot run must end on a CPU session that still gives the right
    answer, and must SAY it degraded."""

    @classmethod
    def setUpClass(cls):
        # Cheapest installed model, so a plain checkout (which has only `default`) still runs these.
        cls.model = next((n for n in ("mobilefacenet", "facerecognizersf", "default") if model_available(n)), None)
        if cls.model is None:
            raise unittest.SkipTest("no recognition model on disk")
        cls.face = noise_face()
        cls.reference, _ = RecognitionInferenceEngine(cls.model, MODELS_DIR, "cpu").compute_embedding(cls.face)

    def _assert_recovered(self, engine):
        self.assertEqual(engine.active_providers, ["CPUExecutionProvider"])
        self.assertTrue(engine.degraded, "a CPU landing after a GPU request must be flagged")
        self.assertTrue(engine.fallback_log, "...and explained")
        vec, _ = engine.compute_embedding(self.face)
        self.assertGreater(float(vec @ self.reference), 0.99999)

    def test_unknown_provider_name_is_rejected_not_guessed(self):
        """A typo must not silently run on some other device."""
        with self.assertRaisesRegex(ValueError, "Unknown device"):
            RecognitionInferenceEngine(self.model, MODELS_DIR, "non_existent_provider_xyz")

    def test_provider_this_build_lacks_recovers_on_cpu(self):
        have = set(ort.get_available_providers())
        for device, ep in (("coreml", "CoreMLExecutionProvider"), ("directml", "DmlExecutionProvider")):
            with self.subTest(device=device):
                if ep in have:
                    self.skipTest("%s exists in this build" % ep)
                engine = RecognitionInferenceEngine(self.model, MODELS_DIR, device)
                self._assert_recovered(engine)

    def test_nonexistent_gpu_id_recovers_on_cpu(self):
        if not cuda_available():
            self.skipTest("no CUDA provider")
        self._assert_recovered(RecognitionInferenceEngine(self.model, MODELS_DIR, "cuda", gpu_id=99))

    def test_provider_dropped_by_onnxruntime_itself_is_caught_and_recovered(self):
        """The REAL silent failure: CUDA is accepted, ORT drops it to CPU, nothing raises."""
        if not cuda_available():
            self.skipTest("no CUDA provider")
        bad = {"device_id": 7, "arena_extend_strategy": "kNextPowerOfTwo"}      # no such device
        with mock.patch.object(RecognitionInferenceEngine, "_cuda_options", return_value=bad):
            engine = RecognitionInferenceEngine(self.model, MODELS_DIR, "cuda")
        self._assert_recovered(engine)

    def test_strict_mode_refuses_to_degrade(self):
        if not cuda_available():
            self.skipTest("no CUDA provider")
        with self.assertRaisesRegex(RuntimeError, "CPU-only"):
            RecognitionInferenceEngine(self.model, MODELS_DIR, "cuda", gpu_id=99, strict=True)


# ============================================================================ benchmark

def summarise(latencies_ms: List[float]) -> Dict[str, float]:
    """Mean / P95 / P99 latency (ms) and throughput (calls per second at the mean)."""
    a = np.asarray(latencies_ms, dtype=np.float64)
    mean = float(a.mean())
    return {"mean_ms": mean, "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "fps": 1000.0 / mean if mean > 0 else 0.0}


def vram_used_mb() -> Optional[float]:
    """Device-wide VRAM in use (MiB), or None without a CUDA device. Includes other processes."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return (total - free) / 1024 ** 2
    except Exception as exc:                     # a harness reports what it could not measure
        print("[bench] VRAM unreadable: %r" % (exc,), file=sys.stderr)
    return None


def gpu_clocks() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,utilization.gpu,driver_version",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()[0]
        name, sm, sm_max, util, driver = [x.strip() for x in out.split(",")]
        return "%s | SM %s/%s MHz | util %s%% | driver %s" % (name, sm, sm_max, util, driver)
    except Exception:
        return "n/a"


def _settled_vram(read: Callable[[], Optional[float]] = vram_used_mb, samples: int = 3, gap: float = 0.2) -> Optional[float]:
    values = []
    for _ in range(samples):
        value = read()
        if value is None:
            return None
        values.append(value)
        time.sleep(gap)
    return float(np.median(values))


def benchmark_model(name: str, device: str, warmup: int = 30, iters: int = 100, ramp_seconds: float = 2.0,
                    face: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Time one model's compute_embedding on `device`; VRAM is the device-wide delta of holding it."""
    face = noise_face() if face is None else face
    gc.collect()
    before = _settled_vram()
    engine = RecognitionInferenceEngine(name, MODELS_DIR, device)
    start = time.perf_counter()
    done = 0
    while done < warmup or time.perf_counter() - start < ramp_seconds:       # iterations AND clock ramp
        engine.compute_embedding(face)
        done += 1
    clocks_before = gpu_clocks()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        engine.compute_embedding(face)
        samples.append((time.perf_counter() - t0) * 1000.0)
    after = _settled_vram()
    row = summarise(samples)
    row.update(model=name, provider=engine.active_providers[0].replace("ExecutionProvider", ""),
               degraded=engine.degraded, warmup_calls=done, clocks=clocks_before,
               vram_mb=None if before is None or after is None else after - before, failed=None)
    del engine
    gc.collect()
    return row


def benchmark_batches(name: str, device: str, sizes: List[int], iters: int = 50, warmup: int = 10,
                      ramp_seconds: float = 1.0) -> List[Dict[str, Any]]:
    """Batch sweep through the raw session (compute_embedding is single-face). Skips static-batch graphs.

    Each batch size gets the same warm-up rule as the per-face table (calls AND clock ramp): a sweep
    that warmed 10 calls read 4.08 ms for a model the ramped single-face run measured at 3.1 ms."""
    engine = RecognitionInferenceEngine(name, MODELS_DIR, device)
    batch_axis = engine.session.get_inputs()[0].shape[0]
    one = engine.preprocess(noise_face())
    rows = []
    for n in sizes:
        row: Dict[str, Any] = {"model": name, "batch": n, "provider": engine.active_providers[0].replace("ExecutionProvider", "")}
        if isinstance(batch_axis, int) and batch_axis != n:
            row["note"] = "static batch %d" % batch_axis
            rows.append(row)
            continue
        blob = np.ascontiguousarray(np.repeat(one, n, axis=0))
        feed = {engine.input_name: blob}
        start, done = time.perf_counter(), 0
        while done < warmup or time.perf_counter() - start < ramp_seconds:
            engine.session.run(None, feed)
            done += 1
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = engine.session.run(None, feed)[0]
            samples.append((time.perf_counter() - t0) * 1000.0)
        assert out.shape[0] == n, "batch %d returned %d rows" % (n, out.shape[0])
        stats = summarise(samples)
        row.update(batch_ms=stats["mean_ms"], p95_ms=stats["p95_ms"], per_face_ms=stats["mean_ms"] / n,
                   faces_per_s=n * 1000.0 / stats["mean_ms"])
        rows.append(row)
    return rows


def _fmt(value: Any, spec: str, unit: str = "") -> str:
    return "n/a" if value is None else format(value, spec) + unit


def render_markdown(header: List[str], rows: List[Dict[str, Any]], batch_rows: List[Dict[str, Any]]) -> str:
    lines = list(header) + [""]
    lines += ["| Model | Provider | Mean (ms) | P95 (ms) | P99 (ms) | Throughput (FPS) | VRAM held (MiB) |",
              "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r.get("failed"):
            lines.append("| `%s` | **FAILED** | %s | | | | |" % (r["model"], r["failed"].replace("|", "/")))
            continue
        flag = " (**degraded**)" if r["degraded"] else ""
        lines.append("| `%s` | %s%s | %.2f | %.2f | %.2f | %.0f | %s |" % (
            r["model"], r["provider"], flag, r["mean_ms"], r["p95_ms"], r["p99_ms"], r["fps"], _fmt(r["vram_mb"], ".0f")))
    if batch_rows:
        lines += ["", "| Model | Provider | Batch | Batch (ms) | Per face (ms) | Faces/s |", "| :--- | :--- | ---: | ---: | ---: | ---: |"]
        for r in batch_rows:
            if "note" in r:
                lines.append("| `%s` | %s | %d | n/a | n/a | n/a (%s) |" % (r["model"], r["provider"], r["batch"], r["note"]))
            else:
                lines.append("| `%s` | %s | %d | %.2f | %.2f | %.0f |" % (
                    r["model"], r["provider"], r["batch"], r["batch_ms"], r["per_face_ms"], r["faces_per_s"]))
    return "\n".join(lines) + "\n"


def run_performance_benchmarks(device: str = "auto", warmup: int = 30, iters: int = 100,
                               batch_sizes: Optional[List[int]] = None, models: Optional[List[str]] = None,
                               ramp_seconds: float = 2.0) -> Tuple[str, bool]:
    """Benchmark every installed model; returns (markdown, all_ok)."""
    names = [n for n in (models or get_registered_models()) if n != "antelopev2"]      # same file as glintr100
    rows: List[Dict[str, Any]] = []
    # Charge the CUDA context to nobody: build one session first, drop it, then measure.
    if names and device != "cpu" and vram_used_mb() is not None and model_available(names[0]):
        RecognitionInferenceEngine(names[0], MODELS_DIR, device).compute_embedding(noise_face())
        gc.collect()
    for name in names:
        if not model_available(name):
            rows.append({"model": name, "failed": "not on disk (ROOP_RECOGNITION_DOWNLOAD=1 fetches it)"})
            continue
        try:
            rows.append(benchmark_model(name, device, warmup, iters, ramp_seconds))
        except Exception as exc:
            print("[bench] %s FAILED: %r" % (name, exc), file=sys.stderr)
            rows.append({"model": name, "failed": "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:120])})
    batch_rows: List[Dict[str, Any]] = []
    sweep_device = "cpu" if device == "cpu" else "cuda"        # a TensorRT session is built for batch 1
    if not cuda_available() and sweep_device == "cuda":
        sweep_device = "cpu"
    for name in names if batch_sizes else []:
        if model_available(name):
            try:
                batch_rows += benchmark_batches(name, sweep_device, batch_sizes)
            except Exception as exc:
                print("[bench] batch sweep for %s FAILED: %r" % (name, exc), file=sys.stderr)
                rows.append({"model": name + " (batch sweep)", "failed": "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:120])})
    header = ["# Recognition benchmark", "",
              "- device requested: `%s` (batch sweep on `%s`)" % (device, sweep_device),
              "- onnxruntime %s | providers %s" % (ort.__version__, ", ".join(ort.get_available_providers())),
              "- GPU at start: %s" % gpu_clocks(),
              "- %d warm-up calls (>= %.1f s clock ramp), %d timed calls per model; latency = preprocess + inference + normalise" % (
                  warmup, ramp_seconds, iters),
              "- VRAM is the device-wide delta of holding the loaded model (other processes included); batch sweep excludes preprocessing"]
    return render_markdown(header, rows, batch_rows), not any(r.get("failed") or r.get("degraded") for r in rows)


class TestBenchmarkHarness(unittest.TestCase):
    """The harness's own arithmetic and failure handling, with no model or GPU."""

    def test_summary_statistics(self):
        s = summarise(list(range(1, 101)))                       # 1..100 ms
        self.assertAlmostEqual(s["mean_ms"], 50.5)
        self.assertAlmostEqual(s["p95_ms"], 95.05)
        self.assertAlmostEqual(s["p99_ms"], 99.01)
        self.assertAlmostEqual(s["fps"], 1000.0 / 50.5)

    def test_vram_is_median_of_settled_readings(self):
        readings = iter([1000.0, 5000.0, 1010.0])
        self.assertEqual(_settled_vram(lambda: next(readings), 3, 0.0), 1010.0)
        self.assertIsNone(_settled_vram(lambda: None, 3, 0.0))

    def test_failed_and_degraded_models_are_rows_and_fail_the_run(self):
        ok = {"model": "m", "provider": "CUDA", "degraded": False, "mean_ms": 2.0, "p95_ms": 2.5, "p99_ms": 3.0,
              "fps": 500.0, "vram_mb": 300.0, "failed": None}
        text = render_markdown(["# h"], [ok, {"model": "broken", "failed": "RuntimeError: boom | pipe"},
                                         dict(ok, model="slow", provider="CPU", degraded=True, vram_mb=None)], [])
        self.assertIn("`broken` | **FAILED** | RuntimeError: boom / pipe", text)
        self.assertIn("CPU (**degraded**)", text)
        self.assertIn("| 300 |", text)
        self.assertIn("| n/a |", text)

    def test_run_reports_a_missing_model_instead_of_dropping_it(self):
        with mock.patch.object(sys.modules[__name__], "model_available", return_value=False):
            text, ok = run_performance_benchmarks("cpu", models=["default", "adaface"])
        self.assertFalse(ok)
        self.assertEqual(text.count("**FAILED**"), 2)

    def test_an_exploding_model_does_not_abort_the_others(self):
        boom = RuntimeError("provider exploded")
        good = {"model": "adaface", "provider": "CPU", "degraded": False, "mean_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0,
                "fps": 1000.0, "vram_mb": None, "failed": None}
        mod = sys.modules[__name__]
        with mock.patch.object(mod, "model_available", return_value=True), \
                mock.patch.object(mod, "benchmark_model", side_effect=[boom, good]), \
                mock.patch.object(mod, "vram_used_mb", return_value=None):
            text, ok = run_performance_benchmarks("cpu", models=["default", "adaface"])
        self.assertFalse(ok)
        self.assertIn("`default` | **FAILED** | RuntimeError: provider exploded", text)
        self.assertIn("`adaface` | CPU", text)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--benchmark", action="store_true", help="run the benchmark (default: only print usage)")
    p.add_argument("--device", default="auto", help="auto|tensorrt|cuda|directml|coreml|cpu")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--ramp-seconds", type=float, default=2.0)
    p.add_argument("--batch-sizes", default="1,4,8,16", help="comma list, '' to skip the sweep")
    p.add_argument("--models", default="", help="comma list of registry keys (default: all)")
    p.add_argument("--out", default="", help="also write the Markdown table here")
    args = p.parse_args(argv)
    if not args.benchmark:
        p.print_help()
        return 0
    sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    text, ok = run_performance_benchmarks(args.device, args.warmup, args.iters, sizes,
                                          [m for m in args.models.split(",") if m] or None, args.ramp_seconds)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
