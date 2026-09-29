"""Swap-model inference benchmark: latency, VRAM and ACCURACY per execution path.

    app/env/Scripts/python.exe tools/benchmark_swapper.py [--models hififace hyperswap]
        [--iters 300] [--warmup 20] [--rounds 2] [--json out.json]

Arms (each in its OWN process, so peak VRAM is that arm's and nothing else's):
    ort_cuda_fp32_run    plain onnxruntime session.run with numpy (the baseline)
    cuda_fp32_iobind     OptimizedInferenceSession, CUDA EP, device IO binding
    trt_fp32_iobind      TensorRT EP, FP32 engine
    trt_fp16_iobind      TensorRT EP, FP16 engine
    cuda_fp16model       the model converted by convert_onnx_fp16, CUDA EP
                         (skipped for a graph that is already FP16: hyperswap)

Per arm: active provider (read AFTER the timed run -- ORT can drop an EP during
the first inference), first-call time (TensorRT engine build or cache load),
warm-up cycles, median / std / p95 latency for the HOST path (numpy in, numpy
out: what session.run callers pay) and the DEVICE path (inputs already in VRAM,
run_binding), peak VRAM over the arm's own baseline (NVML, sampled every 2 ms),
and the SM clock during timing (a GPU idling at a low clock reads slow).

Accuracy, because precision is the point of the exercise: every arm swaps the
same real faces (insightface's t1.jpg, three largest, source = the next face),
and the outputs are compared with the ort_cuda_fp32_run reference in 8-bit
levels, then pasted back and RE-DETECTED with buffalo_l to score identity
(cosine to the source embedding, higher = the swap carried more of the
source). A faster arm that moves identity has not got faster for free.

Rounds: round 2 runs the arms in reverse order. The first arm to touch a model
pays engine builds and clock ramp; A/B then B/A is this repo's rule
(AGENTS.md: counterbalance every A/B).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
MODELS = APP / "models"
T1 = APP / "env" / "Lib" / "site-packages" / "insightface" / "data" / "images" / "t1.jpg"

MODEL_KEYS = {"hififace": "hififace_256", "hyperswap": "hyperswap_1a_256"}
ARMS = ["ort_cuda_fp32_run", "cuda_fp32_iobind", "trt_fp32_iobind", "trt_fp16_iobind", "cuda_fp16model"]


# ── parent: inputs ─────────────────────────────────────────────────────────────

def prepare_inputs(model: str, workdir: Path) -> Path:
    """Aligned crops + latents for the three largest faces of t1.jpg."""
    import cv2
    from insightface.app import FaceAnalysis
    from roop.processors.frame import model_registry as mr
    img = cv2.imread(str(T1))
    fa = FaceAnalysis(name="buffalo_l", root=str(APP), providers=["CPUExecutionProvider"],
                      allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    faces = sorted(fa.get(img), key=lambda f: -(f.bbox[2] - f.bbox[0]))[:4]
    sw = mr.create_swapper(MODEL_KEYS[model])
    sw.initialize_session(None, "cpu")
    blobs, latents, mats, src_emb = [], [], [], []
    for i in range(3):
        tgt, src = faces[i], faces[(i + 1) % 4]
        crop, M = sw.align(img, tgt.kps)
        feed = sw.pre_process(crop, src.embedding)
        blobs.append(feed[sw.image_input_name])
        latents.append(feed[sw.embed_input_name])
        mats.append(M)
        src_emb.append(src.normed_embedding)
    path = workdir / f"{model}_inputs.npz"
    np.savez(path, blobs=np.concatenate(blobs), latents=np.concatenate(latents),
             mats=np.stack(mats), src_emb=np.stack(src_emb),
             image_name=sw.image_input_name, embed_name=sw.embed_input_name,
             model_file=str(MODELS / sw.spec["file"]), denorm=sw.model_denormalize)
    return path


# ── child: one arm ────────────────────────────────────────────────────────────

class VramPeak:
    """Device-used MiB via NVML every 2 ms (WDDM has no per-process figure)."""

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak = 0
        self._stop = threading.Event()
        self.clocks = []

    def used(self) -> float:
        return self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2 ** 20

    def __enter__(self):
        self.base = self.used()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(0.002)

    def clock(self):
        self.clocks.append(self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM))

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def _stats(ms):
    a = np.asarray(ms)
    return {"median_ms": float(np.median(a)), "std_ms": float(a.std()),
            "p95_ms": float(np.percentile(a, 95)), "n": int(a.size)}


def run_arm(arm: str, inputs: Path, out: Path, iters: int, warmup: int) -> None:
    sys.path.insert(0, str(APP))
    from roop.processors.frame.inference_engine import (OptimizedInferenceSession,
                                                        _prepare_runtime, convert_onnx_fp16,
                                                        default_cache_root, is_fp16_graph)
    _prepare_runtime()
    import torch
    torch.cuda.init()
    d = np.load(inputs)
    img_name, emb_name = str(d["image_name"]), str(d["embed_name"])
    model_file = str(d["model_file"])
    feeds = [{img_name: d["blobs"][i:i + 1], emb_name: d["latents"][i:i + 1]} for i in range(len(d["blobs"]))]
    result = {"arm": arm, "model_file": Path(model_file).name}

    if arm == "cuda_fp16model" and is_fp16_graph(model_file):
        result["skipped"] = "graph is already FP16"
        out.write_text(json.dumps(result))
        return

    with VramPeak() as vram:
        t0 = time.perf_counter()
        if arm == "ort_cuda_fp32_run":
            import onnxruntime as ort
            from roop.utilities import get_onnx_session_options
            sess = ort.InferenceSession(model_file, get_onnx_session_options(),
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            host = lambda f: sess.run(None, f)
            device = None
            providers = lambda: sess.get_providers()[0]
            host(feeds[0])
        else:
            prov, prec = {"cuda_fp32_iobind": ("cuda", "fp32"), "trt_fp32_iobind": ("tensorrt", "fp32"),
                          "trt_fp16_iobind": ("tensorrt", "fp16"), "cuda_fp16model": ("cuda", "fp32")}[arm]
            path = model_file
            if arm == "cuda_fp16model":
                path = convert_onnx_fp16(model_file, str(default_cache_root() / "fp16_models" /
                                                         (Path(model_file).stem + ".fp16.onnx")))
            sess = OptimizedInferenceSession(path, prov, prec, warmup=True)
            host = sess.run
            dev_feeds = [{k: torch.as_tensor(v, device="cuda") for k, v in f.items()} for f in feeds]
            device = lambda i: sess.run_binding(dev_feeds[i])
            providers = lambda: sess.session.get_providers()[0]
        result["first_call_s"] = time.perf_counter() - t0
        for i in range(warmup):
            host(feeds[i % len(feeds)])
        result["warmup_cycles"] = warmup

        ms = []
        for i in range(iters):
            f = feeds[i % len(feeds)]
            t = time.perf_counter()
            host(f)
            ms.append((time.perf_counter() - t) * 1000)
            if i % 50 == 0:
                vram.clock()
        result["host"] = _stats(ms)
        if device is not None:
            for i in range(warmup):
                device(i % len(feeds))
            torch.cuda.synchronize()
            ms = []
            for i in range(iters):
                t = time.perf_counter()
                device(i % len(feeds))
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1000)
                if i % 50 == 0:
                    vram.clock()
            result["device"] = _stats(ms)
        outputs = np.concatenate([host(f)[0] for f in feeds])
    result["active_provider"] = providers()
    result["peak_vram_mib"] = vram.peak - vram.base
    result["sm_clock_mhz_median"] = float(np.median(vram.clocks)) if vram.clocks else None
    np.save(str(out) + ".npy", outputs)
    out.write_text(json.dumps(result))


# ── parent: identity ──────────────────────────────────────────────────────────

def identity_scores(inputs: Path, outputs: np.ndarray) -> list:
    """Paste each output into t1.jpg, re-detect, cosine to its source."""
    import cv2
    from insightface.app import FaceAnalysis
    from roop.processors.frame.swapper_base import BaseFaceSwapper
    d = np.load(inputs)
    img = cv2.imread(str(T1))
    fa = FaceAnalysis(name="buffalo_l", root=str(APP), providers=["CPUExecutionProvider"],
                      allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    class _S(BaseFaceSwapper):
        initialize_session = pre_process = infer = post_process = None
    _S.__abstractmethods__ = frozenset()
    s = _S()
    s.model_denormalize = bool(d["denorm"])
    scores = []
    for i, out in enumerate(outputs):
        crop = s.to_crop(out)
        M = d["mats"][i]
        mask = cv2.GaussianBlur(np.pad(np.ones((216, 216), np.float32), 20), (0, 0), 8)
        frame = s.paste_back(crop, M, img, mask)
        centre = cv2.transform(np.array([[[128.0, 128.0]]]), cv2.invertAffineTransform(M))[0, 0]
        faces = fa.get(frame)
        best = min(faces, key=lambda f: np.hypot(*(np.array([(f.bbox[0] + f.bbox[2]) / 2,
                                                             (f.bbox[1] + f.bbox[3]) / 2]) - centre)))
        scores.append(float(np.dot(best.normed_embedding, d["src_emb"][i])))
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["hififace", "hyperswap"], choices=list(MODEL_KEYS))
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--_arm", help=argparse.SUPPRESS)
    ap.add_argument("--_inputs", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--_out", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._arm:
        run_arm(a._arm, a._inputs, a._out, a.iters, a.warmup)
        return 0

    sys.path.insert(0, str(APP))
    os.environ.setdefault("ROOP_ORT_IO_BINDING", "0")
    report = {"gpu": None, "models": {}}
    try:
        import torch
        report["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for model in a.models:
            print(f"\n== {model}: preparing t1.jpg crops", flush=True)
            inputs = prepare_inputs(model, tmp)
            rows = {}
            for rnd in range(a.rounds):
                order = a.arms if rnd % 2 == 0 else list(reversed(a.arms))
                for arm in order:
                    out = tmp / f"{model}_{arm}_{rnd}.json"
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--_arm", arm,
                           "--_inputs", str(inputs), "--_out", str(out),
                           "--iters", str(a.iters), "--warmup", str(a.warmup)]
                    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(APP))
                    if proc.returncode != 0 or not out.exists():
                        print(f"   {arm:18s} round {rnd + 1}: FAILED\n{proc.stderr[-1500:]}", flush=True)
                        continue
                    r = json.loads(out.read_text())
                    r["round"] = rnd + 1
                    if not r.get("skipped"):
                        r["_outputs"] = str(out) + ".npy"
                    rows.setdefault(arm, []).append(r)
                    if r.get("skipped"):
                        print(f"   {arm:18s} round {rnd + 1}: skipped ({r['skipped']})", flush=True)
                    else:
                        dev = r.get("device", {}).get("median_ms")
                        print(f"   {arm:18s} round {rnd + 1}: {r['active_provider']:26s} host "
                              f"{r['host']['median_ms']:6.2f} ms  device "
                              f"{('%6.2f ms' % dev) if dev else '     -   '}  "
                              f"vram +{r['peak_vram_mib']:.0f} MiB  first {r['first_call_s']:.1f}s", flush=True)

            ref_rows = rows.get("ort_cuda_fp32_run")
            ref = np.load(ref_rows[0]["_outputs"]) if ref_rows else None
            summary = {}
            for arm, rs in rows.items():
                live = [r for r in rs if not r.get("skipped")]
                if not live:
                    summary[arm] = {"skipped": rs[0]["skipped"]}
                    continue
                outputs = np.load(live[0]["_outputs"])
                entry = {
                    "active_provider": live[0]["active_provider"],
                    "first_call_s": [round(r["first_call_s"], 2) for r in live],
                    "warmup_cycles": live[0]["warmup_cycles"],
                    "host_median_ms": [round(r["host"]["median_ms"], 3) for r in live],
                    "host_std_ms": [round(r["host"]["std_ms"], 3) for r in live],
                    "device_median_ms": [round(r["device"]["median_ms"], 3) for r in live if "device" in r],
                    "device_std_ms": [round(r["device"]["std_ms"], 3) for r in live if "device" in r],
                    "peak_vram_mib": [round(r["peak_vram_mib"]) for r in live],
                    "sm_clock_mhz": [r["sm_clock_mhz_median"] for r in live],
                    "identity": [round(v, 4) for v in identity_scores(inputs, outputs)],
                }
                if ref is not None:
                    scale = 127.5 if np.load(inputs)["denorm"] else 255.0
                    diff = np.abs(outputs - ref) * scale
                    entry["vs_fp32_levels_mean"] = round(float(diff.mean()), 4)
                    entry["vs_fp32_levels_p999"] = round(float(np.percentile(diff, 99.9)), 3)
                    entry["vs_fp32_levels_max"] = round(float(diff.max()), 2)
                summary[arm] = entry
            report["models"][model] = summary

    print("\n== summary (median per round; identity = cosine to source, 3 faces)")
    for model, summary in report["models"].items():
        print(f"\n{model}")
        for arm, e in summary.items():
            if "skipped" in e:
                print(f"  {arm:18s} skipped: {e['skipped']}")
                continue
            print(f"  {arm:18s} {e['active_provider']:26s} host {e['host_median_ms']} ms (std {e['host_std_ms']})"
                  f"  device {e['device_median_ms'] or '-'} ms  vram {e['peak_vram_mib']} MiB"
                  f"  first {e['first_call_s']} s  sm {e['sm_clock_mhz']} MHz")
            print(f"  {'':18s} identity {e['identity']}  vs fp32 mean {e.get('vs_fp32_levels_mean')}"
                  f" p99.9 {e.get('vs_fp32_levels_p999')} max {e.get('vs_fp32_levels_max')} levels")
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
