"""Swap a short clip under ONE (precision, mask engine, enhancer) combination
and report whether the result is a VALID face -- not merely whether it threw.

Requirement 18. The two precision failures this project has already recorded
both produced garbage with no exception at all: ESRGAN x4 went BLACK under TRT
FP16, and inswapper produced rainbow smudges from an FP16 overflow. So "it ran"
is not the test. Four checks, aimed at those failure shapes:

  detected    a face is still findable in the output at all
  identity    cosine distance to the source faceset -- catches a swap that ran
              but produced something that is not the person
  texture     stddev inside the face box -- catches BLACK and flat output
  channel     how far the R/B means sit from the green -- catches RAINBOW,
              where one channel blows out and the others do not

One config per PROCESS on purpose: ORT builds its sessions during init, so
changing the precision inside a live process would quietly measure the engines
that were already built.

Usage:
    env/Scripts/python.exe tests/compat_one.py --precision mixed \
        --mask-engine RealityUX --enhancer None --clip <tiny.mp4> --source harjot
"""

import argparse
import os
import sys
import time
import subprocess
import threading

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import angle_bench as ab                                    # noqa: E402
from angle_video import ensure_ffmpeg, run_swap             # noqa: E402
from sample_bench import map_mask_engine                    # noqa: E402


class _GpuTelemetry:
    """Best-effort whole-device telemetry during the timed render."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], text=True,
                    stderr=subprocess.DEVNULL, timeout=2).strip().splitlines()[0]
                util, mem = [float(x.strip()) for x in raw.split(",", 1)]
                self.samples.append((util, mem))
            except Exception:
                pass
            self.stop.wait(0.25)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if not self.samples:
            return float("nan"), float("nan")
        utils, memory = zip(*self.samples)
        return float(max(memory) / 1024.0), float(sum(utils) / len(utils))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "mixed"])
    ap.add_argument("--provider", default="tensorrt",
                    choices=["tensorrt", "cuda", "cpu"],
                    help="execution backend; precision applies to TensorRT")
    ap.add_argument("--mask-engine", default="RealityUX")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # core.py selects the enhancer with an if/elif chain that has NO else, so an
    # unknown name silently adds no enhancer at all -- the run then succeeds,
    # measures the un-enhanced pipeline, and reports PASS for a configuration
    # that was never exercised. That happened here: "GPEN 512" is not a name
    # (the 512 variant is plain "GPEN") and it scored exactly the no-enhancer
    # figure. Refuse unknown names rather than quietly testing nothing.
    # DERIVED from core.py, not hand-listed. A hardcoded allowlist drifts the
    # other way from the bug it guards: this one had gone stale and would have
    # REFUSED "GPEN Realistic", "UltraMax" and "GPEN 256 Ultra", all of which
    # core.py matches. Same parse tests/test_enhancer_names.py uses.
    import re as _re
    from roop import core as _core
    with open(_core.__file__, "r", encoding="utf-8") as _fh:
        KNOWN = set(_re.findall(r"selected_enhancer == '([^']+)'", _fh.read()))
    KNOWN.add("None")
    if args.enhancer not in KNOWN:
        raise SystemExit(
            f"unknown enhancer {args.enhancer!r}; core.py would silently ignore "
            f"it and this run would report PASS for an un-enhanced pipeline. "
            f"Known: {sorted(KNOWN)}")

    ensure_ffmpeg()
    me = (map_mask_engine(args.mask_engine)
          if args.mask_engine not in ("", "None", None) else "None")
    if me is None:
        raise SystemExit(f"unknown mask engine {args.mask_engine!r}")

    label = f"{args.provider}/{args.precision}/{args.mask_engine}/{args.enhancer}"
    try:
        t_init = time.perf_counter()
        g = ab.init_pipeline(args.provider, "realswap", args.enhancer, me, 25.0)
        # Set AFTER init_pipeline built CFG, and BEFORE the providers are
        # decoded into sessions -- core.py reads CFG.trt_precision there.
        g.CFG.trt_precision = args.precision
        from roop.core import decode_execution_providers
        g.execution_providers = decode_execution_providers([args.provider])
        g.execution_threads = 8
        g.CFG.track_identities = True
        g.temporal_detection = True
        g.CFG.temporal_detection = True
        options = ab.build_options(g, "realswap", me)
        init_seconds = time.perf_counter() - t_init

        from two_face_video import load_library_faceset, faceset_mean, cos
        fs = load_library_faceset(args.source)
        mean = faceset_mean(fs)

        out_dir = args.out or os.path.join(
            APP, "output", "compat", args.precision,
            f"{args.mask_engine}__{args.enhancer}".replace(" ", "_"))
        telemetry = _GpuTelemetry()
        telemetry.start()
        t_run = time.perf_counter()
        final = run_swap(args.clip, fs, options, out_dir)
        process_seconds = time.perf_counter() - t_run
        peak_gpu_memory_gb, mean_gpu_util = telemetry.finish()

        from roop.face_util import get_all_faces
        cap = cv2.VideoCapture(final)
        det, ids, texs, chans = 0, [], [], []
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            faces = get_all_faces(fr) or []
            if not faces:
                continue
            det += 1
            f = max(faces, key=lambda q: (q.bbox[2] - q.bbox[0]))
            x0, y0, x1, y1 = [int(v) for v in f.bbox]
            x0, y0 = max(0, x0), max(0, y0)
            sub = fr[y0:y1, x0:x1]
            if sub.size:
                texs.append(float(sub.std()))
                b, gg, r = [float(sub[:, :, i].mean()) for i in range(3)]
                mid = (b + gg + r) / 3.0
                chans.append(max(abs(b - mid), abs(gg - mid), abs(r - mid)))
            if mean is not None and getattr(f, "embedding", None) is not None:
                ids.append(cos(f.embedding, mean))
        cap.release()

        idm = float(np.mean(ids)) if ids else float("nan")
        tex = float(np.mean(texs)) if texs else 0.0
        ch = float(np.mean(chans)) if chans else 0.0
        ok_det = det > 0
        ok_id = ids and idm < 0.9
        ok_tex = tex > 10.0
        ok_ch = ch < 40.0
        verdict = "PASS" if (ok_det and ok_id and ok_tex and ok_ch) else "FAIL"
        flags = "".join([
            "" if ok_det else " NO-FACE", "" if ok_id else " IDENTITY",
            "" if ok_tex else " FLAT/BLACK", "" if ok_ch else " CHANNEL-SKEW"])
        try:
            import psutil
            peak_rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except Exception:
            peak_rss_gb, cpu_percent = float("nan"), float("nan")
        peak_vram_allocated_gb = peak_vram_reserved_gb = float("nan")
        try:
            import torch
            if torch.cuda.is_available():
                peak_vram_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                peak_vram_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        except Exception:
            pass
        gpu_util = mean_gpu_util
        try:
            if not np.isfinite(gpu_util):
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True,
                    stderr=subprocess.DEVNULL, timeout=2).strip().splitlines()[0]
                gpu_util = float(raw)
        except Exception:
            pass
        fps = n / process_seconds if process_seconds > 0 else 0.0
        print(f"RESULT {verdict:4s} {label:<44} frames={n} detected={det} "
              f"id={idm:.3f} texture={tex:.1f} channel={ch:.1f}{flags}",
              flush=True)
        print(f"METRICS provider={args.provider} precision={args.precision} "
              f"init_s={init_seconds:.3f} process_s={process_seconds:.3f} "
              f"fps={fps:.3f} peak_rss_gb={peak_rss_gb:.3f} "
              f"peak_vram_allocated_gb={peak_vram_allocated_gb:.3f} "
              f"peak_vram_reserved_gb={peak_vram_reserved_gb:.3f} "
              f"peak_gpu_memory_gb={peak_gpu_memory_gb:.3f} "
              f"cpu_percent={cpu_percent:.1f} gpu_util_percent={gpu_util:.1f} "
              f"transfers=unavailable", flush=True)
    except Exception as e:
        print(f"RESULT ERROR {label:<44} {type(e).__name__}: {e}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
