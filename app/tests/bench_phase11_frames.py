"""Phase 11: the FRAME paths -- super-resolution, colorizers, classical resamplers.

The face paths live in `bench_phase11_enhancers.py`. This covers the other 15
rows of the matrix, which had never been measured reproducibly: the nine frame
super-resolution models, the two DeOldify colorizers (never run at all), and the
four classical ffmpeg/CPU resamplers (never run at all).

The first pass's frame figures are NOT a starting point. They came from the same
uncommitted run whose face rows were wrong by up to 14x, and they were taken on
a synthetic gradient tiled at 128x128 -- an input a production render never
sees. Here the input is a REAL decoded frame at its native size, which is what
`upscale_after_swap` actually hands these models, so the tile count and
therefore the cost are the production ones.

Each model is Initialize -> warm -> timed -> Release so peak VRAM stays bounded;
a x4 model on a 720p frame produces a 4K output and several of these will not
share a card. CPU utilisation is sampled per row, filling a matrix column that
was pending for every path.

    env/Scripts/python.exe tests/bench_phase11_frames.py
    env/Scripts/python.exe tests/bench_phase11_frames.py --only esrganx2,span_x4
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cv2

import angle_bench as ab


UPSCALERS = ("esrganx2", "esrganx4", "esrgan_anime_x4", "ultrasharp_x4",
             "lsdirx4", "clear_reality_x4", "span_x4", "compact_x4",
             "nomos8k_x4")
COLORIZERS = ("deoldify_artistic", "deoldify_stable")
CLASSICAL = ("lanczos", "fsr", "spline", "sinc")


def real_frame(clip, frame_no):
    """A real decoded frame at native size -- the input these models get."""
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % clip)
    frame = None
    for i in range(frame_no + 1):
        ok, buf = cap.read()
        if not ok or buf is None:
            cap.release()
            raise SystemExit("clip ended at frame %d before %d" % (i, frame_no))
        frame = buf
    cap.release()
    return np.ascontiguousarray(frame)


def vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


class CpuWatch:
    """Mean host CPU% across the timed section."""

    def __enter__(self):
        try:
            import psutil
            self.psutil = psutil
            psutil.cpu_percent()
            self.ok = True
        except Exception:
            self.ok = False
        return self

    def __exit__(self, *exc):
        self.value = None
        if self.ok:
            try:
                self.value = self.psutil.cpu_percent()
            except Exception:
                pass
        return False


def check(out, src, scale):
    """Shape and sanity. A silently-black x4 output is a real failure mode here:
    ESRGAN x4 went BLACK under TensorRT FP16 on this machine, which is why the
    frame models do not force TRT."""
    if out is None:
        return "no output"
    a = np.asarray(out)
    if not np.isfinite(a.astype(np.float32)).all():
        return "non-finite"
    exp = (src.shape[0] * scale, src.shape[1] * scale)
    if scale and a.shape[:2] != exp:
        return "shape %s, expected %s" % (a.shape[:2], exp)
    if float(a.astype(np.float32).std()) < 0.5 * float(src.astype(np.float32).std()):
        return "collapsed/black (std %.1f vs %.1f)" % (a.std(), src.std())
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.environ.get(
        "PHASE11_CLIP", "G:/pinokio/roop-keep/double/d4.mp4"))
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    g = ab.init_pipeline(cfg.provider, cfg.swap_model, "None", "None", 0.0)

    import onnxruntime
    hw = {
        "provider_setting": str(cfg.provider),
        "onnxruntime": onnxruntime.__version__,
    }
    try:
        hw["gpu"] = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        hw["gpu"] = "unknown"

    frame = real_frame(args.clip, args.frame)
    hw["input"] = "%dx%d real frame, %s frame %d" % (
        frame.shape[1], frame.shape[0], os.path.basename(args.clip), args.frame)

    print("Phase 11 frame-path matrix")
    for k, v in hw.items():
        print("  %-20s %s" % (k, v))
    print()

    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    rows = {}

    def timed(label, run, scale, init, release, out_note=""):
        if wanted and label not in wanted:
            return
        base = vram_mb()
        t0 = time.perf_counter()
        try:
            init()
        except Exception as exc:
            print("  %-20s INITIALIZE FAILED: %s: %s" % (label, type(exc).__name__, exc))
            rows[label] = {"status": "init_failed",
                           "error": "%s: %s" % (type(exc).__name__, exc)}
            return
        init_ms = (time.perf_counter() - t0) * 1000
        out = None
        try:
            for _ in range(args.warmup):
                out = run(frame)
        except Exception as exc:
            print("  %-20s RUN FAILED: %s: %s" % (label, type(exc).__name__, exc))
            rows[label] = {"status": "run_failed",
                           "error": "%s: %s" % (type(exc).__name__, exc)}
            try:
                release()
            except Exception:
                pass
            return
        peak = vram_mb()
        per_round = []
        with CpuWatch() as cpu:
            for _r in range(args.rounds):
                t0 = time.perf_counter()
                for _ in range(args.calls):
                    out = run(frame)
                per_round.append((time.perf_counter() - t0) / args.calls * 1000)
        warn = check(out, frame, scale)
        ms = statistics.mean(per_round)
        sd = statistics.pstdev(per_round) if len(per_round) > 1 else 0.0
        vram = (peak - base) if (peak and base) else None
        rows[label] = {
            "status": "measured", "ms": round(ms, 2), "ms_sd": round(sd, 2),
            "fps": round(1000.0 / ms, 3) if ms else None,
            "init_ms": round(init_ms, 1),
            "out_shape": tuple(np.asarray(out).shape) if out is not None else None,
            "vram_mb": round(vram) if vram is not None else None,
            "cpu_pct": cpu.value, "rounds": [round(x, 2) for x in per_round],
            "warning": warn, "note": out_note,
        }
        print("  %-20s %9.2f +- %6.2f ms (%7.3f fps) out %-16s init %8.1f ms "
              "vram %-6s cpu %-5s%s"
              % (label, ms, sd, 1000.0 / ms,
                 rows[label]["out_shape"], init_ms,
                 vram if vram is not None else "?",
                 cpu.value if cpu.value is not None else "?",
                 ("  <-- " + warn) if warn else ""))
        try:
            release()
        except Exception:
            pass
        time.sleep(0.5)

    # ── frame super-resolution ──────────────────────────────────────────────
    from roop.processors.Frame_Upscale import Frame_Upscale
    for sub in UPSCALERS:
        scale = 2 if sub.endswith("x2") else 4
        proc = Frame_Upscale()
        opts = {"devicename": "cuda", "subtype": sub}
        timed(sub, lambda f, p=proc: p.Run(f), scale,
              lambda p=proc, o=opts: p.Initialize(o),
              lambda p=proc: p.Release(),
              out_note="Frame_Upscale.Run, tile/batch chosen by the processor")

    # ── colorizers ──────────────────────────────────────────────────────────
    from roop.processors.Frame_Colorizer import Frame_Colorizer
    grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    for sub in COLORIZERS:
        proc = Frame_Colorizer()
        opts = {"devicename": "cuda", "subtype": sub.replace("deoldify_", "")}
        # DeOldify's input is a frame with no colour to recover; feeding it the
        # already-coloured plate would measure it doing nothing interesting.
        timed(sub, lambda _f, p=proc, gg=grey: p.Run(gg), None,
              lambda p=proc, o=opts: p.Initialize(o),
              lambda p=proc: p.Release(),
              out_note="greyscale input, LAB merge back onto the source")

    # ── classical resamplers ────────────────────────────────────────────────
    import post_swap
    for mode in CLASSICAL:
        label = mode + "_x2"
        timed(label,
              lambda f, m=mode: post_swap._classical_image_apply(f, m, 2), 2,
              lambda: None, lambda: None,
              out_note="post_swap._classical_image_apply, CPU only")

    print()
    print("  ms/frame, ascending:")
    for label, r in sorted(rows.items(), key=lambda kv: kv[1].get("ms") or 1e9):
        if r.get("status") == "measured":
            print("    %-20s %9.2f ms %8.3f fps  out %s"
                  % (label, r["ms"], r["fps"], r["out_shape"]))
        else:
            print("    %-20s %s: %s" % (label, r["status"], r.get("error", "")))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"hardware": hw, "rows": rows}, fh, indent=2)
        print("\n  wrote %s" % args.json)


if __name__ == "__main__":
    main()
