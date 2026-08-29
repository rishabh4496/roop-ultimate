"""Phase 11: per-face cost of EVERY face-restoration path, on the production provider.

WHY THIS FILE EXISTS. The RTX 4070 table in docs/PHASE11_ENHANCER_MATRIX.md was
produced by an ad-hoc pass that was never committed, so it cannot be re-run, and
three of its rows disagree with this repo's own measurements on this same card:

    docs table            this repo's 2026-08-24 table (CLAUDE.md)
    GPEN 256   81.52 ms   5.3 ms
    GPEN 512   79.05 ms   --
    GPEN Realistic 512    77.32 ms          27.5 ms

CodeFormer (36.32), UltraMax (30.32) and GFPGAN (45.16) agree with history; the
GPEN family does not, and 256 and 512 coming back within 3% of each other is not
something a 4x pixel change can do on real GPU execution. That is the signature
of a fixed per-call host cost, or of ORT having silently fallen back to CPU.

So: measured through angle_bench.init_pipeline, which is the only way TensorRT's
DLLs are on PATH -- without it ORT falls back to CPU without saying so and a 4 ms
model reports as 210 ms. Provider, swap model and pool sizes come from
config.yaml (the models the user actually runs), not from CLI defaults.

Each path is Initialize -> warm -> timed rounds -> Release, so peak VRAM stays
bounded (GPEN 2048 does not share the card with ten other sessions) and every row
is independent. Output is checked for the GFPGAN-class failure that `is_usable`
cannot see: a finite, in-range, but COLLAPSED face with its dynamic range gone.

    env/Scripts/python.exe tests/bench_phase11_enhancers.py
    env/Scripts/python.exe tests/bench_phase11_enhancers.py --only "GPEN 256,GPEN"
"""
import argparse
import json
import os
import re
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


# (label, module, class, plugin_options). The label is the string core.py
# matches on, so a typo fails loudly instead of benching NO enhancer at all --
# that exact defect invalidated four benchmarks on 2026-08-23.
PATHS = [
    ("Codeformer",        "Enhance_CodeFormer",         "Enhance_CodeFormer",         {"fp16": False}),
    ("Codeformer (fp16)", "Enhance_CodeFormer",         "Enhance_CodeFormer",         {"fp16": True}),
    ("GFPGAN",            "Enhance_GFPGAN",             "Enhance_GFPGAN",             {}),
    ("GPEN 256",          "Enhance_GPEN",               "Enhance_GPEN",               {"size": 256}),
    ("GPEN",              "Enhance_GPEN",               "Enhance_GPEN",               {"size": 512}),
    ("GPEN 1024",         "Enhance_GPEN",               "Enhance_GPEN",               {"size": 1024}),
    ("GPEN 2048",         "Enhance_GPEN",               "Enhance_GPEN",               {"size": 2048}),
    ("GPEN 256 Pro",      "Enhance_GPEN256Pro",         "Enhance_GPEN256Pro",         {}),
    ("GPEN Realistic",    "Enhance_GPENRealistic",      "Enhance_GPENRealistic",      {}),
    ("UltraMax",          "Enhance_UltraMax",           "Enhance_UltraMax",           {}),
    ("Restoreformer++",   "Enhance_RestoreFormerPPlus", "Enhance_RestoreFormerPPlus", {}),
    ("DMDNet",            "Enhance_DMDNet",             "Enhance_DMDNet",             {}),
]

# DMDNet is the ONLY path here that reads its `source_faceset` and `target_face`
# arguments; every other processor ignores both (UltraMax's periocular pass takes
# its eye positions from the fixed FFHQ-512 template via `swap_template_points`,
# not from the face). So real metadata is built once and handed only to DMDNet,
# which keeps every other row byte-identical to a run without it.
NEEDS_FACE_METADATA = {"DMDNet"}


def real_face_crop(clip, frame_no, size):
    """A real aligned face crop at the size the swapper emits.

    Not a synthetic gradient: a gradient has no skin, no pores and no eye
    structure, so every texture and clarity path in these processors is measured
    doing nothing on it. `realswap` emits `size` (256), which is what the
    enhancer is handed in production.
    """
    # Read FORWARD rather than seeking. cv2's seek on long HEVC either fails or
    # returns a frame up to 16 off the one asked for, and a bench that cannot say
    # which frame it measured cannot be re-run.
    from roop.face_util import get_all_faces, align_crop

    cap = cv2.VideoCapture(clip)  # noqa: F841 (kept for the seek note above)
    if not cap.isOpened():
        cap.release()
        raise SystemExit("could not open %s" % clip)

    frame = None
    for i in range(frame_no + 1):
        ok, buf = cap.read()
        if not ok or buf is None:
            cap.release()
            raise SystemExit("clip ended at frame %d, before the requested %d (%s)"
                             % (i, frame_no, clip))
        frame = buf

    # Scan forward for the first frame that actually has a face, so a requested
    # frame that happens to be a cutaway does not abort the run.
    scanned = frame_no
    for _ in range(240):
        faces = get_all_faces(frame)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            cap.release()
            crop, M = align_crop(frame, face.kps, size)
            # The pipeline attaches the crop affine to the face at
            # ProcessMgr.py:3954 from this same align_crop call, and DMDNet
            # reads it (`face.matrix * scale_factor`). A detector-fresh Face has
            # `matrix` None, so without this DMDNet dies on None * float -- it
            # was never a missing-metadata problem in the model, just an
            # un-aligned Face.
            face.matrix = M
            return np.ascontiguousarray(crop), scanned, face
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        scanned += 1
    cap.release()
    raise SystemExit("no face in frames %d..%d of %s" % (frame_no, scanned, clip))


def vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def sm_mhz():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.sm",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def ramp(run_one, seconds):
    """Drive the model continuously until the card leaves its idle clock.

    WHY THIS IS NOT OPTIONAL. A single 512 face crop is a few tens of
    milliseconds of GPU work with host work either side, so a per-face bench is
    a train of short bursts. This RTX 4070 sits at 1065 MHz against a 3135 MHz
    max under that pattern -- 34% of clock, at 44 C and 55 W, with nvidia-smi
    reporting throttle reason 0x1, GpuIdle. It is not thermal; the card simply
    never ramps.

    Measured cost of ignoring it -- same code, same crop, two runs an hour apart:
    UltraMax 86.75 -> 132.95 ms (+53%), GPEN 256 Pro 6.96 -> 12.60 (+81%),
    GPEN 2048 250.43 -> 335.96 (+34%). That is larger than most effects anyone
    would want to measure here, and it is invisible unless the clock is recorded.

    Locking clocks with `nvidia-smi -lgc` needs administrator rights and is not
    available here, so the bench ramps instead and REPORTS the clock it reached.
    Two rows measured at different sm_mhz are not comparable to each other.
    """
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        run_one()


class CpuWatch:
    """Mean host CPU% across the timed section, filling the matrix's CPU column."""

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


def collapsed(out, src):
    """The GFPGAN FP16 failure: finite, in range, and flat.

    `is_usable` was written for NaN/black (the GPEN 1024/2048 overflow) and
    cannot see this one -- a TRT FP16 engine that keeps every value finite while
    losing its dynamic range, returning a grey rectangle 59/255 from the truth.
    It was also FAST that way, which is how it came to be documented as the
    cheapest restorer here. Gate on the ratio to the INPUT so that a legitimately
    soft restorer is not accused of collapsing.
    """
    if out is None:
        return "no output"
    a = np.asarray(out, dtype=np.float32)
    if not np.isfinite(a).all():
        return "non-finite"
    src_std = float(np.asarray(src, np.float32).std())
    ratio = float(a.std()) / max(src_std, 1e-6)
    if ratio < 0.5:
        return "collapsed (std %.1f vs input %.1f)" % (a.std(), src_std)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.environ.get(
        "PHASE11_CLIP", "G:/pinokio/roop-keep/single/s1.mp4"))
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--crop", type=int, default=256,
                    help="crop size the enhancer is handed; realswap emits 256")
    ap.add_argument("--calls", type=int, default=30, help="timed calls per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--ramp", type=float, default=4.0,
                    help="seconds of continuous inference before timing, to get "
                         "the card off its idle clock; see ramp()")
    ap.add_argument("--pool", type=int, default=None,
                    help="pool_size for pooled processors (default: the app's own "
                         "session_pool.pool_size(), i.e. what production resolves)")
    ap.add_argument("--source", default="harjot",
                    help="faceset name; only DMDNet consumes it")
    ap.add_argument("--only", default="", help="comma-separated labels to run")
    ap.add_argument("--json", default="", help="write rows here")
    args = ap.parse_args()

    # Production stack, read live. Tool CLI defaults are not production; that
    # rule has invalidated whole sessions in this repo twice.
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = cfg.provider

    g = ab.init_pipeline(provider, cfg.swap_model, "UltraMax", "mask_realityux",
                         float(getattr(cfg, "swap_model_mask_strength", 0.0) or 0.0))
    g.codeformer_fidelity = float(g.CFG.codeformer_fidelity)

    # The app's own resolution, not int(config). `perf_trt_pool` is commonly
    # 'auto', and the VRAM tier that 'auto' resolves to is exactly the thing a
    # pool number must not be guessed at across two different cards.
    from roop import session_pool
    pool = args.pool if args.pool is not None else session_pool.pool_size()

    import onnxruntime
    hw = {
        "provider_setting": provider,
        "execution_providers": [str(p)[:44] for p in (g.execution_providers or [])],
        "onnxruntime": onnxruntime.__version__,
        "pool_size": pool,
        "crop": args.crop,
    }
    try:
        hw["gpu"] = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        hw["gpu"] = "unknown"

    print("Phase 11 face-enhancer matrix")
    for k, v in hw.items():
        print("  %-22s %s" % (k, v))

    crop, used_frame, target_face = real_face_crop(args.clip, args.frame, args.crop)
    hw["source"] = "%s frame %d" % (os.path.basename(args.clip), used_frame)
    print("  %-22s real face from %s frame %d, %s, std %.1f"
          % ("crop", os.path.basename(args.clip), used_frame, crop.shape, crop.std()))
    print()

    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    paths = [p for p in PATHS if not wanted or p[0] in wanted]
    if wanted:
        missing = set(wanted) - set(p[0] for p in paths)
        if missing:
            raise SystemExit("unknown label(s): %s -- valid: %s"
                             % (sorted(missing), [p[0] for p in PATHS]))

    # A label core.py does not match adds NO enhancer, silently. Fail here.
    import roop.core as core
    with open(core.__file__, "r", encoding="utf-8") as fh:
        valid = set(re.findall(r"selected_enhancer == '([^']+)'", fh.read()))
    unmatched = [p[0] for p in paths if p[0] not in valid]
    if unmatched:
        raise SystemExit("labels not matched by core.get_processing_plugins: %s"
                         % unmatched)

    rows = {}
    for idx, (label, module, clsname, opts) in enumerate(paths):
        mod = __import__("roop.processors." + module, fromlist=[clsname])
        proc = getattr(mod, clsname)()
        options = {"devicename": "cuda", "pool_size": pool}
        options.update(opts)

        base_vram = vram_mb()
        t_init = time.perf_counter()
        try:
            proc.Initialize(options)
        except Exception as exc:
            print("  %-20s INITIALIZE FAILED: %s: %s" % (label, type(exc).__name__, exc))
            rows[label] = {"status": "init_failed",
                           "error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        init_ms = (time.perf_counter() - t_init) * 1000

        # Only DMDNet reads these; see NEEDS_FACE_METADATA.
        src_fs, tgt = (None, None)
        if label in NEEDS_FACE_METADATA:
            try:
                from two_face_video import load_library_faceset
                src_fs, tgt = load_library_faceset(args.source), target_face
            except Exception as exc:
                print("  %-20s SKIPPED: needs a faceset and this failed: %s: %s"
                      % (label, type(exc).__name__, exc))
                rows[label] = {"status": "skipped",
                               "error": "%s: %s" % (type(exc).__name__, exc)}
                try:
                    proc.Release()
                except Exception:
                    pass
                continue

        out = None
        try:
            for _ in range(args.warmup):
                out, _ = proc.Run(src_fs, tgt, crop)
        except Exception as exc:
            print("  %-20s RUN FAILED: %s: %s" % (label, type(exc).__name__, exc))
            rows[label] = {"status": "run_failed",
                           "error": "%s: %s" % (type(exc).__name__, exc)}
            try:
                proc.Release()
            except Exception:
                pass
            continue

        peak_vram = vram_mb()
        # Bring the card off its idle clock before timing anything.
        ramp(lambda: proc.Run(src_fs, tgt, crop), args.ramp)
        clocks = [sm_mhz()]
        per_round = []
        with CpuWatch() as cpu:
            for _r in range(args.rounds):
                t0 = time.perf_counter()
                for _ in range(args.calls):
                    out, _ = proc.Run(src_fs, tgt, crop)
                per_round.append((time.perf_counter() - t0) / args.calls * 1000)
                clocks.append(sm_mhz())

        warn = collapsed(out, crop)
        ms = statistics.mean(per_round)
        sd = statistics.pstdev(per_round) if len(per_round) > 1 else 0.0
        outshape = tuple(np.asarray(out).shape) if out is not None else None
        vram_delta = (peak_vram - base_vram) if (peak_vram and base_vram) else None

        rows[label] = {
            "status": "measured",
            "ms": round(ms, 2),
            "ms_sd": round(sd, 2),
            "fps": round(1000.0 / ms, 2) if ms else None,
            "init_ms": round(init_ms, 1),
            "out_shape": outshape,
            "vram_mb": round(vram_delta) if vram_delta is not None else None,
            "cpu_pct": cpu.value,
            "sm_mhz": (round(sum(c for c in clocks if c)
                             / len([c for c in clocks if c]))
                       if any(c for c in clocks) else None),
            "rounds": [round(x, 2) for x in per_round],
            "warning": warn,
        }
        print("  %-20s %7.2f +- %4.2f ms  (%6.2f fps)  out %-14s init %7.1f ms  "
              "vram %-6s cpu %-5s sm %-6s%s"
              % (label, ms, sd, 1000.0 / ms, outshape, init_ms,
                 vram_delta if vram_delta is not None else "?",
                 cpu.value if cpu.value is not None else "?",
                 rows[label]["sm_mhz"],
                 ("  <-- " + warn) if warn else ""))

        try:
            proc.Release()
        except Exception:
            pass
        if idx + 1 < len(paths):
            time.sleep(0.5)   # let the driver return the freed contexts

    print()
    print("  ms/face, ascending:")
    for label, r in sorted(rows.items(), key=lambda kv: kv[1].get("ms") or 1e9):
        if r.get("status") == "measured":
            print("    %-20s %7.2f ms   %6.2f fps   out %s"
                  % (label, r["ms"], r["fps"], r["out_shape"]))
        else:
            print("    %-20s %s: %s" % (label, r["status"], r.get("error", "")))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"hardware": hw, "rows": rows}, fh, indent=2)
        print("\n  wrote %s" % args.json)


if __name__ == "__main__":
    main()
