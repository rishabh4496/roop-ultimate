"""Phase 5: the model-quality precision matrix that previously produced no result.

WHY THE EARLIER ATTEMPT FAILED, AND IT WAS NOT THE WORKLOAD. The 2026-08-28 run
bounded each arm at 180 s and recorded: "TensorRT FP32 and FP16 GPEN 256 Pro
arms both timed out during the full quality workload and produced no valid
quality result; the remaining arms were stopped." The 2026-08-29 retry of TRT
FP16 "remained CPU-bound with 1-3% GPU use and produced no quality result after
five minutes."

Both are the same thing: **TensorRT keeps a SEPARATE ENGINE CACHE NAMESPACE PER
PRECISION**, and several of those namespaces on this machine are 0 bytes. An arm
that switches precision therefore builds every engine it touches from scratch --
detector, recognition, landmarks, swapper, mask, enhancer. A single GPEN-512
build measured 346 s here on its own. "1-3% GPU use" is what an engine BUILD
looks like from outside: the builder is profiling tactics on the host, so the
arm was never CPU-bound in the sense of falling back to the CPU provider.

So the fix is not a bigger timeout bolted onto one pass. Each arm runs TWICE:

  cold  -- builds whatever its precision namespace is missing, generous bound
  warm  -- the measurement, against caches that now exist

The verdict and every performance number come from the WARM run. The cold
seconds are reported separately because "what does switching precision cost the
first time" is a real question with a real answer, and burying it inside the
measurement is what produced two sessions of unusable results.

The fixture is deliberately tiny. compat_one swaps the clip and then RE-DETECTS
every output frame, so its cost is quadratic in clip length for no extra
signal -- the four checks (face still findable, identity, texture, channel skew)
are about whether the precision produced a valid face at all, not about
throughput.

    env/Scripts/python.exe tests/phase5_quality_matrix.py --tag phase5_4070
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import telemetry as tel


MATRIX = (("tensorrt", "fp32"), ("tensorrt", "fp16"), ("tensorrt", "mixed"),
          ("cuda", "fp32"), ("cuda", "fp16"), ("cpu", "fp32"))


def make_fixture(source, frames, out_path, start=0):
    """Cut a tiny clip. Regenerated rather than committed, so the RTX 3060 host
    builds the identical fixture from the same source clip and frame range."""
    if os.path.isfile(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", source, "-vf", "select=between(n\\,%d\\,%d)" % (start, start + frames - 1),
           "-vsync", "0", "-c:v", "libx264", "-crf", "12", "-an", out_path]
    subprocess.run(cmd, check=True, timeout=300)
    return out_path


def parse_arm(text):
    out = {}
    m = re.search(r"^RESULT\s+(\w+)\s+(\S+)\s+(.*)$", text, re.M)
    if m:
        out["verdict"] = m.group(1)
        out["label"] = m.group(2)
        for key, pat in (("frames", r"frames=(\d+)"), ("detected", r"detected=(\d+)"),
                         ("identity", r"id=([\d.nan]+)"), ("texture", r"texture=([\d.]+)"),
                         ("channel", r"channel=([\d.]+)")):
            mm = re.search(pat, m.group(3))
            if mm:
                out[key] = mm.group(1)
        flags = re.findall(r"(NO-FACE|IDENTITY|FLAT/BLACK|CHANNEL-SKEW)", m.group(3))
        out["flags"] = flags
    m = re.search(r"^METRICS\s+(.*)$", text, re.M)
    if m:
        for pair in m.group(1).split():
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[k] = v
    m = re.search(r"^RESULT ERROR\s+\S+\s+(.*)$", text, re.M)
    if m:
        out["verdict"] = "ERROR"
        out["error"] = m.group(1).strip()
    return out


def run_arm(provider, precision, args, phase, timeout):
    out_dir = os.path.join(args.out, args.tag, "%s_%s" % (provider, precision), phase)
    cmd = [sys.executable, os.path.join(HERE, "compat_one.py"),
           "--provider", provider, "--precision", precision,
           "--mask-engine", args.mask_engine, "--enhancer", args.enhancer,
           "--clip", args.clip, "--source", args.source, "--out", out_dir]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=APP, env=env, text=True,
                              capture_output=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        parsed = parse_arm(proc.stdout or "")
        parsed["returncode"] = proc.returncode
        parsed["wall_s"] = round(elapsed, 2)
        if proc.returncode != 0 and "verdict" not in parsed:
            parsed["verdict"] = "ERROR"
            parsed["error"] = (proc.stderr or proc.stdout or "")[-600:]
        return parsed
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "wall_s": round(time.perf_counter() - started, 2),
                "timeout_s": timeout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source-clip", default="G:/pinokio/roop-keep/single/s4.mp4")
    ap.add_argument("--fixture-frames", type=int, default=24)
    ap.add_argument("--fixture-start", type=int, default=60)
    ap.add_argument("--clip", default=None, help="skip fixture generation")
    ap.add_argument("--source", default="harjot", help="faceset name")
    ap.add_argument("--enhancer", default="GPEN 256 Pro")
    ap.add_argument("--mask-engine", default="RealityUX")
    ap.add_argument("--cold-timeout", type=float, default=2700.0,
                    help="generous: this arm may build every TRT engine it uses")
    ap.add_argument("--warm-timeout", type=float, default=900.0)
    ap.add_argument("--out", default=os.path.join(APP, "output", "phase5_quality"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ffmpeg both makes the fixture and runs inside the render.
    import shutil
    if not shutil.which("ffmpeg"):
        for cand in (os.path.join("G:/pinokio", "bin", "miniforge", "Library", "bin"),
                     os.path.join("G:/pinokio", "bin", "miniconda", "Library", "bin")):
            if os.path.isfile(os.path.join(cand, "ffmpeg.exe")):
                os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
                break
        else:
            raise SystemExit("ffmpeg not found; needed for the fixture and the render")

    if args.clip is None:
        args.clip = make_fixture(
            args.source_clip, args.fixture_frames,
            os.path.join(args.out, "fixture_%d_%dframes.mp4"
                         % (args.fixture_start, args.fixture_frames)),
            start=args.fixture_start)

    print("Phase 5 model-quality precision matrix")
    print("  fixture      %s (%d frames from %s @%d)"
          % (args.clip, args.fixture_frames, os.path.basename(args.source_clip),
             args.fixture_start))
    print("  stack        realswap / %s / %s, source %s"
          % (args.enhancer, args.mask_engine, args.source))
    print("  arms         %s" % ", ".join("%s/%s" % a for a in MATRIX))
    print("  each arm     cold (build, <=%ds) then warm (measured, <=%ds)"
          % (args.cold_timeout, args.warm_timeout))
    print()

    rows = {}
    started = time.perf_counter()
    for provider, precision in MATRIX:
        key = "%s/%s" % (provider, precision)
        print("  [%5.1f min] %-16s cold ..." % ((time.perf_counter() - started) / 60, key),
              flush=True)
        cold = run_arm(provider, precision, args, "cold", args.cold_timeout)
        print("  [%5.1f min] %-16s cold %-8s %8.1fs"
              % ((time.perf_counter() - started) / 60, key,
                 cold.get("verdict", "?"), cold.get("wall_s", 0)), flush=True)

        warm = run_arm(provider, precision, args, "warm", args.warm_timeout)
        rows[key] = {"cold": cold, "warm": warm}
        v = warm.get("verdict", "?")
        detail = ""
        if v in ("PASS", "FAIL"):
            detail = ("id=%s texture=%s channel=%s fps=%s"
                      % (warm.get("identity", "?"), warm.get("texture", "?"),
                         warm.get("channel", "?"), warm.get("fps", "?")))
            if warm.get("flags"):
                detail += "  flags=%s" % ",".join(warm["flags"])
        elif v == "ERROR":
            detail = str(warm.get("error", ""))[:160]
        print("  [%5.1f min] %-16s WARM %-8s %8.1fs  %s"
              % ((time.perf_counter() - started) / 60, key, v,
                 warm.get("wall_s", 0), detail), flush=True)
        print(flush=True)

    js = os.path.join(args.out, args.tag + ".json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"fixture": args.clip, "enhancer": args.enhancer,
                   "mask_engine": args.mask_engine, "source": args.source,
                   "rows": rows}, fh, indent=2)

    print("  %-16s %-8s %10s %10s %8s %9s %9s"
          % ("arm", "verdict", "cold s", "warm s", "fps", "identity", "texture"))
    for key, r in rows.items():
        w = r["warm"]
        print("  %-16s %-8s %10.1f %10.1f %8s %9s %9s"
              % (key, w.get("verdict", "?"), r["cold"].get("wall_s", 0),
                 w.get("wall_s", 0), w.get("fps", "-"),
                 w.get("identity", "-"), w.get("texture", "-")))
    print("\n  wrote %s" % js)
    bad = [k for k, r in rows.items()
           if r["warm"].get("verdict") not in ("PASS", "FAIL")]
    if bad:
        print("  arms without a valid quality result: %s" % ", ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
