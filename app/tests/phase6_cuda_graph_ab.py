"""Phase 6: the provider-level CUDA-graph A/B that never reached steady state.

WHAT WAS OPEN. Phase 6 closed its stream question (auxiliary=1 rejected: 29.57
against 32.58 FPS enhanced) and rejected the torch CUDA-graph filter inside
GPEN 256 Pro on a real timing (2.06 ms against 1.67 ms). What it never got was
the PROVIDER-level graph: `OPTIMIZATION_PROGRESS.md` records that the "global
steady-state run did not reach a valid result", so `trt_cuda_graph_enable` is
currently rejected as a default without a number behind it.

WHY IT DID NOT REACH STEADY STATE, ALMOST CERTAINLY. `ROOP_TRT_CUDA_GRAPH=1`
feeds `trt_tuning_namespace()`, which puts the run in a DIFFERENT TensorRT
engine cache directory. Every engine it touches is therefore built from scratch
on the first run -- the same mechanism that made the Phase 5 quality arms time
out inside a 180 s bound while looking "CPU-bound at 1-3% GPU", which is what a
tactic search looks like from outside. A single GPEN-512 build measured 346 s
here on its own.

So each arm runs COLD first (builds its namespace, untimed) and then WARM
(measured), and the two arms are COUNTERBALANCED A,B,B,A because the first arm
of any process in this project reads several fps slow -- uncounterbalanced, that
alone has manufactured +21.8% and +9.8% on measurements that were really
neutral.

Reads SWAP RATE beside FPS. A configuration that goes faster by finding fewer
faces has not got faster.

    env/Scripts/python.exe tests/phase6_cuda_graph_ab.py --tag phase6_4070
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


ARMS = (("graph_off", "0"), ("graph_on", "1"))
ORDER = ("graph_off", "graph_on", "graph_on", "graph_off")   # counterbalanced


def run_once(tag, value, args, phase, timeout):
    out_root = os.path.join(args.out, args.tag)
    cmd = [sys.executable, os.path.join(HERE, "baseline_controlled.py"),
           "--tag", "%s_%s" % (tag, phase),
           "--env", "ROOP_TRT_CUDA_GRAPH=%s" % value,
           "--end", str(args.frames),
           "--out", out_root]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=APP, text=True, capture_output=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": round(time.perf_counter() - started, 1),
                "timeout_s": timeout}
    js = os.path.join(out_root, "%s_%s.json" % (tag, phase))
    result = {"status": "ok" if proc.returncode == 0 else "failed",
              "returncode": proc.returncode,
              "seconds": round(time.perf_counter() - started, 1)}
    if os.path.isfile(js):
        with open(js, encoding="utf-8") as fh:
            data = json.load(fh)
        run = data.get("run", {})
        result.update({
            "fps": run.get("fps"),
            "processing_seconds": run.get("processing_seconds"),
            "faces_seen": run.get("faces_seen"),
            "faces_swapped": run.get("faces_swapped"),
            "wrong_faceset": run.get("wrong_faceset"),
            "peak_gpu_memory_mb": data.get("telemetry", {}).get("peak_gpu_memory_mb"),
            "peak_rss_gb": data.get("telemetry", {}).get("peak_rss_gb"),
        })
        if run.get("faces_seen"):
            result["swap_rate"] = round(
                100.0 * (run.get("faces_swapped") or 0) / run["faces_seen"], 2)
    else:
        result["stderr_tail"] = (proc.stderr or proc.stdout or "")[-800:]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, default=300,
                    help="shorter than the Phase 2 baseline: this is a paired "
                         "A/B, not the locked reference")
    ap.add_argument("--cold-timeout", type=float, default=5400.0,
                    help="an arm may build its entire TensorRT namespace")
    ap.add_argument("--warm-timeout", type=float, default=1800.0)
    ap.add_argument("--out", default=os.path.join(APP, "output", "phase6_cuda_graph"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Phase 6 provider CUDA-graph A/B")
    print("  arms      ROOP_TRT_CUDA_GRAPH=0 vs =1")
    print("  order     %s (counterbalanced)" % ", ".join(ORDER))
    print("  frames    %d per run" % args.frames)
    print("  each arm  cold build pass (<=%ds), then measured passes"
          % args.cold_timeout)
    print()

    results = {"cold": {}, "warm": []}
    started = time.perf_counter()

    # Cold: build each namespace once, untimed for acceptance purposes.
    for tag, value in ARMS:
        print("  [%5.1f min] %-10s cold build ..." % ((time.perf_counter() - started) / 60, tag),
              flush=True)
        cold = run_once(tag, value, args, "cold", args.cold_timeout)
        results["cold"][tag] = cold
        print("  [%5.1f min] %-10s cold %-8s %8.1fs  fps=%s"
              % ((time.perf_counter() - started) / 60, tag, cold["status"],
                 cold["seconds"], cold.get("fps")), flush=True)

    # Warm, counterbalanced.
    for i, tag in enumerate(ORDER):
        value = dict(ARMS)[tag]
        r = run_once(tag, value, args, "warm%d" % i, args.warm_timeout)
        r["arm"] = tag
        r["position"] = i
        results["warm"].append(r)
        print("  [%5.1f min] pos %d %-10s %-8s fps=%-8s swap_rate=%-7s vram=%s"
              % ((time.perf_counter() - started) / 60, i, tag, r["status"],
                 r.get("fps"), r.get("swap_rate"), r.get("peak_gpu_memory_mb")),
              flush=True)

    js = os.path.join(args.out, args.tag + ".json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print()
    means = {}
    for tag, _v in ARMS:
        vals = [r["fps"] for r in results["warm"]
                if r["arm"] == tag and isinstance(r.get("fps"), (int, float))]
        rates = [r["swap_rate"] for r in results["warm"]
                 if r["arm"] == tag and isinstance(r.get("swap_rate"), (int, float))]
        if vals:
            means[tag] = (sum(vals) / len(vals), vals,
                          sum(rates) / len(rates) if rates else None)
    for tag, (mean, vals, rate) in means.items():
        print("  %-10s mean %6.2f fps  from %s   swap rate %s"
              % (tag, mean, [round(v, 2) for v in vals],
                 ("%.2f%%" % rate) if rate is not None else "?"))
    if len(means) == 2:
        off, on = means["graph_off"][0], means["graph_on"][0]
        print("  graph on / off = %.3fx (%+.1f%%)" % (on / off, (on / off - 1) * 100))
        print("  cold build cost: off %ss, on %ss"
              % (results["cold"]["graph_off"]["seconds"],
                 results["cold"]["graph_on"]["seconds"]))
    else:
        # An arm with no fps is NOT automatically "unmeasured". Distinguish:
        #   * every attempt errored the same way  -> a REPRODUCIBLE FAILURE, which
        #     is a finding and a legitimate basis for rejection;
        #   * mixed or absent results             -> genuinely unmeasured.
        # Phase 6 hit the first case on 2026-08-29 and the original wording would
        # have filed a correctness defect as "we did not get round to it".
        for tag, _v in ARMS:
            attempts = [r for r in results["warm"] if r["arm"] == tag]
            statuses = {r["status"] for r in attempts}
            if attempts and statuses == {"failed"}:
                print("  %s FAILED on all %d attempts -- this is a REPRODUCIBLE "
                      "FAILURE, not an unmeasured arm. Read the per-run logs for "
                      "the error and record the failure as the result."
                      % (tag, len(attempts)))
            elif tag not in means:
                print("  %s produced no fps and no consistent error -- genuinely "
                      "unmeasured; do not record it as a rejection." % tag)
    print("\n  wrote %s" % js)


if __name__ == "__main__":
    main()
