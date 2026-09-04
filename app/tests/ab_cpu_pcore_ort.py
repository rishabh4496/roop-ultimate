"""Counterbalanced A/B: P-core pinning + ORT thread widening, end to end.

Arm A is the shipped default (`cpu_distribution=auto`, ORT serial per session).
Arm B is the requested i9-14900K policy: process pinned to the measured P-core
logicals, `intra_op_num_threads=6`, `inter_op_num_threads=2`.

Everything about the schedule here is forced by prior results on this repo:

  * 600 frames, never 120. A 120-frame window measures TensorRT warm-up and has
    reversed the SIGN of a result on both GPUs (Gate D).
  * NULL CONTROL FIRST. This machine resolves ~50% effects reliably and ~5%
    effects not at all, and its spread has been measured at 3.7% and at 8% in
    different sessions. An A/B run without a same-session null is a delta
    quoted against an unknown floor.
  * ABBA, not AB. The first arm of a process pays the cold TensorRT engine
    build; uncounterbalanced reads of measured-neutral changes have come back
    at +21.8% and +9.8%.
  * SWAP RATE and faces_seen beside fps. A configuration that goes faster by
    finding fewer faces has not got faster, and on this fixture faces_seen also
    discriminates the code path (a RAM-derived stabilizer geometry silently
    switches between sequential and parallel).

Usage::

    env/Scripts/python.exe tests/ab_cpu_pcore_ort.py
    env/Scripts/python.exe tests/ab_cpu_pcore_ort.py --enhancer "GPEN 256 Pro"
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
if APP not in sys.path:
    sys.path.insert(0, APP)

PY = os.path.join(APP, "env", "Scripts", "python.exe")

ARMS = {
    "A_auto": ["--env", "ROOP_CPU_DISTRIBUTION=auto"],
    "B_pcore_ort": ["--env", "ROOP_CPU_DISTRIBUTION=p_only",
                    "--env", "ROOP_ORT_INTRA_THREADS=6",
                    "--env", "ROOP_ORT_INTER_THREADS=2"],
}

# null, null, then ABBA. The nulls are two runs of the SAME arm and their
# spread is the resolution every later delta has to clear.
SCHEDULE = ["A_auto", "A_auto", "A_auto", "B_pcore_ort",
            "B_pcore_ort", "A_auto"]
NULL_COUNT = 2


def run_arm(name, index, args):
    tag = "abcpu_%02d_%s" % (index, name)
    cmd = [PY, os.path.join(HERE, "baseline_controlled.py"),
           "--tag", tag, "--out", args.out,
           "--enhancer", args.enhancer,
           "--start", "0", "--end", str(args.end)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    cmd += ARMS[name]

    started = time.time()
    print("\n[%d/%d] %-12s  starting ..." % (index + 1, len(SCHEDULE), name),
          flush=True)
    proc = subprocess.run(cmd, cwd=APP, capture_output=True, text=True,
                          errors="replace")
    elapsed = time.time() - started

    row = {"arm": name, "position": index + 1, "tag": tag,
           "seconds": round(elapsed, 1), "rc": proc.returncode}
    path = os.path.join(args.out, tag + ".json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            run = data.get("run", data)
            for key in ("fps", "faces_seen", "faces_swapped", "swap_rate",
                        "wrong_faceset"):
                if key in run:
                    row[key] = run[key]
            # P/E utilisation is the direct outcome check on the pin: under
            # p_only the E-core figure must collapse. If it does not, the
            # policy did not reach the worker threads whatever the fps says.
            telem = data.get("telemetry") or {}
            for key in ("mean_cpu_pct", "peak_cpu_p_pct", "peak_cpu_e_pct",
                        "mean_gpu_util_pct", "peak_rss_gb"):
                if key in telem:
                    row[key] = telem[key]
        except Exception as exc:
            row["parse_error"] = "%s: %s" % (type(exc).__name__, exc)
    if row.get("fps") is None:
        tail = [ln for ln in proc.stdout.splitlines() if "fps" in ln.lower()]
        row["stdout_tail"] = tail[-3:] or proc.stdout.splitlines()[-3:]
    print("      %-12s rc=%s fps=%s faces_seen=%s swapped=%s  (%.0fs)"
          % (name, row["rc"], row.get("fps"), row.get("faces_seen"),
             row.get("faces_swapped"), elapsed), flush=True)
    return row


def summarise(rows):
    print("\n" + "=" * 74)
    print("%-4s %-12s %7s %10s %8s %7s %7s %6s" %
          ("pos", "arm", "fps", "faces_seen", "swapped", "P%", "E%", "sec"))
    print("-" * 74)
    for r in rows:
        print("%-4d %-12s %7s %10s %8s %7s %7s %6s" %
              (r["position"], r["arm"], r.get("fps", "-"),
               r.get("faces_seen", "-"), r.get("faces_swapped", "-"),
               r.get("peak_cpu_p_pct", "-"), r.get("peak_cpu_e_pct", "-"),
               r.get("seconds", "-")))
    print("=" * 74)

    nulls = [r.get("fps") for r in rows[:NULL_COUNT] if r.get("fps")]
    floor = None
    if len(nulls) >= 2:
        floor = (max(nulls) - min(nulls)) / statistics.mean(nulls) * 100.0
        print("\nNULL CONTROL  %s  ->  spread %.2f%% "
              "(this is the resolution; a delta below it is not readable)"
              % (" / ".join("%.2f" % n for n in nulls), floor))

    a = [r.get("fps") for r in rows[NULL_COUNT:] if r["arm"] == "A_auto" and r.get("fps")]
    b = [r.get("fps") for r in rows[NULL_COUNT:] if r["arm"] == "B_pcore_ort" and r.get("fps")]
    if not (a and b):
        print("\nnot enough completed arms to compare")
        return
    ma, mb = statistics.mean(a), statistics.mean(b)
    delta = (mb - ma) / ma * 100.0
    print("\nA_auto       mean %.2f fps  (%s)"
          % (ma, ", ".join("%.2f" % v for v in a)))
    print("B_pcore_ort  mean %.2f fps  (%s)"
          % (mb, ", ".join("%.2f" % v for v in b)))
    print("\nDELTA  B vs A = %+.2f%%" % delta)
    if floor is not None:
        if abs(delta) <= floor:
            print("VERDICT: NOT RESOLVABLE -- |%.2f%%| is inside this session's "
                  "%.2f%% null spread. Report as neutral, not as a gain."
                  % (delta, floor))
        else:
            print("VERDICT: %.2f%% exceeds the %.2f%% null spread; the effect "
                  "is readable on this rig." % (abs(delta), floor))

    seen = {r.get("faces_seen") for r in rows if r.get("faces_seen")}
    if len(seen) > 1:
        print("\nWARNING: faces_seen differs across arms %s -- the arms may "
              "not have taken the same code path, so the fps delta is not "
              "attributable to the treatment." % sorted(seen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhancer", default="UltraMax",
                    help="default matches config.yaml's selected_enhancer")
    ap.add_argument("--end", type=int, default=600)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(APP, "output", "ab_cpu_pcore"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("counterbalanced CPU/ORT A/B   enhancer=%s frames=%d schedule=%s"
          % (args.enhancer, args.end, "->".join(SCHEDULE)))

    rows = []
    for index, name in enumerate(SCHEDULE):
        rows.append(run_arm(name, index, args))
        with open(os.path.join(args.out, "summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
    summarise(rows)


if __name__ == "__main__":
    main()
