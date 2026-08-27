"""Run the real compatibility guard once per backend/precision combination.

Each arm is a fresh process because ORT/TensorRT builds and caches sessions at
initialization.  Results are written as JSON-lines beside the output folders;
no precision is declared a winner by this harness.
"""
import argparse
import json
import os
import subprocess
import sys
import time


MATRIX = (("tensorrt", "fp32"), ("tensorrt", "fp16"),
          ("tensorrt", "mixed"), ("cuda", "fp32"), ("cuda", "fp16"),
          ("cpu", "fp32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--mask-engine", default="RealityUX")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                   "..", "output", "precision_matrix"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--timeout", type=float, default=240.0,
                    help="per-arm wall-clock limit (seconds)")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    report_path = args.report or os.path.join(out, "results.jsonl")
    compat = os.path.join(os.path.dirname(__file__), "compat_one.py")
    with open(report_path, "a", encoding="utf-8") as report:
        for provider, precision in MATRIX:
            arm = os.path.join(out, f"{provider}_{precision}")
            cmd = [sys.executable, compat, "--provider", provider,
                   "--precision", precision, "--mask-engine", args.mask_engine,
                   "--enhancer", args.enhancer, "--clip", args.clip,
                   "--source", args.source, "--out", arm]
            started = time.perf_counter()
            try:
                proc = subprocess.run(cmd, text=True, capture_output=True,
                                      check=False, timeout=args.timeout)
                elapsed = time.perf_counter() - started
                metrics = {"provider": provider, "precision": precision,
                           "returncode": proc.returncode,
                           "wall_seconds": round(elapsed, 3),
                           "stdout_tail": proc.stdout[-4000:],
                           "stderr_tail": proc.stderr[-2000:]}
                for line in proc.stdout.splitlines():
                    if line.startswith("METRICS "):
                        metrics["metrics_line"] = line
                    if line.startswith("RESULT "):
                        metrics["result_line"] = line
            except subprocess.TimeoutExpired as exc:
                metrics = {"provider": provider, "precision": precision,
                           "returncode": -2, "timed_out": True,
                           "timeout_seconds": args.timeout,
                           "stdout_tail": str(exc.stdout or "")[-4000:],
                           "stderr_tail": str(exc.stderr or "")[-2000:]}
            except Exception as exc:
                metrics = {"provider": provider, "precision": precision,
                           "returncode": -1, "error": repr(exc)}
            report.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            report.flush()
            print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
