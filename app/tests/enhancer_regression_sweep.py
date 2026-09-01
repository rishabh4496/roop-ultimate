"""Every enhancer, end to end, with proof that it actually ran.

WHY THIS EXISTS, AND WHY IT IS NOT bench_phase11_enhancers.py. That file times
each restorer in isolation, which is the right instrument for per-face cost and
the wrong one for integration: it does not run ProcessMgr's enhance path, so it
cannot see an enhancer that the pipeline never reaches, reaches at the wrong
point in the chain, or reaches and then discards.

The failure this guards is specific and has happened. On the RTX 3060, four
enhancers failed on 60 of 60 frames with a GPU error; ProcessMgr caught it,
wrote the ORIGINAL frame, and the swap audit reported `swapped (every face)
100.0%` -- because that audit counts faces it was HANDED, not work performed. A
throughput bench called those arms fast and fine, precisely because not
enhancing is cheap. Separately, `identity_detail_strength 0.35` restored nothing
on V1 facesets with no message anywhere.

So the acceptance here is a CALL COUNT, not a pixel difference and not a return
code: `ROOP_PROFILE=1` publishes an `enhance` stage whose `calls` must equal the
faces the run swapped. One enhance per swapped face means the processor was
constructed, reached, and executed. `None` correctly has no stage at all, and
that asymmetry is asserted rather than tolerated.

A pixel difference is deliberately NOT the acceptance test: two renders of one
unchanged configuration on this pipeline differ by mean ~0.71/255 (see
tests/measure_output_noise_floor.py), so a small delta proves nothing either
way.

    env/Scripts/python.exe tests/enhancer_regression_sweep.py
    env/Scripts/python.exe tests/enhancer_regression_sweep.py \
        --only "UltraMax,Adaptive" --end 30 --json out.json
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

import fixtures   # noqa: E402

# Every selectable restoration path, in the spelling roop/core.py matches on.
# The spelling matters more than it looks: `get_processing_plugins` compares
# against exact strings and a miss adds NO enhancer at all, silently. Four
# separate benches once ran their "CodeFormer" arm against nothing for exactly
# that reason, which is why tests/test_enhancer_names.py parses the valid set
# out of core.py rather than trusting a list like this one.
ENHANCERS = [
    "None",
    "GFPGAN",
    "Codeformer",
    "Codeformer (fp16)",
    "DMDNet",
    "GPEN 256",
    "GPEN 256 Pro",
    "GPEN Realistic",
    "GPEN",
    "GPEN 1024",
    "GPEN 2048",
    "UltraMax",
    "Restoreformer++",
    # Adaptive is not a model. It is a selector that sits AFTER the mask stages
    # (see get_processing_plugins) rather than before them like every manual
    # enhancer, so it is not a like-for-like substitution for the model it
    # picks, and it needs its own row.
    "Adaptive",
]

_STAGE_ROW = re.compile(r"^\s+(\w+)\s+([\d.]+)s\s+([\d.]+)%\s+(\d+)\s+([\d.]+)\s*$")


def parse_stages(text):
    block = re.search(r"==== STAGE TIMING \(ROOP_PROFILE\).*?\n(.*?)\n=====",
                      text, re.S)
    if not block:
        return {}
    out = {}
    for line in block.group(1).splitlines():
        m = _STAGE_ROW.match(line)
        if m:
            out[m.group(1)] = {"total_s": float(m.group(2)),
                               "calls": int(m.group(4)),
                               "ms_per_call": float(m.group(5))}
    return out


def parse_swapped(text):
    """Faces the pipeline decided to swap, from its own audit."""
    total = 0
    block = re.search(r"==== SWAP AUDIT.*?\n(.*?)(?:\n\n|\n\[|\Z)", text, re.S)
    if not block:
        return None
    for line in block.group(1).splitlines():
        m = re.match(r"\s+swapped[^\d]*?(\d+)\s+[\d.]+%\s*$", line)
        if m:
            total += int(m.group(1))
    return total or None


def parse_fps(text):
    m = re.findall(r"took ([\d.]+) secs, ([\d.]+) frames/s", text)
    return (float(m[-1][1]), float(m[-1][0])) if m else (None, None)


def parse_wrong_faceset(text):
    wrong = 0
    for m in re.finditer(r"WRONG FACESET APPLIED on (\d+) of", text):
        wrong += int(m.group(1))
    return wrong


def run_one(name, args, out_root):
    tag = "enh_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    out = os.path.join(out_root, tag)
    cmd = [sys.executable, os.path.join(HERE, "two_face_video.py"),
           "--tag", tag, "--video", args.video, "--sources", args.sources,
           "--start", str(args.start), "--end", str(args.end),
           "--capture", str(args.capture), "--provider", args.provider,
           "--swap-model", args.swap_model, "--enhancer", name,
           "--mask-engine", args.mask_engine, "--codec", args.codec,
           "--tracking", "1", "--threads", str(args.threads), "--out", out]
    env = dict(os.environ)
    env["ROOP_PROFILE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if args.keep_small_card_enhancer:
        # A sub-7GB card strips the enhancer before it can run. That gate is
        # correct there, but it also makes an enhancer row on such a card
        # meaningless unless it is lifted deliberately.
        env["ROOP_SMALL_CARD_ENHANCER"] = "keep"
    log_path = os.path.join(out_root, tag + ".log")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - started
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    stages = parse_stages(text)
    enhance = stages.get("enhance")
    swapped = parse_swapped(text)
    fps, secs = parse_fps(text)
    # Adaptive is a selector, so a counted `enhance` call proves only that the
    # WRAPPER ran. When it chooses `none` the wrapper returns immediately, so
    # the arm shows one enhance call per face at ~0 ms and comes out as the
    # fastest row in the sweep -- indistinguishable, on the call count alone,
    # from a restorer doing real work. Measured on this 4070: BALANCED on the
    # locked d4 fixture chose `none` for 60 of 60 faces at 0.0 ms/face and
    # 1.95 fps, against 1.87 for `--enhancer None`. So this row is graded on
    # the selector's own decision counts as well.
    decisions = re.search(r"\[AdaptiveEnhancer\] decisions=(\{[^}]*\})", text)
    reasons = re.search(r"reasons=(\{[^}]*\})", text)
    quality_band = re.search(r"quality=(\{[^}]*\})", text)
    row = {
        "enhancer": name, "rc": rc, "wall_s": round(wall, 1),
        "fps": fps, "render_s": secs,
        "swapped_faces": swapped,
        "enhance_calls": enhance["calls"] if enhance else 0,
        "enhance_ms_per_face": enhance["ms_per_call"] if enhance else None,
        "wrong_faceset": parse_wrong_faceset(text),
        "gpu_errors": len(re.findall(r"GPU error|CUDNN|RUN-FAIL|Traceback", text)),
        "adaptive_decisions": decisions.group(1) if decisions else None,
        "adaptive_reasons": reasons.group(1) if reasons else None,
        "adaptive_quality": quality_band.group(1) if quality_band else None,
        "log": log_path,
    }

    if rc != 0:
        row["verdict"] = "FAIL: return code %s" % rc
    elif swapped is None:
        row["verdict"] = "FAIL: no swap audit -- the run did not reach the swap"
    elif name == "None":
        # The asymmetry is the point: no enhancer must mean no stage. If an
        # `enhance` stage appears here, something is enhancing that nobody asked
        # for, and every other row's attribution is wrong.
        row["verdict"] = ("PASS" if row["enhance_calls"] == 0
                          else "FAIL: %d enhance calls with enhancer None"
                               % row["enhance_calls"])
    elif row["enhance_calls"] == 0:
        row["verdict"] = ("FAIL: NOT EXECUTED -- no enhance stage. The render "
                          "still returned 0 and the audit still read 100%; "
                          "that is the documented shape of this failure.")
    elif row["enhance_calls"] < swapped:
        row["verdict"] = ("FAIL: enhanced %d of %d swapped faces"
                          % (row["enhance_calls"], swapped))
    elif row["wrong_faceset"]:
        row["verdict"] = "FAIL: %d wrong-faceset applications" % row["wrong_faceset"]
    elif name == "Adaptive" and row["adaptive_decisions"] is None:
        row["verdict"] = ("FAIL: no selector summary -- the wrapper ran but "
                          "published no decisions")
    elif name == "Adaptive" and re.fullmatch(r"\{'none': \d+\}",
                                             row["adaptive_decisions"] or ""):
        # Not a defect in itself -- refusing to restore an already-good face is
        # this selector's stated policy. It IS a result that must never be
        # reported as an ordinary enhancer pass, because the output is the
        # unenhanced swap.
        row["verdict"] = ("NO-OP: selector chose 'none' for every face "
                          "(%s, %s). Output is the unenhanced swap."
                          % (row["adaptive_reasons"], row["adaptive_quality"]))
    else:
        row["verdict"] = "PASS"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=fixtures.clip("double/d4.mp4"))
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=30)
    ap.add_argument("--capture", type=int, default=4930)
    ap.add_argument("--only", default="", help="comma-separated enhancer names")
    ap.add_argument("--provider", default=None, help="defaults to config.yaml")
    ap.add_argument("--swap-model", default=None, help="defaults to config.yaml")
    ap.add_argument("--mask-engine", default=None, help="defaults to config.yaml")
    ap.add_argument("--codec", default=None, help="defaults to config.yaml")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--keep-small-card-enhancer", action="store_true",
                    help="lift the sub-7GB strip so an enhancer row means "
                         "something on a small card")
    ap.add_argument("--out", default=os.path.join(APP, "output", "enhancer_sweep"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    args.provider = args.provider or str(cfg.provider)
    args.swap_model = args.swap_model or str(cfg.swap_model)
    args.mask_engine = args.mask_engine or str(cfg.mask_engine)
    args.codec = args.codec or str(cfg.output_video_codec)
    args.threads = args.threads or int(cfg.max_threads)

    names = ([n.strip() for n in args.only.split(",") if n.strip()]
             if args.only else list(ENHANCERS))
    unknown = [n for n in names if n not in ENHANCERS]
    if unknown:
        raise SystemExit("unknown enhancer(s) %s; core.py matches exact strings "
                         "and a miss silently adds no enhancer at all" % unknown)

    os.makedirs(args.out, exist_ok=True)
    print("[sweep] %d enhancers | %s / %s / %s / %s / %s threads | frames %s..%s"
          % (len(names), args.swap_model, args.mask_engine, args.provider,
             args.codec, args.threads, args.start, args.end), flush=True)

    rows = []
    for name in names:
        row = run_one(name, args, args.out)
        rows.append(row)
        print("[sweep] %-20s %-6s enhance %4d/%-4s  %6s ms/face  %5s fps  %s"
              % (name, "rc=%d" % row["rc"], row["enhance_calls"],
                 row["swapped_faces"],
                 ("%.1f" % row["enhance_ms_per_face"])
                 if row["enhance_ms_per_face"] else "-",
                 ("%.2f" % row["fps"]) if row["fps"] else "-",
                 row["verdict"]), flush=True)

    failed = [r for r in rows if not r["verdict"].startswith("PASS")]
    print("\n[sweep] %d of %d PASS" % (len(rows) - len(failed), len(rows)))
    for r in failed:
        print("[sweep]   %s -- %s  (%s)" % (r["enhancer"], r["verdict"], r["log"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print("[sweep] wrote %s" % args.json)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
