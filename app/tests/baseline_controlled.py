"""Phase 2: the controlled, reproducible baseline the plan requires.

WHY THIS EXISTS. `PERFORMANCE_BASELINE.md`'s "OFFICIAL BASELINE -- TO BE FILLED
IN PHASE 2" block is still blank in every field, and the plan says the baseline
locks only once Phase 2 produces a reproducible benchmark. The 2026-08-28
"Controlled RTX 4070 reference" was a step toward it but disqualified its own
fixture: on `d1.mp4` the two people overlap in every scanned frame (separation
0.107), so "identity quality is not a clean acceptance workload". It also left
P95 latency, decode FPS, encode FPS, CPU utilisation, P/E-core split, transfer
time, synchronisation time and queue depth unmeasured, and its VRAM figure was
sampled telemetry rather than a peak.

WHAT CHANGED. The fixture is `d4.mp4`, which the 2026-08-23 two-faceset audit
graded at 100% / 97% coverage with ZERO wrong-faceset applications -- the clip
`d1` could not be graded on at all. The window is 600 frames rather than 141 so
the measurement is not dominated by warm-up, and the run reports warm-up and
steady-state separately, which the plan's baseline rules ask for.

Every field the plan's Phase 2 names is filled or explicitly marked
unavailable. Per-stage latency, decode and encode throughput come from the
pipeline's own ROOP_PROFILE probe; host and device telemetry come from
`tests/telemetry.py`, sampled against the child AND its descendants because
decode and encode are ffmpeg subprocesses.

    env/Scripts/python.exe tests/baseline_controlled.py --tag phase2_4070

Run the identical command on the RTX 3060 with a different --tag. Nothing here
is keyed to a GPU: models, threads and pools come from that machine's own
config.yaml and its own VRAM tier.
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

import fixtures
import telemetry as tel


# The controlled workload. Constants, not defaults to be casually overridden --
# "use the same representative workload when comparing optimization phases" is
# the baseline rule this file exists to enforce.
#
# `clip_id` is the workload's identity and is what makes two runs comparable; it
# is recorded verbatim in every result. `video` is only where that same clip
# happens to live on the machine now running, resolved at import because the two
# validation targets do not share a drive layout (the 4070 has
# `G:/pinokio/roop-keep/`, the 3060 has `C:\pinokio\roop keep\` and no G: at
# all). Resolving the location does NOT license changing the clip: a different
# fixture is a different baseline, not a 3060 result.
WORKLOAD = {
    "clip_id": "double/d4.mp4",
    "video": fixtures.clip("double/d4.mp4"),
    "sources": "harjot,gargee",
    # The locked 2026-08-29 RTX 4070 baseline's own fixture identity, from
    # PERFORMANCE_BASELINE.md. Checked at run time because `double/d4.mp4` and
    # `duo/d4.mp4` are DIFFERENT CLIPS THAT SHARE A FILENAME -- 1280x720 versus
    # 854x480. Resolving by name alone put the wrong one into a 3060 run that
    # otherwise looked completely valid.
    "expect": {"width": 1280, "height": 720},
    "start": 0,
    "end": 600,
    # PINNED, and it must stay pinned. The auto-capture scan used to be bounded
    # by WALL CLOCK (`--capture-budget 30`), so the fixture it produced was a
    # function of how fast the machine happened to be at that moment:
    #
    #   arm 1  646 frames scanned in 30.0s -> seed 4930, separation 1.039
    #   arm 2  629 frames scanned in 30.1s -> seed 4930, separation 1.039
    #   arm 3  598 frames scanned in 30.1s -> seed 4930, separation 1.039
    #   arm 4  409 frames scanned in 30.1s -> seed 2930, separation 0.990  <-- !
    #
    # (RTX 4070, 2026-08-31, four arms of one counterbalanced set.) A transient
    # slowdown cut the scan short and silently changed WHICH SOURCE CAPTURES the
    # baseline ran with, so that arm was not the same experiment as the other
    # three -- a benchmark feeding back into its own fixture.
    #
    # It is worse across targets than within one: the 3060 is roughly 2x slower,
    # so the same 30-second box buys it about half the scan, and the two GPUs
    # could compare different captures while both reported the locked fixture.
    #
    # 6 of 7 arms chose 4930 whenever they were allowed to finish, so that is
    # the scan's own answer, recorded rather than re-derived under a stopwatch.
    "capture_frame": 4930,
    "reason": "d4 graded 100%/97% with zero wrong-faceset applications on "
              "2026-08-23; d1 (the previous fixture) is 0% gradeable for "
              "identity because both people overlap in every frame",
}


def parse_stage_timing(text):
    """Pull the ROOP_PROFILE table out of the run's stdout."""
    stages = {}
    block = re.search(r"==== STAGE TIMING \(ROOP_PROFILE\).*?\n(.*?)\n=====",
                      text, re.S)
    if not block:
        return stages
    for line in block.group(1).splitlines():
        m = re.match(r"\s+(\w+)\s+([\d.]+)s\s+([\d.]+)%\s+(\d+)\s+([\d.]+)\s*$",
                     line)
        if m:
            stages[m.group(1)] = {
                "total_s": float(m.group(2)),
                "share_pct": float(m.group(3)),
                "calls": int(m.group(4)),
                "ms_per_call": float(m.group(5)),
            }
    return stages


def parse_detailed_stage_profile(text):
    """Pull the machine-readable Phase 14 profile from child stdout."""
    marker = (r"==== DETAILED STAGE PROFILE \(ROOP_PROFILE_DETAIL\) ====\s*\n"
              r"(.*?)\n=======================================================")
    block = re.search(marker, text, re.S)
    if not block:
        return {}
    try:
        value = json.loads(block.group(1).strip())
    except (TypeError, ValueError):
        return {"parse_error": "invalid JSON in detailed stage profile"}
    return value if isinstance(value, dict) else {"parse_error": "profile is not an object"}


def parse_adaptive_downgrades(text):
    """What the sub-7GB policy actually changed before the render started.

    WHY THIS IS NOT OPTIONAL. The result record used to store `provider` from
    config.yaml -- the REQUEST -- while the laptop's safety policy had already
    disabled TensorRT, dropped the enhancer to None, skipped RealityUX's BiSeNet
    parser and forced CPU decode. So a 3060 row read `provider: tensorrt,
    enhancer: GPEN 256 Pro` for a run that used neither: a record describing a
    stack that did not exist, which is the same "looks wired, is not" defect
    this project keeps finding.

    It also makes the fixture check insufficient on its own. Matching the locked
    clip is necessary for comparability but not sufficient -- an arm that
    silently dropped the enhancer is doing less work, so a slower number does
    not mean a slower machine.
    """
    found = {}
    if re.search(r"sub-7GB GPU: TensorRT disabled", text):
        found["provider"] = "TensorRT disabled by the sub-7GB RSS policy; CUDA/CPU used"
    m = re.search(r"RSS safety: enhancer '([^']+)' -> '([^']+)'", text)
    if m:
        found["enhancer"] = "%s -> %s (sub-7GB RSS gate)" % (m.group(1), m.group(2))
    if re.search(r"skipping the auxiliary BiSeNet parser", text):
        found["mask_engine"] = "RealityUX degraded to XSeg only; BiSeNet parser skipped"
    if re.search(r"decode safety: NVDEC -> CPU", text):
        found["decode"] = "NVDEC -> CPU (sub-7GB RSS policy)"
    return found


def parse_run(text):
    """Frames, seconds, fps and the swap audit, from the pipeline's own output.

    The processing rate comes from ONE authoritative line -- the encoder's
    "took N secs, M frames/s" -- and from nothing else. A loose `([\d.]+)\s*fps`
    search over this log matches the SOURCE CLIP'S frame rate and reports 30.0
    for a render that ran at 12.2. Ask for the exact line or report nothing.
    """
    out = {}
    m = re.search(r"took\s+([\d.]+)\s+secs?,\s+([\d.]+)\s+frames/s", text)
    if m:
        out["processing_seconds"] = float(m.group(1))
        out["fps"] = float(m.group(2))
    m = re.search(r"\[bench\] output:\s*(\d+)\s*frames", text)
    if m:
        out["frames"] = int(m.group(1))
    if out.get("frames") and out.get("processing_seconds"):
        # Recompute rather than trust the printed rate, and ENFORCE the
        # agreement. A cross-check that is computed and never compared is
        # decoration -- this file's first version reported the SOURCE CLIP's
        # 30.0 fps for a render that ran at 12.20, because a loose
        # `([\d.]+)\s*fps` search matched the video geometry line. That number
        # was heading for the LOCKED baseline, where a 3x inflation would have
        # made every later optimisation on both cards read as a regression.
        # 30 fps is a plausible render rate; nothing about it invites a second
        # look. Only the arithmetic does.
        out["fps_check"] = round(out["frames"] / out["processing_seconds"], 2)
        printed = out.get("fps")
        if printed and abs(printed - out["fps_check"]) > max(0.05, 0.02 * printed):
            out["fps_mismatch"] = True
            raise SystemExit(
                "fps disagreement: the run printed %.2f fps but %d frames in "
                "%.2f s is %.2f fps. One of the two is being read from the "
                "wrong line -- refusing to record either."
                % (printed, out["frames"], out["processing_seconds"],
                   out["fps_check"]))
    m = re.search(r"faces seen\s+(\d+)", text, re.I)
    if m:
        out["faces_seen"] = int(m.group(1))
    m = re.search(r"swapped \(identity lock\)\s+(\d+)", text, re.I)
    if m:
        out["faces_swapped"] = int(m.group(1))
    wrong = re.findall(r"WRONG FACESET APPLIED on (\d+) of (\d+) swaps", text)
    if wrong:
        out["wrong_faceset"] = sum(int(a) for a, _b in wrong)
        out["attributed_swaps"] = sum(int(b) for _a, b in wrong)
    m = re.search(r"ROOP_TRT_POOL=(\d+), ROOP_DETMASK_POOL=(\d+)", text)
    if m:
        out["trt_pool"], out["detmask_pool"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"execution_threads=(\d+)", text)
    if m:
        out["threads"] = int(m.group(1))
    return out


def ensure_ffmpeg(env):
    """Put ffmpeg on the child's PATH, or fail loudly.

    The render shells out to ffmpeg for decode and encode. Pinokio's own shell
    has it; an ordinary terminal may not, and the failure surfaces late -- the
    swap completes and the encode dies -- so it is resolved up front. Searched
    rather than hardcoded, so this works on the RTX 3060 host too.
    """
    import shutil
    if shutil.which("ffmpeg", path=env.get("PATH", "")):
        return env
    home = fixtures.pinokio_home()
    candidates = [
        os.path.join(home, "bin", "miniforge", "Library", "bin"),
        os.path.join(home, "bin", "miniconda", "Library", "bin"),
        os.path.join(APP, "env", "Library", "bin"),
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "ffmpeg.exe")) or \
           os.path.isfile(os.path.join(cand, "ffmpeg")):
            env["PATH"] = cand + os.pathsep + env.get("PATH", "")
            return env
    raise SystemExit(
        "ffmpeg is not on PATH and was not found in %s -- the render would "
        "swap successfully and then fail at encode. Add it and re-run."
        % candidates)


def software_stack(device_id=0):
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except Exception:
        pass
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:
        pass
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=10)
        info["ffmpeg"] = out.split("\n")[0].split()[2]
    except Exception:
        info["ffmpeg"] = "not on PATH"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader", "--id=%d" % int(device_id)],
            text=True, timeout=10).strip()
        info["gpu"] = out
    except Exception:
        info["gpu"] = "unknown"
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except Exception:
        for root, _dirs, files in os.walk(os.path.join(APP, "models", "trt_cache")):
            m = re.search(r"trt(\d+\.\d+\.\d+\.\d+)", root)
            if m:
                info["tensorrt"] = m.group(1)
                break
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="device tag, e.g. phase2_4070")
    ap.add_argument("--video", default=WORKLOAD["video"])
    ap.add_argument("--sources", default=WORKLOAD["sources"])
    ap.add_argument("--start", type=int, default=WORKLOAD["start"])
    ap.add_argument("--end", type=int, default=WORKLOAD["end"])
    ap.add_argument("--enhancer", default="GPEN 256 Pro")
    ap.add_argument("--mask-engine", default="RealityUX")
    ap.add_argument("--provider", default=None,
                    help="explicit inference provider for a controlled A/B; "
                         "default uses config.yaml")
    ap.add_argument("--codec", default=None,
                    choices=("libx264", "libx265", "libvpx-vp9",
                             "h264_nvenc", "hevc_nvenc"),
                    help="explicit output codec for a controlled A/B; "
                         "default uses config.yaml's output_video_codec")
    ap.add_argument("--stabilization-mode", choices=("auto", "off", "on"),
                    default="auto",
                    help="controlled Phase 12 override for all stabilizers")
    ap.add_argument("--color-transfer-mode", default=None,
                    choices=("none", "rct", "lct", "mkl", "idt"),
                    help="controlled Phase 12 color-processing override")
    ap.add_argument("--identity-detail-strength", type=float, default=None,
                     help="FaceSet V2 persistent source-detail restoration; "
                          "default uses config.yaml")
    ap.add_argument("--temporal-compositing-mode", choices=("auto", "off", "on"),
                    default="auto", help="controlled Phase 12 compositor override")
    ap.add_argument("--target", choices=("RTX 3060", "RTX 4070"),
                    default=None, help="validation target label for the record")
    ap.add_argument("--cuda-device-id", type=int, default=0,
                    help="physical CUDA device index used by the child")
    ap.add_argument("--threads", type=int, default=None,
                    help="default: the device's own resolved thread count")
    ap.add_argument("--out", default=os.path.join(APP, "output", "phase2_baseline"))
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment for the render; used by the Phase 6 "
                         "CUDA-graph A/B, which is an env switch that also "
                         "changes the TensorRT cache namespace")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    threads = args.threads if args.threads is not None else int(cfg.max_threads)

    provider = args.provider if args.provider is not None else str(cfg.provider)
    # The codec comes from config.yaml for the same reason the provider and the
    # stabilizers do: this harness must render the stack the user actually runs.
    # It used to default to a hardcoded "libx264" while the locked 4070 baseline
    # and the matrix table both record hevc_nvenc, so the documented
    # reproduction command -- which passes no --codec -- did not reproduce the
    # baseline it claims to. Phase 13 measured that difference at +16.97%
    # end-to-end, which is far too large to leave to a stale default.
    if args.codec is None:
        args.codec = str(cfg.output_video_codec)
    cmd = [sys.executable, os.path.join(HERE, "two_face_video.py"),
           "--tag", args.tag,
           "--video", args.video,
           "--sources", args.sources,
           "--start", str(args.start), "--end", str(args.end),
           "--capture", str(WORKLOAD["capture_frame"]),
           "--provider", provider,
           "--swap-model", str(cfg.swap_model),
           "--enhancer", args.enhancer,
           "--mask-engine", args.mask_engine,
           "--codec", args.codec,
           "--tracking", "1",
           # From config.yaml, not from the harness defaults. Production runs all
           # three stabilizers; they are the single largest GPU lever recorded in
           # this project (s1 16.42 -> ~12.6 fps), so a baseline without them
           # would not be the pipeline the user renders and could not be compared
           # to a 3060 run that had them on.
           "--stabilize-face", "1" if bool(cfg.stabilize_face) else "0",
           "--stabilize-mask", "1" if bool(cfg.stabilize_mask) else "0",
           "--stabilize-mask-strength", str(cfg.stabilize_mask_strength),
           "--stabilize-enhancer", "1" if bool(cfg.stabilize_enhancer) else "0",
           "--threads", str(threads),
           "--cuda-device-id", str(args.cuda_device_id),
           "--swap-model-mask-strength", str(cfg.swap_model_mask_strength),
           "--merger-clarity", str(getattr(cfg, "merger_clarity", 0.0)),
           "--identity-detail-strength", str(
               getattr(cfg, "identity_detail_strength", 0.0)
               if args.identity_detail_strength is None
               else args.identity_detail_strength),
           "--temporal-compositing-strength", str(
               getattr(cfg, "temporal_compositing_strength", 0.65)),
           "--out", os.path.join(args.out, args.tag)]

    if args.stabilization_mode != "auto":
        enabled = "1" if args.stabilization_mode == "on" else "0"
        cmd.extend(["--stabilize-face", enabled,
                    "--stabilize-mask", enabled,
                    "--stabilize-enhancer", enabled])
    if args.color_transfer_mode is not None:
        cmd.extend(["--color-transfer-mode", args.color_transfer_mode])
    if args.temporal_compositing_mode != "auto":
        if args.temporal_compositing_mode == "on":
            cmd.append("--temporal-compositing")

    env = dict(os.environ)
    env["ROOP_PROFILE"] = "1"          # per-stage latency, decode and encode
    env["PYTHONIOENCODING"] = "utf-8"
    for pair in args.env:
        if "=" in pair:
            k, v = pair.split("=", 1)
            env[k] = v
    if args.temporal_compositing_mode == "off":
        # This setting is a Python global, so set the child environment before
        # launching two_face_video rather than adding an argument that child
        # does not own.
        env["ROOP_TEMPORAL_COMPOSITING_OFF"] = "1"
    env = ensure_ffmpeg(env)
    os.environ["PATH"] = env["PATH"]   # so software_stack() can read its version

    stack = software_stack(args.cuda_device_id)
    print("Phase 2 controlled baseline")
    for k, v in stack.items():
        print("  %-14s %s" % (k, v))
    print("  %-14s %s frames %d..%d, sources %s"
          % ("workload", os.path.basename(args.video), args.start, args.end,
             args.sources))
    print("  %-14s %s / %s / %s / %s, %d threads"
          % ("stack", cfg.swap_model, args.enhancer, args.mask_engine,
             args.codec, threads))
    print("  %-14s face=%s mask=%s(%.2f) enhancer=%s(%.2f), method %s"
          % ("stabilizers", bool(cfg.stabilize_face), bool(cfg.stabilize_mask),
             float(cfg.stabilize_mask_strength), bool(cfg.stabilize_enhancer),
             float(cfg.stabilize_enhancer_strength), cfg.stabilize_method))
    print("  %-14s %s" % ("fixture reason", WORKLOAD["reason"]))

    fixture_fp = fixtures.fingerprint(args.video)
    fixture_diff = fixtures.matches(fixture_fp, WORKLOAD["expect"])
    if fixture_fp:
        print("  %-14s %s  %sx%s, %s frames"
              % ("fixture", fixture_fp.get("path"), fixture_fp.get("width"),
                 fixture_fp.get("height"), fixture_fp.get("frames")))
    if fixture_diff:
        print()
        print("  !! FIXTURE MISMATCH on %s -- this run is NOT the locked "
              "baseline workload." % ", ".join(fixture_diff))
        print("     expected %s, opened %s"
              % (WORKLOAD["expect"],
                 {k: fixture_fp.get(k) for k in WORKLOAD["expect"]}))
        print("     The result is still recorded, but flagged so it cannot be "
              "filed as a cross-target comparison against the RTX 4070 row.")
    print()

    started = time.perf_counter()
    last = [started]

    def progress(line):
        # The user's standing rule: surface processing fps roughly every three
        # minutes during a run, not only at the end.
        now = time.perf_counter()
        if now - last[0] >= 180:
            last[0] = now
            m = re.search(r"([\d.]+)\s*fps", line, re.I)
            print("  [%5.1f min] %s" % ((now - started) / 60,
                                        line.strip()[:150]), flush=True)
        elif re.search(r"error|traceback|refus|warn", line, re.I):
            print("  ! %s" % line.strip()[:150], flush=True)

    rc, out, telem = tel.run_sampled(
        cmd, env=env, cwd=APP, on_line=progress,
        device_id=args.cuda_device_id)
    elapsed = time.perf_counter() - started

    log = os.path.join(args.out, args.tag + ".log")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(out)

    stages = parse_stage_timing(out)
    run = parse_run(out)
    downgrades = parse_adaptive_downgrades(out)
    if downgrades:
        print()
        print("  !! ADAPTIVE DOWNGRADES -- the requested stack is NOT what ran:")
        for key in sorted(downgrades):
            print("       %-12s %s" % (key, downgrades[key]))
        print("     Recorded as adaptive_downgrades; this run is marked NOT "
              "comparable to the locked baseline, which ran the full stack.")
    frames = run.get("frames") or (args.end - args.start)

    result = {
        "tag": args.tag,
        "validation_target": args.target,
        "returncode": rc,
        "software": stack,
        "workload": {"clip_id": WORKLOAD["clip_id"],
                      "fixture": fixture_fp,
                      "fixture_expected": WORKLOAD["expect"],
                      "fixture_mismatch": fixture_diff,
                      # Comparability needs BOTH the locked clip and an
                      # undowngraded stack. A run that dropped the enhancer is
                      # doing less work, so its fps is not this machine's answer
                      # to the 4070's number.
                      "adaptive_downgrades": downgrades,
                      "comparable_to_locked_baseline": not fixture_diff and not downgrades,
                      "video": args.video, "sources": args.sources,
                      "start": args.start, "end": args.end,
                      "enhancer": args.enhancer, "mask_engine": args.mask_engine,
                      "codec": args.codec,
                      "swap_model": str(cfg.swap_model),
                      "provider": provider, "threads": threads,
                      "identity_detail_strength": (
                          getattr(cfg, "identity_detail_strength", 0.0)
                          if args.identity_detail_strength is None
                          else args.identity_detail_strength),
                      "reason": WORKLOAD["reason"]},
        "extra_env": {p.split("=", 1)[0]: p.split("=", 1)[1]
                      for p in args.env if "=" in p},
        "wall_seconds": round(elapsed, 3),
        "run": run,
        "stages": stages,
        "detailed_stage_profile": parse_detailed_stage_profile(out),
        "telemetry": telem,
        "log": log,
    }

    # Decode and encode throughput come from the stage probe: frames divided by
    # the wall clock that stage actually spent, which is what "decode FPS" means
    # for a pipeline that overlaps decode with everything else.
    for stage, key in (("decode", "decode_fps"), ("encode", "encode_fps")):
        st = stages.get(stage)
        if st and st["total_s"] > 0:
            result[key] = round(st["calls"] / st["total_s"], 2)
    ft = stages.get("frame_total")
    if ft:
        result["mean_frame_latency_ms"] = round(ft["ms_per_call"], 2)

    js = os.path.join(args.out, args.tag + ".json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("  returncode        %d" % rc)
    print("  frames            %s" % frames)
    print("  wall clock        %.2f s" % elapsed)
    if run.get("fps"):
        print("  processing FPS    %s" % run["fps"])
    for key in ("decode_fps", "encode_fps", "mean_frame_latency_ms"):
        if key in result:
            print("  %-17s %s" % (key, result[key]))
    for key in ("peak_rss_gb", "mean_rss_gb", "peak_gpu_memory_mb",
                "mean_gpu_memory_mb", "peak_gpu_util_pct", "mean_gpu_util_pct",
                "peak_cpu_pct", "mean_cpu_pct", "peak_cpu_p_pct",
                "peak_cpu_e_pct", "peak_cpu_frequency_mhz",
                "mean_cpu_frequency_mhz", "peak_cpu_temperature_c",
                "mean_cpu_temperature_c", "peak_power_w", "peak_ram_used_gb"):
        if key in telem:
            print("  %-17s %s" % (key, telem[key]))
    print("  cpu topology      %s" % telem.get("cpu_topology", {}).get("source"))
    if stages:
        print("\n  per-stage (ROOP_PROFILE, thread time summed across workers):")
        for name, st in sorted(stages.items(), key=lambda kv: -kv[1]["total_s"]):
            print("    %-14s %8.2fs %6.1f%% %8d calls %8.2f ms/call"
                  % (name, st["total_s"], st["share_pct"], st["calls"],
                     st["ms_per_call"]))
    detail = result.get("detailed_stage_profile") or {}
    canonical = detail.get("canonical_stages", {}) if isinstance(detail, dict) else {}
    if canonical:
        print("\n  per-stage (Phase 14 detailed profile; CPU/GPU/sync/allocator):")
        for name, st in sorted(canonical.items(),
                               key=lambda kv: -float(kv[1].get("cpu_ms_total", 0.0))):
            print("    %-20s cpu=%8.2fms gpu_event=%8.2fms sync=%8.2fms "
                  "alloc_peak=%8sMB steady=%8sMB calls=%6d %s"
                  % (name, float(st.get("cpu_ms_total", 0.0)),
                     float(st.get("gpu_event_ms_total", 0.0)),
                     float(st.get("sync_ms_total", 0.0)),
                     st.get("alloc_peak_mb", "n/a"),
                     st.get("steady_state_allocated_mb", "n/a"),
                     int(st.get("calls", 0)), st.get("status", "unknown")))
    print("\n  wrote %s" % js)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
