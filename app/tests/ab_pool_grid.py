"""Pool x swap-batch grid: which smaller configs free VRAM at no fps cost?

    env/Scripts/python.exe tests/ab_pool_grid.py plan  [--detector-values 1,2]
    env/Scripts/python.exe tests/ab_pool_grid.py inert-check          # does ROOP_DETECTOR_POOL change anything?
    env/Scripts/python.exe tests/ab_pool_grid.py run   [--detector-values 2] [--out DIR]   # resumable
    env/Scripts/python.exe tests/ab_pool_grid.py report [--out DIR]

GRID. ROOP_TRT_POOL {1,2} x ROOP_DETMASK_POOL {1,2} x ROOP_DETECTOR_POOL {1,2} x ROOP_BATCH_SWAP_MAX {1,4,8}, plus the
SHIPPED reference (pools 2/2/2, batch unset = auto) so "no cost" has something to be compared to. d4 for 1000 frames; s7
for ALL of its 600 (the file has no more). Per clip: one discarded warm-up render (pays the engine builds, fixes the
stabilizer chunk budget and the capture seed frame, both PINNED for every measured arm), then the config list forward and
then REVERSED (ABBA over the whole grid: every config is measured once early and once late, mean position equal).

WHAT IS RECORDED PER RENDER. frame-loop fps and faces/s from the child's own `[Pipeline] done` line, pre-pass fps, faces_seen
and the swap audit (a config that is faster because it finds fewer faces has not got faster), the `[Stabilize] parallel:` /
`[StabGeometry]` stab_width lines, and the lines that prove the knobs executed (`[FaceAnalysis] pool of N`, `[BatchSwap]`,
`[SessionPool]`, the `[Session]` instance counts). NVML (pynvml, 10 Hz, device 0, the child's lifetime): peak device-wide
used VRAM and its share of the card, mean and p95 GPU utilisation over the whole run and over the frame loop only.

REJECTED. Peak VRAM > 92% of the card in any render; run-to-run spread > 5% (|forward - reverse| / mean of the frame-loop fps)
on any clip; a non-zero exit, failed frames, or a killed (timed-out) render. VRAM here is DEVICE-WIDE (the desktop's ~1.2 GB is
in it): that is what the driver's paging cliff is measured against, and per-process VRAM is not available under WDDM.

GUARD. The run refuses to start while the NVIDIA "CUDA - Sysmem Fallback Policy" still lets an allocation spill past VRAM
(tested in a child on the same interpreter): otherwise a config that overflows would silently page and read as merely slow.
"""
import argparse
import datetime
import itertools
import json
import os
import re
import statistics as st
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_snapshot as bs                                   # noqa: E402
import baseline_controlled as bc                                 # noqa: E402

SPECS = dict(bs.CLIPS)
SPECS["s7"] = {"rel": "single/s7.mp4", "sources": "harjot", "window": (0, 600),
               "capture": None, "capture_face": None}
WINDOWS = {"d4": (0, 1000), "s7": (0, 600)}          # s7.mp4 has 600 frames: all of them
REJECT_VRAM_PCT = 92.0
REJECT_SPREAD_PCT = 5.0
SPILL_MARGIN_MIB = 512        # shared GPU memory above the shipped warm-up's own peak = VRAM overflowing into system RAM
RUN_TIMEOUT_S = 30 * 60
SHIPPED = (2, 2, 2, None)                             # trt, detmask, detector, batch (None = unset = auto)

_CURRENT = {"marks": {}, "nvml": None}


def date_default():
    return datetime.date.today().isoformat()


# ── NVML ────────────────────────────────────────────────────────────────────────────────────────
class Nvml:
    """10 Hz poller of device-wide used memory and GPU utilisation for the lifetime of one render."""

    def __init__(self, interval=0.1, index=0):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(self.h).total
        self.interval = interval
        self.samples = []                              # (t, used_bytes, util_pct)
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                mem = self.nv.nvmlDeviceGetMemoryInfo(self.h)
                util = self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu
                self.samples.append((time.time(), mem.used, util))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="nvml", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self, loop=None):
        s = self.samples
        if not s:
            return {"n_samples": 0}
        used = [u for _, u, _ in s]
        util = [g for _, _, g in s]
        out = {"n_samples": len(s), "total_mib": round(self.total / 2 ** 20, 1),
               "idle_before_mib": round(used[0] / 2 ** 20, 1),
               "peak_used_mib": round(max(used) / 2 ** 20, 1),
               "peak_pct": round(100.0 * max(used) / self.total, 2),
               "util_mean_run": round(st.mean(util), 1)}
        if loop and loop[0] and loop[1] and loop[1] > loop[0]:
            lu = [g for t, _, g in s if loop[0] <= t <= loop[1]]
            lm = [u for t, u, _ in s if loop[0] <= t <= loop[1]]
            if lu:
                out.update({"util_mean_loop": round(st.mean(lu), 1),
                            "util_p95_loop": round(sorted(lu)[int(0.95 * (len(lu) - 1))], 1),
                            "loop_peak_used_mib": round(max(lm) / 2 ** 20, 1),
                            "loop_seconds": round(loop[1] - loop[0], 1)})
        return out


_PS_LOOP = (
    "while($true){ try{ $s=(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples;"
    " foreach($x in $s){ if($x.CookedValue -gt 0 -and $x.InstanceName -match '^pid_(\\d+)_'){"
    " '{0} {1}' -f $matches[1],[int]($x.CookedValue/1MB) } }; 'TICK' } catch { 'TICK' }; Start-Sleep -Milliseconds 600 }")


class SharedGpu:
    """Per-process 'GPU Process Memory\\Shared Usage' of the render's process tree (Windows perf counter, 1-2 Hz).

    This is the spill detector that replaces the driver policy: when VRAM overflows under WDDM the excess is placed in
    shared (system) GPU memory, and this counter is where it shows up. It cannot PREVENT a spill; it measures one.
    """

    def __init__(self):
        self.peak = None
        self.first = None
        self.series = []
        self._proc = None
        self._thread = None

    def start(self):
        try:
            self._proc = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_LOOP],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                          creationflags=0x08000000)
        except Exception:
            self._proc = None
            return
        self._thread = threading.Thread(target=self._read, name="shared_gpu", daemon=True)
        self._thread.start()

    def _read(self):
        import psutil
        acc = {}
        for line in self._proc.stdout:
            line = line.strip()
            if line == "TICK":
                try:
                    tree = {psutil.Process().pid} | {c.pid for c in psutil.Process().children(recursive=True)}
                except Exception:
                    tree = set()
                total = sum(v for pid, v in acc.items() if pid in tree)
                self.series.append((time.time(), total))
                if self.first is None:
                    self.first = total
                self.peak = total if self.peak is None else max(self.peak, total)
                acc = {}
            else:
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit():
                    acc[int(parts[0])] = acc.get(int(parts[0]), 0) + int(parts[1])

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self):
        return {"shared_peak_mib": self.peak, "shared_first_mib": self.first, "shared_n": len(self.series)}


def _kill_children():
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
    except Exception:
        pass


def _install_sampler_wrapper():
    """Wrap telemetry.run_sampled (what baseline_snapshot.run_one calls): NVML for exactly the child's lifetime, the
    frame-loop window from the child's own streamed `[Pipeline]` lines, and a watchdog that kills a hung render."""
    real = bs.tel.run_sampled

    def wrapped(cmd, env=None, cwd=None, on_line=None, device_id=0):
        marks = _CURRENT["marks"] = {}
        nvml = _CURRENT["nvml"] = Nvml()
        shared = _CURRENT["shared"] = SharedGpu()

        def on2(line):
            now = time.time()
            if line.startswith("[Pipeline] frames") and "loop_start" not in marks:
                marks["loop_start"] = now
            elif line.startswith("[Pipeline] done"):
                marks["loop_end"] = now
            if on_line:
                on_line(line)

        timer = threading.Timer(RUN_TIMEOUT_S, lambda: (marks.update(timeout=True), _kill_children()))
        timer.daemon = True
        timer.start()
        nvml.start()
        shared.start()
        try:
            return real(cmd, env=env, cwd=cwd, on_line=on2, device_id=device_id)
        finally:
            nvml.stop()
            shared.stop()
            timer.cancel()
    bs.tel.run_sampled = wrapped


# ── plan ────────────────────────────────────────────────────────────────────────────────────────
def label_of(cfg):
    t, m, d, b = cfg
    return "t%dm%dd%db%s" % (t, m, d, "auto" if b is None else b)


def configs(detector_values):
    grid = [(t, m, d, b) for t, m, d, b in itertools.product((1, 2), (1, 2), detector_values, (1, 4, 8))]
    shipped = SHIPPED if SHIPPED[2] in detector_values else (2, 2, detector_values[-1], None)
    return [shipped] + grid


CONTROL_EVERY = 6          # a shipped-config control after every this many measured renders


def plan(detector_values, clips, control_every=CONTROL_EVERY):
    """Per clip: the config list forward, then reversed (ABBA over the grid), with the SHIPPED config re-run as a control
    every `control_every` renders. The grid itself contains the shipped config once per pass; every shipped render
    (listed or inserted) is a control. Each item carries a global `idx` so the report can bracket a render between the
    controls before and after it."""
    seq = []
    for clip in clips:
        cfgs = configs(detector_values)
        shipped = cfgs[0]
        for pass_name, order in (("F", cfgs), ("R", cfgs[::-1])):
            since, k = 0, 0
            for i, cfg in enumerate(order):
                is_ship = cfg == shipped
                seq.append({"clip": clip, "cfg": cfg, "pass": pass_name, "pos": i, "control": is_ship,
                            "label": "%s_%s_%s" % (clip, pass_name, label_of(cfg))})
                since = 0 if is_ship else since + 1
                last = i == len(order) - 1
                if since >= control_every and not last:
                    k += 1
                    seq.append({"clip": clip, "cfg": shipped, "pass": pass_name, "pos": i, "control": True,
                                "label": "%s_%s_ctl%d_%s" % (clip, pass_name, k, label_of(shipped))})
                    since = 0
    for n, s in enumerate(seq):
        s["idx"] = n
    return seq


# ── pre-flight ──────────────────────────────────────────────────────────────────────────────────
_OVERFLOW = r"""
import sys, torch
free, total = torch.cuda.mem_get_info()
want = int(free + 1.5 * 2**30)
try:
    x = torch.empty(want, dtype=torch.uint8, device='cuda'); x.fill_(1); torch.cuda.synchronize()
    print('SPILLED %.1f GiB beyond %.1f GiB free' % (want / 2**30, free / 2**30)); sys.exit(0)
except Exception as e:
    print('OOM ' + type(e).__name__); sys.exit(3)
"""


def sysmem_fallback_active():
    """True when an allocation 1.5 GiB larger than the free VRAM succeeds (the driver spills it into system RAM)."""
    p = subprocess.run([sys.executable, "-c", _OVERFLOW], capture_output=True, text=True, timeout=300)
    out = (p.stdout or "").strip().splitlines()[-1:] or [""]
    return p.returncode == 0 and out[0].startswith("SPILLED"), out[0]


# ── one render ──────────────────────────────────────────────────────────────────────────────────
POOL_LINES = re.compile(r"^\[(?:FaceAnalysis\] pool of|BatchSwap\]|SessionPool\]|RetinaFace\] pool|RetinaFaceGPU\] pool)"
                        r".*$", re.M)


def pick(rec, nvml, marks):
    text = open(rec["log"], encoding="utf-8", errors="ignore").read()
    loop = (marks.get("loop_start"), marks.get("loop_end"))
    stab = [l for l in (rec.get("stab_lines") or []) if "parallel:" in l or "stabilises" in l or "wide" in l]
    geo = (rec.get("stab_geometry") or [""])[0]
    mw = re.search(r"\bworkers=(\d+)", geo)
    mp = re.search(r"\[Stabilize\] parallel: (\d+) workers", text)
    run = rec.get("run") or {}
    out = {k: rec.get(k) for k in ("label", "returncode", "frame_loop_fps", "prepass_fps", "prepass_seconds",
                                   "faces_per_s", "faces_per_frame", "failed_frames", "detector_failed",
                                   "wall_seconds", "stabilizer_path")}
    out.update({
        "faces_seen": run.get("faces_seen"), "faces_swapped": run.get("faces_swapped"),
        "swapped_lines": [l.strip() for l in (rec.get("swapped_lines") or [])][:3],
        "stab_width": int(mw.group(1)) if mw else (int(mp.group(1)) if mp else None),
        "stab_lines": [l[:230] for l in stab][:4], "stab_geometry": geo[:400],
        "pool_lines": [m.group(0)[:200] for m in POOL_LINES.finditer(text)][:12],
        "session_instances": sorted(set(re.findall(r"^\[Session\] (\S+) .*instances=(\d+)", text, re.M))),
        "output_sha256": (rec.get("output") or {}).get("sha256"),
        "nvml": nvml.summary(loop), "timeout": bool(marks.get("timeout"))})
    sh = _CURRENT.get("shared")
    out["shared_gpu"] = sh.summary() if sh else {}
    return out


def rejection(r):
    why = []
    if r.get("returncode") not in (0, None) or r.get("timeout"):
        why.append("failed/timeout rc=%s" % r.get("returncode"))
    if r.get("failed_frames"):
        why.append("%s failed frames" % r["failed_frames"])
    if r.get("frame_loop_fps") is None:
        why.append("no fps line")
    pct = (r.get("nvml") or {}).get("peak_pct")
    if pct is not None and pct > REJECT_VRAM_PCT:
        why.append("peak VRAM %.1f%% > %d%%" % (pct, REJECT_VRAM_PCT))
    peak = (r.get("shared_gpu") or {}).get("shared_peak_mib")
    base = r.get("shared_baseline_mib")
    if peak is not None and base is not None and peak > base + SPILL_MARGIN_MIB:
        why.append("SPILL: shared GPU memory %d MiB > warm-up %d + %d MiB" % (peak, base, SPILL_MARGIN_MIB))
    return why


def base_env_for(args):
    env = dict(os.environ)
    for k in ("ROOP_TRT_POOL", "ROOP_DETMASK_POOL", "ROOP_DETECTOR_POOL", "ROOP_BATCH_SWAP_MAX",
              "ROOP_STAB_CHUNK_MB", "ROOP_TRT_BOUND"):
        env.pop(k, None)
    env["ROOP_PROFILE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("ROOP_PIPELINE_LOG_EVERY", "100")
    env["ROOP_TRT_STATIC_PROFILE"] = "1" if args.static_profile else "0"      # Q2 (inert for scrfd: see the report)
    env = bc.ensure_ffmpeg(env)
    os.environ["PATH"] = env["PATH"]
    return env


def env_for(base, cfg, chunk):
    t, m, d, b = cfg
    env = dict(base)
    env["ROOP_TRT_POOL"], env["ROOP_DETMASK_POOL"], env["ROOP_DETECTOR_POOL"] = str(t), str(m), str(d)
    if b is not None:
        env["ROOP_BATCH_SWAP_MAX"] = str(b)
    if chunk:
        env["ROOP_STAB_CHUNK_MB"] = str(chunk)
    return env


def args_for_harness(args):
    args.detector_engine = None
    args.governor = False
    return args


def render(label, clip, cfg, base, chunk, capture, cfgobj, args, out_root, threads, window):
    env = env_for(base, cfg, chunk) if cfg else dict(base)
    rec = bs.run_one(label, clip, SPECS[clip], cfgobj, args, env, out_root, threads, window=window, capture=capture)
    return pick(rec, _CURRENT["nvml"], _CURRENT["marks"]), rec


def common(args):
    from settings import Settings
    cfgobj = Settings(os.path.join(APP, "config.yaml"))
    threads = int(cfgobj.max_threads)
    out = args.out or os.path.join(APP, "output", "pool_grid_" + args.date)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    _install_sampler_wrapper()
    return cfgobj, threads, out, args_for_harness(args)


def warmup(clip, base, cfgobj, args, out, threads):
    path = os.path.join(out, "warmup_%s.json" % clip)
    if os.path.exists(path):
        return json.load(open(path))
    r, rec = render("warmup_" + clip, clip, None, base, None, None, cfgobj, args, out, threads, (0, 120))
    chunk = (rec.get("stab_ram_line") or {}).get("derived_chunk_budget_mb")
    cap = next((int(m.group(1)) for l in rec.get("capture_lines", []) for m in [re.search(r"seed frame (\d+)", l)] if m), None)
    w = {"chunk_mb": chunk, "capture": cap, "returncode": r["returncode"], "label": "warmup_" + clip,
         "shared_peak_mib": (r.get("shared_gpu") or {}).get("shared_peak_mib")}
    json.dump(w, open(path, "w"))
    print("  warm-up %s: rc %s, ROOP_STAB_CHUNK_MB pinned to %s, capture seed frame pinned to %s, shared GPU peak %s MiB"
          % (clip, r["returncode"], chunk, cap, w["shared_peak_mib"]), flush=True)
    return w


def cmd_plan(args):
    dv = [int(x) for x in args.detector_values.split(",")]
    seq = plan(dv, args.clips.split(","))
    print("%d measured renders + %d warm-ups (configs per clip: %d)" % (len(seq), len(args.clips.split(",")),
                                                                         len(configs(dv))))
    for s in seq[:6] + seq[-3:]:
        print("  ", s["label"])
    return 0


def cmd_run(args):
    if not args.allow_sysmem_fallback:
        active, line = sysmem_fallback_active()
        print("[guard] overflow test: %s" % line, flush=True)
        if active:
            print("REFUSING TO RUN: NVIDIA 'CUDA - Sysmem Fallback Policy' still lets allocations spill into system RAM "
                  "for this interpreter (%s). Set 'Prefer No Sysmem Fallback' for the process image and re-run, or pass "
                  "--allow-sysmem-fallback." % getattr(sys, "_base_executable", sys.executable))
            return 4
    cfgobj, threads, out, args = common(args)
    base = base_env_for(args)
    dv = [int(x) for x in args.detector_values.split(",")]
    seq = plan(dv, args.clips.split(","))
    print("[grid] %d renders, out %s, threads %d, static profile %s" % (len(seq), out, threads, args.static_profile),
          flush=True)
    warm = {}
    done = 0
    t0 = time.time()
    for n, s in enumerate(seq, 1):
        clip = s["clip"]
        path = os.path.join(out, "runs", s["label"] + ".json")
        if os.path.exists(path):
            done += 1
            continue
        if clip not in warm:
            warm[clip] = warmup(clip, base, cfgobj, args, out, threads)
        w = warm[clip]
        r, _ = render(s["label"], clip, s["cfg"], base, w["chunk_mb"], w["capture"], cfgobj, args, out, threads,
                      WINDOWS[clip])
        r.update({"clip": clip, "cfg": list(s["cfg"]), "cfg_label": label_of(s["cfg"]), "pass": s["pass"],
                  "pos": s["pos"], "idx": s["idx"], "control": bool(s["control"]), "chunk_mb": w["chunk_mb"],
                  "capture": w["capture"], "shared_baseline_mib": w.get("shared_peak_mib"),
                  "machine_before": None})
        json.dump(r, open(path, "w"), indent=1, default=str)
        nv = r["nvml"]
        print("[grid %d/%d, %.0f min] %s: %s fps | faces/s %s | seen %s | stab_width %s | NVML peak %s MiB (%s%%) | "
              "util loop %s%% | shared GPU peak %s MiB | %s" % (
                  n, len(seq), (time.time() - t0) / 60, s["label"], r["frame_loop_fps"], r["faces_per_s"],
                  r["faces_seen"], r["stab_width"], nv.get("peak_used_mib"), nv.get("peak_pct"),
                  nv.get("util_mean_loop"), (r.get("shared_gpu") or {}).get("shared_peak_mib"),
                  rejection(r) or "ok"), flush=True)
    print("[grid] ALL_DONE (%d resumed)" % done, flush=True)
    return 0


def cmd_inert(args):
    """Does ROOP_DETECTOR_POOL change anything with the configured detector? Two short renders, only that knob differs."""
    cfgobj, threads, out, args = common(args)
    out = os.path.join(out, "inert")
    os.makedirs(out, exist_ok=True)
    base = base_env_for(args)
    res = {}
    seen = {1: 0, 2: 0}
    for d in (1, 2, 1, 2):
        seen[d] += 1
        label = "inert_d%d_%d" % (d, seen[d])
        r, rec = render(label, "s7", (2, 2, d, 8), base, None, None, cfgobj, args, out, threads, (0, 150))
        res["d%d_%d" % (d, seen[d])] = r
        print("  %s: %s fps | peak %s MiB | pools %s | sessions %s" % (
            label, r["frame_loop_fps"], r["nvml"].get("peak_used_mib"), r["pool_lines"][:3], r["session_instances"]),
              flush=True)
    a, b = res["d1_1"], res["d2_1"]
    same_pools = a["pool_lines"] == b["pool_lines"] and a["session_instances"] == b["session_instances"]
    dv = abs(a["nvml"]["peak_used_mib"] - b["nvml"]["peak_used_mib"])
    print("INERT-CHECK ROOP_DETECTOR_POOL 1 vs 2: identical pool/session lines = %s; peak VRAM differs by %.0f MiB; "
          "faces_seen %s vs %s" % (same_pools, dv, a["faces_seen"], b["faces_seen"]))
    json.dump(res, open(os.path.join(out, "inert_check.json"), "w"), indent=1, default=str)
    return 0


# ── report ──────────────────────────────────────────────────────────────────────────────────────
def load_runs(out):
    runs = []
    d = os.path.join(out, "runs")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            try:
                runs.append(json.load(open(os.path.join(d, f))))
            except Exception:
                pass
    return runs


def _good_controls(runs, clip):
    return sorted((r for r in runs if r["clip"] == clip and r.get("control") and r.get("frame_loop_fps")
                   and not rejection(r)), key=lambda r: r["idx"])


def expected_fps(ctl, idx):
    """The shipped config's fps at plan position `idx`: linear interpolation between the controls either side of it."""
    if not ctl:
        return None
    before = [c for c in ctl if c["idx"] <= idx]
    after = [c for c in ctl if c["idx"] >= idx]
    if not before:
        return after[0]["frame_loop_fps"]
    if not after:
        return before[-1]["frame_loop_fps"]
    a, b = before[-1], after[0]
    if a["idx"] == b["idx"]:
        return a["frame_loop_fps"]
    t = (idx - a["idx"]) / float(b["idx"] - a["idx"])
    return a["frame_loop_fps"] * (1 - t) + b["frame_loop_fps"] * t


def regime_blocks(runs, clip):
    """Consecutive controls whose fps differ by more than the spread limit: the machine changed speed between them, so
    everything measured in between is not comparable with anything outside the block."""
    ctl = _good_controls(runs, clip)
    out = []
    for a, b in zip(ctl, ctl[1:]):
        m = (a["frame_loop_fps"] + b["frame_loop_fps"]) / 2.0
        drift = 100.0 * abs(a["frame_loop_fps"] - b["frame_loop_fps"]) / m
        if drift > REJECT_SPREAD_PCT:
            out.append({"clip": clip, "a": a["idx"], "b": b["idx"], "labels": [a["label"], b["label"]],
                        "fps": [a["frame_loop_fps"], b["frame_loop_fps"]], "drift_pct": round(drift, 1)})
    return out


def cmd_redo(args):
    """Move the renders inside every drifted control-to-control block (and the controls that bound it) out of runs/, so
    `run` re-measures exactly those in the current machine state."""
    out = args.out or os.path.join(APP, "output", "pool_grid_" + args.date)
    runs = load_runs(out)
    moved = []
    for clip in sorted({r["clip"] for r in runs}):
        for blk in regime_blocks(runs, clip):
            for r in runs:
                if r["clip"] == clip and blk["a"] <= r["idx"] <= blk["b"]:
                    moved.append(r["label"])
    moved = sorted(set(moved))
    dst = os.path.join(out, "runs_discarded_" + time.strftime("%H%M%S"))
    if moved:
        os.makedirs(dst, exist_ok=True)
    for lab in moved:
        os.replace(os.path.join(out, "runs", lab + ".json"), os.path.join(dst, lab + ".json"))
    print("moved %d renders to %s: %s" % (len(moved), dst if moved else "-", moved))
    return 0


def cmd_report(args):
    out = args.out or os.path.join(APP, "output", "pool_grid_" + args.date)
    runs = load_runs(out)
    clips = sorted({r["clip"] for r in runs})
    labels = sorted({r["cfg_label"] for r in runs})
    ship = label_of(SHIPPED) if label_of(SHIPPED) in labels else next((l for l in labels if l.endswith("bauto")), None)
    blocks = {c: regime_blocks(runs, c) for c in clips}
    ctl = {c: _good_controls(runs, c) for c in clips}
    for r in runs:                                   # fps relative to the shipped config measured around it
        e = expected_fps(ctl[r["clip"]], r["idx"]) if r.get("frame_loop_fps") else None
        r["rel"] = (r["frame_loop_fps"] / e) if e else None
        r["in_shift"] = any(b["a"] < r["idx"] < b["b"] for b in blocks[r["clip"]])
    rows = []
    for lab in labels:
        row = {"cfg": lab, "clips": {}, "reject": []}
        for clip in clips:
            rs = sorted((r for r in runs if r["clip"] == clip and r["cfg_label"] == lab), key=lambda r: r["idx"])
            meas = [r for r in rs if not r.get("control")] if lab != ship else rs
            fps = [r["frame_loop_fps"] for r in meas if r.get("frame_loop_fps")]
            rel = [r["rel"] for r in meas if r.get("rel")] if lab != ship else fps
            spread = (100.0 * (max(rel) - min(rel)) / st.mean(rel)) if len(rel) >= 2 else None
            nv = [r["nvml"] for r in rs if r.get("nvml")]
            c = {"n": len(meas), "fps": [round(x, 2) for x in fps],
                 "rel": [round(x, 3) for x in rel] if lab != ship else None,
                 "fps_mean": round(st.mean(fps), 3) if fps else None,
                 "rel_mean": round(st.mean(rel), 4) if (rel and lab != ship) else (1.0 if lab == ship else None),
                 "spread_pct": round(spread, 2) if spread is not None else None,
                 "peak_mib": max((x.get("peak_used_mib", 0) for x in nv), default=None),
                 "peak_pct": max((x.get("peak_pct", 0) for x in nv), default=None),
                 "util_loop": round(st.mean([x["util_mean_loop"] for x in nv if "util_mean_loop" in x]), 1)
                 if any("util_mean_loop" in x for x in nv) else None,
                 "shared_peak": max(((r.get("shared_gpu") or {}).get("shared_peak_mib") or 0 for r in rs), default=None),
                 "faces_seen": sorted({r.get("faces_seen") for r in rs}),
                 "faces_per_s": round(st.mean([r["faces_per_s"] for r in meas if r.get("faces_per_s")]), 2)
                 if any(r.get("faces_per_s") for r in meas) else None,
                 "stab_width": sorted({r.get("stab_width") for r in rs if r.get("stab_width") is not None})}
            row["clips"][clip] = c
            for r in rs:
                row["reject"] += ["%s: %s" % (clip, w) for w in rejection(r)]
            if spread is not None and spread > REJECT_SPREAD_PCT:
                row["reject"].append("%s: spread %.1f%% > %d%%" % (clip, spread, REJECT_SPREAD_PCT))
            if len(rel) < 2:
                row["reject"].append("%s: only %d valid observation(s)" % (clip, len(rel)))
            if any(r["in_shift"] for r in meas):
                row["reject"].append("%s: measured inside a machine-speed shift (see regime blocks)" % clip)
            if len({w for w in c["stab_width"]}) > 1:
                row["reject"].append("%s: stab_width varied %s (different code path)" % (clip, c["stab_width"]))
        rows.append(row)
    ref = next((r for r in rows if r["cfg"] == ship), None)
    for row in rows:
        ratios, dv = [], []
        for clip in clips:
            c, rc = row["clips"][clip], (ref or {}).get("clips", {}).get(clip)
            if c["rel_mean"]:
                ratios.append(c["rel_mean"])
            if rc and c["peak_mib"] is not None and rc["peak_mib"] is not None:
                dv.append(c["peak_mib"] - rc["peak_mib"])
        row["fps_vs_shipped"] = round(st.geometric_mean(ratios), 4) if ratios else None
        row["vram_vs_shipped_mib"] = round(st.mean(dv), 0) if dv else None
        sp = [row["clips"][c]["spread_pct"] for c in clips if row["clips"][c]["spread_pct"] is not None]
        row["noise_pct"] = round(max(sp), 2) if sp else None
    ok = [r for r in rows if not r["reject"]]
    ok.sort(key=lambda r: -(r["fps_vs_shipped"] or 0))
    hdr = ["config"] + ["%s rel fps (fwd/rev) [raw] spread" % c for c in clips] + [
        "peak MiB (%)", "util loop %", "stab_width", "shared GPU MiB", "fps vs shipped", "VRAM vs shipped", "verdict"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in sorted(rows, key=lambda r: (bool(r["reject"]), -(r["fps_vs_shipped"] or 0))):
        cells = []
        for c in clips:
            x = r["clips"][c]
            cells.append("%s [%s] %s%%" % ("/".join(str(v) for v in (x["rel"] or ["ctl"])),
                                          "/".join(str(v) for v in x["fps"]), x["spread_pct"]))
        peak = max([r["clips"][c]["peak_mib"] or 0 for c in clips])
        pct = max([r["clips"][c]["peak_pct"] or 0 for c in clips])
        util = ", ".join(str(r["clips"][c]["util_loop"]) for c in clips)
        sw = ", ".join("/".join(str(v) for v in r["clips"][c]["stab_width"]) or "?" for c in clips)
        shr = ", ".join(str(r["clips"][c]["shared_peak"]) for c in clips)
        within = (r["fps_vs_shipped"] is not None and r["noise_pct"] is not None
                  and abs(r["fps_vs_shipped"] - 1) * 100 <= max(r["noise_pct"], 1.0))
        verdict = ("REJECT: " + "; ".join(r["reject"][:3])) if r["reject"] else (
            "shipped (controls)" if r["cfg"] == ship else
            ("within noise (%.1f%%)" % r["noise_pct"] if within else
             ("faster" if r["fps_vs_shipped"] > 1 else "slower")))
        lines.append("| %s | %s | %s (%.1f%%) | %s | %s | %s | %s | %s MiB | %s |" % (
            r["cfg"] + (" (shipped)" if r["cfg"] == ship else ""), " | ".join(cells), peak, pct, util, sw, shr,
            r["fps_vs_shipped"], r["vram_vs_shipped_mib"], verdict))
    table = "\n".join(lines)
    print(table)
    print("\nmachine-speed shifts between shipped-config controls (renders inside are NOT comparable; `redo` re-runs them):")
    for c in clips:
        for b in blocks[c]:
            print("  %s: %s -> %s  (%.2f -> %.2f fps, %.1f%% apart, plan idx %d..%d)" % (
                c, b["labels"][0], b["labels"][1], b["fps"][0], b["fps"][1], b["drift_pct"], b["a"], b["b"]))
        if not blocks[c]:
            print("  %s: none (%d controls: %s fps)" % (c, len(ctl[c]), [round(x["frame_loop_fps"], 2) for x in ctl[c]]))
    payload = {"date": args.date, "clips": clips, "windows": WINDOWS, "shipped": ship, "rows": rows,
               "regime_blocks": blocks, "reject_rules": {"vram_pct": REJECT_VRAM_PCT, "spread_pct": REJECT_SPREAD_PCT,
                                                         "spill_margin_mib": SPILL_MARGIN_MIB,
                                                         "control_every": CONTROL_EVERY}}
    jp = os.path.join(REPO, "docs", "perf", "pool_grid_%s.json" % args.date)
    json.dump(payload, open(jp, "w"), indent=1, default=str)
    open(os.path.join(out, "table.md"), "w", encoding="utf-8").write(table + "\n")
    print("\nsurvivors:", [r["cfg"] for r in ok])
    print("wrote", jp)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "run", "inert-check", "report", "redo"))
    ap.add_argument("--clips", default="d4,s7")
    ap.add_argument("--detector-values", default="1,2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--date", default=date_default())
    ap.add_argument("--no-static-profile", dest="static_profile", action="store_false", default=True)
    ap.add_argument("--allow-sysmem-fallback", action="store_true")
    args = ap.parse_args()
    return {"plan": cmd_plan, "run": cmd_run, "inert-check": cmd_inert, "report": cmd_report,
            "redo": cmd_redo}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
