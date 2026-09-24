"""Live check of the VRAM governor and the auto-tune, on a real backend.

    app/env/Scripts/python.exe tests/autotune_live_smoke.py [--frames 600] [--no-tune]

1. Starts the React backend exactly as start_react.js does, seeds a library
   faceset and the frame-transport clip (b1.mp4 unless ROOP_AB_CLIP is set).
2. Renders --frames frames through /api/swap ("All faces"), then reads the
   backend log for the governor's admission line AND its measured-peak line --
   the proof that `admit()` and `finish()` both ran on the production path, and
   by how much the prior was off on this machine.
3. Runs the full auto-tune on that render and checks the result: complete,
   every arm either honest or disqualified WITH a reason, a baseline that
   swapped faces, confirmation at the requested length.

config.yaml is copied first and put back afterwards: the tuner saves its
winner by design, and verifying it must not rewrite the operator's settings.
Prints the tuner's latest fps every ~3 minutes while it runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from frame_transport_ab import _clip, seed, start_backend, stop  # noqa: E402
from browser_driver import free_port, wait_for_http  # noqa: E402

OUT = os.path.join(APP, "output", "autotune_live")


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode() or "{}")


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.loads(r.read().decode() or "{}")


async def main(frames, tune):
    os.makedirs(OUT, exist_ok=True)
    report = []

    def ok(name, cond, detail=""):
        report.append((name, bool(cond), detail))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)

    cfg_path = os.path.join(APP, "config.yaml")
    backup = os.path.join(OUT, "config.yaml.bak")
    shutil.copyfile(cfg_path, backup)
    port = free_port()
    log_path = os.path.join(OUT, "backend.log")
    backend, blog = start_backend(port, log_path)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for_http(base + "/api/meta", timeout=900)
        await seed(base, _clip())
        _post(base, "/api/target/set_frame", {"which": "end", "frame": frames})
        _post(base, "/api/target/set_frame", {"which": "start", "frame": 0})

        t0 = time.time()
        answer = _post(base, "/api/swap", {"target_index": 0, "detection": "All faces"})
        ok("render started", answer.get("status") not in (None, "error"), json.dumps(answer)[:200])
        while True:
            time.sleep(2)
            prog = _get(base, "/api/progress")
            if not prog.get("processing"):
                break
        ok("render finished without error", not prog.get("error"), prog.get("error") or "")
        print(f"  render wall {time.time() - t0:.0f}s", flush=True)

        text = open(log_path, encoding="utf-8", errors="replace").read()
        admit = re.findall(r"\[VramGovernor\] (\d+)MB free of (\d+)MB, job estimate (\d+)MB.*", text)
        peak = re.findall(r"\[VramGovernor\] measured peak (\d+)MB vs estimate (\d+)MB", text)
        steps = re.findall(r"\[VramGovernor\] step-down: (.*)", text)
        ok("governor admitted the render", bool(admit), str(admit[-1:]))
        ok("governor measured the peak", bool(peak), str(peak[-1:]))
        if peak:
            measured, estimate = map(int, peak[-1])
            print(f"  prior/actual: estimate {estimate}MB, measured {measured}MB "
                  f"({measured / max(1, estimate):.2f}x)", flush=True)
        ok("no step-down on a healthy card", not steps, "; ".join(steps))
        timing = re.findall(r"took ([\d.]+) secs, ([\d.]+) frames/s", text)
        print(f"  render rate: {timing[-1:]}", flush=True)

        if not tune:
            return 0 if all(r[1] for r in report) else 1

        status = _get(base, "/api/autotune")
        ok("auto-tune ready after a render", status.get("ready"), str(status.get("reason")))
        started = _post(base, "/api/autotune/start", {})
        ok("auto-tune started", started.get("status") == "started", json.dumps(started)[:300])
        blocked = None
        try:
            _post(base, "/api/swap", {"target_index": 0, "detection": "All faces"})
        except urllib.error.HTTPError as err:
            blocked = err.code
        ok("/api/swap refused while tuning", blocked == 409, str(blocked))

        last_report = 0.0
        while True:
            time.sleep(5)
            st = _get(base, "/api/autotune")
            prog = st.get("progress") or {}
            if time.time() - last_report >= 180:
                last_report = time.time()
                print(f"  [{time.strftime('%H:%M:%S')}] {prog.get('phase')}: "
                      f"{prog.get('arms_done')}/{prog.get('arms_total')} arms, "
                      f"last {prog.get('last_fps')} fps -- {prog.get('status')}", flush=True)
            if not prog.get("running"):
                break
        result = st.get("result") or {}
        with open(os.path.join(OUT, "autotune_result.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        ok("auto-tune complete", result.get("status") == "complete",
           result.get("error") or result.get("status"))
        runs = result.get("runs") or []
        dishonest = [r for r in runs if not r.get("honest")]
        explained = {row["arm"] for row in result.get("screen", []) if row.get("excluded")}
        ok("every dishonest arm is excluded with a reason",
           all(r["arm"] in explained or r["phase"] == "confirm" for r in dishonest),
           f"{len(dishonest)} of {len(runs)} runs not as labelled")
        base_row = next((r for r in result.get("screen", []) if r["arm"] == result.get("baseline")), {})
        ok("baseline swapped faces", (base_row.get("min_swaps") or 0) > 0, str(base_row))
        ok("confirmation length", result.get("confirm_frames") == min(frames, 600),
           str(result.get("confirm_frames")))
        enc = result.get("encoder")
        if enc:
            ok("every NVENC preset measured", all(r.get("fps") for r in enc.get("rows", [])),
               json.dumps(enc.get("rows"))[:300])
        print("\n  SCREEN")
        for row in result.get("screen", []):
            print(f"    {row['arm']:<16} {row['mean_fps']:>8} fps  swaps {row['min_swaps']:<6} "
                  f"{row.get('excluded') or ''}")
        for c in result.get("confirm", []):
            print(f"  CONFIRM {c['arm']}: {c.get('improvement_pct')}% (noise {c.get('noise_pct')}%) "
                  f"-- {c['reason']}")
        print(f"  WINNER {result.get('winner')}  APPLIED {result.get('applied')}")
        if enc:
            print(f"  NVENC picked {enc.get('picked')} (need {enc.get('need_fps')} fps): "
                  + ", ".join(f"{r['preset']}={r.get('fps')}fps/{r.get('mbps')}Mbps"
                              for r in enc.get("rows", [])))
    finally:
        stop(backend)
        blog.close()
        shutil.copyfile(backup, cfg_path)
        for name in ("autotune_result.json",):
            path = os.path.join(APP, name)
            if os.path.exists(path):
                shutil.copyfile(path, os.path.join(OUT, "app_" + name))
    failed = [r for r in report if not r[1]]
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as fh:
        json.dump([{"check": n, "ok": o, "detail": d} for n, o, d in report], fh, indent=2)
    print(f"\n{len(report) - len(failed)}/{len(report)} checks passed", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.frames, not args.no_tune)))
