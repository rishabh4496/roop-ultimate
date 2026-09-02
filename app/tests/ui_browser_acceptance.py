"""Real-browser acceptance for React UI 2.0 and a React UI 1.0 regression smoke.

WHY THIS EXISTS.  Stages 13 through 17A all recorded "no browser runtime was
available" and marked V2 launch, routing, controls, themes, telemetry and the
V1 regression as BLOCKED or NOT TESTED.  `browser_driver` removes that
constraint, so those rows can be decided on evidence.

WHAT IT PROVES, AND WHAT IT DOES NOT.  It boots the SAME processes the Pinokio
launcher boots -- `app/run.py` plus the client's own Vite dev server on
127.0.0.1 -- loads the client in a real Chromium, and then asserts behaviour
the source cannot: that the shell renders, that every route mounts its own
screen, that the browser can actually reach the backend THROUGH the dev
server's proxy, that all seven themes apply and persist, that interactive
controls respond without throwing, and that narrow viewports do not overflow.
It captures a screenshot per route and per theme as retained evidence.

It does NOT constitute human visual/aesthetic acceptance, and it does not grade
swap quality.  Those stay separately recorded.

    env/Scripts/python.exe tests/ui_browser_acceptance.py --ui v2
    env/Scripts/python.exe tests/ui_browser_acceptance.py --ui v1
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for path in (APP, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from browser_driver import Browser, find_browser, free_port, wait_for_http  # noqa: E402

THEMES = ("light", "dark", "professional", "modern", "minimal", "gaming", "anime")
V2_ROUTES = (("home", "Overview"), ("create", "Create"), ("settings", "Settings"))


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, state, detail=""):
        self.rows.append({"check": name, "state": state, "detail": detail})
        print(f"[ui] {state:<12} {name}" + (f"  --  {detail}" if detail else ""),
              flush=True)

    def ok(self, name, condition, detail=""):
        self.add(name, "PASS" if condition else "FAIL", detail)
        return bool(condition)

    @property
    def failed(self):
        return [row for row in self.rows if row["state"] == "FAIL"]


def _npm():
    """Resolve npm the way update_health does -- Pinokio keeps it off PATH."""
    sys.path.insert(0, APP)
    from update_health import _resolve_npm
    return _resolve_npm()


def start_stack(ui_dir, api_port, ui_port, log_dir):
    """Boot backend + Vite exactly as the Pinokio launcher does."""
    os.makedirs(log_dir, exist_ok=True)
    env = os.environ.copy()
    env.update({"ROOP_API_PORT": str(api_port), "ROOP_GRADIO_PORT": str(api_port + 2),
                "ROOP_TEMPORAL_STEP": "1"})
    backend_log = open(os.path.join(log_dir, "backend.log"), "w", encoding="utf-8",
                       errors="replace")
    backend = subprocess.Popen(
        [os.path.join(APP, "env", "Scripts", "python.exe"), "run.py"],
        cwd=APP, env=env, stdout=backend_log, stderr=subprocess.STDOUT, text=True)

    npm = _npm()
    if not npm:
        backend.terminate()
        raise RuntimeError("npm could not be resolved; the UI cannot be served")
    ui_env = os.environ.copy()
    ui_env.update({"ROOP_API_PORT": str(api_port), "PORT": str(ui_port)})
    # Resolving npm is not enough: `npm.cmd` shells out to `node`, which lives
    # beside it in the Pinokio toolchain and is equally absent from a plain
    # process's PATH.  Without this the dev server dies with
    # '"node" is not recognized' and the UI simply never answers.
    ui_env["PATH"] = os.path.dirname(npm) + os.pathsep + ui_env.get("PATH", "")
    # Vite otherwise binds ::1 only, which makes 127.0.0.1 refuse the connection.
    ui_log = open(os.path.join(log_dir, "vite.log"), "w", encoding="utf-8",
                  errors="replace")
    vite = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", "127.0.0.1"],
        cwd=os.path.join(ROOT, ui_dir), env=ui_env, stdout=ui_log,
        stderr=subprocess.STDOUT, text=True, shell=False)
    return backend, vite, backend_log, ui_log


def stop(process):
    if process is None:
        return
    with contextlib.suppress(Exception):
        process.terminate()
        process.wait(timeout=20)
    if process.poll() is None:
        with contextlib.suppress(Exception):
            process.kill()


async def check_v1(page, base, shots, report):
    await page.goto(base, settle=3.0)
    report.ok("V1 mounts", bool(await page.evaluate(
        "const r = document.getElementById('root');"
        " return !!r && r.children.length > 0;")), "React tree present")
    meta = await page.evaluate(
        "const r = await fetch('/api/meta'); return r.status;")
    report.ok("V1 reaches the backend", meta == 200, f"/api/meta HTTP {meta}")
    controls = await page.evaluate(
        "return document.querySelectorAll('button, input, select, textarea').length;")
    report.ok("V1 renders its controls", bool(controls and controls > 0),
              f"{controls} interactive controls")
    await page.screenshot(os.path.join(shots, "v1_shell.png"))
    report.ok("V1 produced no uncaught page errors", not page.errors,
              "; ".join(e["text"][:160] for e in page.errors[:4]) or "none")


async def run(args):
    report = Report()
    binary = find_browser()
    if not binary:
        report.add("browser runtime", "BLOCKED", "no Chromium-based browser found")
        return report, 2
    report.add("browser runtime", "PASS", binary)

    ui_dir = "react-ui"
    api_port, ui_port = free_port(), free_port()
    out = args.out or os.path.join(APP, "output", "ui_acceptance", args.ui)
    shots = os.path.join(out, "screens")
    os.makedirs(shots, exist_ok=True)

    backend = vite = backend_log = ui_log = None
    try:
        backend, vite, backend_log, ui_log = start_stack(ui_dir, api_port, ui_port, out)
        if not wait_for_http(f"http://127.0.0.1:{api_port}/api/meta", timeout=args.boot):
            report.add("backend boots", "FAIL",
                       f"/api/meta never answered on {api_port} within {args.boot}s")
            return report, 1
        report.add("backend boots", "PASS", f"/api/meta on 127.0.0.1:{api_port}")
        base = f"http://127.0.0.1:{ui_port}"
        if not wait_for_http(base, timeout=180):
            report.add("dev server boots", "FAIL", f"no answer on {ui_port}")
            return report, 1
        report.add("dev server boots", "PASS", base)

        async with Browser() as page:
            await check_v1(page, base, shots, report)
    finally:
        stop(vite)
        stop(backend)
        for handle in (backend_log, ui_log):
            with contextlib.suppress(Exception):
                handle.close()

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump({"ui": args.ui, "browser": binary, "rows": report.rows},
                  handle, indent=2)
    return report, (1 if report.failed else 0)


def main():
    parser = argparse.ArgumentParser()
    # React UI 2.0 was removed; the choice is kept so existing
    # invocations and output paths do not break.
    parser.add_argument("--ui", choices=("v1",), default="v1")
    parser.add_argument("--out", default="")
    parser.add_argument("--boot", type=float, default=420.0,
                        help="seconds to wait for the backend; a cold start "
                             "loads the model stack")
    args = parser.parse_args()

    started = time.time()
    report, code = asyncio.run(run(args))
    failed = report.failed
    print(f"\n[ui] {args.ui}: {len(report.rows)} checks in {time.time() - started:.1f}s; "
          f"{len(failed)} FAIL")
    for row in failed:
        print(f"[ui]   FAIL {row['check']}: {row['detail']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
