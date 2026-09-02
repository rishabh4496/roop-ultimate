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


async def check_v2(page, base, shots, report):
    await page.goto(f"{base}/#/home", settle=2.0)

    report.ok("V2 shell renders",
              await page.wait_for("document.querySelector('.v2-sidebar') && "
                                  "document.querySelector('.v2-topbar')", timeout=30),
              "sidebar and topbar mounted")
    report.ok("V2 mounts without an empty root",
              bool(await page.evaluate(
                  "const r = document.getElementById('root');"
                  " return !!r && r.children.length > 0;")),
              "React tree present")

    # The browser reaching the backend through the dev server proves the whole
    # launcher wiring, not just that two processes are alive.
    meta = await page.evaluate(
        "const r = await fetch('/api/meta');"
        " return { status: r.status, keys: Object.keys(await r.json()).length };")
    report.ok("V2 reaches the backend through the dev-server proxy",
              meta and meta.get("status") == 200,
              f"/api/meta HTTP {meta.get('status')}, {meta.get('keys')} keys")

    for route_id, label in V2_ROUTES:
        await page.evaluate(f"window.location.hash = '#/{route_id}'; return true;")
        await asyncio.sleep(1.2)
        heading = await page.evaluate(
            "const h = document.querySelector('.v2-topbar h1');"
            " return h ? h.textContent.trim() : null;")
        report.ok(f"V2 route '{route_id}' renders", heading == label,
                  f"heading {heading!r}")
        await page.screenshot(os.path.join(shots, f"v2_route_{route_id}.png"))

    # Navigation must work by CLICKING, not only by rewriting the hash.
    await page.evaluate("window.location.hash = '#/home'; return true;")
    await asyncio.sleep(0.8)
    clicked = await page.click_text(".v2-nav button", "Create")
    await asyncio.sleep(1.0)
    heading = await page.evaluate(
        "const h = document.querySelector('.v2-topbar h1'); return h ? h.textContent.trim() : null;")
    report.ok("V2 sidebar navigation responds to a real click",
              clicked and heading == "Create", f"clicked={clicked}, heading={heading!r}")

    # Themes: applied AND persisted, and the applied value must actually change
    # the rendered variable rather than only the dataset attribute.
    await page.evaluate("window.location.hash = '#/settings'; return true;")
    await asyncio.sleep(1.2)
    applied, persisted, distinct = [], [], set()
    for theme in THEMES:
        got = await page.evaluate(
            f"const target = {json.dumps(theme)};"
            " const nodes = Array.from(document.querySelectorAll('button, [role=\"button\"], option'));"
            " const hit = nodes.find((n) => (n.dataset && n.dataset.theme === target)"
            "   || (n.value === target)"
            "   || (n.textContent || '').trim().toLowerCase() === target);"
            " if (hit) { if (hit.tagName === 'OPTION') { const sel = hit.closest('select');"
            "   sel.value = target; sel.dispatchEvent(new Event('change', { bubbles: true })); }"
            "   else { hit.click(); } }"
            " return !!hit;")
        await asyncio.sleep(0.5)
        state = await page.evaluate(
            "return { theme: document.documentElement.dataset.theme,"
            " bg: getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(),"
            " stored: window.localStorage.getItem('roop.ui2.theme') };")
        if got and state.get("theme") == theme:
            applied.append(theme)
            distinct.add(state.get("bg"))
            if state.get("stored") == theme:
                persisted.append(theme)
            await page.screenshot(os.path.join(shots, f"v2_theme_{theme}.png"))
    report.ok("V2 applies every declared theme", len(applied) == len(THEMES),
              f"applied {len(applied)}/{len(THEMES)}: {','.join(applied)}")
    report.ok("V2 themes change the rendered palette",
              len(distinct) == len(applied) and len(applied) > 1,
              f"{len(distinct)} distinct --bg values across {len(applied)} themes")
    report.ok("V2 persists the selected theme", len(persisted) == len(applied),
              f"persisted {len(persisted)}/{len(applied)} to localStorage")

    # Control inventory: every interactive control must be reachable, and no
    # enabled control may throw when activated.
    inventory = await page.evaluate(
        "const out = {};"
        " for (const id of ['home', 'create', 'settings']) {"
        "   window.location.hash = '#/' + id;"
        "   await new Promise((r) => setTimeout(r, 700));"
        "   const nodes = Array.from(document.querySelectorAll("
        "     'button, input, select, textarea, [role=\"button\"], [role=\"tab\"]'));"
        "   out[id] = { total: nodes.length,"
        "     disabled: nodes.filter((n) => n.disabled || n.getAttribute('aria-disabled') === 'true').length,"
        "     unlabelled: nodes.filter((n) => !(n.textContent || '').trim()"
        "       && !n.getAttribute('aria-label') && !n.getAttribute('title')"
        "       && !n.getAttribute('placeholder') && n.type !== 'file' && n.type !== 'range'"
        "       && n.type !== 'checkbox' && n.type !== 'color').length };"
        " }"
        " return out;")
    total = sum(v["total"] for v in inventory.values())
    unlabelled = sum(v["unlabelled"] for v in inventory.values())
    report.ok("V2 exposes interactive controls on every route",
              all(v["total"] > 0 for v in inventory.values()),
              json.dumps(inventory))
    report.ok("V2 controls carry an accessible name", unlabelled == 0,
              f"{unlabelled} unlabelled of {total}")

    before = len(page.errors)
    activated = await page.evaluate(
        "window.location.hash = '#/settings';"
        " await new Promise((r) => setTimeout(r, 700));"
        " const nodes = Array.from(document.querySelectorAll('button'))"
        "   .filter((n) => !n.disabled && n.getAttribute('aria-disabled') !== 'true');"
        " let n = 0;"
        " for (const el of nodes.slice(0, 25)) {"
        "   const t = (el.textContent || '').toLowerCase();"
        "   if (t.includes('delete') || t.includes('remove') || t.includes('reset')"
        "       || t.includes('clean') || t.includes('update')) continue;"
        "   try { el.click(); n += 1; } catch (e) { /* recorded via console */ }"
        "   await new Promise((r) => setTimeout(r, 120));"
        " }"
        " return n;")
    await asyncio.sleep(1.0)
    report.ok("V2 enabled controls activate without throwing",
              len(page.errors) == before,
              f"activated {activated} non-destructive controls; "
              f"{len(page.errors) - before} new page error(s)")

    # Telemetry honesty: the contract is that missing values are stated, not
    # invented.  A number where the backend has nothing is the failure mode.
    await page.evaluate("window.location.hash = '#/settings'; return true;")
    await asyncio.sleep(1.2)
    body = await page.evaluate("return document.body.innerText;")
    report.add("V2 states unavailable runtime values rather than fabricating",
               "PASS" if any(token in body for token in
                             ("UNKNOWN", "NOT AVAILABLE", "NOT APPLICABLE"))
               else "PARTIALLY VERIFIED",
               "explicit sentinel present in the rendered Settings screen"
               if any(token in body for token in ("UNKNOWN", "NOT AVAILABLE",
                                                  "NOT APPLICABLE"))
               else "no sentinel rendered while idle; not proof of fabrication")

    # Responsive: a narrow viewport must not overflow horizontally.
    for width, height in ((1440, 900), (1024, 768), (900, 800), (420, 900)):
        await page.set_viewport(width, height)
        await asyncio.sleep(0.6)
        overflow = await page.evaluate(
            "return document.documentElement.scrollWidth - document.documentElement.clientWidth;")
        report.ok(f"V2 does not overflow horizontally at {width}x{height}",
                  overflow is not None and overflow <= 2, f"overflow {overflow}px")
        await page.screenshot(os.path.join(shots, f"v2_viewport_{width}x{height}.png"))
    await page.set_viewport(1440, 900)

    report.ok("V2 produced no uncaught page errors", not page.errors,
              "; ".join(e["text"][:160] for e in page.errors[:4]) or "none")


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

    ui_dir = "react-ui-v2" if args.ui == "v2" else "react-ui"
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
            if args.ui == "v2":
                await check_v2(page, base, shots, report)
            else:
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
    parser.add_argument("--ui", choices=("v1", "v2"), default="v2")
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
