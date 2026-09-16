"""Live monitor for the running roop-ultimate job.

Polls /api/progress and prints a compact one-line status plus any NEW log
lines. Emits JCODE_PROGRESS lines so the agent harness can track it.
"""
import json
import sys
import time
import urllib.request

# The Pinokio/Windows console is cp1252; job log lines carry emoji. Never let
# the monitor die on an un-encodable glyph.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 42004
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=8) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


seen = set()
last_desc = None
last_pct = -1.0
while True:
    try:
        p = get("/api/progress")
    except Exception as e:  # server busy / restarting
        print(f"[monitor] poll failed: {e}", flush=True)
        time.sleep(3)
        continue

    for entry in p.get("log") or []:
        key = (entry.get("seq"), entry.get("t"), entry.get("msg"))
        if key in seen:
            continue
        seen.add(key)
        print(
            f"[{entry.get('t')}] {entry.get('category','')}/{entry.get('level','')}: {entry.get('msg')}",
            flush=True,
        )

    pct = float(p.get("progress") or 0.0) * 100.0
    desc = p.get("desc") or ""
    if desc != last_desc or abs(pct - last_pct) >= 0.5:
        eta = p.get("eta_s")
        eta_s = f"{float(eta):.0f}s" if isinstance(eta, (int, float)) else str(eta)
        print(
            f"JCODE_PROGRESS {json.dumps({'percent': round(pct, 2), 'message': f'{desc} (eta {eta_s})'})}",
            flush=True,
        )
        last_desc, last_pct = desc, pct

    if p.get("error"):
        print(f"[monitor] ERROR: {p['error']}", flush=True)
    out = p.get("output") or {}
    if out.get("path"):
        print(f"[monitor] OUTPUT ready: {out.get('kind')} {out.get('path')}", flush=True)

    if not p.get("processing"):
        print(f"[monitor] idle (desc={desc})", flush=True)

    time.sleep(2)
