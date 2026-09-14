"""Prove the log/parts freeze if the telemetry socket replaces the poll outright.

The telemetry frame is a deliberate SUBSET of /api/progress: it carries the
numbers that change per frame, not the 250-line rolling log, the parts snapshot
or the nested `runtime` block -- carrying those is precisely what made the poll
17.4 KB. But the Processing tab renders `progress.log`, `progress.parts`,
`progress.status_line` and `progress.runtime`, and the HUD reads
`progress.runtime.sections`.

So "stop polling while the socket is up" is only correct if something still
refreshes those. This prints exactly which keys a telemetry frame does and does
not carry, so the gap is a fact rather than a worry.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


def main() -> int:
    from fastapi.testclient import TestClient
    import api

    with TestClient(api.app) as client:
        poll = set(client.get("/api/progress").json())
        with client.websocket_connect("/ws/telemetry") as ws:
            frame = set(ws.receive_json())

    # Keys the UI reads off `progress` that the socket does NOT supply.
    consumed_by_ui = {
        "log": "Processing tab console",
        "parts": "Processing tab part tabs",
        "status_line": "pinned status line",
        "runtime": "Processing tab + hardware HUD",
        "output": "completed-run link",
    }

    missing = sorted(k for k in consumed_by_ui if k in poll and k not in frame)
    print(f"poll keys : {len(poll)}")
    print(f"frame keys: {len(frame)}")
    print(f"\nUI-consumed keys ABSENT from a telemetry frame ({len(missing)}):")
    for k in missing:
        print(f"  - {k:12} used by {consumed_by_ui[k]}")
    print("\nKeys the frame adds that the poll never had:")
    for k in sorted(frame - poll):
        print(f"  + {k}")

    if missing:
        print("\n=> Cutting the poll entirely WOULD freeze those. A slow poll must")
        print("   remain while a run is active to refresh them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
