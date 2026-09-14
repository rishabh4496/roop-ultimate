"""Measure the REAL /api/progress payload, not a mock of it.

Imports the actual app and drives it through TestClient, so the numbers come
from the endpoint the React UI really polls once a second -- including the
rolling log, the parts snapshot and the nested `runtime` block.

Run: app/env/Scripts/python.exe app/tests/probe_progress_cost.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

os.environ.setdefault("ROOP_REACT_CLIENT", "1")


def main() -> int:
    from fastapi.testclient import TestClient
    import api

    client = TestClient(api.app)

    def sample(label):
        t0 = time.perf_counter()
        n = 20
        size = 0
        for _ in range(n):
            r = client.get("/api/progress")
            size += len(r.content)
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"  {label:22} {ms:7.2f} ms/req   {size / n:10,.0f} bytes/req")
        return size / n

    print("real /api/progress, idle:")
    idle = sample("idle")

    # Simulate a running job the way the pipeline does: fill the rolling log and
    # flip the processing flag. This is the state the UI polls at 1 Hz for the
    # entire length of a render, which is where the cost actually lands.
    api._progress.update({"processing": True, "progress": 0.42,
                          "desc": "128 / 300", "error": ""})
    api._run_stats.update({"start": time.time() - 120, "frames_done": 128,
                           "frames_total": 300})
    for i in range(250):
        api._log_lines.append(f"[swap] frame {i} / 300  ·  23.4 fps  ·  eta 00:07")

    print("\nreal /api/progress, mid-render (250-line log):")
    busy = sample("processing")

    push = json.dumps({"event": "progress", "current_frame": 128,
                       "total_frames": 300, "progress": 42.67,
                       "fps": 23.4, "eta_s": 7}).encode()
    print(f"\n  pushed telemetry frame {len(push):,} bytes")
    print(f"  ratio mid-render       {busy / len(push):.0f}x more bytes per update when polling")

    hours = 1
    polls = 3600 * hours
    print(f"\n  over a {hours}h render at 1 Hz:")
    print(f"    polling: {polls * busy / 1048576:8.1f} MB  ({polls:,} requests)")
    print(f"    push   : {polls * len(push) / 1048576:8.1f} MB  (same update rate, no request overhead)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
