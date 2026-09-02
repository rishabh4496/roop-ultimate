"""Measure whether the running backend talks to anything but loopback.

WHY THIS EXISTS.  Offline operation is an acceptance criterion, and every
previous gate recorded it as simulated: "deterministic disconnected unit tests
pass; the network adapter was not disconnected".  Unit tests with a patched
socket prove the POLICY branch; they cannot prove the running application does
not reach the network somewhere else.

This measures the opposite direction, which needs no disconnection and is
therefore reproducible on any host: it samples the live backend process tree's
own TCP endpoints while real work is happening, and reports every remote peer
that is not loopback.  Zero non-loopback peers across a full render is direct
evidence that the local workflow does not depend on the Internet.

WHAT IT IS NOT.  It is not proof that the app behaves correctly WHEN
disconnected -- a physical disconnection test is still owed, and model
acquisition legitimately needs the network for a model that is not yet on disk.
It bounds the claim to what was measured.

    env/Scripts/python.exe tests/local_only_probe.py --seconds 60
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)


def backend_pids():
    """Every python process in this checkout running the application entry."""
    found = []
    for process in psutil.process_iter(["name", "cmdline"]):
        if (process.info.get("name") or "").lower() != "python.exe":
            continue
        cmdline = process.info.get("cmdline") or []
        if any(str(arg).endswith("run.py") for arg in cmdline):
            found.append(process.pid)
    return found


def sample(pids, seconds, interval=0.5):
    remote = collections.Counter()
    samples = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for pid in list(pids):
            try:
                process = psutil.Process(pid)
                # psutil renamed Process.connections -> net_connections in 6.0;
                # this environment ships the older name, so support both rather
                # than silently observing nothing.
                lookup = getattr(process, "net_connections", None) or process.connections
                for conn in lookup(kind="inet"):
                    address = conn.raddr
                    if address and address.ip not in ("127.0.0.1", "::1", ""):
                        remote[f"{address.ip}:{address.port} {conn.status}"] += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        samples += 1
        time.sleep(interval)
    return samples, remote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    pids = backend_pids()
    if not pids:
        print("[net] BLOCKED  no running backend found; start a render first")
        return 2
    print(f"[net] observing backend pids {pids} for {args.seconds:.0f}s", flush=True)
    samples, remote = sample(pids, args.seconds)

    ok = not remote
    print(f"[net] {'PASS' if ok else 'FAIL'}  {samples} samples; "
          f"non-loopback peers: {dict(remote) or 'NONE'}")
    if not ok:
        print("[net] the local workflow contacted a remote host; that is only "
              "acceptable for an explicit model download")
    report = {"pids": pids, "samples": samples, "seconds": args.seconds,
              "non_loopback": dict(remote), "local_only": ok}
    out = args.out or os.path.join(APP, "output", "local_only_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
