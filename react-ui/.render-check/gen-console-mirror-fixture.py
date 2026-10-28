"""Brings console-mirror.fixture.json up to the dashboard's CURRENT latency columns, using the real backend code.

    app/env/Scripts/python.exe react-ui/.render-check/gen-console-mirror-fixture.py

The fixture is what `roop/terminal_dashboard.py` painted: `running` / `final` are the snapshots `/ws/telemetry` carries under
`dashboard`, `console` the text rendered from them. Stage 17 added a Lip-Sync column, so this script:

  1. checks that `render_telemetry_lines(running)` reproduces the stored console text byte for byte (so the renderer
     parameters used below are the ones that made the file; if they do not, it stops rather than rewrite history);
  2. gives every stored snapshot the new column as NOT MEASURED (None): these were runs with lip-sync off, and the screen
     must show an em dash for it, never 0;
  3. re-renders the console text from the updated snapshots with the real renderer;
  4. records a SECOND run, `lipsync_run`, driven through the real registry + manager with `lipsync` recorded like
     ProcessMgr's `_prof('lipsync')` does, so the measured column, its share of the bottleneck and the text the console
     paints for it are the backend's own numbers.

Idempotent: running it twice changes nothing.
"""
import json
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "app"))
PATH = os.path.join(HERE, "console-mirror.fixture.json")
WIDTH = 160           # >= 120 prints full column labels, which is what the React screen is compared against

from roop import stage_telemetry as st  # noqa: E402
from roop import terminal_dashboard as td  # noqa: E402


def _ordered_latency(lat):
    """The latency dict in the dashboard's column order, new columns present (None = not measured), `other` last."""
    out = {key: lat.get(key) for key, *_ in td.LATENCY_COLUMNS}
    out["other"] = lat.get("other", 0.0)
    return out


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def measured_run():
    """A Stage 3 window with lip-sync on: the same shape the dashboard tests drive (tests/test_terminal_dashboard.py)."""
    st.reset()
    td.reset()
    clock = Clock()
    with mock.patch.object(st.time, "time", clock), \
            mock.patch.object(st._vram, "start", lambda: st._vram.reset()), \
            mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "log"}), \
            mock.patch("builtins.print"):
        st._vram.stop()
        st._vram._probe = lambda: None
        mgr = td.TerminalProgressManager(clock=clock, wall=clock)
        mgr.TICK_S = 3600.0
        try:
            mgr.begin_job({"job_id": "lipsync-run", "mode": "video"}, None)
            st.begin_run(1000)
            st.enter(st.STAGE_3_MAIN_SWAP, 1000)
            clock.advance(0.5)
            mgr._tick()
            snap = None
            for _ in range(4):
                for _ in range(25):
                    for name, sec in (("decode", 0.004), ("detect", 0.006), ("swap", 0.008), ("mask", 0.004),
                                      ("lipsync", 0.005), ("enhance", 0.032), ("blend", 0.002), ("encode", 0.003)):
                        st.record_op(name, sec)
                    st.retire()
                clock.advance(1.0)
                mgr._tick()
                snap = mgr.snapshot()
            return json.loads(json.dumps(snap, default=str))
        finally:
            mgr.end_job(ok=False, _quiet=True)
            st.reset()
            td.reset()


def main():
    with open(PATH, encoding="utf-8") as fh:
        fx = json.load(fh)

    # The stored text was painted before the Lip-Sync column existed, at a width that prints full labels. A second run of
    # this script finds the column already stored, so it must reproduce with the CURRENT columns instead.
    without_lipsync = tuple(c for c in td.LATENCY_COLUMNS if c[0] != "lipsync")
    with mock.patch.object(td, "LATENCY_COLUMNS", without_lipsync):
        before = td.render_telemetry_lines(fx["running"], color=False, width=WIDTH)
    now = td.render_telemetry_lines(fx["running"], color=False, width=WIDTH)
    if fx["console"]["telemetry_lines"] not in (before, now):
        sys.exit("the renderer does not reproduce the stored console text; refusing to rewrite the fixture:\n  %r\n  %r"
                 % (before, fx["console"]["telemetry_lines"]))

    for key in ("running", "final"):
        fx[key]["telemetry"]["latency_ms"] = _ordered_latency(fx[key]["telemetry"]["latency_ms"])
    fx["console"]["telemetry_lines"] = td.render_telemetry_lines(fx["running"], color=False, width=WIDTH)

    run = measured_run()
    t = run["telemetry"]
    if not t["latency_ms"].get("lipsync"):
        sys.exit("the real registry did not record the lipsync column: %r" % (t["latency_ms"],))
    fx["lipsync_run"] = {"telemetry": t, "console": td.render_telemetry_lines(run, color=False, width=WIDTH)}

    # The format the file was already in: 1-space indent, ASCII escapes, and the line endings it had.
    with open(PATH, "rb") as fh:
        crlf = b"\r\n" in fh.read(4096)
    with open(PATH, "w", encoding="utf-8", newline="\r\n" if crlf else "\n") as fh:
        json.dump(fx, fh, indent=1)
        fh.write("\n")
    print("lipsync column measured at %.1f ms; bottleneck %s %.0f%%"
          % (t["latency_ms"]["lipsync"], t["bottleneck"]["label"], t["bottleneck"]["share_pct"]))


if __name__ == "__main__":
    main()
