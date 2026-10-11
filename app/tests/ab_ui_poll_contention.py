"""Does UI traffic cost the render anything? idle vs realistic-UI vs 10x-stress, end to end.

    env/Scripts/python.exe tests/ab_ui_poll_contention.py --clips d4,s7

Question (RECODE_STATUS 2026-10-11, Stage 7): the brief says the React UI's preview polling and HTTP 206 reads contend
for the GIL with the render. The app serves its UI from a uvicorn thread INSIDE the rendering process, so this is
testable: tests/ui_poll_launcher.py starts the real `api.app` on a thread of the render process (as run_api does), and
this driver plays the browser from OUTSIDE it (a separate process, like a real browser) while `baseline_snapshot.run_one`
renders with the live config.yaml.

Three load levels, same server up in all of them (idle = server listening, nobody asking):
  idle    no clients.
  ui      what the shipped UI does with the telemetry socket DOWN (FAST_POLL_MS 1000): /api/progress every 1 s, the live
          JPEG refetched when `live_seq` moves, and a player buffering /outputs/ with one 512 KiB 206 range per second.
  stress  10x that and then some: 4 progress pollers at 100 ms, 2 live-frame pollers at 50 ms, 2 range readers taking
          random 1 MiB ranges every 50 ms - a deliberately hostile browser.
Arm order is the palindrome idle,ui,stress,stress,ui,idle (ABBA generalised) after one discarded warm-up, with
ROOP_STAB_CHUNK_MB pinned from the warm-up. idle-vs-idle is the null control. Every arm is PROVED from its log (server
line) and the client's own counters (requests answered, errors); an arm that did not run as labelled is flagged and not
averaged. Poll latency is taken only while `live_seq` is advancing, i.e. while the frame loop is actually running.
"""
import argparse
import datetime
import http.client
import json
import os
import random
import socket
import statistics as st
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import baseline_snapshot as bs                                   # noqa: E402
import baseline_controlled as bc                                 # noqa: E402
import fixtures                                                  # noqa: E402

EXTRA = {"s7": {"rel": "single/s7.mp4", "sources": "harjot", "window": (0, 600),
                "capture": None, "capture_face": None}}

# (progress pollers, progress interval s, live-frame pollers, live interval s, range readers, range interval s, range bytes)
LOADS = {
    "idle": None,
    "ui": dict(progress=(1, 1.0), live=(1, None), ranges=(1, 1.0, 512 * 1024, False)),
    "stress": dict(progress=(4, 0.1), live=(2, 0.05), ranges=(2, 0.05, 1024 * 1024, True)),
}
ORDER = ["idle", "ui", "stress", "stress", "ui", "idle"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Clients:
    """The browser stand-in. Records (t, kind, ms, status, bytes) for every request."""

    def __init__(self, port, load):
        self.port, self.load = port, load
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.samples = []
        self.seq_trace = []          # (t, live_seq) from /api/progress
        self.seq = [0]
        self.threads = []
        self.media_size = [0]

    def _record(self, kind, t0, status, n):
        with self.lock:
            self.samples.append((t0, kind, (time.perf_counter() - t0) * 1000.0, status, n))

    def _conn(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)

    def _get(self, conn, kind, path, headers=None):
        t0 = time.perf_counter()
        try:
            conn.request("GET", path, headers=headers or {})
            r = conn.getresponse()
            body = r.read()
            self._record(kind, t0, r.status, len(body))
            return r.status, body
        except (OSError, http.client.HTTPException):
            self._record(kind, t0, 0, 0)
            try:
                conn.close()
            except Exception:
                pass
            return 0, b""

    def _wait_up(self):
        while not self.stop.is_set():
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return True
            except OSError:
                time.sleep(0.25)
        return False

    def _progress(self, interval, publish_seq):
        if not self._wait_up():
            return
        conn = self._conn()
        while not self.stop.is_set():
            t0 = time.perf_counter()
            status, body = self._get(conn, "progress", "/api/progress")
            if status == 200 and publish_seq:
                try:
                    seq = int(json.loads(body).get("live_seq") or 0)
                    with self.lock:
                        self.seq_trace.append((t0, seq))
                    self.seq[0] = seq
                except (ValueError, TypeError):
                    pass
            self.stop.wait(max(0.0, interval - (time.perf_counter() - t0)))

    def _live(self, interval):
        """interval None = the UI's behaviour: refetch exactly when live_seq changes."""
        if not self._wait_up():
            return
        conn = self._conn()
        last = -1
        n = 0
        while not self.stop.is_set():
            t0 = time.perf_counter()
            if interval is None:
                if self.seq[0] != last and self.seq[0] > 0:
                    last = self.seq[0]
                    self._get(conn, "live_frame", "/api/live_frame?seq=%d" % last)
                self.stop.wait(0.05)
            else:
                n += 1
                self._get(conn, "live_frame", "/api/live_frame?seq=%d" % n)
                self.stop.wait(max(0.0, interval - (time.perf_counter() - t0)))

    def _ranges(self, interval, nbytes, randomised):
        if not self._wait_up():
            return
        conn = self._conn()
        pos = 0
        rng = random.Random(7)
        size = 0
        while not self.stop.is_set():
            t0 = time.perf_counter()
            if size <= 0:                                   # HEAD-equivalent: ask for the first byte
                st_, _ = self._get(conn, "range_probe", "/outputs/uipoll_media.mp4", {"Range": "bytes=0-0"})
                if st_ == 206:
                    size = self.media_size[0] or 1
                    try:
                        size = os.path.getsize(self.media_path)
                    except OSError:
                        size = 0
                self.stop.wait(0.25)
                continue
            if randomised:
                pos = rng.randrange(0, max(1, size - nbytes))
            end = min(size - 1, pos + nbytes - 1)
            status, _ = self._get(conn, "range", "/outputs/uipoll_media.mp4", {"Range": "bytes=%d-%d" % (pos, end)})
            pos = 0 if end >= size - 1 else end + 1
            self.stop.wait(max(0.0, interval - (time.perf_counter() - t0)))

    def start(self, media_path):
        self.media_path = media_path
        if not self.load:
            return
        n, iv = self.load["progress"]
        for i in range(n):
            self.threads.append(threading.Thread(target=self._progress, args=(iv, i == 0), daemon=True))
        n, iv = self.load["live"]
        for _ in range(n):
            self.threads.append(threading.Thread(target=self._live, args=(iv,), daemon=True))
        n, iv, nb, rnd = self.load["ranges"]
        for _ in range(n):
            self.threads.append(threading.Thread(target=self._ranges, args=(iv, nb, rnd), daemon=True))
        for t in self.threads:
            t.start()

    def finish(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=20)

    def summary(self):
        with self.lock:
            samples, trace = list(self.samples), list(self.seq_trace)
        # Frame-loop window: while live_seq is advancing (the pipeline only publishes inside the frame loop).
        window = None
        moved = [t for (t, s), (_, s0) in zip(trace[1:], trace[:-1]) if s != s0]
        if moved:
            window = (moved[0], moved[-1])
        out = {"window_s": round(window[1] - window[0], 1) if window else None,
               "live_seq_advanced": bool(moved), "total_requests": len(samples)}
        for kind in sorted({s[1] for s in samples}):
            rows = [s for s in samples if s[1] == kind and (window is None or window[0] <= s[0] <= window[1])]
            ok = [r for r in rows if r[3] in (200, 204, 206)]
            ms = sorted(r[2] for r in ok)
            out[kind] = {"n": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
                         "p50_ms": round(ms[len(ms) // 2], 2) if ms else None,
                         "p99_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.99))], 2) if ms else None,
                         "max_ms": round(ms[-1], 2) if ms else None,
                         "mb": round(sum(r[4] for r in ok) / 1048576.0, 1),
                         "statuses": sorted({r[3] for r in rows})}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="d4,s7")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--threads", type=int, default=20, help="AGENTS.md: --threads 20 on every bench run")
    ap.add_argument("--window", nargs=2, type=int, default=None)
    ap.add_argument("--order", default=",".join(ORDER))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.detector_engine = None
    args.governor = False
    order = args.order.split(",")

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    threads = args.threads
    out_root = args.out or os.path.join(APP, "output", "ab_ui_poll_%s" % args.date)
    os.makedirs(out_root, exist_ok=True)
    base_env = dict(os.environ)
    base_env["ROOP_PROFILE"] = "1"
    base_env["PYTHONIOENCODING"] = "utf-8"
    base_env.setdefault("ROOP_PIPELINE_LOG_EVERY", "100")
    base_env["ROOP_BENCH_LAUNCHER"] = os.path.join(HERE, "ui_poll_launcher.py")
    os.environ["ROOP_BENCH_LAUNCHER"] = base_env["ROOP_BENCH_LAUNCHER"]     # build_cmd reads the parent's env
    base_env = bc.ensure_ffmpeg(base_env)
    os.environ["PATH"] = base_env["PATH"]

    specs = dict(bs.CLIPS)
    specs.update(EXTRA)
    summary = {"date": args.date, "head": bs.head_info(), "threads": threads, "order": order,
               "config": {k: str(getattr(cfg, k, None)) for k in (
                   "swap_model", "selected_enhancer", "mask_engine", "provider", "trt_precision", "max_threads",
                   "detector_engine", "face_detector_size", "perf_batch_swap", "perf_trt_pool",
                   "perf_detmask_pool", "perf_detector_pool")},
               "loads": {k: v for k, v in LOADS.items()}, "clips": {}}
    path = os.path.join(REPO, "docs", "perf", "ui_poll_contention_%s.json" % args.date)

    def one(label, clip, spec, load_name, env, window):
        port = free_port()
        env = dict(env)
        env["ROOP_UIPOLL_PORT"] = str(port)
        media = fixtures.clip(spec["rel"], required=True)
        env["ROOP_UIPOLL_MEDIA"] = media
        clients = Clients(port, LOADS[load_name])
        clients.start(media)
        try:
            rec = bs.run_one(label, clip, spec, cfg, args, env, out_root, threads, window=window)
        finally:
            clients.finish()
        text = open(rec["log"], encoding="utf-8", errors="ignore").read()
        rec["load"] = load_name
        rec["client"] = clients.summary()
        rec["server_up"] = "[UIPoll] server up" in text
        c = rec["client"]
        asked = LOADS[load_name] is not None
        served = bool(asked and c.get("progress", {}).get("ok", 0) > 0 and c.get("live_seq_advanced"))
        rec["ran_as_labelled"] = bool(rec["server_up"] and rec["returncode"] == 0 and (served or not asked))
        return rec

    for clip in args.clips.split(","):
        spec = specs[clip]
        window = tuple(args.window) if args.window else None
        w = one("warmup_" + clip, clip, spec, "idle", dict(base_env), (0, 120))
        chunk = (w.get("stab_ram_line") or {}).get("derived_chunk_budget_mb")
        print("  warm-up %s done (rc %s); derived ROOP_STAB_CHUNK_MB = %s -> pinned" % (clip, w["returncode"], chunk), flush=True)
        recs = []
        for i, load_name in enumerate(order):
            env = dict(base_env)
            if chunk:
                env["ROOP_STAB_CHUNK_MB"] = str(chunk)
            rec = one("%s_%d_%s" % (clip, i + 1, load_name), clip, spec, load_name, env, window)
            c = rec["client"]
            print("  %s: frame-loop %s fps | faces/s %s | faces_seen %s | stab %s | reqs %s | progress p99 %s ms | "
                  "live p99 %s ms | range p99 %s ms | ran-as-labelled %s"
                  % (rec["label"], rec.get("frame_loop_fps"), rec.get("faces_per_s"),
                     (rec.get("run") or {}).get("faces_seen"), rec.get("stabilizer_path"), c["total_requests"],
                     c.get("progress", {}).get("p99_ms"), c.get("live_frame", {}).get("p99_ms"),
                     c.get("range", {}).get("p99_ms"), rec["ran_as_labelled"]), flush=True)
            recs.append(rec)
        good = [r for r in recs if r["ran_as_labelled"]]
        fps = {k: [r["frame_loop_fps"] for r in good if r["load"] == k and r["frame_loop_fps"]] for k in LOADS}
        mean = {k: round(st.mean(v), 3) if v else None for k, v in fps.items()}

        def ratio(k):
            return round(mean[k] / mean["idle"], 4) if mean.get(k) and mean.get("idle") else None

        def spread(v):
            return round(100 * abs(v[0] - v[1]) / st.mean(v), 2) if len(v) == 2 else None
        row = {"fps": fps, "mean": mean, "ui_over_idle": ratio("ui"), "stress_over_idle": ratio("stress"),
               "idle_vs_idle_spread_pct": spread(fps["idle"]), "ui_vs_ui_spread_pct": spread(fps["ui"]),
               "stress_vs_stress_spread_pct": spread(fps["stress"]), "chunk_mb": chunk,
               "arms_not_as_labelled": [r["label"] for r in recs if not r["ran_as_labelled"]],
               "arms": [{k: r.get(k) for k in ("label", "load", "returncode", "frame_loop_fps", "faces_per_s",
                                               "faces_per_frame", "stabilizer_path", "server_up", "ran_as_labelled",
                                               "client", "failed_frames", "swapped_lines", "wall_seconds")}
                        for r in recs]}
        summary["clips"][clip] = row
        print("\n== %s: idle %s | ui %s | stress %s | ui/idle %s | stress/idle %s | null %s%% | ui-ui %s%% | stress-stress %s%% "
              "| unlabelled %s\n" % (clip, fps["idle"], fps["ui"], fps["stress"], row["ui_over_idle"], row["stress_over_idle"],
                                     row["idle_vs_idle_spread_pct"], row["ui_vs_ui_spread_pct"],
                                     row["stress_vs_stress_spread_pct"], row["arms_not_as_labelled"]), flush=True)
        json.dump(summary, open(path, "w"), indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
