"""Phase 4 (REST + telemetry): the 3-stage progress payload, the busy preview, the 200 ms start.

Three things the React client depends on, each driven through the real boundary:

* `/ws/telemetry` carries STAGE_1_SPARSE_PRESCAN / STAGE_2_ACTIVE_PREPASS / STAGE_3_MAIN_SWAP with
  stage_progress_pct, active_ranges_count, verified_target_frames, skipped_frames_pct, eta_seconds
  (`roop/stage_telemetry.py`, folded in by `api._telemetry_snapshot`), and stays CONSTANT while idle
  (the sampler sends on change, so an idle backend must send nothing).
* While Stage 1/2 owns the GPU burst, `/api/target/preview*` answers an UNCACHED frame with a fast 503
  `{busy, stage, retry_after_ms}` + `Retry-After`, and still serves a cached one.
* `/api/swap` builds the project on the request thread: a multi-GB target's SHA-256 must not be read
  there (`file_identity(defer_hash=True)`; `complete_hashes` runs on the worker).
"""
import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import stage_telemetry as st  # noqa: E402
import project_checkpoint as pc  # noqa: E402

# `wall_s` (seconds the stage has run / took) was added by Phase 9 with the per-job report.
STAGE_KEYS = {"state", "stage_progress_pct", "active_ranges_count", "verified_target_frames",
              "skipped_frames_pct", "eta_seconds", "wall_s"}


class Registry(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def test_idle_snapshot_is_constant_and_empty(self):
        self.assertEqual(st.snapshot(), {"stage": None, "stages": {}})
        self.assertEqual(st.snapshot(), st.snapshot())      # the change-only sampler must stay quiet
        self.assertIsNone(st.busy_stage())

    def test_every_stage_reports_every_field(self):
        st.begin_run(1000)
        for name, stage in st.snapshot()["stages"].items():
            self.assertEqual(set(stage), STAGE_KEYS, name)
        self.assertEqual(set(st.snapshot()["stages"]), set(st.STAGES))

    def test_stage1_progress_ranges_and_skipped_share(self):
        st.begin_run(1000)
        st.enter(st.STAGE_1_SPARSE_PRESCAN)
        st.progress(st.STAGE_1_SPARSE_PRESCAN, 25, 100)
        s1 = st.snapshot()["stages"][st.STAGE_1_SPARSE_PRESCAN]
        self.assertEqual((s1["state"], s1["stage_progress_pct"]), (st.RUNNING, 25.0))
        self.assertEqual(st.snapshot()["stage"], st.STAGE_1_SPARSE_PRESCAN)
        st.set_ranges(3, 300, 1000)
        st.leave(st.STAGE_1_SPARSE_PRESCAN)
        snap = st.snapshot()["stages"]
        self.assertEqual(snap[st.STAGE_1_SPARSE_PRESCAN]["stage_progress_pct"], 100.0)
        self.assertEqual(snap[st.STAGE_1_SPARSE_PRESCAN]["eta_seconds"], 0.0)
        for stage in snap.values():                          # run-level facts are on every stage
            self.assertEqual(stage["active_ranges_count"], 3)
            self.assertEqual(stage["skipped_frames_pct"], 70.0)

    def test_eta_is_the_stages_own_rate(self):
        st.begin_run(100)
        st.enter(st.STAGE_2_ACTIVE_PREPASS, 100)
        with mock.patch("roop.stage_telemetry.time.time", return_value=time.time() + 10.0):
            st.progress(st.STAGE_2_ACTIVE_PREPASS, 25)
            eta = st.snapshot()["stages"][st.STAGE_2_ACTIVE_PREPASS]["eta_seconds"]
        # 25 of 100 done in ~10 s -> ~30 s left (the clock moved 10 s from `since`).
        self.assertAlmostEqual(eta, 30.0, delta=2.0)
        self.assertIsNone(st.snapshot()["stages"][st.STAGE_3_MAIN_SWAP]["eta_seconds"])

    def test_skipped_stage_and_verified_frames(self):
        st.begin_run(500)
        st.skip(st.STAGE_1_SPARSE_PRESCAN)
        st.enter(st.STAGE_2_ACTIVE_PREPASS, 500)
        st.set_verified(120)
        snap = st.snapshot()["stages"]
        self.assertEqual(snap[st.STAGE_1_SPARSE_PRESCAN]["state"], st.SKIPPED)
        self.assertIsNone(snap[st.STAGE_1_SPARSE_PRESCAN]["stage_progress_pct"])
        self.assertEqual(snap[st.STAGE_2_ACTIVE_PREPASS]["verified_target_frames"], 120)
        self.assertIsNone(snap[st.STAGE_2_ACTIVE_PREPASS]["skipped_frames_pct"])   # no scan, no claim

    def test_busy_stage_covers_stage_1_and_2_but_not_the_render(self):
        st.begin_run(10)
        st.enter(st.STAGE_1_SPARSE_PRESCAN)
        self.assertEqual(st.busy_stage(), st.STAGE_1_SPARSE_PRESCAN)
        st.leave(st.STAGE_1_SPARSE_PRESCAN)
        st.enter(st.STAGE_2_ACTIVE_PREPASS)
        self.assertEqual(st.busy_stage(), st.STAGE_2_ACTIVE_PREPASS)
        st.leave(st.STAGE_2_ACTIVE_PREPASS)
        st.enter(st.STAGE_3_MAIN_SWAP, 10)
        self.assertIsNone(st.busy_stage())      # the render-active guard owns Stage 3

    def test_main_progress_is_overlaid_by_the_web_side(self):
        st.begin_run(200)
        st.enter(st.STAGE_3_MAIN_SWAP, 200)
        st.set_main_progress(50, 200, 12.5)
        s3 = st.snapshot()["stages"][st.STAGE_3_MAIN_SWAP]
        self.assertEqual((s3["stage_progress_pct"], s3["eta_seconds"]), (25.0, 12.5))

    def test_end_run_never_leaves_a_stage_claiming_the_gpu(self):
        st.begin_run(10)
        st.enter(st.STAGE_2_ACTIVE_PREPASS)
        st.end_run(ok=False)
        self.assertIsNone(st.busy_stage())
        self.assertEqual(st.snapshot()["stages"][st.STAGE_2_ACTIVE_PREPASS]["state"], st.FAILED)


class WireFormat(unittest.TestCase):
    """The real app: the hello frame and the sampler carry the stage block."""

    @classmethod
    def setUpClass(cls):
        import api
        from fastapi.testclient import TestClient
        cls.api = api
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def test_idle_hello_has_the_stage_keys_and_two_idle_snapshots_are_equal(self):
        a, b = self.api._telemetry_snapshot(), self.api._telemetry_snapshot()
        self.assertIn("stage", a)
        self.assertEqual(a["stages"], {})
        self.assertEqual(a, b)

    def _stage3_pct(self, desc):
        st.begin_run(200)
        st.skip(st.STAGE_1_SPARSE_PRESCAN)
        st.skip(st.STAGE_2_ACTIVE_PREPASS)
        st.enter(st.STAGE_3_MAIN_SWAP, 200)
        with mock.patch.dict(self.api._progress, {"processing": True, "desc": desc}):
            snap = self.api._telemetry_snapshot()
        return snap["stages"][st.STAGE_3_MAIN_SWAP]["stage_progress_pct"]

    def test_stage3_takes_the_swap_passs_own_counter(self):
        self.assertEqual(self._stage3_pct("Processing frame 50 / 200 (12.0 fps)"), 25.0)

    def test_stage3_ignores_the_previous_stages_text(self):
        # "Analyzing faces" carries no counter, so the fallback accumulator would be read; it must not
        # be taken for the swap pass's progress.
        self.assertEqual(self._stage3_pct("Analyzing faces"), 0.0)

    def test_a_running_stage_reaches_the_websocket(self):
        st.begin_run(1000)
        st.skip(st.STAGE_1_SPARSE_PRESCAN)
        st.enter(st.STAGE_2_ACTIVE_PREPASS, 400)
        st.progress(st.STAGE_2_ACTIVE_PREPASS, 100)
        with self.client.websocket_connect("/ws/telemetry") as ws:
            hello = ws.receive_json()
        self.assertEqual(hello["stage"], st.STAGE_2_ACTIVE_PREPASS)
        s2 = hello["stages"][st.STAGE_2_ACTIVE_PREPASS]
        self.assertEqual(set(s2), STAGE_KEYS)
        self.assertEqual(s2["stage_progress_pct"], 25.0)
        self.assertEqual(hello["stages"][st.STAGE_1_SPARSE_PRESCAN]["state"], st.SKIPPED)


class BusyPreview(unittest.TestCase):
    """An uncached frame during Stage 1/2 is a fast 503; a cached one is still served."""

    @classmethod
    def setUpClass(cls):
        import api
        from fastapi.testclient import TestClient
        cls.api = api
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        self.reads = []
        api = self.api

        def read(filename, frame):
            self.reads.append(frame)
            return np.full((32, 48, 3), frame % 255, dtype=np.uint8)

        patches = [
            mock.patch.object(api, "list_files_process", [SimpleNamespace(filename="clip.png")]),
            mock.patch.object(api, "_read_target_frame", read),
            mock.patch.object(api, "_frame_etag", lambda f, fr, w, fmt, q: f"{f}:{fr}:{w}:{fmt}:{q}"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with api._encoded_cache_lock:
            api._encoded_cache.clear()

    def _busy(self):
        st.begin_run(100)
        st.enter(st.STAGE_2_ACTIVE_PREPASS, 100)

    def test_uncached_frame_is_a_fast_503_with_a_retry_hint_and_no_decode(self):
        self._busy()
        t0 = time.perf_counter()
        r = self.client.get("/api/target/preview?index=0&frame=7")
        elapsed = time.perf_counter() - t0
        self.assertEqual(r.status_code, 503)
        body = r.json()
        self.assertTrue(body["busy"])
        self.assertEqual(body["stage"], st.STAGE_2_ACTIVE_PREPASS)
        self.assertGreater(body["retry_after_ms"], 0)
        self.assertEqual(r.headers["retry-after"], "1")
        self.assertEqual(self.reads, [], "a refused frame must not have been decoded")
        self.assertLess(elapsed, 0.2)

    def test_a_cached_frame_is_still_served_during_the_burst(self):
        r = self.client.get("/api/target/preview?index=0&frame=7")          # idle: decodes + caches
        self.assertEqual(r.status_code, 200)
        self._busy()
        r = self.client.get("/api/target/preview?index=0&frame=7")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.reads, [7], "the second request must come from the cache")

    def test_the_frame_is_served_again_when_the_stage_ends(self):
        self._busy()
        self.assertEqual(self.client.get("/api/target/preview?index=0&frame=9").status_code, 503)
        st.leave(st.STAGE_2_ACTIVE_PREPASS)
        self.assertEqual(self.client.get("/api/target/preview?index=0&frame=9").status_code, 200)

    def test_grid_is_refused_unless_every_frame_is_cached(self):
        self.assertEqual(self.client.get("/api/target/preview_grid?index=0&frames=1,2").status_code, 200)
        self._busy()
        self.assertEqual(self.client.get("/api/target/preview_grid?index=0&frames=1,2").status_code, 200)
        self.assertEqual(self.client.get("/api/target/preview_grid?index=0&frames=1,3").status_code, 503)


class DeferredHash(unittest.TestCase):
    """The request thread must not read the whole target; the worker completes the digest."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, "t.mp4")
        with open(self.target, "wb") as fh:
            fh.write(b"x" * 4096)
        patch = mock.patch.object(pc, "PROJECTS_DIR", self.tmp.name)
        patch.start()
        self.addCleanup(patch.stop)

    def _record(self, identity):
        cfg = SimpleNamespace(provider="tensorrt", trt_precision="mixed", hardware={}, hardware_signature="s")
        return pc.save(pc.new_project(job_id=None, name="t", payload={}, sources=[], target=identity,
                                      frame_start=0, frame_end=10, output={}, cfg=cfg))

    def test_deferred_identity_reads_nothing_and_is_marked_pending(self):
        with mock.patch.object(pc, "file_sha256", side_effect=AssertionError("read the whole file")):
            ident = pc.file_identity(self.target, defer_hash=True)
        self.assertTrue(ident["sha256_pending"])
        self.assertEqual(ident["sha256"], "")
        self.assertEqual(ident["size"], 4096)

    def test_a_slow_digest_does_not_slow_the_deferred_identity(self):
        def slow(path, chunk_size=0):
            time.sleep(1.0)
            return "d" * 64
        with mock.patch.object(pc, "file_sha256", slow):
            t0 = time.perf_counter()
            pc.file_identity(self.target, defer_hash=True)
            deferred = time.perf_counter() - t0
            t0 = time.perf_counter()
            pc.file_identity(self.target)
            eager = time.perf_counter() - t0
        self.assertLess(deferred, 0.05)
        self.assertGreaterEqual(eager, 1.0)

    def test_complete_hashes_fills_the_digest_and_clears_the_flag(self):
        record = self._record(pc.file_identity(self.target, defer_hash=True))
        done = pc.complete_hashes(record["id"])
        target = done["inputs"]["target"]
        self.assertEqual(target["sha256"], pc.file_sha256(self.target))
        self.assertNotIn("sha256_pending", target)
        self.assertEqual(pc.load(record["id"])["inputs"]["target"]["sha256"], target["sha256"])

    def test_pending_identity_still_catches_a_replaced_file(self):
        ident = pc.file_identity(self.target, defer_hash=True)
        reasons = []
        pc._same_file(ident, reasons, "target")
        self.assertEqual(reasons, [])
        with open(self.target, "ab") as fh:                      # truncated/extended = different size
            fh.write(b"y")
        pc._same_file(ident, reasons, "target")
        self.assertTrue(any("changed" in r for r in reasons), reasons)

    def test_a_file_changed_before_the_worker_hashes_is_not_blessed(self):
        record = self._record(pc.file_identity(self.target, defer_hash=True))
        with open(self.target, "ab") as fh:
            fh.write(b"y")
        done = pc.complete_hashes(record["id"])
        self.assertTrue(done["inputs"]["target"].get("sha256_pending"), "new bytes must not get a digest")
        reasons = []
        pc._same_file(done["inputs"]["target"], reasons, "target")
        self.assertTrue(reasons)

    # ── resume / validate on the request thread: size only; the worker re-verifies the digest ──
    def _full_record(self, defer):
        cfg = SimpleNamespace(provider="tensorrt", trt_precision="mixed", hardware={},
                              hardware_signature="s", output_path=self.tmp.name,
                              output_video_format="mp4", output_video_codec="", video_quality=None,
                              output_method="File", output_template="")
        record = pc.new_project(job_id=None, name="t", payload={}, sources=[],
                                target=pc.file_identity(self.target, defer_hash=defer),
                                frame_start=0, frame_end=10, cfg=cfg,
                                output={"directory": os.path.abspath(self.tmp.name), "format": "mp4",
                                        "codec": "", "quality": None, "method": "File", "template": ""})
        return pc.save(record), cfg

    def test_quick_validate_reads_no_file_contents(self):
        record, cfg = self._full_record(defer=False)
        with mock.patch.object(pc, "file_sha256", side_effect=AssertionError("hashed on the request thread")):
            self.assertEqual(pc.validate(record, cfg, quick=True), [])

    def test_quick_validate_still_catches_a_size_change_and_a_missing_file(self):
        record, cfg = self._full_record(defer=False)
        with open(self.target, "ab") as fh:
            fh.write(b"z")
        self.assertTrue(any("size mismatch" in r for r in pc.validate(record, cfg, quick=True)))
        os.remove(self.target)
        self.assertTrue(any("missing" in r for r in pc.validate(record, cfg, quick=True)))

    def test_a_same_size_edit_slips_past_quick_but_not_past_the_worker(self):
        record, cfg = self._full_record(defer=False)
        st_before = os.stat(self.target)
        with open(self.target, "r+b") as fh:                     # same size, different bytes
            fh.write(b"Y" * 4096)
        os.utime(self.target, ns=(st_before.st_atime_ns, st_before.st_mtime_ns))
        self.assertEqual(pc.validate(record, cfg, quick=True), [], "quick compares size only (documented)")
        self.assertTrue(any("sha256 mismatch" in r for r in pc.validate(record, cfg)))
        reasons = pc.finalize_inputs(record["id"])
        self.assertTrue(any("sha256 mismatch" in r for r in reasons), reasons)

    def test_finalize_inputs_does_not_hash_a_file_twice(self):
        record, _ = self._full_record(defer=True)
        real = pc.file_sha256
        calls = []

        def counting(path, chunk_size=1024 * 1024):
            calls.append(path)
            return real(path, chunk_size)
        with mock.patch.object(pc, "file_sha256", counting):
            self.assertEqual(pc.finalize_inputs(record["id"]), [])
        self.assertEqual(len(calls), 1, "a freshly completed digest must not be verified again")
        with mock.patch.object(pc, "file_sha256", counting):
            self.assertEqual(pc.finalize_inputs(record["id"]), [])      # now a stored digest: verified once
        self.assertEqual(len(calls), 2)

    def test_the_resume_endpoints_validate_quickly_and_the_worker_finalizes(self):
        projects = open(os.path.join(APP, "routes_projects.py"), encoding="utf-8").read()
        self.assertNotIn("checkpoints.validate(record, api.roop_globals.CFG)", projects)
        self.assertEqual(projects.count("quick=True"), 2)
        queue = open(os.path.join(APP, "routes_queue.py"), encoding="utf-8").read()
        self.assertIn("quick=True", queue[queue.index("def queue_resume"):])
        api_src = open(os.path.join(APP, "api.py"), encoding="utf-8").read()
        run = api_src[api_src.index("def _run_swap("):]
        self.assertIn("_project_checkpoint.finalize_inputs(project_id)", run[:6500])

    def test_the_request_path_defers_and_the_worker_completes(self):
        src = open(os.path.join(APP, "api.py"), encoding="utf-8").read()
        create = src[src.index("def _create_processing_project("):src.index("\ndef ", src.index("def _create_processing_project(") + 10)]
        sources = src[src.index("def _project_sources("):src.index("\ndef _project_target_faces(")]
        self.assertIn("file_identity(entry.filename, defer_hash=True)", create)
        self.assertIn("file_identity(path, defer_hash=True)", sources)
        run = src[src.index("def _run_swap("):]
        self.assertIn("_project_checkpoint.finalize_inputs(project_id)", run[:6500])


if __name__ == "__main__":
    unittest.main()
