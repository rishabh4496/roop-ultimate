"""Stage 14: TensorRT compile receipts + terminal spinner, the runtime VRAM rungs, Decode/GPU/Encode queue levels.

Every test drives the REAL call site (`predictor.warmup_session`, `UnifiedRuntimeScheduler._update_admission`,
`SwapBatcher`, `PipelineMonitor`, `terminal_dashboard`) with a stub only where the hardware would be: an ORT session
that writes (or does not write) an engine file, a free-VRAM sample, a swap function. A test that stubbed the thing it
claims to cover would pass on code that never runs, which is this repository's recurring failure.
"""
import io
import os
import re
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))

from roop import pipeline_status as ps  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class TrtSession:
    """Just enough of an ORT TensorRT session. `build=True` writes an engine plan into its cache when run."""

    def __init__(self, cache, build, workspace=2 * 1024 ** 3, fp16=True):
        self.cache, self.build = cache, build
        self._opts = {"trt_fp16_enable": fp16, "trt_max_workspace_size": workspace,
                      "trt_engine_cache_enable": True, "trt_engine_cache_path": cache}

    def get_provider_options(self):
        return {"TensorrtExecutionProvider": self._opts}

    def get_inputs(self):
        return [type("M", (), {"name": "x", "shape": [1, 3, 8, 8], "type": "tensor(float)"})()]

    def run(self, _out, _feed):
        if self.build:
            with open(os.path.join(self.cache, "TensorrtExecutionProvider_m_%d.engine" % time.time_ns()), "wb") as f:
                f.write(b"plan")
        return []


class CompileDetail(unittest.TestCase):
    def setUp(self):
        ps.reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = self.tmp.name
        from roop import predictor
        predictor.reset()

    def tearDown(self):
        self.tmp.cleanup()

    def test_trt_info_reads_the_options_the_session_was_built_with(self):
        info = ps.session_trt_info(TrtSession(self.cache, False, workspace=1536 * 1024 ** 2, fp16=False))
        self.assertEqual((info["precision"], info["workspace_mb"], info["cache_path"]), ("FP32", 1536, self.cache))

    def test_a_session_that_is_not_tensorrt_reports_nothing_to_guess(self):
        class Cuda:
            def get_provider_options(self):
                return {"CUDAExecutionProvider": {}}
        self.assertEqual(ps.session_trt_info(Cuda()), {"precision": "n/a", "workspace_mb": None, "cache_path": None})

    def test_cache_hit_needs_both_listings_and_sees_a_new_or_rewritten_plan(self):
        plan = os.path.join(self.cache, "a.engine")
        open(plan, "wb").write(b"1")
        before = ps.engine_cache_files(self.cache)
        self.assertTrue(ps.cache_hit(before, ps.engine_cache_files(self.cache)))
        open(os.path.join(self.cache, "b.engine"), "wb").write(b"2")
        self.assertFalse(ps.cache_hit(before, ps.engine_cache_files(self.cache)), "a new plan = a build")
        before = ps.engine_cache_files(self.cache)
        open(plan, "wb").write(b"123456")
        self.assertFalse(ps.cache_hit(before, ps.engine_cache_files(self.cache)), "a rewritten plan = a build")
        self.assertIsNone(ps.cache_hit(None, before), "an unreadable cache is unknown, not a hit")
        self.assertIsNone(ps.engine_cache_files(os.path.join(self.cache, "missing")))

    def test_warmup_announces_workspace_and_cache_then_leaves_a_miss_receipt(self):
        from roop import predictor
        seen = []
        session = TrtSession(self.cache, build=True)
        real_run = session.run
        session.run = lambda o, f: (seen.append(ps.snapshot()["active"][0]), real_run(o, f))[1]
        self.assertTrue(predictor.warmup_session(session, tag="gpen_256", once=False))
        live = seen[0]
        self.assertEqual((live["stage"], live["model"], live["precision"], live["workspace_mb"], live["cache_path"]),
                         ("TRT_COMPILE", "gpen_256", "FP16", 2048, self.cache))
        last = ps.snapshot()["last_compile"]
        self.assertIs(last["cache_hit"], False)
        self.assertTrue(last["ok"])
        self.assertTrue(ps.format_receipt(last).startswith("[TRT CACHE MISS: built in "))

    def test_a_second_warmup_that_builds_nothing_is_a_cache_hit(self):
        from roop import predictor
        predictor.warmup_session(TrtSession(self.cache, build=True), tag="m", once=False)
        predictor.warmup_session(TrtSession(self.cache, build=False), tag="m", once=False)
        last = ps.snapshot()["last_compile"]
        self.assertIs(last["cache_hit"], True)
        self.assertRegex(ps.format_receipt(last), r"^\[TRT CACHE HIT: \d+\.\d\ds\] m FP16$")

    def test_a_failed_build_is_a_failed_receipt_not_a_hit(self):
        from roop import predictor

        class Boom(TrtSession):
            def run(self, *_):
                raise RuntimeError("engine build failed")
        self.assertFalse(predictor.warmup_session(Boom(self.cache, False), tag="bad", once=False))
        last = ps.snapshot()["last_compile"]
        self.assertFalse(last["ok"])
        self.assertTrue(ps.format_receipt(last).startswith("[TRT BUILD FAILED after "))

    def test_the_pinned_telemetry_contract_did_not_move(self):
        """`/ws/telemetry` carries telemetry_fields(); the compile detail rides /ws/v1/pipeline/status instead."""
        with ps.stage(ps.STAGE_COMPILING_TRT, model="m", precision="FP16", workspace_mb=2048, cache_path="c"):
            self.assertEqual(set(ps.telemetry_fields()), {
                "engine_state", "engine_label", "engine_model", "engine_precision", "engine_since", "engine_active"})


class TerminalSpinner(unittest.TestCase):
    def setUp(self):
        ps.reset()
        from roop import trt_console
        trt_console.detach()
        self.console = trt_console
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.console.detach()
        self.tmp.cleanup()

    def test_off_a_tty_it_prints_one_start_line_and_the_receipt(self):
        from roop import predictor
        said = []
        predictor.reset()
        with patch.object(self.console, "_say", said.append):
            self.console.attach()
            self.assertFalse(self.console.can_spin(), "pytest is not an ANSI terminal")
            predictor.warmup_session(TrtSession(self.tmp.name, build=True), tag="gpen_256", once=False)
        self.assertEqual(len(said), 2, said)
        self.assertIn("[TRT] Engine warmup: gpen_256 · FP16 · workspace 2048 MB", said[0])
        self.assertIn(self.tmp.name, said[0])
        self.assertTrue(said[1].startswith("[TRT CACHE MISS: built in "), said[1])

    def test_a_cuda_warmup_is_not_an_engine_build_and_prints_nothing(self):
        said = []
        with patch.object(self.console, "_say", said.append):
            self.console.attach()
            with ps.stage(ps.STAGE_COMPILING_TRT, model="cuda_model", precision="n/a"):
                pass
        self.assertEqual(said, [])

    def test_attach_twice_still_prints_each_event_once(self):
        said = []
        with patch.object(self.console, "_say", said.append):
            self.console.attach()
            self.console.attach()
            with ps.stage(ps.STAGE_COMPILING_TRT, model="m", precision="FP16"):
                pass
        self.assertEqual(len(said), 2)

    def test_a_listener_fault_never_stops_a_build(self):
        ps.subscribe(lambda _e: 1 / 0)
        with ps.stage(ps.STAGE_COMPILING_TRT, model="m", precision="FP16"):
            pass
        self.assertEqual(ps.snapshot()["events"][-1]["state"], "finished")

    def test_the_frame_never_wraps_and_keeps_the_timer(self):
        info = {"model": "gpen_256", "precision": "FP16", "workspace_mb": 2048,
                "cache_path": os.path.join("trt_cache", "fp16_mixed_lnfp32_seq_c3b1a9752fee69034", "deeply", "nested")}
        for width in (60, 80, 120, 200):
            text = ANSI.sub("", self.console.frame_line(info, 3, 72.0, width))
            self.assertLessEqual(len(text), width - 1, (width, text))
            self.assertIn("01:12", text)
            self.assertIn("TensorRT Engine Warmup", text)
        self.assertIn("cache:", ANSI.sub("", self.console.frame_line(info, 0, 0, 200)))

    def test_the_spinner_thread_paints_and_then_clears_its_line(self):
        out = io.StringIO()
        with patch.object(sys, "stdout", out):
            spinner = self.console._Spinner({"model": "m", "precision": "FP16"}, time.time())
            spinner.start()
            time.sleep(0.35)
            spinner.finish()
        text = out.getvalue()
        self.assertGreaterEqual(text.count("TensorRT Engine Warmup"), 2, "it repaints, not prints once")
        self.assertTrue(text.endswith("\r\x1b[2K"), "it leaves the line clear for the receipt")
        self.assertIn("\x1b[36m", text, "cyan")


class RuntimeRungs(unittest.TestCase):
    def setUp(self):
        ps.reset()
        from roop import vram_governor
        self.vg = vram_governor
        vram_governor.clear_batchers()
        self.batches = []
        from roop.swap_batcher import SwapBatcher

        def run(items):
            self.batches.append(len(items))
            return [i[0] for i in items]
        self.batcher = SwapBatcher(run, lambda: __import__("contextlib").nullcontext(), max_batch=8, max_wait_ms=400.0)
        self.vg.register_batcher(self.batcher)

    def tearDown(self):
        self.batcher.stop()

    def _submit(self, n):
        reqs = [self.batcher.submit(i, i, None) for i in range(n)]
        for r in reqs:
            self.batcher.wait(r)

    def test_the_batch_ladder_and_its_restore(self):
        self.assertEqual([self.vg.runtime_step_down() for _ in range(4)],
                         ["swap batch 8 -> 4", "swap batch 4 -> 2", "swap batch 2 -> 1", None])
        self.assertEqual(self.batcher.max_batch, 1)
        self.assertEqual([self.vg.runtime_restore() for _ in range(4)],
                         ["swap batch 1 -> 2", "swap batch 2 -> 4", "swap batch 4 -> 8", None])
        self.assertEqual(self.batcher.set_max_batch(64), 8, "never above the size the engine was warmed up for")

    def test_a_lowered_ceiling_changes_the_batches_the_gpu_actually_gets(self):
        self._submit(8)
        self.assertEqual(self.batches, [8])
        self.vg.runtime_step_down()
        self.vg.runtime_step_down()                       # 8 -> 2
        self.batches.clear()
        self._submit(8)
        self.assertTrue(self.batches and max(self.batches) <= 2, self.batches)
        self.assertEqual(sum(self.batches), 8, "no frame lost to the smaller batches")

    def test_announce_prints_the_governor_line_with_the_percentage(self):
        said = []
        with patch.object(self.vg, "_say", said.append):
            line = self.vg.announce("in-flight frames 3 -> 2", used_mb=11500, total_mb=12000, detail="VRAM cache flushed")
        self.assertEqual(said, [line])
        self.assertEqual(line, "[VRAM-GOVERNOR] in-flight frames 3 -> 2 at 96% VRAM (11500 of 12000 MiB) - VRAM cache flushed")

    def test_a_finished_render_leaves_no_batcher_for_the_next_one_to_lower(self):
        self.vg.finish()
        self.assertIsNone(self.vg.runtime_step_down())
        self.assertEqual(self.batcher.max_batch, 8)

    def test_tiers(self):
        self.assertEqual([self.vg.tier_label(mb) for mb in (6144, 8188, 12282, 16376, 24576)],
                         ["6GB", "8GB", "12GB", "16GB", "24GB+"])


class SchedulerRungs(unittest.TestCase):
    """UnifiedRuntimeScheduler._update_admission, on a 12 GB card, through `observe`."""

    def setUp(self):
        ps.reset()
        from roop import vram_governor
        from roop.runtime_optimizer import RuntimeTuning
        from roop.runtime_scheduler import UnifiedRuntimeScheduler
        from roop.swap_batcher import SwapBatcher
        from test_runtime_scheduler import _hardware, _workload
        self.vg = vram_governor
        vram_governor.clear_batchers()
        self.batcher = SwapBatcher(lambda items: [i[0] for i in items],
                                   lambda: __import__("contextlib").nullcontext(), max_batch=8)
        vram_governor.register_batcher(self.batcher)
        self._make = lambda: UnifiedRuntimeScheduler(
            _hardware(12.0), _workload(),
            RuntimeTuning(worker_count=4, queue_depth=3, in_flight_frames=4, ram_buffer_mb=1024))
        self.sched = self._make()
        self.said = []
        self.patches = [patch.object(vram_governor, "_say", self.said.append),
                        patch.object(self.sched, "_flush_vram_cache", lambda: self.flushes.append(1))]
        self.flushes = []
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.batcher.stop()

    def _observe(self, used_fraction, times=2, total=12.0):
        sample = {"time": time.time(), "vram_free_gb": total * (1.0 - used_fraction), "vram_total_gb": total,
                  "ram_available_gb": 24.0, "ram_total_gb": 32.0}
        with patch.object(self.sched, "_resource_snapshot", return_value=sample):
            for _ in range(times):
                self.sched.observe({}, force=True)

    def test_past_ninety_percent_it_flushes_steps_down_and_says_so(self):
        # A 2% margin ratio would leave the proportional rule silent at 93% used: only the ceiling can fire.
        with patch.dict(os.environ, {"ROOP_SCHEDULER_VRAM_MARGIN": "0.02", "ROOP_GOVERNOR_MIN_FREE_GB": "0"}):
            self.sched = self._make()
            self.assertAlmostEqual(self.sched.budget.vram_margin_ratio, 0.02)
            limit = self.sched.budget.in_flight_limit
            with patch.object(self.sched, "_flush_vram_cache", lambda: self.flushes.append(1)):
                self._observe(0.93)
        self.assertTrue(self.flushes, "the 90% ceiling flushes even when the margin ratio would not")
        self.assertEqual(self.sched.effective_inflight, limit - 1)
        governor_lines = [line for line in self.said if line.startswith("[VRAM-GOVERNOR]")]
        self.assertEqual(len(governor_lines), 1, self.said)
        self.assertIn("in-flight frames %d -> %d" % (limit, limit - 1), governor_lines[0])
        self.assertIn("at 93% VRAM", governor_lines[0])

    def test_with_in_flight_exhausted_the_swap_batch_is_the_next_rung_and_recovery_restores_it(self):
        with patch.dict(os.environ, {"ROOP_GOVERNOR_MIN_FREE_GB": "0"}):
            self._observe(0.95, times=2 * (self.sched.budget.in_flight_limit + 1))
            self.assertEqual(self.sched.effective_inflight, 1)
            self.assertLess(self.batcher.max_batch, 8, "pressure at in-flight 1 lowered the swap batch")
            lowered = self.batcher.max_batch
            self.assertTrue(any("swap batch 8 ->" in line for line in self.said), self.said)
            throttle = [e for e in ps.snapshot()["events"] if e.get("action", "").startswith("swap batch")]
            self.assertTrue(throttle, "the status channel carries the batch step too")
            self._observe(0.40, times=4 * (self.sched.budget.in_flight_limit + 2))
        self.assertEqual(self.sched.effective_inflight, self.sched.budget.in_flight_limit)
        self.assertGreater(self.batcher.max_batch, lowered, "healthy again: the batch comes back")
        self.assertTrue(any("VRAM healthy again" in line for line in self.said), self.said)

    def test_queue_pressure_alone_never_touches_the_swap_batch(self):
        sample = {"time": time.time(), "vram_free_gb": 9.0, "vram_total_gb": 12.0,
                  "ram_available_gb": 24.0, "ram_total_gb": 32.0}
        full = {"input": self.sched.queue_capacity, "output": self.sched.queue_capacity}
        with patch.object(self.sched, "_resource_snapshot", return_value=sample):
            for _ in range(12):
                self.sched.observe(full, force=True)
        self.assertEqual(self.batcher.max_batch, 8)
        self.assertFalse([line for line in self.said if "swap batch" in line])


class QueueLevels(unittest.TestCase):
    def setUp(self):
        from roop import pipeline_monitor
        self.pm = pipeline_monitor
        pipeline_monitor.clear_levels()

    def tearDown(self):
        self.pm.clear_levels()                    # process-global: do not leak a view into the next test's snapshot

    def test_classify_bound_table(self):
        c = self.pm.classify_bound
        self.assertEqual(c(0.60, 0.00, 0.9), "DISK/DECODE-BOUND")
        self.assertEqual(c(0.00, 0.60, 0.9), "ENCODE-BOUND")
        self.assertEqual(c(0.00, 0.00, 0.9), "GPU-BOUND")
        self.assertEqual(c(0.00, 0.00, 0.1), "BALANCED", "idle workers and healthy queues: waiting on nothing we see")
        self.assertEqual(c(0.00, 0.00, None), "GPU-BOUND", "no worker count: by elimination")
        self.assertEqual(c(0.30, 0.30, 0.9), "DISK/DECODE-BOUND", "a tie goes to the stage upstream")
        self.assertEqual(c(0.10, 0.10, 0.9), "GPU-BOUND", "below the 25% share nothing is called out")

    def _run(self, state, frames=40):
        out, t = [], [0.0]

        def clock():
            return t[0]
        mon = self.pm.PipelineMonitor(20, lambda: state, emit=out.append, clock=clock)
        mon.start()
        for _ in range(frames):
            t[0] += 0.1
            mon.frame_done()
        return out

    def test_the_log_line_carries_the_three_levels_and_the_verdict(self):
        out = self._run({"input": (8, 8), "output": (2, 8), "gpu": (4, 4)})
        self.assertIn("Decode Queue: 8/8 | GPU Queue: 4/4 | Encode Queue: 2/8 | GPU-BOUND", out[0])

    def test_each_bound_is_read_from_the_queue_that_causes_it(self):
        self.assertIn("| DISK/DECODE-BOUND", self._run({"input": (0, 8), "output": (0, 8), "gpu": (1, 4)})[0])
        self.assertIn("| ENCODE-BOUND", self._run({"input": (8, 8), "output": (8, 8), "gpu": (4, 4)})[0])

    def test_without_a_gpu_level_the_existing_line_is_unchanged(self):
        out = self._run({"input": (3, 4), "output": (0, 4)})
        self.assertNotIn("Queue:", out[0])

    def test_levels_are_published_for_the_dashboard_and_cleared_for_a_new_render(self):
        self._run({"input": (5, 8), "output": (1, 8), "gpu": (3, 4)}, frames=20)
        view = self.pm.levels_view()
        self.assertEqual((view["decode"], view["gpu"], view["encode"]),
                         ({"depth": 5, "cap": 8}, {"busy": 3, "cap": 4}, {"depth": 1, "cap": 8}))
        self.assertEqual(view["bound"], "GPU-BOUND")
        self.assertEqual(view["bound_basis"], "workers busy")
        mon = self.pm.PipelineMonitor(20, lambda: {}, emit=lambda _l: None)
        mon.start()
        self.assertIsNone(self.pm.levels_view(), "a new render must not show the last one's queues")

    def test_the_view_is_a_copy(self):
        self._run({"input": (5, 8), "output": (1, 8), "gpu": (3, 4)}, frames=10)
        self.pm.levels_view()["decode"]["depth"] = 99
        self.assertEqual(self.pm.levels_view()["decode"]["depth"], 5)

    def test_processmgr_counts_workers_only_on_paths_that_enter_them(self):
        from roop.ProcessMgr import ProcessMgr

        def q(n, cap):
            import queue
            item = queue.Queue(cap)
            for _ in range(n):
                item.put(0)
            return item
        base = dict(frames_queue=[q(2, 3)], processed_queue=[q(1, 3)], _runtime_scheduler=None,
                    _runtime_read_queue=None, _runtime_write_queue=None, num_threads=4, _gpu_busy={0, 2, 3})
        self.assertEqual(ProcessMgr._pipeline_queue_state(types.SimpleNamespace(**base)),
                         {"input": (2, 3), "output": (1, 3), "gpu": (3, 4)})
        stab = dict(base, _runtime_read_queue=q(1, 2), _runtime_write_queue=q(0, 2))
        self.assertNotIn("gpu", ProcessMgr._pipeline_queue_state(types.SimpleNamespace(**stab)),
                         "the stabilization path never enters a worker: a 0 there would read as idle workers")

    def test_a_real_worker_enter_and_exit_moves_the_count(self):
        from roop.ProcessMgr import ProcessMgr
        mgr = ProcessMgr.__new__(ProcessMgr)
        mgr._gpu_busy = set()
        mgr._runtime_monitor = None
        ProcessMgr._runtime_worker_enter(mgr, 1)
        ProcessMgr._runtime_worker_enter(mgr, 2)
        self.assertEqual(len(mgr._gpu_busy), 2)
        ProcessMgr._runtime_worker_exit(mgr, 1)
        self.assertEqual(mgr._gpu_busy, {2})


class DashboardQueues(unittest.TestCase):
    QUEUES = {"decode": {"depth": 8, "cap": 8}, "gpu": {"busy": 4, "cap": 4}, "encode": {"depth": 2, "cap": 8},
              "unit": "", "bound": "GPU-BOUND", "bound_basis": "workers busy", "frames": 40, "live": True}

    def test_the_console_line(self):
        from roop import terminal_dashboard as td
        self.assertEqual(td.render_queue_line(self.QUEUES, color=False),
                         "Decode Queue: 8/8 | GPU Queue: 4/4 | Encode Queue: 2/8  [GPU-BOUND]")
        enc = dict(self.QUEUES, bound="ENCODE-BOUND", gpu=None, unit="chunks")
        self.assertEqual(td.render_queue_line(enc, color=False),
                         "Decode Queue: 8/8 | Encode Queue: 2/8 chunks  [ENCODE-BOUND]")
        self.assertEqual(td.render_queue_line(None, color=False), "")
        self.assertIn("[bound: measuring]", td.render_queue_line(dict(self.QUEUES, bound=None), color=False))

    def tearDown(self):
        from roop import pipeline_monitor
        pipeline_monitor.clear_levels()           # process-global: a leaked view would appear in another test's snapshot

    def test_log_mode_keeps_the_dashboard_vocabulary(self):
        """Pipes and CI grep these lines: every record starts [Dashboard] or [Telemetry] (test_terminal_dashboard pins it)."""
        from roop import terminal_dashboard as td
        stages = {sid: {"state": "PENDING"} for sid in ("S1", "S2", "S3", "S4")}
        snap = {"updated_at": 0, "stages": stages, "telemetry": {
            "frames": 12, "bottleneck": None, "latency_ms": {c[0]: 1.0 for c in td.LATENCY_COLUMNS},
            "hit_rates": {"smart_skip_roi_reused_pct": None, "stage2_face_free_skipped_pct": None,
                          "vram_allocated_gb": 1.0, "vram_peak_gb": 2.0}, "queues": self.QUEUES}}
        lines = td.render_log_lines(snap)
        self.assertTrue(all(line.startswith(("[Dashboard]", "[Telemetry]")) for line in lines), lines)
        queue_line = next(line for line in lines if " queues " in line)
        self.assertIn("decode_queue=8/8 gpu_queue=4/4 encode_queue=2/8 bound=GPU-BOUND basis=workers busy", queue_line)
        self.assertNotIn("\x1b", queue_line)

    def test_the_snapshot_and_both_renderers_read_the_monitor(self):
        from roop import pipeline_monitor, terminal_dashboard as td
        pipeline_monitor.clear_levels()
        self.assertIsNone(td.TerminalProgressManager._queues(False), "no render, no queues")
        mon = pipeline_monitor.PipelineMonitor(
            10, lambda: {"input": (6, 8), "output": (0, 8), "gpu": (4, 4)}, emit=lambda _l: None)
        mon.start()
        for _ in range(12):
            mon.frame_done()
        q = td.TerminalProgressManager._queues(True)
        self.assertEqual((q["decode"]["depth"], q["gpu"]["busy"], q["bound"], q["live"]), (6, 4, "GPU-BOUND", True))
        self.assertNotIn("updated_at", q, "a wall-clock stamp would make every tick a change")
        snap = {"telemetry": {"latency_ms": {c[0]: 1.0 for c in td.LATENCY_COLUMNS}, "bottleneck": None, "frames": 12,
                              "hit_rates": {"smart_skip_roi_reused_pct": None, "smart_skip_enabled": False,
                                            "stage2_face_free_skipped_pct": None, "vram_allocated_gb": 1.0,
                                            "vram_peak_gb": 2.0, "vram_total_gb": 12.0}, "queues": q}}
        lines = td.render_telemetry_lines(snap, color=False)
        self.assertTrue(any(line.startswith("Decode Queue: 6/8 | GPU Queue: 4/4") for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
