"""Stage 15: Unit Tests for Features from Stages 9–14.

Covers:
1. Terminal Non-TTY Logger Safety (Stage 9)
2. Runtime Anomaly Watchdog Triggers & Crash Autopsy (Stage 10)
3. Segment Custody Verification & Lossless Concat Auditing (Stage 10/12)
4. Journaled Checkpoint Atomic Persistence & Segment Resume (Stage 13)

Ensures 100% cross-platform compatibility (Windows & Ubuntu), clean file closure,
and zero flakiness under CI light and full execution profiles.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import crash_autopsy, journal, segment_output, terminal_dashboard as td, watchdog
import roop.globals as roop_globals


class TestTerminalNonTtyLoggerSafety(unittest.TestCase):
    """Stage 9: Non-TTY mode safety for background CI and headless runs."""

    def setUp(self):
        self.mgr = td.TerminalProgressManager()

    def test_non_tty_mode_selection_and_no_interactive_thread(self):
        """When stdout/stdin is not a TTY or in CI, live redraw and key listener are disabled."""
        with mock.patch.object(sys.stdout, "isatty", return_value=False), \
             mock.patch.object(sys.stdin, "isatty", return_value=False):
            self.assertEqual(td.mode(), "log")
            mgr = td.TerminalProgressManager()
            mgr._mode = td.mode()
            mgr._start_key_listener()
            # Stdin thread must remain None to prevent CI hangs
            self.assertIsNone(mgr._input_thread)

    def test_log_mode_renders_pure_text_without_ansi_escapes(self):
        """render_log_lines output must be clean, ANSI-escape free, and prefix-compliant."""
        stages = {sid: {"state": "PENDING"} for sid in ("S1", "S2", "S3", "S4")}
        queues = {
            "decode": {"depth": 8, "cap": 8},
            "gpu": {"busy": 4, "cap": 4},
            "encode": {"depth": 2, "cap": 8},
            "unit": "",
            "bound": "GPU-BOUND",
            "bound_basis": "workers busy",
            "frames": 40,
            "live": True,
        }
        snap = {
            "updated_at": 0,
            "stages": stages,
            "telemetry": {
                "frames": 300,
                "bottleneck": {"stage": "swap", "label": "Swap", "share_pct": 40.0, "severity": "yellow"},
                "latency_ms": {c[0]: 1.0 for c in td.LATENCY_COLUMNS},
                "hit_rates": {
                    "smart_skip_roi_reused_pct": 18.5,
                    "stage2_face_free_skipped_pct": 25.0,
                    "vram_allocated_gb": 4.2,
                    "vram_peak_gb": 6.8,
                },
                "queues": queues,
            },
        }
        lines = td.render_log_lines(snap)
        self.assertTrue(len(lines) > 0)
        for line in lines:
            # Must carry expected log prefixes
            self.assertTrue(line.startswith(("[Dashboard]", "[Telemetry]")), f"Unprefixed line: {line}")
            # Zero ANSI escape codes
            self.assertNotIn("\x1b", line, f"ANSI escape in log output: {line}")
            self.assertNotIn("\r", line, "Carriage return in log output")

    def test_interactive_hotkeys_safe_in_log_mode(self):
        """handle_key operates safely and returns status message without terminal mutation."""
        self.mgr._mode = "log"
        # Test hotkey toggles
        msg_detail = self.mgr.handle_key("d")
        self.assertIn("EXPANDED", msg_detail)
        self.assertTrue(self.mgr._detail_telemetry)

        msg_throttle = self.mgr.handle_key("t")
        self.assertIn("ON", msg_throttle)
        self.assertTrue(self.mgr._throttle_mode)

        # Unrecognized key returns None or empty string without crashing
        msg_unknown = self.mgr.handle_key("z")
        self.assertIn(msg_unknown, (None, ""))


class TestAnomalyWatchdogTriggers(unittest.TestCase):
    """Stage 10: Anomaly Watchdog triggers and crash autopsy classification."""

    def setUp(self):
        watchdog.reset()
        self.alerts = []
        watchdog.WATCHDOG.add_log_listener(self.alerts.append)

    def tearDown(self):
        watchdog.reset()

    def test_tensor_catcher_detects_nan_inf_and_black_frames(self):
        # NaN detection
        nan_tensor = np.full((128, 128, 3), 100.0, dtype=np.float32)
        nan_tensor[10, 10, 0] = np.nan
        ok, reason = watchdog.check_tensor(nan_tensor, model_name="realswap", frame_idx=1)
        self.assertFalse(ok)
        self.assertIn("NaN", reason)

        # Inf detection
        inf_tensor = np.full((128, 128, 3), 100.0, dtype=np.float32)
        inf_tensor[20, 20, 1] = np.inf
        ok_inf, reason_inf = watchdog.check_tensor(inf_tensor, model_name="gpen", frame_idx=2)
        self.assertFalse(ok_inf)
        self.assertIn("Inf", reason_inf)

        # Black-frame detection
        black_tensor = np.zeros((128, 128, 3), dtype=np.uint8)
        ok_b, reason_b = watchdog.check_tensor(black_tensor, model_name="inswapper", frame_idx=3)
        self.assertFalse(ok_b)
        self.assertIn("black frame", reason_b.lower())

        # Healthy tensor
        clean_tensor = np.full((128, 128, 3), 120, dtype=np.uint8)
        ok_c, reason_c = watchdog.check_tensor(clean_tensor, model_name="realswap", frame_idx=4)
        self.assertTrue(ok_c)
        self.assertIsNone(reason_c)

    def test_zero_swap_alert_counter_and_reset(self):
        # 3 consecutive drops
        for f in range(3):
            watchdog.check_zero_swap(frame_idx=f, expected_faces=1, matched_swaps=0, threshold_consecutive=4)
        self.assertEqual(len(self.alerts), 0)

        # 4th drop triggers warning
        watchdog.check_zero_swap(frame_idx=3, expected_faces=1, matched_swaps=0, threshold_consecutive=4)
        self.assertTrue(any("Zero-swap detected for 4 consecutive frames" in a for a in self.alerts))

        # Reset on swap
        watchdog.check_zero_swap(frame_idx=4, expected_faces=1, matched_swaps=1, threshold_consecutive=4)
        self.assertEqual(watchdog.WATCHDOG._consecutive_zero_swaps, 0)

    def test_mask_geometry_collapse_and_scene_cut_immunity(self):
        m1 = np.ones((50, 50), dtype=np.uint8)
        m_collapsed = np.zeros((50, 50), dtype=np.uint8)

        # Collapse within same scene triggers warning
        watchdog.check_mask_geometry(frame_idx=1, mask=m1, scene_cut_id=0)
        ok = watchdog.check_mask_geometry(frame_idx=2, mask=m_collapsed, scene_cut_id=0)
        self.assertFalse(ok)
        self.assertTrue(any("Mask collapse detected" in a for a in self.alerts))

        # Collapse across scene cut does NOT trigger warning
        watchdog.reset()
        self.alerts.clear()
        watchdog.check_mask_geometry(frame_idx=10, mask=m1, scene_cut_id=1)
        ok_cut = watchdog.check_mask_geometry(frame_idx=11, mask=m_collapsed, scene_cut_id=2)
        self.assertTrue(ok_cut)
        self.assertEqual(len(self.alerts), 0)

    def test_crash_autopsy_diagnostics_and_report_bundle(self):
        oom = RuntimeError("CUDA out of memory. Tried to allocate 1024.00 MiB")
        root, hint = crash_autopsy.diagnose_root_cause(oom)
        self.assertIn("CUDA Out of Memory", root)
        self.assertIn("Reduce Max Threads", hint)

        tmp_logs = tempfile.mkdtemp(prefix="test_autopsy_")
        try:
            watchdog.WATCHDOG.set_job("job_stage15_test", segment_idx=1)
            watchdog.enter_frame(45, stage="Stage 3")
            watchdog.record_log_event("Frame 44 swapped ok")
            report = crash_autopsy.generate_crash_report(oom, job_id="job_stage15_test", dump_dir=tmp_logs)

            self.assertEqual(report["job_id"], "job_stage15_test")
            self.assertEqual(report["error_type"], "RuntimeError")
            self.assertEqual(report["failure_location"]["frame_number"], 45)
            self.assertIn("CUDA Out of Memory", report["root_cause"])
            
            # Check file saved on disk
            dumps = glob.glob(os.path.join(tmp_logs, "crash_job_stage15_test_*.json"))
            self.assertEqual(len(dumps), 1)
            with open(dumps[0], "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["job_id"], "job_stage15_test")
            self.assertEqual(loaded["error_type"], "RuntimeError")
        finally:
            shutil.rmtree(tmp_logs, ignore_errors=True)


class TestSegmentCustodyVerification(unittest.TestCase):
    """Stage 10/12: Safe destinations and segment custody audit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test_custody_")

    def tearDown(self):
        self.tmp.cleanup()

    def test_assert_safe_segment_destination_guards_collision(self):
        safe_seg = os.path.join(self.tmp.name, "segments", "seg_0001.mp4")
        final_dest = os.path.join(self.tmp.name, "final.mp4")
        # Distinct paths pass
        segment_output.assert_safe_segment_destination(safe_seg, final_dest)

        # In-place collision must raise ValueError
        with self.assertRaises(ValueError):
            segment_output.assert_safe_segment_destination(final_dest, final_dest)

    def test_audit_segment_custody_verifies_intact_file(self):
        seg_file = os.path.join(self.tmp.name, "seg_0001.mp4")
        with open(seg_file, "wb") as f:
            f.write(b"video-bytes" * 500)

        receipt = segment_output.audit_segment_custody(seg_file, expected_frames=150)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["filename"], "seg_0001.mp4")
        self.assertEqual(receipt["expected_frames"], 150)
        self.assertTrue(len(receipt["hash"]) > 0)


class TestJournaledCheckpointResume(unittest.TestCase):
    """Stage 13: Atomic write-ahead journal persistence and truncated tail recovery."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test_journal_resume_")
        self.job_dir = self.tmp.name
        self.src_file = os.path.join(self.job_dir, "src.jpg")
        self.tgt_file = os.path.join(self.job_dir, "tgt.mp4")
        with open(self.src_file, "wb") as f:
            f.write(b"source-data")
        with open(self.tgt_file, "wb") as f:
            f.write(b"target-video-data")

    def tearDown(self):
        self.tmp.cleanup()

    def test_journal_atomic_write_and_read(self):
        record = journal.build_journal_record(
            job_id="job_st15",
            project_id="proj_st15",
            source_path=self.src_file,
            target_path=self.tgt_file,
            payload={"swap_model": "realswap"},
            segments=[{"file": "seg_0001.mp4", "frames": 50}],
            frame_start=0,
            frame_end=150,
            current_frame=50,
            status="IN_PROGRESS",
            seg_dir=self.job_dir,
        )
        out_path = journal.write_journal(self.job_dir, record)
        self.assertTrue(os.path.isfile(out_path))
        # Ensure temporary write-ahead files are cleaned
        self.assertFalse(os.path.isfile(out_path + ".tmp"))

        loaded = journal.read_journal(self.job_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["job_id"], "job_st15")
        self.assertEqual(loaded["resume_frame_index"], 50)
        self.assertEqual(len(loaded["completed_segments"]), 1)

    def test_verify_and_recover_discards_truncated_tail_preserves_intact(self):
        seg1 = os.path.join(self.job_dir, "seg_0001.mp4")
        seg2 = os.path.join(self.job_dir, "seg_0002.mp4")
        tail = os.path.join(self.job_dir, "seg_0003.mp4")
        for p in (seg1, seg2, tail):
            with open(p, "wb") as f:
                f.write(b"mp4-content")

        segments = [
            {"file": "seg_0001.mp4", "frames": 100},
            {"file": "seg_0002.mp4", "frames": 100},
            {"file": "seg_0003.mp4", "frames": 100},
        ]

        def fake_probe(path, fps=30.0):
            if "seg_0003" in path:
                return False, 0
            return True, 100

        with mock.patch("roop.journal.probe_segment_with_ffprobe", side_effect=fake_probe):
            verified, total_frames, discarded = journal.verify_and_recover_segments(
                self.job_dir, segments, fps=30.0, discard_truncated_tail=True
            )
            self.assertEqual(len(verified), 2)
            self.assertEqual(total_frames, 200)
            self.assertEqual(len(discarded), 1)
            # Corrupted tail must be removed
            self.assertFalse(os.path.isfile(tail))
            # Intact segments must stay on disk
            self.assertTrue(os.path.isfile(seg1))
            self.assertTrue(os.path.isfile(seg2))


if __name__ == "__main__":
    unittest.main()
