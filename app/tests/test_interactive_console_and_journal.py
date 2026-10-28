"""Stage 13: Interactive Command Console and Journaled Crash Recovery Tests.

Tests:
1. Atomic write-ahead journal persistence (journal.json.tmp -> journal.json)
2. Frame-accurate resume & ffprobe-based truncated tail segment recovery
3. Non-TTY fallback safety for background/CI environments
4. Non-blocking terminal hotkey listener actions ([P], [S], [D], [T], [Q])
5. 4-stage inspection grid debug snapshot creation & tensor stats dump
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import debug_snapshot
from roop import journal
import project_checkpoint as pc
from roop import terminal_dashboard as td
import roop.globals as g


class TestJournalAtomicWrites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test_journal_")
        self.job_dir = self.tmp.name
        self.src_file = os.path.join(self.job_dir, "source.jpg")
        self.tgt_file = os.path.join(self.job_dir, "target.mp4")
        with open(self.src_file, "wb") as f:
            f.write(b"source-content-bytes-1234")
        with open(self.tgt_file, "wb") as f:
            f.write(b"target-video-bytes-5678")

    def tearDown(self):
        self.tmp.cleanup()

    def test_journal_write_is_atomic_and_leaves_no_tmp_file(self):
        payload = {"swap_model": "inswapper", "enhancer": "gpen"}
        segments = [
            {"file": "seg_0001.mp4", "frames": 100, "bytes": 2048, "sha256": "abc1"},
            {"file": "seg_0002.mp4", "frames": 100, "bytes": 2048, "sha256": "abc2"},
        ]
        record = journal.build_journal_record(
            job_id="job_42",
            project_id="proj_42",
            source_path=self.src_file,
            target_path=self.tgt_file,
            payload=payload,
            segments=segments,
            frame_start=0,
            frame_end=500,
            current_frame=200,
            status="IN_PROGRESS",
            seg_dir=self.job_dir,
        )

        out_path = journal.write_journal(self.job_dir, record)
        self.assertTrue(os.path.isfile(out_path))
        self.assertEqual(os.path.basename(out_path), "journal.json")

        # Verify no .tmp files remain
        remaining_files = os.listdir(self.job_dir)
        self.assertNotIn("journal.json.tmp", remaining_files)

        # Verify journal contents
        loaded = journal.read_journal(self.job_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["job_id"], "job_42")
        self.assertEqual(loaded["resume_frame_index"], 200)
        self.assertEqual(loaded["total_verified_frames"], 200)
        self.assertEqual(len(loaded["completed_segments"]), 2)
        self.assertTrue(len(loaded["source_media"]["sha256"]) > 0)
        self.assertTrue(len(loaded["target_media"]["sha256"]) > 0)
        self.assertTrue(len(loaded["recipe_hash"]) > 0)


class TestPartialSegmentRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test_recovery_")
        self.seg_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_intact_segments_preserved(self):
        # Create 2 valid dummy segment files
        seg1 = os.path.join(self.seg_dir, "seg_0001.mp4")
        seg2 = os.path.join(self.seg_dir, "seg_0002.mp4")
        with open(seg1, "wb") as f:
            f.write(b"mp4-data-1")
        with open(seg2, "wb") as f:
            f.write(b"mp4-data-2")

        segments = [
            {"file": "seg_0001.mp4", "frames": 50},
            {"file": "seg_0002.mp4", "frames": 50},
        ]

        # Mock probe_segment_with_ffprobe returning True
        with mock.patch("roop.journal.probe_segment_with_ffprobe", return_value=(True, 50)):
            verified, total_frames, discarded = journal.verify_and_recover_segments(
                self.seg_dir, segments, fps=30.0, discard_truncated_tail=True
            )
            self.assertEqual(len(verified), 2)
            self.assertEqual(total_frames, 100)
            self.assertEqual(len(discarded), 0)

    def test_truncated_tail_file_is_discarded_and_prior_segments_kept(self):
        seg1 = os.path.join(self.seg_dir, "seg_0001.mp4")
        seg2 = os.path.join(self.seg_dir, "seg_0002.mp4")
        tail = os.path.join(self.seg_dir, "seg_0003.mp4")
        for p in (seg1, seg2, tail):
            with open(p, "wb") as f:
                f.write(b"data")

        segments = [
            {"file": "seg_0001.mp4", "frames": 100},
            {"file": "seg_0002.mp4", "frames": 100},
            {"file": "seg_0003.mp4", "frames": 100},
        ]

        # Mock ffprobe: seg1 and seg2 pass, tail fails (corrupted/truncated)
        def mock_probe(path, fps=30.0):
            if "seg_0003" in path:
                return False, 0
            return True, 100

        with mock.patch("roop.journal.probe_segment_with_ffprobe", side_effect=mock_probe):
            verified, total_frames, discarded = journal.verify_and_recover_segments(
                self.seg_dir, segments, fps=30.0, discard_truncated_tail=True
            )
            self.assertEqual(len(verified), 2)
            self.assertEqual(total_frames, 200)
            self.assertEqual(len(discarded), 1)
            self.assertIn("seg_0003.mp4", discarded[0])
            # The truncated tail file must have been deleted from disk
            self.assertFalse(os.path.isfile(tail))
            # The prior segments must remain intact on disk
            self.assertTrue(os.path.isfile(seg1))
            self.assertTrue(os.path.isfile(seg2))


class TestNonTTYFallbackSafety(unittest.TestCase):
    def test_key_listener_does_not_start_in_non_tty(self):
        mgr = td.TerminalProgressManager()
        # When stdin is not a tty (like in pytest or redirected pipe)
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            mgr._mode = "live"
            mgr._start_key_listener()
            # Input thread must remain None
            self.assertIsNone(mgr._input_thread)

    def test_key_listener_does_not_start_in_log_mode(self):
        mgr = td.TerminalProgressManager()
        with mock.patch.object(sys.stdin, "isatty", return_value=True):
            mgr._mode = "log"
            mgr._start_key_listener()
            self.assertIsNone(mgr._input_thread)


class TestInteractiveConsoleHotkeys(unittest.TestCase):
    def setUp(self):
        self.mgr = td.TerminalProgressManager()
        self.mgr._mode = "log"

    def test_handle_pause_resume_gpu(self):
        from roop import procmgr_runtime as pr
        pr.pause_controller.finish()
        g.pause = False

        # First press: pause requested
        msg_pause = self.mgr.handle_key("p")
        self.assertIn("PAUSED", msg_pause)
        snap = pr.pause_controller.snapshot()
        self.assertTrue(snap["requested"])
        self.assertTrue(g.pause)

        # Second press: resume
        msg_resume = self.mgr.handle_key("P")
        self.assertIn("RESUMED", msg_resume)
        snap2 = pr.pause_controller.snapshot()
        self.assertFalse(snap2["requested"])
        self.assertFalse(g.pause)

    def test_handle_toggle_detail(self):
        self.assertFalse(self.mgr._detail_telemetry)
        msg1 = self.mgr.handle_key("d")
        self.assertIn("EXPANDED", msg1)
        self.assertTrue(self.mgr._detail_telemetry)

        msg2 = self.mgr.handle_key("D")
        self.assertIn("COLLAPSED", msg2)
        self.assertFalse(self.mgr._detail_telemetry)

    def test_handle_toggle_throttle(self):
        self.assertFalse(self.mgr._throttle_mode)
        msg1 = self.mgr.handle_key("t")
        self.assertIn("ON", msg1)
        self.assertTrue(self.mgr._throttle_mode)
        self.assertTrue(g.throttle_mode)

        msg2 = self.mgr.handle_key("T")
        self.assertIn("OFF", msg2)
        self.assertFalse(self.mgr._throttle_mode)
        self.assertFalse(g.throttle_mode)

    def test_hotkey_footer_bar_format(self):
        footer = td.render_hotkey_footer(paused=False, throttle=False, detailed=False, color=False)
        self.assertIn("[P] Pause/Resume GPU", footer)
        self.assertIn("[S] Dump Debug Frame", footer)
        self.assertIn("[D] Toggle Kernel Detail", footer)
        self.assertIn("[T] Throttle Mode", footer)
        self.assertIn("[Q] Save Checkpoint & Exit", footer)

    def test_hotkey_footer_bar_active_badges(self):
        footer = td.render_hotkey_footer(paused=True, throttle=True, detailed=True, color=False)
        self.assertIn("[P] Resume GPU", footer)
        self.assertIn("[D] Kernel Detail (ON)", footer)
        self.assertIn("[T] Throttle Mode (ON)", footer)

    def test_telemetry_lines_detailed_expansion(self):
        snap = {
            "telemetry": {
                "latency_ms": {"decode": 5.0, "detect": 15.0, "swap": 30.0, "mask": 10.0, "restore": 20.0, "blend": 5.0, "encode": 15.0, "other": 0.0},
                "bottleneck": {"stage": "swap", "label": "Swap", "share_pct": 30.0, "severity": "yellow"},
                "hit_rates": {"smart_skip_enabled": True, "smart_skip_roi_reused_pct": 20.0, "stage2_face_free_skipped_pct": 10.0, "vram_allocated_gb": 4.5, "vram_peak_gb": 6.0},
            }
        }
        lines_compact = td.render_telemetry_lines(snap, color=False, detailed=False)
        self.assertEqual(len(lines_compact), 2)

        lines_detailed = td.render_telemetry_lines(snap, color=False, detailed=True)
        self.assertGreater(len(lines_detailed), 2)
        joined = " ".join(lines_detailed)
        self.assertIn("Histograms:", joined)
        self.assertIn("Swp:[", joined)


class TestDebugSnapshotEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test_snapshot_")

    def tearDown(self):
        self.tmp.cleanup()

    def test_4stage_inspection_grid_and_json_dump(self):
        orig = np.full((120, 160, 3), 100, dtype=np.uint8)
        mask = np.full((120, 160), 200, dtype=np.uint8)
        raw_swap = np.full((120, 160, 3), 150, dtype=np.uint8)
        restored = np.full((120, 160, 3), 220, dtype=np.uint8)

        png_path, json_path = debug_snapshot.dump_snapshot(
            frame_idx=7,
            original=orig,
            mask=mask,
            raw_swap=raw_swap,
            restored=restored,
            parameters={"model": "inswapper"},
            target_dir=self.tmp.name,
        )

        self.assertTrue(os.path.isfile(png_path))
        self.assertTrue(os.path.isfile(json_path))
        self.assertIn("frame_000007.png", png_path)
        self.assertIn("frame_000007.json", json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["frame_index"], 7)
            self.assertEqual(data["parameters"]["model"], "inswapper")
            self.assertIn("tensor_statistics", data)
            stats = data["tensor_statistics"]
            self.assertEqual(stats["original"]["shape"], [120, 160, 3])
            self.assertEqual(stats["mask"]["shape"], [120, 160])


if __name__ == "__main__":
    unittest.main()
