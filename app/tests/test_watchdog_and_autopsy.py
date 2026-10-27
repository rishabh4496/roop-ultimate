"""Stage 10: Tests for Runtime Anomaly Watchdog & Crash Autopsy Engine."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from roop import crash_autopsy, segment_output, watchdog
from roop.processors.frame.face_enhancer import sanitize_frame_output


class TestWatchdogTensorCatcher(unittest.TestCase):
    """1. NaN / Inf / Black-Frame Tensor Catcher."""

    def setUp(self):
        watchdog.reset()
        self.alerts = []
        watchdog.WATCHDOG.add_log_listener(lambda msg: self.alerts.append(msg))

    def tearDown(self):
        watchdog.reset()

    def test_nan_tensor_detected(self):
        tensor = np.ones((512, 512, 3), dtype=np.float32)
        tensor[100, 100, 0] = np.nan
        ok, issue = watchdog.check_tensor(tensor, model_name="realswap", frame_idx=42, bbox=[10, 20, 100, 120])
        self.assertFalse(ok)
        self.assertIn("NaN", issue)
        self.assertTrue(any("[BUG-DETECTED]" in a and "realswap" in a and "frame 42" in a for a in self.alerts))

    def test_inf_tensor_detected(self):
        tensor = np.ones((512, 512, 3), dtype=np.float32)
        tensor[50, 50, 1] = np.inf
        ok, issue = watchdog.check_tensor(tensor, model_name="gpen_256_pro", frame_idx=105, bbox=[5, 5, 80, 80])
        self.assertFalse(ok)
        self.assertIn("Inf", issue)
        self.assertTrue(any("[BUG-DETECTED]" in a and "gpen_256_pro" in a and "frame 105" in a for a in self.alerts))

    def test_black_frame_tensor_detected(self):
        tensor = np.zeros((512, 512, 3), dtype=np.uint8)
        ok, issue = watchdog.check_tensor(tensor, model_name="inswapper", frame_idx=7, bbox=[0, 0, 512, 512])
        self.assertFalse(ok)
        self.assertIn("All-zero black frame", issue)
        self.assertTrue(any("[BUG-DETECTED]" in a and "black frame" in a.lower() for a in self.alerts))

    def test_healthy_tensor_passes(self):
        tensor = np.full((512, 512, 3), 128, dtype=np.uint8)
        ok, issue = watchdog.check_tensor(tensor, model_name="realswap", frame_idx=1)
        self.assertTrue(ok)
        self.assertIsNone(issue)
        self.assertEqual(len(self.alerts), 0)

    def test_sanitize_frame_output_catches_black_frame(self):
        orig = np.full((256, 256, 3), 100, dtype=np.uint8)
        corrupted = np.zeros((256, 256, 3), dtype=np.uint8)
        res = sanitize_frame_output(corrupted, orig, stage="face_enhancer")
        # Should return original frame when corrupted
        np.testing.assert_array_equal(res, orig)
        self.assertTrue(any("[BUG-DETECTED]" in a and "face_enhancer" in a for a in self.alerts))


class TestWatchdogZeroSwapAlert(unittest.TestCase):
    """2. Zero-Swap / Identity Drop Alert."""

    def setUp(self):
        watchdog.reset()
        self.alerts = []
        watchdog.WATCHDOG.add_log_listener(lambda msg: self.alerts.append(msg))

    def tearDown(self):
        watchdog.reset()

    def test_consecutive_zero_swap_triggers_warning(self):
        # 4 consecutive frames with expected=1, matched=0 -> below threshold 5
        for f in range(4):
            ok = watchdog.check_zero_swap(
                frame_idx=f, expected_faces=1, matched_swaps=0,
                refusal_reason="Similarity score 0.28 fell below threshold 0.45",
                threshold_consecutive=5,
            )
            self.assertTrue(ok)
        self.assertEqual(len(self.alerts), 0)

        # 5th frame hits threshold 5
        ok = watchdog.check_zero_swap(
            frame_idx=4, expected_faces=1, matched_swaps=0,
            refusal_reason="Similarity score 0.28 fell below threshold 0.45",
            threshold_consecutive=5,
        )
        self.assertFalse(ok)
        self.assertEqual(len(self.alerts), 1)
        self.assertIn("[WARN]", self.alerts[0])
        self.assertIn("Zero-swap detected for 5 consecutive frames", self.alerts[0])
        self.assertIn("Similarity score 0.28 fell below threshold 0.45", self.alerts[0])

    def test_successful_swap_resets_counter(self):
        for f in range(4):
            watchdog.check_zero_swap(frame_idx=f, expected_faces=1, matched_swaps=0, threshold_consecutive=5)
        # 5th frame succeeds
        watchdog.check_zero_swap(frame_idx=4, expected_faces=1, matched_swaps=1, threshold_consecutive=5)
        # 6th frame has 0 swaps again -> should start counting from 1, no alert
        watchdog.check_zero_swap(frame_idx=5, expected_faces=1, matched_swaps=0, threshold_consecutive=5)
        self.assertEqual(len(self.alerts), 0)


class TestWatchdogMaskCollapseAndJump(unittest.TestCase):
    """3. Mask Collapse & Seam Jump Detector."""

    def setUp(self):
        watchdog.reset()
        self.alerts = []
        watchdog.WATCHDOG.add_log_listener(lambda msg: self.alerts.append(msg))

    def tearDown(self):
        watchdog.reset()

    def test_mask_collapse_to_zero(self):
        mask1 = np.ones((100, 100), dtype=np.uint8)  # area = 10,000 px
        mask2 = np.zeros((100, 100), dtype=np.uint8)  # area = 0 px
        watchdog.check_mask_geometry(frame_idx=10, mask=mask1, scene_cut_id=1)
        ok = watchdog.check_mask_geometry(frame_idx=11, mask=mask2, scene_cut_id=1)
        self.assertFalse(ok)
        self.assertTrue(any("[WARN] Mask collapse detected" in a and "frame 11" in a for a in self.alerts))

    def test_mask_seam_jump_above_70_percent(self):
        mask1 = np.ones((100, 100), dtype=np.uint8)  # area = 10,000 px
        # Fluctuation to 2,000 px (80% drop)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[:20, :] = 1
        watchdog.check_mask_geometry(frame_idx=10, mask=mask1, scene_cut_id=1)
        ok = watchdog.check_mask_geometry(frame_idx=11, mask=mask2, scene_cut_id=1, jump_threshold=0.70)
        self.assertFalse(ok)
        self.assertTrue(any("[WARN] Mask seam jump detected" in a and "80.0%" in a for a in self.alerts))

    def test_scene_cut_resets_baseline_without_warning(self):
        mask1 = np.ones((100, 100), dtype=np.uint8)  # shot 1
        mask2 = np.zeros((100, 100), dtype=np.uint8)  # shot 2 begins with 0px or small mask
        watchdog.check_mask_geometry(frame_idx=24, mask=mask1, scene_cut_id=1)
        # Frame 25 is in shot 2
        ok = watchdog.check_mask_geometry(frame_idx=25, mask=mask2, scene_cut_id=2)
        self.assertTrue(ok)
        self.assertEqual(len(self.alerts), 0)


class TestWatchdogZeroCopyAndVram(unittest.TestCase):
    """4. Zero-Copy Fallback & VRAM Pressure Watchdog."""

    def setUp(self):
        watchdog.reset()
        self.alerts = []
        watchdog.WATCHDOG.add_log_listener(lambda msg: self.alerts.append(msg))

    def tearDown(self):
        watchdog.reset()

    def test_zero_copy_broken_alert(self):
        watchdog.notify_zero_copy_broken("NVDEC-DLPack", reason="CUDA IPC descriptor failed")
        self.assertTrue(any("[PERF-WARN] Zero-copy broken at stage: NVDEC-DLPack" in a for a in self.alerts))

    def test_vram_pressure_alert(self):
        # Test directly via mock threshold
        ratio = watchdog.check_vram_pressure(threshold_ratio=0.0)  # threshold 0.0 forces alert if CUDA available
        if ratio is not None:
            self.assertTrue(any("[WARN] High VRAM pressure" in a for a in self.alerts))


class TestSegmentCustodyAuditor(unittest.TestCase):
    """5. Segment Custody & Lossless Concat Verification Auditor."""

    def setUp(self):
        watchdog.reset()
        self.alerts = []
        watchdog.WATCHDOG.add_log_listener(lambda msg: self.alerts.append(msg))
        self.tmp_dir = tempfile.mkdtemp(prefix="stage10_custody_")

    def tearDown(self):
        watchdog.reset()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_assert_safe_segment_destination(self):
        seg_dest = os.path.join(self.tmp_dir, ".segments", "job1", "seg_0001.mp4")
        final_dest = os.path.join(self.tmp_dir, "output.mp4")
        # Distinct paths: valid
        segment_output.assert_safe_segment_destination(seg_dest, final_dest)

        # Collision: raises ValueError
        with self.assertRaises(ValueError):
            segment_output.assert_safe_segment_destination(final_dest, final_dest)

    def test_audit_segment_custody_receipt(self):
        seg_file = os.path.join(self.tmp_dir, "seg_0002.mp4")
        with open(seg_file, "wb") as f:
            f.write(b"\x00" * 1024 * 50)  # 50 KB dummy file

        res = segment_output.audit_segment_custody(seg_file, expected_frames=600)
        self.assertTrue(res["ok"])
        self.assertEqual(res["filename"], "seg_0002.mp4")
        self.assertEqual(res["expected_frames"], 600)
        self.assertTrue(any("[CUSTODY OK] seg_0002.mp4" in a and "Hash/Path Verified" in a for a in self.alerts))

    def test_join_segments_captures_ffmpeg_stderr_on_failure(self):
        # Provide invalid manifest path to force ffmpeg error
        bad_manifest = os.path.join(self.tmp_dir, "nonexistent.txt")
        out_video = os.path.join(self.tmp_dir, "out.mp4")
        with self.assertRaises(RuntimeError) as ctx:
            segment_output.join_segments([], out_video, bad_manifest)
        self.assertIn("cannot join segments", str(ctx.exception))


class TestCrashAutopsyReport(unittest.TestCase):
    """6. Structured Crash Autopsy Report Generator."""

    def setUp(self):
        watchdog.reset()
        self.alerts = []
        watchdog.WATCHDOG.add_log_listener(lambda msg: self.alerts.append(msg))
        self.tmp_logs = tempfile.mkdtemp(prefix="stage10_logs_")

    def tearDown(self):
        watchdog.reset()
        shutil.rmtree(self.tmp_logs, ignore_errors=True)

    def test_extract_failure_location(self):
        try:
            raise RuntimeError("Test pipeline failure")
        except RuntimeError as e:
            loc = crash_autopsy.extract_failure_location(e)
            self.assertEqual(loc["exception_type"], "RuntimeError")
            self.assertIn("test_watchdog_and_autopsy.py", loc["file_line"])

    def test_diagnose_cuda_oom(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        root, hint = crash_autopsy.diagnose_root_cause(exc)
        self.assertIn("CUDA Out of Memory", root)
        self.assertIn("Reduce Max Threads", hint)

    def test_diagnose_tensorrt_mismatch(self):
        exc = RuntimeError("The engine plan file is not compatible with this version of TensorRT")
        root, hint = crash_autopsy.diagnose_root_cause(exc)
        self.assertIn("TensorRT Engine Plan was built for a different", root)
        self.assertIn("Purge & Rebuild TRT Cache", hint)

    def test_diagnose_onnx_runtime_error(self):
        exc = RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned")
        root, hint = crash_autopsy.diagnose_root_cause(exc)
        self.assertIn("ONNX Runtime Execution Provider", root)
        self.assertIn("CUDAExecutionProvider", hint)

    def test_diagnose_ffmpeg_broken_pipe(self):
        exc = BrokenPipeError("Broken pipe")
        root, hint = crash_autopsy.diagnose_root_cause(exc)
        self.assertIn("FFmpeg NVENC", root)
        self.assertIn("NVENC hardware sessions", hint)

    def test_diagnose_disk_full(self):
        exc = PermissionError("Permission denied: 'out.mp4'")
        root, hint = crash_autopsy.diagnose_root_cause(exc)
        self.assertIn("Destination drive volume is full or operating system denied", root)

    def test_generate_crash_report_creates_json_bundle(self):
        watchdog.WATCHDOG.set_job("job_autopsy_test", segment_idx=2)
        watchdog.enter_frame(120, stage="Stage 3 (Swap / Restoration)")
        watchdog.update_breadcrumb(
            model_name="realswap",
            input_shape=(1, 3, 512, 512),
            input_dtype="float32",
            bbox=[100, 80, 350, 320],
        )
        watchdog.record_log_event("Frame 119 swapped in 45ms")
        watchdog.record_log_event("Processing frame 120 / 600")

        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
        except RuntimeError as e:
            report = crash_autopsy.generate_crash_report(
                e, job_id="job_autopsy_test", dump_dir=self.tmp_logs
            )

        # Check in-memory report dict
        self.assertEqual(report["job_id"], "job_autopsy_test")
        self.assertEqual(report["failure_location"]["frame_number"], 120)
        self.assertEqual(report["failure_location"]["segment_index"], 2)
        self.assertEqual(report["active_state"]["model_name"], "realswap")
        self.assertIn("CUDA Out of Memory", report["root_cause"])
        self.assertIn("Reduce Max Threads", report["fix_hint"])
        self.assertTrue(len(report["last_50_logs"]) >= 2)

        # Check saved JSON dump on disk
        dumps = glob.glob(os.path.join(self.tmp_logs, "crash_job_autopsy_test_*.json"))
        self.assertEqual(len(dumps), 1)
        with open(dumps[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["job_id"], "job_autopsy_test")
        self.assertEqual(data["error_type"], "RuntimeError")
        self.assertIn("realswap", data["active_state"]["model_name"])

        # Check broadcast to listeners
        self.assertTrue(any("[CRASH AUTOPSY]" in a and "RuntimeError" in a for a in self.alerts))


if __name__ == "__main__":
    unittest.main()
