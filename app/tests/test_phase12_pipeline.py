"""Dependency-light regression checks for the Phase 12 scheduling contract."""
import os
import unittest
from pathlib import Path


APP = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PM = (APP / "roop" / "ProcessMgr.py").read_text(encoding="utf-8")
OPT = (APP / "roop" / "runtime_optimizer.py").read_text(encoding="utf-8")
MASK = (APP / "roop" / "procmgr_masking.py").read_text(encoding="utf-8")
BENCH = (APP / "tests" / "phase12_benchmark.py").read_text(encoding="utf-8")


class Phase12SchedulingContract(unittest.TestCase):
    def test_small_vram_does_not_force_end_to_end_one_worker(self):
        self.assertIn("stabilization_workers = worker if workload.stabilization_enabled",
                      OPT)
        self.assertNotIn("if self._runtime_stab_small and threads > 1:", PM)

    def test_stabilization_geometry_owns_chunk_sizing(self):
        body = PM.split("def _run_stab_parallel", 1)[1].split("def update_progress", 1)[0]
        self.assertIn("if explicit_chunk > 0:", body)
        self.assertIn("CHUNK = max(block, (explicit_chunk // block) * block)", body)
        self.assertNotIn("int(runtime_chunk)", body)
        self.assertIn("_stab_parallel_geometry(threads)", PM)
        self.assertIn("explicit_chunk // block", body)

    def test_retry_gate_is_conservative(self):
        body = PM.split("def retry_rotated", 1)[1].split("def _expression_restorer", 1)[0]
        self.assertIn("retry_detected_faces", body)
        self.assertIn("if detected:", body)
        self.assertIn("if not any(face_rotation_action", body)
        self.assertIn("empty detection", body)

    def test_writer_keeps_absolute_frame_order(self):
        body = PM.split("def _run_stab_parallel", 1)[1].split("def update_progress", 1)[0]
        self.assertRegex(body, r"for gi in range\(cs, cs \+ clen\)")
        self.assertIn("res.pop(gi, None)", body)

    def test_mask_compositor_skips_identity_resize(self):
        body = MASK.split("def _composite_mask", 1)[1].split("def ", 1)[0]
        self.assertIn("if img_mask.shape[:2] != target.shape[:2]", body)
        self.assertIn("np.multiply(result, inv_mask, out=result)", body)


class Phase12BenchmarkContract(unittest.TestCase):
    def test_matrix_has_requested_ab_arms(self):
        for arm in ("baseline", "stabilization_on", "mask_on", "color_on",
                    "postprocess_heavy"):
            self.assertIn('"name": "%s"' % arm, BENCH)

    def test_report_keeps_targets_separate_and_has_required_metrics(self):
        self.assertIn('TARGETS = ("RTX 3060", "RTX 4070")', BENCH)
        for field in ("baseline_fps", "final_fps", "improvement_pct",
                      "peak_vram_mb", "average_vram_mb",
                      "cpu_utilization_pct", "gpu_utilization_pct",
                      "decode_throughput_fps", "inference_throughput_fps",
                      "enhancement_throughput_fps", "encode_throughput_fps",
                      "latency_ms", "stability", "output_quality"):
            self.assertIn('"%s"' % field, BENCH)

    def test_unavailable_target_is_pending(self):
        self.assertIn('report["status"] = "pending"', BENCH)
        self.assertIn('requested GPU is unavailable', BENCH)


if __name__ == "__main__":
    unittest.main()
