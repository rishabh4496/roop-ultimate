"""Architecture guard for ProcessMgr's batch orchestration boundary."""

import ast
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
PROCESS_MGR = APP / "roop" / "ProcessMgr.py"
BATCH = APP / "roop" / "procmgr_batch.py"
STABILIZATION = APP / "roop" / "procmgr_stabilization.py"


class ProcessMgrBoundaryTest(unittest.TestCase):
    def test_batch_orchestration_lives_in_its_own_mixin(self):
        process_source = PROCESS_MGR.read_text(encoding="utf-8")
        batch_source = BATCH.read_text(encoding="utf-8")
        process_tree = ast.parse(process_source, filename=str(PROCESS_MGR))
        batch_tree = ast.parse(batch_source, filename=str(BATCH))

        process_methods = {
            node.name
            for node in ast.walk(process_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        batch_methods = {
            node.name
            for node in ast.walk(batch_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("run_batch_inmem", process_methods)
        self.assertIn("run_batch_inmem", batch_methods)
        self.assertIn("BatchProcessingMixin", process_source)

    def test_stabilization_policy_lives_in_its_own_mixin(self):
        process_source = PROCESS_MGR.read_text(encoding="utf-8")
        stabilization_source = STABILIZATION.read_text(encoding="utf-8")
        process_tree = ast.parse(process_source, filename=str(PROCESS_MGR))
        stabilization_tree = ast.parse(stabilization_source,
                                       filename=str(STABILIZATION))
        process_methods = {
            node.name for node in ast.walk(process_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        stabilization_methods = {
            node.name for node in ast.walk(stabilization_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in ("_stab_warmup_frames", "_stab_live_chunks",
                     "_default_stab_chunk_mb", "_stab_parallel_geometry"):
            self.assertNotIn(name, process_methods)
            self.assertIn(name, stabilization_methods)
        self.assertIn("StabilizationSchedulingMixin", process_source)

    def test_extracted_module_has_a_single_runtime_entry_point(self):
        tree = ast.parse(BATCH.read_text(encoding="utf-8"), filename=str(BATCH))
        mixin = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef)
                     and node.name == "BatchProcessingMixin")
        methods = [node.name for node in mixin.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertEqual(["run_batch_inmem"], methods)


if __name__ == "__main__":
    unittest.main()
