from __future__ import annotations

import hashlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "audit"))
from isolation import AuditError
from local_models import discover, graph_metadata, license_register, verify_registry


class LocalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(__file__).absolute().parents[1] / ".runtime/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        self.weight = self.root / "inswapper_candidate.pth"
        self.weight.write_bytes(b"local artifact, intentionally not executable pickle")
        self.scopes = {"test": self.root}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_hashes_match_actual_bytes_and_consumer_verifies(self) -> None:
        registry = discover(self.scopes)
        model = registry["models"][0]
        self.assertEqual(model["sha256"], hashlib.sha256(self.weight.read_bytes()).hexdigest())
        self.assertEqual(model["state"], "VERIFIED_LOCAL")
        self.assertEqual(model["inference_state"], "UNKNOWN")
        self.assertEqual(model["commercial_permission"], "UNKNOWN")
        self.assertEqual(model["graph"]["state"], "UNKNOWN")
        self.assertTrue(verify_registry(registry, self.scopes)["ok"])

    def test_registry_consumer_detects_mutation(self) -> None:
        registry = discover(self.scopes)
        self.weight.write_bytes(b"different weights")
        self.assertFalse(verify_registry(registry, self.scopes)["ok"])

    def test_license_register_does_not_invent_permissions_or_downloads(self) -> None:
        rows = list(csv.DictReader(io.StringIO(license_register(discover(self.scopes)))))
        self.assertEqual(rows[0]["verification_state"], "VERIFIED_LOCAL")
        for row in rows:
            self.assertEqual(row["commercial_permission"], "UNKNOWN")
            self.assertEqual(row["license_state"], "UNKNOWN")
            self.assertEqual(row["download_url_state"], "UNKNOWN")
            self.assertEqual(row["inference_state"], "UNKNOWN")
        self.assertEqual(next(r for r in rows if r["artifact_name"] == "FLAME")["verification_state"], "CANDIDATE")

    def test_discovery_halts_on_old_hash_drift(self) -> None:
        expected = {os.path.normcase(str(self.weight.absolute())): "0" * 64}
        with self.assertRaises(AuditError):
            discover(self.scopes, expected)

    def test_dependency_and_output_trees_are_not_misidentified_as_models(self) -> None:
        for name in ("env", "node_modules", "output", "roop_3d", "site-packages"):
            directory = self.root / name
            directory.mkdir()
            (directory / "unrelated.pth").write_bytes(b"not a weight")
        (self.root / "timing_cache.bin").write_bytes(b"not a weight")
        self.assertEqual(len(discover(self.scopes)["models"]), 1)

    def test_model_locator_cannot_escape_its_scope(self) -> None:
        registry = discover(self.scopes)
        registry["models"][0]["locator"]["path"] = "../outside.pth"
        self.assertFalse(verify_registry(registry, self.scopes)["ok"])

    def test_empty_registry_fails_closed(self) -> None:
        with self.assertRaises(AuditError):
            verify_registry({"models": []}, self.scopes)

    def test_discovery_does_not_write(self) -> None:
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        discover(self.scopes)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_real_onnx_graph_shapes_and_types_are_read(self) -> None:
        import onnx
        source = onnx.helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, "height", "width"])
        target = onnx.helper.make_tensor_value_info("copy", onnx.TensorProto.FLOAT, [1, 3, "height", "width"])
        graph = onnx.helper.make_graph([onnx.helper.make_node("Identity", ["image"], ["copy"])], "test", [source], [target])
        model = onnx.helper.make_model(graph)
        path = self.root / "actual.onnx"
        onnx.save_model(model, str(path))
        result = graph_metadata(path)
        self.assertEqual(result["state"], "VERIFIED_LOCAL")
        self.assertEqual(result["inputs"], [{"name": "image", "dtype": "FLOAT", "shape": [1, 3, "height", "width"]}])
        self.assertEqual(result["outputs"][0]["name"], "copy")

    def test_corrupt_graph_does_not_get_verified_shapes(self) -> None:
        path = self.root / "broken.onnx"
        path.write_bytes(b"not an ONNX protobuf")
        self.assertEqual(graph_metadata(path)["state"], "UNKNOWN")

    def test_external_tensor_paths_are_reported_without_being_read(self) -> None:
        import onnx
        tensor = onnx.TensorProto()
        tensor.name = "external"
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.append(1)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key, entry.value = "location", "../not_present.bin"
        output = onnx.helper.make_tensor_value_info("copy", onnx.TensorProto.FLOAT, [1])
        graph = onnx.helper.make_graph([onnx.helper.make_node("Identity", ["external"], ["copy"])],
                                       "external-test", [], [output], [tensor])
        path = self.root / "external.onnx"
        path.write_bytes(onnx.helper.make_model(graph).SerializeToString())
        result = graph_metadata(path)
        self.assertEqual(result["state"], "VERIFIED_LOCAL")
        self.assertEqual(result["external_data"][0]["entries"]["location"], "../not_present.bin")
        self.assertFalse((self.root.parent / "not_present.bin").exists())


if __name__ == "__main__":
    unittest.main()
