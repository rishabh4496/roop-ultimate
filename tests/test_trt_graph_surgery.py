from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
module = importlib.import_module("roop.trt_graph_surgery")


class _FakeLayer:
    def __init__(self, name):
        self.name = name
        self.num_outputs = 1
        self.precision = None
        self.output_types = {}

    def set_output_type(self, index, dtype):
        self.output_types[index] = dtype


class _FakeNetwork:
    def __init__(self, *layers):
        self.layers = list(layers)
        self.num_layers = len(self.layers)

    def get_layer(self, index):
        return self.layers[index]


class _FakeTrt:
    float32 = "float32"
    float16 = "float16"


class GraphSurgeryTests(unittest.TestCase):
    def test_activation_metrics_blacklist_only_layers_below_cosine_threshold(self):
        report = module.compare_activation_outputs(
            {"good": np.ones((4,), dtype=np.float32), "bad": np.ones((4,), dtype=np.float32)},
            {"good": np.ones((4,), dtype=np.float32), "bad": -np.ones((4,), dtype=np.float32)},
            tensor_to_node={"good": "node_good", "bad": "node_bad"},
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.blacklisted_nodes, ("node_bad",))
        self.assertEqual(next(item for item in report.metrics if item.tensor_name == "good").mae, 0.0)

    def test_precision_application_sets_layer_and_output_types(self):
        network = _FakeNetwork(_FakeLayer("norm"), _FakeLayer("projection"))
        annotations = (
            module.LayerPrecisionAnnotation("norm", "LayerNormalization", "fp32", "fp32", "norm", False),
            module.LayerPrecisionAnnotation("projection", "Conv", "fp16", "fp16", "output", True),
        )
        applied = module.apply_layer_precision(network, _FakeTrt, annotations)
        self.assertEqual(applied, ("norm", "projection"))
        self.assertEqual(network.layers[0].precision, "float32")
        self.assertEqual(network.layers[0].output_types[0], "float32")
        self.assertEqual(network.layers[1].precision, "float16")
        self.assertEqual(network.layers[1].output_types[0], "float16")

    def test_model_family_resolution_covers_requested_face_models(self):
        self.assertEqual(module.canonical_model_family("inswapper", "model.onnx"), "inswapper")
        self.assertEqual(module.canonical_model_family(None, "restoreformer_plus_plus.onnx"), "restoreformer_pp")
        self.assertEqual(module.canonical_model_family(None, "GPEN-BFR-512.onnx"), "gpen")
        self.assertEqual(module.canonical_model_family(None, "xseg.onnx"), "dfl_xseg")
        self.assertEqual(module.canonical_model_family(None, "scrfd_2.5g_kps.onnx"), "scrfd")

    def test_polygraphy_command_exposes_onnx_and_trt_outputs(self):
        baseline, trt = module.polygraphy_commands(
            "model.onnx", baseline_results="baseline.json", trt_results="trt.json"
        )
        self.assertIn("--model-type", baseline)
        self.assertIn("onnx", baseline)
        self.assertIn("--onnxrt", baseline)
        self.assertIn("--trt", trt)
        self.assertIn("--trt-outputs", trt)

    def test_graph_surgery_classifies_sensitive_nodes(self):
        try:
            import onnx_graphsurgeon as gs
        except ImportError:
            self.skipTest("onnx-graphsurgeon is optional in the light test environment")
        import onnx

        x = gs.Variable("x", dtype=np.float32, shape=(1, 4, 4, 4))
        norm_out = gs.Variable("norm_out", dtype=np.float32)
        soft_out = gs.Variable("self_attn_softmax_out", dtype=np.float32)
        mm_out = gs.Variable("arcface_latent_out", dtype=np.float32)
        y = gs.Variable("output", dtype=np.float32)
        norm = gs.Node("InstanceNormalization", name="norm", inputs=[x, gs.Constant("scale", np.ones(4, dtype=np.float32)), gs.Constant("bias", np.zeros(4, dtype=np.float32))], outputs=[norm_out])
        soft = gs.Node("Softmax", name="self_attn.softmax", inputs=[norm_out], outputs=[soft_out])
        mm = gs.Node("MatMul", name="arcface_embedding_injection", inputs=[soft_out, gs.Constant("arcface_embedding", np.ones((4, 4), dtype=np.float32))], outputs=[mm_out])
        conv = gs.Node("Conv", name="final_projection", inputs=[x, gs.Constant("weight", np.ones((4, 4, 1, 1), dtype=np.float32)), gs.Constant("conv_bias", np.zeros(4, dtype=np.float32))], outputs=[y])
        graph = gs.Graph([norm, soft, mm, conv], [x], [y])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "inswapper.onnx"
            onnx.save(gs.export_onnx(graph), str(path))
            result = module.classify_onnx_graph(path, model_family="inswapper")
        by_name = result.annotation_by_name
        self.assertEqual(by_name["norm"].precision, "fp32")
        self.assertEqual(by_name["self_attn.softmax"].precision, "fp32")
        self.assertEqual(by_name["arcface_embedding_injection"].precision, "fp32")
        self.assertEqual(by_name["final_projection"].precision, "fp16")


if __name__ == "__main__":
    unittest.main()
