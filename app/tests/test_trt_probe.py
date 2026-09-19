#!/usr/bin/env python3
"""Tests for the deterministic, non-Git-LFS TensorRT probe graph."""
from __future__ import annotations

import unittest
import hashlib
import sys
from pathlib import Path

import onnx

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from roop.trt_probe import TINY_ONNX_PROBE_BYTES


class TestTensorRTProbe(unittest.TestCase):
    def test_probe_is_a_real_valid_onnx_graph(self):
        self.assertGreater(len(TINY_ONNX_PROBE_BYTES), 32)
        lfs_marker = b"git-" + b"lfs.github.com/spec/v1"
        self.assertNotIn(lfs_marker, TINY_ONNX_PROBE_BYTES)

        model = onnx.load_from_string(TINY_ONNX_PROBE_BYTES)
        onnx.checker.check_model(model, full_check=False)

        self.assertEqual(model.graph.name, "probe")
        self.assertEqual([node.op_type for node in model.graph.node], ["Relu"])
        self.assertEqual([(item.domain, item.version) for item in model.opset_import], [("", 13)])

    def test_probe_bytes_are_deterministic(self):
        self.assertEqual(len(TINY_ONNX_PROBE_BYTES), 73)
        self.assertEqual(
            hashlib.sha256(TINY_ONNX_PROBE_BYTES).hexdigest(),
            "e73e6442d98ffc86524c3ce351358f5b0e3a063b2ccba1ed67db255f06797876",
        )


if __name__ == "__main__":
    unittest.main()
