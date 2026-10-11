"""The swap net's protobuf is parsed ONCE per Initialize.

`FaceSwapInsightFace.Initialize` called `onnx.load(model_path)` twice for every
emap model (inswapper, reswapper, instyleswapper): once for the 512x512 identity
matrix, once for the graph transforms. inswapper_128 is 554 MB, so the duplicate
cost 0.22 s warm / 0.42 s cold on every render start. These pin that the emap
comes from the same parse the transforms use, and is the array the old path read.
"""
import inspect
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.processors import FaceSwapInsightFace as module  # noqa: E402

Swapper = module.FaceSwapInsightFace


def _tiny_model(path, rng):
    """A graph with a decoy 512x512 initializer, then the real one last (inswapper stores emap last)."""
    decoy = numpy_helper.from_array(rng.standard_normal((512, 512)).astype(np.float32), "decoy_not_emap")
    emap = numpy_helper.from_array(rng.standard_normal((512, 512)).astype(np.float32), "emap")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 512])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 512])
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "emap"], ["y"])], "g", [x], [y],
                              initializer=[decoy, emap])
    onnx.save(helper.make_model(graph), path)
    return numpy_helper.to_array(emap)


class SingleParseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "tiny.onnx")
        self.expected = _tiny_model(self.path, np.random.default_rng(0))

    def test_emap_is_the_array_the_two_step_path_read(self):
        model, emap = Swapper._parse_swap_model(self.path, True)
        old = Swapper._find_emap(onnx.load(self.path).graph)
        np.testing.assert_array_equal(emap, old)
        np.testing.assert_array_equal(emap, self.expected)
        self.assertEqual(emap.dtype, np.float32)
        self.assertIsInstance(model, onnx.ModelProto)

    def test_one_parse_whether_or_not_an_emap_is_wanted(self):
        for want in (True, False):
            with mock.patch.object(module.onnx, "load", wraps=onnx.load) as load:
                model, emap = Swapper._parse_swap_model(self.path, want)
            self.assertEqual(load.call_count, 1, "want_emap=%s" % want)
            self.assertEqual(emap is not None, want)

    def test_initialize_has_no_parse_of_its_own(self):
        src = inspect.getsource(Swapper.Initialize)
        self.assertEqual(src.count("_parse_swap_model("), 1)
        self.assertNotIn("onnx.load(", src)


if __name__ == "__main__":
    unittest.main()
