"""The 5-D GridSample rewrite, checked against onnxruntime's own kernel.

No model download and no GPU: the rewrite is pure graph surgery, so tiny
synthetic graphs exercise exactly the same code that patches the 421 MB warping
module. onnxruntime's CPU GridSample is the reference — it is the very kernel
the rewrite is replacing, and it agrees with torch to 2.4e-07 on these shapes.

What would be silently wrong rather than loudly broken, and is therefore what
gets asserted: the x/y/z channel order (swap two and the face warps sideways),
the align_corners unnormalisation (a half-pixel shift), and zero-padding for
samples that fall outside the volume (a bright rim instead of nothing).
"""

import os
import sys
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.gridsample5d import patch_gridsample_5d  # noqa: E402

ort.set_default_logger_severity(3)

RNG = np.random.default_rng(19)
TOL = 1e-4


def build_model(shape, align=0, mode="bilinear", padding="zeros"):
    n, c, d, h, w, do, ho, wo = shape
    graph = helper.make_graph(
        [helper.make_node("GridSample", ["X", "G"], ["Y"], name="/gs", mode=mode,
                          padding_mode=padding, align_corners=align)],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, c, d, h, w]),
         helper.make_tensor_value_info("G", TensorProto.FLOAT, [n, do, ho, wo, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, c, do, ho, wo])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 9
    return model


def run(model, feeds):
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def feeds_for(shape, spread=3.0):
    """A grid deliberately wider than [-1, 1] so out-of-volume samples occur."""
    n, c, d, h, w, do, ho, wo = shape
    return {"X": RNG.standard_normal((n, c, d, h, w)).astype(np.float32),
            "G": (RNG.random((n, do, ho, wo, 3)).astype(np.float32) * spread
                  - spread / 2).astype(np.float32)}


SHAPES = [
    (1, 2, 4, 5, 6, 3, 4, 5),        # every dimension distinct: catches axis mixups
    (3, 1, 2, 3, 3, 2, 2, 2),        # batched, single channel
    (2, 5, 8, 8, 8, 4, 6, 7),        # non-cubic output
    (1, 8, 16, 16, 16, 16, 16, 16),  # the real module's depth/resolution ratio
]


class TestNumericalParity(unittest.TestCase):
    """The rewritten graph must reproduce the kernel it replaces."""

    def test_matches_onnxruntime_for_both_align_modes(self):
        for align in (0, 1):
            for shape in SHAPES:
                with self.subTest(align=align, shape=shape):
                    feeds = feeds_for(shape)
                    ref = run(build_model(shape, align), feeds)
                    patched, count = patch_gridsample_5d(build_model(shape, align))
                    self.assertEqual(count, 1)
                    self.assertLess(np.abs(run(patched, feeds) - ref).max(), TOL)

    def test_samples_fully_inside_the_volume_also_match(self):
        """Padding must not be the only reason the outputs agree."""
        shape = (1, 3, 6, 6, 6, 4, 4, 4)
        feeds = feeds_for(shape, spread=1.2)     # comfortably within [-1, 1]
        ref = run(build_model(shape), feeds)
        patched, _ = patch_gridsample_5d(build_model(shape))
        self.assertLess(np.abs(run(patched, feeds) - ref).max(), TOL)

    def test_far_outside_the_volume_is_zero_not_clamped(self):
        """padding_mode=zeros: a sample past the edge contributes nothing.

        Clamping instead of zeroing would smear the border voxels outward, which
        looks like a plausible image rather than an obvious failure.
        """
        shape = (1, 2, 4, 4, 4, 3, 3, 3)
        n, c, d, h, w, do, ho, wo = shape
        feeds = {"X": np.abs(RNG.standard_normal((n, c, d, h, w))).astype(np.float32) + 1.0,
                 "G": np.full((n, do, ho, wo, 3), 5.0, np.float32)}
        patched, _ = patch_gridsample_5d(build_model(shape))
        out = run(patched, feeds)
        self.assertTrue(np.allclose(out, 0.0), out.max())
        self.assertLess(np.abs(out - run(build_model(shape), feeds)).max(), TOL)

    def test_a_nan_coordinate_behaves_exactly_as_onnxruntime_does(self):
        """NaN must reach the output the same way, without an unsafe gather.

        onnxruntime's kernel emits NaN for a NaN coordinate, so the rewrite has
        to as well or it would be hiding a genuine upstream problem. The part
        that cannot be asserted from the outside is the reason for the Where
        guard: a NaN cast to int64 is undefined, and without pinning it the
        gather would read from an arbitrary offset instead of index 0.
        """
        shape = (1, 2, 4, 4, 4, 2, 2, 2)
        n, c, d, h, w, do, ho, wo = shape
        grid = np.zeros((n, do, ho, wo, 3), np.float32)
        grid[..., 0] = np.nan
        feeds = {"X": np.ones((n, c, d, h, w), np.float32), "G": grid}
        ref = run(build_model(shape), feeds)
        patched, _ = patch_gridsample_5d(build_model(shape))
        got = run(patched, feeds)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))

    def test_channel_order_is_x_y_z(self):
        """Moving one grid channel must move the matching axis, and only it.

        Built so each axis carries a distinct signal: if the rewrite swapped x
        and z the parity tests above would still pass on symmetric shapes, but
        this one would not.
        """
        shape = (1, 1, 4, 4, 4, 2, 2, 2)
        x = np.zeros((1, 1, 4, 4, 4), np.float32)
        x[0, 0, 3, 0, 0] = 1.0          # a single voxel at max-D, min-H, min-W
        # align_corners=1 makes -1/+1 land exactly on the first/last index.
        grid = np.zeros((1, 2, 2, 2, 3), np.float32)
        grid[..., 0] = -1.0             # x -> W index 0
        grid[..., 1] = -1.0             # y -> H index 0
        grid[..., 2] = 1.0              # z -> D index 3
        feeds = {"X": x, "G": grid}
        patched, _ = patch_gridsample_5d(build_model(shape, align=1))
        out = run(patched, feeds)
        self.assertTrue(np.allclose(out, 1.0), out)


class TestRewriteScope(unittest.TestCase):
    """Anything the rewrite cannot prove equivalent must be left alone."""

    def test_removes_every_gridsample_it_claims(self):
        patched, count = patch_gridsample_5d(build_model(SHAPES[0]))
        self.assertEqual(count, 1)
        self.assertEqual([n for n in patched.graph.node if n.op_type == "GridSample"], [])

    def test_result_is_a_valid_onnx_graph(self):
        patched, _ = patch_gridsample_5d(build_model(SHAPES[0]))
        onnx.checker.check_model(patched, full_check=False)

    def test_leaves_unsupported_padding_and_mode_untouched(self):
        for kwargs in ({"padding": "border"}, {"padding": "reflection"},
                       {"mode": "nearest"}):
            with self.subTest(**kwargs):
                model, count = patch_gridsample_5d(build_model(SHAPES[0], **kwargs))
                self.assertEqual(count, 0)
                self.assertEqual(
                    sum(1 for n in model.graph.node if n.op_type == "GridSample"), 1)

    def test_leaves_4d_gridsample_untouched(self):
        graph = helper.make_graph(
            [helper.make_node("GridSample", ["X", "G"], ["Y"], mode="bilinear")], "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 5, 6]),
             helper.make_tensor_value_info("G", TensorProto.FLOAT, [1, 4, 5, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 4, 5])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
        model.ir_version = 9
        _, count = patch_gridsample_5d(model)
        self.assertEqual(count, 0)

    def test_the_flattened_index_is_computed_in_integers(self):
        """The gather index must never be arithmetic on a float tensor.

        TensorRT runs this graph with FP16 enabled. The real module addresses a
        16*64*64 = 65536-voxel volume, and FP16 cannot represent integers past
        2048 exactly — its largest value at all is 65504, with a spacing of 32
        near the top. A float index therefore selects the wrong voxel or
        overflows to inf, which measured as 0.71 max abs error on a [0,1] output
        while still agreeing to 3e-06 on CPU, where everything is FP32.

        This is asserted structurally because no CPU test can reproduce it: the
        failure only exists once a backend chooses half precision.
        """
        shape = (1, 4, 16, 64, 64, 16, 64, 64)   # the real module's dimensions
        patched, count = patch_gridsample_5d(build_model(shape))
        self.assertEqual(count, 1)

        types = {}
        inferred = onnx.shape_inference.infer_shapes(patched, strict_mode=False)
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
            types[vi.name] = vi.type.tensor_type.elem_type
        produced_by = {o: n for n in patched.graph.node for o in n.output}

        gathers = [n for n in patched.graph.node if n.op_type == "GatherND"]
        self.assertEqual(len(gathers), 8, "trilinear needs exactly 8 corner reads")
        for gather in gathers:
            cast = produced_by[gather.input[1]]
            self.assertEqual(cast.op_type, "Cast")
            self.assertEqual(types.get(cast.input[0]), TensorProto.INT32,
                             "the linear index reaches the gather via float, "
                             "which FP16 will corrupt")

    def test_dynamic_shapes_are_skipped_rather_than_guessed(self):
        graph = helper.make_graph(
            [helper.make_node("GridSample", ["X", "G"], ["Y"], mode="bilinear")], "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                           ["N", 2, 4, 5, 6]),
             helper.make_tensor_value_info("G", TensorProto.FLOAT,
                                           ["N", 3, 4, 5, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                           ["N", 2, 3, 4, 5])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
        model.ir_version = 9
        _, count = patch_gridsample_5d(model)
        self.assertEqual(count, 0)


class TestFallbackBehaviour(unittest.TestCase):
    """A rewrite that cannot be produced must degrade, never raise."""

    def test_missing_source_returns_the_original_path(self):
        from roop.gridsample5d import ensure_patched_model
        missing = os.path.join(os.path.dirname(__file__), "does-not-exist.onnx")
        self.assertEqual(ensure_patched_model(missing, verbose=False), missing)


if __name__ == "__main__":
    unittest.main()
