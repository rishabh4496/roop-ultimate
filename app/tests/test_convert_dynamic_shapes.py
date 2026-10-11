"""tools/convert_dynamic_shapes.py: a model is "dynamic" only if a batch of N == N single runs.

Synthetic graphs with the pitfalls found on the real models:

* a Reshape that bakes batch 1 (fixable: the data is provably batched),
* a weight Reshape with a leading 1 (must be left alone),
* SCRFD's batch folded into axis 0 either batch-major or INTERLEAVED per cell
  (the real det_10g / scrfd_2.5g_kps layout, found only by trying it),
* a graph that mixes samples (ReduceSum over the batch) - unfixable, must fail,
* extra graph outputs with static shapes (RestoreFormer++ leaks 15; ORT treats a
  declared output shape as fact and constant-folds Shape(), pinning the batch).

All run on the CPU provider; no GPU, no model files.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

TOOL = Path(__file__).resolve().parents[2] / "tools" / "convert_dynamic_shapes.py"
spec = importlib.util.spec_from_file_location("convert_dynamic_shapes_under_test", TOOL)
cds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cds)


def _model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _vi(name, *shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _shape_const(name, values):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.array(values, np.int64)))


def conv_reshape_model(extra_static_output=False):
    """x[1,3,8,8] -> Conv -> Relu -> Reshape(const [1,4,64]) -> y[1,4,64]."""
    weight = numpy_helper.from_array(np.random.RandomState(0).randn(4, 3, 3, 3).astype(np.float32), "w")
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        _shape_const("s", [1, 4, 64]),
        helper.make_node("Reshape", ["r", "s"], ["y"]),
    ]
    outputs = [_vi("y", 1, 4, 64)]
    if extra_static_output:
        nodes.append(helper.make_node("Identity", ["r"], ["leak"]))
        outputs.append(_vi("leak", 1, 4, 8, 8))
    return _model(nodes, [_vi("x", 1, 3, 8, 8)], outputs, [weight])


def fold_model(perm, name="y"):
    """x[1,2,2,2] -> Transpose(perm) -> Reshape([-1,1]) : the SCRFD score-head shape."""
    return _model(
        [helper.make_node("Transpose", ["x"], ["t"], perm=perm),
         _shape_const("s", [-1, 1]),
         helper.make_node("Reshape", ["t", "s"], [name])],
        [_vi("x", 1, 2, 2, 2)], [_vi(name, 8, 1)])


def batch_mixer_model():
    """y = sum over the batch axis: loads, runs, and is wrong for any batch > 1."""
    return _model(
        [helper.make_node("Constant", [], ["axes"], value=numpy_helper.from_array(np.array([0], np.int64))),
         helper.make_node("ReduceSum", ["x", "axes"], ["y"], keepdims=1)],
        [_vi("x", 1, 3)], [_vi("y", 1, 3)])


def convert(model, **kwargs):
    converted = onnx.ModelProto()
    converted.CopyFrom(model)
    changes = cds.make_batch_dynamic(converted)
    return converted, changes, cds.verify_equivalence(model, converted, kwargs.pop("batches", (1, 3)), **kwargs)


# ---------------------------------------------------------------- surgery


def test_inputs_and_outputs_get_the_named_batch_dimension():
    converted, changes, _ = convert(conv_reshape_model())
    assert cds._dims(converted.graph.input[0])[0] == "batch_size"
    assert cds._dims(converted.graph.output[0])[0] == "batch_size"
    assert changes["inputs"][0]["before"][0] == 1


def test_a_reshape_that_bakes_batch_1_on_batched_data_is_rewritten_and_verified():
    model = conv_reshape_model()
    converted, changes, result = convert(model)
    assert changes["reshapes"] == 1
    constant = next(n for n in converted.graph.node if n.op_type == "Constant")
    assert numpy_helper.to_array(constant.attribute[0].t).tolist() == [0, 4, 64]
    assert result["ok"], result["reason"]
    assert result["worst_rel_error"] < 1e-5


def test_every_output_loses_its_static_batch_not_only_the_verified_one():
    """A leaked static output would otherwise let ORT constant-fold Shape() to batch 1."""
    converted, _, result = convert(conv_reshape_model(extra_static_output=True), outputs=["y"])
    assert [cds._dims(o)[0] for o in converted.graph.output] == ["batch_size", "batch_size"]
    assert result["ok"], result["reason"]
    assert result["unverified_outputs"] == ["leak"]


def test_a_weight_reshape_with_a_leading_1_is_left_alone():
    weight = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(6), "w")
    model = _model(
        [_shape_const("s", [1, 6]),
         helper.make_node("Reshape", ["w", "s"], ["w2"]),
         helper.make_node("Add", ["x", "w2"], ["y"])],
        [_vi("x", 1, 6)], [_vi("y", 1, 6)], [weight])
    converted, changes, result = convert(model)
    constant = next(n for n in converted.graph.node if n.op_type == "Constant")
    assert numpy_helper.to_array(constant.attribute[0].t).tolist() == [1, 6]
    assert changes["reshapes"] == 0
    assert result["ok"], result["reason"]


def test_a_shared_initializer_shape_is_copied_not_edited():
    shape = numpy_helper.from_array(np.array([1, 4, 64], np.int64), "shared")
    weight = numpy_helper.from_array(np.random.RandomState(0).randn(4, 3, 3, 3).astype(np.float32), "w")
    model = _model(
        [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
         helper.make_node("Reshape", ["c", "shared"], ["y"]),
         helper.make_node("Identity", ["shared"], ["shape_copy"])],
        [_vi("x", 1, 3, 8, 8)],
        [_vi("y", 1, 4, 64),
         helper.make_tensor_value_info("shape_copy", TensorProto.INT64, [3])],
        [weight, shape])
    converted = onnx.ModelProto()
    converted.CopyFrom(model)
    changes = cds.make_batch_dynamic(converted)
    assert changes["reshapes_copied"] == 1
    inits = {i.name: numpy_helper.to_array(i).tolist() for i in converted.graph.initializer}
    assert inits["shared"] == [1, 4, 64]                       # the Identity consumer still sees it
    assert inits["shared__dynbatch"] == [0, 4, 64]


# ---------------------------------------------------------------- verification


@pytest.mark.parametrize("perm,expected", [
    ([0, 2, 3, 1], "batch-major fold"),
    ([2, 3, 0, 1], "cell-major fold (2 anchors per cell)"),     # SCRFD's actual layout
])
def test_the_fold_layout_of_a_batch_in_axis_0_is_found_and_named(perm, expected):
    model = fold_model(perm)
    converted, _, result = convert(model, batches=(1, 4))
    assert cds._dims(converted.graph.output[0])[0] == "batch_size_x_rows"
    assert result["ok"], result["reason"]
    assert result["layouts"]["y"] == expected


def test_a_graph_that_mixes_samples_fails_and_says_which_batch():
    _, _, result = convert(batch_mixer_model())
    assert not result["ok"]
    assert "batch 3" in result["reason"]


def test_a_numerically_wrong_batched_graph_is_caught_even_though_it_loads_and_runs():
    """Cumulative sum across the batch axis: right shape, wrong values."""
    model = _model(
        [helper.make_node("Constant", [], ["axis"], value=numpy_helper.from_array(np.array(0, np.int64))),
         helper.make_node("CumSum", ["x", "axis"], ["y"])],
        [_vi("x", 1, 3)], [_vi("y", 1, 3)])
    _, _, result = convert(model)
    assert not result["ok"]
    assert "differs from per-sample runs" in result["reason"]


def test_rank_zero_outputs_are_skipped_not_failed():
    model = _model(
        [helper.make_node("Relu", ["x"], ["y"]),
         helper.make_node("ReduceSum", ["y"], ["total"], keepdims=0)],
        [_vi("x", 1, 3)],
        [_vi("y", 1, 3), helper.make_tensor_value_info("total", TensorProto.FLOAT, [])])
    _, _, result = convert(model)
    assert result["ok"], result["reason"]
    assert result["skipped_outputs"] == ["total"]


# ---------------------------------------------------------------- driver


def test_a_verified_model_is_written_with_the_suffix_and_passes_the_onnx_checker(tmp_path):
    source = tmp_path / "net.onnx"
    onnx.save(conv_reshape_model(), str(source))
    report = cds.convert_model(source, batches=(1, 2))
    assert report["ok"] and report["written"] == str(tmp_path / "net_dynamic.onnx")
    onnx.checker.check_model(report["written"])
    assert cds._dims(onnx.load(report["written"]).graph.input[0])[0] == "batch_size"


def test_a_model_that_fails_verification_is_not_written_unless_forced(tmp_path):
    source = tmp_path / "mixer.onnx"
    onnx.save(batch_mixer_model(), str(source))
    report = cds.convert_model(source, batches=(1, 3))
    assert not report["ok"] and report["written"] is None
    assert not (tmp_path / "mixer_dynamic.onnx").exists()
    forced = cds.convert_model(source, batches=(1, 3), force=True)
    assert forced["written"] and "written anyway" in forced["reason"]


def test_prepare_dynamic_model_reuses_a_current_file_and_raises_on_an_unconvertible_one(tmp_path, monkeypatch):
    source = tmp_path / "net.onnx"
    onnx.save(conv_reshape_model(), str(source))
    first = cds.prepare_dynamic_model(source, batches=(1, 2))
    monkeypatch.setattr(cds, "convert_model", lambda *a, **k: pytest.fail("a current _dynamic file must be reused"))
    assert cds.prepare_dynamic_model(source) == first

    bad = tmp_path / "mixer.onnx"
    onnx.save(batch_mixer_model(), str(bad))
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="not batch-convertible"):
        cds.prepare_dynamic_model(bad, batches=(1, 3))


def test_restoreformer_pp_verifies_only_its_image_output():
    assert cds.PRIMARY_OUTPUTS["restoreformer_plus_plus"] == ("2701",)
