"""roop.trt_quant: tier selection, stratified calibration, cache validation, Q/DQ.

The light tests need only numpy. The ONNX ones build a three-conv graph in
memory; in the light profile importing onnx skips them (conftest). Nothing
here needs a GPU: engine builds and the quality verdict live in
app/tests/quant_quality_bench.py.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
tq = importlib.import_module("roop.trt_quant")


def _meta(pose="frontal", light="normal", tone="tone_mid", occ="clear", clip="a"):
    return {"pose_bin": pose, "light_bin": light, "tone_bin": tone,
            "occlusion_bin": occ, "clip": clip}


class TierTests(unittest.TestCase):
    def test_capability_maps_to_the_brief_tiers(self):
        self.assertEqual(tq.tier_for_capability((8, 9)), "fp8")    # Ada
        self.assertEqual(tq.tier_for_capability((9, 0)), "fp8")    # Hopper
        self.assertEqual(tq.tier_for_capability((12, 0)), "fp8")   # Blackwell
        self.assertEqual(tq.tier_for_capability((8, 6)), "int8")   # RTX 30
        self.assertEqual(tq.tier_for_capability((8, 0)), "int8")   # A100
        self.assertEqual(tq.tier_for_capability((7, 5)), "fp16")   # Turing
        self.assertEqual(tq.tier_for_capability(None), "fp16")

    def test_explicit_request_wins_over_hardware(self):
        with mock.patch.object(tq, "device_capability", return_value=(8, 9)):
            self.assertEqual(tq.select_tier(requested="int8"), "int8")
            self.assertEqual(tq.select_tier(requested="auto"), "fp8")
            self.assertEqual(tq.select_tier(), "fp8")
        with mock.patch.object(tq, "device_capability", return_value=(8, 6)):
            self.assertEqual(tq.select_tier(requested="bogus"), "int8")


class StratifyTests(unittest.TestCase):
    def test_rare_cells_are_not_swamped_by_the_common_one(self):
        metas = [_meta() for _ in range(900)]
        metas += [_meta(pose="profile") for _ in range(30)]
        metas += [_meta(light="low_light") for _ in range(30)]
        metas += [_meta(tone="tone_dark", occ="occluded") for _ in range(30)]
        picked = tq.stratify(metas, 100)
        self.assertEqual(len(picked), 100)
        self.assertEqual(len(set(picked)), 100)
        chosen = [metas[i] for i in picked]
        # four occupied cells, 25 each: every rare condition got its share
        self.assertEqual(sum(m["pose_bin"] == "profile" for m in chosen), 25)
        self.assertEqual(sum(m["light_bin"] == "low_light" for m in chosen), 25)
        self.assertEqual(sum(m["occlusion_bin"] == "occluded" for m in chosen), 25)

    def test_short_supply_is_not_padded_with_duplicates(self):
        metas = [_meta(clip=str(i % 3)) for i in range(40)]
        picked = tq.stratify(metas, 500)
        self.assertEqual(sorted(picked), list(range(40)))

    def test_a_cell_is_spread_over_clips(self):
        metas = [_meta(clip="big") for _ in range(100)] + [_meta(clip="small") for _ in range(10)]
        chosen = [metas[i]["clip"] for i in tq.stratify(metas, 20)]
        self.assertEqual(chosen.count("small"), 10)

    def test_bins(self):
        self.assertEqual(tq.pose_bin(10, -5), "frontal")
        self.assertEqual(tq.pose_bin(-30, 0), "mid")
        self.assertEqual(tq.pose_bin(5, 70), "profile")      # pitch counts too
        self.assertEqual(tq.pose_bin(80, 0), "profile")
        self.assertEqual(tq.tone_bin(55), "tone_light")
        self.assertEqual(tq.tone_bin(-20), "tone_dark")
        self.assertEqual(tq.light_bin({"luma_mean": 40, "shadow_frac": 0.3,
                                       "highlight_frac": 0}), "low_light")
        self.assertEqual(tq.light_bin({"luma_mean": 150, "shadow_frac": 0,
                                       "highlight_frac": 0.1}), "highlight")


class CacheValidatorTests(unittest.TestCase):
    def _identity(self, **kw):
        base = tq.QuantIdentity(schema=tq.SCHEMA, model_sha256="m", calibration_sha256="c",
                                tier="int8", recipe=tq.QUANT_RECIPE, tensorrt="10.9",
                                gpu="RTX", capability="8.6")
        return replace(base, **kw)

    def test_missing_manifest_and_every_identity_field_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp) / "x.engine"
            ident = self._identity()
            self.assertEqual(tq.validate_artifact(art, ident), ["missing"])
            art.write_bytes(b"engine")
            self.assertEqual(tq.validate_artifact(art, ident), ["no manifest"])
            tq.write_manifest(art, ident, calibration_samples=500)
            self.assertEqual(tq.validate_artifact(art, ident), [])
            self.assertEqual(tq.validate_artifact(art, self._identity(model_sha256="new")),
                             ["model_sha256 changed"])
            self.assertEqual(tq.validate_artifact(art, self._identity(calibration_sha256="n")),
                             ["calibration_sha256 changed"])
            self.assertEqual(tq.validate_artifact(art, self._identity(capability="8.9")),
                             ["capability changed"])
            art.write_bytes(b"tampered")
            self.assertIn("artifact bytes changed", tq.validate_artifact(art, ident))

    def test_ensure_engine_does_not_build_when_the_cache_is_valid(self):
        calib = _calib(4)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"ROOP_QUANT_ENGINE_DIR": tmp}), \
                mock.patch.object(tq, "graph_is_half", return_value=True), \
                mock.patch.object(tq, "device_identity",
                                  return_value={"gpu": "RTX", "capability": "8.6", "tensorrt": "10.9"}), \
                mock.patch.object(tq, "_build") as build:
            model = Path(tmp) / "m.onnx"
            model.write_bytes(b"onnx")
            ident = tq.identity_for(model, calib, "int8", precision_guard=False)
            engine = Path(tmp) / "m.int8.engine"
            engine.write_bytes(b"e")
            tq.write_manifest(engine, ident, layer_precisions={"Int8": 56, "Half": 210})
            path, tier, why = tq.ensure_engine(model, tier="int8", calib=calib)
            self.assertEqual((path, tier, why), (engine, "int8", []))
            build.assert_not_called()

    def test_changed_weights_trigger_recalibration(self):
        calib = _calib(4)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"ROOP_QUANT_ENGINE_DIR": tmp}), \
                mock.patch.object(tq, "graph_is_half", return_value=True), \
                mock.patch.object(tq, "device_identity",
                                  return_value={"gpu": "RTX", "capability": "8.6", "tensorrt": "10.9"}), \
                mock.patch.object(tq, "make_entropy_calibrator", return_value="CAL") as cal, \
                mock.patch.object(tq, "engine_precision_summary", return_value={"Int8": 3}), \
                mock.patch.object(tq, "_build", return_value=b"new-engine") as build:
            model = Path(tmp) / "m.onnx"
            model.write_bytes(b"onnx")
            engine = Path(tmp) / "m.int8.engine"
            engine.write_bytes(b"e")
            tq.write_manifest(engine, tq.identity_for(model, calib, "int8", precision_guard=False))
            model.write_bytes(b"onnx v2")                        # weights changed
            path, tier, why = tq.ensure_engine(model, tier="int8", calib=calib)
            self.assertEqual(why, ["model_sha256 changed"])
            cal.assert_called_once()
            self.assertEqual(build.call_args.kwargs["calibrator"], "CAL")
            self.assertEqual(engine.read_bytes(), b"new-engine")
            self.assertEqual(tq.validate_artifact(engine, tq.identity_for(
                model, calib, "int8", precision_guard=False)), [])

    def test_fp8_build_that_ran_no_fp8_layer_is_rejected_and_remembered(self):
        calib = _calib(4)
        with tempfile.TemporaryDirectory() as tmp,                 mock.patch.dict("os.environ", {"ROOP_QUANT_ENGINE_DIR": tmp}),                 mock.patch.object(tq, "graph_is_half", return_value=True),                 mock.patch.object(tq, "device_identity",
                                  return_value={"gpu": "RTX 4070", "capability": "8.9", "tensorrt": "10.9"}),                 mock.patch.object(tq, "collect_activation_amax", return_value={"x": 1.0}),                 mock.patch.object(tq, "insert_fp8_qdq", return_value=Path(tmp) / "q.onnx"),                 mock.patch.object(tq, "engine_precision_summary",
                                  return_value={"Float": 208, "Half": 234}),                 mock.patch.object(tq, "_build", return_value=b"fake-quant engine") as build:
            model = Path(tmp) / "m.onnx"
            model.write_bytes(b"onnx")
            with self.assertRaisesRegex(tq.QuantizationError, "0 layers in FP8"):
                tq.ensure_engine(model, tier="fp8", calib=calib)
            self.assertFalse((Path(tmp) / "m.fp8.engine").exists())
            with self.assertRaisesRegex(tq.QuantizationError, "already built and rejected"):
                tq.ensure_engine(model, tier="fp8", calib=calib)
            self.assertEqual(build.call_count, 1)        # not rebuilt on the next start

    def test_cached_engine_without_its_precision_is_not_trusted(self):
        calib = _calib(4)
        with tempfile.TemporaryDirectory() as tmp,                 mock.patch.dict("os.environ", {"ROOP_QUANT_ENGINE_DIR": tmp}),                 mock.patch.object(tq, "graph_is_half", return_value=True),                 mock.patch.object(tq, "device_identity",
                                  return_value={"gpu": "RTX", "capability": "8.9", "tensorrt": "10.9"}),                 mock.patch.object(tq, "collect_activation_amax", return_value={}),                 mock.patch.object(tq, "insert_fp8_qdq", return_value=Path(tmp) / "q.onnx"),                 mock.patch.object(tq, "engine_precision_summary", return_value={"Half": 3}),                 mock.patch.object(tq, "_build", return_value=b"e2") as build:
            model = Path(tmp) / "m.onnx"
            model.write_bytes(b"onnx")
            engine = Path(tmp) / "m.fp8.engine"
            engine.write_bytes(b"old")
            tq.write_manifest(engine, tq.identity_for(model, calib, "fp8", precision_guard=False),
                              layer_precisions={"Half": 3})
            with self.assertRaises(tq.QuantizationError):
                tq.ensure_engine(model, tier="fp8", calib=calib)
            build.assert_called_once()

    def test_missing_calibration_set_is_a_distinct_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(tq.CalibrationSetMissing):
                tq.CalibrationSet.load(Path(tmp) / "nope.calib.npz")


def _calib(n, size=8):
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(n, 16)).astype(np.float32)
    return tq.CalibrationSet(
        crops=rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8),
        embeddings=emb / np.linalg.norm(emb, axis=1, keepdims=True),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), image_input="target",
        embed_input="source", model_file="m.onnx", meta=[_meta() for _ in range(n)])


class CalibrationSetTests(unittest.TestCase):
    def test_round_trip_keeps_digest_and_feed(self):
        calib = _calib(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = calib.save(Path(tmp) / "m.calib.npz")
            back = tq.CalibrationSet.load(path)
        self.assertEqual(back.digest(), calib.digest())
        self.assertEqual(back.meta, calib.meta)
        feed = back.feed(1)
        self.assertEqual(feed["target"].shape, (1, 3, 8, 8))
        self.assertEqual(feed["source"].shape, (1, 16))
        # [-1, 1] input, the hyperswap contract, via the live to_blob
        self.assertGreaterEqual(float(feed["target"].min()), -1.0)
        self.assertLessEqual(float(feed["target"].max()), 1.0)

    def test_feed_is_the_live_to_blob(self):
        from roop.procmgr_tiling import to_blob
        calib = _calib(2)
        np.testing.assert_array_equal(calib.feed(0)["target"],
                                      to_blob(calib.crops[0], calib.mean, calib.std))

    def test_ita_orders_light_above_dark(self):
        cv2 = importlib.import_module("cv2")
        light = np.full((64, 64, 3), (170, 190, 230), np.uint8)    # BGR, pale
        dark = np.full((64, 64, 3), (40, 60, 90), np.uint8)
        self.assertGreater(tq.ita_degrees(light), tq.ita_degrees(dark))
        self.assertIsNotNone(cv2)


class HarvesterInverseTests(unittest.TestCase):
    def test_blob_inverts_to_the_exact_crop(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        harvest = importlib.import_module("build_calibration_set")
        from roop.procmgr_tiling import to_blob
        crop = np.random.default_rng(3).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        blob = to_blob(crop, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        np.testing.assert_array_equal(harvest._crop_from_blob(blob, (0.5,) * 3, (0.5,) * 3), crop)
        with self.assertRaises(AssertionError):
            harvest._crop_from_blob(blob + 1e-3, (0.5,) * 3, (0.5,) * 3)


def _tiny_graph(half=False):
    onnx = importlib.import_module("onnx")
    from onnx import TensorProto, helper, numpy_helper
    ft = TensorProto.FLOAT16 if half else TensorProto.FLOAT
    npt = np.float16 if half else np.float32
    rng = np.random.default_rng(0)

    def weight(name, o, i):
        return numpy_helper.from_array(rng.normal(size=(o, i, 3, 3)).astype(npt), name)

    inits = [weight("w43", 4, 3), weight("w44", 4, 4), weight("w34", 3, 4)]
    nodes = [
        helper.make_node("Conv", ["x", "w43"], ["a"], pads=[1, 1, 1, 1], name="c0"),
        helper.make_node("Relu", ["a"], ["b"], name="r0"),
        helper.make_node("Conv", ["b", "w44"], ["c"], pads=[1, 1, 1, 1], name="c1"),
        helper.make_node("Conv", ["c", "w34"], ["d"], pads=[1, 1, 1, 1], name="c2"),
        helper.make_node("Tanh", ["d"], ["y"], name="t0"),
    ]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", ft, [1, 3, 8, 8])],
                              [helper.make_tensor_value_info("y", ft, [1, 3, 8, 8])],
                              initializer=inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])


class QdqTests(unittest.TestCase):
    def test_final_projection_is_left_alone(self):
        model = _tiny_graph()
        self.assertEqual(tq.quantizable_conv_indices(model), [0, 2])

    def test_qdq_is_fp32_even_on_a_half_graph(self):
        onnx = importlib.import_module("onnx")
        from onnx import TensorProto
        for half in (False, True):
            with self.subTest(half=half), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "m.onnx"
                onnx.save(_tiny_graph(half), str(src))
                out = tq.insert_fp8_qdq(src, {"x": 2.0, "b": 4.0}, Path(tmp) / "q.onnx")
                model = onnx.load(str(out))
                onnx.checker.check_model(model)
                self.assertGreaterEqual(model.opset_import[0].version, 19)
                inits = {i.name: i for i in model.graph.initializer}
                # TensorRT folds constants to FP32; an FP16 scale matches no tactic
                self.assertEqual(inits["x__fp8_s"].data_type, TensorProto.FLOAT)
                self.assertEqual(inits["w34"].data_type, TensorProto.FLOAT)
                self.assertEqual(inits["x__fp8_z"].data_type, TensorProto.FLOAT8E4M3FN)
                scale = onnx.numpy_helper.to_array(inits["b__fp8_s"]).astype(np.float32)
                self.assertAlmostEqual(float(scale), 4.0 / 448.0, places=3)
                ops = [n.op_type for n in model.graph.node]
                # two activations + two weights quantized, final conv untouched
                self.assertEqual(ops.count("QuantizeLinear"), 4)
                final = next(n for n in model.graph.node if n.name == "c2")
                self.assertEqual(list(final.input), ["c", "w34"])

    def test_qdq_graph_runs_close_to_the_original(self):
        onnx = importlib.import_module("onnx")
        ort = importlib.import_module("onnxruntime")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "m.onnx"
            onnx.save(_tiny_graph(), str(src))
            x = np.random.default_rng(2).uniform(-1, 1, (1, 3, 8, 8)).astype(np.float32)
            ref_sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
            names = ["a", "b"]
            probe = onnx.load(str(src))
            for n in names:
                probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(n))
            acts = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"]
                                        ).run(["b"], {"x": x})
            amax = {"x": float(np.abs(x).max()), "b": float(np.abs(acts[0]).max())}
            out = tq.insert_fp8_qdq(src, amax, Path(tmp) / "q.onnx")
            try:
                q_sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            except Exception as error:           # ORT CPU without FP8 kernels
                self.skipTest(f"onnxruntime cannot run FP8 Q/DQ on CPU: {error}")
            ref = ref_sess.run(None, {"x": x})[0]
            got = q_sess.run(None, {"x": x})[0]
            cos = float(np.dot(ref.ravel(), got.ravel())
                        / (np.linalg.norm(ref) * np.linalg.norm(got)))
            self.assertGreater(cos, 0.99)


class SwapperWiringTests(unittest.TestCase):
    """The setting reaches the swapper, and every failure is loud, not silent."""

    def _swapper(self, mode, providers=(("TensorrtExecutionProvider", {}),)):
        swap = importlib.import_module("roop.processors.FaceSwapInsightFace")
        sw = swap.FaceSwapInsightFace()
        sw._swap_providers = list(providers)
        sw.model_swap_insightface = mock.Mock()
        cfg = mock.Mock(swap_quantization=mode)
        return swap, sw, cfg

    def test_off_never_touches_trt_quant(self):
        swap, sw, cfg = self._swapper("off")
        with mock.patch.object(swap.roop.globals, "CFG", cfg, create=True), \
                mock.patch.object(tq, "ensure_engine") as ensure:
            sw._load_quantized("m.onnx", "hyperswap")
        ensure.assert_not_called()
        self.assertIsNone(sw._native)

    def test_missing_calibration_set_is_printed_and_ort_stays(self):
        swap, sw, cfg = self._swapper("auto")
        with mock.patch.object(swap.roop.globals, "CFG", cfg, create=True), \
                mock.patch.object(tq, "ensure_engine",
                                  side_effect=tq.CalibrationSetMissing("x.calib.npz")), \
                mock.patch("builtins.print") as out:
            sw._load_quantized("m.onnx", "hyperswap")
        self.assertIsNone(sw._native)
        self.assertIn("build_calibration_set.py", " ".join(str(c) for c in out.call_args_list))

    def test_needs_tensorrt(self):
        swap, sw, cfg = self._swapper("auto", providers=("CUDAExecutionProvider",))
        with mock.patch.object(swap.roop.globals, "CFG", cfg, create=True), \
                mock.patch.object(tq, "ensure_engine") as ensure:
            sw._load_quantized("m.onnx", "hyperswap")
        ensure.assert_not_called()

    def test_native_failure_at_run_time_falls_back_to_ort(self):
        swap, sw, cfg = self._swapper("auto")
        native = mock.Mock(outputs=["output", "mask"])
        native.run.side_effect = RuntimeError("enqueue failed")
        sw._native, sw._native_tier, sw._native_order = native, "fp8", ["output", "mask"]
        sw.image_input_name = "target"
        session = mock.Mock()
        session.run.return_value = ["ort"]
        sw.model_swap_insightface = session
        with mock.patch.dict("os.environ", {"ROOP_ORT_IO_BINDING": "0"}), \
                mock.patch("builtins.print"):
            out = sw._infer({"target": np.zeros((1, 3, 4, 4), np.float32)})
        self.assertEqual(out, ["ort"])
        self.assertIsNone(sw._native)

    def test_native_outputs_are_reordered_to_the_graph(self):
        swap, sw, cfg = self._swapper("auto")
        native = mock.Mock(outputs=["mask", "output"])
        native.run.return_value = ["M", "O"]
        sw._native, sw._native_tier, sw._native_order = native, "fp8", ["output", "mask"]
        sw.image_input_name = "target"
        self.assertEqual(sw._infer({"target": np.zeros((1, 3, 4, 4), np.float32)}), ["O", "M"])

    def test_batch_calls_stay_on_ort(self):
        swap, sw, cfg = self._swapper("auto")
        native = mock.Mock(outputs=["output"])
        sw._native, sw._native_order = native, ["output"]
        sw.image_input_name = "target"
        session = mock.Mock()
        session.run.return_value = ["ort-batch"]
        sw.model_swap_insightface = session
        with mock.patch.dict("os.environ", {"ROOP_ORT_IO_BINDING": "0"}):
            out = sw._infer({"target": np.zeros((2, 3, 4, 4), np.float32)})
        self.assertEqual(out, ["ort-batch"])
        native.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
