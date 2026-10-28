"""Stage 17: the lip-sync model is accounted for like every other model - lifecycle table, TensorRT receipt, VRAM plan.

Each claim names its site and drives it:

  * `Lipsync_Wav2Lip._register_runtime`  puts `lipsync:wav2lip` in the Stage 2 lifecycle table, and pays the engine build on
    a dummy pass at load time through `predictor.warmup_session` - so a TensorRT session emits the Stage 14 compile
    receipt (model, cache_hit) there and not on frame 0 of a render;
  * `vram_governor.estimate_budget_mb`   counts a resident lip-sync model in the plan, and leaves every plan without one
    exactly as it was (the learned-ratio key included - an unchanged key keeps every ratio already learned).

No Wav2Lip weights exist on this machine, so the session is a stub that behaves like an ORT TensorRT session where it matters
here: it reports its providers and cache path, and a first inference writes (or does not write) a plan file.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


class _Meta:
    def __init__(self, name, shape):
        self.name, self.shape, self.type = name, shape, "tensor(float)"


class _TrtSession:
    """An ORT-shaped stub: TensorRT provider, an engine cache dir, two Wav2Lip inputs, a plan file written by run()."""

    def __init__(self, cache, build):
        self.cache, self.build, self.runs = cache, build, 0

    def get_providers(self):
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    def get_provider_options(self):
        return {"TensorrtExecutionProvider": {"trt_fp16_enable": True, "trt_max_workspace_size": 2 * 1024 ** 3,
                                              "trt_engine_cache_enable": True, "trt_engine_cache_path": self.cache}}

    def get_inputs(self):
        return [_Meta("face", [1, 6, 96, 96]), _Meta("mel", [1, 1, 80, 16])]

    def get_outputs(self):
        return [_Meta("out", [1, 3, 96, 96])]

    def run(self, _outputs, feed):
        self.runs += 1
        assert feed["face"].shape == (1, 6, 96, 96) and feed["mel"].shape == (1, 1, 80, 16)
        if self.build:
            with open(os.path.join(self.cache, "wav2lip.engine"), "wb") as fh:
                fh.write(b"plan")
        return []


class Wav2LipRuntimeRegistrationTest(unittest.TestCase):
    def setUp(self):
        from roop import model_lifecycle, pipeline_status, predictor
        from roop.processors.Lipsync_Wav2Lip import Lipsync_Wav2Lip
        self.model_lifecycle, self.ps, self.predictor = model_lifecycle, pipeline_status, predictor
        model_lifecycle.clear_model_lifecycle_records()
        pipeline_status.reset()
        predictor.reset()
        self.addCleanup(model_lifecycle.clear_model_lifecycle_records)
        self.addCleanup(pipeline_status.reset)
        self.addCleanup(predictor.reset)
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.Model = Lipsync_Wav2Lip

    def register(self, build):
        model = self.Model()
        model.session = _TrtSession(self.tmp.name, build)
        model._register_runtime(["TensorrtExecutionProvider"])
        return model

    def test_the_model_is_in_the_lifecycle_table_with_its_real_provider(self):
        self.register(build=True)
        rows = [r for r in self.model_lifecycle.get_model_lifecycle_records() if r.model == "lipsync:wav2lip"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].provider, "TensorrtExecutionProvider")
        self.assertEqual(rows[0].engine_cache, "ENABLED")
        self.assertIn("1x6x96x96", rows[0].input_shape.replace(" ", "").replace(",", "x").replace("(", "").replace(")", "")
                      .replace("[", "").replace("]", "") or rows[0].input_shape)

    def test_a_cold_tensorrt_build_emits_a_miss_receipt_naming_the_model(self):
        model = self.register(build=True)
        self.assertEqual(model.session.runs, 1, "the engine is built on a dummy pass at load, not on frame 0")
        compile_ = self.ps.snapshot(events=0)["last_compile"]
        self.assertIsNotNone(compile_)
        self.assertEqual(compile_["model"], "lipsync:wav2lip")
        self.assertIs(compile_["cache_hit"], False)
        self.assertIn("MISS", self.ps.format_receipt(compile_))

    def test_a_warm_cache_emits_a_hit_receipt(self):
        self.register(build=True)
        self.predictor.reset()                                       # a new process: nothing warmed yet, the plan is on disk
        self.register(build=False)
        compile_ = self.ps.snapshot(events=0)["last_compile"]
        self.assertIs(compile_["cache_hit"], True)
        self.assertIn("HIT", self.ps.format_receipt(compile_))

    def test_a_failure_to_warm_or_register_never_breaks_the_load(self):
        class Broken(_TrtSession):
            def get_inputs(self):
                raise RuntimeError("boom")
        model = self.Model()
        model.session = Broken(self.tmp.name, False)
        model._register_runtime(["TensorrtExecutionProvider"])        # must not raise: lip-sync still runs without either

    def test_the_load_path_calls_it_after_the_session_exists(self):
        with open(os.path.join(APP, "roop", "processors", "Lipsync_Wav2Lip.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertLess(src.index("self.session = create_strictly_accelerated_session("), src.index("self._register_runtime(providers)"))


class VramPlanTest(unittest.TestCase):
    def setUp(self):
        from roop import vram_governor as vg
        self.vg = vg

    def job(self, **kw):
        return self.vg.JobSpec(enhancer="None", swap_model="realswap", threads=8, batch_cap=8, **kw)

    def test_a_resident_lipsync_model_adds_its_own_line_to_the_plan(self):
        base = self.vg.estimate_budget_mb(self.job(), 8, None)
        for model in ("wav2lip", "musetalk"):
            plan = self.vg.estimate_budget_mb(self.job(lipsync=model), 8, None)
            self.assertIn("lipsync", plan)
            self.assertGreater(plan["lipsync"], 0.0)
            for key, value in base.items():
                self.assertEqual(plan[key], value, "%s changed with lipsync on" % key)
        self.assertGreater(self.vg.estimate_budget_mb(self.job(lipsync="musetalk"), 8, None)["lipsync"],
                           self.vg.estimate_budget_mb(self.job(lipsync="wav2lip"), 8, None)["lipsync"],
                           "the diffusion UNet is the bigger model")

    def test_no_lipsync_leaves_the_plan_and_the_learned_ratio_key_exactly_as_they_were(self):
        job = self.job()
        self.assertNotIn("lipsync", self.vg.estimate_budget_mb(job, 8, None))
        self.assertNotIn("lipsync", job.signature())
        self.assertEqual(job.signature(), self.job(lipsync="").signature())
        self.assertIn("lipsync:wav2lip", self.job(lipsync="wav2lip").signature())
        self.assertNotEqual(self.job(lipsync="wav2lip").signature(), job.signature(), "a lip-sync job learns its own ratio")

    def test_the_render_builds_the_job_with_the_model_the_request_chose(self):
        import roop.globals as g
        saved = (getattr(g, "lipsync_enabled", False), getattr(g, "lipsync_model", "wav2lip"))
        try:
            g.lipsync_enabled, g.lipsync_model = True, "MuseTalk"
            self.assertEqual(self.vg.job_from_render([], "None", "realswap", [], 12.0).lipsync, "musetalk")
            g.lipsync_enabled = False
            self.assertEqual(self.vg.job_from_render([], "None", "realswap", [], 12.0).lipsync, "")
        finally:
            g.lipsync_enabled, g.lipsync_model = saved

    def test_the_session_pool_knows_the_two_model_families(self):
        from roop.session_pool import _resource_spec
        self.assertEqual(_resource_spec("lipsync:wav2lip").model_key, "lipsync")
        self.assertEqual(_resource_spec("lipsync:musetalk").model_key, "lipsync-generative")
        self.assertNotEqual(_resource_spec("lipsync:wav2lip").model_key, "unknown")


if __name__ == "__main__":
    unittest.main()
