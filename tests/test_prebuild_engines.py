"""tools/prebuild_engines.py: the pure parts (the build itself needs a GPU)."""

import importlib.util
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "prebuild_engines", os.path.join(ROOT, "tools", "prebuild_engines.py"))
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)


class ClassifyTest(unittest.TestCase):

    def test_growth_means_built_now(self):
        self.assertIn("COLD", pe.classify(40 * 1024 * 1024))
        self.assertIn("COLD", pe.classify(pe.COLD_BYTES))

    def test_small_bookkeeping_writes_are_a_cache_hit(self):
        # TensorRT rewrites small timing/profile files even on a warm start.
        self.assertIn("warm", pe.classify(0))
        self.assertIn("warm", pe.classify(pe.COLD_BYTES - 1))
        self.assertIn("warm", pe.classify(-5))      # cache shrank (clear-stale ran)


TRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


class ClassifySessionsTest(unittest.TestCase):
    """A stage that inspected nothing must not read as 'fine'."""

    def test_all_on_tensorrt_is_verified(self):
        on, off, status = pe.classify_sessions({"a": [TRT, CUDA, CPU], "b": [TRT, CPU]}, True)
        self.assertEqual((sorted(on), off), (["a", "b"], []))
        self.assertEqual(status, "verified 2 session(s)")

    def test_a_cuda_fallback_is_reported_when_tensorrt_was_requested(self):
        on, off, status = pe.classify_sessions({"a": [TRT, CPU], "b": [CUDA, CPU]}, True)
        self.assertEqual(off, ["b"])
        self.assertEqual(status, "NOT ON TENSORRT")

    def test_a_cpu_only_session_is_off_tensorrt(self):
        self.assertEqual(pe.classify_sessions({"a": [CPU]}, True)[2], "NOT ON TENSORRT")

    def test_no_sessions_is_unverified_not_ok(self):
        on, off, status = pe.classify_sessions({}, True)
        self.assertEqual((on, off), ([], []))
        self.assertTrue(status.startswith("UNVERIFIED"))

    def test_cuda_provider_run_does_not_flag_missing_tensorrt(self):
        _on, off, status = pe.classify_sessions({"a": [CUDA, CPU]}, False)
        self.assertNotEqual(status, "NOT ON TENSORRT")
        self.assertIn("TensorRT not requested", status)

    def test_an_empty_provider_list_counts_as_off(self):
        self.assertEqual(pe.classify_sessions({"a": []}, True)[2], "NOT ON TENSORRT")


class FindSessionsTest(unittest.TestCase):

    class Sess:
        def get_inputs(self): return []
        def run(self, *a): return []

    def test_finds_direct_pooled_and_nested_sessions(self):
        S = self.Sess

        class Pool:
            _items = [S(), S()]

        class Secondary:
            def __init__(self): self.model = S()

        class Proc:
            def __init__(self):
                self.model_xseg = S()
                self.pool = Pool()
                self.secondary = Secondary()
                self.unrelated = object()

        found = pe._ort_sessions(Proc())
        self.assertEqual(sorted(found), ["model_xseg", "pool[0]", "pool[1]", "secondary.model"])

    def test_a_processor_with_no_session_yields_nothing(self):
        class Proc:
            def __init__(self): self.x = 1
        self.assertEqual(pe._ort_sessions(Proc()), {})


class DirBytesTest(unittest.TestCase):

    def test_counts_nested_files_and_tolerates_a_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "a", "b"))
            for rel, size in (("x.engine", 1000), (os.path.join("a", "y.cache"), 500),
                              (os.path.join("a", "b", "z.profile"), 25)):
                with open(os.path.join(tmp, rel), "wb") as fh:
                    fh.write(b"\0" * size)
            self.assertEqual(pe.dir_bytes(tmp), 1525)
        self.assertEqual(pe.dir_bytes(os.path.join(tmp, "does-not-exist")), 0)


class ArgsTest(unittest.TestCase):

    def test_defaults_build_every_stage_on_tensorrt(self):
        a = pe.parse_args([])
        self.assertEqual(a.provider, "tensorrt")
        self.assertEqual(a.stages, list(pe.STAGES))
        self.assertIsNone(a.swap_model)

    def test_only_selects_stages(self):
        self.assertEqual(pe.parse_args(["--only", "analyser, swapper"]).stages,
                         ["analyser", "swapper"])

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(SystemExit):
            pe.parse_args(["--only", "analyser,nonsense"])

    def test_overrides_are_passed_through(self):
        a = pe.parse_args(["--swap-model", "realswap", "--enhancer", "GPEN 256 Pro",
                           "--mask-engine", "DFL XSeg"])
        self.assertEqual((a.swap_model, a.enhancer, a.mask_engine),
                         ("realswap", "GPEN 256 Pro", "DFL XSeg"))


class RealPathTest(unittest.TestCase):
    """The reason this tool exists: it must NOT use the old tool's private cache."""

    def setUp(self):
        self.src = open(os.path.join(ROOT, "tools", "prebuild_engines.py"), encoding="utf-8").read()

    def test_reads_the_apps_cache_not_a_private_one(self):
        self.assertIn('os.path.join(APP, "models", "trt_cache")', self.src)
        self.assertNotIn("ROOP_TRT_CACHE_DIR", self.src)
        self.assertNotIn("trt_engine_cache_path", self.src)       # no private provider options

    def test_goes_through_the_apps_own_loader(self):
        for needle in ("init_pipeline", "get_face_analyser", "manager.plugins",
                       "proc.Initialize(", "predictor.warmup_session"):
            self.assertIn(needle, self.src)

    def test_the_old_tool_now_says_it_is_not_read_by_the_app(self):
        old = open(os.path.join(ROOT, "tools", "build_trt_engines.py"), encoding="utf-8").read()
        self.assertIn("are NOT read by the app", old)
        self.assertIn("tools/prebuild_engines.py", old)


if __name__ == "__main__":
    unittest.main()
