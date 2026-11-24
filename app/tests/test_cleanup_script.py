"""scripts/cleanup.py (the sidebar's Clean button): what it may and may not delete.

Every case builds its own tree in a temp dir; nothing here touches a real install.
"""

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

_SPEC = importlib.util.spec_from_file_location(
    "cleanup_script", os.path.join(os.path.dirname(APP), "scripts", "cleanup.py"))
cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup)

DAY = 86400


def _touch(path, age_days=0, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    t = time.time() - age_days * DAY
    os.utime(path, (t, t))


def _age_tree(path, age_days):
    t = time.time() - age_days * DAY
    for dirpath, _d, files in os.walk(path):
        for fn in files:
            os.utime(os.path.join(dirpath, fn), (t, t))
    os.utime(path, (t, t))


class _Launcher(unittest.TestCase):
    """<home>/api/launcher with the module constants pointed at it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cleanup_script_")
        self.home = os.path.realpath(self.tmp.name)
        self.root = os.path.join(self.home, "api", "launcher")
        self.app = os.path.join(self.root, "app")
        os.makedirs(self.app)
        saved = {k: getattr(cleanup, k) for k in ("ROOT", "APP", "PINOKIO_HOME")}
        self.addCleanup(lambda: [setattr(cleanup, k, v) for k, v in saved.items()])
        cleanup.ROOT, cleanup.APP, cleanup.PINOKIO_HOME = self.root, self.app, self.home
        self.addCleanup(self.tmp.cleanup)


class TensorRTCacheTest(_Launcher):
    def setUp(self):
        super().setUp()
        with open(os.path.join(self.app, "config.yaml"), "w") as fh:
            fh.write("trt_precision: mixed\n")
        self.cache = os.path.join(self.app, "models", "trt_cache")
        self.live = ["mixed_NVIDIA_GeForce_RTX_4070_sm0809_cuda12.8_c631",
                     "mixed_NVIDIA_GeForce_RTX_4070_sm0809_cuda12.8_c631_gpen_512_fp32",
                     "recognition_adaface_fp16_sm89_b1", "surgical", "surgical_cache"]
        self.stale = ["_unittest_flags_sp2x512", "fp16_NVIDIA_GeForce_RTX_4070_c1"]
        for name in self.live + self.stale:
            _touch(os.path.join(self.cache, name, "e.engine"))

    def names(self, stale_only):
        return {os.path.basename(p) for p in cleanup._trt_dirs(stale_only)}

    def test_stale_never_includes_the_current_precision_folders(self):
        self.assertEqual(self.names(True), set(self.stale))

    def test_unprovable_folders_are_kept(self):
        # recognition_*_fp16_* and surgical* do not carry a precision prefix, so
        # nothing proves them stale; deleting them costs minutes of recompiling.
        self.assertFalse(self.names(True) & {"recognition_adaface_fp16_sm89_b1", "surgical"})

    def test_all_means_all(self):
        self.assertEqual(self.names(False), set(self.live + self.stale))


class PinokioScratchTest(_Launcher):
    def setUp(self):
        super().setUp()
        self.temp = os.path.join(self.home, "cache", "TEMP")
        os.makedirs(self.temp)
        env = {"TEMP": self.temp}
        saved = {k: os.environ.get(k) for k in ("TEMP", "TMPDIR", "TMP")}
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                                 for k, v in saved.items()])

    def test_only_entries_untouched_for_three_days(self):
        _touch(os.path.join(self.temp, "old_bench", "a.bin"), 10)
        _touch(os.path.join(self.temp, "fresh_bench", "a.bin"), 0)
        # an old folder with one NEW file inside is in use: judged by the newest file
        _touch(os.path.join(self.temp, "busy", "old.bin"), 10)
        _touch(os.path.join(self.temp, "busy", "new.bin"), 0)
        _touch(os.path.join(self.temp, "stray.log"), 10)
        _touch(os.path.join(self.temp, "claude", "proj", "old-session", "x"), 10)
        _touch(os.path.join(self.temp, "claude", "proj", "this-session", "x"), 0)
        for p in ("old_bench", "busy", os.path.join("claude", "proj", "old-session")):
            os.utime(os.path.join(self.temp, p), (time.time() - 10 * DAY,) * 2)
        got = {os.path.relpath(p, self.temp) for p in cleanup._old_scratch()}
        self.assertEqual(got, {"old_bench", "stray.log",
                               os.path.join("claude", "proj", "old-session")})

    def test_env_pointing_outside_pinokio_cache_is_ignored(self):
        outside = os.path.join(self.home, "elsewhere")
        _touch(os.path.join(outside, "old", "a.bin"), 10)
        os.environ["TEMP"] = outside
        self.assertEqual(cleanup._old_scratch(), [])

    def test_the_cache_root_itself_is_never_a_target(self):
        os.environ["TEMP"] = os.path.join(self.home, "cache")
        self.assertIsNone(cleanup._pinokio_cache("TEMP"))

    def test_not_under_api_means_no_home(self):
        cleanup.PINOKIO_HOME = None
        self.assertEqual(cleanup._old_scratch(), [])
        self.assertEqual(cleanup._package_caches(), [])


class UploadsTest(_Launcher):
    def setUp(self):
        super().setUp()
        up = os.path.join(self.app, "temp", "api_uploads")
        self.free, self.used = os.path.join(up, "free.mp4"), os.path.join(up, "used.mp4")
        _touch(self.free), _touch(self.used)
        os.makedirs(os.path.join(self.app, "projects"))

    def project(self, state):
        with open(os.path.join(self.app, "projects", "p.json"), "w") as fh:
            json.dump({"state": state, "target_path": self.used}, fh)

    def names(self):
        return {os.path.basename(p) for p in cleanup._upload_candidates()}

    def test_a_referenced_upload_is_kept_and_an_orphan_is_not(self):
        self.project("DONE")
        self.assertEqual(self.names(), {"free.mp4"})

    def test_nothing_goes_while_a_project_is_resumable(self):
        self.project("PAUSED")
        self.assertEqual(self.names(), set())

    def test_a_corrupt_project_protects_everything(self):
        with open(os.path.join(self.app, "projects", "p.json"), "w") as fh:
            fh.write("{ not json")
        self.assertEqual(self.names(), set())

    def test_the_backend_is_not_imported_and_sys_modules_is_restored(self):
        sentinel = object()
        sys.modules["api"] = sentinel
        try:
            self.project("DONE")
            cleanup._upload_candidates()
            self.assertIs(sys.modules["api"], sentinel)
        finally:
            sys.modules.pop("api", None)
        sys.modules.pop("routes_queue", None)
        cleanup._upload_candidates()
        self.assertNotIn("api", sys.modules)
        self.assertNotIn("routes_queue", sys.modules)


class DriverStaleEngineTest(_Launcher):
    """An engine whose name embeds another driver version can never be looked up again."""

    def setUp(self):
        super().setUp()
        with open(os.path.join(self.app, "config.yaml"), "w") as fh:
            fh.write("trt_precision: mixed\n")
        self.cache = os.path.join(self.app, "models", "trt_cache")
        base = "mixed_NVIDIA_GeForce_RTX_4070_sm0809_cuda12.8_%s_trt10.9.0.34_ort1.23.2"
        self.old_dir = base % "drv617.42"
        self.new_dir = base % "drv621.14"
        _touch(os.path.join(self.cache, self.old_dir, "a.engine"))
        _touch(os.path.join(self.cache, self.new_dir, "a.engine"))
        _touch(os.path.join(self.cache, "mixed_gpu", "b_drv621.01_trt10.9.engine"))   # tag on the FILE
        _touch(os.path.join(self.cache, "mixed_gpu", "c_drv621.14_trt10.9.engine"))
        _touch(os.path.join(self.cache, "TensorrtExecutionProvider_TRTKernel_graph_1.engine"))  # untagged
        # inside a folder that is stale by PRECISION and by driver: must be listed once
        _touch(os.path.join(self.cache, "fp16_gpu", "d_drv617.42_trt10.9.engine"))
        saved = cleanup._gpu_driver
        self.addCleanup(lambda: setattr(cleanup, "_gpu_driver", saved))

    def stale(self, driver):
        cleanup._gpu_driver = lambda: driver
        return {os.path.relpath(p, self.cache) for p in cleanup._trt_stale()}

    def test_other_driver_engines_go_and_current_ones_stay(self):
        got = self.stale("621.14")
        self.assertIn(self.old_dir, got)
        self.assertIn(os.path.join("mixed_gpu", "b_drv621.01_trt10.9.engine"), got)
        self.assertNotIn(self.new_dir, got)
        self.assertNotIn(os.path.join("mixed_gpu", "c_drv621.14_trt10.9.engine"), got)

    def test_untagged_engines_are_never_called_stale(self):
        self.assertNotIn("TensorrtExecutionProvider_TRTKernel_graph_1.engine", self.stale("621.14"))

    def test_unknown_driver_deletes_no_engine_by_driver(self):
        got = self.stale(None)
        self.assertEqual(got, {"fp16_gpu"})          # only the precision rule still applies

    def test_nothing_is_listed_twice(self):
        got = self.stale("621.14")
        self.assertIn("fp16_gpu", got)
        self.assertNotIn(os.path.join("fp16_gpu", "d_drv617.42_trt10.9.engine"), got)


class WorkProductTest(_Launcher):
    """Renders, project caches and uploads: deleted only past the keep window."""

    def setUp(self):
        super().setUp()
        self.keep = os.environ.pop("ROOP_CLEAN_KEEP_DAYS", None)
        self.addCleanup(lambda: os.environ.pop("ROOP_CLEAN_KEEP_DAYS", None))
        self.addCleanup(lambda: self.keep is None or os.environ.__setitem__("ROOP_CLEAN_KEEP_DAYS", self.keep))
        a = self.app
        _touch(os.path.join(a, "projects", "old.stage2.npz"), 10)
        _touch(os.path.join(a, "projects", "old.roop.journal"), 10)
        _touch(os.path.join(a, "projects", "new.stage2.npz"), 0)
        _touch(os.path.join(a, "projects", "old.json"), 10)          # the checkpoint itself: never listed
        _touch(os.path.join(a, "output", "old.mp4"), 10)
        _touch(os.path.join(a, "output", "old.mp4.provenance.json"), 10)
        _touch(os.path.join(a, "output", "new.mp4"), 0)
        _touch(os.path.join(a, "output", "notes.txt"), 10)           # not a video
        _touch(os.path.join(a, "output", "sub", "deep.mp4"), 10)     # only the top level is a render
        _touch(os.path.join(a, "temp", "api_uploads", "old.mp4"), 10)
        _touch(os.path.join(a, "temp", "api_uploads", "new.mp4"), 0)
        _touch(os.path.join(self.root, "output", "swap_quality_stage1", "x.mp4"), 10)
        _touch(os.path.join(self.root, ".roop", "regression", "k", "baseline.json"), 10)
        # a folder counts as young while ANY entry in it is, directory mtime included
        _age_tree(os.path.join(self.root, "output", "swap_quality_stage1"), 10)
        _age_tree(os.path.join(self.root, ".roop", "regression"), 10)

    def names(self, finder, base):
        return {os.path.relpath(p, base) for p in finder()}

    def test_project_caches_only_old_npz_and_journals(self):
        self.assertEqual(self.names(cleanup._project_caches, os.path.join(self.app, "projects")),
                         {"old.stage2.npz", "old.roop.journal"})

    def test_renders_old_video_and_provenance_only(self):
        self.assertEqual(self.names(cleanup._old_renders, os.path.join(self.app, "output")),
                         {"old.mp4", "old.mp4.provenance.json"})

    def test_uploads_only_old_files(self):
        self.assertEqual(self.names(cleanup._old_uploads, os.path.join(self.app, "temp", "api_uploads")),
                         {"old.mp4"})

    def test_quality_evidence_old_only(self):
        got = self.names(cleanup._quality_evidence, self.root)
        self.assertEqual(got, {os.path.join("output", "swap_quality_stage1"),
                               os.path.join(".roop", "regression")})

    def test_keep_days_zero_removes_recent_ones_too(self):
        os.environ["ROOP_CLEAN_KEEP_DAYS"] = "0"
        self.assertIn("new.stage2.npz", self.names(cleanup._project_caches, os.path.join(self.app, "projects")))
        self.assertIn("new.mp4", self.names(cleanup._old_renders, os.path.join(self.app, "output")))
        self.assertIn("new.mp4", self.names(cleanup._old_uploads, os.path.join(self.app, "temp", "api_uploads")))

    def test_a_bad_keep_value_falls_back_to_the_default(self):
        os.environ["ROOP_CLEAN_KEEP_DAYS"] = "soon"
        self.assertEqual(cleanup._keep_seconds(), cleanup.SCRATCH_DAYS * 86400.0)
        self.assertNotIn("new.mp4", self.names(cleanup._old_renders, os.path.join(self.app, "output")))


class DevLeftoverTest(_Launcher):
    def setUp(self):
        super().setUp()
        r, a = self.root, self.app
        for rel in ("scratch/x.txt", "scratch_probe.log", "test-trt-cache_gpen/e.engine",
                    "temp_fs_inspect/f", "roop_3d/.runtime/pytest/t/a", "react-ui/e2e/test-results/r.json",
                    "app/temp/tsx-me/x", "app/temp/node-compile-cache/x", "keep_me/x"):
            _touch(os.path.join(r, *rel.split("/")))
        saved = cleanup._git_tracked
        self.addCleanup(lambda: setattr(cleanup, "_git_tracked", saved))
        cleanup._git_tracked = lambda p: False

    def listed(self):
        return {os.path.relpath(p, self.root).replace(os.sep, "/") for p in cleanup._dev_leftovers()}

    def test_lists_the_leftovers_and_nothing_else(self):
        self.assertEqual(self.listed(), {
            "scratch", "scratch_probe.log", "test-trt-cache_gpen", "temp_fs_inspect",
            "roop_3d/.runtime/pytest", "react-ui/e2e/test-results", "app/temp/tsx-me",
            "app/temp/node-compile-cache"})

    def test_a_folder_git_tracks_is_kept(self):
        cleanup._git_tracked = lambda p: os.path.basename(p) == "scratch"
        self.assertNotIn("scratch", self.listed())
        self.assertIn("temp_fs_inspect", self.listed())


class NeverTouchedTest(_Launcher):
    """Whatever is ticked, with no age protection at all."""

    def test_models_env_facesets_and_config_survive_every_new_target(self):
        a = self.app
        keep = [os.path.join(a, "models", "buffalo_l", "w.onnx"), os.path.join(a, "env", "Scripts", "python.exe"),
                os.path.join(a, "facesets", "me.fsz"), os.path.join(a, "config.yaml"),
                os.path.join(a, "projects", "p.json"), os.path.join(self.root, "pinokio.js")]
        for p in keep:
            _touch(p, 99)
        _touch(os.path.join(a, "output", "r.mp4"), 99)
        _touch(os.path.join(a, "projects", "p.stage2.npz"), 99)
        saved = (cleanup._git_tracked, cleanup._gpu_driver)
        cleanup._git_tracked, cleanup._gpu_driver = (lambda p: False), (lambda: "621.14")
        self.addCleanup(lambda: [setattr(cleanup, n, v) for n, v in
                                 zip(("_git_tracked", "_gpu_driver"), saved)])
        os.environ["ROOP_CLEAN_KEEP_DAYS"] = "0"
        self.addCleanup(lambda: os.environ.pop("ROOP_CLEAN_KEEP_DAYS", None))
        cleanup.clean(["dev-leftovers", "project-caches", "old-uploads", "old-renders",
                       "quality-evidence", "trt-stale"])
        for p in keep:
            self.assertTrue(os.path.exists(p), p)
        self.assertFalse(os.path.exists(os.path.join(a, "output", "r.mp4")))
        self.assertFalse(os.path.exists(os.path.join(a, "projects", "p.stage2.npz")))


class SizeTest(unittest.TestCase):
    def test_hard_linked_files_free_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a"), os.path.join(d, "b")
            _touch(a, data=b"y" * 1000)
            try:
                os.link(a, b)
            except OSError:
                self.skipTest("hard links unavailable")
            self.assertEqual(cleanup._size(d), 0)
            os.remove(b)
            self.assertEqual(cleanup._size(d), 1000)


if __name__ == "__main__":
    unittest.main()
