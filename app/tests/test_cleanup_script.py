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
