"""A swap preview must not run beside an active render.

`/api/preview` sets the process-global `roop_globals.is_preview` (which narrows
`session_pool.pool_size()` to 1 for EVERY thread, the render's workers
included), runs `core.live_swap` on the shared GPU, and ends in
`torch.cuda.empty_cache()`. The React panel defers its own previews while a job
is processing, but the comparison-grid loader, a second tab and a direct API
client do not, so the guard has to live in the backend.

These assert admission, not pixels: whether `live_swap` and the global writes
happen at all. The cheap, cache-only endpoints (the live JPEG, the progress poll,
target stills) must keep answering while a render runs.
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import roop.globals as roop_globals  # noqa: E402


class PreviewRenderLockoutTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self._processing = api._progress.get("processing")
        self._is_preview = roop_globals.is_preview
        self._cfg = getattr(roop_globals, "CFG", None)
        if self._cfg is None:
            roop_globals.CFG = mock.MagicMock()
        roop_globals.is_preview = False

        def restore():
            api._progress["processing"] = self._processing
            roop_globals.is_preview = self._is_preview
            roop_globals.CFG = self._cfg
        self.addCleanup(restore)

    def test_preview_is_refused_with_409_while_a_render_runs(self):
        api._progress["processing"] = True
        with mock.patch("roop.core.live_swap", side_effect=AssertionError("inference ran")) as swap, \
                mock.patch.object(api, "_activate_target_from_payload",
                                  side_effect=AssertionError("target context touched")):
            r = self.client.post("/api/preview", json={"frame": 1, "fake_preview": True})
        self.assertEqual(r.status_code, 409)
        body = r.json()
        self.assertTrue(body["render_active"])
        self.assertIn("render is in progress", body["message"])
        swap.assert_not_called()

    def test_refusal_leaves_the_process_globals_alone(self):
        api._progress["processing"] = True
        roop_globals.is_preview = False
        with mock.patch("torch.cuda.empty_cache") as empty:
            self.client.post("/api/preview", json={"frame": 1})
        self.assertFalse(roop_globals.is_preview)       # was set True, then False, mid-render
        empty.assert_not_called()                        # a render's cached blocks stay put

    def test_preview_reaches_the_target_context_when_idle(self):
        api._progress["processing"] = False
        with mock.patch.object(api, "_activate_target_from_payload",
                               side_effect=RuntimeError("reached")):
            with self.assertRaises(RuntimeError):
                self.client.post("/api/preview", json={"frame": 1})

    def test_cache_only_endpoints_keep_answering_during_a_render(self):
        api._progress["processing"] = True
        self.assertEqual(self.client.get("/api/progress").status_code, 200)
        # 204 (nothing published yet) or 200 (a frame) - never refused.
        self.assertIn(self.client.get("/api/live_frame").status_code, (200, 204))


if __name__ == "__main__":
    unittest.main()
