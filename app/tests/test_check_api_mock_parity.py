"""Stage 16: `scripts/verification/check_api_mock_parity.py` holds the repo to its contract, and can fail.

The script compares route signatures and the /ws/telemetry frame keys between `app/api.py` (+ its routers), the mock
server (`react-ui/mock-server/server.ts`) and what `pillModel.js` reads. A checker that only ever passes proves nothing,
so besides asserting the repo is in parity this file feeds it each kind of drift (a mock route the backend lacks, a
mock frame key the backend never sends, a pill read neither side carries, a stale allow-list entry) and requires a
verdict, and pins that the extractors really see the things a regex-light parser tends to miss (router prefixes,
`api_route`, `**spread` calls). Pure text/AST work: no ML stack, runs in the light profile in well under a second.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCRIPT = os.path.join(ROOT, "scripts", "verification", "check_api_mock_parity.py")


def _load():
    spec = importlib.util.spec_from_file_location("check_api_mock_parity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


parity = _load()


class RepoIsInParityTest(unittest.TestCase):
    def test_no_new_drift_and_no_stale_allow_list_entry(self):
        new, stale = parity.evaluate()
        self.assertEqual(new, {}, "contract drift between backend, mock server and pillModel: %s" % new)
        self.assertEqual(stale, [], "KNOWN_DRIFT entries that no longer drift - delete them: %s" % stale)

    def test_the_command_line_exits_zero(self):
        done = subprocess.run([sys.executable, "-I", SCRIPT], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("[parity] OK", done.stdout)

    def test_every_allow_list_entry_has_a_reason(self):
        for key, why in parity.KNOWN_DRIFT.items():
            self.assertTrue(isinstance(why, str) and len(why) > 20, "%s needs a real reason" % (key,))


class ExtractorsSeeTheRealThingsTest(unittest.TestCase):
    """If an extractor silently returned a subset, every comparison above would pass for the wrong reason."""

    def test_backend_routes_include_plain_prefixed_websocket_and_api_route_forms(self):
        routes = parity.backend_routes()
        self.assertIn(("GET", "/api/state"), routes)                         # @app.get in api.py
        self.assertIn(("WS", "/ws/telemetry"), routes)                       # @router.websocket
        self.assertIn(("GET", "/api/file"), routes)                          # @router.api_route(..., methods=["GET","HEAD"])
        self.assertIn(("HEAD", "/api/file"), routes)
        self.assertIn(("POST", "/api/projects/{}/resume"), routes)           # path parameter normalised
        prefixed = [r for r in routes if r[1].startswith("/api/trt_cache")]  # APIRouter(prefix="/api/trt_cache")
        self.assertTrue(prefixed, "an APIRouter(prefix=...) route was not seen with its prefix")

    def test_mock_routes_normalise_express_params_and_upgrade_paths(self):
        routes = parity.mock_routes()
        self.assertIn(("WS", "/ws/telemetry"), routes)
        self.assertIn(("GET", "/api/state"), routes)
        self.assertIn(("GET", "/api/file/{}"), routes)                       # app.get('/api/file/:name')
        self.assertFalse([r for r in routes if "${" in r[1] or ":" in r[1]], "an unnormalised mock path leaked through")

    def test_real_frame_keys_include_the_spread_functions_not_just_the_literal(self):
        keys = parity.real_frame_keys()
        for literal in ("processing", "fps", "fps_now", "dashboard", "live_seq"):
            self.assertIn(literal, keys)
        for spread in ("stages", "summary", "io_decode", "engine_state", "engine_since"):   # **stage_telemetry / video_io / pipeline_status
            self.assertIn(spread, keys, "`**` spread keys were not resolved")
        self.assertIn("event", keys)                                          # added by the hub

    def test_mock_frame_keys_include_its_literal_and_buildstages_keys(self):
        keys = parity.mock_frame_keys()
        for key in ("fps", "fps_now", "current_frame", "stages", "summary", "stage"):
            self.assertIn(key, keys)

    def test_pill_model_reads_are_found(self):
        self.assertTrue({"stages", "summary", "fps_now", "fps"} <= parity.pill_top_level_reads())
        self.assertTrue({"state", "stage_progress_pct", "skipped_frames_pct"} <= parity.pill_stage_reads())


class TheCheckCanFailTest(unittest.TestCase):
    def test_a_mock_route_the_backend_lacks_is_drift(self):
        invented = parity.mock_routes() | {("GET", "/api/zz_invented_route")}
        with mock.patch.object(parity, "mock_routes", return_value=invented):
            new, _ = parity.evaluate()
        self.assertIn(("route", "GET /api/zz_invented_route"), new)

    def test_a_mock_route_with_the_wrong_method_is_drift(self):
        # /api/state is GET on the backend; a mock that serves it as POST would pass a path-only check.
        wrong = parity.mock_routes() | {("POST", "/api/state")}
        with mock.patch.object(parity, "mock_routes", return_value=wrong):
            new, _ = parity.evaluate()
        self.assertIn(("route", "POST /api/state"), new)

    def test_a_mock_only_test_hook_under_double_underscore_mock_is_exempt(self):
        hooks = parity.mock_routes() | {("POST", "/__mock/anything")}
        with mock.patch.object(parity, "mock_routes", return_value=hooks):
            new, _ = parity.evaluate()
        self.assertNotIn(("route", "POST /__mock/anything"), new)

    def test_a_mock_frame_key_the_backend_never_sends_is_drift(self):
        with mock.patch.object(parity, "mock_frame_keys", return_value=parity.mock_frame_keys() | {"zz_invented_key"}):
            new, _ = parity.evaluate()
        self.assertIn(("frame", "zz_invented_key"), new)

    def test_a_telemetry_key_the_pill_reads_but_the_backend_lacks_is_drift(self):
        reads = parity.pill_top_level_reads() | {"zz_pill_key"}
        with mock.patch.object(parity, "pill_top_level_reads", return_value=reads):
            new, _ = parity.evaluate()
        self.assertIn(("pill", "backend:zz_pill_key"), new)
        self.assertIn(("pill", "mock:zz_pill_key"), new)

    def test_a_stage_key_the_pill_reads_but_the_mock_lacks_is_drift(self):
        reads = parity.pill_stage_reads() | {"zz_stage_key"}
        with mock.patch.object(parity, "pill_stage_reads", return_value=reads):
            new, _ = parity.evaluate()
        self.assertIn(("pill-stage", "mock:zz_stage_key"), new)

    def test_a_removed_backend_frame_key_is_caught_through_the_pill(self):
        # The regression this exists for: the backend stops sending `fps_now`, the pill falls back to `fps` silently.
        without = parity.real_frame_keys() - {"fps_now"}
        with mock.patch.object(parity, "real_frame_keys", return_value=without):
            new, _ = parity.evaluate()
        self.assertIn(("pill", "backend:fps_now"), new)

    def test_an_allow_list_entry_that_no_longer_drifts_is_stale(self):
        bogus = dict(parity.KNOWN_DRIFT)
        bogus[("route", "GET /api/state")] = "this route is in parity, so the entry must be flagged as stale"
        with mock.patch.object(parity, "KNOWN_DRIFT", bogus):
            _, stale = parity.evaluate()
        self.assertEqual(stale, [("route", "GET /api/state")])

    def test_the_command_line_exits_one_on_drift(self):
        invented = parity.mock_routes() | {("GET", "/api/zz_invented_route")}
        with mock.patch.object(parity, "mock_routes", return_value=invented), \
                mock.patch("builtins.print"):
            self.assertEqual(parity.main([]), 1)


if __name__ == "__main__":
    unittest.main()
