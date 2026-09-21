"""Network exposure of the backend: loopback by default, token in share mode,
foreign Origins refused.

Before api_access existed the API bound to 127.0.0.1 but answered ANY browser
page: CORS was `*` with credentials (Starlette echoes the caller's Origin),
and /ws/telemetry accepted every upgrade. A remote web page could therefore
POST to e.g. /api/target/add_path on a visitor's machine. `server_share` only
ever reached the frozen Gradio UI (a gradio.live tunnel); it never widened
the API bind, so enabling it did nothing visible and nothing safe.

These drive the real ASGI app through Starlette's TestClient (client address
"testclient", i.e. NOT loopback, so share mode demands the token) plus the
policy object directly for the loopback exemption.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

os.environ.setdefault('ROOP_REACT_CLIENT', '1')

import api  # noqa: E402
import api_access  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

TOKEN = "test-launch-token-0123456789"
PING = "/api/ui/status"          # cheap, side-effect free, exists in every build


def _yaml(text):
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


class LoopbackDefault(unittest.TestCase):
    """Nothing but an explicit request turns share mode on."""

    def tearDown(self):
        api_access.set_policy(None)

    def test_share_is_off_unless_asked_for(self):
        self.assertFalse(api_access.share_requested(_yaml("max_threads: 4\n"), argv=[]))
        self.assertFalse(api_access.share_requested(_yaml("server_share: false\n"), argv=[]))
        self.assertFalse(api_access.share_requested(_yaml("server_share: 'yes'\n"), argv=[]))
        self.assertFalse(api_access.share_requested("/nonexistent/config.yaml", argv=[]))
        self.assertFalse(api_access.share_requested(_yaml(": not yaml: [\n"), argv=[]))
        # env vars are not a way in
        with mock.patch.dict(os.environ, {"ROOP_SERVER_SHARE": "1"}):
            self.assertFalse(api_access.share_requested(_yaml("a: 1\n"), argv=[]))

    def test_share_turns_on_from_config_or_flag(self):
        self.assertTrue(api_access.share_requested(_yaml("server_share: true\n"), argv=[]))
        self.assertTrue(api_access.share_requested(_yaml("server_share: false\n"),
                                                   argv=["run.py", "--server_share"]))

    def test_policy_hosts(self):
        off = api_access.AccessPolicy(share=False)
        self.assertEqual(off.host, "127.0.0.1")
        self.assertIsNone(off.token)
        on = api_access.AccessPolicy(share=True)
        self.assertEqual(on.host, "0.0.0.0")
        self.assertGreaterEqual(len(on.token), 24)
        # a new launch, a new token
        self.assertNotEqual(on.token, api_access.AccessPolicy(share=True).token)

    def test_run_api_binds_loopback_by_default(self):
        api_access.set_policy(api_access.AccessPolicy(share=False))
        with mock.patch.object(api.uvicorn, "run") as run:
            api.run_api()
        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")

    def test_run_api_binds_all_interfaces_only_in_share_mode_and_says_so(self):
        api_access.set_policy(api_access.AccessPolicy(share=True, token=TOKEN))
        with mock.patch.object(api.uvicorn, "run") as run, \
                mock.patch("builtins.print") as printed:
            api.run_api()
        self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")
        banner = "\n".join(str(c.args[0]) for c in printed.call_args_list if c.args)
        self.assertIn("SHARE MODE IS ON", banner)
        self.assertIn(TOKEN, banner)


class TokenRequiredInShareMode(unittest.TestCase):

    def setUp(self):
        api_access.set_policy(api_access.AccessPolicy(share=True, token=TOKEN))
        self.client = TestClient(api.app)

    def tearDown(self):
        api_access.set_policy(None)

    def test_api_without_token_is_401(self):
        r = self.client.get(PING)
        self.assertEqual(r.status_code, 401)
        self.assertIn("token", r.json()["detail"].lower())

    def test_wrong_token_is_401(self):
        self.assertEqual(self.client.get(PING, headers={"Authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(self.client.get(PING, params={"token": "nope"}).status_code, 401)
        self.assertEqual(self.client.get(PING, cookies={api_access.TOKEN_COOKIE: "nope"}).status_code, 401)

    def test_bearer_header_query_and_cookie_all_admit(self):
        self.assertEqual(self.client.get(PING, headers={"Authorization": f"Bearer {TOKEN}"}).status_code, 200)
        self.assertEqual(self.client.get(PING, params={"token": TOKEN}).status_code, 200)
        self.assertEqual(self.client.get(PING, cookies={api_access.TOKEN_COOKIE: TOKEN}).status_code, 200)

    def test_websocket_needs_the_token_too(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/ws/telemetry"):
                pass
        with self.client.websocket_connect(f"/ws/telemetry?token={TOKEN}") as ws:
            self.assertEqual(ws.receive_json()["event"], "hello")

    def test_preflight_needs_no_token(self):
        # A CORS preflight cannot carry credentials; refusing it would break a
        # local dev page that is about to send the token on the real request.
        r = self.client.options(PING, headers={"Origin": "http://localhost:5173",
                                               "Access-Control-Request-Method": "GET"})
        self.assertEqual(r.status_code, 200)

    def test_loopback_peer_is_exempt(self):
        # The launcher's own stop/pause/resume scripts call from 127.0.0.1 and
        # carry no token; a local browser page is still held to the Origin rule.
        policy = api_access.get_policy()
        self.assertTrue(policy.token_ok(client_host="127.0.0.1"))
        self.assertTrue(policy.token_ok(client_host="::1"))
        self.assertFalse(policy.token_ok(client_host="192.168.1.20"))
        local = TestClient(api.app, client=("127.0.0.1", 50000))
        self.assertEqual(local.get(PING).status_code, 200)
        self.assertEqual(local.get(PING, headers={"Origin": "https://evil.example"}).status_code, 403)

    @unittest.skipUnless(api.ui_dist_ready(), "react-ui/dist not built")
    def test_index_hands_out_the_cookie_only_for_the_token(self):
        self.assertEqual(self.client.get("/").status_code, 401)
        self.assertIn("token required", self.client.get("/").text.lower())
        r = self.client.get("/", params={"token": TOKEN})
        self.assertEqual(r.status_code, 200)
        cookie = r.headers.get("set-cookie", "")
        self.assertIn(f"{api_access.TOKEN_COOKIE}={TOKEN}", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        # and from then on the API works with no header at all
        self.assertEqual(self.client.get(PING).status_code, 200)

    def test_saving_server_share_is_announced_not_silent(self):
        # A stand-in CFG: the real one is not loaded in a test process, and
        # save_settings() persists to config.yaml, which a test must not touch.
        api_access.set_policy(api_access.AccessPolicy(share=False))
        import roop.globals as roop_globals
        saved = []
        stub = types.SimpleNamespace(server_share=False, save=lambda: saved.append(True))
        with mock.patch.object(roop_globals, "CFG", stub), mock.patch("builtins.print") as printed:
            api.save_settings({"server_share": True})
        self.assertTrue(stub.server_share)
        self.assertTrue(saved)
        said = " ".join(str(c.args[0]) for c in printed.call_args_list if c.args)
        self.assertIn("server_share ENABLED", said)
        self.assertIn("next launch", said)


class ForeignOriginRejected(unittest.TestCase):
    """Even on loopback: a browser page from another site gets 403, not CORS."""

    def setUp(self):
        api_access.set_policy(api_access.AccessPolicy(share=False))
        self.client = TestClient(api.app)

    def tearDown(self):
        api_access.set_policy(None)

    def test_foreign_origin_is_403_on_api_and_ws(self):
        for origin in ("https://evil.example", "http://192.168.1.99:8001", "null", "http://localhost.evil.example"):
            with self.subTest(origin=origin):
                self.assertEqual(self.client.get(PING, headers={"Origin": origin}).status_code, 403)
                self.assertEqual(self.client.post("/api/source/clear", headers={"Origin": origin}).status_code, 403)
                with self.assertRaises(WebSocketDisconnect):
                    with self.client.websocket_connect("/ws/telemetry", headers={"Origin": origin}):
                        pass

    def test_foreign_origin_gets_no_cors_grant_on_preflight(self):
        r = self.client.options(PING, headers={"Origin": "https://evil.example",
                                               "Access-Control-Request-Method": "POST"})
        self.assertNotEqual(r.status_code, 200)
        self.assertIsNone(r.headers.get("access-control-allow-origin"))

    def test_served_and_local_origins_are_fine(self):
        for origin in ("http://testserver",            # == the Host header: same origin
                       "http://localhost:5173",        # vite dev server
                       "http://127.0.0.1:8001",
                       "https://8001.localhost"):      # pinokio's https proxy
            with self.subTest(origin=origin):
                self.assertEqual(self.client.get(PING, headers={"Origin": origin}).status_code, 200)
        with self.client.websocket_connect("/ws/telemetry", headers={"Origin": "http://testserver"}) as ws:
            self.assertEqual(ws.receive_json()["event"], "hello")

    def test_no_origin_is_fine(self):
        # curl, python requests, the launcher's stop.js -- no browser, no Origin
        self.assertEqual(self.client.get(PING).status_code, 200)

    def test_cors_grants_only_local_origins(self):
        ok = self.client.options(PING, headers={"Origin": "http://localhost:5173",
                                                "Access-Control-Request-Method": "GET"})
        self.assertEqual(ok.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_static_ui_is_not_gated_by_origin(self):
        # Only /api and /ws are guarded; a page load carries no Origin anyway.
        r = self.client.get("/", headers={"Origin": "https://evil.example"})
        self.assertIn(r.status_code, (200, 503))


if __name__ == "__main__":
    unittest.main()
