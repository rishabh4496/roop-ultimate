"""Network access policy for the backend: bind address, per-launch token, origins.

The backend is a local tool. By default it listens on loopback only, and even
there a browser page from another site must not be able to drive it: every
/api and /ws request carries the page's Origin, and a foreign one is refused
(403) -- the difference between "cannot read the response" (what CORS gives)
and "did not run", which matters for endpoints like /api/target/add_path that
read files off the disk.

`server_share` widens the bind to every interface. It is never applied
silently: it must be asked for (config.yaml or --server_share), it is announced
with a banner, and it comes with a random per-launch bearer token that every
non-loopback /api and /ws request must present -- as `Authorization: Bearer`,
as a `token` query parameter, or as the cookie the UI is given when it is
opened through the printed URL. No caller is exempt, loopback included: the
launcher's own Stop/Pause/Resume scripts get the token from start_react.js,
which captures it from the ready line (`ready_url`).

Everything here is decided from config.yaml and argv, not from
roop.globals.CFG: the API thread starts before core.run() loads CFG.
"""
from __future__ import annotations

import secrets
import sys
from http.cookies import CookieError, SimpleCookie
from typing import Optional
from urllib.parse import parse_qs, urlsplit

LOOPBACK_HOST = "127.0.0.1"
SHARE_HOST = "0.0.0.0"
TOKEN_COOKIE = "roop_api_token"
TOKEN_QUERY = "token"
_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}


def share_requested(config_path: str = "config.yaml", argv: Optional[list] = None) -> bool:
    """Whether share mode was explicitly asked for.

    `--server_share` on the command line wins; otherwise `server_share: true`
    in config.yaml. Anything else -- a missing file, a parse error, an env
    var -- is NOT share mode. The default has to be loopback.
    """
    argv = sys.argv if argv is None else argv
    if "--server_share" in argv:
        return True
    try:
        import yaml
    except ImportError as exc:
        print(f"[api_access] PyYAML unavailable ({exc}); share mode stays OFF", flush=True)
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return False  # first launch: only default_config.yaml exists yet
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[api_access] could not read {config_path} ({exc}); share mode stays OFF", flush=True)
        return False
    return bool(isinstance(data, dict) and data.get("server_share") is True)


class AccessPolicy:
    """The bind host, the token (share mode only) and the two checks."""

    def __init__(self, share: bool, token: Optional[str] = None) -> None:
        self.share = bool(share)
        self.token = (token or secrets.token_urlsafe(24)) if self.share else None

    @property
    def host(self) -> str:
        return SHARE_HOST if self.share else LOOPBACK_HOST

    # -- Origin -------------------------------------------------------------
    def origin_allowed(self, origin: Optional[str], host_header: Optional[str]) -> bool:
        """True when the request may come from a browser page we served.

        No Origin at all is a non-browser client or a same-origin navigation;
        both are fine. A browser Origin must be this server (its host equals
        the request's Host header) or a local page (localhost, 127.0.0.1,
        ::1, *.localhost -- the last covers Pinokio's https proxy). The
        literal "null" origin and everything else is refused.
        """
        if not origin:
            return True
        parts = urlsplit(origin.strip())
        oh = (parts.hostname or "").lower()
        if not oh:
            return False
        if oh in _LOCAL_HOSTNAMES or oh.endswith(".localhost"):
            return True
        if host_header:
            hh = (urlsplit("//" + host_header.strip()).hostname or "").lower()
            if hh and oh == hh:
                return True
        return False

    # -- Token --------------------------------------------------------------
    def token_ok(self, authorization: Optional[str] = None, cookie_header: Optional[str] = None,
                 query_string: Optional[str] = None) -> bool:
        if not self.share:
            return True
        candidates = []
        if authorization and authorization.lower().startswith("bearer "):
            candidates.append(authorization[7:].strip())
        if cookie_header:
            jar = SimpleCookie()
            try:
                jar.load(cookie_header)
            except CookieError:
                jar = SimpleCookie()  # a malformed header simply carries no token
            if TOKEN_COOKIE in jar:
                candidates.append(jar[TOKEN_COOKIE].value)
        if query_string:
            candidates.extend(parse_qs(query_string).get(TOKEN_QUERY, []))
        return any(c and secrets.compare_digest(c, self.token) for c in candidates)

    # -- Announcements ------------------------------------------------------
    def share_url(self, port: int) -> str:
        return f"http://<this-machine's-address>:{port}/?{TOKEN_QUERY}={self.token}"

    def ready_url(self, port: int) -> str:
        """The loopback URL the launcher captures from the ready line.

        In share mode it carries the token, so the Pinokio sidebar link opens
        a UI that works and start_react.js can capture the token for its
        Stop/Pause/Resume scripts. Regex-stable: host:port, then optionally
        `/?token=<url-safe token>`, nothing else on the line.
        """
        base = f"http://{LOOPBACK_HOST}:{port}"
        return f"{base}/?{TOKEN_QUERY}={self.token}" if self.share else base

    def banner(self, port: int) -> str:
        if not self.share:
            return ""
        return "\n".join([
            "",
            "=" * 72,
            "  SHARE MODE IS ON: the backend is listening on EVERY network interface.",
            "  Anyone who can reach this machine can reach the API. EVERY /api and /ws",
            "  request must carry this launch's token (new one every start):",
            "",
            f"      {self.token}",
            "",
            "  The Pinokio sidebar shows the token and its Open link carries it. Or open:",
            f"      {self.share_url(port)}",
            "",
            "  Scripts: Authorization: Bearer <token>   or   ?token=<token>",
            "  To turn this off: Settings -> Server -> Public server (share), then restart.",
            "=" * 72,
            "",
        ])


_policy: Optional[AccessPolicy] = None


def get_policy() -> AccessPolicy:
    """The process-wide policy, decided once from config.yaml / argv."""
    global _policy
    if _policy is None:
        _policy = AccessPolicy(share_requested())
    return _policy


def set_policy(policy: Optional[AccessPolicy]) -> None:
    """Tests install a policy explicitly; None resets to lazy discovery."""
    global _policy
    _policy = policy


class AccessControlMiddleware:
    """Pure ASGI: Origin check on /api and /ws, token check in share mode.

    Pure ASGI rather than BaseHTTPMiddleware so streamed video responses and
    the WebSocket upgrade pass through untouched. A refused WebSocket is
    closed before accept, which the client sees as HTTP 403.
    """

    GUARDED_PREFIXES = ("/api/", "/ws/")

    def __init__(self, app, policy_getter=get_policy) -> None:
        self.app = app
        self._policy = policy_getter

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope.get("path", "") or ""
        if not (path.startswith(self.GUARDED_PREFIXES) or path in ("/api", "/ws")):
            return await self.app(scope, receive, send)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        policy = self._policy()
        if not policy.origin_allowed(headers.get("origin"), headers.get("host")):
            return await self._refuse(scope, send, 403, "Origin not allowed")
        # A CORS preflight carries no credentials by design; it changes nothing.
        if scope["type"] == "http" and scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)
        if not policy.token_ok(headers.get("authorization"), headers.get("cookie"),
                               (scope.get("query_string") or b"").decode("latin-1")):
            return await self._refuse(scope, send, 401, "Bearer token required (share mode)")
        return await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(scope, send, status: int, detail: str):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": detail})
            return
        body = ('{"detail": "%s"}' % detail).encode("utf-8")
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("ascii"))]})
        await send({"type": "http.response.body", "body": body})


def token_from_request_query(query_string: str) -> Optional[str]:
    vals = parse_qs(query_string or "").get(TOKEN_QUERY, [])
    return vals[0] if vals else None


TOKEN_PAGE = """<!DOCTYPE html><html><head><title>Roop Ultimate - token required</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}
.card{background:#222;padding:32px;border-radius:12px;border:1px solid #333;max-width:520px;}
h1{color:#e94560;margin-top:0;}input{width:100%%;padding:8px;margin:12px 0;background:#111;color:#eee;border:1px solid #444;border-radius:6px;}
button{padding:8px 16px;background:#e94560;color:#fff;border:0;border-radius:6px;cursor:pointer;}code{color:#50a070;}</style></head>
<body><div class='card'><h1>Share mode: token required</h1>
<p>This backend is reachable from the network, so it asks for the token printed in the
launcher console at startup (a new one every launch).</p>
<form method='get' action='/'><input name='%s' placeholder='paste the token' autofocus>
<button type='submit'>Open the UI</button></form>
<p><small>Scripts: <code>Authorization: Bearer &lt;token&gt;</code> or <code>?%s=&lt;token&gt;</code>.</small></p>
</div></body></html>""" % (TOKEN_QUERY, TOKEN_QUERY)

