#!/usr/bin/env python3
"""Fast static contract check: backend routes and /ws/telemetry keys vs the mock server and the pill model.

    python scripts/verification/check_api_mock_parity.py            # exit 1 on new drift
    python scripts/verification/check_api_mock_parity.py --report   # also list the allow-listed drift and unmocked routes

The React e2e suite and the dev server run against ``react-ui/mock-server/server.ts``, a hand-written stand-in for
``app/api.py``. When the two disagree, a UI change passes every browser test against the mock and breaks against the
real backend (or the reverse), and nothing says so until someone drives the real app. This script reads all three
sides as TEXT / AST - no import of the app, no ML stack, no server, < 1 s - and fails when:

  1. ROUTES    the mock serves a (method, path) the backend does not define. The mock may serve FEWER routes than
               the backend (it is a stand-in, not a clone); it may not invent ones. ``/__mock/*`` test hooks are exempt.
  2. FRAME     the mock's ``/ws/telemetry`` frame carries a top-level key the real frame does not (a UI built on it
               would work in the mock and read ``undefined`` live).
  3. PILL      ``react-ui/src/components/faceswap/pillModel.js`` reads a telemetry key (or a per-stage key) that the
               real frame, or the mock's, does not carry. The pill makes no REST call, so a missing key is silent.

Known, pre-existing drift is listed in ``KNOWN_DRIFT`` below WITH the reason it exists. The check fails on drift that
is not listed, and ALSO on a listed entry that no longer drifts (fix landed: delete the entry), so the list can only
shrink. Sources of truth for the real side: ``app/api.py`` + ``app/routes_*.py`` + ``app/roop/api/enhancer_routes.py``
(route decorators, ``APIRouter(prefix=...)``), ``api._telemetry_snapshot`` (the frame) and the functions it spreads.

Wired into: ``.githooks/pre-commit`` (when a side changed) and ``app/tests/test_check_api_mock_parity.py`` (pytest).
"""
from __future__ import annotations

import argparse
import ast
import functools
import os
import re
import sys
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(ROOT, "app")
MOCK_SERVER = os.path.join(ROOT, "react-ui", "mock-server", "server.ts")
MOCK_DIR = os.path.join(ROOT, "react-ui", "mock-server")
MOCK_TYPES = os.path.join(MOCK_DIR, "pipelineTypes.ts")
PILL_MODEL = os.path.join(ROOT, "react-ui", "src", "components", "faceswap", "pillModel.js")

HTTP_METHODS = ("get", "post", "put", "delete", "patch")
# Modules whose functions `_telemetry_snapshot` spreads (`**f()`), searched in this order for the function's name.
SPREAD_MODULES = ("app/api.py", "app/roop/stage_telemetry.py", "app/roop/pipeline_status.py",
                  "app/roop/video_io_hw.py", "app/roop/terminal_dashboard.py")
# `routes_telemetry` adds these to every frame it sends (`frame["event"] = ...; frame["ts"] = ...`).
HUB_KEYS = frozenset({"event", "ts"})

# (kind, key) -> why it is allowed to drift. Keep every reason checkable; delete the entry when the drift is gone.
_NO_FRAME_CONSUMER = ("mock-only: no react-ui/src or e2e file reads it off the telemetry frame "
                      "(GPU load, VRAM and temperature come from GET /api/system/telemetry); delete from the mock")
_NO_UI_CALLER = "mock-only: no backend route and no react-ui/src caller; delete from the mock or implement it"
KNOWN_DRIFT: dict[tuple[str, str], str] = {
    ("frame", "active_workers"): _NO_FRAME_CONSUMER,
    ("frame", "gpu_util"): _NO_FRAME_CONSUMER,
    ("frame", "status"): "mock-only: the real frame's status line is `desc`; nothing reads `status` off the frame",
    ("frame", "temp_c"): _NO_FRAME_CONSUMER,
    ("frame", "vram_total_gb"): _NO_FRAME_CONSUMER,
    ("frame", "vram_used_gb"): _NO_FRAME_CONSUMER,
    ("route", "GET /api/file/{}"): "the mock serves outputs at /api/file/<name>; the backend serves them at /api/media/<path> "
                                   "and GET /api/file?path=",
    ("route", "GET /api/keyframes"): _NO_UI_CALLER + " (typed by KeyframeOverride in api-contract.ts only)",
    ("route", "POST /api/keyframes"): _NO_UI_CALLER + " (typed by KeyframeOverride in api-contract.ts only)",
    ("route", "GET /api/segments"): _NO_UI_CALLER + " (typed by SegmentStatus in api-contract.ts only)",
    ("route", "POST /api/segments"): _NO_UI_CALLER + " (typed by SegmentStatus in api-contract.ts only)",
    ("route", "GET /api/settings/benchmark_threads"): "the backend route is POST-only, which is all ProviderBenchmark.jsx calls",
    ("route", "POST /api/mock/target_frames"): "e2e test hook (react-ui/e2e/timeline.spec.js) that sits outside the /__mock/ prefix",
    ("route", "POST /api/projects/{}"): "mock catch-all `/api/projects/:action`; the backend's project routes are "
                                        "/api/projects/{id}[/validate|/load|/resume]",
    ("route", "POST /api/system/profile"): "mock-only hardware-profile switch; the backend's /api/system/profile is a GET of "
                                           "per-stage timing (and the mock's GET returns a different shape - not detectable here)",
}


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────

def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _norm(path: str) -> str:
    """``/api/x/{id}`` and ``/api/x/:id`` both -> ``/api/x/{}``; query strings and trailing slashes dropped."""
    path = path.split("?", 1)[0]
    path = re.sub(r"\{[^}/]*\}", "{}", path)
    path = re.sub(r":[A-Za-z_][A-Za-z0-9_]*", "{}", path)
    return path.rstrip("/") or "/"


def _str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


# ── side 1a: backend routes ──────────────────────────────────────────────────────────────────────────────

def _backend_files() -> list[str]:
    files = [os.path.join(APP, n) for n in sorted(os.listdir(APP)) if n.endswith(".py")]
    extra = os.path.join(APP, "roop", "api", "enhancer_routes.py")
    if os.path.isfile(extra):
        files.append(extra)
    return files


@functools.lru_cache(maxsize=None)
def backend_routes() -> set[tuple[str, str]]:
    """``{(METHOD, normalised path)}`` for every ``@app|router.<verb>("/path")`` / ``.websocket("/path")`` decorator."""
    found: set[tuple[str, str]] = set()
    for path in _backend_files():
        tree = ast.parse(_read(path), filename=path)
        prefixes: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if (isinstance(func, ast.Name) and func.id == "APIRouter") or \
                        (isinstance(func, ast.Attribute) and func.attr == "APIRouter"):
                    prefix = next((_str(k.value) for k in node.value.keywords if k.arg == "prefix"), None) or ""
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            prefixes[target.id] = prefix
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                        and isinstance(dec.func.value, ast.Name) and dec.args):
                    continue
                verb, owner = dec.func.attr, dec.func.value.id
                sub = _str(dec.args[0])
                if sub is None or (verb not in HTTP_METHODS and verb not in ("websocket", "api_route")):
                    continue
                if owner != "app" and owner not in prefixes:
                    continue
                if verb == "api_route":                              # @router.api_route("/p", methods=["GET", "HEAD"])
                    methods = next((k.value for k in dec.keywords if k.arg == "methods"), None)
                    names = [_str(e).upper() for e in getattr(methods, "elts", []) if _str(e)] or ["GET"]
                else:
                    names = ["WS" if verb == "websocket" else verb.upper()]
                for method in names:
                    found.add((method, _norm(prefixes.get(owner, "") + sub)))
    return found


# ── side 1b: mock routes ─────────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def mock_routes() -> set[tuple[str, str]]:
    """Literal ``app.<verb>('/path')`` registrations and the ``pathname === '/ws/...'`` upgrade routes in the mock.

    A path written as a template literal (``${...}``) is skipped: it is not statically comparable.
    """
    source = _read(MOCK_SERVER)
    found: set[tuple[str, str]] = set()
    for verb, _quote, path in re.findall(r"\bapp\.(get|post|put|delete|patch)\(\s*(['\"])((?:(?!\2).)+)\2", source):
        if path.startswith("/") and "${" not in path:
            found.add((verb.upper(), _norm(path)))
    for path in re.findall(r"pathname\s*===\s*['\"](/ws/[^'\"]+)['\"]", source):
        found.add(("WS", _norm(path)))
    return found


def _covered(method: str, path: str, backend: set[tuple[str, str]]) -> bool:
    """True when the backend serves this method on a path with the same shape.

    A ``{}`` segment on EITHER side matches any one segment: the mock often writes ``/api/projects/:action`` where the
    backend has one literal route per action, and the reverse (a backend ``{project_id}`` the mock spells literally).
    """
    want = path.split("/")
    for b_method, b_path in backend:
        if b_method != method:
            continue
        have = b_path.split("/")
        if len(have) == len(want) and all(h == w or h == "{}" or w == "{}" for h, w in zip(have, want)):
            return True
    return False


# ── side 2: /ws/telemetry frame keys ─────────────────────────────────────────────────────────────────────

def _functions(tree: ast.AST, name: str) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]


def _return_dict_keys(func: ast.AST, resolve, depth: int) -> set[str]:
    """Literal keys of every ``return {...}`` in ``func``; a ``**f()`` entry or ``return f()`` is resolved by name."""
    keys: set[str] = set()
    for ret in ast.walk(func):
        if not isinstance(ret, ast.Return) or ret.value is None:
            continue
        value = ret.value
        if isinstance(value, ast.Dict):
            keys |= _dict_keys(value, resolve, depth)
        elif isinstance(value, ast.Call) and depth > 0:
            keys |= resolve(value, depth - 1)
    return keys


def _dict_keys(node: ast.Dict, resolve, depth: int) -> set[str]:
    keys: set[str] = set()
    for key, value in zip(node.keys, node.values):
        if key is None:                                             # `**spread`
            if isinstance(value, ast.Call) and depth > 0:
                keys |= resolve(value, depth - 1)
        elif _str(key) is not None:
            keys.add(_str(key))
    return keys


@functools.lru_cache(maxsize=None)
def real_frame_keys() -> set[str]:
    """Top-level keys of the live ``/ws/telemetry`` frame: ``api._telemetry_snapshot`` + what it spreads + hub keys."""
    trees = {rel: ast.parse(_read(os.path.join(ROOT, rel)), filename=rel) for rel in SPREAD_MODULES}

    def resolve(call: ast.Call, depth: int) -> set[str]:
        func = call.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name is None:
            return set()
        out: set[str] = set()
        for rel in SPREAD_MODULES:
            for fn in _functions(trees[rel], name):
                out |= _return_dict_keys(fn, resolve, depth)
        return out

    snapshots = _functions(trees["app/api.py"], "_telemetry_snapshot")
    if not snapshots:
        raise SystemExit("[parity] app/api.py no longer defines _telemetry_snapshot: update this checker")
    keys = _return_dict_keys(snapshots[0], resolve, 3)
    if len(keys) < 10:
        raise SystemExit("[parity] read only %d keys from _telemetry_snapshot: its shape changed, update this checker"
                         % len(keys))
    return keys | HUB_KEYS


def _brace_block(source: str, open_index: int) -> str:
    """The text between the ``{`` at ``open_index`` and its matching ``}`` (strings and template literals respected)."""
    depth, i, quote = 0, open_index, None
    while i < len(source):
        ch = source[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == "/" and source[i:i + 2] == "//":
            i = source.find("\n", i)
            if i < 0:
                break
            continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[open_index + 1:i]
        i += 1
    return source[open_index + 1:]


def _top_level_keys(block: str) -> set[str]:
    """``key:`` / ``'key':`` / ``...spread`` heads at nesting depth 0 of an object-literal body."""
    keys: set[str] = set()
    depth, i, quote, start = 0, 0, None, 0
    pieces: list[str] = []
    while i < len(block):
        ch = block[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif block[i:i + 2] == "//":
            i = block.find("\n", i)
            if i < 0:
                break
            continue
        elif ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "," and depth == 0:
            pieces.append(block[start:i])
            start = i + 1
        i += 1
    pieces.append(block[start:])
    for piece in pieces:
        piece = re.sub(r"\A(?:\s*//[^\n]*\n)+", "", piece)         # comment lines in front of the key
        m = re.match(r"\s*(?:['\"]([A-Za-z_][\w]*)['\"]|([A-Za-z_][\w]*))\s*:", piece)
        if m:
            keys.add(m.group(1) or m.group(2))
    return keys


def _mock_function_keys(name: str) -> set[str]:
    """Top-level keys of every ``return {...}`` object literal in ``export function <name>`` in the mock-server sources.

    Only the function's own body is read (up to the next ``export`` or section divider), so a helper's later returns are
    not mixed in. Nothing found = the function builds its object some way this reader cannot see: an error, not zero keys.
    """
    for path in sorted(os.path.join(MOCK_DIR, n) for n in os.listdir(MOCK_DIR) if n.endswith(".ts")):
        source = _read(path)
        start = source.find("export function %s(" % name)
        if start < 0:
            continue
        stop = len(source)
        for marker in ("\nexport ", "\n// ──"):
            hit = source.find(marker, start + 1)
            if hit >= 0:
                stop = min(stop, hit)
        body = source[start:stop]
        keys: set[str] = set()
        for match in re.finditer(r"\breturn\s*\{", body):
            keys |= _top_level_keys(_brace_block(body, match.end() - 1))
        if keys:
            return keys
    raise SystemExit("[parity] cannot read the keys %s() puts in the mock frame: update this checker" % name)


@functools.lru_cache(maxsize=None)
def mock_frame_keys() -> set[str]:
    """Top-level keys of the mock's frame: the ``ws.send(JSON.stringify({...}))`` literal + every ``...fn(...)`` it spreads."""
    source = _read(MOCK_SERVER)
    anchor = source.find("wss.on('connection'")
    send = source.find("JSON.stringify({", anchor)
    if anchor < 0 or send < 0:
        raise SystemExit("[parity] mock-server no longer builds a /ws/telemetry frame in wss.on('connection'): update this checker")
    block = _brace_block(source, send + len("JSON.stringify("))
    keys = _top_level_keys(block)
    for spread in re.findall(r"\.\.\.([A-Za-z_]\w*)\(", block):
        keys |= _mock_function_keys(spread)
    if len(keys) < 8:
        raise SystemExit("[parity] read only %d keys from the mock frame: its shape changed, update this checker" % len(keys))
    return keys


# ── side 3: pillModel.js ─────────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def pill_top_level_reads() -> set[str]:
    """Telemetry keys ``pillModel`` reads off its two slices: ``pipeline`` (stages / summary) and ``run`` (fps_now / fps)."""
    source = _read(PILL_MODEL)
    return set(re.findall(r"\b(?:pipeline|run)\??\.([a-z_][a-z0-9_]*)", source))


@functools.lru_cache(maxsize=None)
def pill_stage_reads() -> set[str]:
    """Per-stage keys: ``stages[def.stage]?.state`` / ``.stage_progress_pct`` / ``.skipped_frames_pct`` and ``summary.ok``."""
    source = _read(PILL_MODEL)
    return set(re.findall(r"stages\[[^\]]+\]\??\.([a-z_][a-z0-9_]*)", source))


@functools.lru_cache(maxsize=None)
def real_stage_keys() -> set[str]:
    tree = ast.parse(_read(os.path.join(APP, "roop", "stage_telemetry.py")))
    keys: set[str] = set()
    for fn in _functions(tree, "snapshot"):
        for node in ast.walk(fn):
            if isinstance(node, ast.Dict):
                keys |= {_str(k) for k in node.keys if _str(k)}
    return keys


@functools.lru_cache(maxsize=None)
def mock_stage_keys() -> set[str]:
    types = _read(MOCK_TYPES)
    fn = types.find("export function buildStages")
    body = types[fn:types.find("// ── Structured runtime report", fn)]
    return set(re.findall(r"\b([a-z][a-z0-9_]*)\s*:", body))


# ── comparison ───────────────────────────────────────────────────────────────────────────────────────────

def drift() -> dict[tuple[str, str], str]:
    """Every current disagreement as ``{(kind, key): detail}``."""
    out: dict[tuple[str, str], str] = {}
    real, mock = backend_routes(), mock_routes()
    for method, path in sorted(mock):
        if path.startswith("/__mock/") or path in ("*", "/*") or _covered(method, path, real):
            continue
        out[("route", "%s %s" % (method, path))] = "served by the mock, not defined by the backend"
    real_keys, mock_keys = real_frame_keys(), mock_frame_keys()
    for key in sorted(mock_keys - real_keys):
        out[("frame", key)] = "in the mock's /ws/telemetry frame, never sent by the backend"
    for key in sorted(pill_top_level_reads()):
        for name, keys in (("backend", real_keys), ("mock", mock_keys)):
            if key not in keys:
                out[("pill", "%s:%s" % (name, key))] = "pillModel.js reads telemetry key '%s', which the %s frame lacks" % (key, name)
    for key in sorted(pill_stage_reads()):
        for name, keys in (("backend", real_stage_keys()), ("mock", mock_stage_keys())):
            if key not in keys:
                out[("pill-stage", "%s:%s" % (name, key))] = "pillModel.js reads per-stage key '%s', which the %s stage object lacks" % (key, name)
    return out


def evaluate() -> tuple[dict[tuple[str, str], str], list[tuple[str, str]]]:
    """``(new drift not in KNOWN_DRIFT, KNOWN_DRIFT entries that no longer drift)``."""
    current = drift()
    new = {k: v for k, v in current.items() if k not in KNOWN_DRIFT}
    stale = sorted(k for k in KNOWN_DRIFT if k not in current)
    return new, stale


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", action="store_true", help="also list allow-listed drift and backend routes the mock lacks")
    args = parser.parse_args(list(argv) if argv is not None else None)

    new, stale = evaluate()
    real, mock = backend_routes(), mock_routes()
    print("[parity] backend routes %d | mock routes %d | frame keys backend %d / mock %d"
          % (len(real), len(mock), len(real_frame_keys()), len(mock_frame_keys())))
    if args.report:
        for key, why in sorted(KNOWN_DRIFT.items()):
            print("[parity] known drift  %s %s: %s" % (key[0], key[1], why))
        print("[parity] backend routes with no mock: %d (allowed: the mock is a subset)" % len(real - mock))
    for (kind, key), detail in sorted(new.items()):
        print("[parity] DRIFT  %-9s %s: %s" % (kind, key, detail))
    for kind, key in stale:
        print("[parity] STALE allow-list entry (no longer drifts - delete it from KNOWN_DRIFT): %s %s" % (kind, key))
    if new or stale:
        return 1
    print("[parity] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
