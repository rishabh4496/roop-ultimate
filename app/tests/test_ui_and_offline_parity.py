"""React UI parity: the shell's routes, the Stage 22-27 tools and named controls, their backend bindings, and the
offline contract of the production bundle.

What each group proves, and how far it reaches:

  routes        App.jsx registers the nine shell routes. The runtime part is the Playwright spec
                react-ui/e2e/route-mount.spec.js, run here through the repo's own config (production build,
                vite preview, mock backend): each route mounts with no uncaught exception, no console.error, no
                error-boundary fallback, and no request that leaves the machine.
  tools         Each Stage 22-27 tool exists, is reachable from main.jsx (scripts/unreachable-modules.mjs),
                and the sweep of every /api/ path the UI calls resolves to a backend route
                (scripts/api-callsites.mjs and the path check below).
  controls      Each named control is present in the source. A control that is not there is a strict xfail
                with the reason, never a silent pass.
  offline       The production bundle contains no URL that a browser would fetch. Namespace identifiers,
                an error-message link and a CSS comment are text, not requests, and are allowlisted by name.

The Playwright and Node steps skip (they never pass) when Node or the built bundle is absent.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
UI = REPO / "react-ui"
SRC = UI / "src"
DIST = UI / "dist"

ROUTES = ("home", "faceswap", "batch", "processing", "facemgr", "extras", "gallery", "history", "settings")


def _read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _node():
    env = os.environ.get("ROOP_NODE")
    if env and Path(env).is_file():
        return env
    found = shutil.which("node")
    if found:
        return found
    for cand in (r"C:\Program Files\nodejs\node.exe", "/c/Program Files/nodejs/node.exe"):
        if Path(cand).is_file():
            return cand
    return None


def _run_node_script(script):
    node = _node()
    if node is None:
        pytest.skip("Node is not installed; the UI gate cannot run here")
    if not (UI / "node_modules").is_dir():
        pytest.skip("react-ui/node_modules is missing; run npm install")
    proc = subprocess.run([node, script], cwd=UI, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    return proc.stdout


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def _app_tab_ids():
    text = _read(SRC / "App.jsx")
    block = re.search(r"const ALL_TABS = \[(.*?)\n\];", text, re.S)
    assert block, "ALL_TABS not found in App.jsx"
    return re.findall(r"\{ id: '([a-z]+)'", block.group(1))


def test_app_shell_registers_the_nine_routes():
    assert tuple(_app_tab_ids()) == ROUTES


def test_route_mount_suite_passes_in_a_real_browser():
    """Each route mounts with no runtime, console or foreign-network error (react-ui/e2e/route-mount.spec.js).
    Runs the repo's own Playwright config: production build, vite preview, and the mock backend."""
    node = _node()
    if node is None:
        pytest.skip("Node is not installed")
    cli = UI / "node_modules" / "@playwright" / "test" / "cli.js"
    if not cli.is_file():
        pytest.skip("@playwright/test is not installed in react-ui/node_modules")
    if not (DIST / "index.html").is_file():
        pytest.skip("react-ui/dist is not built; run npm run build")
    proc = subprocess.run(
        [node, str(cli), "test", "-c", "e2e/playwright.config.js", "route-mount", "--reporter=line"],
        cwd=UI, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-4000:]
    assert re.search(r"\b10 passed\b", out), out[-1500:]


def test_unreachable_module_gate_passes():
    """Every React module is reachable from main.jsx, so no tool is shipped as dead code."""
    _run_node_script("scripts/unreachable-modules.mjs")


def test_api_callsite_gate_passes():
    _run_node_script("scripts/api-callsites.mjs")


# --------------------------------------------------------------------------- #
# tools (Stage 22-27)
# --------------------------------------------------------------------------- #
TOOLS = [
    ("Curve Editor", "components/faceswap/curves/CurveEditor.jsx", r"CurveEditor"),
    ("3x3 Sweep contact sheet", "components/faceswap/ParameterSweepModal.jsx", r"3x3 Contact Sheet"),
    ("Node Flow Bar", "components/faceswap/PipelineNodeFlow.jsx", r"PipelineNodeFlow"),
    ("Snapshot Slots", "components/faceswap/SnapshotSlots.jsx", r"setReference"),
    ("Pre-flight Linter", "components/faceswap/PreFlightLinter.jsx", r"/api/render/preflight"),
    ("Frame Flamegraph", "components/faceswap/FrameFlamegraph.jsx", r"FrameFlamegraph"),
    ("LUT bake", "components/faceswap/ColorGradingPanel.jsx", r"Bake LUT into Hardware Encode"),
]


@pytest.mark.parametrize("name,rel,marker", TOOLS, ids=[t[0] for t in TOOLS])
def test_stage_tool_is_present_and_marked(name, rel, marker):
    path = SRC / rel
    assert path.is_file(), f"{name}: {rel} is missing"
    assert re.search(marker, _read(path)), f"{name}: marker {marker!r} not found in {rel}"


# Backend bindings the tools depend on, checked against the Python routes below.
TOOL_ENDPOINTS = {
    "3x3 Sweep": ("components/faceswap/ParameterSweepModal.jsx", "/api/preview/sweep"),
    "Pre-flight Linter (render)": ("components/faceswap/PreFlightLinter.jsx", "/api/render/preflight"),
    "Pre-flight Linter (face radar)": ("components/faceswap/PreFlightLinter.jsx", "/api/facemgr/radar"),
}


@pytest.mark.parametrize("label", sorted(TOOL_ENDPOINTS))
def test_stage_tool_endpoint_is_called_and_served(label):
    rel, endpoint = TOOL_ENDPOINTS[label]
    assert endpoint in _read(SRC / rel), f"{label}: {rel} does not call {endpoint}"
    assert endpoint in _backend_routes_as_text(), f"{label}: no backend route serves {endpoint}"


# --------------------------------------------------------------------------- #
# named controls
# --------------------------------------------------------------------------- #
CONTROLS = [
    ("Pixel Boost", "components/faceswap/curves/curveModel.js", r"label:\s*'Pixel Boost'"),
    ("LivePortrait expression transfer", "components/FaceSwap.jsx",
     r'label="Expression transfer strength"'),
    ("UltraMax restorer bypass ramp", "components/faceswap/studioControls.js", r"wire:\s*'ultramax_ramp'"),
    ("UltraMax temporal restore alpha", "components/faceswap/studioControls.js",
     r"wire:\s*'temporal_restore_alpha'"),
    ("UltraMax restore identity weight", "components/faceswap/studioControls.js",
     r"wire:\s*'restore_identity_weight'"),
    ("ByteTrack track lanes", "components/faceswap/NleTimeline.tsx", r"trackLanes\?:"),
    ("VFR PTS-to-frame map", "utils/ptsFrameMap.ts", r"export (function|const) createPtsFrameMap"),
]


@pytest.mark.parametrize("name,rel,marker", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_named_control_is_present(name, rel, marker):
    path = SRC / rel
    assert path.is_file(), f"{name}: {rel} is missing"
    assert re.search(marker, _read(path)), f"{name}: marker {marker!r} not found in {rel}"


def test_studio_controls_are_consumed_by_a_component():
    """The UltraMax slider table is only a control if a component renders it."""
    importers = [p for p in SRC.rglob("*.jsx") if "studioControls" in _read(p)]
    importers += [p for p in SRC.rglob("*.tsx") if "studioControls" in _read(p)]
    assert any(p.name != "studioControls.js" for p in importers), "studioControls.js is imported by nothing"


def test_gnkps_detector_is_a_selectable_engine_end_to_end():
    """`gnkps` is a data-driven option of the Detector engine select, not a literal in React source.

    An earlier version of this test grepped the UI source for a `gnkps` control, found none, and recorded a
    strict xfail "not built". That was wrong, and the live UI showed it: the dropdown's options come from the
    backend (`options={meta.detector_engines || [...]}`), and the backend serves `gnkps` in that list. So the
    chain is asserted link by link: UI select bound to the backend list -> backend serves 'gnkps' -> the
    detector dispatch and model-file mapping honour it -> the weights exist (skipped where they are absent,
    because the GN-kps file is not redistributable and is fetched by hand).
    """
    face_swap = _read(SRC / "components" / "FaceSwap.jsx")
    assert re.search(r"options=\{\s*meta\.detector_engines\b", face_swap), \
        "the Detector engine select is no longer bound to meta.detector_engines"

    api = _read(APP / "api.py")
    listed = re.search(r'"detector_engines"\s*:\s*\[([^\]]*)\]', api)
    assert listed, "api.py no longer serves detector_engines"
    assert '"gnkps"' in listed.group(1), "api.py does not offer gnkps in detector_engines"

    face_util = _read(APP / "roop" / "face_util.py")
    assert "engine == 'gnkps'" in face_util, "face_util has no dispatch branch for the gnkps engine"
    retinaface = _read(APP / "roop" / "retinaface.py")
    assert "scrfd_10g_gnkps.onnx" in retinaface, "retinaface.py does not map gnkps to its weights file"

    weights = APP / "models" / "scrfd_10g_gnkps.onnx"
    if not weights.is_file():
        pytest.skip("scrfd_10g_gnkps.onnx is not present on this machine (not redistributable)")
    assert weights.stat().st_size > 1_000_000, "scrfd_10g_gnkps.onnx is present but implausibly small"


# --------------------------------------------------------------------------- #
# API paths: every /api/ literal the UI uses must have a backend route
# --------------------------------------------------------------------------- #
_ROUTER_DECL = re.compile(r"""(\w+)\s*=\s*APIRouter\(\s*(?:prefix\s*=\s*)?["']([^"']*)["']""")
_DECORATOR = re.compile(
    r"""@(\w+)\.(?:get|post|put|patch|delete|api_route)\(\s*["']([^"']*)["']""")


_PRUNE = {"env", "__pycache__", "tests", "node_modules", "models", "output", "temp", "facesets", "cache"}


def _backend_sources():
    """Every Python file under app/ that can declare a route (routers live in app/roop/api/ as well as app/)."""
    out = []
    for root, dirs, files in os.walk(APP):
        dirs[:] = [d for d in dirs if d not in _PRUNE and not d.startswith(".")]
        out.extend(Path(root, f) for f in files if f.endswith(".py"))
    return out


def _backend_routes():
    """Full /api/ paths served by the backend. A decorator on an APIRouter is joined to that router's prefix,
    so `@router.post("/start")` under `APIRouter(prefix="/api/autotune")` is /api/autotune/start."""
    routes = set()
    for path in _backend_sources():
        text = _read(path)
        prefixes = {name: prefix for name, prefix in _ROUTER_DECL.findall(text)}
        for var, rel in _DECORATOR.findall(text):
            full = prefixes.get(var, "") + rel
            if full.startswith("/api/") or full == "/api":
                routes.add(full.rstrip("/") or "/api")
    return routes


def _backend_routes_as_text():
    return "\n".join(sorted(_backend_routes()))


def _route_regex(route):
    parts = re.split(r"(\{[^}]+\})", route)
    out = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            out.append(".+" if ":path" in part else "[^/]+")
        else:
            out.append(re.escape(part))
    return re.compile("^" + "".join(out) + "$")


def _ui_api_paths():
    """Every /api/ path literal the UI builds. A template's ${...} becomes a placeholder segment, and anything
    after a query string or an unclosed ${ is dropped, so `/api/update/check${x ? ...}` checks /api/update/check."""
    literal = re.compile(r"""["'`](/api/[^"'`\s]*)""")
    found = {}
    for path in SRC.rglob("*"):
        if path.suffix not in (".js", ".jsx", ".ts", ".tsx"):
            continue
        code = "\n".join(line for line in _read(path).splitlines()
                         if not line.lstrip().startswith(("//", "*", "/*")))   # comments name paths, they do not call them
        for match in literal.findall(code):
            head = re.split(r"\?|\$\{(?![^}]*\})", match)[0]
            normal = re.sub(r"\$\{[^}]*\}", "x", head).rstrip("*/")
            if not normal or normal == "/api":
                continue
            found.setdefault(normal, path.relative_to(UI).as_posix())
    return found


def _verbs_called(rel, call):
    return set(re.findall(call + r"""\(\s*'(\w+)'""", _read(SRC / rel)))


# A UI path whose last segment is a runtime value cannot be matched literally. Each entry names the concrete
# paths the value can take, and the test checks the callers' verb set is exactly those verbs.
DYNAMIC_UI_PATHS = {
    # App.jsx: runControl(action) posts /api/${action}; the callers pass these three.
    "/api/x": (("App.jsx", "runControl"), ("/api/pause", "/api/resume", "/api/stop")),
    # ProjectsPanel.jsx: act(kind) posts /api/projects/<id>/${kind}; the callers pass these three.
    "/api/projects/x/x": (("components/faceswap/ProjectsPanel.jsx", "act"),
                          ("/api/projects/p/validate", "/api/projects/p/load", "/api/projects/p/resume")),
}


def test_every_ui_api_path_has_a_backend_route():
    routes = [_route_regex(r) for r in _backend_routes()]
    assert routes, "no backend /api/ routes were found; the scanner is broken"
    unserved = []
    for ui_path, where in sorted(_ui_api_paths().items()):
        if ui_path in DYNAMIC_UI_PATHS:
            continue
        if not any(r.match(ui_path) for r in routes):
            unserved.append(f"{ui_path}  ({where})")
    assert not unserved, "UI calls with no backend route:\n  " + "\n  ".join(unserved)


@pytest.mark.parametrize("ui_path", sorted(DYNAMIC_UI_PATHS))
def test_dynamic_ui_path_serves_every_verb_its_callers_pass(ui_path):
    (rel, caller), concrete = DYNAMIC_UI_PATHS[ui_path]
    routes = [_route_regex(r) for r in _backend_routes()]
    for path in concrete:
        assert any(r.match(path) for r in routes), f"{ui_path}: backend serves no route for {path}"
    passed = _verbs_called(rel, caller)
    expected = {p.rsplit("/", 1)[-1] for p in concrete}
    assert passed, f"{rel}: no {caller}('<verb>') call sites were found"
    assert passed <= expected, f"{rel} passes verbs the backend does not serve: {sorted(passed - expected)}"


# --------------------------------------------------------------------------- #
# offline: the production bundle
# --------------------------------------------------------------------------- #
# Text, not requests. Each is a namespace identifier, a link inside a minified error message, or a CSS comment.
TEXT_ONLY_HOSTS = {
    "www.w3.org": "XML/SVG/MathML namespace identifiers passed to createElementNS and setAttributeNS",
    "react.dev": "the error-decoder link inside a minified React error string",
    "tailwindcss.com": "a CSS comment in the generated stylesheet",
}
_URL = re.compile(r"https?://([^\s\"'`)<>\\/]+)")
_FETCHING = re.compile(
    r"""(?:src|href)\s*=\s*["']https?://|url\(\s*["']?https?://|(?:fetch|import|new\s+WebSocket|new\s+EventSource)\(\s*["'`]\s*(?:https?|wss?)://""",
    re.I)


def _dist_text_files():
    if not DIST.is_dir():
        pytest.skip("react-ui/dist is not built; run npm run build")
    return [p for p in DIST.rglob("*") if p.suffix in (".html", ".js", ".css", ".json", ".svg", ".mjs")]


def test_bundle_has_no_url_that_a_browser_would_fetch():
    offenders = []
    for path in _dist_text_files():
        text = _read(path)
        for match in _FETCHING.finditer(text):
            offenders.append(f"{path.relative_to(DIST).as_posix()}: {text[match.start():match.end() + 40]!r}")
    assert not offenders, "external fetch in the bundle:\n  " + "\n  ".join(offenders[:20])


def test_bundle_external_hosts_are_text_only_and_named():
    hosts = {}
    for path in _dist_text_files():
        for host in _URL.findall(_read(path)):
            if host in ("", ):
                continue
            hosts[host] = hosts.get(host, 0) + 1
    external = {h: n for h, n in hosts.items() if h not in ("localhost", "127.0.0.1")}
    unknown = sorted(set(external) - set(TEXT_ONLY_HOSTS))
    assert not unknown, f"unexplained external host(s) in the bundle: {unknown}"
