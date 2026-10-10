"""The React UI's API surface: what it calls, what exists, what the README says.

`react-ui/README.md` carried an API table of about a dozen rows while the UI calls
126 distinct endpoints, and its tab list described four of nine tabs. A hand-kept
list of this kind is wrong the day after it is written, and nothing fails when it
is. So the three sets are tied together here:

  * every `/api/...` and `/ws/...` path in `react-ui/src` is a route the backend
    registers (a typo, or a route renamed on the server, is a request that 404s
    only when somebody clicks the button -- the mock server in e2e/ would never
    notice, because it serves whatever it is told to);
  * every path the UI calls is listed in the README;
  * every path the README lists is a real route.

The extraction is static (regexes over source, like the other test_ui_* files):
routes come from FastAPI decorators with `APIRouter(prefix=...)` joined on, and UI
paths from string/template literals with `${...}` segments treated as wildcards.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
SRC = os.path.join(ROOT, 'react-ui', 'src')
README = os.path.join(ROOT, 'react-ui', 'README.md')

SKIP_DIRS = {'env', 'tests', '__pycache__', 'node_modules', 'projects'}

ROUTER_DEF = re.compile(r'^(\w+)\s*=\s*APIRouter\(([^)]*)\)', re.M)
ROUTE_DECORATOR = re.compile(
    r'@(\w+)\.(?:get|post|put|delete|patch|websocket|api_route)\(\s*(?:path\s*=\s*)?f?["\']([^"\']*)["\']')
# `/api/target/preview`, `/ws/frames`, `/api/projects/${id}/${kind}` -- anchored so
# `http://host/api/x` inside prose or a URL constant is still read, but
# `something/api/x` glued to an identifier is not.
SEG = r'(?:[A-Za-z0-9_\-]+|\$\{[^}]*\})'
PATH = re.compile(r'(?<![A-Za-z0-9_.])(/(?:api|ws)(?:/' + SEG + r')+)')


def _walk(root, exts):
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith(exts):
                yield os.path.join(dirpath, name)


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _strip_js_comments(src):
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    # A line comment starts at `//` after whitespace or at the start of the line. Not
    # after `}` or `:` -- `${proto}//${host}/ws/frames` and `http://x` are code, and
    # treating them as comments dropped every WebSocket URL from the scan.
    return '\n'.join(re.sub(r'(^|\s)//.*$', r'\1', line) for line in src.split('\n'))


def _norm(path):
    return re.sub(r'\$\{[^}]*\}|\{[^}]*\}', '{}', path).rstrip('/')


def backend_routes():
    routes = set()
    for path in _walk(APP, ('.py',)):
        src = _read(path)
        prefixes = {}
        for m in ROUTER_DEF.finditer(src):
            pm = re.search(r'prefix\s*=\s*["\']([^"\']*)["\']', m.group(2))
            prefixes[m.group(1)] = pm.group(1) if pm else ''
        for m in ROUTE_DECORATOR.finditer(src):
            routes.add(_norm(prefixes.get(m.group(1), '') + m.group(2)) or '/')
    return routes


def ui_paths():
    """{normalised path: sorted files}. `{}` marks a dynamic segment."""
    found = {}
    for path in _walk(SRC, ('.js', '.jsx', '.ts', '.tsx')):
        for m in PATH.finditer(_strip_js_comments(_read(path))):
            found.setdefault(_norm(m.group(1)), set()).add(os.path.relpath(path, SRC).replace('\\', '/'))
    return {p: sorted(f) for p, f in found.items()}


def _matches(path, routes):
    """A UI path matches a route when they agree segment by segment, `{}` being a wildcard on either side."""
    parts = path.split('/')
    for route in routes:
        rp = route.split('/')
        if len(rp) == len(parts) and all(a == b or a == '{}' or b == '{}' for a, b in zip(parts, rp)):
            return True
    return False


def readme_paths():
    # Only the API section: the rest of the README mentions /api in prose (ports, proxy).
    text = _read(README)
    head = text.find('## API')
    assert head >= 0, 'react-ui/README.md has no "## API" section'
    section = text[head:]
    nxt = re.search(r'\n## ', section[3:])
    if nxt:
        section = section[:nxt.start() + 3]
    return {_norm(m.group(1)) for m in PATH.finditer(section)}


class UiApiSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes = backend_routes()
        cls.ui = ui_paths()

    def test_the_scan_found_something(self):
        # A regex that drifted from the source would make every other test vacuous.
        self.assertGreater(len(self.routes), 100, 'found almost no backend routes')
        self.assertGreater(len(self.ui), 100, 'found almost no UI endpoint paths')
        self.assertTrue({'/ws/telemetry', '/ws/frames', '/ws/angle-scan'} <= set(self.ui),
                        'the WebSocket endpoints are built from template strings; the scan lost them')

    def test_every_endpoint_the_ui_calls_is_a_backend_route(self):
        missing = {p: f for p, f in self.ui.items() if not _matches(p, self.routes)}
        self.assertEqual(missing, {}, 'the UI calls routes the backend does not register: ' + repr(missing))

    def test_the_readme_lists_every_endpoint_the_ui_calls(self):
        listed = readme_paths()
        concrete = {p for p in self.ui if '{}' not in p}
        absent = sorted(p for p in concrete if p not in listed)
        self.assertEqual(absent, [], 'react-ui/README.md "## API" is missing endpoints the UI calls: ' + ', '.join(absent))

    def test_the_readme_lists_only_real_endpoints(self):
        stale = sorted(p for p in readme_paths() if not _matches(p, self.routes))
        self.assertEqual(stale, [], 'react-ui/README.md "## API" lists endpoints that do not exist: ' + ', '.join(stale))


if __name__ == '__main__':
    unittest.main()
