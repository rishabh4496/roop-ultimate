"""A hook's ARGUMENTS must not name something declared further down.

`test_ui_hook_order` guards the other direction — a hook's RESULT read before
the hook runs. This is the mirror image, and it is not covered there:

    const { desktopAlerts } = useRunCompleteAlert({
      processing: progress.processing, error: progress.error, notify,   // line 240
    });
    ...
    const notify = useCallback(..., []);                                // line 517

`const` is not hoisted. A React component body runs top to bottom on every
render, so this throws "Cannot access 'notify' before initialization" on the
FIRST render. There is no ErrorBoundary above the shell, so App taking that
throw is a completely blank page.

This shipped. It survived `vite build` (esbuild does no scope analysis),
`oxlint --deny no-undef` (the name DOES resolve lexically — the problem is
temporal, not lexical), 684 green Python tests, and a read-through. The only
thing that showed it was opening the app.

WHAT IS SCANNED, and why it is narrow. Only hook calls whose first argument is
an OBJECT LITERAL — `useSomething({ a, b })` — and only that literal.

The first version of this scanned every hook argument, skipping arrow-function
bodies so that `useEffect(() => { ... })` would not be flagged for naming a
later const inside a callback that runs after the body finishes. Finding the end
of those bodies means matching braces through real JSX, template literals and
regex literals, and it desynchronised: one `useMemo` in Timeline.jsx swallowed
ninety lines and reported fifteen identifiers that were all inside the callback.
Fifteen false positives is a guard nobody will keep, and a guard nobody keeps is
a guard that reports clean forever.

An object literal has no such ambiguity. It also happens to be exactly the shape
of the bug — a config-object hook lifted above one of the things it is
configured with. Dependency arrays are eagerly evaluated too and are NOT covered
here; `test_ui_hook_order` documents that hazard from the other direction.
"""

import os
import re
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')

HOOK_CALL = re.compile(r'(?<![\w.])(use[A-Z]\w*)\s*\(')
# `  const x =`, `  const { a, b } =`, `  const [a, b] =` at component top level.
DECL_PLAIN = re.compile(r'^  (?:const|let)\s+(\w+)\s*=')
DECL_OBJ = re.compile(r'^  (?:const|let)\s*\{([^}]*)\}\s*=')
DECL_ARR = re.compile(r'^  (?:const|let)\s*\[([^\]]*)\]\s*=')
IDENT = re.compile(r'(?<![\w.$])([a-zA-Z_$][\w$]*)')
# `name:` — a property key, which names nothing. Shorthand `{ name }` has no
# colon and is left alone, which is the case that matters.
KEY = re.compile(r'(?<![\w.$])[a-zA-Z_$][\w$]*\s*:')

# Names that are never component-local bindings.
KEYWORDS = {
    'true', 'false', 'null', 'undefined', 'new', 'typeof', 'void', 'return',
    'await', 'async', 'function', 'const', 'let', 'var', 'if', 'else', 'try',
    'catch', 'finally', 'throw', 'delete', 'in', 'of', 'instanceof', 'this',
    'document', 'window', 'localStorage', 'Math', 'JSON', 'Date', 'Object',
    'Array', 'String', 'Number', 'Boolean', 'Promise', 'Set', 'Map', 'parseInt',
    'parseFloat', 'isNaN', 'setTimeout', 'clearTimeout', 'setInterval',
    'clearInterval', 'requestAnimationFrame', 'console', 'Notification',
}


def _components():
    for root, _dirs, names in os.walk(os.path.join(SRC)):
        for name in sorted(names):
            if name.endswith(('.jsx', '.js')):
                yield os.path.join(root, name)


def _rel(path):
    return os.path.relpath(path, SRC).replace('\\', '/')


def _strip_comment(line):
    return line.split('//', 1)[0]


def _object_literal(src, brace_idx):
    """Text inside the object literal whose '{' is at brace_idx, or None.

    Brace depth only, with string and template-literal contexts skipped. That is
    enough here precisely because the scan is limited to object literals — the
    JSX, regex and arrow-body ambiguity that broke the general version cannot
    appear between these braces without a nested function, and a nested function
    body is deferred anyway.
    """
    depth, i = 0, brace_idx
    while i < len(src):
        c = src[i]
        if c in '{([':
            depth += 1
        elif c in '})]':
            depth -= 1
            if depth == 0:
                return src[brace_idx + 1:i]
        elif c in "\"'`":
            q, i = c, i + 1
            while i < len(src) and src[i] != q:
                i += 2 if src[i] == '\\' else 1
        elif c == '=' and src.startswith('=>', i):
            return None          # a callback snuck in; not our shape
        i += 1
    return None


def _declarations(body_lines):
    """name -> line index, for component-top-level const/let declarations."""
    decls = {}
    for i, raw in enumerate(body_lines):
        line = _strip_comment(raw)
        m = DECL_PLAIN.match(line)
        if m:
            decls.setdefault(m.group(1), i)
        m = DECL_OBJ.match(line)
        if m:
            for part in m.group(1).split(','):
                nm = part.split(':')[-1].split('=')[0].strip()
                if nm:
                    decls.setdefault(nm, i)
        m = DECL_ARR.match(line)
        if m:
            for part in m.group(1).split(','):
                nm = part.split('=')[0].strip()
                if nm:
                    decls.setdefault(nm, i)
    return decls


def scan(path):
    """[(hook, name, used_line, declared_line)] for this file."""
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    if 'export default function' not in src:
        return []
    body = src.split('export default function', 1)[1]
    lines = body.split('\n')
    decls = _declarations(lines)
    if not decls:
        return []

    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    def line_of(idx):
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo

    found = []
    for m in HOOK_CALL.finditer(body):
        hook = m.group(1)
        at = line_of(m.start())
        # Only calls made as component-top-level statements.
        stmt = lines[at]
        if not (stmt.startswith('  ') and not stmt.startswith('   ')):
            continue
        j = m.end()
        while j < len(body) and body[j] in ' \n':
            j += 1
        if j >= len(body) or body[j] != '{':
            continue                      # not a config-object hook — see module docstring
        chunk = _object_literal(body, j)
        if chunk is None:
            continue
        # Drop KEYS. `useState({ start: 1, end: 1 })` names `start`, but as a
        # property name, not as a read — and `start` happens to be a function
        # declared 700 lines later in FaceSwap. Shorthand survives on purpose:
        # `{ notify }` IS a read, and is exactly the bug this exists for.
        chunk = KEY.sub('', chunk)
        for name in {i for i in IDENT.findall(chunk) if i not in KEYWORDS}:
            where = decls.get(name)
            if where is not None and where > at:
                found.append((hook, name, at, where))
    return found


class HookArgumentsAreDeclaredFirst(unittest.TestCase):
    def test_no_component_reads_a_later_declaration_from_a_hook_call(self):
        offenders = []
        for path in _components():
            for hook, name, used, declared in scan(path):
                offenders.append(
                    f'{_rel(path)}: {hook}(...) reads `{name}` at body line '
                    f'{used}, which is declared at line {declared}')
        self.assertFalse(
            offenders,
            'a hook argument names something declared further down; `const` is '
            'not hoisted, so this throws on the first render and blanks the '
            'whole page:\n  ' + '\n  '.join(offenders))


class TheScannerActuallyCatchesIt(unittest.TestCase):
    """A source-scanning guard that cannot fail is worse than none — it reports
    clean forever. These pin it against the real bug and its near misses."""

    def _scan_text(self, text):
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.jsx', delete=False,
                                         encoding='utf-8') as fh:
            fh.write(text)
            tmp = fh.name
        try:
            return scan(tmp)
        finally:
            os.unlink(tmp)

    def test_it_catches_the_bug_that_shipped(self):
        hits = self._scan_text(
            'export default function App() {\n'
            '  const { a } = useRunCompleteAlert({ error: e, notify });\n'
            '  const notify = useCallback(() => {}, []);\n'
            '}\n')
        self.assertTrue(any(n == 'notify' for _h, n, _u, _d in hits),
                        'the scanner misses the exact case it was written for')

    def test_the_right_order_is_clean(self):
        self.assertFalse(self._scan_text(
            'export default function App() {\n'
            '  const notify = useCallback(() => {}, []);\n'
            '  const { a } = useRunCompleteAlert({ error: e, notify });\n'
            '}\n'))

    def test_a_deferred_callback_body_is_not_flagged(self):
        """useEffect's callback runs after the body finishes, so naming a later
        const inside it is legal and extremely common."""
        self.assertFalse(self._scan_text(
            'export default function App() {\n'
            '  useEffect(() => { doThing(later); }, []);\n'
            '  const later = 1;\n'
            '}\n'))

    def test_a_dependency_array_is_a_documented_non_goal(self):
        """Eagerly evaluated, so it CAN throw the same way — but finding it
        means parsing past a callback body, which is what made the general
        scanner unusable. Recorded as a known gap rather than left implicit."""
        self.assertFalse(self._scan_text(
            'export default function App() {\n'
            '  useEffect(() => { go(); }, [later]);\n'
            '  const later = 1;\n'
            '}\n'))


if __name__ == '__main__':
    unittest.main()
