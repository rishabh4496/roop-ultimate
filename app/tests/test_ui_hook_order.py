"""A hook's results must be declared before anything reads them.

This guards the specific way extracting a hook from a large component goes
wrong, which happened while doing exactly that to FaceSwap.jsx.

When state moves into a hook, the natural place to put the call is where the
LOGIC lived — down among the effects that use it. But `const` is not hoisted the
way `var` is: a reference that executes before the declaration throws
"Cannot access before initialization". And in a React component the body runs
top to bottom on every render, so:

    useEffect(() => { ... }, [isPlaying]);      // line 1127
    const scrubbing = isScrubbing || isPlaying; // line 1380
    ...
    const { isPlaying } = usePlaybackBuffer();  // line 1445  <-- too late

is a hard throw on the FIRST render. A dependency array is not deferred like
the effect callback is — it is an argument, evaluated as the body runs — so
even the array alone is enough to break it.

Nothing else catches this:

  * Vite/esbuild do no scope analysis, so it builds cleanly.
  * `oxlint --deny no-undef` passes, because the name DOES resolve lexically —
    the problem is temporal, not lexical.
  * It is invisible in review, because the declaration and the use can be a
    thousand lines apart in a file this size.

The symptom is the whole tab failing to render behind the ErrorBoundary, which
is at least loud — but only if someone opens that tab.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')

# `const { a, b: c } = useSomething(` — the destructured result of a hook call.
# Also matches when the `=` and the call are on different lines.
DESTRUCTURE = re.compile(
    r'const\s*\{(?P<names>[^}]*)\}\s*=\s*(?P<call>use[A-Z]\w*)\s*\(', re.S)

# `a` or `a: b` inside the destructure — the BOUND name is what matters.
BINDING = re.compile(r'(?:\w+\s*:\s*)?(\w+)')


def _components():
    for root, _dirs, names in os.walk(os.path.join(SRC, 'components')):
        for name in sorted(names):
            if name.endswith('.jsx'):
                yield os.path.join(root, name)


def _strip_comments(line):
    """Drop a trailing line comment so prose about a name is not a 'use'."""
    return line.split('//', 1)[0]


# A top-level declaration: where one component (or hook) body can begin.
# The TDZ hazard is per FUNCTION BODY, so the scan for early uses starts at the
# top-level declaration that encloses the hook call, not at line 1. A file with
# one component (FaceSwap.jsx) scans exactly as before; a file with several
# (Processing.jsx's live children each call useLiveRun) is not charged with a
# SIBLING component's use of the same names.
TOP_LEVEL = re.compile(r'^(?:export\s+(?:default\s+)?)?(?:function\b|const\s+\w+\s*=)')


def _enclosing_start(lines, decl_line):
    """1-based line of the last top-level declaration at or before decl_line."""
    for i in range(decl_line - 1, 0, -1):
        if TOP_LEVEL.match(lines[i - 1]):
            return i
    return 1


def _offenders(src, rel):
    """Every read of a hook's destructured result above its declaration,
    within the enclosing top-level function."""
    found = []
    lines = src.split('\n')
    for m in DESTRUCTURE.finditer(src):
        # Line the `const {` sits on — everything at or after it is fine.
        decl_line = src[:m.start()].count('\n') + 1
        names = {b for b in BINDING.findall(m.group('names')) if b}

        first = _enclosing_start(lines, decl_line)
        for i, raw in enumerate(lines[first - 1:decl_line - 1], start=first):
            stripped = raw.strip()
            if stripped.startswith(('//', '*', '/*')):
                continue
            code = _strip_comments(raw)
            for n in sorted(names):
                # Not a property access (`x.isPlaying`) and not a
                # substring of a longer identifier.
                if re.search(r'(?<![\w.$])' + re.escape(n) + r'(?![\w$])', code):
                    found.append(
                        f'{rel}:{i} uses `{n}` from {m.group("call")}(), '
                        f'which is declared at line {decl_line}')
    return found


class HookDeclarationOrder(unittest.TestCase):
    def test_the_checker_still_catches_a_read_above_the_hook(self):
        """Scoping the scan to one function must not blind it inside one."""
        bad = (
            "export default function Panel() {\n"
            "  useEffect(() => {}, [isPlaying]);\n"
            "  const { isPlaying } = usePlaybackBuffer();\n"
            "}\n"
        )
        self.assertEqual(len(_offenders(bad, 'x.jsx')), 1)

    def test_a_sibling_component_is_not_charged(self):
        ok = (
            "function A() {\n"
            "  const { prog } = useLiveRun();\n"
            "  return prog;\n"
            "}\n"
            "\n"
            "function B() {\n"
            "  const { prog } = useLiveRun();\n"
            "  return prog;\n"
            "}\n"
        )
        self.assertEqual(_offenders(ok, 'x.jsx'), [])

    def test_hook_results_are_declared_before_they_are_used(self):
        offenders = []

        for path in _components():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            rel = os.path.relpath(path, SRC).replace('\\', '/')
            offenders.extend(_offenders(src, rel))

        self.assertEqual(
            offenders, [],
            'these read a hook result before the line that declares it. `const` '
            'is not hoisted, and a React component body runs top to bottom on '
            'every render, so this throws "Cannot access before initialization" '
            'on the first render. A dependency array counts — it is an argument, '
            'evaluated with the body, not deferred like the effect callback. '
            'Move the hook call above its first use: '
            + '; '.join(offenders))


if __name__ == '__main__':
    unittest.main()
