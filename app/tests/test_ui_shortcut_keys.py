"""No two keyboard actions that can be live together may share a key.

Every keydown handler in the React UI is registered on `window`, and
`preventDefault()` does not stop a sibling listener on the same target — both
run. So a key claimed twice is not "one wins": it is BOTH, every time, and the
failure mode depends on what the two do:

  * If they drive the same state through an updater, they cancel out. `C` was
    bound by FaceSwap (setCompare) and by InteractivePreview (onToggleCompare,
    which is the same setter), so the app's documented compare shortcut flipped
    the flag and flipped it straight back — it did nothing at all whenever the
    normal preview stage was mounted.
  * Otherwise both effects land. `M` set a timeline marker AND opened the
    magnifier (InteractivePreview matched `e.code === 'KeyM'`, which this test's
    old `e.key` scan could not see); `H` opened the shortcuts HUD AND flipped the
    compare split axis underneath it.

None of that is visible in a diff of either file, and neither lint nor the build
can see it, because each handler is correct on its own.

HOW IT IS CHECKED (since Stage 26)
----------------------------------
The keys are no longer literals scattered through nine handlers. Each is an
action in `react-ui/src/shortcuts/actions.js`, and the handlers ask
`eventMatches(e, 'action.id')`; a user can remap them in Settings. So the
invariant moved with them:

  1. the action table itself: no two actions whose scopes can be live together
     (`global` overlaps everything; `faceswap` and `monitor` are separate) share a
     default key;
  2. every action is actually CONSUMED by a handler. A row nothing reads is a
     cheat-sheet entry for a key that does nothing - the project's recurring
     failure of something reporting success while not running;
  3. the compare/curve keys still have exactly one owner (the old regression);
  4. nobody has gone around the registry: a window keydown handler that tests a
     literal letter/symbol is a hard-coded key the remapper cannot move, and a
     second claim on a key the table already gives to an action.

The registry's own logic (combos, conflicts, persistence) is covered by
`react-ui/.render-check/render-stage26-logic.mjs`.

History: at edfba3c this file failed - `Escape` was claimed by three files and `f`
by two. `f` was a false positive (the reference monitor is a separate window mode
that replaces the whole app); Escape is reserved and several layers dismissing what
is in front of them is intended (see RESERVED_KEYS).
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
ACTIONS_JS = os.path.join(SRC, 'shortcuts', 'actions.js')

# Only mounted behind a modal that the other handlers already yield to.
EXEMPT_FILES = {'confirm.jsx'}

# Keys that are not commands. Escape dismisses whatever is in front (a solo stage, a pasted-files notice, a modal): several
# layers reacting to it is the design, and the remapper refuses to bind it. Space appears in a guard that stops Play/Pause
# stealing the activation key from a focused button, and in the pan gesture - neither is an action.
RESERVED_KEYS = {'escape', ' '}

# One action per line in actions.js (its header says so): id, scope, keys.
ACTION_ROW = re.compile(
    r"""\{\s*id:\s*'(?P<id>[^']+)'.*?scope:\s*'(?P<scope>\w+)',\s*keys:\s*\[(?P<keys>[^\]]*)\]""")
KEY_LITERAL = re.compile(r"'((?:[^'\\]|\\.)*)'")

# `e.key === 'x'`, `e.key.toLowerCase() === 'x'`, `e.key == "ArrowLeft"`, and the negated guard form.
KEY_TEST = re.compile(r"""e\.key(?:\.toLowerCase\(\))?\s*[!=]==?\s*['"]([^'"]+)['"]""")
# `window.addEventListener('keydown', handleKeyDown)` -> the handler's name.
REGISTERED = re.compile(r"""window\.addEventListener\(\s*['"]keydown['"]\s*,\s*(\w+)""")
# A handler that cannot fire without a modifier does not compete for the bare key.
MODIFIER_REQUIRED = re.compile(r'if\s*\(\s*!\s*\(\s*e\.(metaKey|ctrlKey)')


def _actions():
    with open(ACTIONS_JS, encoding='utf-8') as fh:
        text = fh.read()
    rows = []
    for line in text.splitlines():
        m = ACTION_ROW.search(line)
        if m:
            keys = [k.replace('\\\\', '\\') for k in KEY_LITERAL.findall(m.group('keys'))]
            rows.append({'id': m.group('id'), 'scope': m.group('scope'), 'keys': keys})
    return rows


def _jsx_files():
    for root, _dirs, names in os.walk(SRC):
        for name in names:
            if name.endswith(('.jsx', '.js')):
                yield os.path.join(root, name)


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _overlap(a, b):
    return a == b or 'global' in (a, b)


def _function_bodies(src, name):
    """Every `const name = (…) => {…}` / `function name(…) {…}` body, brace-matched."""
    out = []
    for m in re.finditer(r'\b(?:const|let|var|function)\s+' + re.escape(name) + r'\b', src):
        start = src.find('{', m.end())
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(src)):
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    out.append(src[start:i + 1])
                    break
    return out


def _literal_claims(src):
    """The bare keys a file binds on WINDOW keydown by literal `e.key` tests (scoped to the registered handler's body, so
    element-level Enter/Escape on a text field are not counted)."""
    keys = set()
    for handler in set(REGISTERED.findall(src)):
        for body in _function_bodies(src, handler):
            if MODIFIER_REQUIRED.search(body):
                continue
            keys.update(k.lower() if len(k) == 1 else k.lower() for k in KEY_TEST.findall(body))
    return keys - RESERVED_KEYS


class ActionTableIsClean(unittest.TestCase):
    def setUp(self):
        self.actions = _actions()

    def test_the_scan_found_the_table(self):
        self.assertGreaterEqual(len(self.actions), 40,
                                'actions.js parsed to almost nothing - the one-action-per-line format has probably drifted')
        self.assertEqual(len({a['id'] for a in self.actions}), len(self.actions), 'duplicate action ids')

    def test_no_two_actions_that_can_be_live_together_share_a_default_key(self):
        owners = {}
        for a in self.actions:
            for key in a['keys']:
                owners.setdefault(key, []).append(a)
        clashes = {}
        for key, rows in owners.items():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if _overlap(rows[i]['scope'], rows[j]['scope']):
                        clashes.setdefault(key, set()).update((rows[i]['id'], rows[j]['id']))
        self.assertEqual(clashes, {}, 'these keys are the default of more than one action that can be live at once, '
                                      'so every one of them fires: ' + repr({k: sorted(v) for k, v in clashes.items()}))

    def test_m_is_the_marker_key_alone(self):
        """The old M clash: a timeline marker AND the magnifier. Only the marker owns it now."""
        owners = [a['id'] for a in self.actions if any(k.lower() == 'm' for k in a['keys'])]
        self.assertEqual(owners, ['timeline.marker'])


class EveryActionIsConsumed(unittest.TestCase):
    def test_every_action_is_read_by_a_handler(self):
        """A table row nothing reads documents a key that does nothing."""
        actions = _actions()
        sources = {}
        for path in _jsx_files():
            if os.path.abspath(path) == os.path.abspath(ACTIONS_JS):
                continue
            sources[path] = _read(path)
        slot_ids = {a['id'] for a in actions if a['id'].startswith('snapshot.slot')}
        slot_loop_users = [p for p, s in sources.items() if 'SLOT_ACTION_IDS' in s and 'eventMatches' in s]
        unread = []
        for a in actions:
            literal = f"'{a['id']}'"
            if any(literal in s and 'eventMatches' in s or (literal in s and 'eventFiresInText' in s) for s in sources.values()):
                continue
            if a['id'] in slot_ids and len(slot_loop_users) >= 2:    # Face Swap's preview and the Editor tab
                continue
            unread.append(a['id'])
        self.assertEqual(unread, [], 'these actions are in the table but no handler asks eventMatches() about them: ' + repr(unread))

    def test_compare_and_curve_keys_have_exactly_one_owner(self):
        """The specific regression: `C` toggling compare twice per press (two components calling one setCompare)."""
        for action in ('viewport.compare', 'timeline.curveEditor'):
            owners = sorted(os.path.basename(p) for p in _jsx_files()
                            if os.path.abspath(p) != os.path.abspath(ACTIONS_JS)
                            and f"eventMatches(e, '{action}')" in _read(p))
            self.assertEqual(owners, ['FaceSwap.jsx'],
                             f'{action} must be handled by FaceSwap alone - a second handler that reaches the same '
                             'setCompare updater cancels it out')


class NobodyBypassesTheRegistry(unittest.TestCase):
    def test_no_window_handler_tests_a_literal_key(self):
        """A literal `e.key === 'x'` in a window keydown handler is a key the remapper cannot move, and a claim the action
        table cannot see. Add an action to shortcuts/actions.js and ask eventMatches() instead."""
        offenders = {}
        for path in _jsx_files():
            name = os.path.basename(path)
            if name in EXEMPT_FILES:
                continue
            claims = _literal_claims(_read(path))
            if claims:
                offenders[name] = sorted(claims)
        self.assertEqual(offenders, {}, 'these files hard-code keys in a window keydown handler: ' + repr(offenders))


if __name__ == '__main__':
    unittest.main()
