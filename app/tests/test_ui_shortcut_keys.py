"""No two components may claim the same bare keyboard shortcut.

Every keydown handler in the React UI is registered on `window`, and
`preventDefault()` does not stop a sibling listener on the same target — both
run. So a letter claimed in two components is not "one wins": it is BOTH, every
time, and the failure mode depends on what the two do:

  * If they drive the same state through an updater, they cancel out. `C` was
    bound by FaceSwap (setCompare) and by InteractivePreview (onToggleCompare,
    which is the same setter), so the app's documented compare shortcut flipped
    the flag and flipped it straight back — it did nothing at all whenever the
    normal preview stage was mounted.
  * Otherwise both effects land. `M` set a timeline marker AND opened the
    magnifier; `H` opened the shortcuts HUD AND flipped the compare split axis
    underneath it.

None of that is visible in a diff of either file, and neither lint nor the build
can see it, because each handler is correct on its own. So the invariant is
checked here instead: collect the bare keys each keydown handler claims and
require the claims to be disjoint.

Excluded, deliberately:
  * handlers that require Ctrl/Cmd (App.jsx's palette and page zoom) — a bare
    letter and its Ctrl-modified twin are different shortcuts, and every bare
    handler returns early on a modifier;
  * confirm.jsx, which is only mounted while a modal is open and which every
    other handler explicitly stands down for (`[role="dialog"]`).
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')

# Only mounted behind a modal that the other handlers already yield to.
EXEMPT_FILES = {'confirm.jsx'}

# A handler that cannot fire without a modifier does not compete for the bare key.
MODIFIER_REQUIRED = re.compile(r'if\s*\(\s*!\s*\(\s*e\.(metaKey|ctrlKey)')

# `e.key === 'x'`, `e.key.toLowerCase() === 'x'`, `e.key == "ArrowLeft"` — and
# the negated guard form, `if (e.key.toLowerCase() !== 'm' || …) return;`, which
# is how Timeline claims its marker key. Both directions name the key the
# handler acts on.
KEY_TEST = re.compile(r"""e\.key(?:\.toLowerCase\(\))?\s*[!=]==?\s*['"]([^'"]+)['"]""")

# `window.addEventListener('keydown', handleKeyDown)` -> the handler's name.
REGISTERED = re.compile(r"""window\.addEventListener\(\s*['"]keydown['"]\s*,\s*(\w+)""")


def _jsx_files():
    for root, _dirs, names in os.walk(SRC):
        for name in names:
            if name.endswith(('.jsx', '.js')):
                yield os.path.join(root, name)


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


def _claimed_keys(src):
    """The bare keys this file binds on WINDOW keydown.

    Scoped to the registered handler's own body on purpose. A whole-file scan
    also picks up element-level `onKeyDown` on text inputs — Enter/Escape to
    commit or cancel a field — which never compete for a global shortcut."""
    keys = set()
    for handler in set(REGISTERED.findall(src)):
        for body in _function_bodies(src, handler):
            if MODIFIER_REQUIRED.search(body):
                continue
            keys.update(k.lower() if len(k) == 1 else k for k in KEY_TEST.findall(body))
    return keys


class ShortcutKeysAreUnique(unittest.TestCase):
    def test_no_two_components_claim_the_same_key(self):
        claims = {}
        for path in _jsx_files():
            name = os.path.basename(path)
            if name in EXEMPT_FILES:
                continue
            with open(path, encoding='utf-8') as fh:
                keys = _claimed_keys(fh.read())
            if keys:
                claims[name] = keys

        self.assertGreaterEqual(len(claims), 3,
                                'the scan found almost no keydown handlers — the '
                                'regex has probably drifted from the source')

        owners = {}
        for name, keys in claims.items():
            for key in keys:
                owners.setdefault(key, []).append(name)

        clashes = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
        self.assertEqual(clashes, {},
                         'these keys are bound by more than one window keydown '
                         'handler, so every one of them fires: ' + repr(clashes))

    def test_the_compare_shortcut_has_exactly_one_owner(self):
        """The specific regression: `C` toggling compare twice per press."""
        owners = []
        for path in _jsx_files():
            name = os.path.basename(path)
            if name in EXEMPT_FILES:
                continue
            with open(path, encoding='utf-8') as fh:
                if 'c' in _claimed_keys(fh.read()):
                    owners.append(name)
        self.assertEqual(owners, ['FaceSwap.jsx'],
                         'C must be owned by FaceSwap alone — a second binding '
                         'that calls the same setCompare updater cancels it out')


if __name__ == '__main__':
    unittest.main()
