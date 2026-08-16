"""Every `size`/`variant` a call site asks a primitive for must actually exist.

The shared primitives pick their classes out of a lookup:

    className={`… ${variants[variant]} ${sizes[size]} …`}

which means an unknown key is not an error. `sizes['xs']` is `undefined`, the
template stringifies it, and the element receives a class literally named
"undefined" plus NONE of the padding, font-size or radius it was supposed to
get. That was live: two `<Button size="xs">` call sites existed while the map
only defined sm/md/lg, so both rendered unpadded with square corners.

Nothing catches this. It is valid JavaScript, valid JSX, and Tailwind is happy
to not emit a rule for a class nobody defined — so the build passes, oxlint
passes, and the only symptom is a control that looks slightly wrong in one
corner of one panel.

The component now falls back to a default, so a typo degrades gracefully rather
than losing all styling. This test is the other half: it fails the build so the
typo gets FIXED instead of silently landing on the fallback.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
UI = os.path.join(SRC, 'components', 'ui.jsx')

# `<Button … size="sm" …>` across newlines, capturing the prop value.
def _usages(src, component, prop):
    pattern = re.compile(
        r'<' + component + r'\b(?P<attrs>(?:[^<>]|\n)*?)/?>', re.S)
    prop_re = re.compile(prop + r'="([a-zA-Z0-9_]+)"')
    for m in pattern.finditer(src):
        for pm in prop_re.finditer(m.group('attrs')):
            yield pm.group(1), src[:m.start()].count('\n') + 1


def _defined_keys(ui_src, component, obj_name):
    """The keys of e.g. the `sizes` object inside the Button component."""
    body = ui_src.split(f'export const {component} =', 1)[1]
    body = body.split('export const', 1)[0]
    obj = body.split(f'const {obj_name} = {{', 1)[1]
    depth, i = 1, 0
    while i < len(obj) and depth:
        if obj[i] == '{':
            depth += 1
        elif obj[i] == '}':
            depth -= 1
        i += 1
    return set(re.findall(r'^\s*(\w+):', obj[:i], re.M))


def _module_object_keys(ui_src, obj_name):
    """The keys of a module-level object literal, e.g. `const ELEVATION = {…}`.

    Separate from `_defined_keys` because a lookup shared by two components
    (Card and the Section that forwards to it) belongs at module scope, not
    inside either one's body.
    """
    obj = ui_src.split(f'const {obj_name} = {{', 1)[1]
    depth, i = 1, 0
    while i < len(obj) and depth:
        if obj[i] == '{':
            depth += 1
        elif obj[i] == '}':
            depth -= 1
        i += 1
    return set(re.findall(r'^\s*(\w+):', obj[:i], re.M))


def _jsx_files():
    for root, _dirs, names in os.walk(SRC):
        for name in sorted(names):
            if name.endswith('.jsx'):
                yield os.path.join(root, name)


class PrimitiveProps(unittest.TestCase):
    def _check(self, component, prop, obj_name):
        with open(UI, encoding='utf-8') as fh:
            defined = _defined_keys(fh.read(), component, obj_name)
        self.assertTrue(defined, f'parsed no {obj_name} keys out of {component}')

        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            for value, line in _usages(src, component, prop):
                if value not in defined:
                    rel = os.path.relpath(path, SRC).replace('\\', '/')
                    offenders.append(f'{rel}:{line} {prop}="{value}"')
        self.assertEqual(
            offenders, [],
            f'{component} has no such {prop} — the lookup yields undefined and '
            f'the element loses every class it should have had. Defined: '
            f'{sorted(defined)}. Offenders: ' + ', '.join(offenders))

    def test_button_sizes_exist(self):
        self._check('Button', 'size', 'sizes')

    def test_button_variants_exist(self):
        self._check('Button', 'variant', 'variants')

    def test_card_elevations_exist(self):
        """`elevation` decides whether a surface tilts under the cursor.

        A typo here does not lose styling the way a bad `size` does — it falls
        back to `panel`, which looks entirely reasonable. That is worse, not
        better: a media tile meant to read as `hero` would just quietly stop
        doing so, and nothing would ever point at the line.
        """
        with open(UI, encoding='utf-8') as fh:
            defined = _module_object_keys(fh.read(), 'ELEVATION')
        self.assertTrue(defined, 'parsed no ELEVATION keys out of ui.jsx')

        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            # Section forwards the prop straight to Card, so both are checked.
            for component in ('Card', 'Section'):
                for value, line in _usages(src, component, 'elevation'):
                    if value not in defined:
                        rel = os.path.relpath(path, SRC).replace('\\', '/')
                        offenders.append(f'{rel}:{line} elevation="{value}"')
        self.assertEqual(
            offenders, [],
            'Card has no such elevation, so this surface silently falls back '
            f'to "panel". Defined: {sorted(defined)}. Offenders: '
            + ', '.join(offenders))


if __name__ == '__main__':
    unittest.main()
