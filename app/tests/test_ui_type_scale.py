"""The type scale is named, and nothing may quietly step outside it.

This UI is dense, and most of its text sits BELOW Tailwind's `text-xs` (12px)
where Tailwind offers no steps. With nothing filling that gap, 303 one-off
`text-[Npx]` values had accumulated across eleven sizes from 7px to 34px —
including 7px and 8px uppercase text, which is past legible. A scale nobody can
enumerate is not a scale, so the sizes are now named in the `@theme` block of
index.css and these two rules keep them that way.

Both failures they guard are silent:

  * A fresh `text-[13px]` renders perfectly and simply widens the ramp again,
    one author at a time. Nothing in a diff, in oxlint, or in the build says so.
  * A MISTYPED token is worse. Tailwind generates utilities on demand, so
    `text-micoro` matches no rule and emits no CSS at all — the element silently
    inherits whatever size its parent had. It looks like a styling accident, not
    a typo, and there is nothing to grep for after the fact.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
CSS = os.path.join(SRC, 'index.css')

ARBITRARY = re.compile(r'text-\[\d+(?:\.\d+)?px\]')
# `text-foo` as written in a className, minus the variant prefix.
TOKEN_USE = re.compile(r'(?<![\w-])text-([a-z][a-z0-9]*)(?![\w-])')
TOKEN_DEF = re.compile(r'--text-([a-z][a-z0-9]*)\s*:')

# Sizes Tailwind ships itself, which need no definition of ours.
BUILTIN = {'xs', 'sm', 'base', 'lg', 'xl'}
# `text-` is also the colour namespace, a few keyword utilities, and — inside
# the plain .js that builds the pop-out window's stylesheet — real CSS property
# names. None of those are sizes.
NOT_A_SIZE = {
    # CSS properties, written as CSS rather than as a class.
    'align', 'decoration', 'indent', 'transform', 'overflow', 'shadow',
    'rendering', 'emphasis', 'orientation', 'combine', 'underline',
    'left', 'right', 'center', 'justify', 'start', 'end',
    'wrap', 'nowrap', 'balance', 'pretty', 'clip', 'ellipsis',
    'white', 'black', 'transparent', 'current', 'inherit',
    # Colour utilities defined with `@utility` in index.css (not font sizes):
    # `text-accent` (readable accent ink), `text-person` (per-person colour ink).
    'accent', 'person',
    'red', 'green', 'blue', 'amber', 'emerald', 'rose', 'sky', 'slate',
    'zinc', 'gray', 'grey', 'neutral', 'stone', 'orange', 'yellow', 'lime',
    'teal', 'cyan', 'indigo', 'violet', 'purple', 'fuchsia', 'pink',
    'top', 'bottom', 'middle', 'super', 'sub',
}


def _jsx_files():
    # .tsx too: BiometricAngleHUD/*.tsx carried 36 arbitrary text-[9|10|11px]
    # values for as long as this scan skipped it.
    for root, _dirs, names in os.walk(SRC):
        for name in sorted(names):
            if name.endswith(('.jsx', '.js', '.tsx')):
                yield os.path.join(root, name)


def _rel(path):
    return os.path.relpath(path, SRC).replace('\\', '/')


class TypeScale(unittest.TestCase):
    def test_no_arbitrary_font_sizes(self):
        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            for m in ARBITRARY.finditer(src):
                line = src[:m.start()].count('\n') + 1
                offenders.append(f'{_rel(path)}:{line} {m.group(0)}')
        self.assertEqual(
            offenders, [],
            'use a named step from the @theme type scale in index.css instead '
            'of a one-off pixel size (add a step there if none fits): '
            + ', '.join(offenders))

    def test_every_size_token_used_is_defined(self):
        with open(CSS, encoding='utf-8') as fh:
            defined = set(TOKEN_DEF.findall(fh.read())) | BUILTIN

        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            for m in TOKEN_USE.finditer(src):
                name = m.group(1)
                if name in defined or name in NOT_A_SIZE:
                    continue
                # Colour utilities carry a shade (`text-red-300`); the regex
                # already excludes those, so what is left is a bare word that
                # looks like a size token and matches no rule.
                line = src[:m.start()].count('\n') + 1
                offenders.append(f'{_rel(path)}:{line} {m.group(0)}')
        self.assertEqual(
            offenders, [],
            'these look like font-size tokens but are defined nowhere, so '
            'Tailwind emits no rule and the text silently inherits its parent '
            'size: ' + ', '.join(offenders))


# ── Body text is 12px or more ───────────────────────────────────────────────
# `nano` (9px) and `micro` (10px) are CHROME sizes: a pill badge, a <kbd>, a
# 1-3 character tick, an uppercase letter-spaced tag. A label, a value, a caption
# or a button's text is body text and does not use them. Whether a given piece of
# text is chrome is a property of the rendered element, so the whole page is
# checked in the browser (react-ui/e2e/faceswap-layout.spec.js); what is checked
# HERE is what can be said from the source alone.
MIN_BODY_PX = 12
CHROME_TOKENS = {'nano', 'micro'}


def _theme_sizes():
    with open(CSS, encoding='utf-8') as fh:
        css = fh.read()
    return {name: int(px) for name, px in re.findall(r'--text-([a-z]+)\s*:\s*(\d+)px', css)}


def _block(src, start_pat):
    """From the first match of start_pat to the brace that closes its first `{`."""
    m = re.search(start_pat, src)
    if not m:
        return ''
    i = src.find('{', m.end())
    depth = 0
    for j in range(i, len(src)):
        depth += (src[j] == '{') - (src[j] == '}')
        if depth == 0:
            return src[m.start():j + 1]
    return ''


class BodyTextFloor(unittest.TestCase):
    def test_only_the_chrome_steps_are_under_12px(self):
        small = {n for n, px in _theme_sizes().items() if px < MIN_BODY_PX}
        self.assertEqual(
            small, CHROME_TOKENS,
            'the steps under 12px must be exactly nano and micro (chrome). '
            '`mini` was 11px and is body text (captions on 150 call sites): '
            'raise a body step rather than adding another small one. '
            f'got {sorted(small)}')

    def test_body_primitives_do_not_use_chrome_sizes(self):
        """Field/Slider/Toggle/Button and a tracker slider card are all labels,
        values and button text: none may reach for text-nano / text-micro."""
        def code(path):
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
            return '\n'.join(l.split('//', 1)[0] for l in src.split('\n'))

        ui = code(os.path.join(SRC, 'components', 'ui.jsx'))
        tracker = code(os.path.join(SRC, 'components', 'faceswap', 'SliderTrackerBar.jsx'))
        regions = {
            'ui.jsx Field': _block(ui, r'export const Field\s*='),
            'ui.jsx Slider': _block(ui, r'export const Slider\s*='),
            'ui.jsx Toggle': _block(ui, r'export const Toggle\s*='),
            'ui.jsx Button': _block(ui, r'export const Button\s*='),
            'SliderTrackerBar TrackerSlider': _block(tracker, r'function TrackerSlider'),
        }
        offenders = []
        for name, body in regions.items():
            self.assertTrue(body, f'could not find {name}; the pattern has drifted from the source')
            offenders += [f'{name}: text-{t}' for t in CHROME_TOKENS if re.search(rf'\btext-{t}\b', body)]
        self.assertEqual(offenders, [], 'body-text components use a chrome size: ' + ', '.join(offenders))


if __name__ == '__main__':
    unittest.main()
