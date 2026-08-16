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
    'red', 'green', 'blue', 'amber', 'emerald', 'rose', 'sky', 'slate',
    'zinc', 'gray', 'grey', 'neutral', 'stone', 'orange', 'yellow', 'lime',
    'teal', 'cyan', 'indigo', 'violet', 'purple', 'fuchsia', 'pink',
    'top', 'bottom', 'middle', 'super', 'sub',
}


def _jsx_files():
    for root, _dirs, names in os.walk(SRC):
        for name in sorted(names):
            if name.endswith(('.jsx', '.js')):
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


if __name__ == '__main__':
    unittest.main()
