"""Light themes must repoint `--color-white`, and none may be able to miss it.

Tailwind v4 compiles the whole `*-white` utility family — text, background,
border, ring, at every alpha, under every variant — into a single expression:
`color-mix(in oklab, var(--color-white) N%, transparent)`. On a dark theme that
variable is `#fff` and everything reads correctly. On a LIGHT theme, anything
still resolving to white is invisible.

The bug this guards was live in both directions:

  * The old fix enumerated `.text-white\\/30 … \\/95` and set `opacity`. It
    therefore missed every alpha nobody thought to list (/15 /20 /25 /35 /45
    /55 /85 — 88 uses) and every variant-prefixed one, so `hover:text-white`
    (55 uses) turned invisible the moment you hovered it.
  * The fix that replaced it enumerated the eight light theme CLASS names
    instead. A theme added to `themes.js` with `mode: 'light'` but not to that
    list got no correction at all and was white-on-white everywhere — and a
    user-authored theme could never be in the list, because it has no class.

Both were the same shape of mistake: a hand-maintained list, far from the thing
it mirrors, whose failure is silent. The CSS stays valid and the app still
renders; the text is simply not there.

So the correction is now keyed on `data-theme-mode`, which `applyThemeToDom`
derives from the resolved theme's own `mode` field. There is no list to keep in
step, and these tests hold that mechanism in place rather than re-checking a
list that no longer exists.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
CSS = os.path.join(SRC, 'index.css')
THEMES = os.path.join(SRC, 'themes.js')


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _light_theme_block(css):
    """The selector list + body of the light-theme correction block."""
    start = css.index('Light Theme Readability')
    # First `{` after the selector list, then match to its balanced close.
    brace = css.index('{', start)
    depth, i = 0, brace
    while i < len(css):
        if css[i] == '{':
            depth += 1
        elif css[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return css[start:brace], css[brace:i + 1]


class LightThemeCorrection(unittest.TestCase):
    def test_correction_is_keyed_on_the_mode_attribute(self):
        selectors, _body = _light_theme_block(_read(CSS))
        self.assertIn(
            '[data-theme-mode="light"]', selectors,
            'the light-theme block must key off data-theme-mode, which is '
            'derived from each theme\'s own `mode`. Anything else has to be '
            'maintained by hand and will eventually miss a theme.')

    def test_no_per_theme_class_list_returns(self):
        """A list of `.theme-*-light` classes is the bug, not the fix."""
        selectors, _body = _light_theme_block(_read(CSS))
        enumerated = re.findall(r'\.theme-[\w-]*light\b', selectors)
        self.assertEqual(
            enumerated, [],
            'listing light themes by class name cannot cover a user-authored '
            'theme (it has no class) and silently misses any preset nobody '
            'remembered to add — key off data-theme-mode instead: '
            + ', '.join(enumerated))

    def test_the_mode_attribute_is_actually_set_from_the_theme(self):
        """The CSS above is inert unless something writes the attribute."""
        themes = _read(THEMES)
        self.assertRegex(
            themes,
            r"setAttribute\(\s*'data-theme-mode'\s*,[^)]*mode",
            "applyThemeToDom in themes.js must set data-theme-mode from the "
            "theme's `mode` field — without it every light theme renders "
            "white-on-white and the block in index.css never matches")

    def test_the_correction_repoints_the_variable(self):
        _selectors, body = _light_theme_block(_read(CSS))
        self.assertRegex(
            body, r'--color-white:\s*var\(--text-main\)',
            'the light-theme block must repoint --color-white at the theme ink '
            'colour; that one variable is what every *-white utility reads')

    def test_dark_surfaces_keep_white_ink(self):
        """Scrims and the HUD stay dark in light themes, so they opt back in."""
        _selectors, body = _light_theme_block(_read(CSS))
        self.assertIn('[class*="bg-black/"]', body)
        self.assertIn('.hud-glass', body)
        self.assertRegex(body, r'--color-white:\s*#fff')

    def test_no_per_alpha_text_enumeration_returns(self):
        """The enumeration is the bug, not the fix — it cannot cover variants."""
        _selectors, body = _light_theme_block(_read(CSS))
        enumerated = re.findall(r'\.text-white\\/\d+', body)
        self.assertEqual(
            enumerated, [],
            'listing individual text-white alphas cannot cover hover:, '
            'group-hover: or placeholder: variants, and silently misses any '
            'alpha not written down — repoint --color-white instead: '
            + ', '.join(enumerated))


class CustomThemeDerivation(unittest.TestCase):
    """A custom theme must define everything a preset block defines.

    A missing variable does not error — it falls through to whatever the
    :root default left behind, so a custom theme would silently wear one of
    the default theme's colours. The light-only `--input-bg-focus` is the
    subtle one: without it, focusing an input on a light custom theme turns
    its background dark while the ink stays dark (see `.glass-input:focus`).
    """

    def test_derivation_covers_the_root_variable_set(self):
        vars_js = _read(os.path.join(SRC, 'themeVars.js'))
        css = _read(CSS)

        # Everything :root declares is what a theme is made of.
        root = css[css.index(':root {'):]
        root = root[:root.index('\n}')]
        declared = set(re.findall(r'(--[\w-]+)\s*:', root))

        derived = set(re.findall(r"'(--[\w-]+)':", vars_js))
        missing = sorted(declared - derived)
        self.assertEqual(
            missing, [],
            'deriveThemeVars must emit every variable :root defines, or a '
            'custom theme inherits the default theme for these: '
            + ', '.join(missing))

    def test_light_custom_themes_get_the_input_focus_fix(self):
        vars_js = _read(os.path.join(SRC, 'themeVars.js'))
        self.assertIn(
            '--input-bg-focus', vars_js,
            'light themes need --input-bg-focus or focusing an input paints it '
            'dark under dark ink — the preset light blocks set it, so the '
            'derivation must too')

    def test_accent_ink_is_computed_not_assumed_white(self):
        """Ink on an accent fill must be derived from the accent's lightness.

        The standing assumption was "the accent fill is dark enough to carry
        white ink in every theme". Measured against each theme's own accent,
        white reaches 4.5:1 on NONE of them — lime 1.19, gold 1.40, mauve 2.03,
        teal 2.22, emerald 2.54, and the DEFAULT crimson 3.83. So this was never
        a light-accent edge case, and a hardcoded `color: #fff` is the bug.

        `oklch(from var(--accent) …)` reads the accent's own lightness and picks
        black or white, which is correct for all 38 presets and any user theme
        without a per-theme value. A plain `#fff` immediately before it is the
        intended fallback for an engine without relative colour syntax, so this
        checks that a computed declaration FOLLOWS the literal rather than
        banning the literal outright.
        """
        css = _read(CSS)

        # Scoped to the .fill-accent rule BODY, not the stylesheet. Matching
        # anywhere made this vacuous: the light block's own accent rule carries
        # the same pattern, so deleting the override from .fill-accent still
        # passed. Caught by deliberately perturbing it.
        def rule_body(selector):
            i = css.index(selector)
            body = css[i + len(selector):]
            return body[:body.index('}')]

        for selector in ('.fill-accent {', '.fill-accent:hover {'):
            body = rule_body(selector)
            self.assertIn(
                'oklch(from var(--accent', body,
                f'{selector} sets ink without computing it from the accent — '
                f'white is below 4.5:1 on EVERY theme accent, the default '
                f'crimson included (3.83)')

        # And the accent-fill rule in the light block must not be left asserting
        # plain white as its final word.
        _sel, body = _light_theme_block(css)
        accent_rule = re.search(r'\.bg-\\\[var\\\(--accent\\\)\\\][^{]*\{([^}]*)\}', body)
        self.assertIsNotNone(
            accent_rule, 'the light block no longer scopes the accent fill')
        self.assertIn(
            'oklch(from var(--accent)', accent_rule.group(1),
            'the light-mode accent fill still forces white ink unconditionally')

    def test_status_colours_agree_between_the_css_and_the_derivation(self):
        """The status hues are written twice and cannot be written once.

        `:root` in index.css needs literal values (nothing evaluates JS there),
        and `deriveThemeVars` needs them to emit inline for custom themes. So the
        two lists are duplicated by necessity — and a drift between them is
        invisible: preset themes would read one set and custom themes the other,
        which shows up as "the warning colour is slightly different on my theme"
        and nothing else.

        Both modes are checked, because the light set is the one that matters
        (the dark values as ink on a white card are ~1.9:1).
        """
        css = _read(CSS)
        vars_js = _read(os.path.join(SRC, 'themeVars.js'))
        names = ('--ok', '--warn', '--danger', '--info')

        def css_block(selector):
            i = css.index(selector)
            body = css[i:]
            return body[:body.index('\n}')]

        def js_block(mode):
            m = re.search(mode + r":\s*\{([^}]*)\}", vars_js)
            self.assertIsNotNone(m, f'STATUS_COLORS.{mode} not found in themeVars.js')
            return m.group(1)

        for mode, selector in (('dark', ':root {'), ('light', '[data-theme-mode="light"] {')):
            block = css_block(selector)
            js = js_block(mode)
            for name in names:
                cm = re.search(re.escape(name) + r':\s*([^;]+);', block)
                jm = re.search(re.escape(name) + r"':\s*'([^']+)'", js)
                self.assertIsNotNone(cm, f'{name} missing from {selector} in index.css')
                self.assertIsNotNone(jm, f'{name} missing from STATUS_COLORS.{mode}')
                self.assertEqual(
                    cm.group(1).strip().lower(), jm.group(1).strip().lower(),
                    f'{name} disagrees between index.css ({selector}) and '
                    f'STATUS_COLORS.{mode} in themeVars.js — preset themes and '
                    f'custom themes would render different status colours')


if __name__ == '__main__':
    unittest.main()
