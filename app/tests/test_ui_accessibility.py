"""Two accessibility rules that fail silently, so they are checked mechanically.

Neither shows up in a diff, in oxlint, or in the build. Both were live:

  * `Toggle` rendered its real <input type="checkbox"> with Tailwind's `hidden`,
    which is display:none — that takes an element out of the tab order
    ENTIRELY. The visible switch beside it is a <div>, so there was nothing
    focusable at all: every toggle in the app (most of Settings, most of the
    Face Swap panel) could not be reached or operated from a keyboard. The fix
    is `sr-only`, which hides it visually but keeps it focusable, so this
    checks the class rather than trusting the memory of why.

  * An icon-only button whose whole content is an emoji or an SVG has no
    accessible name. `title` is not one: it is a tooltip, exposed
    inconsistently, and never on touch. Such a button is announced as
    "button" — the row of them in the preview HUD was unusable without sight.

Both rules are about controls that LOOK fine and simply cannot be operated, so
the cost of a regression is total for the people affected and invisible to
everyone else.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')

BUTTON_OPEN = re.compile(r'<button\b')
INPUT_OPEN = re.compile(r'<input\b')
# A form control hidden with display:none is unreachable; `sr-only` is the
# visually-hidden-but-focusable idiom and is what these must use instead.
HIDDEN_CLASS = re.compile(r'className="[^"]*\bhidden\b[^"]*"')


def _jsx_files():
    for root, _dirs, names in os.walk(SRC):
        for name in sorted(names):
            if name.endswith('.jsx'):
                yield os.path.join(root, name)


def _rel(path):
    return os.path.relpath(path, SRC).replace('\\', '/')


def _tags(src, opener):
    """Yield (line, attrs, end_index) for each JSX tag `opener` matches.

    The attribute half of a JSX tag cannot be found by scanning to the first
    `>`. This was a regex — `<button\\b([^>]*?)>(.*?)</button>` — and it quietly
    got the split wrong on the most common shape in this codebase: an inline
    arrow handler. In `onClick={(e) => setX(1)}` the FIRST `>` belongs to the
    arrow, not to the tag, so the attribute half was cut at `onClick={(e) =`
    and everything after it, `aria-label` included, was treated as the button's
    BODY. The name check then found the letters in `aria-label="Close"` sitting
    in the body text and concluded the button had a visible label, so 100+
    controls were passing for entirely the wrong reason.

    Scanning with brace/quote depth finds the real end of the tag.

    Backticks count as a quote context, and that is not a detail: a template
    literal like `Copy part ${tab}'s log` contains an APOSTROPHE. Treating `'`
    as a quote opener there desynchronises everything after it, and the tag is
    dropped from the scan entirely — silently exempting it from the check.
    Opening a backtick context and running to its partner skips the apostrophe,
    and any `>` inside the string with it, which is also correct.

    Shared by both checks below. It used to be inlined in the button scan while
    the hidden-input scan kept a hand-written `(?:[^>]|\\n)*?` regex — which is
    the naive form, with the identical arrow-function blind spot. It could not
    see FileDrop's picker (`onChange={(e) => …}` sits before its className), so
    the app's two primary upload zones were exempt from the one check that
    would have caught them being keyboard-unreachable.
    """
    for m in opener.finditer(src):
        i, depth, quote = m.end(), 0, None
        while i < len(src):
            ch = src[i]
            if quote:
                if ch == quote:
                    quote = None
            elif ch in '"\'`':
                quote = ch
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == '>' and depth == 0:
                break
            i += 1
        else:
            continue
        yield src[:m.start()].count('\n') + 1, src[m.end():i], i


def _buttons(src):
    """Yield (line, attrs, body) for each <button> that has a body."""
    for line, attrs, i in _tags(src, BUTTON_OPEN):
        if src[i - 1] == '/':          # self-closing, so it has no body
            continue
        end = src.find('</button>', i)
        if end == -1:
            continue
        yield line, attrs, src[i + 1:end]


# Text a `{…}` expression can be seen to render: any string literal inside it,
# plus the bare `{value}` / `{obj.prop}` case where the whole expression is one
# identifier and therefore renders whatever that holds.
_LITERAL = re.compile(r"'([^']*)'|\"([^\"]*)\"|`([^`\\{]*)`")
_BARE_IDENT = re.compile(r'^[\w.?\[\]]+$')


def _expr_text(m):
    inner = m.group(0)[1:-1].strip()
    lits = [g for lit in _LITERAL.finditer(inner) for g in lit.groups() if g]
    if lits:
        return ' ' + ' '.join(lits) + ' '
    # `{title}`, `{c.label}` — a value standing alone IS the button's text.
    return ' x ' if _BARE_IDENT.match(inner) else ' '


def _visible_text(body):
    """What a sighted user reads on the button, minus markup.

    Expressions used to be deleted wholesale, which made every button whose
    label is computed — `{expanded ? 'Collapse' : 'Expand'}`, or just `{title}`
    — look like it had no text at all and get reported as a nameless icon
    button. Reading the literals back out is what separates those from the
    controls that really are an icon and nothing else.
    """
    text = re.sub(r'<svg(?:.|\n)*?</svg>', '', body, flags=re.S)
    text = re.sub(r'\{[^{}]*\}', _expr_text, text)
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()


# An `<Icon.foo />` from icons.jsx renders an SVG, so a button containing only
# one is exactly as nameless as a button containing a raw <svg> or an emoji.
# It has to be recognised explicitly: the tag carries no ASCII letters once
# markup is stripped, so without this a button whose whole content is an icon
# component would be read as an empty layout element and skipped — which is
# precisely how the emoji-to-icon migration could have quietly removed the
# accessible name from every icon-only control in the app.
ICON_COMPONENT = re.compile(r'<Icon\.\w+')


class AccessibleNames(unittest.TestCase):
    def test_icon_only_buttons_have_an_accessible_name(self):
        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            for line, attrs, body in _buttons(src):
                if 'aria-label' in attrs:
                    continue
                text = _visible_text(body)
                # Any ASCII letter is a readable name already.
                if re.search(r'[A-Za-z]', text):
                    continue
                # A button with no content at all is a layout element, not a
                # control a user is meant to find; the ones that matter here are
                # the emoji/SVG/<Icon.*> icons.
                if not text and '<svg' not in body and not ICON_COMPONENT.search(body):
                    continue
                offenders.append(f'{_rel(path)}:{line}')
        self.assertEqual(offenders, [],
                         'icon-only buttons with no aria-label (a `title` is a '
                         'tooltip, not an accessible name): ' + ', '.join(offenders))


class FocusableControls(unittest.TestCase):
    def test_no_form_control_is_hidden_with_display_none(self):
        """…unless a real, focusable control drives it.

        The exempt pattern is a `ref`'d file input clicked by a <button> beside
        it: there the button IS the control, it is in the tab order, and hiding
        the input is correct. What is not correct — and is what six file pickers
        here did — is hiding the input inside a bare <label>: a <label> is not a
        tab stop, so nothing focusable was left and the picker could not be
        opened from a keyboard at all.

        FileDrop was the last one, and it survived because the scan could not
        see it, not because it complied: its `onChange={(e) => …}` arrow ended
        the old regex's attribute match before `className`. Since FileDrop is
        both upload zones, and source faces have no other entry point, adding a
        face was impossible without a mouse.
        """
        offenders = []
        for path in _jsx_files():
            with open(path, encoding='utf-8') as fh:
                src = fh.read()
            for line, attrs, _i in _tags(src, INPUT_OPEN):
                if not HIDDEN_CLASS.search(attrs):
                    continue
                if 'ref=' in attrs:
                    continue
                offenders.append(f'{_rel(path)}:{line}')
        self.assertEqual(offenders, [],
                         'these inputs use `hidden` (display:none), which removes '
                         'them from the tab order, and no focusable control drives '
                         'them — use `sr-only` so they stay focusable: '
                         + ', '.join(offenders))

    def test_the_toggle_switch_is_operable_from_the_keyboard(self):
        """Belt and braces on the specific control that was broken: the checkbox
        must be sr-only AND a `peer`, so the focus ring lands on the visible
        switch rather than on an element nobody can see."""
        with open(os.path.join(SRC, 'components', 'ui.jsx'), encoding='utf-8') as fh:
            src = fh.read()
        toggle = src.split('export const Toggle', 1)[1].split('export const', 1)[0]
        self.assertIn('sr-only peer', toggle)
        self.assertIn('peer-focus-visible', toggle,
                      'the switch must show a focus ring when the input behind it '
                      'is focused, or keyboard users cannot see where they are')


class DisclosureState(unittest.TestCase):
    def test_collapsible_sections_report_whether_they_are_open(self):
        with open(os.path.join(SRC, 'components', 'ui.jsx'), encoding='utf-8') as fh:
            src = fh.read()
        section = src.split('export const Section', 1)[1].split('export const', 1)[0]
        self.assertIn('aria-expanded', section)


if __name__ == '__main__':
    unittest.main()
