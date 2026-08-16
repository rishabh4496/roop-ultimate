"""The palette's settings catalogue and the Settings panel must list the same keys.

App.jsx offers every setting by name in the command palette, and it cannot read
them out of Settings.jsx: that panel is a lazy chunk, and its controls are JSX
rather than a list. So the catalogue is a separate file — which makes it the
same shape of hazard as every other hand-mirrored list in this UI.

Both directions fail silently:

  * A key in the catalogue that the panel does not bind gives a palette entry
    that opens Settings, finds no `[data-setting]` element, and quietly does
    nothing. The command is there, it highlights, it just goes nowhere.
  * A key the panel binds but the catalogue omits is simply unfindable from the
    palette. Nothing is broken; the setting is just invisible to the one
    surface built for looking settings up.

Neither shows up in a build, in oxlint, or in a diff.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
SETTINGS_JSX = os.path.join(SRC, 'components', 'Settings.jsx')
CATALOG_JS = os.path.join(SRC, 'components', 'settingsCatalog.js')

BOUND = re.compile(r"\bbind(?:Toggle)?\(\s*'([a-z0-9_]+)'", re.I)
CATALOGUED = re.compile(r"key:\s*'([a-z0-9_]+)'", re.I)


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


class SettingsCatalog(unittest.TestCase):
    def setUp(self):
        self.bound = set(BOUND.findall(_read(SETTINGS_JSX)))
        self.catalogued = set(CATALOGUED.findall(_read(CATALOG_JS)))
        self.assertTrue(self.bound, 'parsed no bind() keys out of Settings.jsx')
        self.assertTrue(self.catalogued, 'parsed no keys out of settingsCatalog.js')

    def test_catalogued_settings_all_exist_in_the_panel(self):
        missing = sorted(self.catalogued - self.bound)
        self.assertEqual(
            missing, [],
            'the command palette offers these but the Settings panel binds no '
            'such control, so choosing one opens Settings and does nothing: '
            + ', '.join(missing))

    def test_panel_settings_are_all_findable_from_the_palette(self):
        missing = sorted(self.bound - self.catalogued)
        self.assertEqual(
            missing, [],
            'these are in the Settings panel but missing from '
            'settingsCatalog.js, so they cannot be found from the command '
            'palette: ' + ', '.join(missing))

    def test_the_panel_stamps_the_key_onto_the_dom(self):
        """The jump works by querying `[data-setting]`, so it must be rendered."""
        ui = _read(os.path.join(SRC, 'components', 'ui.jsx'))
        self.assertIn(
            'data-setting={settingKey}', ui,
            'Field/Toggle must stamp settingKey onto the DOM as data-setting, '
            'or the palette has nothing to scroll to and every settings jump '
            'silently no-ops')


if __name__ == '__main__':
    unittest.main()
