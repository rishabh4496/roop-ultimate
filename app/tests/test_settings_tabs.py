"""The Settings tabs, the runtime report and the workspace store agree with the backend.

Three lists in the React UI are copies of something the backend owns, and each can
drift without anything failing at runtime:

  * `runtimeRows.js` RUNTIME_SECTIONS is `roop/runtime_state.SECTION_NAMES`. A section
    the backend gains and the panel lacks is a diagnostics page that quietly omits it.
  * `settingsTabs.js` maps every searchable setting to a tab. The command palette opens
    a setting by key; one that maps to a tab that does not contain it would open the
    wrong tab and then find nothing to flash.
  * `workspaceStore.js` SETTINGS_TABS must be the tabs `settingsTabs.js` labels.
"""

import os
import re
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
COMP = os.path.join(SRC, 'components')


def _read(*parts):
    with open(os.path.join(*parts), encoding='utf-8') as fh:
        return fh.read()


class RuntimeSections(unittest.TestCase):
    def test_the_panel_lists_exactly_the_engine_s_sections(self):
        if APP not in sys.path:
            sys.path.insert(0, APP)
        from roop.runtime_state import SECTION_NAMES
        js = _read(COMP, 'runtimeRows.js')
        block = js.split('export const RUNTIME_SECTIONS = [', 1)[1].split('];', 1)[0]
        self.assertEqual(tuple(re.findall(r"'([A-Z]+)'", block)), tuple(SECTION_NAMES),
                         'runtimeRows.js RUNTIME_SECTIONS differs from runtime_state.SECTION_NAMES '
                         '(names AND order: the panel renders them in the backend\'s order)')
        self.assertEqual(len(SECTION_NAMES), 14)


class SettingsTabs(unittest.TestCase):
    def setUp(self):
        self.tabs_js = _read(COMP, 'settingsTabs.js')
        self.store_js = _read(SRC, 'store', 'workspaceStore.js')

    def test_the_store_and_the_labels_name_the_same_tabs(self):
        store = re.search(r"SETTINGS_TABS = \[([^\]]*)\]", self.store_js).group(1)
        store_tabs = re.findall(r"'([a-z]+)'", store)
        labels = re.search(r"TAB_LABELS = \{(.*?)\n\};", self.tabs_js, re.S).group(1)
        label_tabs = re.findall(r"^\s+([a-z]+):", labels, re.M)
        self.assertEqual(store_tabs, label_tabs)
        # Five settings tabs plus the Keyboard Shortcuts remapper (Stage 26), which is not a backend setting.
        self.assertEqual(len(store_tabs), 6)
        self.assertEqual(store_tabs[-1], 'shortcuts')

    def test_every_override_key_is_a_real_setting(self):
        """KEY_TAB entries that name a key settings.py does not have would never match anything."""
        declared = set(re.findall(r"(?:default_get|_hw_get)\(\s*data\s*,\s*'([a-z0-9_]+)'",
                                  _read(APP, 'settings.py')))
        block = re.search(r"const KEY_TAB = \{(.*?)\n\};", self.tabs_js, re.S).group(1)
        keys = re.findall(r"^\s+([a-z0-9_]+):", block, re.M)
        self.assertGreater(len(keys), 10)
        missing = sorted(set(keys) - declared)
        self.assertEqual(missing, [], f'settingsTabs.js KEY_TAB names settings the backend does not have: {missing}')

    def test_every_tab_has_at_least_one_catalog_setting_or_is_diagnostics(self):
        """A tab with nothing the palette can open would be unreachable by search."""
        catalog = _read(COMP, 'settingsCatalog.js')
        sections = set(re.findall(r"section:\s*'([^']+)'", catalog))
        self.assertTrue(sections, 'settingsCatalog.js sections not parsed')

        def tab(section):  # mirrors SECTION_TAB in settingsTabs.js
            s = section.lower()
            if s.startswith('advanced performance'):
                return 'cpu'
            if s.startswith('performance'):
                return 'providers'
            if s.startswith('server') or s.startswith('output'):
                return 'paths'
            return 'general'

        reached = {tab(s) for s in sections}
        for needed in ('providers', 'cpu', 'paths', 'general'):
            self.assertIn(needed, reached, f'no catalog section lands on the {needed} tab')


class ShellContract(unittest.TestCase):
    def test_the_header_no_longer_has_a_second_navigation(self):
        """The rail is THE navigation. A leftover HeaderNav would be a second `nav` named Main."""
        self.assertFalse(os.path.exists(os.path.join(COMP, 'HeaderNav.jsx')))
        app = _read(SRC, 'App.jsx')
        self.assertIn('WorkstationRail', app)
        self.assertNotIn('HeaderNav', app)

    def test_ctrl_k_is_handled_before_the_text_field_guard(self):
        """The palette must open from inside a field (the Settings search box is where people reach for it)."""
        # Since Stage 26 the key is the `app.palette` action; "may fire from a field" is its `modInText` flag, and the
        # handler checks it before the early return for a focused field.
        app = _read(SRC, 'App.jsx')
        body = app.split('const onKey = (e) => {', 1)[1]
        self.assertLess(body.index("fires('app.palette')"), body.index('if (inField) return;'),
                        'the palette is checked after the "typing in a field" early return, so it cannot open from a field')
        actions = _read(SRC, 'shortcuts', 'actions.js')
        palette = next(line for line in actions.splitlines() if "id: 'app.palette'" in line)
        self.assertIn('modInText: true', palette, 'app.palette must opt in to firing from a text field')
        self.assertIn("'Mod+K'", palette)

    def test_the_telemetry_poll_has_one_home(self):
        """GPU/VRAM/CPU is shown by the header pill only: the HUD banner, Home's GPU tile and
        Face Swap's hardware card were three more views of one poll."""
        self.assertFalse(os.path.exists(os.path.join(COMP, 'faceswap', 'SystemTelemetryHud.jsx')))
        home = _read(COMP, 'Home.jsx')
        self.assertNotIn('useTelemetry', home)
        self.assertNotIn('showHud', _read(SRC, 'App.jsx'))


if __name__ == '__main__':
    unittest.main()
