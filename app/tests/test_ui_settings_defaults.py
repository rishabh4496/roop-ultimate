"""Every setting the Settings panel binds must be a real key in settings.py.

The panel's controls are wired through `bind('key')` / `bindToggle('key')`,
which hands a control four things at once: its value, its change handler, its
"changed from default" marker and its reset. That is a nice consolidation, and
it has one sharp edge — the key is now a string in exactly one place, and a
typo in it fails silently in a particular way:

  * `p['typo']` is undefined, so the control falls back to its literal default
    and looks completely normal.
  * `'typo' in defaults` is false, so `isModified` returns false. The control
    can never be marked as changed and its reset never appears.
  * `set('typo', v)` writes a key the backend drops on the floor, because
    save_settings only assigns keys that already exist on CFG. The control
    appears to work, and the value is gone on the next reload.

Nothing errors at any point. So this test pins the string set against the
defaults the backend actually declares.

It also checks the reverse for the theme keys specifically: those are written
through `setMany` rather than `bind`, and they must be persisted by save(), or
a custom theme survives until the next restart and then vanishes.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
SETTINGS_JSX = os.path.join(SRC, 'components', 'Settings.jsx')
SETTINGS_PY = os.path.join(APP, 'settings.py')

BOUND = re.compile(r"\bbind(?:Toggle)?\(\s*'([a-z0-9_]+)'", re.I)
# Deliberately matches the default_get CALL rather than an assignment shape.
# The question is only "does settings.py read this key out of the config", and
# the assignment around it varies: some are wrapped in float(), and max_threads
# lands in a local first so it can be reconciled with the VRAM auto-tuner.
# _hw_get is default_get for hardware-derived values (pool sizes, thread count):
# same load, but it falls back to 'auto' when config.yaml was stamped on another
# GPU. Both count as "settings.py reads this key".
DECLARED = re.compile(r"(?:default_get|_hw_get)\(\s*data\s*,\s*'([a-z0-9_]+)'", re.I)
SAVED = re.compile(r"'([a-z0-9_]+)'\s*:\s*self\.", re.I)


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


class SettingsBindings(unittest.TestCase):
    def test_every_bound_key_exists_in_settings(self):
        bound = set(BOUND.findall(_read(SETTINGS_JSX)))
        self.assertTrue(bound, 'parsed no bind() keys out of Settings.jsx — '
                               'the binding helper must have been renamed')

        declared = set(DECLARED.findall(_read(SETTINGS_PY)))
        self.assertTrue(declared, 'parsed no defaults out of settings.py')

        missing = sorted(bound - declared)
        self.assertEqual(
            missing, [],
            'these are bound by the Settings panel but are not declared in '
            'settings.py, so they can never be marked as changed, never be '
            'reset, and are silently dropped when saved: ' + ', '.join(missing))

    def test_theme_keys_survive_a_restart(self):
        """load() defaults and save() persistence must both list the theme keys.

        A key read in load() but missing from save()'s dict round-trips fine in
        memory and is lost the moment the config is written — so a theme the
        user built would come back as "Default" after a restart.
        """
        py = _read(SETTINGS_PY)
        declared = set(DECLARED.findall(py))
        saved = set(SAVED.findall(py))

        for key in ('custom_themes', 'theme_follow_system', 'theme_dark', 'theme_light'):
            self.assertIn(key, declared,
                          f'{key} has no default in settings.py load()')
            self.assertIn(key, saved,
                          f'{key} is never written by settings.py save(), so it '
                          f'is lost on restart')


if __name__ == '__main__':
    unittest.main()
