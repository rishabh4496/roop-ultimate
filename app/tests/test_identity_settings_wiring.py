"""The Identity & tracking settings must reach the flags they claim to drive.

A setting can look completely wired — a control in the panel, an entry in the
catalogue, a key in config.yaml — and still do nothing, because the half that
matters is a string in run.py matching a string in a roop module. Nothing else
in the suite compares those two: test_ui_settings_catalog keeps the panel and
the catalogue in step, and the settings round-trip tests only prove the value
persists. This closes the last link.

It also pins the value VOCABULARIES. `keep_awake` falls back to 'high' for any
priority name it does not recognise, so an option offered in the UI but absent
from _PRIORITY_CLASSES would present as a working choice and silently do
nothing — which is exactly what the first draft of this feature shipped with
('background', a name that module has never accepted).
"""

import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)


def _read(rel):
    with open(os.path.join(APP, rel), encoding='utf-8') as fh:
        return fh.read()


class TestIdentitySettingsReachTheirFlags(unittest.TestCase):

    # setting key -> the environment variable run.py must set from it
    TRISTATE = {
        'face_demarcate': 'ROOP_FACE_DEMARCATE',
        'track_stitch': 'ROOP_TRACK_STITCH',
        'verify_swap': 'ROOP_VERIFY_SWAP',
        'upright_remeasure': 'ROOP_UPRIGHT_REMEASURE',
    }

    def test_settings_exist_with_auto_default(self):
        """'auto' must mean "leave the environment alone", so exposing these
        changed no shipped behaviour for anyone who never touches them."""
        src = _read('settings.py')
        for key in list(self.TRISTATE) + ['process_priority']:
            self.assertRegex(
                src, rf"default_get\(data, '{key}', 'auto'\)",
                f"{key} must default to 'auto' (leave the env alone)")
        self.assertRegex(src, r"default_get\(data, 'recognizer', 'default'\)")

    def test_run_py_maps_each_setting_to_its_flag(self):
        src = _read('run.py')
        for key, var in self.TRISTATE.items():
            self.assertIn(f"'{var}', '{key}'", src,
                          f"run.py must map {key} -> {var}")
        self.assertIn('ROOP_ADAFACE', src)
        self.assertIn('ROOP_PRIORITY', src)

    def test_each_flag_is_actually_read_by_a_roop_module(self):
        """The names must match something that reads them, or the setting is a
        no-op with a convincing label."""
        blob = ''
        roop_dir = os.path.join(APP, 'roop')
        for root, _dirs, files in os.walk(roop_dir):
            for f in files:
                if f.endswith('.py'):
                    with open(os.path.join(root, f), encoding='utf-8',
                              errors='ignore') as fh:
                        blob += fh.read()
        for var in list(self.TRISTATE.values()) + ['ROOP_ADAFACE', 'ROOP_PRIORITY']:
            self.assertIn(f"environ.get('{var}'", blob,
                          f"{var} is set by run.py but nothing reads it")

    def test_priority_options_are_only_names_keep_awake_accepts(self):
        """keep_awake falls back to 'high' for an unknown name, so an option
        outside its table would look like a choice and do nothing."""
        accepted = set(re.findall(r"'([a-z_]+)': 0x[0-9A-Fa-f]+",
                                  _read('roop/keep_awake.py')))
        self.assertTrue(accepted, 'could not read _PRIORITY_CLASSES')
        offered = re.search(r'"priorities":\s*\[([^\]]+)\]', _read('api.py'))
        self.assertIsNotNone(offered, '/api/meta must publish the priority list')
        names = set(re.findall(r'"([a-z_]+)"', offered.group(1))) - {'auto'}
        self.assertTrue(
            names <= accepted,
            f"offered priorities {sorted(names - accepted)} are not in "
            f"keep_awake._PRIORITY_CLASSES {sorted(accepted)}")

        # run.py must not forward a name outside that table either.
        forwarded = re.search(r"_pri in \(([^)]+)\)", _read('run.py'))
        self.assertIsNotNone(forwarded)
        self.assertTrue(set(re.findall(r"'([a-z_]+)'", forwarded.group(1))) <= accepted)

    def test_recognizer_options_match_what_run_py_understands(self):
        offered = re.search(r'"recognizers":\s*\[([^\]]+)\]', _read('api.py'))
        self.assertIsNotNone(offered)
        names = set(re.findall(r'"([a-z]+)"', offered.group(1)))
        self.assertEqual(names, {'default', 'adaface'})


if __name__ == '__main__':
    unittest.main()
