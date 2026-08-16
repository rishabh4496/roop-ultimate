"""Every preset must set REAL settings to values the backend accepts.

A preset is a block of `key: value` pairs applied straight into the live params.
Nothing validates them at runtime: an invented key is written, ignored by the
backend and echoed back into the settings blob, and a value outside an option
list leaves a `<select>` with no matching entry. Either way the preset looks
like it applied and changed nothing — which is exactly what the first Preset
Studio recipes did (`enhancer_blend`, `face_upscaler`, `mask_blur`,
`'GPEN-BFR-512'`, `no_face_action: 'skip'`).

So the presets are checked against the two places that define what is real:
faceswap/defaults.js for the key set, and api.py's meta lists (plus the UI's own
AI_UPSCALE_MODELS) for the values. Covers both preset sources — the Preset
Studio recipes and SliderTrackerBar's built-in presets.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(os.path.dirname(APP), 'react-ui', 'src', 'components')


def _read(*parts):
    with open(os.path.join(*parts), encoding='utf-8') as fh:
        return fh.read()


def _js_list(src, name):
    """Values of a JS array-of-strings literal `name = [ 'a', 'b' ]`."""
    m = re.search(re.escape(name) + r'\s*[:=]\s*\[(.*?)\]', src, re.S)
    return set(re.findall(r"""['"]([^'"]+)['"]""", m.group(1))) if m else set()


def _py_list(src, key):
    """Values of a Python list literal `"key": [ "a", "b" ]` in api.py's meta."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[(.*?)\]', src, re.S)
    return set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()


def _blocks(src, opener):
    """Every `opener {...}` object literal body, brace-matched.

    `opener` is a REGEX, not a literal, so a preset spelled as a helper call
    (`preset('Name', { … })`) can be matched as readily as a bare `values:`
    property. It has been both.
    """
    out = []
    for m in re.finditer(opener + r'\s*\{', src):
        i, depth = m.end() - 1, 0
        for j in range(i, len(src)):
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
                if depth == 0:
                    out.append(src[i + 1:j])
                    break
    return out


def _pairs(body):
    """`key: value` pairs of a flat object-literal body (skips nested objects)."""
    pairs = {}
    for km, vm in re.findall(r"""([A-Za-z_][A-Za-z0-9_]*)\s*:\s*("""
                             r"""'[^']*'|"[^"]*"|true|false|-?\d+(?:\.\d+)?)""", body):
        v = vm.strip()
        if v in ('true', 'false'):
            pairs[km] = v == 'true'
        elif v[:1] in ('"', "'"):
            pairs[km] = v[1:-1]
        else:
            pairs[km] = float(v) if '.' in v else int(v)
    return pairs


API_SRC = _read(APP, 'api.py')
# key -> value, not just the keys: the values are what Reset defaults writes,
# so they need checking against the same enums the presets are checked against.
# (_js_list would give a bare set of strings — it was that, and unused.)
DEFAULTS = _pairs(_blocks(_read(UI, 'faceswap', 'defaults.js'),
                          'FACESWAP_DEFAULTS =')[0])
VALID_KEYS = set(re.findall(r'^\s{2}([a-z_0-9]+):',
                            _read(UI, 'faceswap', 'defaults.js'), re.M))

# key -> the set of values the backend will accept for it.
ENUMS = {
    'selected_enhancer': _py_list(API_SRC, 'enhancers'),
    'swap_model': _py_list(API_SRC, 'swap_models'),
    'face_detection_mode': _py_list(API_SRC, 'face_detection_modes'),
    'mask_engine': _py_list(API_SRC, 'mask_engines'),
    'sam2_model_size': _py_list(API_SRC, 'sam2_model_sizes'),
    'color_transfer_mode': _py_list(API_SRC, 'color_transfer_modes'),
    'detector_engine': _py_list(API_SRC, 'detector_engines'),
    'subsample_upscale': _py_list(API_SRC, 'upscale'),
    'video_swapping_method': _py_list(API_SRC, 'video_methods'),
    'output_method': _py_list(API_SRC, 'output_methods'),
    'no_face_action': set(re.findall(r'"([^"]+)"', re.search(
        r'no_face_choices = \[(.*?)\]', API_SRC, re.S).group(1))),
    'upscale_model_after': set(re.findall(
        r"value:\s*'([^']+)'", re.search(r'AI_UPSCALE_MODELS = \[(.*?)\];',
                                         _read(UI, 'FaceSwap.jsx'), re.S).group(1))),
}

# Every setting the Face Swap tab can change, straight from its set('key', …)
# calls, intersected with the real settings in settings.py.
_SETTINGS_KEYS = set(re.findall(r"self\.default_get\(data,\s*'([^']+)'",
                                _read(APP, 'settings.py')))
FACESWAP_TAB_KEYS = set(re.findall(
    r"\bset\(\s*'([a-z0-9_]+)'", _read(UI, 'FaceSwap.jsx'))) & _SETTINGS_KEYS


class PresetKeysTest(unittest.TestCase):

    def _check(self, label, params):
        for key, value in params.items():
            with self.subTest(preset=label, key=key):
                self.assertIn(key, VALID_KEYS,
                              f"{label}: '{key}' is not a real setting "
                              f"(not in FACESWAP_DEFAULTS) — applying it does nothing")
                allowed = ENUMS.get(key)
                if allowed and isinstance(value, str):
                    self.assertIn(value, allowed,
                                  f"{label}: {key}={value!r} is not one of {sorted(allowed)}")

    def test_option_lists_were_actually_found(self):
        """Guard the parsing itself — empty sets would make this suite vacuous."""
        self.assertGreater(len(VALID_KEYS), 40)
        for key, values in ENUMS.items():
            self.assertTrue(values, f"could not parse the option list for {key}")

    def test_preset_studio_recipes(self):
        src = _read(UI, 'faceswap', 'PresetStudioModal.jsx')
        recipes = _blocks(src, 'params:')
        self.assertGreaterEqual(len(recipes), 4, 'recipes not parsed')
        for i, body in enumerate(recipes):
            params = _pairs(body)
            self.assertTrue(params, f'recipe {i} parsed to no params')
            self._check(f'PresetStudio recipe {i}', params)

    def test_slider_tracker_builtin_presets(self):
        src = _read(UI, 'faceswap', 'SliderTrackerBar.jsx')
        # Both spellings: the `preset(name, {…})` helper the built-ins use now,
        # and a bare `values: {…}` in case one is ever written by hand again.
        presets = _blocks(src, r"(?:preset\('[^']+',|values:)")
        self.assertGreaterEqual(len(presets), 4, 'built-in presets not parsed')
        for i, body in enumerate(presets):
            self._check(f'SliderTrackerBar preset {i}', _pairs(body))

    def test_the_factory_defaults_are_themselves_valid(self):
        """FACESWAP_DEFAULTS was the one table nothing checked.

        It was parsed here and then never used, so every preset was validated
        against the enums while the values "Reset defaults" actually writes were
        not. An invalid one there is worse than in a preset: it is what the
        reset button restores to.
        """
        self.assertTrue(DEFAULTS, 'FACESWAP_DEFAULTS parsed to nothing')
        for key, value in DEFAULTS.items():
            with self.subTest(key=key):
                self.assertIn(key, _SETTINGS_KEYS,
                              f"'{key}' is not a real setting in settings.py — "
                              f"Reset defaults would write a key nothing reads")
                allowed = ENUMS.get(key)
                if allowed and isinstance(value, str):
                    self.assertIn(value, allowed,
                                  f"default {key}={value!r} is not one of {sorted(allowed)}")

    def test_every_face_swap_control_can_be_reset(self):
        """Reset defaults must cover every control the tab owns.

        A setting the tab can change but that is missing from FACESWAP_DEFAULTS
        survives a reset untouched, so "Reset defaults" leaves the tab in a
        state that is not the defaults — silently, and only for that one
        control. It has happened twice: expression_restore_strength, then
        yaw_align.
        """
        self.assertGreater(len(FACESWAP_TAB_KEYS), 40, 'set() calls not parsed')
        missing = sorted(FACESWAP_TAB_KEYS - VALID_KEYS)
        self.assertEqual(missing, [],
                         f"editable in the Face Swap tab but not in "
                         f"FACESWAP_DEFAULTS, so Reset skips them: {missing}")


if __name__ == '__main__':
    unittest.main()
