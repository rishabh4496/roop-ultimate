"""Slider Tracker bar — that its sliders agree with the panel they mirror.

The tracker is a SECOND view of settings the Face Swap panel already exposes.
Two views of one number is a drift risk with no runtime symptom: a range that
disagreed would let the tracker produce a value the panel's own slider cannot
represent (or clamp one the panel allows), and nothing would report it.

Also guards the grouping the bar renders by, which is derived rather than
hand-listed — the failure there is a slider in a group no section draws, so it
would vanish from the UI while every other check still passed.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(os.path.dirname(APP), 'react-ui', 'src', 'components')


def _read(*parts):
    with open(os.path.join(*parts), encoding='utf-8') as fh:
        return fh.read()


TRACKER_SRC = _read(UI, 'faceswap', 'trackerConfig.js')
FACESWAP_SRC = _read(UI, 'FaceSwap.jsx')
DEFAULTS_SRC = _read(UI, 'faceswap', 'defaults.js')
BAR_SRC = _read(UI, 'faceswap', 'SliderTrackerBar.jsx')


def tracker_sliders():
    """Each `{ key, group, min, max, step, defaultVal, bypassVal }` entry."""
    out = []
    for block in re.findall(r"\{\s*key:\s*'([a-z_0-9]+)',(.*?)\n  \},", TRACKER_SRC, re.S):
        key, body = block
        fields = {'key': key}
        gm = re.search(r'group:\s*([A-Z_]+)', body)
        fields['group'] = gm.group(1) if gm else None
        for name in ('min', 'max', 'step', 'defaultVal', 'bypassVal'):
            m = re.search(name + r':\s*(-?[\d.]+)', body)
            if m:
                fields[name] = float(m.group(1))
        out.append(fields)
    return out


def panel_sliders():
    """`key -> (min, max, step)` for every <Slider> in the settings panel."""
    pat = re.compile(r"min=\{(-?[\d.]+)\}\s*max=\{(-?[\d.]+)\}\s*step=\{(-?[\d.]+)\}"
                     r".*?set\(\s*'([a-z_0-9]+)'", re.S)
    found = {}
    for mn, mx, st, key in pat.findall(FACESWAP_SRC):
        found.setdefault(key, (float(mn), float(mx), float(st)))
    return found


def builtin_presets():
    """`name -> {key: number}` for each `preset('Name', { … })` built-in.

    Only the keys a preset OVERRIDES; the rest come from the spread over
    TRACKER_DEFAULT_VALUES, so an absent key means "ships at the default",
    which is what the assertions below rely on.
    """
    out = {}
    for name, body in re.findall(r"preset\('([^']+)',\s*\{(.*?)\n  \}\)", BAR_SRC, re.S):
        out[name] = {k: float(v) for k, v in
                     re.findall(r'^\s+([a-z_0-9]+):\s*(-?[\d.]+),', body, re.M)}
    return out


SLIDERS = tracker_sliders()
PANEL = panel_sliders()
PRESETS = builtin_presets()
DEFAULT_KEYS = set(re.findall(r'^\s{2}([a-z_0-9]+):', DEFAULTS_SRC, re.M))
SETTINGS_KEYS = set(re.findall(r"self\.default_get\(data,\s*'([^']+)'",
                               _read(APP, 'settings.py')))


class ParsingIsNotVacuous(unittest.TestCase):
    def test_both_sides_parsed(self):
        self.assertGreaterEqual(len(SLIDERS), 14, 'tracker sliders not parsed')
        self.assertGreaterEqual(len(PANEL), 20, 'panel sliders not parsed')

    def test_presets_parsed(self):
        """Named individually: every assertion below indexes PRESETS by name,
        so a rename would otherwise fail as a KeyError with no explanation."""
        self.assertEqual(set(PRESETS), {'Ultra Realism', 'Cinematic',
                                        'Subtle Touchup', 'Strong Likeness',
                                        'Natural Soft'})
        for name, values in PRESETS.items():
            self.assertTrue(values, f'{name} parsed to no values')


class TrackerMatchesThePanel(unittest.TestCase):
    def test_ranges_agree_with_the_settings_panel(self):
        """The drift this file exists for. Both views write the same setting,
        so a value one can reach and the other cannot is a real bug."""
        checked = 0
        for s in SLIDERS:
            if s['key'] not in PANEL:
                continue
            checked += 1
            with self.subTest(key=s['key']):
                self.assertEqual((s['min'], s['max'], s['step']), PANEL[s['key']],
                                 f"{s['key']}: tracker range disagrees with FaceSwap.jsx")
        self.assertGreaterEqual(checked, 10, 'almost nothing was cross-checked')

    def test_every_slider_is_a_real_setting(self):
        for s in SLIDERS:
            with self.subTest(key=s['key']):
                self.assertIn(s['key'], DEFAULT_KEYS,
                              f"{s['key']} is missing from FACESWAP_DEFAULTS, so "
                              f"Reset defaults would skip it")
                self.assertIn(s['key'], SETTINGS_KEYS,
                              f"{s['key']} is not persisted in settings.py")


class ValuesAreInRange(unittest.TestCase):
    def test_default_and_bypass_are_reachable(self):
        """Both are written straight into params — one outside [min, max] would
        put the slider in a state the user cannot get back to by dragging."""
        for s in SLIDERS:
            for field in ('defaultVal', 'bypassVal'):
                with self.subTest(key=s['key'], field=field):
                    self.assertGreaterEqual(s[field], s['min'])
                    self.assertLessEqual(s[field], s['max'])


class GroupingHoldsTogether(unittest.TestCase):
    def test_every_slider_names_a_group_in_the_order_list(self):
        """TRACKER_GROUPS filters by GROUP_ORDER, so a slider whose group is not
        in that list is silently dropped from the rendered bar."""
        order = re.search(r'GROUP_ORDER\s*=\s*\[(.*?)\]', TRACKER_SRC, re.S).group(1)
        known = set(re.findall(r'([A-Z_]+)', order))
        self.assertTrue(known, 'GROUP_ORDER not parsed')
        for s in SLIDERS:
            with self.subTest(key=s['key']):
                self.assertIsNotNone(s['group'], f"{s['key']} has no group")
                self.assertIn(s['group'], known,
                              f"{s['key']} is in {s['group']}, which GROUP_ORDER "
                              f"never lists — it would not render")

    def test_the_merger_group_holds_the_merger_settings(self):
        merger = {s['key'] for s in SLIDERS if s['group'] == 'GROUP_MERGER'}
        self.assertEqual(
            merger,
            {'merger_hist_match', 'merger_sharpen', 'merger_motion_blur',
             'merger_grain_match', 'merger_degrade', 'merger_clarity',
             'output_face_scale'})

    def test_the_bar_renders_by_group(self):
        """A flat TRACKER_SLIDERS.map in the render would ignore the sections
        entirely while every test above still passed."""
        self.assertIn('TRACKER_GROUPS.map', BAR_SRC)


class PresetsCoverEverySlider(unittest.TestCase):
    def test_builtins_spread_the_defaults(self):
        """A preset that omits a key leaves that slider untouched on apply while
        valuesMatch still compares it, so the pill you just clicked reads
        'Custom'. Spreading the defaults makes that unrepresentable."""
        recipes = re.findall(r"preset\('([^']+)',\s*\{", BAR_SRC)
        self.assertGreaterEqual(len(recipes), 4, 'built-in presets not parsed')
        self.assertIn('...TRACKER_DEFAULT_VALUES', BAR_SRC)

    def test_preset_keys_are_all_real_sliders(self):
        keys = {s['key'] for s in SLIDERS}
        for values in PRESETS.values():
            for key in values:
                with self.subTest(key=key):
                    self.assertIn(key, keys,
                                  f"preset sets '{key}', which is not a tracker slider")


class PresetsFollowTheirNames(unittest.TestCase):
    """The merger half of each preset has to follow from the preset's name.

    These values are easy to fill in arbitrarily — five presets, six sliders,
    thirty numbers that all "look reasonable". Each test below is one of the
    decisions that made a number what it is, so a later edit that reverts to
    plausible-looking noise fails rather than passes quietly.
    """

    def test_no_preset_touches_face_size(self):
        """It depends on how THIS source's face compares to THIS target's,
        which no preset can know. A value here would be a guess wearing a
        preset's name."""
        for name, values in PRESETS.items():
            with self.subTest(preset=name):
                self.assertNotIn('output_face_scale', values)

    def test_every_preset_asks_for_grain(self):
        """The face coming back cleaner than the plate is the one tell that
        applies to all of them, so it is the one slider none can skip."""
        for name, values in PRESETS.items():
            with self.subTest(preset=name):
                self.assertGreater(values.get('merger_grain_match', 0), 0)

    def test_grain_tracks_how_hard_the_preset_pushes(self):
        """Ultra Realism exists to beat exactly this tell; Subtle Touchup
        barely intervenes at all. If those two ever swap order the scale has
        stopped meaning anything."""
        self.assertGreater(PRESETS['Ultra Realism']['merger_grain_match'],
                           PRESETS['Subtle Touchup']['merger_grain_match'])

    def test_strong_likeness_holds_back_histogram_match(self):
        """Histogram match pulls tonality toward the TARGET, so on the preset
        whose whole job is source identity it must be the lowest of the set —
        this is a trade-off, not a quality dial."""
        others = [v['merger_hist_match'] for n, v in PRESETS.items()
                  if n != 'Strong Likeness' and 'merger_hist_match' in v]
        self.assertTrue(others, 'no other preset sets histogram match')
        self.assertLess(PRESETS['Strong Likeness']['merger_hist_match'], min(others))

    def test_strong_likeness_does_not_blur_away_what_it_paid_for(self):
        """Three swap passes then a motion blur or a degrade would be spending
        time to discard the result."""
        v = PRESETS['Strong Likeness']
        self.assertEqual(v.get('merger_motion_blur', 0), 0)
        self.assertEqual(v.get('merger_degrade', 0), 0)

    def test_cinematic_blurs_the_most(self):
        """A 180-degree shutter is the defining artefact of the look."""
        others = [v.get('merger_motion_blur', 0) for n, v in PRESETS.items()
                  if n != 'Cinematic']
        self.assertGreater(PRESETS['Cinematic']['merger_motion_blur'], max(others))

    def test_sharpen_sign_matches_the_texture_in_the_name(self):
        """Film and 'soft' are not video-crisp; likeness is."""
        self.assertLess(PRESETS['Cinematic']['merger_sharpen'], 0)
        self.assertLess(PRESETS['Natural Soft']['merger_sharpen'], 0)
        self.assertGreater(PRESETS['Strong Likeness']['merger_sharpen'], 0)

    def test_natural_soft_is_the_softest(self):
        soft = PRESETS['Natural Soft']['merger_sharpen']
        for name, values in PRESETS.items():
            if name != 'Natural Soft':
                with self.subTest(preset=name):
                    self.assertGreater(values.get('merger_sharpen', 0), soft)


if __name__ == '__main__':
    unittest.main()
