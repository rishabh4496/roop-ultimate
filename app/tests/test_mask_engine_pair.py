"""Two occlusion engines instead of one.

When something comes in front of a face and the swap gets painted over it
anyway, the usual cause is not that masking is off — it is that the single
engine selected was not trained on that particular object. The engines are not
interchangeable: XSeg and XSeg-3 come from DeepFaceLab's hand/object data, the
Face Occluder from a different set, the Face Parser segments facial regions
rather than occluders at all, and the SAM variants segment anything.

They compose in the right direction for this. Each mask processor blends the
swapped crop back toward the untouched one wherever it says "not face", so two
of them restore the UNION of what either recognised — a second engine can only
ever hand back more of the original footage, never eat into the swap. That is
what makes this safe to offer as an addition rather than a replacement.

What is guarded here is the wiring, because the failure mode is silent: a second
engine that is selected, sent and saved, and never reaches the processor list,
produces exactly the output the user was already complaining about.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PluginListTakesEitherForm(unittest.TestCase):
    def setUp(self):
        from roop.core import get_processing_plugins
        self.build = get_processing_plugins

    def test_one_engine_as_a_string_is_unchanged(self):
        """Every existing caller passes a bare string — the Gradio tab,
        virtualcam, the per-frame mask path. None of them may be affected."""
        self.assertEqual(list(self.build('mask_xseg')),
                         ['faceswap', 'mask_xseg'])

    def test_no_engine_is_still_no_engine(self):
        self.assertEqual(list(self.build(None)), ['faceswap'])
        self.assertEqual(list(self.build([])), ['faceswap'])

    def test_two_engines_both_reach_the_processor_list(self):
        got = list(self.build(['mask_xseg', 'mask_occluder']))
        self.assertEqual(got, ['faceswap', 'mask_xseg', 'mask_occluder'])

    def test_the_mask_stage_runs_last(self):
        """Dict order is execution order, and the mask has to land after the
        swap and after the enhancer — it is what blends them back toward the
        plate, so running it earlier would have the enhancer paint over the
        restored occluder."""
        import roop.globals as g
        old = g.selected_enhancer
        try:
            g.selected_enhancer = 'GPEN'
            keys = list(self.build(['mask_xseg', 'mask_occluder']))
        finally:
            g.selected_enhancer = old
        self.assertEqual(keys[0], 'faceswap')
        self.assertEqual(keys[-2:], ['mask_xseg', 'mask_occluder'])

    def test_the_same_engine_twice_is_not_run_twice(self):
        self.assertEqual(list(self.build(['mask_xseg', 'mask_xseg'])),
                         ['faceswap', 'mask_xseg'])


class TheUiNamesMapToEngines(unittest.TestCase):
    def setUp(self):
        from api import map_mask_engines
        self.map = map_mask_engines

    def test_one_engine_maps_to_a_bare_string(self):
        """So the single-engine case is byte-for-byte the previous call."""
        self.assertEqual(self.map('DFL XSeg', 'None', ''), 'mask_xseg')

    def test_none_and_none_is_none(self):
        self.assertIsNone(self.map('None', 'None', ''))

    def test_a_second_engine_alone_still_works(self):
        """Nothing says the first slot has to be the filled one."""
        self.assertEqual(self.map('None', 'Face Occluder', ''), 'mask_occluder')

    def test_the_pair_maps_to_both_in_order(self):
        self.assertEqual(self.map('DFL XSeg', 'Face Occluder', ''),
                         ['mask_xseg', 'mask_occluder'])

    def test_picking_the_same_engine_twice_collapses(self):
        self.assertEqual(self.map('DFL XSeg', 'DFL XSeg', ''), 'mask_xseg')

    def test_clip2seg_still_needs_its_prompt(self):
        """It is the one engine that is a no-op without text, and that rule has
        to survive being routed through the pair."""
        self.assertIsNone(self.map('Clip2Seg', 'None', ''))
        self.assertEqual(self.map('Clip2Seg', 'None', 'hands'), 'mask_clip2seg')


class TheSettingIsWiredEndToEnd(unittest.TestCase):
    """Read as text, across the Python/JavaScript boundary, the same way
    test_settings_wiring.py does — there is no other way to check that the
    control, the payload and the reader all agree on one name."""

    def setUp(self):
        self.api = open(os.path.join(APP, 'api.py'), encoding='utf-8').read()
        ui = os.path.join(os.path.dirname(APP), 'react-ui', 'src', 'components')
        self.jsx = open(os.path.join(ui, 'FaceSwap.jsx'), encoding='utf-8').read()
        self.defaults = open(os.path.join(ui, 'faceswap', 'defaults.js'),
                             encoding='utf-8').read()

    def test_the_ui_has_a_control_for_it(self):
        self.assertIn("set('mask_engine_2'", self.jsx)

    def test_the_ui_sends_it(self):
        self.assertIn('mask_engine_2: activeParams.mask_engine_2', self.jsx)
        self.assertIn('mask_engine_2: sp.mask_engine_2', self.jsx)

    def test_reset_defaults_covers_it(self):
        self.assertIn('mask_engine_2:', self.defaults)

    def test_both_render_paths_read_it(self):
        """The preview and the run are separate code paths that have drifted
        before; a setting that only one of them reads makes the preview lie."""
        self.assertEqual(self.api.count('map_mask_engines('), 3)   # def + 2 calls
        self.assertIn('payload.get("mask_engine_2", "None")', self.api)
        self.assertIn('"mask_engine_2", getattr(roop_globals.CFG, "mask_engine_2"',
                      self.api)

    def test_it_round_trips_through_the_config(self):
        import json
        import tempfile

        import yaml

        from settings import Settings
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'config.json')
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump({'mask_engine_2': 'Face Occluder'}, fh)
            s = Settings(path)
            self.assertEqual(s.mask_engine_2, 'Face Occluder')
            s.save()
            with open(path, encoding='utf-8') as fh:
                self.assertEqual(yaml.safe_load(fh)['mask_engine_2'],
                                 'Face Occluder')

    def test_it_defaults_to_the_previous_single_engine_behaviour(self):
        from settings import Settings
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            s = Settings(os.path.join(d, 'config.json'))
            self.assertEqual(s.mask_engine_2, 'None')


if __name__ == '__main__':
    unittest.main()
