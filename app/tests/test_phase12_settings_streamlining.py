"""Phase 12: settings that were removed stay removed, and what the pipeline now decides for itself is decided.

  * Sparse pre-scan, disk-frame extraction, Keep frames, Wait before creating video, Skip audio and the Video method are no
    longer settings: not in config.yaml's loader or saver, not in settings.schema.json, not in default_config.yaml, and
    POST /api/swap renders through the in-memory streaming pipeline with the 3-stage scan whatever the payload says
    (`roop/render_policy.py`), except for the two INTERNAL payload channels that are documented there.
  * The global TensorRT precision offers 'mixed' and 'fp32'. 'fp16' is not a choice: a global FP16 request reaches
    every model whose policy row is only a `candidate` (the GPEN family, detectors, recognition, masks) - precision the
    policy has not gated. The per-model pins (xseg, 2d106det, 1k3d68, resnet18, RestoreFormer++) win over EVERY global
    value; that is asserted here too, because the brief that removed the choice assumed they could conflict.
  * The frozen Gradio UI reads the retired attributes while it builds its widgets, so they survive as class constants
    that are neither loaded, saved nor listed in the schema.
"""
import json
import os
import re
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

RETIRED = ('sparse_auto_range', 'keep_frames', 'skip_audio', 'wait_after_extraction', 'video_swapping_method')


def _read(*parts):
    with open(os.path.join(APP, *parts), encoding='utf-8') as fh:
        return fh.read()


class RetiredSettings(unittest.TestCase):
    def test_not_in_the_schema_or_the_default_config(self):
        schema = json.loads(_read('settings.schema.json'))
        keys = set(schema.get('properties', schema))
        for name in RETIRED:
            self.assertNotIn(name, keys, f'{name} is back in settings.schema.json')
            self.assertNotIn(f'"{name}"', _read('settings.schema.json'), name)
            self.assertIsNone(re.search(rf'^{name}\s*:', _read('default_config.yaml'), re.M), f'{name} is back in default_config.yaml')

    def test_not_loaded_or_saved(self):
        from settings import Settings
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'config.yaml')
            with open(path, 'w', encoding='utf-8') as fh:
                # an OLD config.yaml that still holds every retired key, set to non-default values
                yaml.safe_dump({'keep_frames': True, 'skip_audio': True, 'wait_after_extraction': True,
                                'video_swapping_method': 'Extract Frames to media', 'sparse_auto_range': False}, fh)
            cfg = Settings(path)
            for name in RETIRED:
                self.assertNotIn(name, cfg.__dict__, f'{name} was loaded from config.yaml into the instance')
            # ...and the class constants the Gradio UI reads are the pipeline's fixed choices, not the old file's
            self.assertEqual(cfg.video_swapping_method, 'In-Memory processing')
            self.assertFalse(cfg.keep_frames or cfg.skip_audio or cfg.wait_after_extraction)
            self.assertTrue(cfg.sparse_auto_range)
            cfg.save()
            saved = yaml.safe_load(open(path, encoding='utf-8')) or {}
            for name in RETIRED:
                self.assertNotIn(name, saved, f'{name} was written back by Settings.save()')

    def test_the_settings_object_the_api_serialises_has_none_of_them(self):
        from settings import Settings
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = Settings(os.path.join(d, 'config.yaml'))
            for name in RETIRED:
                self.assertNotIn(name, cfg.__dict__)          # GET /api/settings serialises __dict__

    def test_the_catalog_and_the_panel_agree(self):
        catalog = _read('..', 'react-ui', 'src', 'components', 'settingsCatalog.js')
        panel = _read('..', 'react-ui', 'src', 'components', 'Settings.jsx')
        for name in RETIRED:
            self.assertNotIn(name, catalog)
            self.assertNotIn(name, panel)


class StreamingRenderFlags(unittest.TestCase):
    def test_a_payload_cannot_pick_the_disk_route(self):
        from roop import render_policy
        hostile = {'video_method': 'Extract Frames to media', 'keep_frames': True, 'wait_after_extraction': True}
        flags = render_policy.streaming_render_flags(hostile)
        self.assertTrue(flags['use_new_method'])
        self.assertFalse(flags['keep_frames'])
        self.assertFalse(flags['wait_after_extraction'])

    def test_the_three_stage_pipeline_is_the_default_and_audio_is_kept(self):
        from roop import render_policy
        for payload in (None, {}, {'enhancer': 'GPEN'}):
            flags = render_policy.streaming_render_flags(payload)
            self.assertTrue(flags['sparse_auto_range'])
            self.assertFalse(flags['skip_audio'])

    def test_the_two_internal_channels_still_work(self):
        """distributed_render renders audio-less chunks; the harness builds the dense 1-stage baseline."""
        from roop import render_policy
        flags = render_policy.streaming_render_flags({'skip_audio': True, 'sparse_auto_range': False})
        self.assertTrue(flags['skip_audio'])
        self.assertFalse(flags['sparse_auto_range'])
        self.assertIn('"skip_audio"', _read('roop', 'distributed_render.py').replace("'", '"'))

    def test_run_swap_uses_the_policy_and_no_longer_reads_the_retired_settings(self):
        api = _read('api.py')
        self.assertIn('render_flags = _render_policy.streaming_render_flags(payload)', api)
        self.assertIn('render_flags["use_new_method"]', api)
        for stale in ('CFG.keep_frames', 'CFG.skip_audio', 'CFG.wait_after_extraction', 'CFG.video_swapping_method',
                      'CFG, "sparse_auto_range"', 'payload.get("video_method"'):
            self.assertNotIn(stale, api, stale)

    def test_the_meta_lists_no_video_method(self):
        self.assertNotIn('"video_methods"', _read('api.py'))


class RunSwapExecutesTheFixedFlags(unittest.TestCase):
    """The real `api._run_swap`, run up to the render call, with the render stubbed to record what it was handed.

    The source-level pins above only read api.py; this EXECUTES the assignments (the standing failure class of this
    project is a flag that is read and never reaches the stage).
    """

    def _run(self, payload):
        import tempfile
        from unittest import mock
        import api
        import roop.globals as g
        from settings import Settings

        seen = {}

        def render(*args, **kwargs):
            seen['use_new_method'] = args[4]
            seen['globals'] = {n: getattr(g, n, None) for n in
                               ('keep_frames', 'wait_after_extraction', 'skip_audio', 'sparse_auto_range')}
            raise RuntimeError('render stubbed')

        request = {'swap_mode': 'all', 'source_index': 0, 'source_index_mapping': [],
                   'selection_state': {}, 'processing_selection': {}}
        # _run_swap writes dozens of roop.globals attributes (merger_clarity, blend_ratio, ...): restore ALL of them, or the
        # tests that read the module's defaults (test_settings_wiring) fail depending on what ran before this one.
        keep = dict(vars(g))
        with tempfile.TemporaryDirectory() as d:
            g.CFG = Settings(os.path.join(d, 'config.yaml'))
            try:
                with mock.patch.object(api, '_target_index_for_ref', lambda p, index=None: (0, 'm1')),                         mock.patch.object(api, '_activate_target_media', lambda **k: None),                         mock.patch.object(api, '_update_mask_offsets_from_payload', lambda p: None),                         mock.patch.object(api._routes_identity, 'apply_identity_blend_from_payload', lambda p: None),                         mock.patch.object(api, '_log_normalized_selection', lambda *a, **k: None),                         mock.patch.object(api, '_clear_output_folder', lambda *a, **k: None),                         mock.patch('ui.main.prepare_environment', lambda: None),                         mock.patch('roop.core.batch_process_regular', render):
                    api._run_swap(dict(payload, normalized_request=request))
            finally:
                for n in [n for n in vars(g) if n not in keep]:
                    delattr(g, n)
                for n, v in keep.items():
                    setattr(g, n, v)
        self.assertIn('use_new_method', seen, 'the render call was never reached: ' + str(api._progress.get('error')))
        return seen

    def test_a_payload_asking_for_the_disk_route_still_streams_in_memory(self):
        seen = self._run({'video_method': 'Extract Frames to media', 'keep_frames': True,
                          'wait_after_extraction': True})
        self.assertTrue(seen['use_new_method'])
        self.assertEqual(seen['globals'], {'keep_frames': False, 'wait_after_extraction': False,
                                           'skip_audio': False, 'sparse_auto_range': True})

    def test_the_internal_channels_reach_the_pipeline(self):
        seen = self._run({'skip_audio': True, 'sparse_auto_range': False})
        self.assertTrue(seen['globals']['skip_audio'])
        self.assertFalse(seen['globals']['sparse_auto_range'])


class TrtPrecisionChoices(unittest.TestCase):
    def test_the_choices_are_mixed_and_fp32(self):
        from settings import TRT_PRECISION_CHOICES
        self.assertEqual(TRT_PRECISION_CHOICES, ('mixed', 'fp32'))
        self.assertIn('list(TRT_PRECISION_CHOICES)', _read('api.py'))

    def test_an_old_config_with_fp16_reads_as_mixed(self):
        from settings import Settings
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'config.yaml')
            with open(path, 'w', encoding='utf-8') as fh:
                yaml.safe_dump({'trt_precision': 'fp16'}, fh)
            self.assertEqual(Settings(path).trt_precision, 'mixed')
            with open(path, 'w', encoding='utf-8') as fh:
                yaml.safe_dump({'trt_precision': 'fp32'}, fh)
            self.assertEqual(Settings(path).trt_precision, 'fp32')

    def test_per_model_pins_win_over_every_global_value(self):
        """The premise of the brief: the global setting could conflict with the pins. It cannot."""
        from roop import precision_policy as pp
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        pinned = {'xseg': 'xseg.onnx', '2d106det': '2d106det.onnx', '1k3d68': '1k3d68.onnx',
                  'resnet18': 'resnet18.onnx', 'restoreformer_plus_plus': 'restoreformer_plus_plus.onnx'}
        with patch.object(pp, 'write_decision_cache', lambda decision: None):
            for glob in ('mixed', 'fp32', 'fp16'):
                with patch.object(pp, '_global_precision', lambda g=glob: g):
                    for key, path in pinned.items():
                        _, decision = pp.providers_for(key, providers, path)
                        self.assertEqual(decision.effective, 'fp32', f'{key} left FP32 under global {glob}')

    def test_a_global_fp16_reaches_models_the_policy_has_not_gated(self):
        """Why 'fp16' is not offered: with the choice set to it, a `candidate` row runs at FP16 while 'mixed' is `safe`."""
        from roop import precision_policy as pp

        class HW:                       # a card that can do everything the request names
            cuda_available = tensorrt_available = fp16_supported = True
            bf16_supported = int8_supported = fp8_supported = False

        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        # every row the policy calls fp16=candidate / mixed=safe, addressed by the key it resolves to itself
        rows = [k for k, pol in pp.POLICIES.items()
                if pol.fp16 == 'candidate' and pol.mixed == 'safe' and pp.canonical_model_key(k, None) == k]
        self.assertIn('gpen_256', rows)
        self.assertGreaterEqual(len(rows), 3, rows)
        for key in rows:
            self.assertEqual(pp.resolve(key, 'fp16', providers, hardware=HW()).effective, 'fp16', key)
            self.assertEqual(pp.resolve(key, 'mixed', providers, hardware=HW()).effective, 'mixed', key)


if __name__ == '__main__':
    unittest.main()
