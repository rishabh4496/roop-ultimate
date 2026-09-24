"""The per-render performance settings reach their flags without a restart,
and each flag is actually READ by the code it names (AGENTS.md: a control bound
to a value nothing consumes looks completely wired)."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import settings                                  # noqa: E402


class LiveEnv(unittest.TestCase):

    def setUp(self):
        self.env = {}
        self.owned = set(settings._SETTINGS_OWNED_VARS)
        settings._SETTINGS_OWNED_VARS.clear()

    def tearDown(self):
        settings._SETTINGS_OWNED_VARS.clear()
        settings._SETTINGS_OWNED_VARS.update(self.owned)

    def test_every_live_key_is_an_env_setting(self):
        keys = {k for k, _v, _kind in settings.ENV_SETTINGS}
        self.assertTrue(set(settings.LIVE_ENV_SETTINGS) <= keys)

    def test_save_exports_the_new_value(self):
        settings.apply_live_env({'perf_batch_max': '4', 'perf_nvenc_preset': 'p7',
                                 'perf_gpu_affine': 'off', 'perf_pinned_buffers': 'off',
                                 'temporal_step': 1}, self.env)
        self.assertEqual(self.env, {'ROOP_BATCH_SWAP_MAX': '4', 'ROOP_NVENC_PRESET': 'p7',
                                    'ROOP_GPU_AFFINE': '0', 'ROOP_PINNED_BUFFERS': '0',
                                    'ROOP_TEMPORAL_STEP': '1'})

    def test_only_live_keys_are_touched(self):
        settings.apply_live_env({'perf_batch_swap': 'on', 'recognizer': 'adaface'}, self.env)
        self.assertEqual(self.env, {}, 'non-live flags stay startup-only')

    def test_back_to_auto_really_returns_to_the_default(self):
        settings._SETTINGS_OWNED_VARS.add('ROOP_BATCH_SWAP_MAX')
        self.env['ROOP_BATCH_SWAP_MAX'] = '4'
        settings.apply_live_env({'perf_batch_max': 'auto'}, self.env)
        self.assertNotIn('ROOP_BATCH_SWAP_MAX', self.env)

    def test_a_launcher_value_is_never_overwritten(self):
        self.env['ROOP_NVENC_PRESET'] = 'p3'          # set outside settings
        settings.apply_live_env({'perf_nvenc_preset': 'p7'}, self.env)
        self.assertEqual(self.env['ROOP_NVENC_PRESET'], 'p3')

    def test_defaults_are_the_shipped_behaviour(self):
        """'auto' / 1 / 1.5 reproduce what ran before these settings existed."""
        source = open(settings.__file__, encoding='utf-8').read()
        for key, default in (('perf_batch_max', 'auto'), ('perf_nvenc_preset', 'auto'),
                             ('perf_gpu_affine', 'auto'), ('perf_pinned_buffers', 'auto'),
                             ('temporal_step', 1), ('vram_safety_margin_gb', 1.5)):
            self.assertIn(f"default_get(data, '{key}', {default!r})", source)


class FlagsAreRead(unittest.TestCase):

    def test_gpu_affine_off_sends_every_caller_to_opencv(self):
        from roop import utilities
        img = np.zeros((8, 8, 3), np.uint8)
        with mock.patch.dict(os.environ, {'ROOP_GPU_AFFINE': '0'}):
            self.assertIsNone(utilities.cuda_warp_affine(img, np.eye(2, 3), (8, 8)))

    def test_pinned_off_disables_page_locked_buffers(self):
        from roop import buffer_pool
        with mock.patch.dict(os.environ, {'ROOP_PINNED_BUFFERS': '0'}):
            self.assertFalse(buffer_pool.is_pinned_supported())
            buf = buffer_pool.allocate_pinned_buffer((4, 4, 3))
            self.assertIsInstance(buf, np.ndarray)

    def test_each_live_flag_has_a_reader(self):
        """Grep, not trust: every live variable is read somewhere in roop/."""
        root = os.path.join(os.path.dirname(settings.__file__), 'roop')
        corpus = ''
        for folder, _d, files in os.walk(root):
            for name in files:
                if name.endswith('.py'):
                    corpus += open(os.path.join(folder, name), encoding='utf-8',
                                   errors='ignore').read()
        for key, var, _kind in settings.ENV_SETTINGS:
            if key in settings.LIVE_ENV_SETTINGS:
                self.assertIn(f"'{var}'", corpus.replace('"', "'"), var)


class TrtCacheClassification(unittest.TestCase):

    NS = ('mixed_NVIDIA_GeForce_RTX_4070_sm0809_cuda12.8_drv616.56_trt10.9.0.34_'
          'ort1.23.2_lnfp32_seq_heur_b3_a-1_g0_c40b4a7d8494df527')
    OLD = NS.replace('a-1_g0_c40b4a7d8494df527', 'a0_g0_c3b1a9752fee69034')

    def test_active_namespace_and_its_children_are_kept(self):
        import routes_trt_cache as r
        kinds = r.classify([self.NS, self.NS + '_sp12x512input8x3x1280x1280',
                            self.NS + '_gpen_1024_fp32', self.OLD,
                            self.OLD + '_spx512x512', 'stage3_gpen_smoke',
                            'mixed_gpen_fp32'], '/x/' + self.NS)
        self.assertEqual(kinds[self.NS], 'active')
        self.assertEqual(kinds[self.NS + '_sp12x512input8x3x1280x1280'], 'active')
        self.assertEqual(kinds[self.NS + '_gpen_1024_fp32'], 'active')
        self.assertEqual(kinds[self.OLD], 'stale')
        self.assertEqual(kinds[self.OLD + '_spx512x512'], 'stale')
        self.assertEqual(kinds['stage3_gpen_smoke'], 'other')
        self.assertEqual(kinds['mixed_gpen_fp32'], 'other')

    def test_unknown_active_means_nothing_is_stale(self):
        import routes_trt_cache as r
        kinds = r.classify([self.NS, self.OLD], None)
        self.assertNotIn('stale', kinds.values())


if __name__ == '__main__':
    unittest.main()
