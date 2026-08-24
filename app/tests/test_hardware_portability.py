"""One config.yaml, two machines: an RTX 4070 12GB and an RTX 3060 6GB.

Everything in this project's perf surface is derived from the GPU — pool sizes
(session_pool's VRAM tiers), thread count (Settings.resolve_threads), the
stabilization chunk budget (ProcessMgr._default_stab_chunk_mb). The defaults for
all of them are already 'auto' and already correct per card. The hazard is that
once a value is SAVED it stops being auto, and config.yaml is a file that gets
copied between machines.

The concrete numbers that make this not theoretical. The 4070 box's live config
carries `perf_detmask_pool: '4'`, `max_threads: 10`, `auto_thread_selection:
false`. session_pool's tier for a card under 7 GB is 0/0, because on 6 GB the
extra TensorRT contexts thrash — measured at 2-2.5 fps against 45.3 fps at the
right size. So carrying those three keys across is not a mis-tune, it is the
difference between a render and a hang.

So: hardware-derived values are stamped with the hardware, and reset when it
changes. PREFERENCES are not — a user who picked RealSwap and GPEN 256 Pro
picked those for how they look, not for the card, and must keep them.
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import yaml                                                   # noqa: E402
import settings as settings_mod                               # noqa: E402


# The two real machines this has to serve.
MAIN = {'gpu': 'NVIDIA GeForce RTX 4070', 'vram_gb': 12.0, 'ram_gb': 32}
OTHER = {'gpu': 'NVIDIA GeForce RTX 3060', 'vram_gb': 6.0, 'ram_gb': 16}


class _AsMachine:
    """Load Settings as if this process were running on `hw`."""

    def __init__(self, hw):
        self.hw = hw

    def __enter__(self):
        self._real = settings_mod.detect_hardware
        settings_mod.detect_hardware = lambda: dict(self.hw)
        return self

    def __exit__(self, *a):
        settings_mod.detect_hardware = self._real


class HardwarePortability(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = os.path.join(self._tmp.name, 'config.yaml')

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, **keys):
        with open(self.cfg, 'w') as fh:
            yaml.safe_dump(keys, fh)

    def _load_on(self, hw):
        with _AsMachine(hw):
            return settings_mod.Settings(self.cfg)

    # ── the signature itself ─────────────────────────────────────────────────

    def test_the_two_machines_do_not_share_a_signature(self):
        self.assertNotEqual(settings_mod.hardware_signature(MAIN),
                            settings_mod.hardware_signature(OTHER))

    def test_signature_is_stable_for_the_same_machine(self):
        self.assertEqual(settings_mod.hardware_signature(MAIN),
                         settings_mod.hardware_signature(dict(MAIN)))

    def test_unknown_hardware_has_no_signature(self):
        """An empty signature must never MATCH anything, and never trigger a
        reset either — a machine we cannot identify gets left alone."""
        self.assertEqual(settings_mod.hardware_signature(
            {'gpu': '', 'vram_gb': 0.0, 'ram_gb': 0}), '')

    # ── the reset ────────────────────────────────────────────────────────────

    def test_pinned_perf_values_do_not_follow_the_config_to_another_card(self):
        """The whole point. These exact values are what the 4070 box has saved."""
        self._write(hardware_signature=settings_mod.hardware_signature(MAIN),
                    perf_detmask_pool='4', perf_trt_pool='4', perf_detector_pool='4',
                    perf_expr_pool='2', max_threads=10, auto_thread_selection=False,
                    benchmark_results={'best_threads': {'standard': 10}})
        s = self._load_on(OTHER)
        self.assertEqual(s.perf_detmask_pool, 'auto')
        self.assertEqual(s.perf_trt_pool, 'auto')
        self.assertEqual(s.perf_detector_pool, 'auto')
        self.assertEqual(s.perf_expr_pool, 'auto')
        self.assertTrue(s.auto_thread_selection)
        self.assertEqual(s.benchmark_results, {},
                         "best_threads from the other card outranks the VRAM tier")

        # RE-DERIVED, not merely "different from 10". `_load_on` swaps the
        # signature, not torch's idea of the card, so the value this machine
        # derives is whatever ITS tier gives -- which can coincide with the
        # saved one. Asserting inequality made this test pass by arithmetic
        # accident: it broke the moment the derivation's own knee landed on 10.
        self._write()                       # no saved max_threads at all
        fresh = self._load_on(OTHER).max_threads
        self.assertEqual(s.max_threads, fresh,
                         "the saved thread count must be replaced by this "
                         "machine's derived default, whatever that is")

    def test_the_same_machine_keeps_everything(self):
        """A reset that fires on every load would be a bug, not a safeguard."""
        self._write(hardware_signature=settings_mod.hardware_signature(MAIN),
                    perf_detmask_pool='4', max_threads=10, auto_thread_selection=False,
                    benchmark_results={'best_threads': {'standard': 10}})
        s = self._load_on(MAIN)
        self.assertEqual(s.perf_detmask_pool, '4')
        self.assertEqual(s.max_threads, 10)
        self.assertFalse(s.auto_thread_selection)
        self.assertEqual(s.benchmark_results, {'best_threads': {'standard': 10}})

    def test_an_unstamped_config_is_left_alone(self):
        """Upgrading into this feature must not wipe the settings of the machine
        that is already working. First load stamps; it does not reset."""
        self._write(perf_detmask_pool='4', max_threads=10, auto_thread_selection=False)
        s = self._load_on(MAIN)
        self.assertEqual(s.perf_detmask_pool, '4')
        self.assertEqual(s.max_threads, 10)
        self.assertEqual(s.hardware_signature, settings_mod.hardware_signature(MAIN))

    def test_preferences_survive_the_move(self):
        """Model and output choices are the user's, not the card's."""
        self._write(hardware_signature=settings_mod.hardware_signature(MAIN),
                    perf_detmask_pool='4',
                    swap_model='realswap', mask_engine='RealityUX',
                    selected_enhancer='GPEN 256 Pro', output_template='{file}_{time}',
                    selected_theme='Midnight', video_quality=14)
        s = self._load_on(OTHER)
        self.assertEqual(s.perf_detmask_pool, 'auto')      # reset
        self.assertEqual(s.swap_model, 'realswap')         # kept
        self.assertEqual(s.mask_engine, 'RealityUX')
        self.assertEqual(s.selected_enhancer, 'GPEN 256 Pro')
        self.assertEqual(s.output_template, '{file}_{time}')
        self.assertEqual(s.selected_theme, 'Midnight')
        self.assertEqual(s.video_quality, 14)

    def test_nvenc_survives_an_nvidia_to_nvidia_move(self):
        """Both machines are NVIDIA. Resetting the encoder here would be a
        pointless downgrade to CPU encoding."""
        self._write(hardware_signature=settings_mod.hardware_signature(MAIN),
                    output_video_codec='hevc_nvenc')
        self.assertEqual(self._load_on(OTHER).output_video_codec, 'hevc_nvenc')

    def test_nvenc_is_dropped_when_the_card_is_not_nvidia(self):
        self._write(hardware_signature=settings_mod.hardware_signature(MAIN),
                    output_video_codec='hevc_nvenc')
        amd = {'gpu': 'AMD Radeon RX 7900 XTX', 'vram_gb': 24.0, 'ram_gb': 32}
        self.assertEqual(self._load_on(amd).output_video_codec, 'libx264')

    def test_the_signature_is_written_back(self):
        """A reset that does not re-stamp would fire again on every single load."""
        self._write(hardware_signature=settings_mod.hardware_signature(MAIN),
                    perf_detmask_pool='4')
        with _AsMachine(OTHER):
            s = settings_mod.Settings(self.cfg)
            s.save()
        with open(self.cfg) as fh:
            saved = yaml.safe_load(fh)
        self.assertEqual(saved['hardware_signature'], settings_mod.hardware_signature(OTHER))
        self.assertEqual(saved['perf_detmask_pool'], 'auto')

        # ...and a second load on that machine is now a no-op.
        s2 = self._load_on(OTHER)
        self.assertFalse(s2._hardware_changed)


class AutoTiersCoverBothCards(unittest.TestCase):
    """With the pins gone, 'auto' has to actually produce different numbers."""

    def test_pool_tiers_differ_between_6gb_and_12gb(self):
        from roop import session_pool
        real = session_pool._detect_vram_gb
        try:
            session_pool._detect_vram_gb = lambda: 6.0
            small = session_pool._auto_pool_defaults()
            session_pool._detect_vram_gb = lambda: 12.0
            large = session_pool._auto_pool_defaults()
        finally:
            session_pool._detect_vram_gb = real
        self.assertEqual(small, (0, 0), "a 6GB card must not get pooled contexts")
        self.assertNotEqual(small, large)
        self.assertTrue(all(v > 0 for v in large))

    def test_expression_pool_is_off_on_a_small_card(self):
        from roop import session_pool
        real = session_pool._detect_vram_gb
        try:
            session_pool._detect_vram_gb = lambda: 6.0
            self.assertEqual(session_pool._auto_expression_pool(), 0)
            session_pool._detect_vram_gb = lambda: 12.0
            self.assertGreater(session_pool._auto_expression_pool(), 0)
        finally:
            session_pool._detect_vram_gb = real

    def test_thread_tiers_differ_between_the_two_cards(self):
        """resolve_threads reads VRAM directly, so a 6GB card must ask for fewer
        workers than a 12GB one even with the same CPU."""
        s = object.__new__(settings_mod.Settings)
        s.auto_thread_selection = True
        s.benchmark_results = {}
        s.max_threads = 8
        import torch
        if not torch.cuda.is_available():
            self.skipTest('no CUDA device to read tiers against')
        real = torch.cuda.get_device_properties

        class _P:
            def __init__(self, gb):
                self.total_memory = int(gb * 1024 ** 3)

        try:
            torch.cuda.get_device_properties = lambda i: _P(6.0)
            small = s.resolve_threads('standard')
            torch.cuda.get_device_properties = lambda i: _P(12.0)
            large = s.resolve_threads('standard')
        finally:
            torch.cuda.get_device_properties = real
        self.assertLess(small, large, f'6GB asked for {small}, 12GB for {large}')

    def test_the_derived_default_is_not_one_worker_per_1_5gb(self):
        """The fresh-install `max_threads`, which is what a NEW card gets.

        It used to be `min(cores - 1, vram_gb / 1.5)` -- one worker per 1.5GB --
        on the premise that workers cost VRAM. They do not: the models live in
        per-model SessionPools sized by session_pool, not by the worker count.
        Measured under the <7GB policy (pools 0/0), own VRAM over a whole
        render, threads 4/6/8/10/12: 2317/2339/2374/2329/2369 MB -- flat, while
        fps went 14.4/17.5/18.1/17.8/17.5. The old rule derived FOUR on a 6GB
        card, a fifth below the knee, and seven on a 12GB/24-core one where 10
        measures 22.2 fps against 8's 20.1.

        Asserted against the tier knees, not against the old arithmetic.
        """
        import psutil
        import torch
        if not torch.cuda.is_available():
            self.skipTest('no CUDA device')
        real_props = torch.cuda.get_device_properties
        real_cores = psutil.cpu_count

        class _P:
            def __init__(self, gb):
                self.total_memory = int(gb * 1024 ** 3)

        def derive(gb, phys):
            torch.cuda.get_device_properties = lambda i: _P(gb)
            psutil.cpu_count = lambda logical=True: (64 if logical else phys)
            cfg = settings_mod.Settings.__new__(settings_mod.Settings)
            cfg.config_file = 'nonexistent-for-this-test.yaml'
            cfg.load()
            return cfg.max_threads

        try:
            self.assertEqual(derive(11.99, 24), 10,
                             'a 12GB/24-core machine must derive the measured knee')
            self.assertEqual(derive(6.0, 8), 7,
                             'a 6GB/8-core machine must not be held at 4')
            self.assertEqual(derive(6.0, 4), 3,
                             'still bounded by the CPU on a small host')
        finally:
            torch.cuda.get_device_properties = real_props
            psutil.cpu_count = real_cores


if __name__ == '__main__':
    unittest.main()
