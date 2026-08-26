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
            self.assertEqual(derive(6.0, 8), 8,
                             'a 6GB/8-core machine must reach its measured knee')
            self.assertEqual(derive(6.0, 4), 3,
                             'still bounded by the CPU on a small host')
        finally:
            torch.cuda.get_device_properties = real_props
            psutil.cpu_count = real_cores


class DerivedValuesDoNotOutliveTheirRule(unittest.TestCase):
    """A derived default must not be mistaken for a user's choice.

    `save()` writes `max_threads` on every settings save, so the first save
    freezes whatever formula was in force -- and `_hw_get` only re-derives when
    the GPU changes, not when the RULE does. The RTX 3060 6GB that the current
    formula was rewritten FOR (commit 0eda23b) was found on 2026-08-25 still
    running max_threads 4, a full tier below its measured knee of 8, with the
    new rule sitting in the source unable to reach it.

    So the number now carries provenance. These tests pin both directions: a
    stale DERIVED value gets corrected, and a value a person chose never does.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = os.path.join(self._tmp.name, 'config.yaml')

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, **keys):
        with open(self.cfg, 'w') as fh:
            yaml.safe_dump(keys, fh)

    def _load(self):
        return settings_mod.Settings(self.cfg)

    def _derived(self):
        """What this machine derives with nothing saved."""
        empty = os.path.join(self._tmp.name, 'empty.yaml')
        return settings_mod.Settings(empty).max_threads

    def test_a_fresh_derive_is_marked_as_derived(self):
        self._write()
        s = self._load()
        self.assertTrue(s._threads_auto)
        self.assertEqual(s.max_threads, self._derived())

    def test_a_stale_legacy_value_below_the_knee_is_corrected(self):
        """The 3060 case, exactly: an unstamped config carrying an old rule's
        output, which nothing could tell apart from a choice."""
        derived = self._derived()
        self._write(max_threads=max(1, derived - 4))
        self.assertEqual(self._load().max_threads, derived)

    def test_a_legacy_value_ABOVE_the_derived_one_is_left_alone(self):
        """One-directional on purpose. Raising a thread count costs nothing
        measurable (workers hold frame buffers, not model weights), but lowering
        one somebody raised deliberately is the silent downgrade this project
        keeps having to hunt down."""
        self._write(max_threads=self._derived() + 6)
        s = self._load()
        self.assertEqual(s.max_threads, self._derived() + 6)
        self.assertFalse(s._threads_auto)

    def test_a_legacy_value_EQUAL_to_the_derived_one_stays_derived(self):
        """The 4070's case, and the subtlest of the three.

        Its config already carries the knee, so nothing needs correcting — but
        filing it as "the user chose this" would pin it, and the NEXT rule
        change would then fail to reach the main machine. That is precisely the
        bug this stamp exists to fix, so a value indistinguishable from the
        derived one is treated as derived. It costs nothing if it really was
        typed: re-deriving reproduces the same number.
        """
        derived = self._derived()
        self._write(max_threads=derived)
        s = self._load()
        self.assertEqual(s.max_threads, derived)
        self.assertTrue(s._threads_auto,
                        'a value equal to the derived one must not be pinned')

    def test_a_users_choice_survives_a_reload(self):
        derived = self._derived()
        s = self._load()
        s.max_threads = 2                    # deliberately below the knee
        s.save()
        again = self._load()
        self.assertEqual(again.max_threads, 2,
                         'a number the user set must never be re-derived')
        self.assertFalse(again._threads_auto)
        self.assertNotEqual(derived, 2, 'test is vacuous if these coincide')

    def test_echoing_an_untouched_value_back_is_not_a_choice(self):
        """The settings panel POSTs the WHOLE object on any unrelated save, so
        the thread slider arrives here having not been touched. Counting that as
        a choice would re-pin the derived value and recreate the bug."""
        self._write()
        s = self._load()
        self.assertTrue(s._threads_auto)
        s.max_threads = s.max_threads        # the echo
        s.video_quality = 18                 # what actually changed
        self.assertTrue(s._threads_auto)

    def test_a_derived_value_is_re_derived_when_the_rule_version_moves(self):
        derived = self._derived()
        self._write(max_threads=max(1, derived - 3),
                    _threads_auto=True, _threads_basis='v1|8|8')
        self.assertEqual(self._load().max_threads, derived)

    def test_a_derived_value_is_re_derived_when_the_core_count_moves(self):
        """Same rule, different CPU. `hardware_signature` deliberately excludes
        the CPU (it does not change a pool size) -- but since 0eda23b the thread
        default IS derived from the core count, so the basis has to carry it."""
        derived = self._derived()
        self._write(max_threads=max(1, derived - 3), _threads_auto=True,
                    _threads_basis=f'v{settings_mod._THREAD_RULE}|2|8')
        self.assertEqual(self._load().max_threads, derived)

    def test_provenance_round_trips_through_save(self):
        self._write()
        self._load().save()
        with open(self.cfg) as fh:
            saved = yaml.safe_load(fh)
        self.assertIn('_threads_auto', saved)
        self.assertIn('_threads_basis', saved)


class SettingsInternalsAreNotSettings(unittest.TestCase):
    """`GET /api/settings` serialises `CFG.__dict__`, which had been shipping
    Settings' private bookkeeping to the UI as though each key were a setting.

    That matters beyond tidiness for `_threads_auto`: it records whether the
    thread count in the very same payload was chosen by a person, so a stale
    copy of it POSTed back after `max_threads` would undo the record -- and
    which of the two lands last is dict ordering, i.e. luck.
    """

    def test_private_attributes_are_filtered_out(self):
        import api
        cfg = settings_mod.Settings(
            os.path.join(tempfile.gettempdir(), '__no_such_config__.yaml'))
        public = api._public_settings(cfg)
        self.assertIn('max_threads', public)
        for private in ('_threads_auto', '_threads_basis', '_hardware_changed', '_loading'):
            self.assertNotIn(private, public)

    def test_save_settings_skips_private_keys(self):
        """Source-level: the endpoint needs a live app to call."""
        import inspect
        import api
        src = inspect.getsource(api.save_settings)
        self.assertIn("startswith('_')", src,
                      'save_settings must refuse Settings-internal keys')


if __name__ == '__main__':
    unittest.main()
