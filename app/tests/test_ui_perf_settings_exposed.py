"""Every performance tunable the app persists must be reachable in the UI.

WHY THIS IS A SEPARATE GUARD FROM test_ui_settings_catalog.py

That one checks the Settings panel and the command-palette catalogue list the
SAME keys. It is symmetric, so it is blind to the case that actually happened:
a setting added to ``settings.py`` and wired all the way through to a
``ROOP_*`` environment variable, but never surfaced in the UI at all. Both
sides are empty, parity holds, and the guard passes while the user has no way
to see or change the setting.

Measured, 2026-09-05: ``perf_gpu_mem_limit``, ``perf_ort_arena_strategy`` and
``perf_cudnn_conv_algo`` were added for the benchmark's Apply path and reached
run.py and core.py, but nothing in ``react-ui``. In the same audit
``cpu_ort_intra_threads``, ``cpu_ort_inter_threads`` and ``cpu_ffmpeg_threads``
turned out to have been invisible for much longer -- while their sibling
``cpu_opencv_threads`` was right there in the panel.

SCOPE, AND WHY IT IS NARROW

Only ``perf_*`` and ``cpu_*`` are required. Those are unambiguously operator
tunables: each maps to a ROOP_* flag the runtime reads, and a benchmark can
recommend changing one. The project has other persisted keys that are
DELIBERATELY not in the UI -- the twelve ``temporal_*`` / ``parser_*`` /
``identity_detail_strength`` settings whose own handoffs record them as
OPEN/INCOMPLETE -- and this guard must not become a reason to expose work that
was consciously held back.
"""
import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')
SETTINGS_PY = os.path.join(APP, 'settings.py')
SETTINGS_JSX = os.path.join(SRC, 'components', 'Settings.jsx')
CATALOG_JS = os.path.join(SRC, 'components', 'settingsCatalog.js')
RUN_PY = os.path.join(APP, 'run.py')

BOUND = re.compile(r"\bbind(?:Toggle)?\(\s*'([a-z0-9_]+)'")
CATALOGUED = re.compile(r"key:\s*'([a-z0-9_]+)'")

# Persisted keys that are intentionally absent from the panel. Each needs a
# reason, because "it is in the allowlist" is how a real gap gets normalised.
DELIBERATELY_HIDDEN = {
    # Derived per machine and rewritten on save; showing it would invite
    # editing a value the app owns.
    'hardware_signature': 'app-derived provenance, not a user choice',
    # The benchmark's own result blob, rendered by the benchmark panel rather
    # than as a settings field.
    'benchmark_results': 'rendered by the benchmark panel',
}


def _read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def _persisted_keys():
    """Keys written by Settings.save() -- i.e. what actually survives a restart."""
    source = _read(SETTINGS_PY)
    match = re.search(r"def save\(self\):\s*\n\s*data = \{(.*?)\n        \}",
                      source, re.S)
    assert match, "could not parse Settings.save() -- the guard is dead"
    return set(re.findall(r"'([a-z0-9_]+)'\s*:", match.group(1)))


class PerformanceSettingsAreReachable(unittest.TestCase):
    def setUp(self):
        self.persisted = _persisted_keys()
        self.bound = set(BOUND.findall(_read(SETTINGS_JSX)))
        self.catalogued = set(CATALOGUED.findall(_read(CATALOG_JS)))
        self.tunables = {key for key in self.persisted
                         if key.startswith(('perf_', 'cpu_'))
                         and key not in DELIBERATELY_HIDDEN}

    def test_the_guard_actually_found_something_to_check(self):
        self.assertGreater(len(self.tunables), 5,
                           "parsed almost nothing out of settings.py")

    def test_every_persisted_perf_tunable_is_bound_in_the_panel(self):
        missing = sorted(self.tunables - self.bound)
        self.assertEqual([], missing,
                         "persisted and wired to the runtime, but invisible in "
                         "the UI: %s" % missing)

    def test_every_persisted_perf_tunable_is_findable_in_the_palette(self):
        missing = sorted(self.tunables - self.catalogued)
        self.assertEqual([], missing,
                         "bound in the panel but unfindable from the command "
                         "palette: %s" % missing)

    def test_a_bound_setting_is_one_the_app_actually_persists(self):
        """A control that writes a key save() drops is a control that does
        nothing the moment the app restarts."""
        phantom = sorted(key for key in self.bound
                         if key.startswith(('perf_', 'cpu_'))
                         and key not in self.persisted)
        self.assertEqual([], phantom,
                         "bound in the panel but never persisted: %s" % phantom)

    def test_private_bookkeeping_keys_are_never_bound(self):
        """`_threads_auto` records whether a HUMAN chose the thread count.

        Round-tripping it through the UI lets a stale copy be written back
        after max_threads, undoing the record -- and which lands last is dict
        ordering, i.e. luck.
        """
        leaked = sorted(key for key in self.bound if key.startswith('_'))
        self.assertEqual([], leaked)

    def test_env_backed_tunables_are_exported_by_run_py(self):
        """A setting that saves but is never exported reaches no runtime.

        Scoped to the three this session added, because a general rule would
        have to encode every ROOP_* name and would decay into a list nobody
        updates.
        """
        exported = _read(RUN_PY)
        for key in ('perf_ort_arena_strategy', 'perf_cudnn_conv_algo',
                    'perf_gpu_mem_limit'):
            with self.subTest(setting=key):
                self.assertIn(key, exported,
                              "%s is saved but never reaches the environment" % key)

    def test_the_exported_env_names_are_the_ones_core_reads(self):
        """The names had to be corrected once already.

        The optimizer originally declared ROOP_ORT_ARENA_STRATEGY and
        ROOP_ORT_GPU_MEM_LIMIT_GB; core.py reads ROOP_CUDA_ARENA_STRATEGY and
        ROOP_CUDA_MEM_LIMIT. A setting exported under a name nothing reads
        saves, displays, and does nothing.
        """
        core = _read(os.path.join(APP, 'roop', 'core.py'))
        run = _read(RUN_PY)
        for name in ('ROOP_CUDA_ARENA_STRATEGY', 'ROOP_CUDA_MEM_LIMIT',
                     'ROOP_CUDNN_CONV_ALGO'):
            with self.subTest(env=name):
                self.assertIn(name, core, "%s is exported but core.py never "
                                          "reads it" % name)
                self.assertIn(name, run, "%s is read but never exported" % name)


if __name__ == '__main__':
    unittest.main()
