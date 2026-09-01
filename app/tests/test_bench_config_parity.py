"""The benchmark harness must render the stack config.yaml describes.

This guards the defect class that has now invalidated measurements three times,
each time in a different harness and each time silently:

  * every saved `yaw_*` arm ran the swap-model mask OFF while production ran 25;
  * every `angle_bench` arm before 2026-08-23 ran the whole merger stage OFF;
  * `two_face_video.py` -- the end-to-end harness `baseline_controlled.py` and
    the entire Phase/Gate campaign run through -- ran with
    `target_conditioned_appearance` False against a config carrying True,
    `detail_transfer_strength` 0.0 against 0.4, `color_match_after_enhance`
    False against True, `codeformer_fidelity` 0.5 against 0.55 and
    `parser_regions` None against the configured five regions.

Each was invisible because an unstated setting does not error, does not warn,
and does not appear in the arm's own record -- it simply takes roop/globals.py's
module default. The fix is `config_sync.sync_globals_from_config`, wired into
`angle_bench.init_pipeline(sync_config=True)`; these tests fail if that wiring
is removed or if the sync stops being exhaustive.

No models are loaded here on purpose: the contract under test is which values
reach `roop.globals`, and that is decidable without a GPU.
"""
import ast
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_sync import TRANSLATED, sync_globals_from_config   # noqa: E402
from settings import Settings                                  # noqa: E402


class _FakeGlobals(types.SimpleNamespace):
    pass


def _globals_defaults():
    """roop/globals.py's module-level defaults, read without importing roop."""
    with open(os.path.join(APP, "roop", "globals.py"), encoding="utf-8") as fh:
        src = fh.read()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


class ConfigSyncExhaustive(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings(os.path.join(APP, "config.yaml"))
        self.defaults = _globals_defaults()

    def _synced(self):
        g = _FakeGlobals(**dict(self.defaults))
        g.CFG = self.cfg
        sync_globals_from_config(g, verbose=False)
        return g

    def test_no_shared_key_is_left_at_a_module_default(self):
        """After the sync, every key both layers define agrees with config.

        This is the assertion that would have caught all three historical
        misses. It is deliberately exhaustive rather than a named list: a list
        is exactly what fails to grow when somebody adds a setting.
        """
        g = self._synced()
        divergent = []
        for key, want in vars(self.cfg).items():
            if key.startswith('_') or key in TRANSLATED:
                continue
            if key not in self.defaults:
                continue
            got = getattr(g, key)
            numeric = (isinstance(got, (int, float))
                       and isinstance(want, (int, float))
                       and not isinstance(got, bool)
                       and not isinstance(want, bool))
            same = float(got) == float(want) if numeric else got == want
            if not same:
                divergent.append("%s: globals %r != config %r" % (key, got, want))
        self.assertEqual([], divergent,
                         "config.yaml keys the sync failed to apply:\n  "
                         + "\n  ".join(divergent))

    def test_the_keys_that_were_actually_missed_are_covered(self):
        """Named regression cases, so the general test above cannot go vacuous.

        If `parser_regions` were dropped from Settings the exhaustive test would
        pass by having nothing to check; these named keys fail loudly instead.
        """
        g = self._synced()
        for key in ("detail_transfer_strength", "color_match_after_enhance",
                    "codeformer_fidelity", "parser_regions",
                    "target_conditioned_appearance",
                    "target_conditioned_appearance_strength",
                    "temporal_compositing", "temporal_quality_control"):
            self.assertTrue(hasattr(self.cfg, key), key + " left Settings")
            self.assertTrue(key in self.defaults, key + " left roop/globals.py")
            self.assertEqual(getattr(self.cfg, key), getattr(g, key),
                             key + " not carried onto globals by the sync")

    def test_translated_keys_are_not_blind_copied(self):
        """A type-mismatched key must go through its translator, not a copy.

        `no_face_action` is a dropdown LABEL in config and an int enum in
        globals; copying the string makes every comparison against the enum
        false and no no-face action fires, with nothing in the output saying so.
        """
        g = self._synced()
        self.assertIsInstance(g.no_face_action, int)
        self.assertNotIsInstance(g.no_face_action, str)
        self.assertIsInstance(g.verify_swap, bool)


class HarnessWiring(unittest.TestCase):
    """The sync only helps if the end-to-end harness asks for it."""

    def test_init_pipeline_accepts_sync_config(self):
        with open(os.path.join(HERE, "angle_bench.py"), encoding="utf-8") as fh:
            src = fh.read()
        fn = next((n for n in ast.parse(src).body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "init_pipeline"), None)
        self.assertIsNotNone(fn, "angle_bench.init_pipeline disappeared")
        names = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
        self.assertIn("sync_config", names,
                      "init_pipeline lost its sync_config parameter")

    def test_two_face_video_requests_the_sync(self):
        """The end-to-end harness must pass sync_config=True.

        Losing this keyword is silent: the run still completes, still reports
        100% swapped, and still prints an fps -- for a stack the user does not
        run. That is precisely how the defect survived.
        """
        with open(os.path.join(HERE, "two_face_video.py"), encoding="utf-8") as fh:
            src = fh.read()
        found = False
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "init_pipeline":
                for kw in node.keywords:
                    if kw.arg == "sync_config":
                        self.assertIs(getattr(kw.value, "value", None), True,
                                      "two_face_video passes sync_config, but "
                                      "not True")
                        found = True
        self.assertTrue(found, "two_face_video.py no longer calls "
                               "init_pipeline(sync_config=True)")

    def test_config_backed_toggles_can_be_turned_off_explicitly(self):
        """Silence now means "the user's config", so off must be sayable.

        Before the sync, an A/B expressed "off" by passing no flag. That is no
        longer a disable for any key whose config value is True, so each toggle
        carries an explicit negative and `baseline_controlled` uses it.
        """
        with open(os.path.join(HERE, "two_face_video.py"), encoding="utf-8") as fh:
            src = fh.read()
        for flag in ("--no-target-conditioned-appearance",
                     "--no-temporal-compositing",
                     "--no-temporal-quality-control"):
            self.assertIn(flag, src, flag + " missing from two_face_video.py")
        with open(os.path.join(HERE, "baseline_controlled.py"),
                  encoding="utf-8") as fh:
            base = fh.read()
        self.assertIn("--no-target-conditioned-appearance", base,
                      "baseline_controlled's 'off' arm still expresses off by "
                      "silence, which now inherits config.yaml instead")


if __name__ == "__main__":
    unittest.main()
