"""The comparison bench must apply the same ROOP_* env the app does.

A bench that sets a different set of performance flags than `run.py` is
measuring a different machine. That is not hypothetical here:
`tests/two_face_video.py` shipped without `_apply_perf_env` at all, so every fps
number it printed before 2026-08-16 was taken at 4 threads with no pooling.

This is a SOURCE-level guard, deliberately. Importing `run.py` to compare the
real dicts is not an option — it parses `sys.argv` at module scope, so it dies
under any test runner's arguments.
"""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)


def _tristate_pairs(path):
    """The (ENV_VAR, config_key) pairs inside the tri-state `for var, key in (...)`
    loop of an `_apply_perf_env`."""
    with open(path, encoding='utf-8') as f:
        src = f.read()
    m = re.search(r"for var, key in \((.*?)\):", src, re.S)
    assert m, f"no tri-state loop found in {path}"
    return set(re.findall(r"\('([A-Z0-9_]+)',\s*'([a-z0-9_]+)'\)", m.group(1)))


def _set_calls(path):
    """The `_set('ROOP_X', cfg.get('key'))` pairs."""
    with open(path, encoding='utf-8') as f:
        src = f.read()
    return set(re.findall(r"_set\('([A-Z0-9_]+)',\s*cfg\.get\('([a-z0-9_]+)'\)\)", src))


class TestBenchPerfEnvMatchesApp(unittest.TestCase):
    RUN = os.path.join(APP, 'run.py')
    BENCH = os.path.join(APP, 'tests', 'compare_enhancers_video.py')

    def test_tristate_flags_match(self):
        app, bench = _tristate_pairs(self.RUN), _tristate_pairs(self.BENCH)
        self.assertTrue(app, "parsed nothing out of run.py — the guard is dead")
        self.assertEqual(app, bench,
                         f"run.py only: {sorted(app - bench)}; "
                         f"bench only: {sorted(bench - app)}")

    def test_direct_set_flags_match(self):
        app, bench = _set_calls(self.RUN), _set_calls(self.BENCH)
        self.assertTrue(app, "parsed nothing out of run.py — the guard is dead")
        self.assertEqual(app, bench,
                         f"run.py only: {sorted(app - bench)}; "
                         f"bench only: {sorted(bench - app)}")

    def test_guard_fails_when_a_key_is_dropped(self):
        """The guard has to actually fail — verified, not assumed."""
        pairs = _tristate_pairs(self.BENCH)
        self.assertGreater(len(pairs), 1)
        trimmed = set(list(pairs)[1:])
        self.assertNotEqual(_tristate_pairs(self.RUN), trimmed)


if __name__ == '__main__':
    unittest.main()
