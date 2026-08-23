"""Every enhancer name a bench hands to `init_pipeline` must be one core matches.

`roop.core.get_processing_plugins` selects the enhancer by comparing
`selected_enhancer` against a chain of exact strings. A name outside that set
falls off the end of the chain and NO enhancer is added — no exception, no
warning, and a perfectly normal-looking render.

That is not hypothetical. Three harnesses in this directory named the baseline
arm `'codeformer'` or `"CodeFormer"`, and core only matches `'Codeformer'` and
`'Codeformer (fp16)'`. So their "CodeFormer" arm ran with no enhancer at all,
and every "x faster than CodeFormer" figure they produced compared UltraMax
against nothing. Found 2026-08-23, when the same pair was measured properly and
came out 1.13x rather than the 2.5x on the banner.

The valid set is parsed out of core.py rather than duplicated, so adding an
enhancer there cannot leave this guard stale.
"""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)


def valid_enhancer_names():
    with open(os.path.join(APP, 'roop', 'core.py'), encoding='utf-8') as f:
        src = f.read()
    names = set(re.findall(
        r"roop\.globals\.selected_enhancer\s*==\s*'([^']+)'", src))
    names.add('None')
    return names


def enhancer_args_in(path):
    """The 3rd positional argument of every `init_pipeline(...)` literal call,
    plus every literal passed as `enhancer_name=`/`"..."` to a run helper."""
    with open(path, encoding='utf-8') as f:
        src = f.read()
    out = set()
    for m in re.finditer(r"init_pipeline\(\s*'[^']*'\s*,\s*[^,]+,\s*'([^']*)'", src):
        out.add(m.group(1))
    for m in re.finditer(r"run_pipeline_for_clip\([^)]*?,\s*\"([^\"]+)\"\s*,", src):
        out.add(m.group(1))
    return out


class TestEnhancerNames(unittest.TestCase):
    def test_core_exposes_a_name_set(self):
        names = valid_enhancer_names()
        self.assertIn('Codeformer (fp16)', names)
        self.assertIn('UltraMax', names)
        self.assertGreater(len(names), 5)

    def test_every_bench_names_a_real_enhancer(self):
        valid = valid_enhancer_names()
        bad = []
        for fn in sorted(os.listdir(HERE)):
            if not fn.endswith('.py'):
                continue
            for name in enhancer_args_in(os.path.join(HERE, fn)):
                # A variable, not a literal, is out of scope for a source scan.
                if name and name not in valid:
                    bad.append(f"{fn}: {name!r}")
        self.assertEqual(bad, [], "these render with NO enhancer: " + "; ".join(bad))

    def test_the_guard_actually_catches_the_bug_it_was_written_for(self):
        valid = valid_enhancer_names()
        self.assertNotIn('codeformer', valid)   # the lowercase spelling that shipped
        self.assertNotIn('CodeFormer', valid)   # and the camel-case one


if __name__ == '__main__':
    unittest.main()
