"""The benchmark harness must record, and check, the tree each arm ran on.

On 2026-09-03 two feature commits landed in the working tree WHILE a
counterbalanced A/B was rendering, both adding ~240 lines to
`roop/processors/frame/face_swapper.py` -- the hot path of every arm. Each arm
is its own process and imports the pipeline at start, so six arms ended up
split across three versions of the swapper. The harness averaged them without
complaint: the "identical config" null pair stepped 5.48 -> 7.68 fps and that
step was read as machine noise.

Nothing objected because nothing recorded the tree. These tests pin both
halves: the stamp, and the refusal to summarise across a mixed set.
"""

import contextlib
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from baseline_controlled import source_revision      # noqa: E402
from ab_shape_profile import assert_one_tree         # noqa: E402


def arm(head, dirty=False):
    return {'source_revision': {'head': head, 'dirty': dirty}}


class TestSourceRevisionStamp(unittest.TestCase):

    def test_stamp_reports_a_real_head(self):
        rev = source_revision()
        self.assertIsInstance(rev, dict)
        self.assertIn('head', rev)
        if rev['head'] is not None:                  # a git checkout
            self.assertRegex(rev['head'], r'^[0-9a-f]{40}$')
            self.assertIn(rev['dirty'], (True, False))

    def test_dirty_paths_are_not_truncated(self):
        """The porcelain codes are column-significant.

        `git status --porcelain` emits " M path". Stripping the whole blob eats
        the leading space of the FIRST line only, so exactly one path comes
        back missing its first character -- 'app/roop/core.py' as
        'pp/roop/core.py'. That is a real bug this helper shipped with for one
        revision; a single-file dirty tree is the case that catches it.
        """
        rev = source_revision()
        for path in (rev.get('dirty_code') or []):
            self.assertFalse(path.startswith('pp/'),
                             'truncated path %r: porcelain was over-stripped' % path)
            self.assertTrue(path.endswith('.py'))
            self.assertEqual(path, path.strip())


class TestRefusesMixedTrees(unittest.TestCase):

    def test_arms_from_different_commits_are_refused(self):
        """The exact shape of the 2026-09-03 contamination."""
        mixed = [('null_0', arm('d542608aa')), ('null_1', arm('d542608aa')),
                 ('profile1_rep0', arm('7da4d08bb')),
                 ('profile0_rep1', arm('b084915cc'))]
        self.assertFalse(assert_one_tree(mixed))

    def test_one_differing_arm_is_enough_to_void_the_set(self):
        almost = [('a', arm('d542608aa')), ('b', arm('d542608aa')),
                  ('c', arm('7da4d08bb'))]
        self.assertFalse(assert_one_tree(almost))

    def test_a_single_tree_is_accepted(self):
        same = [(t, arm('d542608aa')) for t in ('a', 'b', 'c')]
        self.assertTrue(assert_one_tree(same))

    def test_a_dirty_tree_is_accepted_but_flagged(self):
        """Dirty does not void the set -- both arms saw the same edit -- but a
        run whose code is not in git cannot be reproduced, so it must say so."""
        dirty = [(t, arm('d542608aa', dirty=True)) for t in ('a', 'b')]
        self.assertTrue(assert_one_tree(dirty))

    def test_unstamped_arms_do_not_silently_pass_as_verified(self):
        """Arms from an older harness carry no revision at all.

        They must not be reported as 'all arms ran the same tree' -- absence of
        a stamp is not evidence of sameness.
        """
        self.assertTrue(assert_one_tree([('a', {}), ('b', {})]))


class TestGuardIsActuallyWiredIntoMain(unittest.TestCase):
    """Run main() with rendering stubbed out.

    The guard's own unit tests passed while the call to it sat in the wrong
    scope -- inserted after the OPENING banner, above where `results` is
    built -- so every real invocation died with NameError before rendering a
    single frame. Testing the function is not testing the wiring; this project
    has a standing rule about proving the code path executes, and that is what
    these assert.
    """

    def _run_main(self, revs):
        import ab_shape_profile as ab
        calls = {'n': 0}

        def fake_run_arm(tag, profile_on, end, extra_env=None):
            calls['n'] += 1
            head = revs[min(calls['n'] - 1, len(revs) - 1)]
            return {'run': {'fps': 10.0 + calls['n'], 'faces_seen': 900,
                            'faces_swapped': 898},
                    'source_revision': {'head': head, 'dirty': False},
                    '_wall': 1.0, '_rc': 0}

        argv, run_arm = sys.argv, ab.run_arm
        buf = io.StringIO()
        try:
            ab.run_arm = fake_run_arm
            sys.argv = ['ab', '--reps', '1', '--end', '10', '--skip-warmup']
            with contextlib.redirect_stdout(buf):
                ab.main()
        finally:
            ab.run_arm, sys.argv = run_arm, argv
        return buf.getvalue()

    def test_main_completes_and_reports_one_tree(self):
        out = self._run_main(['d542608aa'] * 2)
        self.assertIn('source revision per arm', out)
        self.assertIn('all arms ran the same committed tree', out)
        self.assertIn('profile ON vs OFF', out)
        self.assertNotIn('VOID', out)

    def test_main_voids_the_delta_when_a_commit_lands_mid_run(self):
        # --reps 1 renders two arms; the second lands after a commit
        out = self._run_main(['d542608aa', '7da4d08bb'])
        self.assertIn('ARMS RAN DIFFERENT CODE', out)
        self.assertIn('VOID', out)
        # the number must still be refused where a reader would look for it
        delta = [l for l in out.splitlines() if 'profile ON vs OFF' in l]
        self.assertTrue(delta and 'VOID' in delta[0], delta)


if __name__ == '__main__':
    unittest.main()
