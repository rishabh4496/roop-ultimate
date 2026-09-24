"""Auto-tune decisions (roop/benchmark/autotune.py), driven by a fake GPU.

The executor that runs real renders is exercised live; everything that DECIDES
is here: which arms exist, the order they run in, which arms are disqualified
and why, the counterbalanced verdict, and the NVENC preset rule.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.benchmark import autotune as at   # noqa: E402

A = at.Arm


def run(arm, fps, swaps=100, eff_batch=None, eff_provider=None, error=None, phase='screen'):
    return at.ArmRun(arm=arm, phase=phase, frames=100, fps=fps, swaps=swaps,
                     effective_batch=arm.batch if eff_batch is None else eff_batch,
                     effective_provider=eff_provider or arm.provider, error=error)


class Arms(unittest.TestCase):

    def test_batches_above_the_worker_count_are_not_arms(self):
        arms = at.candidate_arms(['cuda'], threads=3)
        self.assertEqual([a.batch for a in arms], [1, 2])

    def test_every_provider_gets_every_reachable_batch(self):
        arms = at.candidate_arms(['cuda', 'tensorrt'], threads=10)
        self.assertEqual(len(arms), 8)

    def test_screening_is_counterbalanced_and_grouped_by_provider(self):
        arms = at.candidate_arms(['cuda', 'tensorrt'], threads=10)
        order = at.screening_order(arms)
        self.assertEqual(order, order[:len(arms)] + list(reversed(order[:len(arms)])))
        providers = [a.provider for a in order]
        switches = sum(1 for x, y in zip(providers, providers[1:]) if x != y)
        self.assertEqual(switches, 2, 'a provider switch rebuilds every model')


class Honesty(unittest.TestCase):

    def test_a_clamped_batch_is_not_its_label(self):
        self.assertFalse(run(A('cuda', 8), 10, eff_batch=4).honest)

    def test_a_provider_fallback_is_not_its_label(self):
        self.assertFalse(run(A('tensorrt', 4), 10, eff_provider='cuda').honest)

    def test_auto_batch_is_never_compared(self):
        self.assertTrue(run(A('cuda', 0), 10, eff_batch=10).honest)

    def test_an_error_or_zero_fps_is_not_a_measurement(self):
        self.assertFalse(run(A('cuda', 2), 0).honest)
        self.assertFalse(run(A('cuda', 2), 10, error='boom').honest)


class Finalists(unittest.TestCase):
    base = A('tensorrt', 0)

    def summary(self, rows):
        return at.summarize([run(a, f, s) for a, f, s in rows])

    def test_faster_by_swapping_less_is_disqualified(self):
        s = self.summary([(self.base, 10, 100), (A('cuda', 8), 20, 90)])
        finals, why = at.pick_finalists(s, self.base)
        self.assertEqual(finals, [])
        self.assertIn('doing less', why['cuda/b8'])

    def test_slower_arms_are_not_finalists(self):
        s = self.summary([(self.base, 10, 100), (A('cuda', 2), 9, 100)])
        self.assertEqual(at.pick_finalists(s, self.base)[0], [])

    def test_top_two_by_screening_fps(self):
        rows = [(self.base, 10, 100), (A('tensorrt', 2), 11, 100),
                (A('tensorrt', 4), 13, 100), (A('tensorrt', 8), 12, 100)]
        finals, why = at.pick_finalists(self.summary(rows), self.base)
        self.assertEqual([a.key for a in finals], ['tensorrt/b4', 'tensorrt/b8'])
        self.assertIn('top 2', why['tensorrt/b2'])

    def test_a_baseline_that_swapped_nothing_stops_the_run(self):
        s = self.summary([(self.base, 10, 0), (A('cuda', 2), 20, 0)])
        finals, why = at.pick_finalists(s, self.base)
        self.assertEqual(finals, [])
        self.assertIn('swapped no faces', why[self.base.key])


class Verdict(unittest.TestCase):
    b, c = A('cuda', 0), A('cuda', 4)

    def v(self, b1, c1, c2, b2, cs=100):
        return at.verdict([run(self.b, b1), run(self.b, b2)],
                          [run(self.c, c1, cs), run(self.c, c2, cs)])

    def test_wins_both_pairs_above_noise(self):
        self.assertTrue(self.v(10.0, 11.0, 11.1, 10.1)['accepted'])

    def test_a_lost_pair_rejects(self):
        self.assertFalse(self.v(10.0, 11.0, 9.9, 10.1)['accepted'])

    def test_inside_the_baselines_own_spread_rejects(self):
        # Baseline replicates 10 and 12 disagree by ~18%; +9% cannot be seen.
        out = self.v(10.0, 12.1, 12.1, 12.0)
        self.assertFalse(out['accepted'])
        self.assertIn('noise', out['reason'])

    def test_below_min_effect_rejects(self):
        self.assertFalse(self.v(10.0, 10.2, 10.2, 10.0)['accepted'])

    def test_fewer_swaps_rejects(self):
        self.assertFalse(self.v(10.0, 12.0, 12.0, 10.0, cs=90)['accepted'])


class NvencPreset(unittest.TestCase):
    rows = [{'preset': 'p%d' % i, 'fps': f}
            for i, f in zip(range(1, 8), (600, 520, 450, 380, 300, 150, 60))]

    def test_best_quality_that_keeps_up_with_headroom(self):
        # Render at 13 fps needs 26 fps of encode; p7 (60) already clears it.
        self.assertEqual(at.pick_nvenc_preset(self.rows, 13.0)['preset'], 'p7')

    def test_a_fast_render_pushes_the_preset_down(self):
        self.assertEqual(at.pick_nvenc_preset(self.rows, 100.0)['preset'], 'p5')

    def test_nothing_keeps_up_means_the_fastest(self):
        self.assertEqual(at.pick_nvenc_preset(self.rows, 1000.0)['preset'], 'p1')

    def test_failed_presets_are_ignored(self):
        rows = [{'preset': 'p7', 'fps': None}, {'preset': 'p5', 'fps': 300}]
        self.assertEqual(at.pick_nvenc_preset(rows, 10)['preset'], 'p5')


class Session(unittest.TestCase):
    """The whole protocol against a fake GPU with a known best arm."""

    def fake(self, truth, swaps=None, clamp=None):
        calls = []

        def measure(arm, frames, phase):
            calls.append((arm.key, frames, phase))
            eff = clamp.get(arm.key, arm.batch) if clamp else arm.batch
            return run(arm, truth[arm.key], (swaps or {}).get(arm.key, 100),
                       eff_batch=eff, phase=phase)
        return measure, calls

    def test_a_real_win_is_confirmed_and_named(self):
        truth = {'cuda/bauto': 10, 'cuda/b1': 9, 'cuda/b2': 10.5, 'cuda/b4': 13, 'cuda/b8': 12}
        measure, calls = self.fake(truth)
        s = at.AutoTuneSession(measure, None, ['cuda'], threads=10, baseline=A('cuda', 0))
        out = s.run()
        self.assertEqual(out['status'], 'complete')
        self.assertEqual(out['winner']['arm'], 'cuda/b4')
        self.assertTrue(out['winner']['changed'])
        screen = [c for c in calls if c[2] == 'screen']
        confirm = [c for c in calls if c[2] == 'confirm']
        self.assertTrue(all(f == 100 for _, f, _ in screen))
        self.assertTrue(all(f == 600 for _, f, _ in confirm))
        self.assertEqual(len(confirm), 8, 'two finalists x A/B/B/A')

    def test_no_resolvable_win_keeps_the_baseline(self):
        truth = {'cuda/bauto': 10, 'cuda/b1': 10.1, 'cuda/b2': 10.2, 'cuda/b4': 10.1, 'cuda/b8': 10.1}
        measure, _ = self.fake(truth)
        out = at.AutoTuneSession(measure, None, ['cuda'], 10, A('cuda', 0)).run()
        self.assertFalse(out['winner']['changed'])

    def test_a_clamped_arm_cannot_win_under_its_label(self):
        truth = {'cuda/bauto': 10, 'cuda/b1': 9, 'cuda/b2': 9, 'cuda/b4': 9, 'cuda/b8': 20}
        measure, _ = self.fake(truth, clamp={'cuda/b8': 4})
        out = at.AutoTuneSession(measure, None, ['cuda'], 10, A('cuda', 0)).run()
        self.assertFalse(out['winner']['changed'])
        row = next(r for r in out['screen'] if r['arm'] == 'cuda/b8')
        self.assertIn('labelled', row['excluded'])

    def test_short_target_caps_confirmation_and_says_so(self):
        truth = {'cuda/bauto': 10, 'cuda/b1': 9, 'cuda/b2': 9, 'cuda/b4': 9, 'cuda/b8': 9}
        measure, calls = self.fake(truth)
        out = at.AutoTuneSession(measure, None, ['cuda'], 10, A('cuda', 0),
                                 available_frames=240).run()
        self.assertEqual(out['confirm_frames'], 240)
        self.assertFalse(out['confirm_meets_600_rule'])

    def test_encoder_runs_against_the_confirmed_rate(self):
        truth = {'cuda/bauto': 13, 'cuda/b1': 9, 'cuda/b2': 9, 'cuda/b4': 9, 'cuda/b8': 9}
        measure, _ = self.fake(truth)
        enc = lambda presets: {'rows': NvencPreset.rows, 'applies': True}
        out = at.AutoTuneSession(measure, enc, ['cuda'], 10, A('cuda', 0)).run()
        self.assertEqual(out['encoder']['picked'], 'p7')

    def test_cancel_stops_between_arms(self):
        truth = {'cuda/bauto': 10, 'cuda/b1': 9, 'cuda/b2': 9, 'cuda/b4': 9, 'cuda/b8': 9}
        holder = {}

        def measure(arm, frames, phase):
            holder['s'].cancel()
            return run(arm, truth[arm.key])
        s = at.AutoTuneSession(measure, None, ['cuda'], 10, A('cuda', 0))
        holder['s'] = s
        out = s.run()
        self.assertEqual(out['status'], 'cancelled')
        self.assertEqual(len(out['runs']), 1)

    def test_a_crashing_arm_is_recorded_not_fatal(self):
        truth = {'cuda/bauto': 10, 'cuda/b1': 9, 'cuda/b2': 9, 'cuda/b4': 9, 'cuda/b8': 9}

        def measure(arm, frames, phase):
            if arm.batch == 8:
                raise RuntimeError('CUDA out of memory')
            return run(arm, truth[arm.key])
        out = at.AutoTuneSession(measure, None, ['cuda'], 10, A('cuda', 0)).run()
        self.assertEqual(out['status'], 'complete')
        self.assertTrue(any('out of memory' in (r['error'] or '') for r in out['runs']))


if __name__ == '__main__':
    unittest.main()
