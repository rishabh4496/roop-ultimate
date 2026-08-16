"""Where the pipeline gets a head's yaw and pitch from, and what breaks when it lies.

Two consumers act on the target face's angles, and both are invisible failures
when the number is wrong — nothing errors, the render just comes out doubled:

  * `apply_mouth_area` and `apply_eyes_area` composite the PLATE's own mouth and
    eyes back over the swap, and fade that out between 25 and 38 degrees. Past
    that the plate's mouth sits on the side of a turned head and the plate's eye
    ellipses sit over the nose bridge, so pasting them lays the original
    features on top of the swapped ones — the doubled nose/mouth people report
    on profiles and steep tilts.
  * `nonfrontal_score` adds a pitch term, which decides which of two different
    mask derivations runs.

The angles used to come from an EPnP fit to `landmark_3d_68` mapped into crop
space, and it was wrong in both directions at once:

  * that model is only loaded when one of four off-by-default features asks for
    it (ProcessMgr.initialize), so by DEFAULT the block never ran and both
    angles stayed at their 0.0 initialiser. Not "unknown" — 0.0 reads as
    "perfectly frontal", so the fade never engaged and both restores composited
    at full strength onto 90-degree profiles;
  * with the model loaded it reports yaw ~= 180 - true_yaw (-178.5 degrees for a
    dead-frontal face). max(|yaw|, |pitch|) is then always past the end of the
    fade, so both restores were switched fully off instead.

A feature that is 100% on when it should be off and 0% on when it should be on,
depending on an unrelated toggle, is worth pinning down mechanically.

These tests do not exercise ProcessMgr (it needs models and a GPU). They pin the
SOLVER the fix routes through, and read the wiring out of the source to check the
fix is actually the one in the call path.
"""

import math
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.face_3d_recon import _REF3D_68                          # noqa: E402
from roop.face_util import solve_pose_5pt                         # noqa: E402
from roop.nonfrontal import nonfrontal_score                      # noqa: E402

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESS_MGR = os.path.join(APP, 'roop', 'ProcessMgr.py')

# The fade both restores apply, lifted from procmgr_masking so a change there
# fails here rather than silently retuning what these tests assert.
FADE_START, FADE_END = 25.0, 38.0


def fade(angle):
    if angle <= FADE_START:
        return 1.0
    return max(0.0, min(1.0, (FADE_END - angle) / (FADE_END - FADE_START)))


# _REF3D_68 is a unit-scale model (extent ~1.6 x 2.2), so `scale` is roughly the
# head height in pixels. 150 puts the eyes ~86 px apart, an ordinary mid-shot
# face. It matters for anything measuring NOISE: the first draft of these tests
# used scale=3, which is a face 1.7 px wide, and 1 px of jitter there is a 54 deg
# pose error. Noiseless geometry is scale-free and unaffected.
FACE_SCALE = 150.0


def project_68(yaw, pitch, roll=0.0, scale=FACE_SCALE, cx=400.0, cy=300.0):
    """The reference head's 68 landmarks at a pose, in image coordinates."""
    y, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    pts = np.asarray(_REF3D_68, dtype=np.float64) @ (rx @ ry).T
    xy = np.column_stack([pts[:, 0], -pts[:, 1]])
    c, s = math.cos(r), math.sin(r)
    xy = xy @ np.array([[c, -s], [s, c]]).T
    xy = xy - xy.mean(axis=0)
    return (xy * scale + np.array([cx, cy])).astype(np.float32)


def kps_from_68(lm):
    """The 5 arcface keypoints, taken off the 68 the same way the pipeline does."""
    return np.array([lm[36:42].mean(axis=0), lm[42:48].mean(axis=0),
                     lm[30], lm[48], lm[54]], dtype=np.float32)


POSES = [(0, 0), (10, 5), (25, 0), (0, 25), (35, 0), (0, -35),
         (45, 20), (60, 0), (75, -20), (85, 0), (-60, 15)]


class TestTheSolverTheFixUses(unittest.TestCase):
    """solve_pose_5pt has to be right for the fix to be worth anything, and it
    has to be right from the 5 keypoints ALONE — that is the whole reason it can
    replace a path gated on an optional model."""

    def test_recovers_the_pose_it_was_given(self):
        worst = 0.0
        for yaw in range(-90, 91, 5):
            for pitch in range(-40, 41, 5):
                got = solve_pose_5pt(kps_from_68(project_68(yaw, pitch)))
                self.assertIsNotNone(got, f"no solution at yaw={yaw} pitch={pitch}")
                worst = max(worst, abs(got[0] - yaw), abs(got[1] - pitch))
        self.assertLess(worst, 0.05, f"worst pose error {worst:.4f} deg")

    def test_roll_does_not_leak_into_yaw_or_pitch(self):
        """An in-plane tilt is not a turn. The alignment represents roll exactly,
        so a rolled frontal face must not read as angled and fade the restores
        out — the mask router had this exact bug against a different metric."""
        for roll in (-30, -15, 15, 30):
            for yaw, pitch in ((0, 0), (30, 10)):
                got = solve_pose_5pt(kps_from_68(project_68(yaw, pitch, roll)))
                self.assertAlmostEqual(got[0], yaw, delta=0.05, msg=f"roll={roll}")
                self.assertAlmostEqual(got[1], pitch, delta=0.05, msg=f"roll={roll}")

    def test_survives_keypoint_noise_without_swinging_the_fade(self):
        """1 px of detector jitter must not move the restore strength much on an
        ordinary face. Making the fade live also makes it a thing that can
        chatter — it used to be pinned at a constant 1.0 or 0.0 — so the size of
        that chatter is worth a number rather than a hope."""
        rng = np.random.default_rng(4)
        for yaw, pitch in ((30, 0), (0, 30), (33, 12)):
            k = kps_from_68(project_68(yaw, pitch))
            vals = []
            for _ in range(400):
                got = solve_pose_5pt(k + rng.normal(0, 1.0, k.shape))
                vals.append(fade(max(abs(got[0]), abs(got[1]))))
            self.assertLess(float(np.std(vals)), 0.12,
                            f"restore strength sd {np.std(vals):.3f} at {yaw}/{pitch}")

    def test_the_chatter_shrinks_as_the_face_gets_bigger(self):
        """Recorded because it decides where the fade is worth trusting, and
        because it is the opposite way round from the reported symptoms.

        Only a pose sitting INSIDE the 25-38 deg band can chatter at all, and the
        pose error scales as 1/face size. Measured restore-strength sd at yaw 31
        under 1 px of keypoint noise: 0.22 at 34 px interocular, 0.09 at 86 px,
        0.03 at 229 px. So the wobble belongs to distant faces, where the
        restores are a few pixels wide, and close-ups — which is what the flicker
        reports are about — are the quiet end.
        """
        rng = np.random.default_rng(4)
        last = None
        for scale in (60.0, 150.0, 400.0):
            k = kps_from_68(project_68(31, 0, scale=scale))
            vals = [fade(max(abs(g[0]), abs(g[1])))
                    for g in (solve_pose_5pt(k + rng.normal(0, 1.0, k.shape))
                              for _ in range(400))]
            sd = float(np.std(vals))
            if last is not None:
                self.assertLess(sd, last, 'a bigger face should be steadier')
            last = sd
        self.assertLess(last, 0.05, f'close-up chatter {last:.3f} is too high')


class TestTheFadeNowTracksTheHead(unittest.TestCase):
    def test_full_strength_frontal_and_off_on_profiles(self):
        for yaw, pitch in POSES:
            got = solve_pose_5pt(kps_from_68(project_68(yaw, pitch)))
            self.assertAlmostEqual(fade(max(abs(got[0]), abs(got[1]))),
                                   fade(max(abs(yaw), abs(pitch))),
                                   delta=0.01, msg=f"yaw={yaw} pitch={pitch}")

    def test_neither_old_default_reproduces_it(self):
        """Both of the values the old code could produce are constants, so they
        cannot track anything. Stated as a test so a revert cannot pass."""
        turned = [(45, 20), (60, 0), (85, 0)]
        for yaw, pitch in turned:
            self.assertEqual(fade(max(abs(yaw), abs(pitch))), 0.0)
        # 0.0 deg — what the default configuration produced — pastes at 100%.
        self.assertEqual(fade(0.0), 1.0)
        # ~180 deg — what the 68-point path produced — pastes at 0% everywhere,
        # including on the frontal faces the restores exist for.
        self.assertEqual(fade(178.5), 0.0)


class TestMaskRoutingIsNotDisturbed(unittest.TestCase):
    """The same angles feed nonfrontal_score's pitch term. Handing it a real
    pitch instead of a fixed 0.0 is a correction, but the routing decision
    should barely move — the keypoint terms already dominate. A large shift
    would mean this fix quietly re-routed everyone's masks as a side effect."""

    def test_routing_verdict_barely_moves(self):
        flips = total = 0
        for yaw in range(-90, 91, 10):
            for pitch in range(-40, 41, 10):
                for roll in (-20, 0, 20):
                    k = kps_from_68(project_68(yaw, pitch, roll))
                    pose = solve_pose_5pt(k)
                    was = nonfrontal_score(k, 0.0) > 1.0
                    now = nonfrontal_score(k, pose[1] if pose else 0.0) > 1.0
                    total += 1
                    flips += (was != now)
        self.assertLess(flips / total, 0.05,
                        f"routing changed on {flips}/{total} poses")

    def test_a_bogus_180_degree_pitch_would_have_routed_everything_non_frontal(self):
        """Why the old value could not simply be left in place."""
        k = kps_from_68(project_68(0, 0))
        self.assertLessEqual(nonfrontal_score(k, 0.0), 1.0)
        self.assertGreater(nonfrontal_score(k, 178.5), 1.0)


class TestTheWiringInProcessMgr(unittest.TestCase):
    """The solver being right is useless if process_face still reads the old
    value. These check the call path in the source, because standing the real
    thing up needs models and a GPU."""

    def setUp(self):
        with open(PROCESS_MGR, encoding='utf-8') as fh:
            self.src = fh.read()
        # Comments in this file discuss the very patterns being searched for.
        self.code = re.sub(r'#[^\n]*', '', self.src)

    def test_target_angles_come_from_the_five_point_solve(self):
        self.assertRegex(
            self.code,
            r'tgt_yaw_deg,\s*tgt_pitch_deg\s*=\s*float\(_pose5\[0\]\),\s*float\(_pose5\[1\]\)',
            'process_face no longer takes its target angles from solve_pose_5pt')
        # Bounded to process_face's OWN body rather than a fixed 4000-character
        # prefix of everything after it. The prefix was a proxy for "somewhere
        # in this function", and it broke twice on additions near the top that
        # had nothing to do with the pose solve — a guard that fails for where
        # code sits rather than what it does. Cutting at the next method keeps
        # the same property and stops it drifting.
        body = self.code.split('def process_face')[1]
        body = re.split(r'\n    (?:@|def )', body)[0]
        self.assertIn('solve_pose_5pt', body)

    def test_the_epnp_angles_no_longer_reach_the_restores(self):
        """tgt_* must not be assigned from decompose_yaw_pitch any more. The
        source bank keeps that convention deliberately — under its own names."""
        block = self.code.split('def process_face')[1]
        self.assertNotRegex(
            block, r'tgt_yaw_deg\s*=\s*_math\.degrees',
            'the EPnP angles are being written back into tgt_yaw_deg')
        self.assertRegex(block, r'bank_yaw_deg\s*=\s*_math\.degrees',
                         'the source bank lost its own pose variables')

    def test_the_source_bank_comparison_stays_in_one_convention(self):
        """Both sides of the bank's distance must use the EPnP angles: the error
        cancels in the difference, and correcting only one side would break a
        matcher that currently works."""
        m = re.search(r'dist\s*=\s*\(([^)]*)\)\s*\*\*\s*2\s*\+\s*\(([^)]*)\)\s*\*\*\s*2',
                      self.code)
        self.assertIsNotNone(m, 'source-bank distance not found')
        self.assertIn('bank_yaw_deg', m.group(1))
        self.assertIn('bank_pitch_deg', m.group(2))

    def test_the_restores_are_handed_the_jaw_aware_angles(self):
        """And specifically NOT the jaw-blind pair.

        The restore fades ask "how far is this head turned" — past ~25 deg the
        plate's mouth and eyes no longer sit where the swap's do and pasting
        them doubles the feature. `solve_pose_5pt` cannot answer that question,
        because two of its five points are the MOUTH CORNERS: on a dead-frontal
        head it reads a dropped jaw as -28 deg of pitch. Fed that, the fades

          * cut the restore to 0.765 on a frontal face at full mouth opening,
            varying continuously through a sentence — including the EYE restore,
            which has nothing to do with the jaw;
          * and move the WRONG WAY where the fade is meant to act: at a true yaw
            of 35 deg the phantom pitch pulls the solved yaw down to 33, so
            opening the mouth takes the fade from 0.231 to 0.388 and the restore
            gets STRONGER on the turned head.

        So they take `_head_angles()`, which is the jaw-aware solve with a
        fallback to the jaw-blind pair when it declines.
        """
        for call in ('apply_mouth_area', 'apply_eyes_area'):
            idx = self.code.find(call + '(')
            seen = 0
            while idx != -1:
                if 'def ' + call not in self.code[max(0, idx - 12):idx]:
                    chunk = self.code[idx:idx + 600]
                    if 'yaw=' in chunk:
                        seen += 1
                        self.assertIn('yaw=_hy', chunk, f'{call} lost the head yaw')
                        self.assertIn('pitch=_hp', chunk, f'{call} lost the head pitch')
                        self.assertNotIn('yaw=tgt_yaw_deg', chunk,
                                         f'{call} is back on the jaw-blind angles')
                idx = self.code.find(call + '(', idx + 1)
            self.assertTrue(seen, f'no call site found for {call}')

    def test_the_jaw_solve_runs_at_most_once_per_face(self):
        """It costs 84us against solve_pose_5pt's 11 — nothing beside a masking
        stage of ~42ms, but there are four consumers in process_face and paying
        it four times would be careless. One memoised helper, one call."""
        block = self.code.split('def process_face')[1]
        self.assertEqual(block.count('solve_pose_jaw_5pt('), 1,
                         'process_face solves the jaw pose more than once')
        self.assertIn('nonlocal _jaw_pose', block,
                      'the jaw solve is no longer memoised across consumers')


if __name__ == '__main__':
    unittest.main()
