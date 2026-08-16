"""face_overlap — who owns a pixel when two swapped faces meet.

Each face is pasted under a matte built from its own geometry alone: a hull
that has been dilated, given a forehead extension reaching 60 % of the face
height past the brow, and feathered outward. None of that knows there is
somebody standing next to it, so on interacting faces the mattes overlap and
each face paints a band of its own swap over the other. Which band survives is
decided by whoever happens to be pasted last.

Properties asserted here:

  * SEPARATE FACES COST NOTHING. Faces that are not near each other return no
    regions at all, so the frames that make up the overwhelming majority of any
    render do not pay for this and cannot be changed by it.

  * NOTHING IS LEFT UNPAINTED. The faces composite in sequence, so the test is
    not that the two fields sum to 1 — it is that the SEQUENTIAL result covers
    the contested area completely. Two mattes at 0.5 sum to 1 and still leave
    a quarter of the plate showing down the join.

  * THE BOUNDARY IS BETWEEN THEM. Deep inside a face, that face owns
    everything; the hand-over happens in a narrow band around the midline.

  * IT DOES NOT DEPEND ON ORDER. The same pair of faces produces the same
    boundary whichever order they arrive in — which is the whole point, since
    match order changes frame to frame and used to decide the join.

  * NEARER WINS. A face twice the size on screen is in front, and takes the
    contested pixels rather than being painted over by the one behind it.

The compositing tests at the bottom cover the other half: that ProcessMgr
actually pastes far-to-near, and that it reads every face from the untouched
plate rather than from the half-finished composite.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_overlap                           # noqa: E402
from roop.face_overlap import build_regions             # noqa: E402
from roop.ProcessMgr import ProcessMgr                  # noqa: E402

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHAPE = (400, 800, 3)


class _Face:
    """A face with only the geometry face_overlap reads."""

    def __init__(self, cx, cy, size, landmarks=True):
        self.bbox = np.array([cx - size / 2, cy - size / 2,
                              cx + size / 2, cy + size / 2], dtype=np.float32)
        self.kps = np.array([[cx - size * 0.2, cy - size * 0.1],
                             [cx + size * 0.2, cy - size * 0.1],
                             [cx, cy + size * 0.05],
                             [cx - size * 0.15, cy + size * 0.25],
                             [cx + size * 0.15, cy + size * 0.25]],
                            dtype=np.float32)
        self.landmark_2d_106 = _ring(cx, cy, size) if landmarks else None


def _ring(cx, cy, size):
    """106 points around an oval face, ordered so indices 33-52 land on the
    brow line the way the real landmark model's do."""
    pts = np.zeros((106, 2), dtype=np.float32)
    # 0-32: jaw / silhouette, lower half of the oval.
    a = np.linspace(np.pi, 2 * np.pi, 33)
    pts[:33, 0] = cx + np.cos(a) * size * 0.5
    pts[:33, 1] = cy - np.sin(a) * size * 0.5
    # 33-52: brow line across the top of the face.
    pts[33:53, 0] = np.linspace(cx - size * 0.4, cx + size * 0.4, 20)
    pts[33:53, 1] = cy - size * 0.25
    # 53-105: interior features, well inside the hull.
    rng = np.random.RandomState(0)
    pts[53:, 0] = cx + rng.uniform(-size * 0.2, size * 0.2, 53)
    pts[53:, 1] = cy + rng.uniform(-size * 0.1, size * 0.2, 53)
    return pts


def _own_at(region, x, y):
    return float(region.own[int(y) - region.y0, int(x) - region.x0])


class Separated(unittest.TestCase):
    def test_one_face_has_nobody_to_argue_with(self):
        self.assertIsNone(build_regions([_Face(200, 200, 120)], SHAPE))

    def test_faces_across_the_frame_are_left_alone(self):
        faces = [_Face(120, 200, 120), _Face(680, 200, 120)]
        self.assertIsNone(
            build_regions(faces, SHAPE),
            'faces nowhere near each other must not pay for the demarcation, '
            'nor have their mattes touched by it')

    def test_adjacent_faces_still_get_a_boundary(self):
        """Not overlapping, but close enough that the dilation and the feather
        reach across — the case that produces the smear along the join."""
        faces = [_Face(320, 200, 120), _Face(440, 200, 120)]
        self.assertIsNotNone(build_regions(faces, SHAPE))

    def test_a_face_with_no_geometry_is_skipped(self):
        class _Blank:
            bbox = None
            landmark_2d_106 = None
        self.assertIsNone(build_regions([_Blank(), _Face(200, 200, 120)], SHAPE))

    def test_the_switch_turns_it_off(self):
        faces = [_Face(340, 200, 120), _Face(420, 200, 120)]
        self.assertIsNotNone(build_regions(faces, SHAPE))
        old = face_overlap.ENABLED
        try:
            face_overlap.ENABLED = False
            self.assertIsNone(build_regions(faces, SHAPE))
        finally:
            face_overlap.ENABLED = old


class Partition(unittest.TestCase):
    def setUp(self):
        # Face 0 is painted first (it is the one further away), face 1 on top.
        self.faces = [_Face(340, 200, 140), _Face(460, 200, 140)]
        self.regions = build_regions(self.faces, SHAPE, order=[0, 1])
        self.assertEqual(sorted(self.regions), [0, 1])

    def test_the_join_never_shows_the_plate(self):
        """The mattes are applied in sequence, so what has to reach 1 is the
        accumulated coverage — `b + (1-b)·a`, not `a + b`. Getting this wrong
        replaces the smear with a hairline of untouched footage down the join,
        which is just as visible and harder to explain."""
        a, b = self.regions[0].own, self.regions[1].own
        covered = b + (1.0 - b) * a
        contested = (a > 0) | (b > 0)
        worst = float(covered[contested].min())
        self.assertGreater(
            worst, 0.999,
            f'{(covered[contested] < 0.999).sum()} px of the contested area are '
            f'only {worst:.3f} covered — the plate shows through there')

    def test_a_face_owns_its_own_middle(self):
        self.assertEqual(_own_at(self.regions[0], 340, 200), 1.0)
        self.assertEqual(_own_at(self.regions[1], 460, 200), 1.0)

    def test_the_face_on_top_takes_none_of_the_one_below(self):
        self.assertEqual(_own_at(self.regions[1], 340, 200), 0.0)

    def test_the_face_below_stops_just_past_the_boundary(self):
        """It carries on a little way to back the ramp above it, and no further
        — the whole point is that it no longer paints across its neighbour."""
        self.assertEqual(_own_at(self.regions[0], 460, 200), 0.0)

    def test_the_hand_over_is_narrow(self):
        """A wide transition is a band where two different faces are mixed."""
        r = self.regions[1]
        row = r.own[200 - r.y0]
        soft = int(((row > 0.02) & (row < 0.98)).sum())
        self.assertLess(soft, 20, f'{soft}px hand-over across the join is a band, '
                                  'not a demarcation')
        self.assertGreater(soft, 0, 'a hard step would alias along the join')

    def test_the_boundary_sits_between_the_faces(self):
        r = self.regions[1]
        row = r.own[200 - r.y0]
        mid = int(np.argmin(np.abs(row - 0.5))) + r.x0
        self.assertGreater(mid, 340)
        self.assertLess(mid, 460)


class UnswappedBystanders(unittest.TestCase):
    """A face that is NOT being swapped still owns its own pixels.

    Ownership used to be decided among the matched faces alone, so anybody else
    in the frame was outside the competition entirely and simply took whatever
    a neighbour's dilated matte, feather and invented forehead reached over them.

    That is worse than a static artefact, because membership of the matched set
    is not stable: a borderline identity match, a gender verdict, a source that
    ran out — any of them drops a face for a frame or two. The face then
    alternates between defending itself and being smeared, which reads as both
    faces flickering, and it only happens when two people are close enough to
    interact. Which is the report.
    """

    def setUp(self):
        self.swapped = _Face(340, 200, 140)
        self.bystander = _Face(460, 200, 140)
        # Index 1 is present as a claimant but absent from `order`.
        self.regions = build_regions([self.swapped, self.bystander], SHAPE,
                                     order=[0])

    def test_the_bystander_gets_no_region_of_its_own(self):
        """Nothing is pasted for it, so there is nothing to trim."""
        self.assertIsNotNone(self.regions)
        self.assertEqual(sorted(self.regions), [0])

    def test_the_swap_does_not_reach_into_the_bystanders_face(self):
        self.assertEqual(_own_at(self.regions[0], 460, 200), 0.0)

    def test_the_swapped_face_still_owns_its_own_middle(self):
        self.assertEqual(_own_at(self.regions[0], 340, 200), 1.0)

    def test_it_stops_at_the_boundary_and_does_not_back_past_it(self):
        """A bystander counts as ALREADY painted — what is on the canvas there is
        the original footage, and it stays. Treating it as still to come would
        have the swap carry on a band past the midline to back a ramp that is
        never going to be drawn over it."""
        r = self.regions[0]
        row = r.own[200 - r.y0]
        mid = int(np.argmin(np.abs(row - 0.5))) + r.x0
        self.assertAlmostEqual(mid, 400, delta=6)

    def test_a_bystander_far_away_costs_nothing(self):
        self.assertIsNone(build_regions([_Face(340, 200, 140), _Face(700, 200, 60)],
                                        SHAPE, order=[0]))

    def test_the_pipeline_hands_them_over(self):
        """The wiring, not the geometry: `_composite_faces` has to be given the
        unmatched faces, or none of the above is reachable from a real render."""
        mgr = ProcessMgr.__new__(ProcessMgr)
        painted = []
        mgr.process_face = lambda i, f, frame, plate=None, region=None: (
            painted.append(region) or frame)
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, _Face(340, 200, 140))], plate, plate.copy(),
                             [_Face(460, 200, 140)])
        self.assertEqual(len(painted), 1)
        self.assertIsNotNone(painted[0],
                             'the swapped face got no region, so the bystander '
                             'was not competing for pixels')

    def test_a_duplicate_box_on_the_swapped_face_is_not_a_bystander(self):
        """A face cannot be its own bystander.

        Detectors do produce two boxes on one head — a partial beside a full one,
        or two engines' worth of the same face. Only one can hold the source, so
        the other arrives here as an unpainted claimant sitting on top of a
        painted one. Competed with, it wins half of that face's own pixels and
        carves the swap apart along a line through the middle of it — on only the
        frames where the duplicate was detected, so it flickers too. Worse than
        the smearing this module exists to fix, and introduced BY the fix that
        gave bystanders a claim.
        """
        face = _Face(340, 200, 140)
        dup = _Face(345, 205, 130)          # same head, slightly different box
        self.assertIsNone(build_regions([face, dup], SHAPE, order=[0]),
                          'the duplicate competed for the face it is a copy of')

    def test_a_smaller_partial_box_on_the_same_face_is_also_dropped(self):
        """Containment, not IoU: a partial detection is much smaller than the
        real one, which puts IoU low while the two are plainly the same face."""
        face = _Face(340, 200, 200)
        partial = _Face(350, 210, 70)
        self.assertIsNone(build_regions([face, partial], SHAPE, order=[0]))

    def test_the_duplicate_is_dropped_whichever_box_holds_the_source(self):
        """Which of two duplicate boxes ends up swapped is ARBITRARY — it is
        whichever one the tracker happened to associate — so the leftover is as
        often the larger box as the smaller.

        Tested in one direction only ("is the leftover inside the swapped
        face?") half of all duplicates sail straight through: containment of a
        big box in a small one is 0.12, nothing fires, and the full-size
        duplicate carves the swapped face in half exactly as if the guard were
        not there. Same pair as the test above, swapped over.
        """
        full = _Face(340, 200, 200)
        partial = _Face(350, 210, 70)
        self.assertIsNone(build_regions([partial, full], SHAPE, order=[0]),
                          'the duplicate is only caught when it is the smaller '
                          'of the two boxes')

    def test_a_smaller_separate_face_nested_in_a_bigger_ones_box_still_competes(self):
        """And this is why containment cannot simply be made symmetric.

        A smaller head standing behind and to one side of a bigger one is 95%
        contained in the bigger one's padded box while being a completely
        different person. On containment alone that is indistinguishable from a
        partial duplicate; what separates them is that duplicates are
        CONCENTRIC, and these two sit 0.71 radii apart.

        The claim is doing real work here, which is what makes dropping it the
        smear this module exists to remove: the bigger face's ownership goes to
        zero over the smaller one's face.
        """
        behind = _Face(300, 160, 110)
        near = _Face(360, 230, 190)
        regions = build_regions([behind, near], SHAPE, order=[1])
        self.assertIsNotNone(regions, 'the bystander was swallowed as a duplicate')
        self.assertEqual(float(regions[1].own.min()), 0.0,
                         'the bystander is in the competition but defends nothing')

    def test_a_genuine_neighbour_is_still_a_claimant(self):
        """The guard must not swallow the case it was built alongside — two
        people close enough to interact still each own their own pixels."""
        regions = build_regions([_Face(340, 200, 140), _Face(460, 200, 140)],
                                SHAPE, order=[0])
        self.assertIsNotNone(regions)
        self.assertEqual(_own_at(regions[0], 460, 200), 0.0)

    def test_the_call_site_passes_the_unmatched_faces(self):
        """Source guard. The bug this fixes is a missing ARGUMENT, and everything
        keeps working without it — silently, and only on frames with two people
        in them."""
        src = open(os.path.join(APP, 'roop', 'ProcessMgr.py'),
                   encoding='utf-8').read()
        call = src[src.index('temp_frame = self._composite_faces(pending'):]
        self.assertIn('_others', call[:120])


class OrderIndependence(unittest.TestCase):
    """Match order changes frame to frame — in identity-lock mode the faces are
    re-sorted by identity distance every frame. If the boundary moved with it,
    a pair standing close together would hand the join back and forth and
    flicker, which is exactly the reported symptom."""

    def test_the_same_pair_gives_the_same_boundary_either_way(self):
        """Same faces, same paint order, listed the other way round."""
        a, b = _Face(340, 200, 140), _Face(460, 200, 140)
        fwd = build_regions([a, b], SHAPE, order=[0, 1])
        rev = build_regions([b, a], SHAPE, order=[1, 0])
        np.testing.assert_array_equal(fwd[0].own, rev[1].own)
        np.testing.assert_array_equal(fwd[1].own, rev[0].own)

    def test_equal_faces_meet_halfway(self):
        a, b = _Face(340, 200, 140), _Face(460, 200, 140)
        r = build_regions([a, b], SHAPE, order=[0, 1])
        row = r[1].own[200 - r[1].y0]
        mid = int(np.argmin(np.abs(row - 0.5))) + r[1].x0
        self.assertAlmostEqual(mid, 400, delta=4,
                               msg='two identical faces side by side must meet '
                                   'halfway; anything else is a bias nobody asked for')


class JoinStability(unittest.TestCase):
    """The join has to sit still on two people who are standing still.

    The boundary is where two normalised distance fields cross, and both are
    rasterised from that frame's raw 106-point landmarks — which nothing
    smooths. So a pixel of detector jitter moves the boundary about a pixel,
    every frame. Whether anyone can see that depends entirely on how wide the
    ramp it is moving inside is, and expressing the band purely as a fraction of
    face radius left that to chance: at a 117 px radius the fraction gives a
    4.7 px band, and the worst pixel along the join changed hands by 0.383 per
    frame — a third of the way from one face to the other, on nobody moving.

    Measured swing against band width at that radius:

        band      4.7px   9.4px   14.0px   18.7px   28.1px
        swing     0.383   0.260    0.182    0.135    0.070

    Hence the pixel floor: the noise is a fixed number of pixels, so the ramp
    that has to absorb it must be too.
    """

    @staticmethod
    def _noisy_pair(rng, size=140, sep=120, jitter=1.0):
        a, b = _Face(340, 200, size), _Face(340 + sep, 200, size)
        for f in (a, b):
            f.landmark_2d_106 = (f.landmark_2d_106
                                 + rng.normal(0, jitter, f.landmark_2d_106.shape)
                                 ).astype(np.float32)
        return [a, b]

    def _swing(self, size, sep, frames=120, jitter=1.0):
        """Mean worst per-frame change in ownership along the join.

        Sampled through `region.crop` over a FIXED frame-space box, not out of
        `region.own` directly: the ROI is sized from the jittered claims, so its
        width changes frame to frame and rows taken from it are not comparable
        to each other. Reading a fixed box is also what the real consumers do.
        """
        rng = np.random.default_rng(21)
        y = 200
        box = (340, y - 1, 340 + sep, y + 1)
        prev, deltas = None, []
        for _ in range(frames):
            regions = build_regions(self._noisy_pair(rng, size, sep, jitter),
                                    SHAPE, order=[0, 1])
            if regions is None or 0 not in regions:
                continue
            own = regions[0].crop(*box)
            if own is None:
                continue
            row = own[1].copy()
            if prev is not None:
                deltas.append(float(np.abs(row - prev).max()))
            prev = row
        self.assertGreater(len(deltas), frames // 2,
                           'not enough contested frames to judge')
        return float(np.mean(deltas))

    def test_the_join_does_not_shimmer_on_a_still_pair(self):
        self.assertLess(self._swing(140, 120), 0.25,
                        'the worst pixel along the join is changing hands too '
                        'fast for two people who are not moving')

    def test_the_band_does_not_shrink_with_the_face(self):
        """A fraction-only band is narrowest exactly where the noise is
        relatively largest. The floor is what decouples the two, so stability
        should hold across a range of face sizes rather than degrading."""
        for size in (100, 160, 260):
            self.assertLess(self._swing(size, int(size * 0.8)), 0.30,
                            f'join is unstable at face size {size}')

    def test_a_narrower_band_really_is_worse(self):
        """The floor is only worth its cost if removing it is measurably worse.

        `feather` cannot express this — the band takes the LARGER of the
        fraction and the floor, so asking for a narrow feather changes nothing
        while the floor stands. The floor itself has to come out, which is also
        exactly what the env override does, so this doubles as a check that
        turning it off restores the old behaviour rather than breaking.
        """
        with_floor = self._swing(140, 120)
        saved = face_overlap.MIN_BAND_PX
        try:
            face_overlap.MIN_BAND_PX = 0.0
            without_floor = self._swing(140, 120)
        finally:
            face_overlap.MIN_BAND_PX = saved
        self.assertGreater(without_floor, with_floor * 1.5,
                           f'floor made little difference: {without_floor:.3f} '
                           f'without vs {with_floor:.3f} with')

    def test_the_floor_is_capped_so_a_small_face_is_not_all_ramp(self):
        """16px of hand-over on a 50px-radius head would be a dissolve, not a
        demarcation. Small faces get a proportionally narrower band instead."""
        faces = [_Face(340, 200, 46), _Face(378, 200, 46)]
        regions = build_regions(faces, SHAPE, order=[0, 1])
        self.assertIsNotNone(regions)
        r = regions[1]
        row = r.own[200 - r.y0]
        soft = int(((row > 0.02) & (row < 0.98)).sum())
        self.assertLess(soft, 46 * 0.5,
                        f'{soft}px of ramp across a 46px face is a dissolve')


class Depth(unittest.TestCase):
    """Index 0 is the near (larger) face throughout, painted last."""

    def _boundary(self, near_size, far_size):
        r = build_regions([_Face(340, 200, near_size), _Face(470, 200, far_size)],
                          SHAPE, order=[1, 0])
        row = r[0].own[200 - r[0].y0]
        return int(np.argmin(np.abs(row - 0.5))) + r[0].x0

    def test_the_bigger_face_is_in_front(self):
        """On-screen size is the only depth cue available here, and a head twice
        the size of another is in front of it, not behind."""
        self.assertGreater(
            self._boundary(220, 110), self._boundary(220, 220),
            'the nearer (larger) face must take MORE of the contested area')

    def test_size_alone_does_not_swallow_a_face(self):
        """The signed distance is normalised by each face's own size for this
        reason: without it a big face is deeper everywhere just for being big,
        and would take the whole of a small face standing in front of it."""
        r = build_regions([_Face(300, 200, 260), _Face(470, 200, 90)],
                          SHAPE, order=[1, 0])
        self.assertEqual(_own_at(r[0], 470, 200), 0.0,
                         'the big face must not claim the small one\'s centre')


class Trimming(unittest.TestCase):
    def setUp(self):
        self.faces = [_Face(340, 200, 140), _Face(460, 200, 140)]
        self.regions = build_regions(self.faces, SHAPE, order=[0, 1])

    def test_trim_frame_clears_the_neighbours_half(self):
        matte = np.ones(SHAPE[:2], dtype=np.float32)
        self.regions[0].trim_frame(matte)
        self.assertEqual(matte[200, 460], 0.0)   # over face 1
        self.assertEqual(matte[200, 340], 1.0)   # over face 0
        self.assertEqual(matte[200, 40], 1.0)    # outside the ROI entirely

    def test_trim_frame_refuses_a_mismatched_matte(self):
        small = np.ones((50, 50), dtype=np.float32)
        self.regions[0].trim_frame(small)
        self.assertTrue((small == 1.0).all(),
                        'a matte at another resolution must be left alone '
                        'rather than silently mis-registered')

    def test_crop_is_none_when_the_box_is_all_ours(self):
        self.assertIsNone(self.regions[0].crop(0, 0, 30, 30))

    def test_crop_fills_the_part_outside_the_roi_with_ownership(self):
        r = self.regions[0]
        out = r.crop(r.x0 - 20, 190, r.x0 + 20, 210)
        self.assertEqual(out.shape, (20, 40))
        self.assertTrue((out[:, :20] == 1.0).all(),
                        'outside the contested rectangle the face owns everything')


class CompositeOrder(unittest.TestCase):
    """ProcessMgr._composite_faces decides paint order for the whole frame."""

    def _mgr(self):
        mgr = ProcessMgr.__new__(ProcessMgr)
        mgr.painted = []

        def _stub(face_index, target_face, frame, plate=None, region=None):
            mgr.painted.append((face_index, target_face, plate is frame, region))
            return frame

        mgr.process_face = _stub
        return mgr

    def test_faces_are_pasted_far_to_near(self):
        mgr = self._mgr()
        big, small = _Face(340, 200, 220), _Face(600, 200, 90)
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, big), (1, small)], plate, plate.copy())
        self.assertEqual([p[0] for p in mgr.painted], [1, 0],
                         'the smaller (further) face must be painted first so '
                         'the nearer one ends up on top')

    def test_paint_order_ignores_match_order(self):
        mgr_a, mgr_b = self._mgr(), self._mgr()
        big, small = _Face(340, 200, 220), _Face(600, 200, 90)
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr_a._composite_faces([(0, big), (1, small)], plate, plate.copy())
        mgr_b._composite_faces([(1, small), (0, big)], plate, plate.copy())
        self.assertEqual([p[0] for p in mgr_a.painted],
                         [p[0] for p in mgr_b.painted])

    def test_every_face_reads_the_untouched_plate(self):
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, _Face(340, 200, 140)), (1, _Face(460, 200, 140))],
                             plate, plate.copy())
        self.assertEqual([p[2] for p in mgr.painted], [False, False],
                         'reading the running composite is how one face ends up '
                         'swapping its neighbour\'s already-swapped pixels')

    def test_overlapping_faces_get_regions_and_lone_ones_do_not(self):
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, _Face(340, 200, 140)), (1, _Face(460, 200, 140))],
                             plate, plate.copy())
        self.assertTrue(all(p[3] is not None for p in mgr.painted))

        mgr = self._mgr()
        mgr._composite_faces([(0, _Face(120, 200, 120)), (1, _Face(680, 200, 120))],
                             plate, plate.copy())
        self.assertTrue(all(p[3] is None for p in mgr.painted))

    def test_near_equal_faces_keep_a_stable_order(self):
        """Detection noise moves a bbox by a pixel or two every frame. Sorting on
        raw area lets that flip which of two similarly sized faces is on top, so
        the join hands back and forth — the flicker the ordering exists to stop."""
        plate = np.zeros(SHAPE, dtype=np.uint8)
        seen = set()
        for jitter in (-2, -1, 0, 1, 2):
            mgr = self._mgr()
            left = _Face(340, 200, 140)
            right = _Face(460, 200, 140 + jitter)
            mgr._composite_faces([(0, left), (1, right)], plate, plate.copy())
            seen.add(tuple(p[0] for p in mgr.painted))
        self.assertEqual(len(seen), 1,
                         f'paint order changed under bbox noise: {seen}')

    def test_a_clearly_bigger_face_still_wins(self):
        """The quantisation must not be so coarse that a real depth difference
        stops registering."""
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        mgr._composite_faces([(0, _Face(340, 200, 140)), (1, _Face(460, 200, 175))],
                             plate, plate.copy())
        self.assertEqual([p[0] for p in mgr.painted], [0, 1])

    def test_nothing_pending_is_a_no_op(self):
        mgr = self._mgr()
        plate = np.zeros(SHAPE, dtype=np.uint8)
        temp = plate.copy()
        self.assertIs(mgr._composite_faces([], plate, temp), temp)


class SwapFacesPaintsOnlyAtTheEnd(unittest.TestCase):
    """A source guard, because the regression is invisible at runtime: pasting
    inside a match loop still produces a plausible frame, it just restores the
    order-dependent join. The match loops must only ever APPEND."""

    def test_the_match_loops_do_not_paste(self):
        src = open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8').read()
        body = src.split('def swap_faces(', 1)[1].split('\n    def ', 1)[0]
        calls = [ln.strip() for ln in body.splitlines()
                 if 'self.process_face(' in ln]
        self.assertEqual(
            len(calls), 1,
            'swap_faces may only call process_face once — the "first" mode, '
            'which has a single face. Every other mode collects its matches and '
            f'hands them to _composite_faces. Found: {calls}')
        self.assertIn('plate=frame', calls[0],
                      'even the single-face path must name its plate, or the '
                      'next reader assumes frame is both')


if __name__ == '__main__':
    unittest.main()
