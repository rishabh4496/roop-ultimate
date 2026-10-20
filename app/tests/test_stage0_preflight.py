"""Stage 0 (Phase 6): the frozen source-identity table.

The table's one promise is that it answers EXACTLY what the live per-face selection would have, for every
head pose. So the central tests run both and compare bit for bit - over a dense pose grid, on a V2 set with
missing cells, and with an Identity Blender recipe whose dials run into the 0.80 identity guard - and the
rest pin the cases where it must refuse to answer.

CPU only: the facesets are built from the same fields a `.fsz` load produces (pose bins, a centroid, Face
objects with a 512-d embedding); the blend uses the real shipped identity_directions.npz.
"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from insightface.app.common import Face  # noqa: E402

from roop import stage0_preflight as s0  # noqa: E402
from roop.FaceSet import FaceSet  # noqa: E402
from roop.faceset_v2 import (FORMAT_NAME, POSE_MATRIX_CELLS, pose_matrix_cell,  # noqa: E402
                             pose_matrix_key)
from roop.identity_algebra import BlendRecipe, load_directions, resolve_recipe  # noqa: E402


def unit(rng):
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def v2_set(seed, sid, cells=None, n_faces=2):
    """A V2 faceset as `attach_v2_metadata` leaves it: pose bins, a centroid, detector Faces."""
    rng = np.random.default_rng(seed)
    fs = FaceSet()
    for i in range(n_faces):
        fs.faces.append(Face(bbox=np.array([0, 0, 100, 100], np.float32),
                             kps=(rng.random((5, 2)) * 100).astype(np.float32),
                             embedding=(unit(rng) * 21.0).astype(np.float32),     # raw ArcFace norm ~20
                             det_score=np.float32(0.9)))
    cells = list(POSE_MATRIX_CELLS) if cells is None else cells
    meta = {
        'schema': FORMAT_NAME, 'version': 2,
        'pose_bins': {pose_matrix_key(c): {'embedding': unit(rng).tolist()} for c in cells},
        'default_embedding': unit(rng).tolist(),
        'sources': [{'geometry': {'yaw': 0.0 if i == 0 else 40.0, 'pitch': 0.0}} for i in range(n_faces)],
    }
    fs.attach_v2_metadata(meta)
    fs._source_id = sid
    return fs


def live(fs, blend, yaw, pitch, roll=0.0):
    """What process_face's two live blocks produce for a target at this pose."""
    base = fs.faces[fs.canonical_face_index()]
    target = Face(_adaptive_yaw=yaw, _adaptive_pitch=pitch, _adaptive_roll=roll)
    face = s0.pose_conditioned_input(base, fs, target)
    if blend is not None:
        face = s0.blended_input(face, blend, fs, target)
    return face


def same_face(a, b):
    """Field-for-field, dtype-for-dtype."""
    if sorted(map(str, a.keys())) != sorted(map(str, b.keys())):
        return 'keys %s != %s' % (sorted(map(str, a.keys())), sorted(map(str, b.keys())))
    for k in a.keys():
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            if not isinstance(y, np.ndarray) or x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                return 'field %r differs' % k
        elif x != y:
            return 'field %r: %r != %r' % (k, x, y)
    return ''


def build(sets, blend=None, **kw):
    kw.setdefault('swap_mode', 'normed')
    kw.setdefault('use_source_bank', False)
    kw.setdefault('quality_engine', False)
    lines = []
    table = s0.build(sets, blend=blend, log=lines.append, **kw)
    return table, lines


class CellIndex(unittest.TestCase):
    def test_yaw_pitch_agree_with_the_fsz_pose_matrix_everywhere(self):
        names = {'left': 0, 'center': 1, 'right': 2, 'down': 0, 'up': 2}
        for yaw in list(np.arange(-90, 91, 2.5)) + [-25.0, -25.0001, 25.0, 25.0001, float('nan')]:
            for pitch in list(np.arange(-45, 46, 2.5)) + [-15.0, -15.0001, 15.0, 15.0001, float('nan')]:
                yc, pc_ = pose_matrix_cell((yaw, pitch))
                yi, pi, _ = s0.cell_index(yaw, pitch, 0.0)
                self.assertEqual((yi, pi), (names[yc], names[pc_]), (yaw, pitch))

    def test_roll_bins_and_unknown_roll(self):
        self.assertEqual([s0.roll_bin(r) for r in (-90, -15.01, -15, 0, 15, 15.01, 90)], [0, 0, 1, 1, 1, 2, 2])
        for bad in (None, float('nan'), float('inf')):
            self.assertEqual(s0.roll_bin(bad), 1)

    def test_a_stand_in_target_lies_inside_its_own_cell(self):
        for yi in range(3):
            for pi in range(3):
                for ri in range(3):
                    t = s0.stand_in_target((yi, pi, ri))
                    self.assertEqual(s0.cell_index(t['_adaptive_yaw'], t['_adaptive_pitch'], t['_adaptive_roll']),
                                     (yi, pi, ri))


class Equivalence(unittest.TestCase):
    """The table equals the live path."""

    def assert_equivalent(self, sets, blend, n=400, seed=7):
        table, _ = build(sets, blend)
        rng = np.random.default_rng(seed)
        checked = 0
        for _ in range(n):
            yaw, pitch, roll = rng.uniform(-95, 95), rng.uniform(-50, 50), rng.uniform(-70, 70)
            for idx, fs in enumerate(sets):
                if idx not in table.sources:
                    continue
                target = Face(_adaptive_yaw=float(yaw), _adaptive_pitch=float(pitch), _adaptive_roll=float(roll))
                base = table.sources[idx].base
                got = table.lookup(idx, base, target)
                want = live(fs, blend, float(yaw), float(pitch), float(roll))
                self.assertIsNotNone(got)
                self.assertEqual(same_face(got, want), '', msg=(idx, yaw, pitch, roll))
                checked += 1
        self.assertGreater(checked, 0)
        return table

    def test_a_v2_set_with_every_cell(self):
        self.assert_equivalent([v2_set(1, 'a')], None)

    def test_a_v2_set_with_missing_cells_falls_back_to_the_centroid_exactly_as_live(self):
        fs = v2_set(2, 'a', cells=[('center', 'center'), ('left', 'center')])
        table = self.assert_equivalent([fs], None)
        cell = table.sources[0].cells[2][2][1]                      # right / up: no such cell stored
        np.testing.assert_array_equal(cell['embedding'], fs.default_embedding.astype(np.float32))

    def test_two_sources(self):
        self.assert_equivalent([v2_set(3, 'a'), v2_set(4, 'b', cells=[('center', 'center')])], None)

    def test_with_a_blend_of_two_sources_and_a_writable_dial(self):
        sets = [v2_set(5, 'a'), v2_set(6, 'b')]
        recipe = BlendRecipe.from_payload({'enabled': True, 'components': [{'source_id': 'a', 'weight': 70},
                                                                          {'source_id': 'b', 'weight': 30}],
                                           'dials': {'jawline': 0.6}})
        blend = resolve_recipe(recipe, sets, load_directions())
        self.assertIsNotNone(blend)
        table = self.assert_equivalent(sets, blend)
        self.assertTrue(all(e.blended for e in table.sources.values()))

    def test_with_dials_only_on_one_source(self):
        sets = [v2_set(7, 'a')]
        recipe = BlendRecipe.from_payload({'enabled': True, 'components': [], 'dials': {'jawline': -0.8}})
        blend = resolve_recipe(recipe, sets, load_directions())
        self.assertIsNotNone(blend)
        self.assert_equivalent(sets, blend)

    def test_the_identity_guard_holds_in_every_cell_of_a_pushed_dial(self):
        """A dial that would take the vector below the guard's cosine is scaled back: the guard is evaluated
        ONCE per cell now, and no cell may end up past it. (The shipped jawline dial reaches exactly cos 0.80
        at full scale, so the default guard never clamps it; 0.90 makes the guard bite.)"""
        sets = [v2_set(8, 'a')]
        recipe = BlendRecipe.from_payload({'enabled': True, 'components': [], 'dials': {'jawline': 1.0},
                                           'min_cosine': 0.90})
        blend = resolve_recipe(recipe, sets, load_directions())
        table, _ = build(sets, blend)
        entry = table.sources[0]
        for yi in range(3):
            for pi in range(3):
                face = entry.cells[yi][pi][1]
                target = Face(_adaptive_yaw=s0._YAW_CENTRE[yi], _adaptive_pitch=s0._PITCH_CENTRE[pi])
                anchor = s0.pose_conditioned_input(entry.base, sets[0], target)['embedding']
                a = np.asarray(anchor, np.float64)
                b = np.asarray(face['embedding'], np.float64)
                cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                self.assertGreaterEqual(cos, 0.90 - 1e-6, (yi, pi))
        self.assertGreater(blend.clamped, 0, 'the test must actually push a dial into the guard')

    def test_the_summary_names_the_dials_the_guard_and_the_unwritable_ones(self):
        sets = [v2_set(11, 'a')]
        recipe = BlendRecipe.from_payload({'enabled': True, 'components': [],
                                           'dials': {'jawline': 1.0, 'age': 10.0}, 'min_cosine': 0.90})
        blend = resolve_recipe(recipe, sets, load_directions())
        table, lines = build(sets, blend)
        entry = table.sources[0]
        self.assertEqual(entry.dials_applied, ('jawline',))
        self.assertEqual(entry.dials_unavailable, ('age',))      # age does not write through hyperswap (render validation)
        self.assertEqual(entry.clamped_cells, 9)
        text = ' '.join(lines)
        self.assertIn("dials pre-applied ['jawline']", text)
        self.assertIn("unavailable (not writable through the swapper, identity_render_validation.json) ['age']", text)
        self.assertIn('bit in 9 of 9 cells', text)

    def test_no_clamping_is_reported_when_the_guard_never_bites(self):
        sets = [v2_set(12, 'a')]
        recipe = BlendRecipe.from_payload({'enabled': True, 'components': [], 'dials': {'jawline': 0.2}})
        blend = resolve_recipe(recipe, sets, load_directions())
        table, lines = build(sets, blend)
        self.assertEqual(table.sources[0].clamped_cells, 0)
        self.assertIn('bit in 0 of 9 cells', ' '.join(lines))

    def test_roll_does_not_change_the_vector_but_is_part_of_the_key(self):
        fs = v2_set(9, 'a')
        table, _ = build([fs])
        base = table.sources[0].base
        for yaw, pitch in ((0, 0), (40, 20), (-40, -20)):
            faces = [table.lookup(0, base, Face(_adaptive_yaw=yaw, _adaptive_pitch=pitch, _adaptive_roll=r))
                     for r in (-45.0, 0.0, 45.0, float('nan'), None)]
            self.assertTrue(all(f is faces[0] for f in faces))

    def test_lookup_reads_a_dict_target_too(self):
        fs = v2_set(10, 'a')
        table, _ = build([fs])
        base = table.sources[0].base
        got = table.lookup(0, base, {'_adaptive_yaw': 40.0, '_adaptive_pitch': 0.0})
        self.assertIs(got, table.sources[0].cells[2][1][1])


class Refusals(unittest.TestCase):
    def reason(self, fs, **kw):
        table, lines = build([fs], **kw)
        return table, table.stats.declined.get(0), lines

    def test_an_image_swapper(self):
        table, why, _ = self.reason(v2_set(1, 'a'), swap_mode='image')
        self.assertFalse(table.armed())
        self.assertIn('image crop', why)

    def test_a_cscs_swapper(self):
        _, why, _ = self.reason(v2_set(1, 'a'), swap_mode='cscs_dual')
        self.assertIn('CSCS', why)

    def test_the_source_bank_with_several_faces(self):
        _, why, _ = self.reason(v2_set(1, 'a', n_faces=3), use_source_bank=True)
        self.assertIn('use_source_bank', why)

    def test_the_source_bank_with_one_face_is_fine(self):
        table, why, _ = self.reason(v2_set(1, 'a', n_faces=1), use_source_bank=True)
        self.assertIsNone(why)
        self.assertTrue(table.armed())

    def test_the_quality_engine(self):
        _, why, _ = self.reason(v2_set(1, 'a'), quality_engine=True)
        self.assertIn('quality engine', why)

    def test_an_angle_portfolio(self):
        fs = v2_set(1, 'a')
        fs.angle_portfolio = SimpleNamespace()
        _, why, _ = self.reason(fs)
        self.assertIn('angle portfolio', why)

    def test_a_folder_bank_is_declined_not_approximated(self):
        fs = v2_set(1, 'a')
        fs.reference_embeddings = [unit(np.random.default_rng(0))]
        _, why, _ = self.reason(fs)
        self.assertIn('continuously', why)

    def test_a_v1_set_without_a_recipe_has_nothing_to_pre_apply(self):
        fs = FaceSet()
        fs.faces.append(Face(embedding=np.ones(512, np.float32), kps=np.zeros((5, 2), np.float32)))
        table, why, _ = self.reason(fs)
        self.assertFalse(table.armed())
        self.assertIn('nothing to pre-apply', why)

    def test_an_empty_source(self):
        _, why, _ = self.reason(FaceSet())
        self.assertIn('no face', why)

    def test_the_kill_switch(self):
        os.environ['ROOP_STAGE0'] = '0'
        self.addCleanup(os.environ.pop, 'ROOP_STAGE0', None)
        table, lines = build([v2_set(1, 'a')])
        self.assertFalse(table.armed())
        self.assertTrue(any('ROOP_STAGE0=0' in l for l in lines))

    def test_every_decline_is_printed(self):
        _, _, lines = self.reason(v2_set(1, 'a'), swap_mode='image')
        self.assertTrue(any('source 0: live path' in l for l in lines), lines)

    def test_a_source_whose_build_raises_runs_the_live_path_and_the_rest_are_unaffected(self):
        good, bad = v2_set(1, 'a'), v2_set(2, 'b')
        bad.canonical_face_index = lambda: (_ for _ in ()).throw(RuntimeError('boom'))
        table, _ = build([good, bad])
        self.assertIn(0, table.sources)
        self.assertNotIn(1, table.sources)
        self.assertIn('boom', table.stats.declined[1])

    def test_a_recipe_that_skips_the_source_is_treated_as_no_recipe(self):
        sets = [v2_set(1, 'a'), v2_set(2, 'b'), v2_set(3, 'c')]
        recipe = BlendRecipe.from_payload({'enabled': True, 'components': [{'source_id': 'a', 'weight': 1},
                                                                          {'source_id': 'b', 'weight': 1}]})
        blend = resolve_recipe(recipe, sets, load_directions())
        table, _ = build(sets, blend)
        self.assertTrue(table.sources[0].blended and table.sources[1].blended)
        self.assertFalse(table.sources[2].blended)


class Lookup(unittest.TestCase):
    def test_a_face_that_is_not_the_one_the_table_was_built_from_misses(self):
        """Whatever re-selects the source upstream (source bank, quality engine) must win."""
        fs = v2_set(1, 'a')
        table, _ = build([fs])
        other = fs.faces[1] if fs.faces[0] is table.sources[0].base else fs.faces[0]
        self.assertIsNone(table.lookup(0, other, Face(_adaptive_yaw=0.0, _adaptive_pitch=0.0)))
        self.assertEqual((table.stats.hits, table.stats.misses), (0, 1))

    def test_an_unknown_source_index_is_none_and_not_a_miss(self):
        table, _ = build([v2_set(1, 'a')])
        self.assertIsNone(table.lookup(5, object(), Face()))
        self.assertEqual(table.stats.misses, 0)

    def test_hits_are_counted(self):
        fs = v2_set(1, 'a')
        table, _ = build([fs])
        for _ in range(5):
            table.lookup(0, table.sources[0].base, Face(_adaptive_yaw=1.0, _adaptive_pitch=1.0))
        self.assertEqual(table.stats.hits, 5)

    def test_the_prebuilt_faces_are_not_the_canonical_one_and_do_not_alias_it(self):
        fs = v2_set(1, 'a')
        table, _ = build([fs])
        base = table.sources[0].base
        before = base['embedding'].copy()
        for yi in range(3):
            for pi in range(3):
                self.assertIsNot(table.sources[0].cells[yi][pi][1], base)
        np.testing.assert_array_equal(base['embedding'], before)       # building never touched the source itself


class Prewarm(unittest.TestCase):
    def test_the_latent_is_built_once_per_distinct_face_not_per_frame(self):
        fs = v2_set(1, 'a')
        seen = []
        table, _ = build([fs], prewarm=lambda f: seen.append(id(f)))
        self.assertEqual(len(seen), 9)
        self.assertEqual(len(set(seen)), 9)

    def test_a_prewarm_that_fails_does_not_stop_the_build(self):
        def boom(face):
            raise RuntimeError('no latent')
        table, _ = build([v2_set(1, 'a')], prewarm=boom)
        self.assertTrue(table.armed())


class Digest(unittest.TestCase):
    def test_same_sources_same_digest_and_a_changed_cell_changes_it(self):
        a, _ = build([v2_set(1, 'a')])
        b, _ = build([v2_set(1, 'a')])
        self.assertEqual(a.digest(), b.digest())
        c, _ = build([v2_set(2, 'a')])
        self.assertNotEqual(a.digest(), c.digest())

    def test_the_summary_says_how_much_was_frozen(self):
        _, lines = build([v2_set(1, 'a')])
        self.assertTrue(any('[Stage0] source 0: 9 of 9 pose cells' in l for l in lines), lines)


if __name__ == '__main__':
    unittest.main()
