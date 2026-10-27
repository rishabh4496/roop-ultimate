"""Face Manager Pose Radar, Auto-Prune and the Morph Lab (roop/identity_lab.py + routes_identity_lab.py).

Pure rules are checked against hand-built sets where the right answer is known by construction (an
orthogonal pair blends to cosine 0.7071 with each; a blurry face that is the only profile must stay).
The routes are driven through the real ASGI app with the Face Manager lists seeded directly, and a
saved blend is read back through the real V2 reader and `FaceSet.attach_v2_metadata` to prove the
vector the swapper will read is the blend and not the photo's own embedding.

NOT covered here (said in the changelog): `/api/facemgr/morph/preview` and the load of a saved blend
through `source_gallery._ingest_faceset` need the face detector, so they are exercised by hand only.
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault('ROOP_SKIP_STARTUP', '1')

from roop import identity_lab as lab  # noqa: E402


def unit(*components, dim=512):
    v = np.zeros(dim, np.float32)
    for i, c in enumerate(components):
        v[i] = c
    return v / np.linalg.norm(v)


def row(i, yaw=0.0, pitch=0.0, sharp=0.8, score=0.8):
    return {'index': i, 'yaw': yaw, 'pitch': pitch, 'bucket': lab.pose_bucket(yaw, pitch),
            'sharpness': sharp, 'sharp_var': 100.0, 'score': score}


class PoseBuckets(unittest.TestCase):
    def test_table(self):
        cases = [((0, 0), 'frontal'), ((39.9, 0), 'frontal'), ((40, 0), 'right_profile'),
                 ((-40, 0), 'left_profile'), ((-85, 5), 'left_profile'), ((10, 20), 'high_angle'),
                 ((10, -20), 'low_angle'), ((10, 19.9), 'frontal'),
                 ((-60, 35), 'left_profile'),              # a profile wins over a steep pitch
                 ((None, None), 'frontal'), ((float('nan'), 0), 'frontal')]
        for (yaw, pitch), want in cases:
            self.assertEqual(lab.pose_bucket(yaw, pitch), want, (yaw, pitch))

    def test_radar_counts_and_what_is_missing(self):
        rows = [row(0), row(1), row(2, yaw=-60), row(3, pitch=25)]
        r = lab.radar(rows)
        self.assertEqual({k: v['count'] for k, v in r['buckets'].items()},
                         {'frontal': 2, 'left_profile': 1, 'right_profile': 0, 'high_angle': 1, 'low_angle': 0})
        self.assertEqual(r['missing'], ['right_profile', 'low_angle'])
        self.assertEqual(r['coverage'], 0.6)
        self.assertEqual(r['buckets']['left_profile']['indices'], [2])
        self.assertEqual(lab.radar([])['coverage'], 0.0)

    def test_rows_read_the_face_managers_parallel_lists(self):
        meta = [{'yaw': -50.4, 'pitch': 3, 'sharpness': 0.6, 'sharp_var': 210.0}, {}]
        rows = lab.face_rows(meta, [0.9])
        self.assertEqual((rows[0]['bucket'], rows[0]['yaw'], rows[0]['score']), ('left_profile', -50.4, 0.9))
        self.assertEqual((rows[1]['bucket'], rows[1]['score']), ('frontal', 0.0))      # missing data is not a crash


class AutoPrune(unittest.TestCase):
    def test_a_blurry_face_goes_unless_it_is_the_only_one_of_its_pose(self):
        rows = [row(0, sharp=0.8), row(1, sharp=0.8), row(2, sharp=0.04),               # frontal x3, one blurry
                row(3, yaw=-70, sharp=0.03)]                                           # the ONLY left profile, blurry
        plan = lab.plan_prune(rows, [unit(1), unit(0, 1), unit(0, 0, 1), unit(0, 0, 0, 1)])
        gone = {d['index']: d['reason'] for d in plan['remove']}
        self.assertEqual(gone, {2: 'blurry'})
        self.assertEqual([d['index'] for d in plan['kept_for_coverage']], [3])
        self.assertEqual(plan['keep'], [0, 1, 3])

    def test_duplicates_keep_the_better_shot_and_only_within_a_pose(self):
        a = unit(1, 0.01)
        rows = [row(0, score=0.7), row(1, score=0.9), row(2, score=0.8, yaw=-60)]
        # 0 and 1 are the same shot twice (cosine ~1); 2 has the same embedding but a different pose
        plan = lab.plan_prune(rows, [a, a, a])
        self.assertEqual([(d['index'], d['reason'], d['duplicate_of']) for d in plan['remove']], [(0, 'duplicate', 1)])
        self.assertEqual(plan['keep'], [1, 2])

    def test_distinct_faces_are_not_duplicates(self):
        rows = [row(0), row(1)]
        plan = lab.plan_prune(rows, [unit(1), unit(0.8, 0.6)])      # cosine 0.8 < 0.96
        self.assertEqual(plan['remove'], [])

    def test_no_embeddings_means_no_duplicate_removal(self):
        plan = lab.plan_prune([row(0), row(1)], [None, None])
        self.assertEqual(plan['remove'], [])

    def test_a_uniformly_soft_set_is_not_emptied(self):
        rows = [row(i, sharp=0.12 + 0.01 * i) for i in range(6)]
        plan = lab.plan_prune(rows, [unit(*([0] * i), 1) for i in range(6)])
        # median 0.145 -> 0.45 x median = 0.065, under the 0.10 floor: the threshold is the floor, so the
        # faces at 0.12+ (all of them) are NOT blurry. A relative rule alone would have condemned the
        # softest of a soft set.
        self.assertEqual(plan['thresholds']['blur_below'], lab.BLUR_ABS_FLOOR)
        self.assertEqual(plan['remove'], [])
        self.assertEqual(plan['keep'], [0, 1, 2, 3, 4, 5])

    def test_never_removes_every_face(self):
        rows = [row(0, sharp=0.01, score=0.3), row(1, sharp=0.01, score=0.9)]
        # both blurry, both frontal: the better one must survive
        plan = lab.plan_prune(rows, [unit(1), unit(0, 1)], blur_relative=50.0)
        self.assertTrue(plan['keep'])

    def test_empty(self):
        self.assertEqual(lab.plan_prune([], [])['remove'], [])


class MorphSummary(unittest.TestCase):
    def test_equal_blend_of_two_orthogonal_identities(self):
        s = lab.morph_summary([unit(1), unit(0, 1)], [1, 1])
        self.assertEqual(s['weights'], [0.5, 0.5])
        for c in s['cosine_to_each']:
            self.assertAlmostEqual(c, 0.7071, places=3)
        self.assertAlmostEqual(float(np.linalg.norm(s['embedding'])), 1.0, places=5)
        self.assertIsNone(s['warning'])

    def test_free_weights_and_normalised_weights_are_the_same_blend(self):
        a = lab.morph_summary([unit(1), unit(0, 1), unit(0, 0, 1)], [50, 30, 20])
        b = lab.morph_summary([unit(1), unit(0, 1), unit(0, 0, 1)], [0.5, 0.3, 0.2])
        self.assertEqual(a['weights'], b['weights'])
        self.assertTrue(np.allclose(a['embedding'], b['embedding']))
        self.assertEqual(a['dominant'], 0)

    def test_a_weight_of_100_percent_is_that_identity(self):
        s = lab.morph_summary([unit(1), unit(0, 1)], [100, 0])
        self.assertAlmostEqual(s['cosine_to_each'][0], 1.0, places=4)
        self.assertAlmostEqual(s['cosine_to_each'][1], 0.0, places=4)

    def test_raw_vectors_of_any_length_are_normalised_first(self):
        s = lab.morph_summary([unit(1) * 20.0, unit(0, 1) * 0.5], [1, 1])
        self.assertEqual(s['weights'], [0.5, 0.5])
        self.assertAlmostEqual(s['cosine_to_each'][0], s['cosine_to_each'][1], places=4)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, '2 to 4'):
            lab.morph_summary([unit(1)], [1])
        with self.assertRaisesRegex(ValueError, '2 to 4'):
            lab.morph_summary([unit(i) for i in range(5)], [1] * 5)
        with self.assertRaisesRegex(ValueError, 'above zero'):
            lab.morph_summary([unit(1), unit(0, 1)], [0, 0])
        with self.assertRaisesRegex(ValueError, 'no usable embedding'):
            lab.morph_summary([unit(1), None], [1, 1])
        with self.assertRaisesRegex(ValueError, 'one weight'):
            lab.morph_summary([unit(1), unit(0, 1)], [1])
        with self.assertRaisesRegex(ValueError, 'cancel'):
            lab.morph_summary([unit(1), -unit(1)], [1, 1])

    def test_negative_and_nonfinite_weights_count_as_zero(self):
        with self.assertRaisesRegex(ValueError, 'above zero'):
            lab.morph_summary([unit(1), unit(0, 1)], [-5, float('nan')])
        s = lab.morph_summary([unit(1), unit(0, 1)], [-5, 3])      # the negative one is simply 0
        self.assertEqual(s['weights'], [0.0, 1.0])


class RoutesOverTheRealApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        import routes_facemgr as fm
        import routes_faceset
        from fastapi.testclient import TestClient
        cls.api, cls.fm, cls.routes_faceset = api, fm, routes_faceset
        cls.client = TestClient(api.app)

    def setUp(self):
        fm = self.fm
        self._saved = [list(x) for x in (fm.fm_thumbs, fm.fm_images, fm.fm_scores, fm.fm_meta, fm.fm_faces)]
        self.lib = tempfile.mkdtemp(prefix='lab_lib_')
        self.patch = mock.patch.object(self.routes_faceset, '_faceset_library_dir', lambda: self.lib)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        fm = self.fm
        for lst, saved in zip((fm.fm_thumbs, fm.fm_images, fm.fm_scores, fm.fm_meta, fm.fm_faces), self._saved):
            lst[:] = saved
        fm._update_fm_mean_embedding()
        shutil.rmtree(self.lib, ignore_errors=True)

    def _seed(self, specs):
        """specs: (yaw, pitch, sharpness, score, embedding) per face."""
        fm = self.fm
        for lst in (fm.fm_thumbs, fm.fm_images, fm.fm_scores, fm.fm_meta, fm.fm_faces):
            lst.clear()
        rng = np.random.default_rng(3)
        for yaw, pitch, sharp, score, emb in specs:
            img = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
            fm.fm_images.append(img)
            fm.fm_thumbs.append(img[:, :, ::-1].copy())
            fm.fm_scores.append(score)
            fm.fm_meta.append({'yaw': yaw, 'pitch': pitch, 'sharpness': sharp, 'sharp_var': sharp * 300})
            fm.fm_faces.append({'embedding': emb, 'normed_embedding': emb, 'bbox': np.array([0, 0, 96, 96], np.float32)})
        fm._update_fm_mean_embedding()

    def _lens(self):
        fm = self.fm
        n = len(fm.fm_images)
        self.assertTrue(n == len(fm.fm_thumbs) == len(fm.fm_scores) == len(fm.fm_meta) == len(fm.fm_faces),
                        'the five parallel lists drifted apart')
        return n

    def test_radar_route(self):
        self._seed([(0, 0, 0.8, 0.8, unit(1)), (-60, 0, 0.7, 0.7, unit(0, 1)), (5, 30, 0.9, 0.9, unit(0, 0, 1))])
        body = self.client.get('/api/facemgr/radar').json()
        self.assertEqual(body['total'], 3)
        self.assertEqual(sorted(body['missing']), ['low_angle', 'right_profile'])
        self.assertEqual(body['bucket_order'], list(lab.BUCKETS))
        self.assertEqual(len(body['faces']), 3)

    def test_radar_on_an_empty_set(self):
        self._seed([])
        body = self.client.get('/api/facemgr/radar').json()
        self.assertEqual((body['total'], body['coverage'], body['prune_preview']), (0, 0.0, None))

    def test_auto_prune_dry_run_changes_nothing_and_apply_keeps_the_lists_in_step(self):
        a = unit(1, 0.01)
        self._seed([(0, 0, 0.8, 0.9, a), (0, 0, 0.8, 0.7, a), (0, 0, 0.03, 0.2, unit(0, 1)),
                    (-60, 0, 0.8, 0.8, unit(0, 0, 1))])
        dry = self.client.post('/api/facemgr/auto_prune', json={'dry_run': True}).json()
        self.assertEqual((dry['dry_run'], dry['removed'], self._lens()), (True, 0, 4))
        self.assertEqual({d['index']: d['reason'] for d in dry['plan']['remove']}, {1: 'duplicate', 2: 'blurry'})
        real = self.client.post('/api/facemgr/auto_prune', json={}).json()
        self.assertEqual((real['removed'], self._lens(), len(real['faces'])), (2, 2, 2))
        self.assertEqual([m['yaw'] for m in self.fm.fm_meta], [0, -60])         # survivors, in order
        self.assertEqual(self.fm.fm_scores, [0.9, 0.8])
        self.assertIsNotNone(self.fm.fm_mean_embedding)

    def test_auto_prune_with_no_faces_is_a_400(self):
        self._seed([])
        self.assertEqual(self.client.post('/api/facemgr/auto_prune', json={}).status_code, 400)

    # ── morph ──
    def _library_identity(self, name, emb):
        from routes_identity_lab import write_blended_fsz
        rng = np.random.default_rng(abs(hash(name)) % 1000)
        img = (rng.random((160, 160, 3)) * 255).astype(np.uint8)
        path = os.path.join(self.lib, f'{name}.fsz')
        write_blended_fsz(path, emb, img, None, name)
        return path

    def test_candidates_blend_and_save_round_trip(self):
        self._library_identity('alice', unit(1))
        self._library_identity('bob', unit(0, 1))
        cands = self.client.get('/api/facemgr/morph/candidates').json()['candidates']
        lib = {c['id']: c for c in cands if c['kind'] == 'library'}
        self.assertEqual(sorted(lib), ['lib:alice.fsz', 'lib:bob.fsz'])
        self.assertTrue(all(c['available'] for c in lib.values()))

        comps = [{'id': 'lib:alice.fsz', 'weight': 75}, {'id': 'lib:bob.fsz', 'weight': 25}]
        blend = self.client.post('/api/facemgr/morph/blend', json={'components': comps}).json()
        self.assertEqual(blend['weights'], [0.75, 0.25])
        self.assertEqual(blend['dominant'], 0)
        self.assertAlmostEqual(blend['cosine_to_each'][0], 0.9487, places=3)      # 0.75/sqrt(0.75^2+0.25^2)
        self.assertEqual([c['label'] for c in blend['components']], ['alice', 'bob'])

        saved = self.client.post('/api/facemgr/morph/save', json={'components': comps, 'name': 'Mix A'}).json()
        self.assertEqual(saved['saved'], 'Mix A')
        path = os.path.join(self.lib, 'Mix A.fsz')
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(os.path.isfile(os.path.join(self.lib, 'Mix A.png')), 'the library thumbnail sidecar')

        # Read it back the way the app does: V2 archive -> FaceSet.attach_v2_metadata. The identity the
        # swapper uses is the blend, not whatever the stored photo's own embedding happens to be.
        from roop.faceset_v2 import read_faceset_archive
        from roop.FaceSet import FaceSet
        md = read_faceset_archive(path)
        fs = FaceSet()
        fs.attach_v2_metadata(md)
        expect = lab.morph_summary([unit(1), unit(0, 1)], [75, 25])['embedding']
        self.assertGreater(float(np.dot(fs.identity_embedding, expect)), 0.9999)
        self.assertTrue(any(c['id'] == 'lib:Mix A.fsz' for c in
                            self.client.get('/api/facemgr/morph/candidates').json()['candidates']))

    def test_saving_twice_never_overwrites(self):
        self._library_identity('alice', unit(1))
        self._library_identity('bob', unit(0, 1))
        comps = [{'id': 'lib:alice.fsz', 'weight': 1}, {'id': 'lib:bob.fsz', 'weight': 1}]
        a = self.client.post('/api/facemgr/morph/save', json={'components': comps, 'name': 'Mix'}).json()
        b = self.client.post('/api/facemgr/morph/save', json={'components': comps, 'name': 'Mix'}).json()
        self.assertNotEqual(a['saved'], b['saved'])
        self.assertEqual(len([f for f in os.listdir(self.lib) if f.endswith('.fsz')]), 4)

    def test_bad_requests_are_400_with_a_reason(self):
        self._library_identity('alice', unit(1))
        for comps, why in (([{'id': 'lib:alice.fsz', 'weight': 1}], '2 to 4'),
                           ([{'id': 'lib:alice.fsz', 'weight': 1}, {'id': 'lib:alice.fsz', 'weight': 1}], 'once'),
                           ([{'id': 'lib:alice.fsz', 'weight': 1}, {'id': 'lib:nobody.fsz', 'weight': 1}], 'not found'),
                           ([{'id': 'lib:alice.fsz', 'weight': 0}, {'id': 'lib:alice.fsz', 'weight': 0}], 'once')):
            for route in ('blend', 'save', 'preview'):
                r = self.client.post(f'/api/facemgr/morph/{route}', json={'components': comps})
                self.assertEqual(r.status_code, 400, (route, why))
                self.assertIn(why, r.json()['message'])

    # ── regressions found by running the lab on the real library (see the changelog) ──
    def _legacy_fsz(self, name):
        """A legacy (V1) archive: PNG members and NO metadata.json - what the whole real library is."""
        import zipfile
        import cv2
        path = os.path.join(self.lib, f'{name}.fsz')
        img = (np.random.default_rng(abs(hash(name)) % 1000).random((96, 96, 3)) * 255).astype(np.uint8)
        ok, buf = cv2.imencode('.png', img)
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr('0.png', buf.tobytes())
            zf.writestr('1.png', buf.tobytes())
        return path

    def test_legacy_archives_are_usable_and_the_detector_runs_only_when_needed(self):
        import routes_identity_lab as L
        self._legacy_fsz('old_a')
        self._legacy_fsz('old_b')
        calls = []
        faces = {'old_a': unit(1), 'old_b': unit(0, 1)}

        def fake_detect(path):
            calls.append(os.path.basename(path))
            who = os.path.splitext(os.path.basename(path))[0]
            img = np.full((96, 96, 3), 128, np.uint8)
            return [({'embedding': faces[who] * 20.0, 'det_score': 0.9, 'bbox': np.array([0, 0, 96, 96], np.float32)}, img),
                    ({'embedding': faces[who] * 20.0, 'det_score': 0.5, 'bbox': np.array([0, 0, 96, 96], np.float32)}, img)]
        with mock.patch.object(L, '_legacy_detect', fake_detect), mock.patch.object(L, '_embedding_cache', {}):
            cands = {c['id']: c for c in self.client.get('/api/facemgr/morph/candidates').json()['candidates']}
            self.assertTrue(cands['lib:old_a.fsz']['available'] and cands['lib:old_a.fsz']['legacy'])
            self.assertEqual(calls, [], 'listing a library must never run the detector')

            comps = [{'id': 'lib:old_a.fsz', 'weight': 1}, {'id': 'lib:old_b.fsz', 'weight': 1}]
            for _ in range(3):                                      # a slider drag is many requests
                body = self.client.post('/api/facemgr/morph/blend', json={'components': comps}).json()
            self.assertAlmostEqual(body['cosine_to_each'][0], 0.7071, places=3)
            self.assertEqual(sorted(calls), ['old_a.fsz', 'old_b.fsz'],
                             'the blend readout runs the detector once per file, then reads the cache')
            n = len(calls)
            saved = self.client.post('/api/facemgr/morph/save', json={'components': comps, 'name': 'From legacy'})
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertEqual(len(calls) - n, 1, 'only the dominant identity is re-detected, for its anchor image')

    def test_a_legacy_archive_with_no_face_is_a_400_that_says_which(self):
        import routes_identity_lab as L
        self._legacy_fsz('faceless')
        self._library_identity('alice', unit(1))
        with mock.patch.object(L, '_legacy_detect', lambda p: []), mock.patch.object(L, '_embedding_cache', {}):
            r = self.client.post('/api/facemgr/morph/blend', json={'components': [
                {'id': 'lib:faceless.fsz', 'weight': 1}, {'id': 'lib:alice.fsz', 'weight': 1}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("no face found in 'faceless.fsz'", r.json()['message'])

    def test_a_real_insightface_face_can_anchor_a_blend(self):
        """insightface's Face is a dict subclass that copy.copy cannot copy and whose normed_embedding is a
        read-only property. Both broke the first real save; a plain dict stand-in hid it."""
        import routes_identity_lab as L

        class RealishFace(dict):
            def __getattr__(self, name):                 # the real one answers None for anything unknown
                return self.get(name)

            def __setattr__(self, name, value):          # ...and forbids setting its derived properties
                if name == 'normed_embedding':
                    raise AttributeError("can't set attribute")
                self[name] = value

            @property
            def normed_embedding(self):
                e = self.get('embedding')
                return None if e is None else e / np.linalg.norm(e)

        with self.assertRaises(TypeError):
            import copy
            copy.copy(RealishFace(bbox=np.array([0, 0, 96, 96], np.float32)))      # the failure being guarded
        face = RealishFace(bbox=np.array([0, 0, 96, 96], np.float32), det_score=0.9, embedding=unit(1) * 21.0,
                           pose=[0.0, 0.0, 0.0])
        img = (np.random.default_rng(5).random((96, 96, 3)) * 255).astype(np.uint8)
        path = os.path.join(self.lib, 'anchored.fsz')
        L.write_blended_fsz(path, unit(0, 1), img, face, 'anchored')
        self.assertAlmostEqual(float(face['embedding'][0]), 21.0, places=3,
                               msg='the anchor face passed in must not be mutated (it is the loaded source\'s own)')
        from roop.faceset_v2 import read_faceset_archive
        from roop.FaceSet import FaceSet
        fs = FaceSet()
        fs.attach_v2_metadata(read_faceset_archive(path))
        self.assertGreater(float(np.dot(fs.identity_embedding, unit(0, 1))), 0.9999)

    def test_a_bug_is_a_500_not_a_400(self):
        """A TypeError is a defect, not a bad request. Catching it as 400 once turned 'NoneType is not
        callable' into a message that read like the user's mistake."""
        import routes_identity_lab as L
        with mock.patch.object(L, '_prepared', side_effect=TypeError('bug')):
            r = self.client.post('/api/facemgr/morph/save', json={'components': []})
        self.assertEqual(r.status_code, 500)

    def test_a_weight_that_is_not_a_number_is_a_400(self):
        self._library_identity('alice', unit(1))
        self._library_identity('bob', unit(0, 1))
        r = self.client.post('/api/facemgr/morph/blend', json={'components': [
            {'id': 'lib:alice.fsz', 'weight': 'lots'}, {'id': 'lib:bob.fsz', 'weight': 1}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn('numbers', r.json()['message'])

    def test_the_face_manager_set_is_a_candidate_once_it_has_faces(self):
        self._seed([])
        ids = [c['id'] for c in self.client.get('/api/facemgr/morph/candidates').json()['candidates']]
        self.assertNotIn('fm:set', ids)
        self._seed([(0, 0, 0.8, 0.8, unit(1))])
        ids = [c['id'] for c in self.client.get('/api/facemgr/morph/candidates').json()['candidates']]
        self.assertIn('fm:set', ids)


if __name__ == '__main__':
    unittest.main()
