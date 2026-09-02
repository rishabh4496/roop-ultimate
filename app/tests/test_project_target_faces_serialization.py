"""A project must be able to record the target faces it exists to restore.

`_project_target_faces` hands the checkpoint the detector's own facts — bbox,
kps, embedding, landmark_2d_106, landmark_3d_68 — and every one of them is a
numpy ARRAY. `_json_default` asked for `item` before `tolist`, and an ndarray
has both: `ndarray.item()` raises "can only convert an array of size 1 to a
Python scalar" for anything bigger than one element.

So the serializer worked on the numpy SCALARS (age, gender) and failed on the
arrays. /api/swap answered "cannot create a recoverable project: can only
convert an array of size 1 to a Python scalar" and refused to start, but ONLY
once the user had captured a target face — a swap with none was unaffected,
which is why the suite stayed green and why every project on disk reads
`target_faces: 0`.

The tests below use the real shapes rather than a token array, because the
scalar fields are exactly the ones the broken order got right.
"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import project_checkpoint as checkpoints


def _detector_facts():
    return {
        'bbox': np.array([10.0, 20.0, 110.0, 140.0], dtype=np.float32),
        'kps': np.arange(10, dtype=np.float32).reshape(5, 2),
        'embedding': np.linspace(-1.0, 1.0, 512, dtype=np.float32),
        'landmark_2d_106': np.zeros((106, 2), dtype=np.float32),
        'landmark_3d_68': np.zeros((68, 3), dtype=np.float32),
        'gender': np.int64(1),
        'age': np.float32(31.0),
    }


class JsonDefaultShapes(unittest.TestCase):

    def test_arrays_survive_as_nested_lists(self):
        facts = _detector_facts()
        restored = json.loads(json.dumps(facts, default=checkpoints._json_default))
        for key in ('bbox', 'kps', 'embedding', 'landmark_2d_106', 'landmark_3d_68'):
            self.assertEqual(np.asarray(restored[key]).shape, facts[key].shape,
                             f'{key} must round-trip with its shape intact')
            np.testing.assert_allclose(np.asarray(restored[key], dtype=np.float32),
                                       facts[key])

    def test_numpy_scalars_stay_scalars(self):
        # The branch that used to run first. It must keep working: a scalar
        # written as a one-element list would break `int(record['gender'])`.
        restored = json.loads(json.dumps(
            {'gender': np.int64(1), 'age': np.float32(31.0), 'score': np.float64(0.5)},
            default=checkpoints._json_default))
        self.assertEqual(restored['gender'], 1)
        self.assertEqual(restored['age'], 31.0)
        self.assertEqual(restored['score'], 0.5)
        for value in restored.values():
            self.assertNotIsInstance(value, list)

    def test_zero_dimensional_array_is_its_element(self):
        restored = json.loads(json.dumps({'v': np.array(7.0, dtype=np.float32)},
                                         default=checkpoints._json_default))
        self.assertEqual(restored['v'], 7.0)

    def test_unknown_object_still_falls_back_to_str(self):
        class Opaque:
            def __repr__(self):
                return 'opaque'
        restored = json.loads(json.dumps({'v': Opaque()},
                                         default=checkpoints._json_default))
        self.assertEqual(restored['v'], 'opaque')

    def test_fingerprint_accepts_arrays(self):
        # fingerprint() shares the serializer, and new_project() hashes the
        # payload before it writes anything.
        self.assertEqual(len(checkpoints.fingerprint(_detector_facts())), 64)


class ProjectRecordsTargetFaces(unittest.TestCase):
    """The end-to-end shape: write a project carrying target faces, read it back."""

    def setUp(self):
        self.created = []

    def tearDown(self):
        for project_id in self.created:
            try:
                os.unlink(checkpoints.project_path(project_id))
            except OSError:
                pass

    def test_new_project_writes_and_reloads_target_faces(self):
        faces = [{'data': _detector_facts(), 'group': 0, 'thumbnail': 'data:,'}]
        record = checkpoints.new_project(
            job_id=None, name='probe', payload={'blend_ratio': np.float32(0.8)},
            sources=[], target={'name': 'probe.mp4', 'path': 'probe.mp4'},
            frame_start=0, frame_end=1,
            output={'directory': '', 'format': 'mp4'}, cfg=None,
            target_faces=json.loads(json.dumps(faces, default=checkpoints._json_default)),
            app_version='test')
        self.created.append(record['id'])

        reloaded = checkpoints.load(record['id'])
        stored = (reloaded.get('inputs') or {}).get('target_faces') or []
        self.assertEqual(len(stored), 1,
                         'a captured target face must reach the project file')
        data = stored[0]['data']
        np.testing.assert_allclose(np.asarray(data['bbox'], dtype=np.float32),
                                   _detector_facts()['bbox'])
        self.assertEqual(np.asarray(data['landmark_2d_106']).shape, (106, 2))
        self.assertEqual(data['gender'], 1)


if __name__ == '__main__':
    unittest.main()
