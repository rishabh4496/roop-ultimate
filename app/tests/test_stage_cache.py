"""Stage 1 / Stage 2 persistence (Phase 6): the codec, the project record, and the adapter.

The cache has one job: a resumed project must get back EXACTLY what the interrupted run's pre-passes
sealed. "Nearly" is a different face swapped, so these tests compare type, dtype, shape and value, not
equality of a printout, and pin every refusal (a stale key, a tampered file, a window the cache does not
cover) because a cache that is wrongly accepted is worse than none.

No GPU, no models: the faces are built from the field types `procmgr_tracking` actually produces
(ndarray float32 bbox / kps / landmarks / embeddings, numpy and python scalars, tuples, dicts, strings).
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from insightface.app.common import Face  # noqa: E402

import project_checkpoint as pc  # noqa: E402
import stage_checkpoint as sck  # noqa: E402
from roop import stage_cache as sc  # noqa: E402
from roop.stage2_plan import build_stage2_plan  # noqa: E402


# ── builders ─────────────────────────────────────────────────────────────────────────────────────────
def make_face(rng, track, frame, interpolated=False, extra=True):
    f = Face(
        bbox=(rng.random(4) * 500).astype(np.float32),
        kps=(rng.random((5, 2)) * 500).astype(np.float32),
        landmark_2d_106=(rng.random((106, 2)) * 500).astype(np.float32),
        landmark_3d_68=(rng.random((68, 3)) * 500).astype(np.float32),
        embedding=rng.standard_normal(512).astype(np.float32),
        det_score=np.float32(0.5 + rng.random() / 2),
        gender=np.int64(1), age=int(20 + rng.integers(40)),
        pose=(rng.random(3) * 30).astype(np.float32),
    )
    f['_track_id'] = int(track)
    f['_interpolated'] = bool(interpolated)
    if extra:
        # Every JSON-column value is DIFFERENT per face and frame. Constant values here made a wrong block
        # index return an equal-looking face: a mutation of the block arithmetic passed the whole suite.
        f['_expression_blink_state'] = 'open%d_%d' % (frame, track)
        f['_expression_eye_strengths'] = (0.25 + frame, np.float32(0.5 + track))
        f['_pose_v5'] = {'yaw': 1.5 + frame, 'pitch': np.float32(-2.0 - track), 'tag': 'x%d' % frame}
        f['_temporal_other_track_ids'] = [3, 4, frame * 10 + track]
        f['_temporal_interaction_score'] = float(frame) / 7.0
        f['_kps_detector'] = f['kps'].copy()
    return f


def make_pm(n_frames=40, start_rel=0, tracks=(1, 2), seed=0, cuts=(17,)):
    """A stand-in ProcessMgr carrying what _precompute_temporal leaves behind."""
    rng = np.random.default_rng(seed)
    emb = {t: rng.standard_normal(512).astype(np.float32) for t in tracks}
    faces, assign = {}, {}
    for fr in range(start_rel, n_frames):
        row = []
        for t in tracks:
            if t == 2 and fr % 5 == 0:          # holes, like a real track
                continue
            row.append(make_face(rng, t, fr, interpolated=(fr % 7 == 0), extra=(fr % 2 == 0)))
        faces[fr] = row
        assign[fr] = [((10.0 * t, 20.0), (0 if t == 1 else None), emb[t]) for t in tracks if not (t == 2 and fr % 5 == 0)]
    smap = {1: (0, emb[1]), 2: (None, emb[2])}
    pm = SimpleNamespace(_temporal_faces=faces, _track_assignments=assign, _track_source_map=smap,
                         _shot_boundaries=set(cuts), _temporal_covered=n_frames, _track_scanned=n_frames,
                         _active_ranges=None, stage2_frame_plan=None)
    pm.stage2_frame_plan, _ = build_stage2_plan(faces, smap)
    return pm


def deep_equal(a, b, path='$'):
    """Strict: same type, dtype, shape and value. Returns '' when equal, else where it differs."""
    if type(a) is not type(b):
        return f'{path}: type {type(a).__name__} != {type(b).__name__}'
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f'{path}: {a.dtype}{a.shape} != {b.dtype}{b.shape}'
        return '' if np.array_equal(a, b, equal_nan=a.dtype.kind == 'f') else f'{path}: values differ'
    if isinstance(a, np.generic):
        return '' if (a == b or (a != a and b != b)) else f'{path}: {a!r} != {b!r}'
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()) and set(a.keys()) != set(b.keys()):
            return f'{path}: keys {sorted(map(str, a))} != {sorted(map(str, b))}'
        for k in a:
            r = deep_equal(a[k], b[k], f'{path}[{k!r}]')
            if r:
                return r
        return ''
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f'{path}: len {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            r = deep_equal(x, y, f'{path}[{i}]')
            if r:
                return r
        return ''
    return '' if a == b else f'{path}: {a!r} != {b!r}'


class Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='stagecache-')
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, 'p.stage2.npz')

    def write(self, pm, start=100, count=None, ranges=None):
        return sc.write_stage2(self.path, pm, frame_start=start, frame_count=count or pm._temporal_covered,
                               ranges_abs=ranges, plan=pm.stage2_frame_plan)

    def restore(self, start, count, ranges=None):
        target = SimpleNamespace()
        return target, sc.restore_stage2(self.path, target, new_start=start, new_count=count,
                                         current_ranges_abs=ranges, build_plan=build_stage2_plan)


# ── the value codec ──────────────────────────────────────────────────────────────────────────────────
class ValueCodec(unittest.TestCase):
    def roundtrip(self, value):
        reg = sc._Arrays()
        enc = json.loads(json.dumps(sc._enc(value, reg)))
        arrays = {g: a for (g, _), a in zip(sorted(reg._groups.items()), [np.stack(v) for _, v in sorted(reg._groups.items())])}
        return sc._dec(enc, arrays)

    def test_scalars_keep_their_exact_type(self):
        for v in (None, True, False, 0, -7, 2 ** 70, 1.5, 'x', np.float32(0.1), np.float64(0.1),
                  np.int64(5), np.uint8(7), np.bool_(True), np.bool_(False)):
            out = self.roundtrip(v)
            self.assertEqual(deep_equal(v, out), '', msg=repr(v))

    def test_np_float64_is_not_collapsed_into_a_python_float(self):
        # np.float64 SUBCLASSES float; the codec must not take the python branch for it.
        out = self.roundtrip(np.float64(0.25))
        self.assertIsInstance(out, np.float64)

    def test_containers_keep_their_type(self):
        for v in ((1, 2.0, 'a'), [1, (2, 3)], {1: 'a', 'b': (1, 2)}, {2, 3, 5}, frozenset({1, 2}),
                  {'yaw': np.float32(1.0), 'n': [1, 2]}):
            self.assertEqual(deep_equal(v, self.roundtrip(v)), '', msg=repr(v))

    def test_non_finite_floats(self):
        out = self.roundtrip([float('nan'), float('inf'), -float('inf'), np.float32('nan')])
        self.assertTrue(np.isnan(out[0]) and out[1] == float('inf') and out[2] == -float('inf'))
        self.assertIsInstance(out[3], np.float32)
        self.assertTrue(np.isnan(out[3]))

    def test_an_array_inside_a_value_keeps_dtype_and_shape(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertEqual(deep_equal({'k': a}, self.roundtrip({'k': a})), '')

    def test_a_shared_array_is_stored_once(self):
        a = np.ones(512, np.float32)
        reg = sc._Arrays()
        sc._enc([(0, a), (1, a), (2, a)], reg)
        (name, stacked), = list(reg.members())
        self.assertEqual(stacked.shape, (1, 512))

    def test_values_the_codec_cannot_restore_exactly_are_refused(self):
        class Odd:
            pass
        for v in (Odd(), object(), np.array(['a']), 1 + 2j, np.complex64(1), b'raw', np.array([None], dtype=object)):
            with self.assertRaises(sc.Unserializable, msg=repr(v)):
                sc._enc(v, sc._Arrays())


# ── Stage 2 sidecar ──────────────────────────────────────────────────────────────────────────────────
class Stage2RoundTrip(Tmp):
    def test_everything_comes_back_exactly(self):
        pm = make_pm(40)
        entry = self.write(pm, start=0)
        self.assertEqual(entry['faces'], sum(len(v) for v in pm._temporal_faces.values()))
        target, report = self.restore(0, 40)
        self.assertEqual(deep_equal(pm._temporal_faces, target._temporal_faces), '')
        self.assertEqual(deep_equal(pm._track_assignments, target._track_assignments), '')
        self.assertEqual(deep_equal(pm._track_source_map, target._track_source_map), '')
        self.assertEqual(deep_equal(pm._shot_boundaries, target._shot_boundaries), '')
        self.assertEqual(target._temporal_covered, 40)
        self.assertEqual(sorted(pm.stage2_frame_plan), sorted(target.stage2_frame_plan))

    def test_faces_come_back_as_insightface_faces_not_dicts(self):
        pm = make_pm(8)
        self.write(pm, start=0)
        target, _ = self.restore(0, 8)
        face = target._temporal_faces[0][0]
        self.assertIs(type(face), Face)
        self.assertEqual(face.bbox.dtype, np.float32)            # attribute access works (Face.__getattr__)
        self.assertEqual(face.normed_embedding.shape, (512,))    # derived property works

    def test_faces_that_lack_a_field_still_lack_it(self):
        pm = make_pm(10)
        self.write(pm, start=0)
        target, _ = self.restore(0, 10)
        for fr in pm._temporal_faces:
            for a, b in zip(pm._temporal_faces[fr], target._temporal_faces[fr]):
                self.assertEqual(sorted(a.keys()), sorted(b.keys()))

    def test_the_track_mean_embedding_stays_one_shared_array(self):
        pm = make_pm(20)
        self.write(pm, start=0)
        target, _ = self.restore(0, 20)
        a = target._track_source_map[1][1]
        entries = [e for v in target._track_assignments.values() for e in v if e[1] == 0]
        self.assertTrue(entries and all(e[2] is a for e in entries),
                        'assignment entries and the source map must share the track embedding, as in the pre-pass')

    def test_restored_arrays_own_their_memory(self):
        pm = make_pm(6)
        self.write(pm, start=0)
        target, _ = self.restore(0, 6)
        arr = target._temporal_faces[0][0].bbox
        self.assertTrue(arr.flags.owndata or arr.base is not None)
        arr[0] = 12345.0                                            # writable, and not a view onto a closed file
        self.assertEqual(arr[0], np.float32(12345.0))

    def test_a_resumed_window_is_rebased_and_clipped(self):
        pm = make_pm(40)
        self.write(pm, start=100)                                   # frames 100..139 absolute
        target, report = self.restore(112, 28)                      # resumed at 112: 28 frames left
        self.assertEqual(min(target._temporal_faces), 0)
        self.assertEqual(max(target._temporal_faces), 27)
        for rel in (0, 5, 27):
            self.assertEqual(deep_equal(pm._temporal_faces[rel + 12], target._temporal_faces[rel]), '', msg=rel)
        self.assertEqual(deep_equal(pm._track_assignments[12], target._track_assignments[0]), '')
        self.assertEqual(target._shot_boundaries, {17 - 12})        # the cut at rel 17 is now rel 5
        self.assertEqual(target._temporal_covered, 28)
        self.assertEqual(sorted(target.stage2_frame_plan), sorted(f - 12 for f in pm.stage2_frame_plan if f >= 12))

    def test_a_cut_before_the_resumed_window_is_dropped(self):
        pm = make_pm(40, cuts=(3, 17))
        self.write(pm, start=0)
        target, _ = self.restore(10, 30)
        self.assertEqual(target._shot_boundaries, {7})

    def test_a_none_shot_list_stays_none(self):
        pm = make_pm(10)
        pm._shot_boundaries = None
        self.write(pm, start=0)
        target, _ = self.restore(0, 10)
        self.assertIsNone(target._shot_boundaries)

    def test_the_plan_rebuilt_from_restored_faces_is_the_plan_that_was_sealed(self):
        pm = make_pm(30)
        self.write(pm, start=0)
        target, _ = self.restore(5, 25)
        self.assertEqual(sc.serialize_plan(target.stage2_frame_plan, 5),
                         sc.plan_window(sc.serialize_plan(pm.stage2_frame_plan, 0), 5, 30, 0))


class Stage2Blocks(Tmp):
    """The JSON columns (the nested pose block, mixed scalar types) are written and parsed in blocks of
    `JSON_BLOCK_ROWS` faces. With the real 2048 these tests would never cross a block boundary, so the block
    size is forced down to 7."""

    def setUp(self):
        super().setUp()
        from unittest import mock
        patcher = mock.patch.object(sc, 'JSON_BLOCK_ROWS', 7)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_clip_spanning_many_blocks_round_trips_exactly(self):
        pm = make_pm(60)                         # ~100 faces -> 15 blocks per JSON column
        self.write(pm, start=0)
        target, _ = self.restore(0, 60)
        self.assertEqual(deep_equal(pm._temporal_faces, target._temporal_faces), '')

    def test_every_window_start_gives_the_same_faces_as_the_original(self):
        """Includes starts in the middle of a block, on a block edge, and one past it."""
        pm = make_pm(60)
        self.write(pm, start=1000)
        for shift in (0, 1, 3, 6, 7, 8, 13, 14, 29, 41, 59):
            count = 60 - shift
            target, _ = self.restore(1000 + shift, count)
            for rel in range(count):
                a, b = pm._temporal_faces.get(rel + shift) or [], target._temporal_faces.get(rel) or []
                self.assertEqual(deep_equal(a, b, 'faces[%d]' % rel), '', msg='shift %d' % shift)

    def test_blocks_are_the_unit_on_disk(self):
        pm = make_pm(60)
        self.write(pm, start=0)
        with zipfile.ZipFile(self.path) as z:
            blocks = [n for n in z.namelist() if '.j' in n and n.endswith(('.j0', '.j1', '.j14'))]
        self.assertTrue(blocks, 'no block members found')

    def test_only_the_blocks_a_late_window_touches_are_parsed(self):
        pm = make_pm(60)
        self.write(pm, start=0)
        parsed = []
        original = sc.SidecarReader.json

        def spy(self_, name):
            parsed.append(name)
            return original(self_, name)

        from unittest import mock
        with mock.patch.object(sc.SidecarReader, 'json', spy):
            self.restore(55, 5)
        face_blocks = [p for p in parsed if p.startswith('faces/')]
        total_blocks = 0
        with zipfile.ZipFile(self.path) as z:
            total_blocks = len([n for n in z.namelist() if n.startswith('faces/') and n.endswith(tuple('.j%d' % i for i in range(40)))])
        self.assertTrue(face_blocks)
        self.assertLess(len(face_blocks), total_blocks, 'a window at the end must not parse the whole clip')


class Stage2Refusals(Tmp):
    def assertRefused(self, start, count, ranges=None, fragment=''):
        target = SimpleNamespace(_temporal_faces='UNTOUCHED')
        with self.assertRaises(sc.Unserializable) as cm:
            sc.restore_stage2(self.path, target, new_start=start, new_count=count,
                              current_ranges_abs=ranges, build_plan=build_stage2_plan)
        self.assertIn(fragment, str(cm.exception))
        self.assertEqual(target._temporal_faces, 'UNTOUCHED', 'a refused restore must not touch the pipeline')

    def test_a_window_that_starts_before_the_cache_is_refused(self):
        self.write(make_pm(40), start=100)
        self.assertRefused(90, 50, fragment='covers frames 100-139')

    def test_a_window_that_runs_past_the_cache_is_refused(self):
        self.write(make_pm(40), start=100)
        self.assertRefused(110, 40, fragment='covers frames')

    def test_a_different_stage1_result_is_refused(self):
        self.write(make_pm(40), start=0, ranges=[[0, 20]])
        self.assertRefused(0, 40, ranges=[[0, 21]], fragment='Stage 1 ranges differ')

    def test_stage1_that_did_not_run_is_not_stage1_that_found_nothing(self):
        self.write(make_pm(40), start=0, ranges=None)
        self.assertRefused(0, 40, ranges=[], fragment='Stage 1 ranges differ')
        self.write(make_pm(40), start=0, ranges=[])
        self.assertRefused(0, 40, ranges=None, fragment='Stage 1 ranges differ')

    def test_ranges_are_compared_inside_the_resumed_window_only(self):
        """Saved against [0,12] and [20,38]; the run resumes at 25 and its Stage 1 is clipped to [25,38]."""
        self.write(make_pm(40), start=0, ranges=[[0, 12], [20, 38]])
        target, _ = self.restore(25, 15, ranges=[[25, 38]])
        self.assertEqual(min(target._temporal_faces), 0)
        self.assertRefused(25, 15, ranges=[[26, 38]], fragment='Stage 1 ranges differ')

    def test_a_window_with_no_cached_face_is_refused_so_the_pre_pass_can_fall_back(self):
        pm = make_pm(40)
        for fr in range(20, 40):
            pm._temporal_faces[fr] = []
        pm.stage2_frame_plan, _ = build_stage2_plan(pm._temporal_faces, pm._track_source_map)
        self.write(pm, start=0)
        self.assertRefused(25, 15, fragment='no cached face')

    def test_a_scan_that_stopped_short_is_not_written(self):
        pm = make_pm(40)
        pm._temporal_covered = 25
        with self.assertRaises(sc.Unserializable):
            sc.write_stage2(self.path, pm, frame_start=0, frame_count=40, ranges_abs=None,
                            plan=pm.stage2_frame_plan)
        self.assertFalse(os.path.exists(self.path))

    def test_a_pre_pass_with_no_faces_is_not_written(self):
        pm = make_pm(5)
        pm._temporal_faces = {}
        with self.assertRaises(sc.Unserializable):
            self.write(pm, start=0, count=5)

    def test_an_unserializable_face_field_leaves_no_file_and_no_temp(self):
        pm = make_pm(6)
        pm._temporal_faces[2][0]['_odd'] = object()
        with self.assertRaises(sc.Unserializable):
            self.write(pm, start=0)
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_good_cache_survives_a_failed_rewrite(self):
        pm = make_pm(10)
        self.write(pm, start=0)
        before = open(self.path, 'rb').read()
        bad = make_pm(10, seed=1)
        bad._temporal_faces[3][0]['_odd'] = object()
        with self.assertRaises(sc.Unserializable):
            self.write(bad, start=0)
        self.assertEqual(open(self.path, 'rb').read(), before)
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith('.tmp')], [])

    def test_a_plan_that_disagrees_with_the_faces_is_refused(self):
        pm = make_pm(20)
        self.write(pm, start=0)
        # rewrite the sidecar with a tampered plan but untouched faces
        tampered = {f: v for f, v in pm.stage2_frame_plan.items() if f != 3}
        sc.write_stage2(self.path, pm, frame_start=0, frame_count=20, ranges_abs=None, plan=tampered)
        self.assertRefused(0, 20, fragment='plan rebuilt from the restored faces differs')

    def test_a_pickle_is_never_loaded(self):
        # allow_pickle=False: a member that is an object array must fail, not execute
        pm = make_pm(6)
        self.write(pm, start=0)
        evil = os.path.join(self.dir, 'evil.npz')
        with zipfile.ZipFile(self.path) as zin, zipfile.ZipFile(evil, 'w') as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == 'faces/frame.npy':
                    import io
                    buf = io.BytesIO()
                    np.save(buf, np.array([{'x': 1}] * 3, dtype=object), allow_pickle=True)
                    data = buf.getvalue()
                zout.writestr(item.filename, data)
        with self.assertRaises(Exception) as cm:
            sc.restore_stage2(evil, SimpleNamespace(), new_start=0, new_count=6, current_ranges_abs=None,
                              build_plan=build_stage2_plan)
        self.assertIn('pickle', str(cm.exception).lower())


# ── ranges ───────────────────────────────────────────────────────────────────────────────────────────
class Ranges(unittest.TestCase):
    def test_rebase_clips_and_builds_the_mask(self):
        ranges, mask = sc.rebase_ranges([[100, 130], [150, 160], [200, 220]], 120, 50)
        self.assertEqual(ranges, [(0, 10), (30, 40)])
        self.assertEqual(int(mask.sum()), 22)
        self.assertTrue(mask[0] and mask[10] and not mask[11] and mask[30] and mask[40] and not mask[41])

    def test_empty_ranges_give_an_all_false_mask(self):
        ranges, mask = sc.rebase_ranges([], 0, 30)
        self.assertEqual((ranges, int(mask.sum()), len(mask)), ([], 0, 30))

    def test_a_range_wholly_before_the_window_is_dropped(self):
        ranges, mask = sc.rebase_ranges([[0, 50]], 60, 40)
        self.assertEqual((ranges, int(mask.sum())), ([], 0))

    def test_digest_is_order_and_value_sensitive_and_none_is_distinct(self):
        a, b = sc.ranges_digest([[0, 5], [9, 12]]), sc.ranges_digest([[0, 5], [9, 13]])
        self.assertNotEqual(a, b)
        self.assertNotEqual(sc.ranges_digest([]), sc.ranges_digest(None))

    def test_covers_window(self):
        self.assertTrue(sc.covers_window(0, 100, 0, 100))
        self.assertTrue(sc.covers_window(0, 100, 40, 60))
        self.assertFalse(sc.covers_window(10, 100, 0, 50))
        self.assertFalse(sc.covers_window(0, 100, 50, 51))

    def test_mask_equals_what_build_active_ranges_produced(self):
        from roop.sparse_range_scan import build_active_ranges
        ranges, mask = build_active_ranges([10, 12, 80, 200], 300, 6)
        back, mask2 = sc.rebase_ranges(sc.ranges_to_abs(ranges, 0), 0, 300)
        self.assertEqual([tuple(r) for r in ranges], back)
        self.assertTrue(np.array_equal(mask, mask2))


# ── the project record ───────────────────────────────────────────────────────────────────────────────
class RecordCase(unittest.TestCase):
    """A real project record in a temp projects dir."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='stagecache-proj-')
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        old = pc.PROJECTS_DIR
        pc.PROJECTS_DIR = self.dir
        self.addCleanup(setattr, pc, 'PROJECTS_DIR', old)
        self.media = os.path.join(self.dir, 'target.mp4')
        open(self.media, 'wb').write(b'0' * 4096)
        self.src = os.path.join(self.dir, 'src.jpg')
        open(self.src, 'wb').write(b'1' * 512)
        cfg = SimpleNamespace(provider='tensorrt', trt_precision='mixed', swap_model='realswap',
                              output_path=self.dir, hardware={}, hardware_signature='', output_video_format='mp4',
                              output_video_codec='', video_quality=14, output_method='File', output_template='')
        payload = {'swap_model': 'realswap'}
        src = pc.file_identity(self.src)
        src['source_id'] = 's1'
        self.record = pc.new_project(
            job_id=None, name='t', payload=payload, sources=[src], target=pc.file_identity(self.media),
            frame_start=0, frame_end=300, output={'directory': self.dir}, cfg=cfg,
            target_faces=[{'bbox': [1, 2, 3, 4], 'embedding': [0.5] * 4}])
        self.pid = self.record['id']
        self.live = {'app_version': 'abc123', 'env': 'e1'}


class ProjectRecord(RecordCase):
    def test_a_stage_is_recorded_and_found_again(self):
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 300, 'active_ranges': [[0, 50]],
                                             'mode': 'sparse_auto_range', 'digest': 'd'}, live=self.live)
        found = pc.usable_stage_cache(self.pid, self.live)
        self.assertEqual(found['reasons'], [])
        self.assertEqual(found['stage1']['active_ranges'], [[0, 50]])
        self.assertIsNone(found['stage2'])

    def test_a_pending_digest_means_no_key_and_nothing_stored(self):
        record = pc.load(self.pid)
        record['inputs']['target']['sha256'] = ''
        record['inputs']['target']['sha256_pending'] = True
        pc.save(record)
        self.assertEqual(pc.stage_cache_key(pc.load(self.pid), self.live), '')
        self.assertIsNone(pc.record_stage(self.pid, 'stage1', {'x': 1}, live=self.live))

    def test_the_key_moves_with_every_input_it_depends_on(self):
        base = pc.stage_cache_key(pc.load(self.pid), self.live)
        self.assertTrue(base)

        def changed(mutate, live=None):
            record = json.loads(json.dumps(pc.load(self.pid)))
            mutate(record)
            return pc.stage_cache_key(record, live or self.live) != base

        self.assertTrue(changed(lambda r: r['inputs']['target'].update(sha256='f' * 64)), 'target bytes')
        self.assertTrue(changed(lambda r: r['inputs']['sources'][0].update(sha256='e' * 64)), 'a source')
        self.assertTrue(changed(lambda r: r['inputs'].update(frame_start=5)), 'trim start')
        self.assertTrue(changed(lambda r: r['inputs'].update(frame_end=299)), 'trim end')
        self.assertTrue(changed(lambda r: r['inputs']['target_faces'][0].update(embedding=[0.4] * 4)), 'target face')
        self.assertTrue(changed(lambda r: r['settings'].update(fingerprint='zz')), 'settings')
        self.assertTrue(changed(lambda r: r['runtime']['models'].update(swap_model='inswapper')), 'models')
        self.assertTrue(changed(lambda r: None, live={'app_version': 'other', 'env': 'e1'}), 'code version')
        self.assertTrue(changed(lambda r: None, live={'app_version': 'abc123', 'env': 'e2'}), 'ROOP_* env')
        self.assertFalse(changed(lambda r: r.update(name='renamed')), 'a rename must not invalidate')

    def test_a_cache_under_another_key_is_refused_with_a_reason(self):
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 300, 'active_ranges': [], 'mode': 'x',
                                             'digest': 'd'}, live=self.live)
        found = pc.usable_stage_cache(self.pid, {'app_version': 'newer', 'env': 'e1'})
        self.assertIsNone(found['stage1'])
        self.assertTrue(any('changed since the cache was written' in r for r in found['reasons']), found['reasons'])
        self.assertIn('application version or a ROOP_* environment variable', ' '.join(found['reasons']))

    def test_the_refusal_names_every_component_that_changed(self):
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 300, 'active_ranges': [], 'mode': 'x',
                                             'digest': 'd'}, live=self.live)
        record = pc.load(self.pid)
        record['inputs']['target']['sha256'] = 'f' * 64
        record['inputs']['frame_start'] = 7
        record['settings']['fingerprint'] = 'changed'
        pc.save(record)
        found = pc.usable_stage_cache(self.pid, self.live)
        text = ' '.join(found['reasons'])
        for part in ('the target video', 'the trim points', 'the processing settings'):
            self.assertIn(part, text)
        for part in ('a source file', 'the captured target faces', 'the model configuration'):
            self.assertNotIn(part, text, 'a component that did not change must not be blamed')

    def test_the_basis_holds_digests_and_fingerprints_only(self):
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 1, 'active_ranges': [], 'mode': 'x',
                                             'digest': 'd'}, live=self.live)
        basis = pc.load(self.pid)['stage_cache']['basis']
        self.assertNotIn(self.media, json.dumps(basis))
        self.assertNotIn(self.src, json.dumps(basis))

    def test_recording_under_a_new_key_drops_what_was_stored_under_the_old_one(self):
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 300, 'active_ranges': [], 'mode': 'x',
                                             'digest': 'd'}, live=self.live)
        side = pc.stage_sidecar_path(self.pid, 'stage2')
        open(side, 'wb').write(b'stale')
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 300, 'active_ranges': [[1, 2]],
                                             'mode': 'x', 'digest': 'd2'}, live={'app_version': 'b', 'env': 'e1'})
        self.assertFalse(os.path.exists(side), 'a sidecar from another key must be deleted')
        self.assertEqual(pc.load(self.pid)['stage_cache']['stage1']['active_ranges'], [[1, 2]])

    def test_a_new_stage1_invalidates_stage2(self):
        pm = make_pm(10)
        entry = sc.write_stage2(pc.stage_sidecar_path(self.pid, 'stage2'), pm, frame_start=0, frame_count=10,
                                ranges_abs=None, plan=pm.stage2_frame_plan)
        pc.record_stage(self.pid, 'stage2', entry, live=self.live)
        self.assertIsNotNone(pc.usable_stage_cache(self.pid, self.live)['stage2'])
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 10, 'active_ranges': [[0, 9]],
                                             'mode': 'x', 'digest': 'd'}, live=self.live)
        self.assertIsNone(pc.usable_stage_cache(self.pid, self.live)['stage2'])
        self.assertFalse(os.path.exists(pc.stage_sidecar_path(self.pid, 'stage2')))

    def _with_sidecar(self):
        pm = make_pm(10)
        entry = sc.write_stage2(pc.stage_sidecar_path(self.pid, 'stage2'), pm, frame_start=0, frame_count=10,
                                ranges_abs=None, plan=pm.stage2_frame_plan)
        pc.record_stage(self.pid, 'stage2', entry, live=self.live)
        return entry

    def test_a_sidecar_that_was_edited_by_one_byte_is_refused(self):
        self._with_sidecar()
        path = pc.stage_sidecar_path(self.pid, 'stage2')
        data = bytearray(open(path, 'rb').read())
        data[len(data) // 2] ^= 0x01
        open(path, 'wb').write(bytes(data))
        found = pc.usable_stage_cache(self.pid, self.live)
        self.assertIsNone(found['stage2'])
        self.assertTrue(any('digest differs' in r for r in found['reasons']), found['reasons'])

    def test_a_truncated_or_missing_sidecar_is_refused(self):
        self._with_sidecar()
        path = pc.stage_sidecar_path(self.pid, 'stage2')
        data = open(path, 'rb').read()
        open(path, 'wb').write(data[:-10])
        self.assertTrue(any('size differs' in r for r in pc.usable_stage_cache(self.pid, self.live)['reasons']))
        os.remove(path)
        self.assertTrue(any('missing' in r for r in pc.usable_stage_cache(self.pid, self.live)['reasons']))

    def test_completing_a_project_deletes_its_cache(self):
        self._with_sidecar()
        path = pc.stage_sidecar_path(self.pid, 'stage2')
        self.assertTrue(os.path.exists(path))
        pc.update_state(self.pid, 'COMPLETED')
        self.assertFalse(os.path.exists(path))
        self.assertNotIn('stage_cache', pc.load(self.pid))

    def test_a_failed_or_paused_project_keeps_its_cache(self):
        self._with_sidecar()
        for state in ('PAUSED', 'FAILED', 'INTERRUPTED', 'RECOVERABLE'):
            pc.update_state(self.pid, state, 'x')
            self.assertTrue(os.path.exists(pc.stage_sidecar_path(self.pid, 'stage2')), state)

    def test_segment_commits_and_stage_records_do_not_overwrite_each_other(self):
        """Both read-modify-write the same JSON from different threads; without the lock one wins."""
        import threading
        errors = []

        def commits():
            try:
                for i in range(40):
                    pc.update_checkpoint(self.pid, safe_frame=i, next_frame=i, segments=[{'frames': i}])
            except Exception as exc:       # noqa: BLE001
                errors.append(exc)

        def stages():
            try:
                for i in range(40):
                    pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 9, 'active_ranges': [[0, i]],
                                                         'mode': 'x', 'digest': str(i)}, live=self.live)
            except Exception as exc:       # noqa: BLE001
                errors.append(exc)

        ts = [threading.Thread(target=commits), threading.Thread(target=stages)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errors, [])
        record = pc.load(self.pid)
        self.assertEqual(record['checkpoint']['safe_frame'], 39, 'the last segment commit was lost')
        self.assertEqual(record['stage_cache']['stage1']['active_ranges'], [[0, 39]], 'the last stage record was lost')


# ── the adapter ──────────────────────────────────────────────────────────────────────────────────────
class Adapter(RecordCase):
    def adapter(self):
        logs = []
        cache = sck.ProjectStageCache(self.pid, app_version='abc123', log=lambda *a, **k: logs.append(' '.join(map(str, a))))
        cache._live = self.live            # the test pins the live identity
        return cache, logs

    def test_stage1_round_trip_with_a_resumed_window(self):
        cache, _ = self.adapter()
        ranges = [(0, 40), (100, 150)]
        stats = {'mode': 'sparse_auto_range', 'active_frames': 92, 'total_frames': 300}
        self.assertTrue(cache.save_stage1(frame_start=0, frame_count=300, ranges=ranges, stats=stats))
        fresh, logs = self.adapter()
        got = fresh.restore_stage1(frame_start=30, frame_count=270)
        self.assertIsNotNone(got, logs)
        r, mask, st = got
        self.assertEqual(r, [(0, 10), (70, 120)])
        self.assertEqual(int(mask.sum()), 11 + 51)
        self.assertEqual((st['total_frames'], st['active_frames'], st['restored']), (270, 62, True))

    def test_a_failed_scan_is_not_frozen(self):
        cache, logs = self.adapter()
        self.assertFalse(cache.save_stage1(frame_start=0, frame_count=300, ranges=[(0, 299)],
                                           stats={'mode': 'fail_safe_full'}))
        self.assertIsNone(pc.load(self.pid).get('stage_cache'))
        self.assertTrue(any('not a settled result' in line for line in logs))

    def test_an_empty_scan_result_is_a_result(self):
        cache, _ = self.adapter()
        self.assertTrue(cache.save_stage1(frame_start=0, frame_count=300, ranges=[], stats={'mode': 'no_target_found'}))
        r, mask, st = self.adapter()[0].restore_stage1(frame_start=0, frame_count=300)
        self.assertEqual((r, int(mask.sum())), ([], 0))

    def test_a_window_the_stage1_record_does_not_cover_is_refused(self):
        cache, _ = self.adapter()
        cache.save_stage1(frame_start=50, frame_count=100, ranges=[(0, 99)], stats={'mode': 'sparse_auto_range'})
        fresh, logs = self.adapter()
        self.assertIsNone(fresh.restore_stage1(frame_start=0, frame_count=100))
        self.assertTrue(any('covers frames 50-149' in line for line in logs), logs)

    def test_stage2_round_trip_and_the_stage1_ranges_it_was_built_against(self):
        cache, _ = self.adapter()
        ranges = [(0, 30)]
        cache.save_stage1(frame_start=0, frame_count=40, ranges=ranges, stats={'mode': 'sparse_auto_range'})
        pm = make_pm(40)
        pm._active_ranges = ranges
        self.assertTrue(cache.save_stage2(pm, frame_start=0, frame_count=40))

        fresh, logs = self.adapter()
        target = SimpleNamespace(_active_ranges=[(0, 30)])
        self.assertTrue(fresh.restore_stage2(target, frame_start=0, frame_count=40), logs)
        self.assertEqual(deep_equal(pm._temporal_faces, target._temporal_faces), '')

        other, logs = self.adapter()
        different = SimpleNamespace(_active_ranges=[(0, 31)])        # a re-run Stage 1 that came out differently
        self.assertFalse(other.restore_stage2(different, frame_start=0, frame_count=40))
        self.assertTrue(any('Stage 1 ranges differ' in line for line in logs), logs)

    def test_a_real_resume_clips_stage1_and_stage2_must_still_be_accepted(self):
        """The case that matters: the run is resumed at frame 25, Stage 1 comes back CLIPPED to the new
        window (the first range is gone, the second starts at the window edge), and Stage 2 - which was sealed
        against the UNCLIPPED ranges - has to agree with it, or a resume would throw Stage 2 away exactly when
        it is useful."""
        cache, _ = self.adapter()
        ranges = [(0, 12), (20, 38)]
        cache.save_stage1(frame_start=0, frame_count=40, ranges=ranges, stats={'mode': 'sparse_auto_range'})
        pm = make_pm(40)
        pm._active_ranges = ranges
        self.assertTrue(cache.save_stage2(pm, frame_start=0, frame_count=40))

        fresh, logs = self.adapter()
        got = fresh.restore_stage1(frame_start=25, frame_count=15)
        self.assertEqual(got[0], [(0, 13)], 'precondition: stage 1 is clipped to the window')
        target = SimpleNamespace(_active_ranges=got[0])
        self.assertTrue(fresh.restore_stage2(target, frame_start=25, frame_count=15), logs)
        self.assertEqual(min(target._temporal_faces), 0)

    def test_stage2_without_stage1_is_keyed_on_no_ranges(self):
        cache, _ = self.adapter()
        pm = make_pm(12)
        self.assertTrue(cache.save_stage2(pm, frame_start=0, frame_count=12))
        fresh, _ = self.adapter()
        self.assertTrue(fresh.restore_stage2(SimpleNamespace(_active_ranges=None), frame_start=0, frame_count=12))
        stale, _ = self.adapter()
        self.assertFalse(stale.restore_stage2(SimpleNamespace(_active_ranges=[(0, 5)]), frame_start=0, frame_count=12))

    def test_nothing_is_ever_raised(self):
        cache, logs = self.adapter()
        bad = make_pm(6)
        bad._temporal_faces[1][0]['_odd'] = object()
        self.assertFalse(cache.save_stage2(bad, frame_start=0, frame_count=6))
        self.assertTrue(any('not recorded' in line for line in logs))
        self.assertFalse(os.path.exists(pc.stage_sidecar_path(self.pid, 'stage2')))

    def test_a_project_with_no_cache_says_why_once(self):
        cache, logs = self.adapter()
        self.assertIsNone(cache.restore_stage1(frame_start=0, frame_count=10))
        self.assertFalse(cache.restore_stage2(SimpleNamespace(_active_ranges=None), frame_start=0, frame_count=10))
        self.assertEqual(sum('no Stage 1/2 result was recorded' in line for line in logs), 1)

    def test_variables_the_application_writes_while_running_do_not_change_the_key(self):
        """A pause-and-resume inside one app session must see the key the first run had: the runtime
        optimizer exports ROOP_RUNTIME_PROFILE_* in the middle of a run and they stay in os.environ."""
        before = sck.live_identity('v1')
        for name, value in (('ROOP_RUNTIME_PROFILE_KEY', 'abc'), ('ROOP_RUNTIME_PROFILE_JSON', '{"x":1}'),
                            ('ROOP_NVDEC', '0'), ('ROOP_BATCH_SWAP_MAX', '4')):
            os.environ[name] = value
            self.addCleanup(os.environ.pop, name, None)
        self.assertEqual(before, sck.live_identity('v1'))

    def test_a_real_knob_still_changes_the_key(self):
        before = sck.live_identity('v1')
        os.environ['ROOP_TEMPORAL_GAP'] = '3'
        self.addCleanup(os.environ.pop, 'ROOP_TEMPORAL_GAP', None)
        self.assertNotEqual(before['env'], sck.live_identity('v1')['env'])

    def test_live_identity_follows_the_roop_environment(self):
        a = sck.live_identity('v1')
        os.environ['ROOP_TEMPORAL_GAP'] = '99'
        self.addCleanup(os.environ.pop, 'ROOP_TEMPORAL_GAP', None)
        self.assertNotEqual(a['env'], sck.live_identity('v1')['env'])
        self.assertNotEqual(a, sck.live_identity('v2'))


class Telemetry(unittest.TestCase):
    def test_restored_is_its_own_state_and_reads_complete(self):
        from roop import stage_telemetry as st
        st.begin_run(100)
        st.enter(st.STAGE_2_ACTIVE_PREPASS, 50)
        st.restored(st.STAGE_2_ACTIVE_PREPASS)
        s = st.snapshot()['stages'][st.STAGE_2_ACTIVE_PREPASS]
        self.assertEqual((s['state'], s['stage_progress_pct'], s['eta_seconds']), ('RESTORED', 100.0, 0.0))
        self.assertIsNone(st.busy_stage(), 'a restored stage is not holding the GPU')
        st.reset()


class ApiWiring(unittest.TestCase):
    """`api._run_swap` is too large to drive in a unit test; these pin the three sites that decide whether a
    project's run gets the cache, in the repo's usual source-level style (mutation-checked in the session
    log: removing any one makes its test fail)."""

    @classmethod
    def setUpClass(cls):
        cls.src = open(os.path.join(APP, 'api.py'), encoding='utf-8').read()

    def test_a_project_run_installs_the_cache_for_its_own_project(self):
        i = self.src.index('roop_globals._stage_cache = _stage_checkpoint.ProjectStageCache(')
        block = self.src[self.src.rfind('if project_id:', 0, i):i + 200]
        self.assertIn('project_id, app_version=_get_git_version()', block)

    def test_a_run_without_a_project_gets_no_cache(self):
        text = self.src.replace('\r\n', '\n')
        self.assertIn('        else:\n            roop_globals._stage_cache = None', text)

    def test_the_cache_is_cleared_when_the_run_ends(self):
        tail = self.src[self.src.index('roop_globals._checkpoint_segment_callback = None'):]
        self.assertIn('roop_globals._stage_cache = None', tail[:300])

    def test_the_pipeline_reads_the_cache_from_globals_and_only_when_present(self):
        batch = open(os.path.join(APP, 'roop', 'procmgr_batch.py'), encoding='utf-8').read()
        self.assertIn("getattr(roop.globals, '_stage_cache', None)", batch)
        self.assertIn('_stage_cache.restore_stage1(', batch)
        self.assertIn('_stage_cache.restore_stage2(', batch)
        self.assertIn('_stage_cache.save_stage1(', batch)
        self.assertIn('_stage_cache.save_stage2(', batch)


if __name__ == '__main__':
    unittest.main()
