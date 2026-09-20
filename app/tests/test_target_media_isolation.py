"""Stage 11 regression tests for target-media context isolation.

These tests use fake ProcessEntry-like objects and the real target selection,
removal, preview, and queue-request boundaries.  No detector or GPU work is
needed: the assertions are about which identity-bearing context is active.
"""

import os
import sys
import types
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import api  # noqa: E402
import api_state as state  # noqa: E402
import project_checkpoint  # noqa: E402
import roop.globals as roop_globals  # noqa: E402
import ui.globals as ui_globals  # noqa: E402
from target_media_state import TargetMediaContextStore  # noqa: E402


class _Entry:
    def __init__(self, name, media_id=None):
        self.filename = name
        self.media_id = media_id
        self.startframe = 0
        self.endframe = 1
        self.total_frames = 1
        self.fps = 0


class TargetMediaIsolation(unittest.TestCase):
    def setUp(self):
        self.old_entries = list(api.list_files_process)
        self.old_faces = list(roop_globals.TARGET_FACES)
        self.old_groups = list(roop_globals.TARGET_FACE_GROUP)
        self.old_names = dict(getattr(roop_globals, 'TARGET_FACE_NAMES', {}) or {})
        self.old_thumbs = list(ui_globals.ui_target_thumbs)
        self.old_selected = state.selected_target_index
        self.old_active = getattr(state, 'active_target_media_id', None)
        self.old_selected_face = getattr(state, 'selected_target_face_index', 0)
        self.old_mapping = getattr(state, 'active_target_source_mapping', {})
        self.old_cfg = roop_globals.CFG
        self.old_refresh = api._refresh_target_frames

        api.list_files_process.clear()
        api._target_contexts.clear()
        roop_globals.TARGET_FACES.clear()
        roop_globals.TARGET_FACE_GROUP.clear()
        roop_globals.TARGET_FACE_NAMES.clear()
        ui_globals.ui_target_thumbs.clear()
        state.selected_target_index = 0
        state.active_target_media_id = None
        state.selected_target_face_index = 0
        state.active_target_source_mapping = {}
        # Media loading is orthogonal to these tests; keep selection deterministic.
        api._refresh_target_frames = lambda _idx: None

    def tearDown(self):
        api._refresh_target_frames = self.old_refresh
        api.list_files_process.clear()
        api.list_files_process.extend(self.old_entries)
        api._target_contexts.clear()
        roop_globals.TARGET_FACES.clear()
        roop_globals.TARGET_FACES.extend(self.old_faces)
        roop_globals.TARGET_FACE_GROUP.clear()
        roop_globals.TARGET_FACE_GROUP.extend(self.old_groups)
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.TARGET_FACE_NAMES.update(self.old_names)
        ui_globals.ui_target_thumbs.clear()
        ui_globals.ui_target_thumbs.extend(self.old_thumbs)
        state.selected_target_index = self.old_selected
        state.active_target_media_id = self.old_active
        state.selected_target_face_index = self.old_selected_face
        state.active_target_source_mapping = self.old_mapping
        roop_globals.CFG = self.old_cfg

    def _add(self, name):
        entry = _Entry(name)
        api.list_files_process.append(entry)
        return api._ensure_target_media_id(entry)

    def _configure(self, index, label, mapping=None):
        api._activate_target_media(index=index, refresh=False)
        roop_globals.TARGET_FACES[:] = [label]
        roop_globals.TARGET_FACE_GROUP[:] = [0]
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.TARGET_FACE_NAMES[0] = label
        ui_globals.ui_target_thumbs[:] = [np.zeros((2, 2, 3), dtype=np.uint8)]
        state.selected_target_face_index = 0
        state.active_target_source_mapping = mapping or {}
        api._save_active_target_context_locked()

    def _active_labels(self):
        return list(roop_globals.TARGET_FACES)

    def test_each_media_starts_without_another_media_identity(self):
        a = self._add('duplicate.mp4')
        b = self._add('duplicate.mp4')
        self.assertNotEqual(a, b, 'duplicate filenames must still receive distinct ids')
        self._configure(0, 'Harjot')
        response = api.target_select({'index': 1, 'target_media_id': b})
        self.assertEqual(response['target_media_id'], b)
        self.assertEqual(self._active_labels(), [], 'B must not inherit A identity')

    def test_switch_a_b_a_restores_exact_context(self):
        a = self._add('a.mp4')
        b = self._add('b.mp4')
        self._configure(0, 'Harjot', {'0': 2})
        api.target_select({'target_media_id': b})
        self._configure(1, 'Other', {'0': 1})
        api.target_select({'target_media_id': a})
        self.assertEqual(self._active_labels(), ['Harjot'])
        self.assertEqual(roop_globals.TARGET_FACE_NAMES, {0: 'Harjot'})
        self.assertEqual(state.active_target_source_mapping, {'0': 2})

    def test_removing_first_and_middle_targets_preserves_survivors(self):
        a = self._add('a.mp4')
        b = self._add('b.mp4')
        c = self._add('c.mp4')
        self._configure(0, 'A')
        self._configure(1, 'B')
        self._configure(2, 'C')

        api.target_remove({'target_media_id': b})
        api.target_select({'target_media_id': a})
        self.assertEqual(self._active_labels(), ['A'])
        api.target_select({'target_media_id': c})
        self.assertEqual(self._active_labels(), ['C'])

        api.target_remove({'target_media_id': a})
        api.target_select({'target_media_id': c})
        self.assertEqual(self._active_labels(), ['C'])
        self.assertEqual(
            {entry.media_id for entry in api.list_files_process}, {c})

    def test_removed_slot_gets_fresh_context_for_new_media(self):
        a = self._add('same.mp4')
        old_b = self._add('same.mp4')
        self._configure(1, 'B')
        api.target_remove({'target_media_id': old_b})
        new_b = self._add('same.mp4')
        self.assertNotEqual(old_b, new_b)
        api.target_select({'target_media_id': new_b})
        self.assertEqual(self._active_labels(), [])
        self.assertEqual(api._target_contexts.has(old_b), False)
        self.assertEqual(api._target_contexts.load(new_b).target_faces, [])
        # Keep the first target in the fixture meaningful as well.
        api.target_select({'target_media_id': a})
        self.assertEqual(self._active_labels(), [])

    def test_failed_target_load_does_not_replace_active_context(self):
        self._add('a.mp4')
        b = self._add('missing.mp4')
        self._configure(0, 'Harjot')

        def fail(index):
            if index == 1:
                raise OSError('unreadable target')

        api._refresh_target_frames = fail
        response = api.target_select({'target_media_id': b})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self._active_labels(), ['Harjot'])
        self.assertEqual(state.active_target_media_id, api.list_files_process[0].media_id)

    def test_preview_and_final_request_carry_the_active_media_id(self):
        a = self._add('a.mp4')
        b = self._add('b.mp4')
        self._configure(0, 'A')
        api.target_select({'target_media_id': b})
        self._configure(1, 'B')

        request = api._canonical_processing_request(
            {'target_index': 1, 'target_media_id': b, 'face_mapping': {'0': 7}},
            target_media_index=1, target_media_id=b)
        self.assertEqual(request['target_media_id'], b)

        old_ready = api._configuration_ready
        old_frame = api.get_image_frame
        old_cfg = roop_globals.CFG
        old_faces = None
        import roop.face_util as face_util
        old_faces = face_util.get_all_faces
        try:
            api._configuration_ready = lambda: True
            roop_globals.CFG = types.SimpleNamespace(
                default_det_size=False, face_detector_size='320x320',
                face_detector_threshold=0.5, face_detector_nms=0.4,
                refine_landmarks=False, swap_model_mask_strength=0.0,
                jaw_reshape=False, jaw_reshape_strength=0.5,
                detail_transfer_strength=0.0, mask_edge_mode='gaussian',
                boundary_illumination_strength=0.0,
                identity_detail_strength=0.0, expression_restore_strength=0.0,
                expression_restore_region='all', rescue_small_faces=False,
                detector_engine='scrfd', detector_scale_pyramid='auto',
                max_face_distance=0.75, blend_ratio=0.8,
                no_face_action='Keep', max_threads=1,
                color_transfer_mode='rct', sam2_model_size='tiny',
                autorotate_faces=False,
            )
            api.get_image_frame = lambda _path: np.zeros((4, 4, 3), dtype=np.uint8)
            face_util.get_all_faces = lambda _frame: []
            preview = api.preview({'target_media_id': b, 'index': 1, 'frame': 1})
            self.assertEqual(preview['target_media_id'], b)
            self.assertEqual(self._active_labels(), ['B'])
        finally:
            api._configuration_ready = old_ready
            api.get_image_frame = old_frame
            roop_globals.CFG = old_cfg
            face_util.get_all_faces = old_faces

    def test_final_processing_request_is_bound_to_selected_media_context(self):
        self._add('a.mp4')
        b = self._add('b.mp4')
        self._configure(0, 'A')
        api.target_select({'target_media_id': b})
        self._configure(1, 'B')

        old_ready = api._configuration_ready
        old_unavailable = api._unavailable_target_entries
        old_create = api._create_processing_project
        old_start = api._start_existing_project
        old_diagnostic = api._selection_diagnostic_for_mode
        old_sources = list(roop_globals.INPUT_FACESETS)
        captured = {}
        try:
            api._configuration_ready = lambda: True
            api._unavailable_target_entries = lambda _payload: []
            api._selection_diagnostic_for_mode = lambda *_args: None
            api._create_processing_project = lambda payload, job_id=None: {'id': 'stage11-job'}
            def start(project_id, payload):
                captured.update(payload)
                return {'accepted': True, 'project_id': project_id}
            api._start_existing_project = start
            roop_globals.INPUT_FACESETS[:] = [object()]
            result = api.trigger_swap({
                'target_index': 1,
                'target_media_id': b,
                'face_mapping': [3],
            })
            self.assertEqual(result['project_id'], 'stage11-job')
            self.assertEqual(captured['target_media_id'], b)
            self.assertEqual(captured['normalized_request']['target_media_id'], b)
            self.assertEqual(self._active_labels(), ['B'])
        finally:
            api._configuration_ready = old_ready
            api._unavailable_target_entries = old_unavailable
            api._create_processing_project = old_create
            api._start_existing_project = old_start
            api._selection_diagnostic_for_mode = old_diagnostic
            roop_globals.INPUT_FACESETS[:] = old_sources

    def test_context_store_removes_identity_and_does_not_reuse_id(self):
        store = TargetMediaContextStore()
        first = types.SimpleNamespace(filename='same.mp4')
        second = types.SimpleNamespace(filename='same.mp4')
        first_id = store.ensure_media_id(first)
        second_id = store.ensure_media_id(second)
        self.assertNotEqual(first_id, second_id)
        store.save(first_id, target_faces=['Harjot'], source_mapping={'0': 1})
        store.remove(first_id)
        self.assertEqual(store.load(first_id).target_faces, [])
        self.assertEqual(store.load(second_id).target_faces, [])

    def test_project_reload_keeps_target_context_associated_with_media_id(self):
        media_id = self._add('duplicate.mp4')
        record = project_checkpoint.new_project(
            job_id=None,
            name='stage11-context-reload',
            payload={'target_media_id': media_id},
            sources=[],
            target={'name': 'duplicate.mp4', 'path': 'duplicate.mp4',
                    'target_media_id': media_id},
            frame_start=0,
            frame_end=1,
            output={'directory': '', 'format': 'mp4'},
            cfg=None,
            target_context={
                'target_media_id': media_id,
                'target_face_names': {'0': 'Harjot'},
                'selected_target_face_index': 0,
                'face_mapping': {'0': 4},
            },
            app_version='stage11-test',
        )
        try:
            restored = project_checkpoint.load(record['id'])
            restored_context = (restored.get('inputs') or {}).get('target_context') or {}
            self.assertEqual(restored_context['target_media_id'], media_id)
            self.assertEqual(restored_context['target_face_names'], {'0': 'Harjot'})
            self.assertEqual(restored_context['face_mapping'], {'0': 4})
        finally:
            try:
                os.unlink(project_checkpoint.project_path(record['id']))
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()
