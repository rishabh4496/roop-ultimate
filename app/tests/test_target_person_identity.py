"""Stage 13 regressions for stable target-person identity and mapping."""

import os
import sys
import types
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import api  # noqa: E402
import api_state as state  # noqa: E402
import roop.globals as roop_globals  # noqa: E402
import ui.globals as ui_globals  # noqa: E402


class _Entry:
    def __init__(self, name):
        self.filename = name
        self.media_id = None
        self.startframe = 0
        self.endframe = 1
        self.total_frames = 1
        self.fps = 0


class _Source:
    def __init__(self, source_id):
        self._source_id = source_id
        self._source_path = source_id
        self.faces = []


class TargetPersonIdentityTests(unittest.TestCase):
    def setUp(self):
        self.old_entries = list(api.list_files_process)
        self.old_faces = list(roop_globals.TARGET_FACES)
        self.old_groups = list(roop_globals.TARGET_FACE_GROUP)
        self.old_people = list(roop_globals.TARGET_FACE_PERSON_IDS)
        self.old_refs = list(roop_globals.TARGET_REFERENCE_FACE_IDS)
        self.old_names = dict(roop_globals.TARGET_FACE_NAMES)
        self.old_thumbs = list(ui_globals.ui_target_thumbs)
        self.old_sources = list(roop_globals.INPUT_FACESETS)
        self.old_active = state.active_target_media_id
        self.old_selected = state.selected_target_index
        self.old_face = state.selected_target_face_index
        self.old_mapping = state.active_target_source_mapping
        self.old_person_mapping = state.active_target_person_source_mapping
        self.old_person_names = state.active_target_person_names
        self.old_person = state.selected_target_person_id
        self.old_ref = state.selected_reference_face_id
        self.old_refresh = api._refresh_target_frames

        api.list_files_process.clear()
        api._target_contexts.clear()
        roop_globals.TARGET_FACES.clear()
        roop_globals.TARGET_FACE_GROUP.clear()
        roop_globals.TARGET_FACE_PERSON_IDS.clear()
        roop_globals.TARGET_REFERENCE_FACE_IDS.clear()
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.INPUT_FACESETS[:] = [_Source("source-a"), _Source("source-b"), _Source("source-c")]
        ui_globals.ui_target_thumbs.clear()
        state.active_target_media_id = None
        state.selected_target_index = 0
        state.selected_target_face_index = 0
        state.active_target_source_mapping = {}
        state.active_target_person_source_mapping = {}
        state.active_target_person_names = {}
        state.selected_target_person_id = None
        state.selected_reference_face_id = None
        api._refresh_target_frames = lambda _idx: None

    def tearDown(self):
        api._refresh_target_frames = self.old_refresh
        api.list_files_process.clear()
        api.list_files_process.extend(self.old_entries)
        api._target_contexts.clear()
        roop_globals.TARGET_FACES[:] = self.old_faces
        roop_globals.TARGET_FACE_GROUP[:] = self.old_groups
        roop_globals.TARGET_FACE_PERSON_IDS[:] = self.old_people
        roop_globals.TARGET_REFERENCE_FACE_IDS[:] = self.old_refs
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.TARGET_FACE_NAMES.update(self.old_names)
        roop_globals.INPUT_FACESETS[:] = self.old_sources
        ui_globals.ui_target_thumbs[:] = self.old_thumbs
        state.active_target_media_id = self.old_active
        state.selected_target_index = self.old_selected
        state.selected_target_face_index = self.old_face
        state.active_target_source_mapping = self.old_mapping
        state.active_target_person_source_mapping = self.old_person_mapping
        state.active_target_person_names = self.old_person_names
        state.selected_target_person_id = self.old_person
        state.selected_reference_face_id = self.old_ref

    def _add(self, name):
        entry = _Entry(name)
        api.list_files_process.append(entry)
        return api._ensure_target_media_id(entry)

    def _configure(self, index, labels, mapping):
        api._activate_target_media(index=index, refresh=False)
        roop_globals.TARGET_FACES[:] = list(labels)
        roop_globals.TARGET_FACE_GROUP[:] = list(range(len(labels)))
        roop_globals.TARGET_FACE_PERSON_IDS.clear()
        roop_globals.TARGET_REFERENCE_FACE_IDS.clear()
        roop_globals.TARGET_FACE_NAMES.clear()
        ui_globals.ui_target_thumbs[:] = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in labels]
        state.active_target_source_mapping = {}
        state.active_target_person_source_mapping = dict(mapping)
        state.selected_target_face_index = 0
        api._save_active_target_context_locked()
        return api._target_person_records()

    def test_person_ids_are_opaque_and_mapping_survives_delete(self):
        a = self._add("a.mp4")
        records = self._configure(0, ["A", "B"], {})
        a_id, b_id = [record["target_person_id"] for record in records]
        self.assertNotIn(a_id, {"0", "1"})
        state.active_target_person_source_mapping = {a_id: "source-a", b_id: "source-b"}
        api._save_active_target_context_locked()
        api.target_remove_face({"target_media_id": a, "face_index": 0})
        self.assertEqual(roop_globals.TARGET_FACE_PERSON_IDS, [b_id])
        self.assertEqual(state.active_target_person_source_mapping, {b_id: "source-b"})

    def test_media_contexts_have_independent_people_and_duplicate_names(self):
        a = self._add("duplicate.mp4")
        b = self._add("duplicate.mp4")
        self._configure(0, ["person_a"], {})
        a_id = api._target_person_records()[0]["target_person_id"]
        state.active_target_person_source_mapping = {a_id: "source-a"}
        api._save_active_target_context_locked()
        api.target_select({"target_media_id": b})
        self.assertEqual(api._target_person_records(), [])
        self.assertNotEqual(a, b)

    def test_three_people_delete_middle_keeps_a_and_c_bindings(self):
        media = self._add("a.mp4")
        records = self._configure(0, ["A", "B", "C"], {})
        ids = [r["target_person_id"] for r in records]
        state.active_target_person_source_mapping = {
            ids[0]: "source-a", ids[1]: "source-b", ids[2]: "source-c",
        }
        api._save_active_target_context_locked()
        api.target_remove_face({"target_media_id": media, "face_index": 1})
        self.assertEqual(state.active_target_person_source_mapping,
                         {ids[0]: "source-a", ids[2]: "source-c"})

    def test_remove_first_target_does_not_rebind_remaining_media(self):
        a = self._add("a.mp4")
        b = self._add("b.mp4")
        self._configure(0, ["A"], {})
        a_person = api._target_person_records()[0]["target_person_id"]
        state.active_target_person_source_mapping = {a_person: "source-a"}
        api._save_active_target_context_locked()
        records_b = self._configure(1, ["B"], {})
        b_person = records_b[0]["target_person_id"]
        state.active_target_person_source_mapping = {b_person: "source-b"}
        api._save_active_target_context_locked()
        api.target_remove({"target_media_id": a})
        api.target_select({"target_media_id": b})
        self.assertEqual(api._target_person_records()[0]["target_person_id"], b_person)
        self.assertEqual(state.active_target_person_source_mapping, {b_person: "source-b"})

    def test_replacement_media_gets_a_fresh_context(self):
        self._add("same.mp4")
        old = self._add("same.mp4")
        records = self._configure(1, ["Old"], {})
        old_person = records[0]["target_person_id"]
        api.target_remove({"target_media_id": old})
        new = self._add("same.mp4")
        api.target_select({"target_media_id": new})
        self.assertEqual(api._target_person_records(), [])
        self.assertFalse(api._target_contexts.has(old))
        self.assertNotEqual(old_person, None)

    def test_add_and_remove_angle_keep_the_same_person_id(self):
        media = self._add("a.mp4")
        records = self._configure(0, ["A"], {})
        person_id = records[0]["target_person_id"]
        # Exercise the same stable mutation used by add-angle without invoking
        # detector/model work.
        roop_globals.TARGET_FACES.append("A-profile")
        roop_globals.TARGET_FACE_GROUP.append(0)
        roop_globals.TARGET_FACE_PERSON_IDS.append(person_id)
        roop_globals.TARGET_REFERENCE_FACE_IDS.append(api.new_target_reference_face_id())
        api._save_active_target_context_locked()
        self.assertEqual(roop_globals.TARGET_FACE_PERSON_IDS, [person_id, person_id])
        api.target_remove_face({"target_media_id": media, "face_index": 1})
        self.assertEqual(roop_globals.TARGET_FACE_PERSON_IDS, [person_id])

    def test_reorder_does_not_change_identity_or_mapping(self):
        self._add("a.mp4")
        records = self._configure(0, ["A", "B"], {})
        ids = [r["target_person_id"] for r in records]
        state.active_target_person_source_mapping = {ids[0]: "source-a", ids[1]: "source-b"}
        roop_globals.TARGET_FACES[:] = ["B", "A"]
        roop_globals.TARGET_FACE_GROUP[:] = [1, 0]
        roop_globals.TARGET_FACE_PERSON_IDS[:] = [ids[1], ids[0]]
        api._save_active_target_context_locked()
        self.assertEqual(state.active_target_person_source_mapping,
                         {ids[0]: "source-a", ids[1]: "source-b"})

    def test_autocluster_explicitly_invalidates_old_mapping(self):
        self._add("a.mp4")
        records = self._configure(0, [types.SimpleNamespace(embedding=np.ones(3)),
                                     types.SimpleNamespace(embedding=np.zeros(3))], {})
        ids = [r["target_person_id"] for r in records]
        state.active_target_person_source_mapping = {ids[0]: "source-a", ids[1]: "source-b"}
        response = api.target_autocluster({"target_media_id": state.active_target_media_id})
        self.assertTrue(response["mappings_invalidated"])
        self.assertEqual(state.active_target_person_source_mapping, {})
        self.assertTrue(all(pid not in ids for pid in roop_globals.TARGET_FACE_PERSON_IDS))

    def test_invalid_person_and_source_fail_structured(self):
        self._add("a.mp4")
        records = self._configure(0, ["A"], {})
        invalid_person = api.target_name({
            "target_media_id": state.active_target_media_id,
            "target_person_id": "tp_missing", "name": "wrong",
        })
        self.assertEqual(invalid_person.status_code, 422)
        invalid_source = api.target_context({
            "target_media_id": state.active_target_media_id,
            "target_person_source_mapping": {
                records[0]["target_person_id"]: "source-missing",
            },
        })
        self.assertEqual(invalid_source.status_code, 422)
        self.assertEqual(state.active_target_person_source_mapping, {})

    def test_selected_person_uses_stable_id_in_canonical_request(self):
        self._add("a.mp4")
        records = self._configure(0, ["A", "B"], {})
        ids = [r["target_person_id"] for r in records]
        state.active_target_person_source_mapping = {ids[0]: "source-a", ids[1]: "source-b"}
        request = api._canonical_processing_request({
            "target_media_id": state.active_target_media_id,
            "detection": "Selected face",
            "target_person_source_mapping": {ids[0]: "source-a", ids[1]: "source-b"},
            "selection_state": {"selection_mode": "selected", "person_id": ids[1]},
            "selected_source_id": "source-b",
        }, target_media_id=state.active_target_media_id)
        self.assertEqual(request["selection_state"]["person_id"], ids[1])
        self.assertEqual(request["target_person_source_mapping"][ids[1]], "source-b")
        self.assertEqual(request["face_mapping"], [0, 1])

    def test_invalid_reference_face_id_is_structured_and_not_redirected(self):
        self._add("a.mp4")
        self._configure(0, ["A"], {})
        request = api._canonical_processing_request({
            "target_media_id": state.active_target_media_id,
            "detection": "Selected face",
            "selection_state": {
                "selection_mode": "selected",
                "person_id": api._target_person_records()[0]["target_person_id"],
                "target_reference_face_id": "tr_missing",
            },
        }, target_media_id=state.active_target_media_id)
        self.assertFalse(request["selection_state"]["valid"])
        self.assertEqual(request["selection_state"]["diagnostic"],
                         "invalid_reference_face_id")
        self.assertEqual(request["selection_state"]["person_ids"], [])

    def test_preview_and_final_canonical_requests_match_after_reorder(self):
        media = self._add("a.mp4")
        records = self._configure(0, ["A", "B"], {})
        ids = [r["target_person_id"] for r in records]
        mapping = {ids[0]: "source-a", ids[1]: "source-b"}
        roop_globals.TARGET_FACES[:] = ["B", "A"]
        roop_globals.TARGET_FACE_GROUP[:] = [1, 0]
        roop_globals.TARGET_FACE_PERSON_IDS[:] = [ids[1], ids[0]]
        state.active_target_person_source_mapping = mapping
        api._save_active_target_context_locked()
        payload = {
            "target_media_id": media,
            "detection": "Selected people",
            "target_person_source_mapping": mapping,
            "selection_state": {"selection_mode": "multi_person", "person_ids": ids},
        }
        preview_request = api._canonical_processing_request(payload, target_media_id=media)
        final_request = api._canonical_processing_request(payload, target_media_id=media)
        self.assertEqual(preview_request["target_person_source_mapping"],
                         final_request["target_person_source_mapping"])
        self.assertEqual(preview_request["face_mapping"], final_request["face_mapping"])


if __name__ == "__main__":
    unittest.main()
