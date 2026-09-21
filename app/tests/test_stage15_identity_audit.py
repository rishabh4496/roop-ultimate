"""Stage 15 audit regressions: faceset pipeline and identity routing.

Three defects found by the Stage 15 audit, each pinned here:

  B-1  In single-person "Selected face" mode the source slot handed to
       ProcessMgr was re-resolved from the gallery-HIGHLIGHTED source instead
       of the selected person's mapping, so P mapped to source B while source
       A was highlighted was swapped with A (when another person mapped there)
       or refused.
  A-1  An archive with no PNG members / no detectable face loaded as nothing,
       and a corrupt archive was swallowed by /api/source/add into a 200.
  A-2  A project record did not carry the source id the person->source
       mapping is keyed by; restoring a single-image source recreated it under
       a different id and the mapping silently skipped.
"""

import os
import sys
import tempfile
import types
import unittest
import zipfile

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import api  # noqa: E402
import roop.globals as roop_globals  # noqa: E402
import ui.globals as ui_globals  # noqa: E402
import source_gallery  # noqa: E402
from roop.processing_request import normalize_processing_request  # noqa: E402
from roop.selected_routing import compute_selected_assignment  # noqa: E402


class _Source:
    def __init__(self, source_id, path=None):
        self._source_id = source_id
        self._source_path = path or source_id
        self.faces = []


COMMON = dict(
    target_groups=[0, 1], source_count=3,
    current_source_ids=["source-a", "source-b", "source-c"],
    target_person_ids=["tp_p1", "tp_p2"], request_id="b1",
)


def _selected(person, mapping, highlighted):
    return {
        "detection": "Selected face",
        "selection_state": {"selection_mode": "selected", "person_id": person},
        "target_person_source_mapping": mapping,
        "selected_source_id": highlighted,
        "target_media_id": "m",
    }


class SelectedPersonOwnsItsSource(unittest.TestCase):
    """B-1."""

    def test_highlighted_source_does_not_redirect_the_selected_person(self):
        # P1 -> source-b, P2 -> source-a, gallery highlight on source-a.
        request = normalize_processing_request(
            _selected("tp_p1", {"tp_p1": "source-b", "tp_p2": "source-a"}, "source-a"),
            **COMMON)
        self.assertEqual(request["source_index_mapping"], [1, 0])
        # The mapped-list slot for P1 is rank 0 (source-b), not P2's slot.
        self.assertEqual(request["source_index"], 0)

    def test_selected_person_with_a_mapping_is_not_refused_when_nobody_maps_the_highlight(self):
        request = normalize_processing_request(
            _selected("tp_p1", {"tp_p1": "source-b"}, "source-a"), **COMMON)
        self.assertEqual(request["source_index_mapping"], [1, -1])
        self.assertEqual(request["source_index"], 0)

    def test_selected_person_without_a_mapping_is_still_refused(self):
        request = normalize_processing_request(
            _selected("tp_p2", {"tp_p1": "source-b"}, "source-b"), **COMMON)
        self.assertEqual(request["source_index_mapping"], [1, -1])
        self.assertEqual(request["source_index"], -1)

    def test_second_person_selected_uses_its_own_rank(self):
        request = normalize_processing_request(
            _selected("tp_p2", {"tp_p1": "source-b", "tp_p2": "source-c"}, "source-b"),
            **COMMON)
        self.assertEqual(request["source_index"], 1)

    def test_common_ui_path_is_unchanged(self):
        # No explicit dropdown: the UI maps the selected person to the
        # highlighted source, so rank == old resolution.
        request = normalize_processing_request(
            _selected("tp_p2", {"tp_p2": "source-c"}, "source-c"), **COMMON)
        self.assertEqual(request["source_index"], 1)

    def test_processmgr_single_person_path_reads_the_slot(self):
        """The routing decision, given that slot, swaps the selected person
        with ITS source (the mapped list is person-ordered)."""
        request = normalize_processing_request(
            _selected("tp_p1", {"tp_p1": "source-b", "tp_p2": "source-a"}, "source-a"),
            **COMMON)
        mapped = [f"faceset:{request['source_index_mapping'][0]}",
                  f"faceset:{request['source_index_mapping'][1]}"]
        refs = ["angle-p1", "angle-p2"]
        faces = ["face-x"]
        assign = compute_selected_assignment(
            faces, refs, request["target_groups"], {0}, request["source_index"],
            len(mapped), 0.6,
            identity_match=lambda r, f: (0, 0.1), unreliable=lambda f: False)
        self.assertEqual(assign.pending, [(0, 0)])
        self.assertEqual(mapped[assign.pending[0][0]], "faceset:1")  # source-b


class StableIdSelectionReachesProcessMgr(unittest.TestCase):
    """The real-file acceptance finding: [MATCH] persons=0 on every frame.

    ProcessOptions re-normalized the stable-id selection without its id
    universe, so ``tp_...`` parsed as a missing rank, the structured failure
    was preserved by ProcessMgr's second normalization, and no person was ever
    selected -- preview and render both swapped nothing while reporting Done.
    """

    def _request(self):
        return normalize_processing_request(
            _selected("tp_p1", {"tp_p1": "source-b"}, "source-b"), **COMMON)

    def test_process_options_keep_the_selection_valid(self):
        from roop.ProcessOptions import ProcessOptions
        request = self._request()
        options = ProcessOptions([], 0.6, 0.8, request["swap_mode"], request["source_index"],
                                 "", None, 1, 256, False, False,
                                 selection_state=request["selection_state"],
                                 processing_request=request)
        self.assertTrue(options.selection_state["valid"], options.selection_state)
        self.assertEqual(options.selection_state["person_ids"], ["tp_p1"])

    def test_processmgr_selects_the_person_group(self):
        from roop.ProcessOptions import ProcessOptions
        from roop.target_selection import resolve_processing_selection
        request = self._request()
        options = ProcessOptions([], 0.6, 0.8, request["swap_mode"], request["source_index"],
                                 "", None, 1, 256, False, False,
                                 selection_state=request["selection_state"],
                                 processing_request=request)
        groups, selection, selected = resolve_processing_selection(options, 2)
        self.assertEqual(groups, [0, 1])
        self.assertTrue(selection["valid"])
        self.assertEqual(selected, {0})
        # ...and the routing decision then swaps a matching face.
        assign = compute_selected_assignment(
            ["face"], ["ref-p1", "ref-p2"], groups, selected, request["source_index"], 2, 0.6,
            identity_match=lambda r, f: (0, 0.2), unreliable=lambda f: False)
        self.assertEqual(assign.pending, [(0, 0)])

    def test_normalizer_without_a_universe_accepts_stable_ids(self):
        from roop.target_selection import normalize_target_selection
        sel = normalize_target_selection({"selection_mode": "selected", "person_id": "tp_z"})
        self.assertTrue(sel["valid"])
        self.assertEqual(sel["person_ids"], ["tp_z"])
        # Legacy ranks are still ranks.
        sel = normalize_target_selection({"selection_mode": "selected", "person_id": 1}, person_count=2)
        self.assertTrue(sel["valid"])
        self.assertEqual(sel["person_ids"], [1])
        sel = normalize_target_selection({"selection_mode": "selected", "person_id": 5}, person_count=2)
        self.assertFalse(sel["valid"])
        # An explicit universe still rejects an unknown stable id.
        sel = normalize_target_selection({"selection_mode": "selected", "person_id": "tp_z"},
                                         target_person_ids=["tp_a"])
        self.assertFalse(sel["valid"])

    def test_second_person_selected_does_not_select_the_first(self):
        from roop.ProcessOptions import ProcessOptions
        from roop.target_selection import resolve_processing_selection
        request = normalize_processing_request(
            _selected("tp_p2", {"tp_p2": "source-c"}, "source-c"), **COMMON)
        options = ProcessOptions([], 0.6, 0.8, "selected", request["source_index"], "", None, 1, 256,
                                 False, False, processing_request=request)
        _groups, _selection, selected = resolve_processing_selection(options, 2)
        self.assertEqual(selected, {1})


class FacesetErrorsAreStructured(unittest.TestCase):
    """A-1."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="stage15_fsz_")
        self.old_sources = list(roop_globals.INPUT_FACESETS)
        self.old_thumbs = list(ui_globals.ui_input_thumbs)
        self.old_cfg = roop_globals.CFG
        roop_globals.CFG = types.SimpleNamespace(
            mask_top=0, mask_bottom=0, mask_left=0, mask_right=0,
            face_mask_blend=0, mouth_mask_blend=0, mouth_top_scale=1,
            mouth_bottom_scale=1, mouth_left_scale=1, mouth_right_scale=1)
        roop_globals.INPUT_FACESETS.clear()
        ui_globals.ui_input_thumbs.clear()

    def tearDown(self):
        roop_globals.INPUT_FACESETS[:] = self.old_sources
        ui_globals.ui_input_thumbs[:] = self.old_thumbs
        roop_globals.CFG = self.old_cfg

    def test_mask_offsets_are_safe_before_configuration_is_ready(self):
        roop_globals.CFG = None
        self.assertEqual(
            source_gallery._mask_offsets_from_cfg(),
            [0.0, 0.0, 0.0, 0.0, 12.0, 10.0, 1.0, 1.0, 1.0, 1.0],
        )

    def _archive(self, name, members):
        path = os.path.join(self.tmp, name)
        with zipfile.ZipFile(path, "w") as zf:
            for member, data in members.items():
                zf.writestr(member, data)
        return path

    def test_archive_without_png_members_raises(self):
        path = self._archive("nothing.fsz", {"readme.txt": b"no members"})
        with self.assertRaises(ValueError) as ctx:
            source_gallery._ingest_faceset(path)
        self.assertIn("no PNG reference members", str(ctx.exception))
        self.assertEqual(roop_globals.INPUT_FACESETS, [])

    def test_archive_without_a_detectable_face_raises(self):
        import cv2
        ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
        path = self._archive("blank.fsz", {"0.png": buf.tobytes()})
        import roop.face_util as face_util
        old = face_util.extract_face_images
        face_util.extract_face_images = lambda *_a, **_k: []
        source_gallery.extract_face_images = face_util.extract_face_images
        try:
            with self.assertRaises(ValueError) as ctx:
                source_gallery._ingest_faceset(path)
        finally:
            face_util.extract_face_images = old
            source_gallery.extract_face_images = old
        self.assertIn("no detectable faces", str(ctx.exception))
        self.assertEqual(roop_globals.INPUT_FACESETS, [])
        self.assertEqual(ui_globals.ui_input_thumbs, [])

    def test_source_add_reports_a_corrupt_archive(self):
        path = os.path.join(self.tmp, "corrupt.fsz")
        with open(path, "wb") as fh:
            fh.write(b"not a zip" * 32)

        class _Upload:
            filename = "corrupt.fsz"
            file = open(path, "rb")

        old_save = api._save_upload
        api._save_upload = lambda f, **kw: path   # bypass the boundary: this tests ingestion
        try:
            payload = api.source_add([_Upload()])
        finally:
            api._save_upload = old_save
            _Upload.file.close()
        self.assertEqual(payload["faceset_count"], 0)
        self.assertEqual(payload["errors"][0]["error"], "invalid_faceset")
        self.assertIn("corrupt", payload["errors"][0]["message"])
        self.assertEqual(payload["errors"][0]["file"], "corrupt.fsz")

    def test_library_load_reports_the_reason(self):
        import routes_faceset
        path = os.path.join(self.tmp, "corrupt.fsz")
        with open(path, "wb") as fh:
            fh.write(b"garbage" * 32)
        old = routes_faceset._faceset_library_dir
        routes_faceset._faceset_library_dir = lambda: self.tmp
        try:
            res = routes_faceset.faceset_library_load({"filename": "corrupt.fsz"})
        finally:
            routes_faceset._faceset_library_dir = old
        self.assertEqual(res.status_code, 422)
        self.assertIn(b"invalid_faceset", res.body)


class ProjectSourceIdentitySurvivesRestore(unittest.TestCase):
    """A-2."""

    def setUp(self):
        self.old_sources = list(roop_globals.INPUT_FACESETS)
        self.tmp = tempfile.mkdtemp(prefix="stage15_project_")

    def tearDown(self):
        roop_globals.INPUT_FACESETS[:] = self.old_sources

    def test_project_sources_record_the_source_id(self):
        image = os.path.join(self.tmp, "two.png")
        with open(image, "wb") as fh:
            fh.write(b"\x89PNG fake")
        roop_globals.INPUT_FACESETS[:] = [
            _Source(f"{os.path.abspath(image)}#face-0", image),
            _Source(f"{os.path.abspath(image)}#face-1", image),
        ]
        sources = api._project_sources()
        self.assertEqual([s["source_id"] for s in sources],
                         [f"{os.path.abspath(image)}#face-0",
                          f"{os.path.abspath(image)}#face-1"])
        self.assertEqual(sources[0]["path"], os.path.abspath(image))

    def test_restore_reuses_the_recorded_id_for_image_sources(self):
        import routes_projects
        image = os.path.join(self.tmp, "two.png")
        with open(image, "wb") as fh:
            fh.write(b"\x89PNG fake")
        src = routes_projects.__dict__
        # Drive only the source-restore loop with a fake detector.
        record = {"inputs": {"sources": [
            {"path": image, "source_id": f"{os.path.abspath(image)}#face-1"}],
            "target": {}, "target_context": {}}}
        import source_gallery as sg
        import roop.face_util as face_util
        fake_faces = [[{"bbox": np.array([0, 0, 1, 1])}, np.zeros((2, 2, 3), np.uint8)],
                      [{"bbox": np.array([1, 1, 2, 2])}, np.zeros((2, 2, 3), np.uint8)]]
        old = face_util.extract_face_images
        face_util.extract_face_images = lambda *_a, **_k: fake_faces
        old_list = list(api.list_files_process)
        try:
            roop_globals.INPUT_FACESETS.clear()
            ui_globals.ui_input_thumbs.clear()
            try:
                routes_projects._load_into_runtime(record)
            except Exception:
                pass  # the target half of the restore has no media; only sources matter here
        finally:
            face_util.extract_face_images = old
            api.list_files_process[:] = old_list
        ids = [fs._source_id for fs in roop_globals.INPUT_FACESETS]
        self.assertEqual(ids, [f"{os.path.abspath(image)}#face-1"])
        del src


if __name__ == "__main__":
    unittest.main()
