import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import project_checkpoint
from roop import nle_interchange, project_io, project_render


class ProjectIoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="roop_project_io_")
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.png"
        self.target = self.root / "target.mp4"
        self.source.write_bytes(b"source")
        self.target.write_bytes(b"target")
        self.project = self.root / "session.roop"

    def tearDown(self):
        self.tmp.cleanup()

    def document(self):
        return {
            "format": project_io.PROJECT_FORMAT,
            "project_version": project_io.PROJECT_VERSION,
            "id": "session-1",
            "name": "Timeline",
            "media": {
                "sources": [project_io.media_reference(str(self.source), str(self.project), kind="source", asset_id="src-1")],
                "target": project_io.media_reference(str(self.target), str(self.project), kind="target", asset_id="tgt-1"),
            },
            "face_bank": {
                "embeddings": [{"data": {"embedding": [0.1, 0.2]}, "group": 0}],
                "clusters": [{"id": "person-1", "members": [0]}],
                "target_to_source": {"person-1": "src-1"},
            },
            "automation": {
                "keyframes": [{"frame": 0, "fidelity": 0.6}],
                "fidelity_ramps": [{"in": 0, "out": 30, "from": 0.4, "to": 0.8}],
                "mask_parameters": [{"frame": 10, "face_mask_blend": 12}],
                "in_out": {"in": 0, "out": 90},
            },
            "timeline": {
                "frame_start": 0,
                "frame_end": 90,
                "fps": 30,
                "scene_cuts": [30, 60],
                "face_segments": [{
                    "face_id": "person-1", "in": 12, "out": 42, "track": 1,
                    "alpha_asset": project_io.media_reference(str(self.source), str(self.project), kind="alpha")
                }],
            },
            "settings": {"selected_enhancer": "UltraMax"},
            "render": {"directory": str(self.root / "renders")},
        }

    def test_round_trip_preserves_portable_relative_reference(self):
        project_io.save_project(str(self.project), self.document())
        loaded = project_io.load_project(str(self.project))
        target = loaded["media"]["target"]
        self.assertEqual(project_io.resolve_media(target, str(self.project)), str(self.target))
        self.assertTrue(Path(project_io.journal_path(str(self.project))).is_file())
        self.assertEqual(loaded["face_bank"]["target_to_source"]["person-1"], "src-1")

    def test_journal_recovers_newer_snapshot(self):
        project_io.save_project(str(self.project), self.document(), journal=False)
        updated = self.document()
        updated["timeline"]["markers"] = [{"frame": 24, "name": "hero"}]
        # Deliberately append a journal snapshot without rewriting the main file.
        project_io.append_journal(str(self.project), {"op": "snapshot", "updated_at": 9999999999, "project": updated})
        recovered = project_io.load_project(str(self.project))
        self.assertEqual(recovered["timeline"]["markers"][0]["name"], "hero")

    def test_autosave_controller_writes_on_demand(self):
        value = self.document()
        controller = project_io.AutosaveController(str(self.project), lambda: value, interval=60)
        controller.start()
        controller.save_now()
        controller.stop(save=False)
        self.assertEqual(project_io.load_project(str(self.project), recover=False)["name"], "Timeline")

    def test_output_override_accepts_a_filename(self):
        project_id = "output-override-test"
        record = {"id": project_id, "output": {}}
        requested = self.root / "renders" / "output.mov"
        try:
            project_render._apply_output_override(record, str(requested))
            self.assertEqual(record["output"]["directory"], str(requested.parent))
            self.assertEqual(record["output"]["filename"], "output.mov")
        finally:
            try:
                os.remove(project_checkpoint.project_path(project_id))
            except FileNotFoundError:
                pass

    def test_render_project_reaches_the_worker(self):
        """`--project --render` died on its first statement after the loader:
        `import roop.globals as globals_` was followed by `globals.output_path`,
        i.e. the builtin `globals()` function, so no headless render ever started."""
        import types
        from unittest import mock
        import roop.globals as globals_

        started, events = [], []
        fake_api = types.SimpleNamespace(
            _progress={"processing": False, "error": ""},
            _start_existing_project=lambda pid, payload: (
                started.append((pid, payload)) or {"status": "started"}))
        fake_routes = types.SimpleNamespace(
            _load_into_runtime=lambda record: events.append("load"))
        machine = types.SimpleNamespace(
            execute_phase=lambda phase, fn, *args: events.append(("phase", phase.value, args)))
        cfg = object()
        record = {"id": "headless-1", "output": {"directory": str(self.root / "out"),
                                                 "filename": "clip.mp4"},
                  "settings": {"payload": {"swap_model": "hyperswap"}}}
        self.project.write_text(json.dumps(self.document()), encoding="utf-8")
        saved = (getattr(globals_, "output_path", None),
                 getattr(globals_, "_project_output_file", None))
        try:
            with mock.patch.dict(sys.modules, {"api": fake_api, "routes_projects": fake_routes}), \
                    mock.patch("roop.startup_state_machine.get_startup_state_machine",
                               return_value=machine), \
                    mock.patch.object(project_render, "_ensure_checkpoint", return_value=record):
                code = project_render.render_project(str(self.project), cfg=cfg)
            self.assertEqual(code, 0)
            # The provider phase the UI runs (ui/main.py) must run here too, with the
            # config, and BEFORE the project loads: otherwise the render stays on the
            # module default (CUDA, fp32) and silently writes original frames.
            self.assertEqual(events, [("phase", "MODEL_RUNTIME_INIT", (cfg,)), "load"])
            self.assertEqual(started, [("headless-1", {"swap_model": "hyperswap"})])
            self.assertEqual(globals_.output_path, str(self.root / "out"))
            self.assertEqual(globals_._project_output_file, "clip.mp4")
        finally:
            globals_.output_path, globals_._project_output_file = saved


class NleInterchangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="roop_nle_")
        self.root = Path(self.tmp.name)
        self.project = self.root / "session.roop"
        self.project.write_text("{}", encoding="utf-8")
        self.alpha = self.root / "face_rgba.mov"
        self.alpha.write_bytes(b"alpha")
        self.document = {
            "format": project_io.PROJECT_FORMAT,
            "project_version": 1,
            "name": "NLE Test",
            "media": {"target": project_io.media_reference(str(self.root / "target.mp4"), str(self.project), kind="target")},
            "timeline": {
                "frame_start": 0, "frame_end": 90, "fps": 30,
                "scene_cuts": [30, 60],
                "face_segments": [{"face_id": "A", "in": 10, "out": 40, "track": 1, "alpha_asset": str(self.alpha)}],
            },
        }
        (self.root / "target.mp4").write_bytes(b"target")

    def tearDown(self):
        self.tmp.cleanup()

    def test_fcpxml_contains_markers_and_face_lane(self):
        out = self.root / "timeline.fcpxml"
        nle_interchange.export_fcpxml(self.document, str(self.project), str(out))
        text = out.read_text(encoding="utf-8")
        self.assertIn("Scene Cut", text)
        self.assertIn("Face A", text)
        self.assertIn("alpha=straight", text)
        self.assertIn('lane="1"', text)

    def test_resolve_edl_emits_master_and_face_track(self):
        out = self.root / "timeline.edl"
        generated = nle_interchange.export_resolve_edl(self.document, str(self.project), str(out))
        self.assertEqual(len(generated), 2)
        self.assertIn("SCENE CUT", out.read_text(encoding="utf-8"))
        face_edl = Path(generated[1]).read_text(encoding="utf-8")
        self.assertIn("ALPHA: STRAIGHT", face_edl)
        self.assertIn("PRORES 4444", face_edl)


if __name__ == "__main__":
    unittest.main()
