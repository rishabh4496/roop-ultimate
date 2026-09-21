"""Stage 14: stale asynchronous state across preview, queue, render, session.

Every test here drives the REAL boundaries (api.preview, api.trigger_swap,
api._run_swap's request derivation, routes_queue._run_one, /api/target/context,
/api/state) with fake media entries and fake sources, and no detector, model
or GPU work.  What is asserted is identity: which target media, which target
person, which source identity and which mapping a request/job/render actually
carries, and whether a late or stale write can change it.

The client half of the same contract is exercised by
react-ui/.render-check/stage14-async-check.mjs, which replays the delayed
response races against the pure decision function the UI ships.
"""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import api  # noqa: E402
import api_state as state  # noqa: E402
import roop.globals as roop_globals  # noqa: E402
import routes_queue as q  # noqa: E402
import ui.globals as ui_globals  # noqa: E402
from roop.processing_selection import (  # noqa: E402
    apply_processing_selection,
    build_processing_selection,
    is_stale_version,
    selection_invalidation_reasons,
    selection_signature,
)
from roop.processing_request import normalize_processing_request, selection_log_line  # noqa: E402
from roop.target_selection import (  # noqa: E402
    normalize_target_selection,
    selection_diagnostic_for_mode,
)


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


def _fake_cfg():
    return types.SimpleNamespace(
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
        autorotate_faces=False, clear_output=False,
        selected_enhancer='None', output_method='File',
        video_swapping_method='In-Memory processing',
        subsample_upscale='256px', mask_engine='None', mask_clip_text='',
        keep_frames=False, wait_after_extraction=False, skip_audio=False,
        track_identities=False, vr_mode=False, upscale_after_swap=False,
        upscale_model_after='esrganx2', auto_thread_selection=False,
        output_video_codec='libx264', video_quality=14, memory_limit=0,
        restore_original_mouth=False, num_swap_steps=1, use_3d_recon=False,
        use_source_bank=False, use_frontalization=False,
        frontalization_threshold=30.0, swap_model='inswapper',
        stabilize_face=False, stabilize_method='one_euro',
        stabilize_min_cutoff=0.05, stabilize_beta=0.5, stabilize_enhancer=False,
        stabilize_enhancer_strength=0.5, stabilize_mask=False,
        stabilize_mask_strength=0.5, stabilize_landmarks=False,
        stabilize_hf_texture=False, stabilize_hf_texture_weight=0.15,
        temporal_detection=False, codeformer_fidelity=0.5,
        interp_after_swap='off', trt_precision='mixed', provider='cuda',
    )


class Stage14Base(unittest.TestCase):
    """Two targets (A, B), each with its own people; three sources."""

    def setUp(self):
        self.old_entries = list(api.list_files_process)
        self.old_faces = list(roop_globals.TARGET_FACES)
        self.old_groups = list(roop_globals.TARGET_FACE_GROUP)
        # These tests drive /api/swap; run them as an install that has accepted the
        # intended-use terms (the gate itself is tested in test_synthetic_label.py).
        _accepted = mock.patch('intended_use.acknowledged', return_value=True)
        _accepted.start(); self.addCleanup(_accepted.stop)
        self.old_people = list(roop_globals.TARGET_FACE_PERSON_IDS)
        self.old_refs = list(roop_globals.TARGET_REFERENCE_FACE_IDS)
        self.old_names = dict(roop_globals.TARGET_FACE_NAMES)
        self.old_thumbs = list(ui_globals.ui_target_thumbs)
        self.old_sources = list(roop_globals.INPUT_FACESETS)
        self.old_state = {k: getattr(state, k) for k in (
            "active_target_media_id", "selected_target_index",
            "selected_target_face_index", "active_target_source_mapping",
            "active_target_person_source_mapping", "active_target_person_names",
            "selected_target_person_id", "selected_reference_face_id",
            "selected_input_face_index")}
        self.old_refresh = api._refresh_target_frames
        self.old_versions = dict(api._selection_versions)
        self.old_cfg = roop_globals.CFG
        self.old_ready = api._configuration_ready
        self.old_frame = api.get_image_frame

        api.list_files_process.clear()
        api._target_contexts.clear()
        api._selection_versions.clear()
        roop_globals.TARGET_FACES.clear()
        roop_globals.TARGET_FACE_GROUP.clear()
        roop_globals.TARGET_FACE_PERSON_IDS.clear()
        roop_globals.TARGET_REFERENCE_FACE_IDS.clear()
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.INPUT_FACESETS[:] = [
            _Source("source-a"), _Source("source-b"), _Source("source-c")]
        ui_globals.ui_target_thumbs.clear()
        state.active_target_media_id = None
        state.selected_target_index = 0
        state.selected_target_face_index = 0
        state.active_target_source_mapping = {}
        state.active_target_person_source_mapping = {}
        state.active_target_person_names = {}
        state.selected_target_person_id = None
        state.selected_reference_face_id = None
        state.selected_input_face_index = 0
        api._refresh_target_frames = lambda _idx: None
        roop_globals.CFG = _fake_cfg()
        api._configuration_ready = lambda: True
        api.get_image_frame = lambda _path: np.zeros((4, 4, 3), dtype=np.uint8)

        self.a = self._add("a.mp4")
        self.b = self._add("b.mp4")
        self.a_people = self._configure(0, ["A1", "A2"], {})
        self.b_people = self._configure(1, ["B1", "B2"], {})
        api._activate_target_media(index=0, refresh=False)

    def tearDown(self):
        api._refresh_target_frames = self.old_refresh
        roop_globals.CFG = self.old_cfg
        api._configuration_ready = self.old_ready
        api.get_image_frame = self.old_frame
        api.list_files_process.clear()
        api.list_files_process.extend(self.old_entries)
        api._target_contexts.clear()
        api._selection_versions.clear()
        api._selection_versions.update(self.old_versions)
        roop_globals.TARGET_FACES[:] = self.old_faces
        roop_globals.TARGET_FACE_GROUP[:] = self.old_groups
        roop_globals.TARGET_FACE_PERSON_IDS[:] = self.old_people
        roop_globals.TARGET_REFERENCE_FACE_IDS[:] = self.old_refs
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.TARGET_FACE_NAMES.update(self.old_names)
        roop_globals.INPUT_FACESETS[:] = self.old_sources
        ui_globals.ui_target_thumbs[:] = self.old_thumbs
        for key, value in self.old_state.items():
            setattr(state, key, value)

    # ── fixture helpers ─────────────────────────────────────────────────
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
        return [r["target_person_id"] for r in api._target_person_records()]

    def _selection(self, media_id, person_id, source_id, version, mapping=None,
                   request_id=None, detection="Selected face"):
        """What the React client sends as `processing_selection`."""
        return {
            "schema": 1,
            "request_id": request_id or f"req-{version}",
            "selection_version": version,
            "mapping_version": "m",
            "target_media_id": media_id,
            "target_person_id": person_id,
            "target_person_ids": [person_id] if person_id else [],
            "target_reference_face_id": None,
            "source_identity_id": source_id,
            "detection_mode": detection,
            "target_person_source_mapping": mapping if mapping is not None
            else ({person_id: source_id} if person_id and source_id else {}),
            "selection_state": {
                "selection_mode": "selected" if detection == "Selected face" else "none",
                "person_id": person_id,
                "person_ids": [person_id] if person_id else [],
            },
        }

    def _preview(self, selection, frame=1, fake=False, **extra):
        body = {
            "target_media_id": selection["target_media_id"],
            "frame": frame, "fake_preview": fake,
            "processing_selection": selection,
            # Deliberately STALE flat fields: the canonical object must win.
            "detection": "All faces",
            "selection_state": {"selection_mode": "none"},
            "target_person_source_mapping": {},
            **extra,
        }
        return api.preview(body)


# ── Canonical selection ───────────────────────────────────────────────────
class CanonicalSelection(Stage14Base):
    def test_processing_selection_overrides_stale_flat_fields(self):
        a1 = self.a_people[0]
        selection = self._selection(self.a, a1, "source-b", 10)
        payload = {
            "detection": "All faces",                    # stale
            "selection_state": {"selection_mode": "none"},  # stale
            "target_person_source_mapping": {a1: "source-c"},  # stale
            "selected_source_id": "source-a",            # stale
            "processing_selection": selection,
        }
        request = api._canonical_processing_request(
            payload, target_media_index=0, target_media_id=self.a)
        self.assertEqual(request["swap_mode"], "selected")
        self.assertEqual(request["selection_state"]["person_id"], a1)
        self.assertEqual(request["target_person_source_mapping"], {a1: "source-b"})
        self.assertEqual(request["source_index_mapping"], [1, -1])
        self.assertEqual(request["selected_source_gallery_index"], 1)
        self.assertEqual(request["source_index"], 0)
        self.assertEqual(request["request_id"], "req-10")
        self.assertEqual(request["selection_version"], 10)
        echo = request["processing_selection"]
        self.assertEqual(echo["target_person_id"], a1)
        self.assertEqual(echo["source_identity_id"], "source-b")
        self.assertEqual(echo["target_media_id"], self.a)

    def test_legacy_flat_payload_yields_the_same_canonical_object(self):
        a1 = self.a_people[0]
        flat = {
            "detection": "Selected face",
            "selection_state": {"selection_mode": "selected", "person_id": a1},
            "target_person_source_mapping": {a1: "source-b"},
            "selected_source_id": "source-b",
            "target_media_id": self.a,
            "request_id": "legacy-1",
        }
        canonical = {"processing_selection": self._selection(
            self.a, a1, "source-b", None, request_id="legacy-1")}
        r_flat = api._canonical_processing_request(flat, target_media_index=0, target_media_id=self.a)
        r_can = api._canonical_processing_request(canonical, target_media_index=0, target_media_id=self.a)
        for key in ("swap_mode", "selection_state", "target_person_source_mapping",
                    "source_index_mapping", "source_index", "target_media_id"):
            self.assertEqual(r_flat[key], r_can[key], key)
        for key in ("target_media_id", "target_person_id", "source_identity_id",
                    "detection_mode", "target_person_source_mapping"):
            self.assertEqual(r_flat["processing_selection"][key],
                             r_can["processing_selection"][key], key)

    def test_log_line_carries_the_diagnostic_fields_and_no_faces(self):
        a1 = self.a_people[0]
        request = api._canonical_processing_request(
            {"processing_selection": self._selection(self.a, a1, "source-a", 3)},
            target_media_index=0, target_media_id=self.a)
        line = selection_log_line(request, "preview", preview_signature="sig123")
        for token in ("request=req-3", f"target_media_id={self.a}",
                      f"target_person_id={a1}", "source_identity_id=source-a",
                      "selection_version=3", "preview_signature=sig123"):
            self.assertIn(token, line)
        self.assertNotIn("embedding", line)
        self.assertLess(len(line), 600)

    def test_signature_depends_on_identity_frame_and_context_only(self):
        a1, a2 = self.a_people
        s1 = self._selection(self.a, a1, "source-a", 1, request_id="x")
        s2 = self._selection(self.a, a1, "source-a", 99, request_id="y")  # same identity
        s3 = self._selection(self.a, a2, "source-a", 1, request_id="x")   # other person
        s4 = self._selection(self.a, a1, "source-b", 1, request_id="x")   # other source
        self.assertEqual(selection_signature(s1, frame=1), selection_signature(s2, frame=1))
        self.assertNotEqual(selection_signature(s1, frame=1), selection_signature(s3, frame=1))
        self.assertNotEqual(selection_signature(s1, frame=1), selection_signature(s4, frame=1))
        self.assertNotEqual(selection_signature(s1, frame=1), selection_signature(s1, frame=2))
        self.assertNotEqual(selection_signature(s1, frame=1, context="c1"),
                            selection_signature(s1, frame=1, context="c2"))


# ── Preview: response identity ────────────────────────────────────────────
class PreviewResponses(Stage14Base):
    def test_preview_echoes_request_identity_on_plain_and_diagnostic_responses(self):
        a1 = self.a_people[0]
        plain = self._preview(self._selection(self.a, a1, "source-a", 5, request_id="p-1"))
        self.assertEqual(plain["request_id"], "p-1")
        self.assertEqual(plain["target_media_id"], self.a)
        self.assertEqual(plain["selection_version"], 5)
        self.assertEqual(plain["frame"], 1)
        self.assertEqual(plain["processing_selection"]["target_person_id"], a1)
        self.assertEqual(plain["processing_selection"]["source_identity_id"], "source-a")
        self.assertTrue(plain["preview_signature"])
        self.assertEqual(plain["diagnostics"]["preview_signature"], plain["preview_signature"])
        self.assertNotIn("embedding", str(plain["diagnostics"]))

        # A person that no longer exists is answered with a diagnostic AND the
        # same echo, so the client can still match it to its request.
        gone = self._preview(self._selection(self.a, "tp_gone", "source-a", 6, request_id="p-2"))
        self.assertEqual(gone["request_id"], "p-2")
        self.assertEqual(gone["selection_diagnostic"], "invalid_person_id")
        self.assertIsNone(gone["processing_selection"]["target_person_id"])

    def test_two_people_and_two_frames_yield_distinct_signatures(self):
        a1, a2 = self.a_people
        r1 = self._preview(self._selection(self.a, a1, "source-a", 1))
        r2 = self._preview(self._selection(self.a, a2, "source-a", 2))
        r3 = self._preview(self._selection(self.a, a1, "source-a", 3), frame=2)
        r4 = self._preview(self._selection(self.a, a1, "source-a", 4))
        self.assertNotEqual(r1["preview_signature"], r2["preview_signature"])
        self.assertNotEqual(r1["preview_signature"], r3["preview_signature"])
        self.assertEqual(r1["preview_signature"], r4["preview_signature"])

    def test_preview_swap_receives_the_canonical_selection(self):
        """Prove the code path: live_swap is handed the frozen selection."""
        a1 = self.a_people[0]
        seen = {}

        def fake_live_swap(frame, options, input_facesets=None):
            seen["request"] = options.processing_request
            seen["facesets"] = input_facesets
            return frame

        import roop.core as core
        with mock.patch.object(core, "live_swap", fake_live_swap), \
                mock.patch.object(core, "get_processing_plugins", lambda *_a, **_k: []):
            res = self._preview(self._selection(self.a, a1, "source-c", 7), fake=True)
        self.assertEqual(res["request_id"], "req-7")
        self.assertIn("request", seen, "live_swap never ran")
        self.assertEqual(seen["request"]["processing_selection"]["target_person_id"], a1)
        self.assertEqual(seen["request"]["source_index_mapping"], [2, -1])
        self.assertEqual(seen["request"]["selection_state"]["person_id"], a1)

    def test_preview_for_media_b_activates_b_and_reports_b(self):
        b1 = self.b_people[0]
        res = self._preview(self._selection(self.b, b1, "source-a", 8))
        self.assertEqual(res["target_media_id"], self.b)
        self.assertEqual(state.active_target_media_id, self.b)
        self.assertEqual(res["processing_selection"]["target_person_id"], b1)


# ── Stale writes ──────────────────────────────────────────────────────────
class StaleWrites(Stage14Base):
    def test_older_preview_cannot_overwrite_a_newer_mapping(self):
        """RACE: preview built with mapping A lands AFTER the user set mapping B."""
        a1 = self.a_people[0]
        newer = api.target_context({
            "target_media_id": self.a, "selection_version": 20,
            "face_mapping": {a1: "source-b"}, "target_person_source_mapping": {a1: "source-b"},
        })
        self.assertEqual(newer["target_person_source_mapping"], {a1: "source-b"})
        # The late preview carries version 10 and mapping source-a.
        self._preview(self._selection(self.a, a1, "source-a", 10, mapping={a1: "source-a"}),
                      target_person_source_mapping={a1: "source-a"})
        self.assertEqual(state.active_target_person_source_mapping, {a1: "source-b"})
        # A newer preview (version 30) is applied.
        self._preview(self._selection(self.a, a1, "source-c", 30, mapping={a1: "source-c"}),
                      target_person_source_mapping={a1: "source-c"})
        self.assertEqual(state.active_target_person_source_mapping, {a1: "source-c"})

    def test_older_person_selection_write_is_dropped(self):
        a1, a2 = self.a_people
        api.target_context({"target_media_id": self.a, "selection_version": 50,
                            "selected_target_person_id": a2})
        self.assertEqual(state.selected_target_person_id, a2)
        api.target_context({"target_media_id": self.a, "selection_version": 40,
                            "selected_target_person_id": a1})
        self.assertEqual(state.selected_target_person_id, a2, "a stale write won")

    def test_unversioned_writes_stay_accepted(self):
        a1, a2 = self.a_people
        api.target_context({"target_media_id": self.a, "selection_version": 50,
                            "selected_target_person_id": a2})
        api.target_context({"target_media_id": self.a, "selected_target_person_id": a1})
        self.assertEqual(state.selected_target_person_id, a1)

    def test_versions_are_per_target_media(self):
        a1 = self.a_people[0]
        b1 = self.b_people[0]
        api.target_context({"target_media_id": self.a, "selection_version": 100,
                            "selected_target_person_id": a1})
        # B has never been written at version 100; a lower version for B is fine.
        api.target_context({"target_media_id": self.b, "selection_version": 5,
                            "selected_target_person_id": b1})
        self.assertEqual(state.active_target_media_id, self.b)
        self.assertEqual(state.selected_target_person_id, b1)
        self.assertTrue(is_stale_version(4, 5))
        self.assertFalse(is_stale_version(5, 5))
        self.assertFalse(is_stale_version(None, 5))


# ── Direct render (/api/swap): frozen before the worker starts ────────────
class DirectRender(Stage14Base):
    def test_render_uses_the_selection_frozen_at_post_time_not_later_ui_state(self):
        """RACE 3: preview B shown -> start render for B -> UI switches to A."""
        b1, b2 = self.b_people
        captured = {}

        def fake_start(project_id, payload):
            captured["payload"] = payload
            return {"status": "started", "project_id": project_id}

        with mock.patch.object(api, "_unavailable_target_entries", lambda _p: []), \
                mock.patch.object(api, "_create_processing_project",
                                  lambda payload, job_id=None: {"id": "p14"}), \
                mock.patch.object(api, "_start_existing_project", fake_start):
            res = api.trigger_swap({
                "target_media_id": self.b,
                "processing_selection": self._selection(self.b, b2, "source-c", 70),
            })
        self.assertEqual(res["status"], "started")
        self.assertEqual(res["processing_selection"]["target_person_id"], b2)
        frozen = captured["payload"]["normalized_request"]
        self.assertEqual(frozen["processing_selection"]["target_person_id"], b2)
        self.assertEqual(frozen["source_index_mapping"], [-1, 2])

        # The UI moves on: person b1, mapping to source-a, and even target A.
        api.target_context({"target_media_id": self.b, "selection_version": 71,
                            "selected_target_person_id": b1,
                            "target_person_source_mapping": {b1: "source-a"}})
        api.target_select({"target_media_id": self.a})
        state.selected_input_face_index = 0

        # What the worker derives from that payload is still B/b2/source-c.
        with api._target_context_lock:
            api._activate_target_media(media_id=self.b, refresh=False)
            worker_request = captured["payload"]["normalized_request"]
            self.assertEqual(worker_request["processing_selection"]["target_media_id"], self.b)
            self.assertEqual(worker_request["processing_selection"]["target_person_id"], b2)
            self.assertEqual(worker_request["source_index_mapping"], [-1, 2])
            # And even if the worker had to re-normalize (a payload without the
            # frozen request), the canonical object on the payload wins over
            # the mutated globals.
            payload = dict(captured["payload"])
            payload.pop("normalized_request")
            rederived = api._canonical_processing_request(
                payload, target_media_index=1, target_media_id=self.b)
            self.assertEqual(rederived["processing_selection"]["target_person_id"], b2)
            self.assertEqual(rederived["source_index_mapping"], [-1, 2])
            self.assertEqual(rederived["selection_state"]["person_id"], b2)


# ── Queue: serialized at creation, validated at dispatch ──────────────────
class QueueSelection(Stage14Base):
    def setUp(self):
        super().setUp()
        import tempfile
        self._tmp = tempfile.mkdtemp(prefix="stage14_queue_")
        self._old_file = q.QUEUE_FILE
        q.QUEUE_FILE = os.path.join(self._tmp, "queue.json")
        self._old_hooks = {k: getattr(q, k) for k in (
            "_progress", "list_files_process", "_run_swap", "_stop_current",
            "_snapshot_outputs", "_outputs_since", "_create_project",
            "_validate_project", "_project_source_list", "_activate_target",
            "_ensure_target_media_id", "_benchmark_running", "_selection_invalidation")}
        q._queue["jobs"] = []
        q._queue.update({"running": False, "paused": False, "current": None})
        q.pause_controller.cancel()
        q.pause_controller.start()
        self.ran = []
        self.progress = {"processing": False, "progress": 0.0, "error": "", "paused": False}
        q._progress = self.progress
        q.list_files_process = api.list_files_process
        q._project_source_list = api.list_files_process
        q._stop_current = lambda: None
        q._run_swap = self._fake_run
        q._snapshot_outputs = None
        q._outputs_since = None
        q._create_project = None
        q._validate_project = None
        q._activate_target = api._activate_target_media
        q._ensure_target_media_id = api._ensure_target_media_id
        q._benchmark_running = lambda: False
        q._selection_invalidation = api._selection_invalidation_for_active_context

    def tearDown(self):
        q._queue["running"] = False
        q._queue["jobs"] = []
        q.QUEUE_FILE = self._old_file
        for key, value in self._old_hooks.items():
            setattr(q, key, value)
        super().tearDown()

    def _fake_run(self, payload):
        # The worker's own derivation, exactly as api._run_swap performs it.
        with api._target_context_lock:
            request = api._canonical_processing_request(
                payload, target_media_index=payload.get("target_index"),
                target_media_id=payload.get("target_media_id"))
        self.ran.append({"payload": dict(payload), "request": request})
        self.progress["progress"] = 1.0
        self.progress["processing"] = False

    def _drain(self, timeout=5.0):
        import time
        roop_globals.processing = True
        q.queue_start()
        deadline = time.time() + timeout
        while q._queue["running"] and time.time() < deadline:
            time.sleep(0.01)
        self.assertFalse(q._queue["running"], "runner did not finish")

    def _job(self, media_id, person_id, source_id, version, **extra):
        selection = self._selection(media_id, person_id, source_id, version)
        return {
            "target_name": os.path.basename(api.list_files_process[
                0 if media_id == self.a else 1].filename),
            "target_media_id": media_id,
            "source_index": 0, "source_name": "stale-name", "source_id": source_id,
            "processing_selection": selection,
            "selection_version": version,
            "payload": {"swap_model": "inswapper", "detection": "All faces",
                        "processing_selection": selection},
            **extra,
        }

    def test_queue_add_freezes_the_selection(self):
        b2 = self.b_people[1]
        snap = q.queue_add(self._job(self.b, b2, "source-b", 100))
        job = snap["jobs"][0]
        self.assertEqual(job["processing_selection"]["target_person_id"], b2)
        self.assertEqual(job["processing_selection"]["source_identity_id"], "source-b")
        self.assertEqual(job["processing_selection"]["target_media_id"], self.b)
        self.assertEqual(job["selection_version"], 100)
        self.assertEqual(job["request_id"], "req-100")
        # And it survives the persisted round trip.
        q._queue["jobs"] = []
        q.load()
        self.assertEqual(q._snapshot()["jobs"][0]["processing_selection"]["target_person_id"], b2)

    def test_worker_renders_the_queued_selection_after_the_ui_moved_on(self):
        """RACE 3 through the queue: queue B/b2, switch UI to A/a1, dispatch."""
        a1 = self.a_people[0]
        b1, b2 = self.b_people
        q.queue_add(self._job(self.b, b2, "source-c", 200))
        # The UI moves on before the job runs.
        api.target_context({"target_media_id": self.b, "selection_version": 201,
                            "selected_target_person_id": b1,
                            "target_person_source_mapping": {b1: "source-a"}})
        api.target_select({"target_media_id": self.a})
        api.target_context({"target_media_id": self.a, "selection_version": 202,
                            "selected_target_person_id": a1})
        state.selected_input_face_index = 0
        self._drain()
        self.assertEqual(len(self.ran), 1)
        request = self.ran[0]["request"]
        self.assertEqual(self.ran[0]["payload"]["target_media_id"], self.b)
        self.assertEqual(request["processing_selection"]["target_media_id"], self.b)
        self.assertEqual(request["processing_selection"]["target_person_id"], b2)
        self.assertEqual(request["selection_state"]["person_id"], b2)
        self.assertEqual(request["source_index_mapping"], [-1, 2])
        self.assertEqual(request["source_index"], 1)
        self.assertEqual(request["request_id"], "req-200")
        self.assertEqual(q._snapshot()["jobs"][0]["state"], "COMPLETED")

    def test_mapping_change_after_queueing_does_not_reach_the_job(self):
        """RACE 4: the queued mapping is immutable."""
        b2 = self.b_people[1]
        q.queue_add(self._job(self.b, b2, "source-c", 300))
        api.target_context({"target_media_id": self.b, "selection_version": 301,
                            "target_person_source_mapping": {b2: "source-a"}})
        self.assertEqual(state.active_target_person_source_mapping, {b2: "source-a"})
        self._drain()
        request = self.ran[0]["request"]
        self.assertEqual(request["target_person_source_mapping"], {b2: "source-c"})
        self.assertEqual(request["source_index_mapping"], [-1, 2])

    def test_removed_source_invalidates_the_job_explicitly(self):
        """RACE 4b: an identity that disappeared fails the job, never a silent skip."""
        b2 = self.b_people[1]
        q.queue_add(self._job(self.b, b2, "source-c", 400))
        roop_globals.INPUT_FACESETS[:] = [_Source("source-a"), _Source("source-b")]
        self._drain()
        self.assertEqual(self.ran, [], "the worker ran with a re-interpreted selection")
        job = q._snapshot()["jobs"][0]
        self.assertEqual(job["state"], "FAILED")
        self.assertIn("selection invalidated", job["error"])
        self.assertIn("source-c", job["error"])

    def test_removed_person_invalidates_the_job_explicitly(self):
        b2 = self.b_people[1]
        q.queue_add(self._job(self.b, b2, "source-c", 500))
        api.target_remove_face({"target_media_id": self.b, "face_index": 1})
        self._drain()
        self.assertEqual(self.ran, [])
        job = q._snapshot()["jobs"][0]
        self.assertEqual(job["state"], "FAILED")
        self.assertIn(b2, job["error"])

    def test_legacy_rank_addressed_job_still_dispatches(self):
        """BatchSwap queues rank-addressed selections; a valid rank must not
        read as a removed person, and the worker must resolve it to the id."""
        b1, b2 = self.b_people
        legacy = {
            "target_name": "b.mp4", "target_media_id": self.b, "source_index": 1,
            "payload": {"detection": "Selected face",
                        "selection_state": {"selection_mode": "selected",
                                            "person_id": 1, "person_ids": [1]},
                        "face_mapping": [-1, 1]},
        }
        q.queue_add(legacy)
        self._drain()
        self.assertEqual(len(self.ran), 1)
        request = self.ran[0]["request"]
        self.assertEqual(request["selection_state"]["person_id"], b2)
        self.assertEqual(request["processing_selection"]["target_person_id"], b2)
        self.assertEqual(q._snapshot()["jobs"][0]["state"], "COMPLETED")
        # …and a rank that no longer exists is an explicit invalidation.
        q._queue["jobs"] = []
        q.queue_add(dict(legacy, payload=dict(legacy["payload"], selection_state={
            "selection_mode": "selected", "person_id": 5, "person_ids": [5]})))
        self._drain()
        self.assertEqual(q._snapshot()["jobs"][0]["state"], "FAILED")
        self.assertIn("target person 5", q._snapshot()["jobs"][0]["error"])
        self.assertNotEqual(b1, b2)

    def test_explicit_edit_refreezes_the_selection(self):
        b1, b2 = self.b_people
        job_id = q.queue_add(self._job(self.b, b2, "source-c", 600))["jobs"][0]["id"]
        edited = self._selection(self.b, b1, "source-a", 601)
        q.queue_update({"id": job_id, "processing_selection": edited,
                        "payload": {"processing_selection": edited}})
        job = q._snapshot()["jobs"][0]
        self.assertEqual(job["processing_selection"]["target_person_id"], b1)
        self.assertEqual(job["selection_version"], 601)

    def test_pure_invalidation_reasons(self):
        sel = self._selection("m1", "tp_x", "s1", 1, mapping={"tp_x": "s1", "tp_y": "s2"})
        self.assertEqual(selection_invalidation_reasons(
            sel, target_media_ids=["m1"], target_person_ids=["tp_x", "tp_y"],
            source_identity_ids=["s1", "s2"]), [])
        reasons = selection_invalidation_reasons(
            sel, target_media_ids=["m2"], target_person_ids=["tp_y"],
            source_identity_ids=["s2"])
        joined = " ".join(reasons)
        for token in ("target media m1", "target person tp_x", "source s1"):
            self.assertIn(token, joined)


# ── Session persistence ───────────────────────────────────────────────────
class SessionPersistence(Stage14Base):
    def _reload(self):
        """Simulate a webview reload: drop every in-memory selection the
        server has not been told about, then rehydrate from /api/state."""
        return api.get_state()

    def test_selected_person_source_and_mapping_survive_a_reload(self):
        """RACE 5."""
        a1, a2 = self.a_people
        api.target_context({"target_media_id": self.a, "selection_version": 1,
                            "selected_target_person_id": a2,
                            "target_person_source_mapping": {a2: "source-b"}})
        api.source_select({"index": 2})
        st = self._reload()
        self.assertEqual(st["target_media_id"], self.a)
        self.assertEqual(st["selected_target_person_id"], a2)
        self.assertEqual(st["target_person_source_mapping"], {a2: "source-b"})
        self.assertEqual(st["selected_source_id"], "source-c")
        self.assertEqual(st["selected_source_index"], 2)
        # Identity, not position: reorder the persons' angles and the selected
        # id is unchanged even though its array position moved.
        roop_globals.TARGET_FACES[:] = ["A2", "A1"]
        roop_globals.TARGET_FACE_GROUP[:] = [1, 0]
        roop_globals.TARGET_FACE_PERSON_IDS[:] = [a2, a1]
        refs = list(roop_globals.TARGET_REFERENCE_FACE_IDS)
        roop_globals.TARGET_REFERENCE_FACE_IDS[:] = [refs[1], refs[0]]
        api._save_active_target_context_locked()
        st = self._reload()
        self.assertEqual(st["selected_target_person_id"], a2)
        self.assertEqual(st["target_person_ids"][0], a2)

    def test_selection_is_per_target_media_across_a_switch_and_reload(self):
        a2 = self.a_people[1]
        b1 = self.b_people[0]
        api.target_context({"target_media_id": self.a, "selection_version": 1,
                            "selected_target_person_id": a2})
        api.target_select({"target_media_id": self.b})
        api.target_context({"target_media_id": self.b, "selection_version": 2,
                            "selected_target_person_id": b1})
        st = self._reload()
        self.assertEqual(st["target_media_id"], self.b)
        self.assertEqual(st["selected_target_person_id"], b1)
        api.target_select({"target_media_id": self.a})
        st = self._reload()
        self.assertEqual(st["target_media_id"], self.a)
        self.assertEqual(st["selected_target_person_id"], a2)

    def test_deleted_person_cannot_come_back(self):
        """RACE 6."""
        a1, a2 = self.a_people
        api.target_context({"target_media_id": self.a, "selection_version": 1,
                            "selected_target_person_id": a2,
                            "target_person_source_mapping": {a2: "source-b"}})
        api.target_remove_face({"target_media_id": self.a, "face_index": 1})
        st = self._reload()
        self.assertNotIn(a2, st["target_person_ids"])
        self.assertNotEqual(st["selected_target_person_id"], a2)
        self.assertNotIn(a2, st["target_person_source_mapping"])
        # A late client write naming the deleted person is refused, not applied.
        res = api.target_context({"target_media_id": self.a, "selection_version": 2,
                                  "selected_target_person_id": a2})
        self.assertEqual(getattr(res, "status_code", 200), 422)
        self.assertEqual(state.selected_target_person_id, a1)
        # A late preview naming it gets a diagnostic, and swaps nothing.
        res = self._preview(self._selection(self.a, a2, "source-b", 3))
        self.assertEqual(res["selection_diagnostic"], "invalid_person_id")

    def test_deleted_target_media_cannot_come_back(self):
        b1 = self.b_people[0]
        api.target_remove({"target_media_id": self.b})
        self.assertFalse(api._target_contexts.has(self.b))
        res = api.target_context({"target_media_id": self.b, "selection_version": 9,
                                  "selected_target_person_id": b1})
        self.assertEqual(getattr(res, "status_code", 200), 404)
        res = self._preview(self._selection(self.b, b1, "source-a", 10))
        self.assertEqual(getattr(res, "status_code", 200), 404)
        st = self._reload()
        self.assertEqual([t["media_id"] for t in st["targets"]], [self.a])


# ── Pure helpers ──────────────────────────────────────────────────────────
class PureHelpers(unittest.TestCase):
    def test_apply_projects_canonical_onto_flat_fields(self):
        selection = build_processing_selection({
            "processing_selection": {
                "request_id": "r", "selection_version": 3,
                "target_media_id": "m", "target_person_id": "tp_1",
                "source_identity_id": "s", "detection_mode": "Selected face",
                "target_person_source_mapping": {"tp_1": "s", "tp_2": -1, "tp_3": None},
                "selection_state": {"selection_mode": "selected", "person_id": "tp_1"},
            },
            "detection": "All faces", "selected_source_id": "old",
        })
        self.assertEqual(selection["target_person_source_mapping"], {"tp_1": "s"})
        flat = apply_processing_selection({"detection": "All faces"}, selection)
        self.assertEqual(flat["detection"], "Selected face")
        self.assertEqual(flat["selected_source_id"], "s")
        self.assertEqual(flat["target_media_id"], "m")
        self.assertEqual(flat["selection_state"]["person_id"], "tp_1")
        self.assertEqual(flat["request_id"], "r")
        self.assertEqual(flat["selection_version"], 3)

    def test_stable_id_selection_is_admitted(self):
        """Inherited defect: the admission check re-normalized a stable-id
        selection without the id universe, parsed "tp_..." as a missing rank
        and refused EVERY Selected-face preview/render with
        selection_required."""
        sel = normalize_target_selection(
            {"selection_mode": "selected", "person_id": "tp_a"},
            target_person_ids=["tp_a", "tp_b"])
        self.assertTrue(sel["valid"])
        self.assertIsNone(selection_diagnostic_for_mode(
            "selected", sel, 2, target_person_ids=["tp_a", "tp_b"]))
        self.assertIsNone(selection_diagnostic_for_mode("selected", sel, 2))
        self.assertEqual(selection_diagnostic_for_mode(
            "selected", sel, 2, target_person_ids=["tp_b"]), "invalid_person_id")
        self.assertEqual(selection_diagnostic_for_mode("selected", sel, 0), "target_required")

    def test_normalize_request_without_boundary_still_builds_the_selection(self):
        request = normalize_processing_request(
            {"detection": "Selected face",
             "selection_state": {"selection_mode": "selected", "person_id": "tp_a"},
             "target_person_source_mapping": {"tp_a": "s1"},
             "selected_source_id": "s1", "target_media_id": "m"},
            target_groups=[0, 1], source_count=2,
            current_source_ids=["s0", "s1"], target_person_ids=["tp_a", "tp_b"],
            request_id="direct")
        self.assertEqual(request["processing_selection"]["target_person_id"], "tp_a")
        self.assertEqual(request["processing_selection"]["source_identity_id"], "s1")
        self.assertEqual(request["request_id"], "direct")
        self.assertEqual(request["source_index_mapping"], [1, -1])


if __name__ == "__main__":
    unittest.main()
