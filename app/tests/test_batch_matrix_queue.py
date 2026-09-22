"""A Batch Matrix job list, pushed through the real queue.

docs/development/BATCH_MATRIX_DATA_FLOW.md traces how the four strategies
become N single-target jobs in POST /api/queue/add_batch and how the queue
dispatches each one. These tests hold that contract to the code:

  * the recipe split A+alice, B+bob, C+alice (three targets, two facesets)
    creates exactly three jobs, each carrying ITS target, ITS faceset and ITS
    face_mapping, and dispatches each with the right target index;
  * per-job progress and cancellation stay with the job they belong to;
  * a faceset removed while the batch is queued behaves as documented (§7):
    a removed PRIMARY fails that job alone at dispatch, a removed non-primary
    row silently becomes a skip, a removed-and-re-added faceset is found by id
    at its new gallery position;
  * per-file matrix and grouped pairings survive the trip (a disabled row must
    not shift the next row onto the wrong target; a target in two groups gets
    both groups' mappings, not one of them twice).

Two layers, because CI has no ML stack:

  BatchMatrixQueue      routes_queue only (light profile runs it): job shape,
                        order, progress/cancel isolation, the all-or-nothing 400.
  BatchMatrixEndToEnd   the real api.py target contexts and source gallery,
                        with `_run_swap` replaced by the worker's own request
                        derivation stopped just before the swap model is called
                        (needs `import api`; skipped in the light profile).
  BatchMatrixStagingJS  the strategies' real JS (batchMatrix.js) via node, and
                        its emitted request bodies pushed through the queue.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import fixtures  # noqa: E402
import api_state as state  # noqa: E402
import roop.globals as roop_globals  # noqa: E402
import routes_queue as q  # noqa: E402
from tests.test_queue import QueueTestBase, _Entry as _QueueEntry  # noqa: E402

try:
    import api  # noqa: E402
    from tests.test_stage14_async_state import Stage14Base, _Entry, _Source  # noqa: E402
except ImportError:  # the light profile turns the ML stack into an ImportError
    api = None
    Stage14Base = unittest.TestCase

REACT_UI = Path(APP).parent / "react-ui"
CHECKER = REACT_UI / ".render-check" / "batch-matrix-check.mjs"

ALICE = "/facesets/alice.fsz"
BOB = "/facesets/bob.fsz"
GALLERY = [(ALICE, "alice.fsz"), (BOB, "bob.fsz")]


def _node():
    found = shutil.which("node")
    if found:
        return found
    home = Path(fixtures.pinokio_home())
    for cand in (home / "bin" / "miniconda" / "node.exe",
                 home / "bin" / "miniforge" / "node.exe"):
        if cand.exists():
            return str(cand)
    return None


def wire_job(target_name, mappings, mode="Selected face", gallery=None,
             frame_start=None, frame_end=None, label="", **payload_extra):
    """One entry of POST /api/queue/add_batch, in the shape BatchSwap sends
    (BATCH_MATRIX_DATA_FLOW.md §3-§4). `mappings` is [(rank, gallery index)];
    a rank that is not listed is an explicit -1 (skip), and an index outside
    the gallery is -1 too -- never source 0."""
    gallery = gallery if gallery is not None else GALLERY
    ranks = [r for r, _ in mappings]
    face_mapping = [-1] * (max(ranks) + 1 if ranks else 0)
    for rank, idx in mappings:
        face_mapping[rank] = idx if 0 <= idx < len(gallery) else -1
    ids = [gallery[i][0] if i >= 0 else None for i in face_mapping]
    names = [gallery[i][1] if i >= 0 else None for i in face_mapping]
    primary = mappings[0][1] if mappings and 0 <= mappings[0][1] < len(gallery) else -1
    people = sorted(set(ranks))
    if mode == "Selected people":
        selection = {"selection_mode": "multi_person", "person_id": None, "person_ids": people}
    elif mode == "Selected face":
        selection = {"selection_mode": "selected", "person_id": people[0] if people else None,
                     "person_ids": people[:1]}
    else:
        selection = {"selection_mode": "none", "person_id": None, "person_ids": []}
    payload = {
        "swap_model": "inswapper",
        "detection": mode,
        "enhancer": "None",
        "auto_fallback": False,
        "face_mapping": face_mapping,
        "source_mapping_names": names,
        "source_mapping_ids": ids,
        "selected_source_name": gallery[primary][1] if primary >= 0 else None,
        "selected_source_id": gallery[primary][0] if primary >= 0 else None,
        "selection_state": selection,
    }
    payload.update(payload_extra)
    job = {
        "target_name": target_name,
        "source_index": primary,
        "source_name": gallery[primary][1] if primary >= 0 else f"Faceset {primary + 1}",
        "source_id": payload["selected_source_id"],
        "payload": payload,
        "label": label or f"{target_name} {mappings}",
    }
    if frame_start is not None:
        job["frame_start"] = frame_start
    if frame_end is not None:
        job["frame_end"] = frame_end
    return job


# The recipe split the user asked about: sequential match over 3 targets and
# 2 facesets pairs tIdx % 2 -> A+alice, B+bob, C+alice.
def sequential_split():
    return [
        wire_job("a.mp4", [(0, 0)], label="Sequential | a.mp4 -> alice.fsz"),
        wire_job("b.mp4", [(0, 1)], label="Sequential | b.mp4 -> bob.fsz"),
        wire_job("c.mp4", [(0, 0)], label="Sequential | c.mp4 -> alice.fsz"),
    ]


def triple(job):
    return (job["target_name"], job["source_index"], job["source_name"],
            job["source_id"], tuple(job["payload"]["face_mapping"]),
            tuple(job["payload"]["source_mapping_ids"]))


# ══════════════════════════════════════════════════════════════════════════
class BatchMatrixQueue(QueueTestBase):
    """routes_queue alone: what add_batch stores and how _loop hands it out."""

    def setUp(self):
        super().setUp()
        self.entries.extend([_QueueEntry("/media/a.mp4", 300),
                             _QueueEntry("/media/b.mp4", 120),
                             _QueueEntry("/media/c.mp4", 600)])
        # Dispatch re-resolves the primary faceset against the live gallery;
        # without api.py loaded the gallery is this attribute (the code's own
        # first choice), which does not exist on api_state by default.
        state.source_faces_info = [{"id": sid, "name": name, "count": 1, "poses": []}
                                   for sid, name in GALLERY]
        self.addCleanup(lambda: delattr(state, "source_faces_info"))
        self.dispatched = []            # (target_index, selected_input_face_index, face_mapping)

        def run(payload):
            self.dispatched.append((payload["target_index"],
                                    state.selected_input_face_index,
                                    list(payload["face_mapping"])))
            self._fake_run(payload)
        q._run_swap = run

    # ── 1. three targets, two facesets, one recipe split ──────────────────
    def test_sequential_split_queues_three_jobs_each_with_its_own_triple(self):
        snap = q.queue_add_batch({"jobs": sequential_split()})
        self.assertNotIsInstance(snap, q.JSONResponse, getattr(snap, "body", b""))
        jobs = snap["jobs"]
        self.assertEqual(len(jobs), 3)
        self.assertEqual([triple(j) for j in jobs], [
            ("a.mp4", 0, "alice.fsz", ALICE, (0,), (ALICE,)),
            ("b.mp4", 1, "bob.fsz", BOB, (1,), (BOB,)),
            ("c.mp4", 0, "alice.fsz", ALICE, (0,), (ALICE,)),
        ])
        self.assertEqual([j["state"] for j in jobs], ["QUEUED"] * 3)
        self.assertEqual(len({j["id"] for j in jobs}), 3)
        # The selection is frozen per job from ITS payload (§4).
        self.assertEqual([j["processing_selection"]["source_identity_id"] for j in jobs],
                         [ALICE, BOB, ALICE])
        self.assertEqual([j["processing_selection"]["target_person_id"] for j in jobs],
                         ["0", "0", "0"], "ranks, converted at dispatch")
        # Documented gaps, pinned so a change is noticed: F1 (no media id on
        # the wire) and F2 (no per-person stable mapping for BatchSwap jobs).
        self.assertEqual([j["target_media_id"] for j in jobs], ["", "", ""])
        self.assertEqual([j["processing_selection"]["target_person_source_mapping"] for j in jobs],
                         [{}, {}, {}])
        # And it survives the persisted round trip unchanged.
        q._queue["jobs"] = []
        q.load()
        self.assertEqual([triple(j) for j in q._snapshot()["jobs"]], [triple(j) for j in jobs])

    def test_sequential_split_dispatches_each_job_to_its_own_target_and_faceset(self):
        q.queue_add_batch({"jobs": sequential_split()})
        self._drain()
        self.assertEqual(self.dispatched, [(0, 0, [0]), (1, 1, [1]), (2, 0, [0])])
        self.assertEqual([j["state"] for j in q._snapshot()["jobs"]], ["COMPLETED"] * 3)
        # Each dispatched payload is the job's own object, not a shared one.
        self.assertEqual([p["face_mapping"] for p in self.ran], [[0], [1], [0]])
        self.assertEqual([p["selected_source_id"] for p in self.ran], [ALICE, BOB, ALICE])
        self.assertEqual([p["target_index"] for p in self.ran], [0, 1, 2])
        self.assertEqual(len({id(p) for p in self.ran}), 3)

    def test_the_primary_faceset_is_re_resolved_by_id_when_the_gallery_is_reordered(self):
        q.queue_add_batch({"jobs": sequential_split()})
        state.source_faces_info.reverse()        # bob is gallery 0 now, alice 1
        self._drain()
        self.assertEqual([d[1] for d in self.dispatched], [1, 0, 1],
                         "the stored numeric source_index must not be trusted")

    # ── 2. per-job progress and cancel stay with their job ────────────────
    def test_progress_is_recorded_per_job_not_inherited_from_the_previous_one(self):
        def run(payload):
            n = payload["target_index"]
            self.progress.update({"frames_done": 100 * (n + 1), "frames_total": 100 * (n + 1),
                                  "fps": 10.0 * (n + 1), "desc": f"job {n}"})
            self._fake_run(payload)
        q._run_swap = run
        jobs = sequential_split()
        jobs.insert(2, wire_job("gone.mp4", [(0, 1)]))     # fails before dispatch
        q.queue_add_batch({"jobs": jobs})
        self._drain()
        snap = q._snapshot()["jobs"]
        self.assertEqual([j["state"] for j in snap], ["COMPLETED", "COMPLETED", "FAILED", "COMPLETED"])
        self.assertEqual([j["progress"]["frames_total"] for j in snap], [100, 200, None, 300])
        self.assertEqual([j["progress"]["fraction"] for j in snap], [1.0, 1.0, 0.0, 1.0])
        self.assertEqual(snap[2]["progress"]["phase"], "FAILED",
                         "a pre-dispatch failure must not wear the previous job's snapshot")
        self.assertIn("no longer loaded", snap[2]["error"])
        self.assertEqual([j["error"] for j in snap if j["state"] == "COMPLETED"], ["", "", ""])

    def test_cancelling_a_queued_job_does_not_touch_its_neighbours(self):
        gate = threading.Event()
        entered = threading.Event()

        def run(payload):
            if payload["target_index"] == 0:
                entered.set()
                gate.wait(5.0)
            self.dispatched.append(payload["target_index"])
            self._fake_run(payload)
        q._run_swap = run
        ids = [j["id"] for j in q.queue_add_batch({"jobs": sequential_split()})["jobs"]]
        roop_globals.processing = True
        q.queue_start()
        self.assertTrue(entered.wait(2.0))
        cancelled = q.queue_cancel({"id": ids[1]})["jobs"]
        self.assertEqual([j["state"] for j in cancelled], ["PROCESSING", "CANCELLED", "QUEUED"])
        self.assertEqual(cancelled[1]["error"], "cancelled before processing")
        gate.set()
        deadline = time.time() + 5.0
        while q._queue["running"] and time.time() < deadline:
            time.sleep(0.01)
        snap = q._snapshot()["jobs"]
        self.assertEqual([j["state"] for j in snap], ["COMPLETED", "CANCELLED", "COMPLETED"])
        self.assertEqual(self.dispatched, [0, 2], "the cancelled job never ran; the next one did")
        # The cancelled job keeps its flag as the record of what happened;
        # its neighbours must not have caught it.
        self.assertEqual([j["cancel_requested"] for j in snap], [False, True, False])
        self.assertEqual([j["error"] for j in snap], ["", "cancelled before processing", ""])

    def test_cancelling_the_running_job_does_not_cancel_the_next_one(self):
        gate = threading.Event()
        entered = threading.Event()
        stops = []
        q._stop_current = lambda: (stops.append(True), gate.set())

        def run(payload):
            if payload["target_index"] == 0:
                entered.set()
                gate.wait(5.0)
            self.dispatched.append(payload["target_index"])
            self._fake_run(payload)
        q._run_swap = run
        ids = [j["id"] for j in q.queue_add_batch({"jobs": sequential_split()})["jobs"]]
        roop_globals.processing = True
        q.queue_start()
        self.assertTrue(entered.wait(2.0))
        q.queue_cancel({"id": ids[0]})
        deadline = time.time() + 5.0
        while q._queue["running"] and time.time() < deadline:
            time.sleep(0.01)
        snap = q._snapshot()["jobs"]
        self.assertEqual(stops, [True], "cancel of the current job asks the render to stop")
        self.assertEqual([j["state"] for j in snap], ["CANCELLED", "COMPLETED", "COMPLETED"])
        self.assertEqual(snap[0]["error"], "cancelled by user")
        self.assertEqual(self.dispatched, [0, 1, 2])
        self.assertEqual([j["cancel_requested"] for j in snap], [False, False, False])
        self.assertEqual(q._cancel_requested, set())

    # ── the all-or-nothing 400 (§7 row 2) ─────────────────────────────────
    def test_a_stale_primary_row_rejects_the_whole_batch(self):
        jobs = sequential_split()
        jobs[1] = wire_job("b.mp4", [(0, 7)])        # gallery index 7 does not exist
        self.assertEqual(jobs[1]["source_index"], -1)
        res = q.queue_add_batch({"jobs": jobs})
        self.assertEqual(res.status_code, 400)
        self.assertIn("source_index must be a non-negative integer", res.body.decode())
        self.assertEqual(q._snapshot()["jobs"], [], "nothing from the batch is queued")

    # ── per-file matrix and grouped shapes through the queue ──────────────
    def test_matrix_with_a_disabled_middle_row_dispatches_to_the_right_targets(self):
        # Row A: two people, bob then alice, frames 5-50. Row B disabled (no
        # job). Row C: alice. A disabled row must not shift C onto B's index.
        jobs = [
            wire_job("a.mp4", [(0, 1), (1, 0)], "Selected people", frame_start=5, frame_end=50),
            wire_job("c.mp4", [(0, 0)]),
        ]
        q.queue_add_batch({"jobs": jobs})
        seen = []
        q._run_swap = lambda p: (seen.append((p["target_index"],
                                              self.entries[p["target_index"]].startframe,
                                              self.entries[p["target_index"]].endframe,
                                              p["face_mapping"], p["source_mapping_ids"])),
                                 self._fake_run(p))
        self._drain()
        self.assertEqual(seen, [
            (0, 5, 50, [1, 0], [BOB, ALICE]),
            (2, 0, 600, [0], [ALICE]),
        ])
        self.assertEqual([j["state"] for j in q._snapshot()["jobs"]], ["COMPLETED"] * 2)

    def test_grouped_jobs_each_carry_their_own_groups_mapping(self):
        # Group X = {A, C} -> bob; Group Y = {B, C} -> alice. C is in both.
        jobs = [
            wire_job("a.mp4", [(0, 1)], enhancer="CodeFormer", label="Group X | a.mp4"),
            wire_job("c.mp4", [(0, 1)], enhancer="CodeFormer", label="Group X | c.mp4"),
            wire_job("b.mp4", [(0, 0)], "All faces", label="Group Y | b.mp4"),
            wire_job("c.mp4", [(0, 0)], "All faces", label="Group Y | c.mp4"),
        ]
        q.queue_add_batch({"jobs": jobs})
        self._drain()
        self.assertEqual(self.dispatched, [(0, 1, [1]), (2, 1, [1]), (1, 0, [0]), (2, 0, [0])])
        self.assertEqual([p["enhancer"] for p in self.ran], ["CodeFormer", "CodeFormer", "None", "None"])
        self.assertEqual([p["detection"] for p in self.ran],
                         ["Selected face", "Selected face", "All faces", "All faces"])


# ══════════════════════════════════════════════════════════════════════════
class _EndToEndBase(Stage14Base):
    """Real target contexts (A: 2 people, B: 1, C: 2), real gallery, real
    dispatch validation, and the worker's own request derivation -- stopped
    where `batch_process_regular` would be handed the mapped facesets."""

    def setUp(self):
        super().setUp()
        self.c = self._add("c.mp4")
        self.b_people = self._configure(1, ["B1"], {})
        self.c_people = self._configure(2, ["C1", "C2"], {})
        api._activate_target_media(index=0, refresh=False)      # page-load active = A
        self.people = {0: self.a_people, 1: self.b_people, 2: self.c_people}
        self._set_gallery(GALLERY)
        # Stage14's entries are one frame long; the matrix rows trim real ranges.
        for entry, total in zip(api.list_files_process, (300, 120, 600)):
            entry.total_frames = total
            entry.endframe = total

        self._tmp = tempfile.mkdtemp(prefix="batch_matrix_")
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
        self.progress = {"processing": False, "progress": 0.0, "error": "", "paused": False}
        q._progress = self.progress
        q.list_files_process = api.list_files_process
        q._project_source_list = api.list_files_process
        q._stop_current = lambda: None
        q._run_swap = self._worker
        q._snapshot_outputs = None
        q._outputs_since = None
        q._create_project = None
        q._validate_project = None
        q._activate_target = api._activate_target_media
        q._ensure_target_media_id = api._ensure_target_media_id
        q._benchmark_running = lambda: False
        q._selection_invalidation = api._selection_invalidation_for_active_context
        self.ran = []
        self.before_job = {}            # target_index -> callable run before that job's swap

    def tearDown(self):
        q._queue["running"] = False
        q._queue["jobs"] = []
        q.QUEUE_FILE = self._old_file
        for key, value in self._old_hooks.items():
            setattr(q, key, value)
        super().tearDown()

    def _set_gallery(self, gallery):
        roop_globals.INPUT_FACESETS[:] = [_Source(sid) for sid, _name in gallery]

    def _worker(self, payload):
        """api._run_swap up to, but not including, the swap model: the canonical
        request, the admission gate, and the person-ordered faceset list."""
        hook = self.before_job.pop(payload.get("target_index"), None)
        if hook:
            hook()
        with api._target_context_lock:
            request = api._canonical_processing_request(
                payload, target_media_index=payload.get("target_index"),
                target_media_id=payload.get("target_media_id"))
        diagnostic = api._selection_diagnostic_for_mode(
            request["swap_mode"], request["selection_state"])
        mapped = api.mapped_facesets(request["source_index_mapping"], request["swap_mode"])
        self.ran.append({
            "target_index": payload["target_index"],
            "media_id": payload["target_media_id"],
            "selected_gallery": state.selected_input_face_index,
            "request": request,
            "diagnostic": diagnostic,
            "facesets": None if mapped is None else
                        [getattr(fs, "_source_id", None) for fs in mapped],
            "entry_range": (api.list_files_process[payload["target_index"]].startframe,
                            api.list_files_process[payload["target_index"]].endframe),
        })
        if diagnostic:
            self.progress.update({"processing": False,
                                  "error": api._selection_message(diagnostic)})
            return
        self.progress.update({"progress": 1.0, "processing": False})

    def _drain(self, timeout=10.0):
        roop_globals.processing = True
        q.queue_start()
        deadline = time.time() + timeout
        while q._queue["running"] and time.time() < deadline:
            time.sleep(0.01)
        self.assertFalse(q._queue["running"], "runner did not finish")

    def _states(self):
        return [(j["state"], j["error"]) for j in q._snapshot()["jobs"]]

    def _assert_run(self, run, *, target, gallery_indices, facesets, person_ranks,
                    source_index, media=None):
        """`run` reached the swap for `target` with each person rank bound to
        the expected gallery index / faceset id, and the selection resolved to
        THAT target's stable person ids."""
        self.assertEqual(run["target_index"], target)
        self.assertEqual(run["media_id"], media or {0: self.a, 1: self.b, 2: self.c}[target])
        req = run["request"]
        self.assertEqual(req["source_index_mapping"], gallery_indices)
        self.assertEqual(run["facesets"], facesets)
        self.assertEqual(req["source_index"], source_index)
        expected_people = [self.people[target][r] for r in person_ranks]
        self.assertEqual(req["selection_state"]["person_ids"], expected_people)
        self.assertEqual(req["processing_selection"]["target_person_ids"], expected_people)
        self.assertIsNone(run["diagnostic"], run["diagnostic"])


@unittest.skipIf(api is None, "needs api.py (the ML stack); not in the light profile")
class BatchMatrixEndToEnd(_EndToEndBase):
    # ── 1. the recipe split, end to end ───────────────────────────────────
    def test_sequential_split_reaches_the_swap_with_the_right_faceset_per_target(self):
        q.queue_add_batch({"jobs": sequential_split()})
        self._drain()
        self.assertEqual(self._states(), [("COMPLETED", "")] * 3)
        self.assertEqual(len(self.ran), 3)
        self._assert_run(self.ran[0], target=0, gallery_indices=[0], facesets=[ALICE],
                         person_ranks=[0], source_index=0)
        self._assert_run(self.ran[1], target=1, gallery_indices=[1], facesets=[BOB],
                         person_ranks=[0], source_index=0)
        self._assert_run(self.ran[2], target=2, gallery_indices=[0], facesets=[ALICE],
                         person_ranks=[0], source_index=0)
        self.assertEqual([r["selected_gallery"] for r in self.ran], [0, 1, 0])
        # Rank 0 on each target is a DIFFERENT person; nothing leaked across.
        self.assertEqual(len({r["request"]["selection_state"]["person_id"] for r in self.ran}), 3)

    # ── 2. a faceset removed mid-queue ────────────────────────────────────
    def test_removing_the_primary_faceset_mid_queue_fails_only_the_jobs_that_use_it(self):
        q.queue_add_batch({"jobs": sequential_split()})
        # bob disappears while A (alice) is rendering; B needs bob, C alice.
        self.before_job[0] = lambda: self._set_gallery([(ALICE, "alice.fsz")])
        self._drain()
        states = self._states()
        self.assertEqual(states[0], ("COMPLETED", ""))
        self.assertEqual(states[1][0], "FAILED")
        self.assertIn("selection invalidated", states[1][1])
        self.assertIn(f"source {BOB} was removed", states[1][1])
        self.assertEqual(states[2], ("COMPLETED", ""))
        self.assertEqual([r["target_index"] for r in self.ran], [0, 2], "B never reached the worker")
        self._assert_run(self.ran[1], target=2, gallery_indices=[0], facesets=[ALICE],
                         person_ranks=[0], source_index=0)

    def test_removing_a_non_primary_faceset_mid_queue_becomes_a_silent_skip(self):
        """§7: only the primary is validated at dispatch (F2); the other rows
        resolve by id at render time and a missing id is -1 -> an empty
        FaceSet for that person, and the job still completes."""
        q.queue_add_batch({"jobs": [
            wire_job("b.mp4", [(0, 0)]),
            wire_job("a.mp4", [(0, 0), (1, 1)], "Selected people"),   # alice primary, bob rank 1
        ]})
        self.before_job[1] = lambda: self._set_gallery([(ALICE, "alice.fsz")])
        self._drain()
        self.assertEqual(self._states(), [("COMPLETED", "")] * 2)
        run = self.ran[1]
        self._assert_run(run, target=0, gallery_indices=[0, -1], facesets=[ALICE, None],
                         person_ranks=[0, 1], source_index=0)
        self.assertEqual(run["request"]["source_mapping_ids"], [ALICE, None])
        self.assertEqual(run["request"]["source_mapping_errors"], [],
                         "F4: a removed id is not reported as a mapping error")

    def test_a_faceset_removed_and_re_added_is_found_by_id_at_its_new_position(self):
        q.queue_add_batch({"jobs": sequential_split()})
        # alice is removed and re-added: gallery becomes [bob, alice].
        self.before_job[0] = lambda: self._set_gallery([(BOB, "bob.fsz"), (ALICE, "alice.fsz")])
        self._drain()
        self.assertEqual(self._states(), [("COMPLETED", "")] * 3)
        # Job A was already past dispatch (selected_gallery resolved before
        # the hook), but its render-time mapping still resolves by id.
        self.assertEqual(self.ran[0]["request"]["source_index_mapping"], [1])
        self.assertEqual(self.ran[0]["facesets"], [ALICE])
        self._assert_run(self.ran[1], target=1, gallery_indices=[0], facesets=[BOB],
                         person_ranks=[0], source_index=0)
        self._assert_run(self.ran[2], target=2, gallery_indices=[1], facesets=[ALICE],
                         person_ranks=[0], source_index=0)
        self.assertEqual([r["selected_gallery"] for r in self.ran[1:]], [0, 1])

    def test_removing_every_faceset_fails_each_remaining_job_explicitly(self):
        q.queue_add_batch({"jobs": sequential_split()})
        self.before_job[0] = lambda: self._set_gallery([])
        self._drain()
        states = self._states()
        self.assertEqual(states[0], ("COMPLETED", ""))
        for st, err in states[1:]:
            self.assertEqual(st, "FAILED")
            self.assertIn("was removed", err)
        self.assertEqual(len(self.ran), 1)

    # ── 3. per-file matrix and grouped ────────────────────────────────────
    def test_per_file_matrix_rows_reach_their_own_targets_people_and_frames(self):
        q.queue_add_batch({"jobs": [
            wire_job("a.mp4", [(0, 1), (1, 0)], "Selected people", frame_start=5, frame_end=50),
            # b.mp4 disabled -> no job
            wire_job("c.mp4", [(0, 0)]),
        ]})
        self._drain()
        self.assertEqual(self._states(), [("COMPLETED", "")] * 2)
        # Two people: `source_index` is the mapped slot of the highlighted
        # (primary) source -- bob sits at slot 0 of [bob, alice].
        self._assert_run(self.ran[0], target=0, gallery_indices=[1, 0], facesets=[BOB, ALICE],
                         person_ranks=[0, 1], source_index=0)
        self.assertEqual(self.ran[0]["entry_range"], (5, 50))
        self.assertEqual(self.ran[0]["request"]["swap_mode"], "selected_multi")
        self._assert_run(self.ran[1], target=2, gallery_indices=[0], facesets=[ALICE],
                         person_ranks=[0], source_index=0)
        self.assertEqual(self.ran[1]["entry_range"], (0, 600), "no segment: the whole clip")

    def test_per_file_matrix_row_addressing_a_second_person_selects_that_person(self):
        # A's second person only, mapped to bob: rank 0 is an explicit skip.
        q.queue_add_batch({"jobs": [wire_job("a.mp4", [(1, 1)])]})
        self._drain()
        self.assertEqual(self._states(), [("COMPLETED", "")])
        self._assert_run(self.ran[0], target=0, gallery_indices=[-1, 1], facesets=[None, BOB],
                         person_ranks=[1], source_index=1)
        self.assertEqual(self.ran[0]["request"]["selection_state"]["person_id"], self.a_people[1])

    def test_grouped_jobs_dispatch_each_groups_mapping_to_each_of_its_targets(self):
        # Group X = {A, C} -> bob (Selected face); Group Y = {B, C} -> alice (All faces).
        q.queue_add_batch({"jobs": [
            wire_job("a.mp4", [(0, 1)], label="Group X | a.mp4"),
            wire_job("c.mp4", [(0, 1)], label="Group X | c.mp4"),
            wire_job("b.mp4", [(0, 0)], "All faces", label="Group Y | b.mp4"),
            wire_job("c.mp4", [(0, 0)], "All faces", label="Group Y | c.mp4"),
        ]})
        self._drain()
        self.assertEqual(self._states(), [("COMPLETED", "")] * 4)
        self._assert_run(self.ran[0], target=0, gallery_indices=[1], facesets=[BOB],
                         person_ranks=[0], source_index=0)
        self._assert_run(self.ran[1], target=2, gallery_indices=[1], facesets=[BOB],
                         person_ranks=[0], source_index=0)
        # "All faces" addresses nobody by rank; the highlighted source is the
        # mapped one (alice at gallery 0 -> mapped slot 0).
        for run, target in ((self.ran[2], 1), (self.ran[3], 2)):
            self.assertEqual(run["target_index"], target)
            self.assertEqual(run["request"]["swap_mode"], "all")
            self.assertEqual(run["request"]["source_index_mapping"], [0])
            self.assertEqual(run["facesets"], [ALICE])
            self.assertEqual(run["request"]["source_index"], 0)
            self.assertIsNone(run["diagnostic"])
        # Target C ran twice, once per group, with DIFFERENT facesets.
        self.assertEqual([r["facesets"] for r in self.ran if r["target_index"] == 2],
                         [[BOB], [ALICE]])

    def test_a_target_with_no_captured_people_fails_selected_mode_at_dispatch(self):
        """§7 last row: the strategies default to 'Selected face', which
        needs a captured person on THAT target."""
        self._configure(1, [], {})
        api._activate_target_media(index=0, refresh=False)
        q.queue_add_batch({"jobs": sequential_split()})
        self._drain()
        states = self._states()
        self.assertEqual(states[0], ("COMPLETED", ""))
        self.assertEqual(states[1][0], "FAILED")
        self.assertIn("target person 0 was removed", states[1][1])
        self.assertEqual(states[2], ("COMPLETED", ""))

    # ── the active person bank vs. the job's target ───────────────────────
    def test_a_rank_beyond_the_active_targets_bank_still_addresses_the_jobs_target(self):
        """Repro for the BatchSwap finding: with target B (one person) active
        at page load, a matrix row for A that addresses A's second person is
        emitted by the client with `selection_state.person_id = null`
        (normalizeTargetSelectionState range-checks the rank against the
        ACTIVE bank). On the server that job must still reach A's second
        person -- this test feeds the CORRECT wire form to prove the server
        side is sound; BatchMatrixStagingJS pins the client side."""
        api._activate_target_media(index=1, refresh=False)       # B active, one person
        q.queue_add_batch({"jobs": [wire_job("a.mp4", [(1, 1)])]})
        self._drain()
        self.assertEqual(self._states(), [("COMPLETED", "")])
        self._assert_run(self.ran[0], target=0, gallery_indices=[-1, 1], facesets=[None, BOB],
                         person_ranks=[1], source_index=1)

    def test_a_job_whose_client_dropped_the_rank_is_refused_not_misrouted(self):
        """What the server did with the client's output for the case above
        before the fix (person_id null, valid false, diagnostic
        invalid_person_id): the job is refused at the admission gate with that
        diagnostic. The user saw a FAILED job, never the wrong person."""
        job = wire_job("a.mp4", [(1, 1)])
        job["payload"]["selection_state"] = {
            "selection_mode": "selected", "person_id": None, "person_ids": [],
            "valid": False, "diagnostic": "invalid_person_id"}
        q.queue_add_batch({"jobs": [job]})
        self._drain()
        state_, error = self._states()[0]
        self.assertEqual(state_, "FAILED")
        self.assertEqual(error, api._SELECTION_MESSAGES["invalid_person_id"])
        self.assertEqual(self.ran[0]["diagnostic"], "invalid_person_id")


# ══════════════════════════════════════════════════════════════════════════
class BatchMatrixStagingJS(unittest.TestCase):
    """The strategies' real code (react-ui/src/components/faceswap/batchMatrix.js)."""

    def _run_checker(self, emit_to=None):
        node = _node()
        if not node:
            self.skipTest("node not available to run the JS staging suite")
        self.assertTrue(CHECKER.exists(), f"missing checker: {CHECKER}")
        args = [node, str(CHECKER)]
        if emit_to:
            args += ["--emit", emit_to]
        return subprocess.run(args, cwd=str(REACT_UI), capture_output=True,
                              text=True, timeout=180, encoding="utf-8")

    def test_staging_behaviour_suite_passes(self):
        proc = self._run_checker()
        self.assertEqual(proc.returncode, 0,
                         f"JS staging checks failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("ALL GREEN", proc.stdout)

    def test_emitted_requests_are_the_documented_wire_shape(self):
        """The bodies the real builders produce, against the contract
        BATCH_MATRIX_DATA_FLOW.md §4 documents and routes_queue validates."""
        tmp = tempfile.mkdtemp(prefix="batch_emit_")
        out = os.path.join(tmp, "scenarios.json")
        proc = self._run_checker(emit_to=out)
        self.assertTrue(os.path.exists(out), f"no emission:\n{proc.stdout}\n{proc.stderr}")
        with open(out, encoding="utf-8") as fh:
            emitted = json.load(fh)
        scenarios = emitted["scenarios"]
        sources = emitted["fixture"]["sources"]
        alice, bob = sources[0]["id"], sources[1]["id"]

        def wire(name):
            return [(j["target_name"], j["source_index"], j["source_id"],
                     tuple(j["payload"]["face_mapping"]),
                     tuple(j["payload"]["source_mapping_ids"]),
                     j["payload"]["detection"],
                     j["payload"]["selection_state"]["selection_mode"],
                     tuple(j["payload"]["selection_state"]["person_ids"]))
                    for j in scenarios[name]["jobs"]]

        self.assertEqual(wire("sequential"), [
            ("clip_a.mp4", 0, alice, (0,), (alice,), "Selected face", "selected", (0,)),
            ("clip_b.mp4", 1, bob, (1,), (bob,), "Selected face", "selected", (0,)),
            ("clip_c.mp4", 0, alice, (0,), (alice,), "Selected face", "selected", (0,)),
        ])
        self.assertEqual(wire("matrix"), [
            ("clip_a.mp4", 1, bob, (1, 0), (bob, alice), "Selected people", "multi_person", (0, 1)),
            ("clip_c.mp4", 0, alice, (0,), (alice,), "Selected face", "selected", (0,)),
        ])
        self.assertEqual([(j["frame_start"], j["frame_end"]) for j in scenarios["matrix"]["jobs"]],
                         [(5, 50), (1, 600)])
        self.assertEqual(wire("grouped"), [
            ("clip_a.mp4", 1, bob, (1,), (bob,), "Selected face", "selected", (0,)),
            ("clip_c.mp4", 1, bob, (1,), (bob,), "Selected face", "selected", (0,)),
            ("clip_b.mp4", 0, alice, (0,), (alice,), "All faces", "none", ()),
            ("clip_c.mp4", 0, alice, (0,), (alice,), "All faces", "none", ()),
        ])
        self.assertEqual(len(scenarios["cartesian"]["jobs"]), 6)
        # Every emitted job passes the queue's own validation, except the
        # documented stale-primary case which rejects the whole batch.
        for name, scenario in scenarios.items():
            errors = [q._validate_job_payload(j) for j in scenario["jobs"]]
            if name == "stale_primary":
                self.assertEqual(errors, ["source_index must be a non-negative integer"])
            else:
                self.assertEqual(errors, [None] * len(errors), name)
        # No job carries the media id (F1) or a stable person mapping (F2).
        for scenario in scenarios.values():
            for j in scenario["jobs"]:
                self.assertNotIn("target_media_id", j)
                self.assertNotIn("target_person_source_mapping", j["payload"])


@unittest.skipIf(api is None, "needs api.py (the ML stack); not in the light profile")
class BatchMatrixStagingJSThroughTheQueue(_EndToEndBase):
    """The real builders' output, dispatched by the real queue against real
    target contexts named like the JS fixture."""

    def setUp(self):
        super().setUp()
        node = _node()
        if not node:
            self.skipTest("node not available to run the JS staging suite")
        out = os.path.join(self._tmp, "scenarios.json")
        subprocess.run([node, str(CHECKER), "--emit", out], cwd=str(REACT_UI),
                       capture_output=True, text=True, timeout=180, encoding="utf-8")
        with open(out, encoding="utf-8") as fh:
            self.emitted = json.load(fh)
        # The JS fixture names its targets clip_a/b/c; rename ours to match.
        for entry, name in zip(api.list_files_process, ("clip_a.mp4", "clip_b.mp4", "clip_c.mp4")):
            entry.filename = name
        self._set_gallery([(s["id"], s["name"]) for s in self.emitted["fixture"]["sources"]])
        self.alice, self.bob = [s["id"] for s in self.emitted["fixture"]["sources"]]

    def _push(self, scenario):
        snap = q.queue_add_batch({"jobs": self.emitted["scenarios"][scenario]["jobs"]})
        self.assertNotIsInstance(snap, q.JSONResponse, getattr(snap, "body", b""))
        self._drain()
        return snap

    def test_sequential_recipe_from_the_real_builder(self):
        self._push("sequential")
        self.assertEqual(self._states(), [("COMPLETED", "")] * 3)
        self._assert_run(self.ran[0], target=0, gallery_indices=[0], facesets=[self.alice],
                         person_ranks=[0], source_index=0)
        self._assert_run(self.ran[1], target=1, gallery_indices=[1], facesets=[self.bob],
                         person_ranks=[0], source_index=0)
        self._assert_run(self.ran[2], target=2, gallery_indices=[0], facesets=[self.alice],
                         person_ranks=[0], source_index=0)

    def test_per_file_matrix_from_the_real_builder(self):
        self._push("matrix")
        self.assertEqual(self._states(), [("COMPLETED", "")] * 2)
        self._assert_run(self.ran[0], target=0, gallery_indices=[1, 0],
                         facesets=[self.bob, self.alice], person_ranks=[0, 1], source_index=0)
        self.assertEqual(self.ran[0]["entry_range"], (5, 50))
        self._assert_run(self.ran[1], target=2, gallery_indices=[0], facesets=[self.alice],
                         person_ranks=[0], source_index=0)

    def test_grouped_from_the_real_builder(self):
        self._push("grouped")
        self.assertEqual(self._states(), [("COMPLETED", "")] * 4)
        self._assert_run(self.ran[0], target=0, gallery_indices=[1], facesets=[self.bob],
                         person_ranks=[0], source_index=0)
        self._assert_run(self.ran[1], target=2, gallery_indices=[1], facesets=[self.bob],
                         person_ranks=[0], source_index=0)
        self.assertEqual([(r["target_index"], r["facesets"]) for r in self.ran[2:]],
                         [(1, [self.alice]), (2, [self.alice])])

    def test_matrix_rank1_with_a_smaller_active_bank_reaches_the_second_person(self):
        """The client-side finding, end to end: staged while B (one person)
        was active, the row addresses A's second person."""
        api._activate_target_media(index=1, refresh=False)
        self._push("matrix_rank1_active_bank_one_person")
        self.assertEqual(self._states(), [("COMPLETED", "")])
        self._assert_run(self.ran[0], target=0, gallery_indices=[-1, 1], facesets=[None, self.bob],
                         person_ranks=[1], source_index=1)

    def test_stale_primary_from_the_real_builder_rejects_the_batch(self):
        res = q.queue_add_batch({"jobs": self.emitted["scenarios"]["stale_primary"]["jobs"]})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(q._snapshot()["jobs"], [])


if __name__ == "__main__":
    unittest.main()
