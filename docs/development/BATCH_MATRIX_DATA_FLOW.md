# Batch Matrix Data Flow

Traced 2026-09-22 from the code, not from the UI copy. Line numbers are as of
commit `d355803`; the function names are the stable handles.

The Batch Matrix (`react-ui/src/components/BatchSwap.jsx`) offers four
strategies. **All four converge on one job shape and one request**: each
strategy is only a different way of producing a list of *staged jobs*, and a
staged job is always *one target × one person→faceset mapping*. There is no
batch-level request that carries a target list; the target list exists only in
the browser until the moment it becomes N single-target jobs in
`POST /api/queue/add_batch`. Everything below the queue is the ordinary
single-render path (`app/api.py:_run_swap`) run once per job.

```
BatchSwap.jsx                       app/routes_queue.py            app/api.py                   app/roop/
────────────────────────────────    ───────────────────────────    ─────────────────────────    ──────────────────────────────
GET /api/state ─► targets[],        POST /api/queue/add_batch      _run_swap(payload)           processing_request.
  source_faces_info[], target_*     {jobs:[...]}                    ├ _target_index_for_ref       normalize_processing_request
      │                                │                            ├ _canonical_processing_      ├ resolve_source_mapping_ids
strategy → stagedJobs[]              _validate_job_payload           │   request                  ├ source_index_mapping_errors
  (one job per target)               _normalize_job                  ├ mapped_facesets            └ resolve_selected_source_index
      │                                └ _job_selection (FROZEN)    ├ batch_process_regular ───► core.batch_process_regular
createJobPayload()                   app/queue.json                  ├ _outputs_since               └ ProcessMgr
      │                                │                            ├ _record_run_history               ├ selected_routing.
enqueueStagedJobs()                  _loop → _run_one (per job)      │   (run_history.json)              │   compute_selected_assignment
      │                                ├ target by NAME              └ project checkpoint                └ process_face
      ▼                                ├ _selection_invalidation
                                       ├ source by id/name
                                       ├ _apply_segment
                                       └ _run_swap(payload) ─────►
```

## Vocabulary

| Term | What it is in code |
|---|---|
| **target** | One entry of `list_files_process` (a `ProcessEntry`: file path, frame range, `media_id`). Exposed by `GET /api/state` as `targets[i] = {id, media_id, name, frames, start_frame, end_frame, fps, preview_available}` (`app/api.py:_target_entry_dict`). |
| **faceset / source** | One entry of `roop_globals.INPUT_FACESETS` (a `FaceSet` with 1..n reference faces), added through `/api/source/add`, `/api/source/add-folder` or the Faceset Library. `GET /api/state.source_faces_info[i] = {id, name, count, poses}`; `id` is the absolute source path or `memory:<uuid>` (`app/source_gallery.py:_get_source_faces_info`). |
| **person rank** | Display rank of a captured *target person* in the ACTIVE target's person bank (`target_groups` / `target_names` in `/api/state`). The person bank is per target media. |
| **mapping** | `[{personRank, sourceIdx}]` in the UI; `face_mapping[rank] = sourceIdx | -1` on the wire. `-1` (`SKIP`) means "do not swap this person". |
| **staged job** | Browser-only object `{target_name, target_index, source_index, source_name, mappings, frame_start, frame_end, label, payload}`; becomes one queue job. |

## 1. State the strategies read (`refreshBackendState`, BatchSwap.jsx:90)

`GET /api/state` → `targets`, `source_faces`, `source_faces_info`,
`target_faces`, `target_groups`, `target_names`. Note two things the
strategies inherit from this single read:

* `target_groups`/`target_names` describe the person bank of the target that
  is active **at page load**, and `createJobPayload` normalizes every job's
  `selection_state` against that one bank (`normalizeTargetSelectionState(…,
  targetGroups)`). Rank→person-id conversion for the *job's own* target
  happens later, server-side, at dispatch (§5).
* `targets[i].media_id` is available here but is **not** forwarded into the
  job (see §4, finding F1).

## 2. Strategy → staged jobs

All four call `createJobPayload(mappings, swapMode, overrides)`
(BatchSwap.jsx:270) once per (target, mapping) pair. Which UI state carries the
target list and the assignment differs; the resulting job is identical in shape.

| Strategy | UI state that carries the **target list** | UI state that carries the **faceset assignment** | Jobs generated | Builder |
|---|---|---|---|---|
| **One-to-many** (`batchMode === 'one_to_many'`) | `mode1SelectedTargets: number[]` (target indices; defaults to every target) | `mode1Mappings: [{personRank, sourceIdx}]` — ONE mapping applied to every selected target; `mode1SwapMode`, `mode1Enhancer`, `mode1FaceDistance` | one per selected target | `generateMode1Jobs` (575) |
| **Grouped** (`'grouped'`) | `groups[g].targetIndices: number[]` per group | `groups[g].mappings` + `swapMode`/`enhancer`/`faceDistance` per group | one per (group, target in group); a target in two groups yields two jobs | `generateGroupJobs` (663) |
| **Per-file matrix** (`'matrix'`) | every `targets[i]` whose `matrixConfig[i].enabled` is true | `matrixConfig[i].mappings` (per target), plus per-target `swapMode`, `enhancer`, `faceDistance`, `frameStart`, `frameEnd`; `autoMatchFacesetsToTargets` (414) fills `mappings=[{0, bestSourceIdx}]` by filename-token match | one per enabled target | `generateMatrixJobs` (738) |
| **Recipe matrix** (`'recipes'`) | `targets[]` (all of them) | none stored — the recipe computes it: Cartesian = every `(target, source)` pair with `[{personRank:0, sourceIdx:s}]`; Sequential = `sourceIdx = tIdx % sourceFaces.length`; Segment splitter = one target, `[{0, 0}]`, N frame ranges | Cartesian: `targets × sources`; Sequential: `targets`; Splitter: N segments | `recipeCartesianProduct` (455), `recipeSequentialMatch` (487), `splitTargetIntoSegments` (519) |

Preconditions enforced in the browser only: every builder refuses with a toast
when `sourceFaces.length === 0` ("Add a source faceset first") and when its
target list is empty. A `mappings` entry whose `sourceIdx` is not a valid
gallery index is not refused — it is normalized to `-1` (§3).

## 3. The per-job payload (`createJobPayload`, BatchSwap.jsx:270)

`mappings` → a **dense** `face_mapping` indexed by person rank:

```
maxRank = max(personRank); face_mapping = Array(maxRank+1).fill(-1)
face_mapping[personRank] = sourceIdx            // per mapping row
face_mapping = normalizeSourceMapping(face_mapping, sourceFaces.length)
                                                // out-of-range / NaN → -1, never 0
source_mapping_names[r] = source_faces_info[face_mapping[r]].name | null
source_mapping_ids[r]   = source_faces_info[face_mapping[r]].id   | null
primarySourceIdx        = mappings[0].sourceIdx (if valid) else -1
```

The server allocates every gallery id (`_get_source_faces_info`: the absolute
source path, else `memory:<uuid>`), so `.id` is always present when the
`source_faces_info` entry is. The client's `memory-slot-<idx>` fallback only
fires when `source_faces` (thumbnails) is longer than `source_faces_info`;
such an id matches nothing at render time and resolves to `-1` (§6).

Returned `payload` = `{...FACESWAP_DEFAULTS, ...settings}` (the whole Settings
snapshot) plus these job-specific fields:

```jsonc
{
  "enhancer": "<overrides.enhancer | settings.selected_enhancer>",
  "detection": "<swapMode>",              // 'Selected face' | 'Selected people' | 'All faces' | 'First found' | ...
  "output_method": "...", "video_method": "...", "upscale": "...",
  "mask_engine": "...", "mask_engine_2": "...", "clip_text": "...", "sam2_model_size": "...",
  "track_identities": bool, "autorotate": bool,
  "face_distance": <overrides.faceDistance | max_face_distance>,
  "blend_ratio": float, "num_swap_steps": int,
  "auto_fallback": <autoFallbackEnabled>,   // retry with enhancer=None on error
  "face_mapping":          [<sourceIdx | -1>, ...],   // ← THE per-person faceset assignment
  "source_mapping_names":  [<name | null>, ...],
  "source_mapping_ids":    [<faceset id | null>, ...], // ← stable identity of each assignment
  "selected_source_name":  "<name of mappings[0]>" | null,
  "selected_source_id":    "<id of mappings[0]>"   | null, // ← becomes processing_selection.source_identity_id
  "selection_state": {                             // ← WHICH target people are addressed, by RANK
    "selection_mode": "selected" | "multi_person" | "none",
    "person_id": <rank> | null,  "person_ids": [<rank>, ...],
    "target_reference_index": null, "target_detection_index": null,
    "track_id": null, "target_media_index": null
  }
}
```

`selection_mode` follows `swapMode`: `'Selected face'` → `selected` with
`person_id = lowest personRank`; `'Selected people'` → `multi_person` with every
mapped rank; anything else → `none` (no people addressed; the mapping still
translates the highlighted source, §6). What it does **not** contain:
`target_media_id`, `target_index` (the queue supplies that at dispatch),
`processing_selection` (the queue builds it), `target_person_source_mapping`
(BatchSwap addresses people by rank, not by stable person id).

## 4. The request: `POST /api/queue/add_batch` (`enqueueStagedJobs`, BatchSwap.jsx:888)

Sent through `useQueue().addMany` (`react-ui/src/components/faceswap/useQueue.js:102`).
**Identical for all four strategies**; N staged jobs → N entries:

```jsonc
POST /api/queue/add_batch
{ "jobs": [
    { "target_name":  "clip_a.mp4",            // ← the target, BY BASENAME (no media_id, no index)
      "source_index": 2,                       // primarySourceIdx (gallery position at staging time)
      "source_name":  "alice.fsz",             // ← primary faceset name (dispatch re-resolves by this)
      "source_id":    "G:\\...\\alice.fsz",    // ← primary faceset stable id (dispatch prefers this)
      "payload":      { …§3… },                // ← face_mapping / source_mapping_ids / selection_state live HERE
      "frame_start":  1, "frame_end": 812,     // segment; recipes/matrix may narrow it
      "label":        "Matrix | clip_a.mp4 (P#1➔F#3)" },
    …
] }
```

Server (`routes_queue.py:queue_add_batch`, 402): `_validate_job_payload` on
every entry first — `source_index` must be a non-negative int, `frame_*` ints,
`frame_end ≥ frame_start`; any failure rejects the **whole** batch with
`400 {"message": "<first error>"}` and nothing is appended. Otherwise every
entry goes through `_normalize_job` (326), the list is appended under the lock,
`app/queue.json` is written, and the response is the full queue snapshot:

```jsonc
200
{ "schema_version": 2,
  "job_states": ["QUEUED","PREPARING","PROCESSING","PAUSE_REQUESTED","PAUSED","COMPLETED","FAILED","CANCELLED","INTERRUPTED","RECOVERABLE"],
  "running": false, "paused": false, "current": null,
  "jobs": [ { "id": "3f9c1a2b7d4e", "position": 1, "state": "QUEUED", "status": "pending",
              "target_name": "clip_a.mp4", "target_media_id": "",          // ← empty for BatchSwap jobs (F1)
              "source_index": 2, "source_name": "alice.fsz", "source_id": "G:\\...\\alice.fsz",
              "payload": { … },
              "processing_selection": {                                      // ← FROZEN at queue time (_job_selection, 303)
                  "schema": 1, "request_id": "…", "selection_version": null, "mapping_version": null,
                  "target_media_id": null,
                  "target_person_id": "0", "target_person_ids": ["0"],       // still RANKS here
                  "target_reference_face_id": null,
                  "source_identity_id": "G:\\...\\alice.fsz",                // from payload.selected_source_id
                  "detection_mode": "Selected face",
                  "target_person_source_mapping": {},                         // ← EMPTY for BatchSwap (F2)
                  "selection_state": { … } },
              "frame_start": 1, "frame_end": 812, "label": "…",
              "error": "", "added": 1789…, "started": 0.0, "finished": 0.0,
              "progress": {…}, "outputs": [], "cancel_requested": false, "recoverable": false, "project_id": "" },
            … ] }
```

The queue is then driven by `POST /api/queue/start` (the UI's "enqueue and
start" calls it right after). Nothing about a job changes between here and
dispatch except through `POST /api/queue/update`.

## 5. Dispatch: `_run_one` (routes_queue.py:558), once per job, in order

1. **Target resolution.** `target_media_id` is empty for every Batch Matrix
   job, so the *legacy* branch runs: `os.path.basename(target_name)` is matched
   against the basenames of `list_files_process`. Zero matches →
   `FAILED "target "<name>" is no longer loaded"`; two or more (the same
   filename loaded twice) → `FAILED "legacy target … is ambiguous; requeue it"`.
   One match → `_activate_target(index=idx)` loads that media's person bank.
2. **Selection invalidation** (`api.py:_selection_invalidation_for_active_context`,
   1121). The frozen `processing_selection` is checked against what is loaded
   NOW for *that* target: ranks in `target_person_id(s)` are converted to that
   media's stable person ids; a rank the media does not have, a person that was
   removed, or `source_identity_id` not in the current gallery →
   `FAILED "selection invalidated: <reasons>"`. Only `source_identity_id` (the
   *primary* faceset) is checked — `target_person_source_mapping` is `{}` for
   these jobs, so the other per-person assignments are **not** validated here.
3. **Source re-resolution** (619): `source_index` is re-derived from
   `source_id` (case-insensitive) or, failing that, `source_name`, against the
   current `source_faces_info`; the stored numeric index is only the fallback.
   Result → `state.selected_input_face_index`.
4. `_apply_segment` (532) writes `frame_start/frame_end` onto the
   `ProcessEntry` (clamped to the file's total frames).
5. `payload["target_index"] = idx`, `payload["processing_selection"] = frozen
   selection`, a project checkpoint is created (`_create_project`), state →
   `PROCESSING`, and `_run_swap(payload)` blocks until the render ends. If it
   ends with an error and `payload.auto_fallback` is true and the enhancer was
   not `None`, it runs once more with `enhancer = "None"`.
6. Outcome → `COMPLETED` / `FAILED` (`_progress.error`) / `CANCELLED` /
   `INTERRUPTED` (deliberate stop); `job.outputs = _outputs_since(before)`.

## 6. Render: `_run_swap` → the pipeline (api.py:4603)

* `_canonical_processing_request` (1057) rebuilds the selection from the
  frozen object, converts ranks → stable ids for the active target
  (`_target_selection_for_payload`), and calls
  `roop.processing_request.normalize_processing_request` (222) with the
  current gallery's ids/names.
* **The faceset assignment is resolved by identity, not position**:
  `resolve_source_mapping_ids(face_mapping, source_mapping_ids, current_ids)`
  (106) maps each `source_mapping_ids[r]` to its *current* gallery index; an id
  no longer present → `-1`. Only if `source_mapping_ids` is absent does it fall
  back to names, then to the raw numeric `face_mapping`.
  `source_index_mapping_errors` (58) runs on the already-resolved list, where
  `-1` is legal, so a removed id is never reported as an error (F4).
* `source_index` (the single-source slot used by `selected` / non-person
  modes) is the mapped-list position of the selected person's source, or
  `-1` if that person maps to nobody (`resolve_selected_source_index`, 207 and
  the `single_person` block at 322-341).
* `mapped_facesets(source_index_mapping, swap_mode)` (api.py:189) builds the
  person-ordered `FaceSet` list handed to the swap: index `r` is
  `INPUT_FACESETS[source_index_mapping[r]]`, or an **empty `FaceSet()`** when
  the entry is `-1`. Mode `all_input` opts out and keeps gallery order.
* `core.batch_process_regular(…, input_facesets=run_facesets,
  selection_state=…, processing_request=…)` → `ProcessMgr`, per frame:
  * `selected` / `selected_multi`: `selected_routing.compute_selected_assignment`
    pairs detected faces with target people by identity distance, then picks
    `src_index = selected_index` (single person) or `rank[g]` (several). A
    negative / out-of-range index is refused with the audit reason
    `refused: no source faceset for that person` (`selected_routing.py:38,157`).
  * `process_face(face_index, …)` (ProcessMgr.py:4339) reads
    `input_face_datas[face_index]`; when that `FaceSet` has no faces
    (`inputface is None`, 4738) it **returns the frame untouched**.
* Outputs: every file new or newer in `roop_globals.output_path` since the
  pre-run snapshot (`post_swap._outputs_since`). History:
  `_record_run_history(payload, produced)` (def 3902, called at 4999)
  prepends `{id, time, outputs:[basenames], settings:<payload minus
  _HISTORY_STRIP (3872) = face_mapping, enhancer, detection, video_method,
  upscale, clip_text, face_distance, autorotate>, duration_s, frames, fps}`
  to `app/run_history.json`
  (`GET /api/history`). `source_mapping_ids`, `selected_source_id`,
  `selection_state` and `processing_selection` survive in `settings`;
  `face_mapping` does not (F5). The project checkpoint is marked `COMPLETED`.

## 7. What happens when a target has no faceset, or its faceset is gone

"No faceset assigned" can mean three different things on the wire. The
outcome depends on the swap mode the job was staged with, not on the strategy.

| Situation | Wire form | Queue time | Dispatch (`_run_one`) | Render |
|---|---|---|---|---|
| **No sources loaded at all** | — | never reaches the wire: every builder refuses ("Add a source faceset first") | — | — |
| **Row 0 (the primary) points at a gallery index that does not exist** (preset imported with `sourceIdx: 7`; or a source was removed after the mapping was built — BatchSwap never remaps its `mappings` on removal) | `source_index: -1` on the job, `selected_source_id = null`, `face_mapping[0] = -1` | **`400 {"message": "source_index must be a non-negative integer"}` — the ENTIRE batch is rejected**, nothing is queued (`_validate_job_payload`, 359) | — | — |
| **A row ≥1 points at a gallery index that does not exist** | `face_mapping[r] = -1`, `source_mapping_ids[r] = null` | accepted | passes (nothing to check for that rank) | as "person mapped to skip" below |
| **Person rank mapped to skip** (`-1`), mode `Selected face` (single person) | `face_mapping[r] = -1` | accepted | passes | `source_index = -1` → every face of that person hits `refused: no source faceset for that person`; frame written unswapped; job `COMPLETED` |
| **Person rank mapped to skip**, mode `Selected people` with ≥2 people | `face_mapping[r] = -1` | accepted | passes | `mapped_facesets` puts an empty `FaceSet()` at rank `r`; routing counts the face as `swapped (identity match)` but `process_face` returns it untouched (F3). Job `COMPLETED`, output shows that person un-swapped |
| **Person rank mapped to skip**, mode `All faces` / `First found` / gender | `face_mapping[r] = -1` | accepted | passes | `resolve_selected_source_index` → `-1` when the highlighted gallery source is mapped to nobody → `refused: no valid selected source`; nothing swapped |
| **Target has no captured people** and mode is `Selected face` / `Selected people` (the strategies' default) | `selection_state.person_id = 0` (a rank) | accepted | `FAILED "selection invalidated: target person 0 was removed"` — rank 0 cannot be resolved to a person id on that media | never runs |
| **Primary faceset (row 0) removed from the gallery after queueing** | `source_id` / `selected_source_id` = the old id | accepted | `FAILED "selection invalidated: source <id> was removed"` (`source_identity_id` check) | never runs |
| **A non-primary faceset (row ≥1) removed after queueing** | `source_mapping_ids[r]` = the old id | accepted | passes (only the primary is validated, F2) | `resolve_source_mapping_ids` → `-1` for that rank → behaves as "mapped to skip" above, silently |
| **Faceset removed and re-added** (same file) | same `id` (absolute path) | — | re-resolves to the new gallery position by id | normal |
| **Gallery reordered** | ids unchanged | — | re-resolved by id | normal — indices are never trusted once ids exist |
| **Target file removed after queueing** | `target_name` | — | `FAILED "target … is no longer loaded"` | — |
| **Same filename loaded twice** | `target_name` | — | `FAILED "legacy target … is ambiguous; requeue it"` | — |

A `FAILED` job does not stop the batch; `_loop` (743) moves to the next
`QUEUED` job. Failed jobs stay in the snapshot with `error` set and can be
retried from the queue panel (`POST /api/queue/retry`).

## 8. Findings worth fixing (not fixed here)

* **F1 — `target_media_id` is dropped.** `/api/state.targets[i].media_id`
  exists and `_run_one` prefers it, but `enqueueStagedJobs` forwards only
  `target_name`, so every Batch Matrix job takes the legacy basename path and
  two targets with the same filename cannot be batched at all. The fix is one
  field in the job (`target_media_id: targets[j.target_index].media_id`).
* **F2 — only the primary faceset is validated at dispatch.** BatchSwap sends
  `face_mapping` + `source_mapping_ids` (rank-indexed) rather than
  `target_person_source_mapping` (person-id-keyed), so `processing_selection.
  target_person_source_mapping` is `{}` and a removed non-primary faceset is
  silently downgraded to a skip at render time instead of failing the job.
* **F3 — multi-person skip is audited as swapped.** With ≥2 selected people a
  rank mapped to `-1` becomes an empty `FaceSet()`; `compute_selected_assignment`
  only checks the index range, so the face is counted `swapped (identity
  match)` and then returned untouched by `process_face`. The single-person path
  refuses it correctly.
* **F4 — `source_mapping_errors` cannot see a removed id.** It runs after ids
  are resolved to `-1`, and `-1` is legal, so the `[Selection] mapping_errors`
  log line never fires for this case.
* **F5 — history strips `face_mapping`.** `run_history.json` keeps
  `source_mapping_ids` and `selected_source_id`, so the assignment is still
  reconstructible, but the `-1` skips are only implied by `null` ids.
* **A stale primary row rejects the whole batch.** `source_index: -1` (row 0
  mapped to a gallery index that no longer exists) fails `_validate_job_payload`
  and `add_batch` is all-or-nothing, so one stale job blocks every other
  staged job with a 400 the UI shows as "Failed to enqueue jobs". The
  client normalizes the mapping to `-1` deliberately (never to source 0),
  but does not refuse to stage it.
* **Default mode needs captured people.** Every strategy defaults to
  `'Selected face'`, which requires the target to have a captured person; a
  fresh multi-file batch with no captures fails every job at dispatch with
  "target person 0 was removed". `All faces` / `First found` do not.
