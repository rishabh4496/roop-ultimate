# React UI 1.0 / 2.0 Migration Audit

**Method.** The comparison below is derived from the repository, not from the
UI2 contract documents. Three independent passes:

1. **Endpoint diff** — every `/api/...` literal in `react-ui/src` and
   `react-ui-v2/src`, diffed against the 104 routes actually registered by
   `app/api.py` + `app/routes_*.py`.
2. **Screen/feature diff** — component inventory and per-domain endpoint
   coverage.
3. **Backend reachability** — for each V2-only call, whether the route exists
   and whether any V2 screen actually invokes it.

**Headline numbers.** V1 is 22,093 LOC across 50 modules; V2 is 5,170 LOC
across 54 modules. V1 calls 94 distinct endpoint literals, V2 calls 88.

Baseline before any change: backend suite **1831 tests, OK (1 skipped)**;
`react-ui` production build clean.

---

## A. The only genuinely V2-unique endpoint calls

| V2 call | exists in backend? | reachable from a V2 screen? | verdict |
| :-- | :-- | :-- | :-- |
| `/api/projects`, `/api/projects/{id}/validate\|load\|resume` | **YES** (`routes_projects.py`) | YES — `ProjectsPanel.jsx` | **PORT** |
| `/api/queue/cancel` | **YES** (`routes_queue.py:778`) | YES — `useQueue.js` / `QueuePanel.jsx` | **PORT** |
| `/api/runtime/state` | YES (`api.py:3377`) | YES — `useOperationsStatus` | **PORT AS EMBEDDED FIELD, not a second poll** (see C.6) |
| `/api/system/hardware` | YES (`routes_diagnostics.py:362`) | YES — `useOperationsStatus` | **PORT** (Settings evidence card) |
| `/api/facemgr/save` | **NO — route does not exist** | no (adapter only) | **REJECT — phantom** |
| `/api/faceset/library/reveal` | **NO — route does not exist** | no (adapter only) | **REJECT — phantom** |
| `/api/history/clear` | **NO — route does not exist** | no (adapter only) | **REJECT — phantom** |

The three phantoms sit in `src/adapters/*.js` wrappers that no screen imports.
The UI2 contract's "complete endpoint parity (87 routes)" was satisfied by
*writing wrapper functions*, not by wiring them to controls — the same
"bound to something nothing reads" failure class this project has hit before.
They cause no user-visible break because nothing calls them; they are also
therefore not features.

## B. Endpoints V1 calls that V2 dropped

`/api/advisor`, `/api/facemgr/frame`, `/api/faceset/library/open`,
`/api/faceset/library/rebuild_thumbs`, `/api/livecam/{start,stop,status,frame}`,
`/api/profiles`, `/api/queue/{clear,duplicate,update}`, `/api/runtime_estimate`.

Live camera, the settings advisor, saved profiles, queue duplicate/update/clear
and the learned runtime estimate exist only in V1. This is the main reason V1 is
the correct base.

---

## C. Feature matrix

Legend — **V2 better?** answers "is V2's implementation superior *as
implemented*", not "is V2's idea good".

| # | Feature | V1 status | V2 status | V2 better? | Required in final? | Port to V1? | Notes |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | **Persistent projects** (list/validate/load/resume) | **ABSENT** | `ProjectsPanel.jsx`, 4 routes | **YES** | YES | **YES** | Backend creates a project on *every* `/api/swap` (`api.py:2854`). V1 users already generate them and have no way to see or resume one. |
| 2 | **Canonical job states** | reads legacy `status` (5 values) | reads `state` (10 values) | **YES** | YES | **YES** | Backend emits **both** (`routes_queue.py::_snapshot`). V1 collapses PREPARING / PAUSE_REQUESTED / PAUSED → "Running", CANCELLED / INTERRUPTED → "Stopped", RECOVERABLE → "Pending". |
| 3 | **Per-job cancel** | ABSENT | `/api/queue/cancel` | **YES** | YES | **YES** | Distinct from `remove` (refuses the running job) and `stop` (stops the whole queue): cooperatively cancels *one* job, running or not. |
| 4 | **Per-job progress in queue rows** | ignores `job.progress` | renders fraction + phase | **YES** | YES | **YES** | Backend already emits `job.progress` per row. |
| 5 | **URL routing / deep links** | none; always opens Face Swap | hash router, 8 routes | **YES** | YES | **YES** | Pinokio reloads the frontend on every tab switch, so V1 silently loses your place. |
| 6 | **Environment health evidence, always-on** | only inside a run (Processing tab) | Settings card | **PARTLY** | YES | **YES** | V1 renders *all 14* runtime sections but only during/after a render. A standing Settings card closes §16. |
| 7 | **Update centre** | ABSENT | static notice card, no call | NO (it is a placeholder) | YES | **YES, improved** | `app/update_manager.py` (868 LOC) already does manifest gating, snapshot, health check and rollback — but is CLI/Pinokio-only. V1 gets a **read-only** `/api/update/check` surfacing SAFE / REQUIRES REVIEW / UNVERIFIED. It cannot install. |
| 8 | Live preview | `InteractivePreview.jsx` 1304 LOC: magnifier lens, mask brush (paint/erase/size), compare slider, auto-swipe, blend, diff, crossfade, zoom/pan, popout window | 443 LOC + 6 helper components, same ideas | **NO** | YES | NO | V2 is a reimplementation of a subset. |
| 9 | Timeline / scrubber | `Timeline.jsx` 788 LOC, filmstrip, in/out, chapters, speeds; `usePlaybackBuffer` | none | NO | YES | NO | V2-only regression. |
| 10 | Batch matrix | `BatchSwap.jsx` 2162 LOC, 4 strategies, segment splitting | `BatchScreen.jsx` 151 LOC | NO | YES | NO | |
| 11 | Settings surface | `Settings.jsx` 613 + `settingsCatalog.js`, `/api/advisor`, `/api/profiles` | **no settings at all** — theme + evidence + storage only | NO | YES | NO | V2's "Settings" screen cannot change a single processing parameter. |
| 12 | Themes | 37 themes + ThemeStudio + ThemeGallery | 7 themes | NO | YES | NO | §18 explicitly cautions against porting visual experimentation. |
| 13 | Storage cleanup | `StorageManager.jsx`, `/api/storage`, `/api/storage/delete` | same two routes | NO — identical | YES | NO | Already at parity, incl. SAFE / REVIEW / PROTECTED gating. |
| 14 | Pause / resume | `App.jsx` + `Processing.jsx` → `/api/pause`, `/api/resume` | same two routes | NO — identical | YES | NO | Real cooperative pause at a safe checkpoint, already in V1. |
| 15 | Telemetry / structured runtime report | all 14 sections incl. PROJECT + CHECKPOINT, log lines classified by `level` / `category`, per-part tabs | 6 fields on a card | **NO** | YES | NO | V1 already consumes the authoritative state via `progress.runtime`. |
| 16 | Benchmark / auto-tune | `BenchmarkPanel.jsx` 581 LOC | none | NO | YES | NO | V2-only regression. |
| 17 | Run history + settings diff | `RunHistory.jsx` 536 + `settingsDiff.js` | `HistoryScreen.jsx` 84 | NO | YES | NO | |
| 18 | Outputs gallery + compare | `Gallery.jsx` 750 + `OutputCompare.jsx` 340 | `GalleryScreen.jsx` 232 | NO | YES | NO | |
| 19 | Live camera | `/api/livecam/*` | none | NO | YES | NO | V2-only regression. |
| 20 | Dedicated Processing screen | `Processing.jsx` 419 + terminal + quality report | none (inline) | NO | YES | NO | |
| 21 | Notifications | `notify()` + `useRunCompleteAlert` (sound + OS notification) | `NotificationCenter.jsx` toast list | NO | YES | NO | |
| 22 | Command palette | `CommandPalette.jsx` (Ctrl+K) | none | NO | YES | NO | The contract claimed V2 had it. It does not. |
| 23 | **Offline operation** | zero external URLs; fonts self-hosted | **imports Google Fonts over the network** (`styles.css:1`) | **NO — regression** | YES | **NO — actively rejected** | Porting this would break §13. |
| 24 | Extras / AI enhancers | 3 routes | same 3 routes | NO — identical | YES | NO | |
| 25 | Face Manager | 8 routes incl. `/frame` | 7 routes + 1 phantom | NO | YES | NO | |

---

## D. Migration decisions

**Porting (7 items):** rows 1–7 above.

**Rejected, with reason:** everything in rows 8–25. In every case V1's existing
implementation is equal or strictly larger in capability; three V2 behaviours
(the Google Fonts import, the phantom endpoints, the settings-free Settings
screen) would be regressions if carried across.

**Not a port but required by §14:** the read-only update check endpoint.
Justification is in row 7 — the compatibility logic already exists and is
verified; only a read-only view of it was missing from every UI.

---

## E. What migrating row 1 uncovered

Persistent projects were not merely missing a UI. Writing one exposed **three
pre-existing defects that made the whole system non-functional**, each
invisible for the same reason: the backend had written a project record on
every `/api/swap` since that code existed, and **no client had ever listed
one**, so nothing exercised the path far enough to fail.

| defect | effect | fixed by |
|---|---|---|
| `runtime_identity` unwrapped a provider one level, so a LIST of `(name, options)` tuples stored the whole `('TensorrtExecutionProvider', {...})` literal | every project written under TensorRT was reported RECOVERABLE and then permanently refused by the machine that wrote it (4 of 5 records here) | `normalize_provider`, `test_project_provider_identity.py` |
| `_resume_context["base"]` came from `safe_frame`, which advances on frame index with nothing committed | a resumed render reported 100% for its entire duration — observed at "frame 777 / 899" with progress 1.0 | committed-segment derivation, `test_resume_progress_base.py` |
| `api.py` called `_project_checkpoint.manifest_path`, which lives on `roop.segment_writer`, inside a broad `except` | `AttributeError` on **every** segment commit; `update_checkpoint` never reached from the writer path, so no project ever recorded a committed segment, manifest or partial file | one-line fix, `test_checkpoint_segment_commit.py` |

All three verified on hardware. After the third fix an interrupted 900-frame
render recorded a real segment for the first time:

    segments  : [{'file': '.d4__temp.seg0000.mp4', 'frames': 899,
                  'bytes': 20679948, 'sha256': 'a948e338a1ae...'}]
    manifest  : ...\output\d4__temp.mp4.resume.json

Before it, that was `[]` and `''` on every run this application has ever done.

**The general lesson, and it is the third instance of it in this repo:** a
feature with no consumer is not "working", it is untested. The swap audit, the
return code and the green suite all reported success throughout.
