# React UI 2.0 migration review and plan

Audit date: 2026-09-02

This document is the Stage 17A non-destructive retirement review. It records
what is verified in the repository and what must be completed before React UI
1.0 can be retired. It is not authorization to delete V1 or switch the
production launcher.

## Decision - SUPERSEDED for activation, UNCHANGED for retirement (Stage 18, 2026-09-02)

**React UI 2.0 has been ACTIVATED as the default client.** `start.js` now
re-exports `start_react_v2.js` and the Pinokio menu's default action starts V2.
That is an ACTIVATION, not a migration: nothing was deleted, and rollback is one
line in `start.js` plus the `react-ui-v1` tag created in Stage 18.

**V1 RETIREMENT REMAINS NOT AUTHORIZED.** Every exit condition below still
stands. Stage 18 closed several of them - a physical RTX 3060 session, real
browser acceptance for both clients, an immutable rollback tag, and a project
record surviving a real backend restart - but the decisive one did not move:
V2 is measurably not a replacement. V1 references 87 API routes to V2's 31, 62
of them V1-only, and renders 179 interactive controls to V2's 47. Until those
families exist in V2, V1 is the supported route to them and must stay.

The Stage 17A audit below is retained verbatim as the record of what was true
before activation.

### Stage 17A decision (historical)

React UI 1.0 must remain available. React UI 2.0 is not ready for migration.
The Stage 16 acceptance matrix contains `FAIL`, `BLOCKED`, and `NOT TESTED`
rows, including the RTX 4070 still-image failure, missing RTX 3060 evidence,
missing browser acceptance, and untested application-close/PC-shutdown
recovery. No V1 files were deleted or renamed in this audit.

## Evidence-based audit

| Criterion | Status | Verified evidence |
|---|---|---|
| V2 complete required functionality | FAIL | `react-ui-v2/README.md` defines a deliberately parallel, partial scope. `CreateScreen.jsx` explicitly renders unavailable functionality, and V1 still contains feature surfaces and API consumers not present in V2. |
| V2 passed the acceptance matrix | FAIL | `docs/development/VALIDATION_MATRIX.md` Stage 16 report explicitly concludes that V2 is not production-ready; several required rows are blocked, failed, or not tested. |
| V2 validated on RTX 4070 | FAIL | Device A runtime evidence exists, but the still-image smoke failed and the long-run quality harness re-measured 71 of 467 gradable `harjot` frames as the other identity. V2 browser execution was not verified. |
| V2 validated on RTX 3060 | BLOCKED | The physical RTX 3060 Laptop was not present on the test host. No Device A result is extrapolated. |
| Persistent projects remain recoverable | BLOCKED | Atomic checkpoint and control-plane validation passed, but application-close and PC-shutdown continuation were not physically tested and V2 browser reload was not verified. |
| Existing user projects remain safe | BLOCKED | Protection and atomic persistence guards are tested, but a complete V1-to-V2 project migration and real-user project reload have not been accepted. |
| V1-specific functionality has a verified V2 replacement | FAIL | V1 contains verified consumers for face management, faceset library, extras, livecam, history, quality, benchmark, advanced source/target operations, and other controls. V2 does not provide complete replacements; some controls explicitly show unavailable. |
| No required backend API depends uniquely on V1 | BLOCKED | The backend is shared and does not show a React-V1 import dependency, but many backend route families are currently consumed only by V1 in this repository. Replacement coverage and required-route ownership are not established. |
| Pinokio launch behavior remains correct | BLOCKED for V2 migration; PASS for current V1 path | `start_react.js:40-42` launches `react-ui`; `start.js:1` aliases it; `install.js:21-23` and `reset.js:12` manage V1 dependencies. The URL capture at `start_react.js:44-53` follows the required Pinokio pattern, but it does not launch V2. |
| Rollback to V1 remains possible until migration is complete | BLOCKED | V1 remains active, but `react-ui-v1-backup/` is ignored and no `react-ui-v1` tag was found. A tested, immutable V1 rollback artifact and post-rollback validation are not verified. |

## Files and components that must remain now

The following are protected until a later, explicitly authorized migration
gate has passed:

- `react-ui/`, the active V1 client and its V1-specific feature surfaces.
- `react-ui-v2/`, the parallel V2 client and its current source of migration
  evidence.
- `react-ui-v1-backup/`, until its provenance is replaced by a verified,
  independently restorable snapshot or tag.
- `pinokio.js`, `start.js`, `start_react.js`, `start_legacy.js`, `install.js`,
  `reset.js`, `update.js`, and `clean.js`.
- `app/` and all backend route modules, including project, queue, checkpoint,
  storage, diagnostics, export, extras, face-manager, faceset, livecam,
  quality, and update/health paths. These are shared or still V1-consumed;
  none is a safe retirement candidate from this audit.
- Existing project data, checkpoints, partial outputs, completed outputs,
  models, environments, and files referenced by active or resumable jobs.

The current V2 source itself records the incomplete boundary: `CreateScreen.jsx`
uses an explicit unavailable control at line 67 and an unavailable-capabilities
summary at lines 100 and 112; `SettingsScreen.jsx:49-50` states that browser
update checking is unavailable. The V1 backup README records its active React
snapshot and broader tab surface at `react-ui-v1-backup/README.md:3-17`.

## Candidates that may be retired only after the final migration gate

No file is approved for retirement in Stage 17A. The eventual review may
consider the following only after parity, migration, rollback, and dual-device
acceptance are complete:

1. The active `react-ui/` tree, after a verified V2 replacement exists for
   every accepted V1 feature and an immutable V1 rollback artifact is stored.
2. Any V1-only launcher entry, only after Pinokio launch/install/reset/update
   behavior has been switched to V2 and tested end to end. `start_legacy.js`
   is a separate legacy UI path and must not be removed as part of this review.
3. Temporary migration adapters or compatibility code, only after all
   supported projects have been migrated or remain readable without them.

Backend route modules, project schemas, checkpoints, models, outputs, and
cleanup protection code are not retirement candidates merely because V2 does
not currently call them.

## Required compatibility shims before retirement

These are required work items, not implemented claims:

- A V1/V2 feature coverage table mapped to actual backend routes and payloads,
  including V1-only source/target operations, face management, facesets,
  extras, livecam, history, quality, benchmark, output, and export behavior.
- A project-schema compatibility reader/writer that validates source/target
  identity, settings, model/provider/precision identity, checkpoint metadata,
  output state, and application compatibility before continuation.
- A launcher migration switch with an explicit V1 fallback, plus matching
  Pinokio install/reset/update/clean behavior for the selected client.
- A verified V1 snapshot/tag with documented restore steps and a post-restore
  `/api/meta` and known-project smoke test.
- A migration path for V1 settings, profiles, recipes, and local UI state where
  those values affect project interpretation or user-visible behavior.
- A compatibility policy for backend route removal: no route may be removed
  until all retained clients and persisted projects no longer require it.

## Migration risks

- V2 is not feature-complete and currently does not replace several V1 route
  families and workflows.
- The production Pinokio React launcher currently installs and starts V1, so a
  V2 switch would change installation and launch behavior.
- The ignored V1 backup (`.gitignore:33-34`) and absent verified `react-ui-v1`
  tag make rollback provenance weaker than required; the only listed tag was
  `pre-tensorrt-optimization-20260828`.
- Project/checkpoint/output compatibility and partial-output recovery have not
  been accepted through a real close/reopen or shutdown/restart workflow.
- Device B, browser interaction, real offline operation, and final visual
  quality are unverified; Device A has known processing/quality failures.
- Removing backend routes or cleanup paths based only on V2 references could
  break V1, existing projects, recovery, or user data safety.

## Planned rollback procedure (not yet executed)

Until migration is complete:

1. Keep the current V1 launcher and source tree intact. Do not delete V1,
   project data, outputs, checkpoints, models, or environments.
2. If a future V2 launcher trial fails, stop the V2 process and restore the
   last verified launcher/source snapshot, or revert the specific migration
   change using the repository's approved, reviewable Git process.
3. Restore/reinstall the V1 `react-ui` dependencies using the existing
   Pinokio installation path, then launch through `start_react.js` (the current
   verified V1 path). Use `start_legacy.js` only when the legacy Gradio path is
   specifically required.
4. Verify the launcher reaches the loopback API, `/api/meta` responds, the
   provider/model health checks are valid for the actual device, and a known
   project can be opened without altering its source, target, checkpoint, or
   output.
5. Record any failed restore, preserve diagnostics, and do not report the
   migration as successful until the known-project and data-integrity checks
   pass.

The procedure is a plan, not evidence that shutdown recovery or a full V2
rollback has already been tested.

## Exit conditions for a later retirement gate

V1 retirement may be reconsidered only when every Stage 16 matrix row is
accepted at the required status, both physical GPUs have separate evidence,
all V1-specific workflows have verified V2 replacements, persistent projects
round-trip through migration and recovery, Pinokio install/start/reset/update
behavior is switched and tested, and the immutable V1 rollback procedure has
passed. Any `FAIL`, `BLOCKED`, or `NOT TESTED` critical item keeps V1 active.
