# Stages 10–15 — Workstation Hardening Audit

**Range:** commits `53cf2c7` (Stage 10) to `df332de` (Stage 15), all 2026-10-28. **Written:** 2026-10-28, Stage 16.
**Sources:** `docs/CHANGELOG.md`, the six commit messages, the code and tests named below, and one test run made while writing
this report (see §8). Nothing here was re-measured on a GPU; where a claim rests on a changelog entry rather than on something
run today, it says so.

> **Two numbering series.** `docs/audits/STAGE0…STAGE9_*.md` are the *core-engine* audit (detector, alignment, swapper, restorer,
> masks, compositing, tracking). Stages 6–15 of the *workstation* series (this report covers 10–15) are the React-UI / pipeline
> hardening that followed. `docs/CHANGELOG.md` also holds older, unrelated "Stage 10 … Stage 15 brief" entries dated 2026-10-12 to
> 2026-10-19 (colour transfer, pre-pass, CUDA streams, …): those are engine briefs and are not summarised here.

Each stage lists what shipped (with the file that implements it), the **invariants** a later change must not break, and what is
**open**. Per `AGENTS.md`: a claim needs a named site and a test; anything without one is marked *unverified*.

---

## 1. Stage 10 — Runtime Anomaly Watchdog & Crash Autopsy (`53cf2c7`)

**Shipped.** `app/roop/watchdog.py` and `app/roop/crash_autopsy.py` (new). Hooks: `ProcessMgr.py` (tensor sanity check on swap and
enhance output, zero-swap alert, mask-collapse / seam-jump detector, breadcrumbs), `processors/frame/face_enhancer.py`
(`sanitize_frame_output`), `segment_output.py` (`audit_segment_custody`, destination-collision assertion, concat failure capture,
duration / frame-sum parity), `core.py` (safe segment destination, `batch_process` autopsy handler), `api.py` (watchdog lifecycle),
`routes_queue.py` (segment custody and queue-level autopsy), `video_io_hw.py` (`[PERF-WARN] Zero-copy broken`),
`runtime_state.py` + `react-ui/src/components/faceswap/logRing.js` (classifies `[BUG-DETECTED]`, `[WARN]`, `[PERF-WARN]`, `[CUSTODY OK]`).

| Part | Behaviour |
|---|---|
| Tensor catcher | NaN / Inf / all-black on swapper and enhancer ROI output; logs frame, model, bbox |
| Zero-swap alert | N consecutive frames with expected faces (Stage 2) and 0 swaps → warning naming the refusal cause |
| Mask collapse / seam jump | BiSeNet mask area 0 px, or >70 % swing inside one scene cut (reset at cut boundaries) |
| Zero-copy / VRAM | `[PERF-WARN]` when device tensors fall to host copies; alert at 92 % VRAM before a hard OOM |
| Segment custody | unique zero-padded names, never colliding with the final output; receipt `[CUSTODY OK] seg_0002.mp4 \| Frames: 600/600 \| …` |
| Concat failure | ffmpeg command and last 20 stderr lines in a boxed message |
| Crash autopsy | classifies CUDA OOM, TRT deserialisation, ORT provider, broken pipe / NVENC session, disk / permission; prints stage, frame, segment, `file:line`, breadcrumbs, fix hint; writes `logs/crash_<job_id>_<timestamp>.json` (profile, last 50 log events, traceback) |

**Invariants.**
1. **The watchdog may never fail a render.** Every detector observes and logs; none raises into the frame loop.
2. **Scene cuts reset the mask-jump detector** — a cut is not an anomaly.
3. **An intermediate segment may never be written to the final output path** (collision assertion).
4. **A concat is verified by frame and duration sums**, not by exit code (this repo has been bitten by an exit-0 concat that lost frames).
5. The log tags above are a vocabulary the UI parses; renaming one silently drops it from the severity chips.

**Tests.** `app/tests/test_watchdog_and_autopsy.py` — 22 passed (changelog). **Open / unverified.** No record of a real CUDA OOM or
NVENC session-limit crash driven through the autopsy; classification is tested on constructed exceptions.

---

## 2. Stage 11 — Expression & Gaze Fidelity, Identity Crossover Lock, Vector Roto Zones (`e9b9ced`)

**Shipped.** `app/roop/expression_fidelity.py`, `identity_lock.py`, `roto_polygons.py` (new); hooks in `ProcessMgr.py`
(`process_frame`, `process_face`, the identity veto ladder), `procmgr_tracking._assign_track_sources` (pins), `oral_cavity.py`
(`force_wide`, `repair`), `face_bank.py` (per-sample scores), `api.py` (`_apply_stage11_settings`, `anchors` in the preview response,
`identity_confidence` on `/api/target/timeline_tracks`), `settings.py` (6 settings). UI: `faceswap/{rotoModel.js, identityLock.js, RotoLayer.jsx}`,
`InteractivePreview`, `Timeline`.

| Feature | Behaviour |
|---|---|
| Gaze retention | moves the swapped **iris** to the plate's position, keeping its colour and brightness; contrast-matched, never absolute-luma matched |
| Crease transplant | mid-frequency luma at nasolabial folds, inner brows, mouth corners (shading, not geometry) |
| Wide-mouth teeth | opt-in `force_wide` / `repair` (restoring the cavity unconditionally was measured to give grey, doubled teeth) |
| Identity pins | a user box binds the pre-pass track to a source **before** any automatic gate runs |
| `TrackLock` | cross-frame hysteresis on the per-frame veto: `hold`, extra challenge margin, trajectory term, hard `max_defend` ceiling, reset on out-of-order frames; OFF by default |
| Confidence strip | per-person amber cells, 1-click seek, prev / next amber |
| Pen tool (`P`) | Bézier / polygon include / exclude zones with feather and frame range; 2-point nose + chin pin (stateless similarity) |

**Invariants.**
1. **Every new control defaults to a bit-exact no-op** (0 / off). A render with defaults must equal a render without the stage.
2. **`TrackLock` delays a real identity switch and never forbids one** (`max_defend`); any out-of-order frame restores the old memoryless behaviour.
3. **Python and JS roto geometry are pinned to shared reference values**; a change to either side must move both.
4. **`Number(null)` is 0**: a pin without a frame must not become frame 0 (regression test exists).
5. Pen keyboard handlers are scoped to the focusable layer so they do not fire FaceSwap's window handlers (`test_ui_shortcut_keys`).

**Tests.** `test_roto_polygons`, `test_expression_fidelity`, `test_identity_lock`, `test_identity_pins`, `test_stage11_backend`,
`react-ui/.render-check/render-stage11-logic.mjs`, `react-ui/e2e/stage11.spec.js` (18 specs, two mutation-checked),
`tests/verify_stage11_render.py` (real 30-frame render: every new function ran and changed pixels with controls on; none ran at defaults).
**Open.** Wide-mouth threshold (0.22) and crease gain (1.6) on real speech footage; no crossing-clip A/B of the lock (the simulated
crossing shows the hysteresis logic, not the real matcher's flip rate); pins and zones are not persisted across reload; the 3060 was not run;
very pale irises are skipped by the gaze pass. Full-suite comparison against HEAD is recorded as partial.

---

## 3. Stage 12 — Directional Relighting, Specular Recovery, 10-bit / HDR Custody, Zebra Scope (`e3a6b2d`)

> **Documentation gap.** This commit has no message body and `docs/CHANGELOG.md` has **no Stage 12 entry** (Stages 10, 11, 13, 14
> have one). What follows is read from the diff and its test; no real-footage or HDR-clip proof is on record, so none is claimed.

**Shipped.**
- **Relighting** — `procmgr_color.py` `ColorTransferMixin._apply_directional_and_specular_lighting`: (1) ambient shadow / spherical-harmonics
  directional match, (2) specular skin-highlight recovery (high-luminance peaks and rim peaks from the target composited onto the swap, with
  a desaturation term so highlights stay neutral), (3) sub-dermal warmth / halation balance per shadow / mid / highlight zone. Called from
  `apply_color_transfer`. Three settings, all default `0.0`: `specular_highlight_recovery`, `ambient_shadow_match`, `subdermal_warmth_match`
  (`ROOP_SPECULAR_HIGHLIGHT_RECOVERY`, `ROOP_AMBIENT_SHADOW_MATCH`, `ROOP_SUBDERMAL_WARMTH_MATCH`; group "Lighting & Harmonization").
- **10-bit / HDR custody** — `capturer.py` reads `bits_per_raw_sample` / `pix_fmt` (ffprobe) into `bit_depth`; `source_color.py` carries
  `bit_depth`, maps BT.2020 / PQ / HLG tags, adds `format_color_badge()` (`BT.2020 HDR10  10-bit HEVC`) and `plan_for_hdr()`; `ffmpeg_writer.py`
  chooses `yuv420p10le` / `p010le` for ≥10-bit and `-profile:v main10` for `hevc_nvenc`. The badge reaches the terminal dashboard.
- **Zebra scope** — `react-ui/src/components/faceswap/falseColorZebra.js` (`drawZebraOverlay`): clipped highlights (>95 IRE) and crushed shadows
  (<5 IRE) as diagonal stripes, a 55–72 IRE skin-exposure band, Rec.709 luma weights, optional face-ROI-only; mounted in `PreviewCanvas` /
  `InteractivePreview`.

**Invariants.**
1. **Neutral settings (0.0) are a bit-identical no-op** (`test_lighting_specular_directional.py`, test 1).
2. **A 10-bit source must not be quietly written as 8-bit**: bit depth is read from the probe, not assumed from the container.
3. **The zebra scope is a client-side monitor only**; it never alters the rendered frame.

**Tests.** `app/tests/test_lighting_specular_directional.py` (5 tests; needs `cv2`, so the CI light profile skips the module and only a full-profile run exercises it). **Open / unverified.** No 10-bit or HDR clip has been
rendered end to end on record; the relighting passes have no A/B on real faces; no 3060 run; no CHANGELOG entry (add one with the first real measurement).

---

## 4. Stage 13 — Interactive Command Console & Journaled Sub-Segment Crash Recovery (`f8a7aae`)

**Shipped.** `app/roop/debug_snapshot.py`, `app/roop/journal.py` (new); `terminal_dashboard.py` (`_key_listener_loop`, `handle_key`,
`render_hotkey_footer`), `ProcessMgr.py` (snapshot stashing, throttle delay), `segment_writer.py` (`_active_writer`, journal persistence,
ffprobe validation on resume), `project_checkpoint.py` (journal sync), `run.py` (`--resume <job_id>`), `routes_projects.py` / `routes_queue.py` (tail recovery).

| Key | Action |
|---|---|
| `P` | pause / resume dispatch after the current batch; `torch.cuda.empty_cache()` and `release_frame_buffer_pools()` while paused |
| `S` | 2×2 inspection grid (Original \| Mask \| Raw Swap \| Final) + tensor stats to `logs/snapshots/frame_<N>.png/.json` |
| `D` | per-model latency histograms and per-core CPU |
| `T` | thermal cooldown delay in dispatch |
| `Q` | flush active segment writer, write `.checkpoint.json` to `logs/`, exit 0 |

**Journal.** Write-ahead and atomic (`journal.json.tmp` → `journal.json`, exponential back-off on Windows) recording SHA-256 and nanosecond
mtime of source and target, the recipe hash, verified completed segments and the resume frame index. On resume, `ffprobe` validates each segment
and **only the truncated tail is discarded**; intact segments are kept.

**Invariants.**
1. **The key listener never starts without a TTY** (CI, pipes, `TERM=dumb`): a hung `stdin` read in CI is a failure. Pinned by `test_non_tty_mode_selection_and_no_interactive_thread`.
2. **A journal write is atomic**; a crash mid-write must leave the previous journal intact.
3. **Resume trusts nothing it did not verify**: source / target hash and recipe hash must match, and each segment must pass `ffprobe`, before it is reused.
4. **Recovery discards the tail only**, never intact segments.

**Tests.** `app/tests/test_interactive_console_and_journal.py` — 12 passed (changelog); Stage 15 re-covers the journal and non-TTY paths (§7).
**Open / unverified.** The hotkeys on a real Windows console and a real POSIX TTY; resume after a genuine process kill mid-render (the tests truncate files, they do not kill a render).

---

## 5. Stage 14 — TensorRT Compile Receipts, Runtime VRAM Batch Rung, Queue Levels (`7524d5c`)

**Shipped.** `app/roop/trt_console.py` (new), `pipeline_status.py`, `predictor.py` (`warmup_session`), `vram_governor.py`,
`runtime_scheduler.py` (`_update_admission`), `swap_batcher.py` (`set_max_batch`), `pipeline_monitor.py`, `ProcessMgr.py`
(`_pipeline_queue_state`, `_runtime_worker_enter/exit`), `terminal_dashboard.py`; UI `CompileBanner.jsx`, `pipelineStatus.js`,
`faceswap/{ConsoleMirror.jsx, dashboardModel.js}`.

- **Compile receipts.** Events carry `stage: TRT_COMPILE`, `workspace_mb`, `cache_path` and, on finish, `cache_hit` (a plan file appeared or
  changed = built; untouched = deserialised; unreadable = unknown, never reported as a hit). Terminal: `TensorRT Engine Warmup` spinner on an
  ANSI TTY, else one `[TRT]` line; both end `[TRT CACHE HIT: 0.18s]` / `[TRT CACHE MISS: built in 143.2s]`. React banner shows workspace, cache dir, precision, an 8 s receipt.
- **Runtime VRAM rung.** ≥90 % used always counts as pressure and flushes the cache; with in-flight frames already at 1 the live `SwapBatcher`
  ceiling steps 8→4→2→1 and is restored step by step on recovery; every action prints a `[VRAM-GOVERNOR]` line.
- **Queue levels.** Decode / GPU / Encode fill and a verdict (`DISK/DECODE-BOUND`, `GPU-BOUND`, `ENCODE-BOUND`, `BALANCED`) in the `[Pipeline]`
  line, the dashboard (`telemetry.queues`) and the Processing page — one snapshot feeds terminal and React.

**Invariants.**
1. **The TensorRT workspace cap is not changed here and must not be changed casually**: it is hashed into the engine-cache namespace
   (`builder_config['workspace_bytes']`), so a new value orphans every compiled engine (a 27-minute rebuild was measured once). It is *reported*, not altered.
2. **"GPU Queue" is workers inside a frame ÷ worker count, not a queue.** On the parallel-stabilisation path there is no GPU cell and `GPU-BOUND` is *by elimination*, and is labelled so.
3. **An unreadable cache is "unknown", never a hit.** A failed build produces a failed receipt.
4. **The spinner is off while the live dashboard block is up** (it would shred a `\r` line).
5. The `[Telemetry] … queues …` log vocabulary is pinned; a new prefix broke it once.

**Tests.** `app/tests/test_stage14_trt_vram_queues.py` — 38 (changelog); `react-ui/.render-check/render-stage14.mjs` (fixture generated by the backend);
e2e `compile-banner.spec.js` (+2), `console-mirror.spec.js` (+3); both rung mutations (second rung off, ceiling off) fail the tests.
**Open (changelog).** No real cold TensorRT build through a live TTY; the 90 % rung proven through the scheduler with injected free-VRAM samples, not under real pressure in a render;
no 3060 / 4070 run; tile step-down and enhancer-specific cache flush **not built**.

---

## 6. Stage 15 — Contract Locks and Regression Suites (`df332de`)

**Shipped.**
- `react-ui/src/types/api-contract.ts` (544 lines): the canonical TypeScript contract. `react-ui/mock-server/pipelineTypes.ts` now *imports and re-exports* from it
  instead of declaring its own copies (the single source for `StageTelemetry`, `PipelineTelemetry`, `TrtCacheSnapshot`, `AutotuneSnapshot`, `RecognitionCurrent`, `BatchQueueResponse`, …);
  `react-ui/src/store/recipeStore.ts` is slimmed onto it. `tsconfig.json` now includes `react-ui/src/types/**/*.ts`, so `npm run typecheck` checks the contract.
- `app/tests/test_api_schema_parity.py` (fast, light profile): `get_state` / `_target_context_payload` return the keys `AppStateResponse` needs; `get_progress` keeps its operational keys;
  `settings.schema.json` ↔ `Settings` for the core settings; the dashboard, TRT-cache, autotune, recognition and identity-blend routes export their documented keys; `api-contract.ts` still declares the 14 required interfaces.
- `app/tests/test_stage15_features_regression.py`: four classes covering Stage 9 (non-TTY logger safety, no ANSI in log mode, hotkeys inert), Stage 10 (NaN / Inf / black-frame catcher, zero-swap counter reset, mask-collapse and scene-cut immunity, autopsy bundle),
  Stage 10/12 (segment-destination collision guard, custody audit of an intact file) and Stage 13 (atomic journal, truncated-tail recovery).
- **Cross-platform flakiness removed**: file handles opened as `open(p,'wb').write(...)` became `with open(...)` (an unclosed handle blocks `TemporaryDirectory` cleanup on Windows),
  and a test temp directory uses `ignore_cleanup_errors=True` (`test_aux_canary`, `test_finalize_output`, `test_stage14_trt_vram_queues`, `test_stage_cache`, `test_stage_telemetry_and_busy`).
  One worker call is typed `(self.postMessage as any)` so the new `include` typechecks.

**Invariants.**
1. **A route or payload key the UI consumes may not change without the contract file and the mock moving with it.**
2. The parity tests read source as text / AST — they stay fast and need no ML stack. They are *structural*; they do not prove a live route returns those keys (the Stage 16 parity script adds the cross-side comparison, §9).
3. Test temp files are closed before the directory is removed, on every OS CI runs.

**Open.** The parity tests assert key presence by string match; they would not notice a key whose *type* changed. The contract lock does not cover routes served only by the mock (the Stage 16 check lists them).

---

## 7. Cross-stage invariants worth keeping in view

| # | Invariant | Where it lives |
|---|---|---|
| 1 | New controls default to a bit-exact no-op; a feature that defaults off is proven by a render with it on and one with it off | Stages 11, 12 |
| 2 | Monitoring and recovery code never fails or slows the render it watches | Stages 10, 13, 14 |
| 3 | A status line says what it measures: `GPU-BOUND` by elimination, `cache_hit` unknown ≠ hit, fps = retired frames | Stages 9, 14 |
| 4 | A recovery path verifies before it reuses (hash, `ffprobe`, recipe), and discards the minimum | Stages 10, 13 |
| 5 | Interactive features are inert off a TTY | Stages 9, 13, 15 |
| 6 | The engine-cache namespace inputs (workspace, precision) are not tuned to fix a symptom | Stage 14 |
| 7 | UI ↔ backend ↔ mock share one contract, enforced by tests | Stage 15, 16 |

## 8. Verification made for this report

On 2026-10-28, one run of the six suites that pin Stages 10-15 (`test_watchdog_and_autopsy`, `test_interactive_console_and_journal`,
`test_stage14_trt_vram_queues`, `test_lighting_specular_directional`, `test_api_schema_parity`, `test_stage15_features_regression`):

| Profile | Result |
|---|---|
| Full (`app/env`, no `ROOP_TEST_LIGHT`) | **91 passed, 0 failed, 0 skipped** (24 s) |
| CI light (`ROOP_TEST_LIGHT=1`) | 83 passed, 4 skipped (the relighting module and three others that import `cv2`) |

This confirms the *named tests still pass*; it is not a render, not a GPU run, and not the full suite. The light profile does **not** exercise
Stage 12's relighting at all, so CI alone would not catch a regression there. The full suite already carries pre-existing failures at HEAD
(the Stage 14 commit records 45 failed / 30 errors on clean HEAD against 46 / 30 with that change); none was introduced or fixed here.

## 9. Open items carried forward (owner decisions in bold)

1. **Add the missing Stage 12 CHANGELOG entry** once a real 10-bit clip and a real-face relighting A/B exist.
2. Real-hardware runs absent for Stages 11–14: RTX 3060 for all; real cold TRT build through a TTY; real VRAM pressure; a genuine mid-render kill for resume.
3. Mock-only surface that has no backend route and no UI caller (`/api/segments`, `/api/keyframes`, the mock's frame keys `vram_used_gb`, `gpu_util`, …):
   recorded with reasons in `KNOWN_DRIFT` in `scripts/verification/check_api_mock_parity.py`. **Delete from the mock, or implement the backend.**
4. `GET /api/system/profile` has different shapes in the backend (per-stage timing) and the mock (`{active_profile, telemetry}`); a path-level check cannot see this.
5. `fps_now` was missing from the mock's `/ws/telemetry` frame although `pillModel.js` reads it; fixed in Stage 16 (`react-ui/mock-server/server.ts`).
