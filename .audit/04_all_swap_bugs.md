# Master bug list — remaining face-swap paths (sweep 4)

**Scope.** A targeted code sweep of the swap path outside what `.audit/01`–`03` already cover: edge cases, tensor/IO handling, video I/O, and UI/config propagation. No source file was changed. Bug #1 (disorientation) and Bug #2 (halos) are cross-referenced, not repeated.

**This is not exhaustive.** I read the files named below and ran small checks against the repo's own code. §5 lists what I did not read. Nothing here comes from a real clip or a full test-suite run.

**Evidence tags** (per `AGENTS.md`: a claim needs a named site; otherwise say "unverified"):
- **[CODE]**: read at the named `file:line`, not executed.
- **[RAN]**: I executed the repo's own code on a synthetic input and the result is quoted.
- **[UNVERIFIED]**: suspicious, but a dependency of the claim was not checked.

**Headline.** I found **no confirmed P0** (crash or data loss on a normal path). I am not promoting anything to P0 to fill the slot. The three P1s are *silent* wrong-behaviour bugs: a control that does nothing, or a mode that does the wrong thing without any error. That is the failure class `AGENTS.md` says this project keeps hitting.

### Fix status (uncommitted working tree)

| ID | Status | Change | Test |
|---|---|---|---|
| B-03 | **Fixed** | `normalize_swap_mode` accepts UI labels (case-insensitive) and canonical values, treats absent as `"all"`, and raises `UnknownSwapMode` for anything else; the API maps it to HTTP 400 via one app-level handler (`api.py`, next to the `UploadRejected` handler). `"All faces"` is now an explicit label. | `app/tests/test_swap_mode_normalisation.py` (11) |
| B-01 / B-02 | **Fixed** | New `roop/mask_offsets.py`: `resolve(face)` = explicit per-face override → live `CFG` → historic fallback. Ingest no longer stamps faces (`source_gallery.py`, `api.py`), the slider handler writes `CFG` only (floats, non-numbers ignored), `ProcessMgr` reads through `resolve` at both sites (manual-mask parse, paste), and the frozen Gradio slider handler writes `CFG` too. The benchmark harnesses that pin an override still win. | `app/tests/test_mask_offsets_resolution.py` (13) |
| B-07 | **Fixed** | `swap_faces` drops pending faces whose source faceset is empty and sets `num_faces_found = len(pending)` once matching and overlap suppression are done; `first` mode refuses an empty source up front. New `_source_has_faces`. Audit bucket: `refused: source faceset has no faces`. | `app/tests/test_swap_count_is_what_was_painted.py` (5; 4 of them fail on HEAD's `ProcessMgr.py`) |
| B-04 | **Fixed (reporting)** | `/api/settings` returns `restart_required: ["provider"]` and logs when the saved provider differs from the one the loaded sessions run on. It does not rebuild sessions live. | `app/tests/test_provider_restart_required.py` (7) |
| B-05 | **Open: needs your decision** | What should "Skip frame" mean in a video render: drop the frame (shortens the video, desyncs audio) or write the original (today's behaviour, mislabelled)? Not changed. | none |
| B-06, B-08 … B-13 | Open | Not touched. | none |

Behaviour changes to know about:
- **The feather now follows `config.yaml` / the sliders.** Faces that previously fell through to the hard-coded feather of 20 (or a stale ingest snapshot) now use the live `face_mask_blend`. With the live config that is 25. Any render or benchmark that was implicitly running at 20 will move. Counterbalance A/B accordingly (`AGENTS.md`).
- **B-07 changes the number `swap_faces` returns** in the cases above, so the `SKIP_FRAME_IF_DISSIMILAR` comparison and the swap audit now see painted faces. The audit's earlier `swapped (…)` intent counters are still bumped at match time, so they can exceed the painted count by exactly the new `refused: source faceset has no faces` bucket.

Light suite after the change: 2752 passed, 7 failed. **The same 7 fail on HEAD** (checked by stashing the five modified source files): 3 in `test_lipsync_acoustics.py`, 1 in `test_phase_harness_contracts.py` (`test_nvenc_writer_lut_kwargs.py` lacks the `load_tests` hook), 3 in `test_update_manifest_head.py` (`react-ui/package.json` hash).

| Priority | Count | IDs |
|---|---|---|
| P0 Critical | 0 | none confirmed |
| P1 High | 3 | B-01, B-02, B-03 |
| P2 Medium | 6 | B-04 … B-09 |
| P3 Low / hardening | 4 | B-10 … B-13 |

---

## 1. P1 — High

### B-01 — A painted manual mask is silently discarded for facesets built outside the main ingest path [CODE]

- **Where.** `roop/ProcessMgr.py:1462` reads `self.input_face_datas[0].faces[0].mask_offsets[4]` inside a `try`; the handler at `:1515-1516` prints `Failed to parse mask JSON: …` and leaves `self.face_masks = {}`.
- **Why it fires.** `insightface.app.common.Face.__getattr__` returns `None` for any missing attribute (`env/lib/site-packages/insightface/app/common.py:30-31`). Only the two ingest functions in `source_gallery.py:215,247` and `api.py:2139` stamp `face.mask_offsets`. Facesets built here do not:
  - `routes_facemgr.py:376-377` (`fs.faces = list(fm_faces)`)
  - `routes_identity_lab.py:365-366` (`fs.faces = [face]`)
  - `routes_projects.py:93-98` (restoring a saved project)
  - `None[4]` → `TypeError` → caught → mask JSON dropped.
- **Impact.** The user's painted exclude/include regions have no effect, with a log line that blames the JSON. The same unstamped faces also render with the hard-coded fallback `[0,0,0,0,20.0,10.0]` at `ProcessMgr.py:7225-7227`, i.e. feather 20 regardless of `face_mask_blend`.
- **Not checked.** Whether the React UI routes mask painting through one of those three sources in practice.
- **Fix direction.** Stamp `mask_offsets` in one place (a `FaceSet` accessor that falls back to `_mask_offsets_from_cfg()` at read time) instead of at ingest, and read it through that accessor at `:1462` and `:7225`.

### B-02 — Mask/feather sliders update one face, not the one the swap reads [CODE]

- **Where.** `api.py:1105-1155` (`_update_mask_offsets_from_payload`) writes `top/bottom/left/right`, `face_mask_blend`, `mouth_mask_blend` and the four mouth scales to **`faceset.faces[0]` of the currently selected faceset only**.
- **What the swap reads.** `inputface = fs.faces[canonical_idx]` (`ProcessMgr.py:5918-5919`), and the paste takes `inputface.mask_offsets` (`:7225`). `FaceSet.canonical_face_index` (`roop/FaceSet.py:305-327`) returns the most frontal face for V2 sets (`min(yaw²+pitch²)`), which need not be index 0. Every other faceset is never updated; the multi-source modes (`all_input`, `selected_multi`) use them.
- **Compounding.** `source_gallery.py:36-48` stamps a snapshot of the config at ingest time, and returns a hard-coded `12.0` blend when `CFG` is not ready yet (`:42-44`, "applied on the next update"). Settings changed later reach `CFG` and `roop.globals` (`api.py:1537-1539`) but not faces already ingested. So `face_mask_blend` in `config.yaml` (live: 25) applies only to faces ingested after it was set, and only to the faces the payload touches.
- **Impact.** Mask blur/feather/offset sliders appear to work in the UI and do nothing for V2 sets, non-selected sources, or sources loaded before the change. Directly affects the Bug #2 feather finding (F2): the number the halo depends on is not the number the user sees.
- **Fix direction.** Make these offsets a render-time read from `CFG` (with a per-faceset override only when the user sets one), not a per-face attribute snapshot.

### B-03 — Unrecognised swap-mode strings silently become "swap every face" [CODE]

- **Where.** `roop/processing_request.py:28-29`: `normalize_swap_mode(label) = SWAP_MODE_LABELS.get(label, "all")`. The dict (`:15-25`) is keyed by UI *labels* (`"Selected face"`, `"First found"`, …). The canonical values (`"first"`, `"selected"`) are not keys. Callers: `processing_request.py:252` (default `"All faces"`, also not a key), `api.py:1158-1159`.
- **Impact.** A typo, a renamed label, or an API client that sends the canonical value (`"first"`) gets `"all"`: every face in the frame, bystanders included. No error, no log.
- **Fix direction.** Accept canonical values as identity, and reject (HTTP 400) or at least warn on anything else. `_validate_settings_payload` (`api.py:1494+`) is the model.

---

## 2. P2 — Medium

### B-04 — Provider change in the UI reports success but only takes effect after restart [CODE]
- `api.py:1540-1550` updates `CFG.provider` and the `provider_requested/admitted` bookkeeping, then returns `{"status": "success"}`. Nothing in `api.py` assigns `roop.globals.execution_providers`; that is set once at startup (`roop/startup_state_machine.py:810`).
- Already-built processors are reused by name only (`ProcessMgr.py:1037-1041`), and e.g. `Mask_XSeg.Initialize` rebuilds only when `devicename` changes (`processors/Mask_XSeg.py:34-39`). `get_device()` returns `"cuda"` for both TensorRT and CUDA (`utilities.py:1115-1118`), so a TRT↔CUDA switch does not rebuild.
- **Impact.** The user believes they switched provider; the render uses the old sessions. No "restart required" message.

### B-05 — "Skip frame" no-face policies do not skip, and one compares the wrong counts [CODE]
- Video render: a `None` result is replaced by the original frame (`ProcessMgr.py:1965-1971`, one log line per frame), so `SKIP_FRAME` / `SKIP_FRAME_IF_DISSIMILAR` behave as "use original". Plausibly deliberate (keeps A/V in step), but the UI label says skip.
- `SKIP_FRAME_IF_DISSIMILAR` (`ProcessMgr.py:3502-3504`) returns `None` when `len(self.input_face_datas) > num_swapped`. That compares *sources loaded* to *faces swapped*, not "faces that did not match". With more sources loaded than people in frame, every frame — including fully swapped ones — is dropped/replaced. **[UNVERIFIED]**: whether `input_face_datas` holds only the mapped sources or the whole gallery (the preview path builds `mapped`, `api.py:268-274`).

### B-06 — `all_input` pairs people with sources in different orders on different paths [CODE]
- Prepass faces and `get_all_faces` are x-sorted (`procmgr_tracking.py:3091-3095`, `face_util.py:2086-2101`). On the dispatch-tracker path the list is re-sorted by `_track_id` (`ProcessMgr.py:4058-4060`), i.e. order of first appearance.
- "nth face → nth source" (`:4106-4111`) therefore maps the same person to different sources depending on path (prepass vs fallback detection vs preview).

### B-07 — A face that is not swapped is still counted as swapped [CODE]
- `process_face` returns `frame` unchanged when `inputface is None` (`ProcessMgr.py:5982-5983`), after `self.total_swaps += 1` (`:5631`). Callers count the face first (`num_faces_found += 1`, `:4097`, `:4107`) and `_audit_hit('swapped …')` (`:4099`, `:4110`).
- Trigger: a selected faceset with an empty `.faces` list. The frame is reported as swapped, so the no-face policy and the swap audit read success. This is the "reports success while not running" class from `AGENTS.md`.

### B-08 — Legacy temp-frame extraction forces constant frame rate [CODE]
- `util_ffmpeg.py:339-346` extracts with `-vsync cfr -fps_mode cfr` (and `cfr_video_filter`), duplicating or dropping frames on VFR sources. The modern path avoids this (`vfr_timing` retime in `finalize_output.py:286-287`, source colour plan in `procmgr_batch.py:695-703`), so this only matters when the temp-folder path is taken (`01_pipeline_map.md` §1.3: "fallback when random seeking is required"). **[UNVERIFIED]** how often that path is taken today.

### B-09 — Retry-rotated re-enters `swap_faces` without a frame index [CODE]
- `ProcessMgr.py:3629` and `:3636` call `swap_faces(copyframe, temp_frame)` with no `frame_idx`, so `self._tls.frame_idx = None` (`:3890`).
- `_roto_frame_ui(None)` then falls back to `roto_preview_frame` (`:3575`), a leftover from the last preview, and the roto inclusion zones drawn on the upright frame are evaluated against a rotated frame (the guard at `:6524-6525` only checks `rotation_action`, not this retry). Reached only for frames the prepass did not scan (`:3549-3563`). Low frequency; flagged because the result is wrong zones, not an error.

---

## 3. P3 — Low / hardening

| ID | Where | What | Evidence |
|---|---|---|---|
| B-10 | `procmgr_masking.py:2025`, `:2061` → `procmgr_color.py:524-526` | Eye and mouth restore run the full LCT on their small ROI. LCT samples every 16th pixel, so a clipped ROI gives it 1-15 samples and a near-singular covariance. Measured with the real `_color_transfer_lct` (200 trials per size, plate patch → swapped patch): worst pixel distance from the target mean is **63 px at 100×100** (the noise floor for this input), **86 at 20×30**, **140 at 12×20**, **193 at 8×8**. Normal ROIs are ≥ ~58 px per side (`pad ≥ 29`), so this only bites when the face sits at the frame edge. | [RAN] |
| B-11 | `ffmpeg_writer.py:227-229`, `:328-329` | Odd frame sizes are squashed by one pixel with `scale=` rather than cropped or padded. Sub-pixel aspect error. | [CODE] |
| B-12 | `ffmpeg_writer.py:234` | `-hwaccel auto` precedes a `rawvideo` stdin input, where it does nothing. | [CODE] |
| B-13 | `ffmpeg_writer.py:414-415` | `_spawn` `print`s the full ffmpeg argv, including file paths, on every spawn (and on every NVENC→software fallback). | [CODE] |

---

## 4. Checked and cleared (kept so they are not re-investigated)

| Suspicion | Result |
|---|---|
| `uint8` overflow when the swap output is cast (`ProcessMgr.py:6421`) | Cleared. `normalize_swap_frame` rounds and clips to [0,255] before the cast (`procmgr_tiling.py:90-99`). |
| BGR/RGB swap in the swap blob | Cleared. `to_blob` flips on the way in (`:79-80`, `:68-73`), `normalize_swap_frame` flips back (`:97`). |
| `[0,1]` vs `[-1,1]` normalisation | Cleared for the single-net and multi-step paths. Mean/std come from the model spec (`FaceSwapInsightFace.py:1004-1006`), `[-1,1]` outputs are denormalised (`procmgr_tiling.py:93-95`), the float multi-step re-entry is handled (`:70-77`), and RealSwap's second net converts range explicitly (`FaceSwapInsightFace.py:2216-2217`, `:2456-2457`). |
| IOBinding output buffer reuse overwriting a result | Cleared for the swap path. `CudaOrtIOBinding.run` copies to host (`utilities.py:151-159`) and bindings are per session, keyed on `id(session)`, initialised at `FaceSwapInsightFace.py:766` and cleared on TRT rebuild (`:1293-1294`, `:2517-2518`). `run_gpu` hands back **shared** device buffers (`utilities.py:109-143`); its callers are in `optimized_processor.py:1700,2291,2447`. **[UNVERIFIED]** whether any live path calls them; the zero-copy pools the enhancers import are a different object. |
| Non-finite / damaged detector output | Cleared. `sanitize_face_attributes` drops faces with bad bbox or `kps` and nulls bad 106/68/pose/embeddings (`face_sanity.py:109-141`). |
| Face partly outside the frame | Cleared for the paste (ROI clamps and zeroed border rows, `procmgr_masking.py:608-615`, `:785-786`), the mouth restore (`box_width <= 0` guard, `:2054`; the negative-index slice in `create_mouth_mask` at `:1806-1810` is harmless because of it), the eye restore (`:1966-1973`, shape check `:2020`), the unwarped mask box (`_mask_crop_box` returns `None` off-frame, `:1273`) and ROI re-detect (`ProcessMgr.py:477-490`). |
| Empty detection list | Cleared. Early returns at `ProcessMgr.py:3955-3959`, `:3974`, `:4061-4069`; `get_faces_detector_only` returns `[]` on any failure (`face_util.py:2021-2032`). |
| `aligned_img` (pooled crop buffer) aliasing in the smart-skip cache | Cleared. `set_track_patch` copies both arrays (`temporal_smart_skip.py:307-309`). |
| `color_match_after_enhance` re-colouring untouched pixels | Cleared as a visible effect: 0.5-0.6 LAB levels when the swap background matches the plate (`03_halo_audit.md` E3). |
| BT.601/709 shift and VFR handling in the main writer | Not reproduced here. The writer uses the source's own `ColorPlan` and tags (`ffmpeg_writer.py:180-182`, `:352-376`; `procmgr_batch.py:695-696`) and VFR retiming (`finalize_output.py:286-287`). The old "bt709/tv forced" issue appears addressed in this tree; I did not render a clip to prove it. |

---

## 5. Not swept

Reading `ProcessMgr.py` (7,653 lines) and `procmgr_*` fully was already a large share of the budget. I did **not** read or test:

- Frame readers and seeking: `capturer.py`, `nvdec_reader.py`, `video_stream.py`, `video_io_hw.py`, `segment_writer.py`, `StreamWriter.py` beyond the greps above.
- Audio: `lipsync_audio.py`, `live_audio.py`, the audio cut/mux code in `util_ffmpeg.py` past the greps above.
- Temp-file lifecycle on cancel or exception (no leak was looked for beyond the cleared aliasing above).
- Tiny-face policy: the detector's minimum size and `EnhanceGate`. The live config has `enhance_min_face_px: 0`, which switches the gate **off**, so tiny faces reach the restorer; `should_bypass_restorer` (`smart_restorer_bypass.py:142+`, ≤ 320 px) then replaces the restorer with a DoG sharpen for hyperswap.
- Source-to-target identity matching (`face_reference.py`, `identity_lock.py`, `tracker.py`) beyond `swap_faces`'s dispatch; `02_disorientation_audit.md` §3 covers part of it.
- `routes_queue.py`, `project_render.py` and the headless `--project --render` option mapping; I traced the settings POST and the preview path only.
- The frozen Gradio UI (`ui/`), the React UI, `face_overlap`, TensorRT engine building, live camera.
- Any test run. The prior audit notes the full suite stalls in `face_engine/tests/test_cuda_streams.py`.

## 6. Related, already documented

- **Bug #1** (`02_disorientation_audit.md`): unchecked five-point fit, orientation threshold mismatch, interpolation across bad endpoints, `first` = leftmost.
- **Bug #2** (`03_halo_audit.md`): F1 yaw-blind crop stabilisers (ghost contours), F2 feather 2× the calibrated value, F3 non-original "original" pixels, F4 matte leak past the crop footprint, F5 whole-crop colour statistics, F6 segmenter/hull reliability. **B-02 above bears directly on F2**: the feather value in use is a per-face snapshot, not the live setting.

## 7. Suggested order

1. **B-03** (swap-mode fallback): one-line behaviour change, high privacy value, easy to test.
2. **B-02 and B-01** together (mask offsets read from one place at render time). Also settles which feather value Bug #2's A/B measures.
3. **B-07** (count only faces that were painted), then **B-05** (decide what "skip" means and say it in the UI).
4. **B-04** (either rebuild on provider change or return `restart_required: true`).
5. B-06, B-08, B-09 after the owner confirms how the paths are used; P3 items opportunistically.

Each fix needs a test that drives the named failure rather than stubbing it (`AGENTS.md`, "Never mark a stage …"): e.g. B-01 builds a `FaceSet` through `routes_projects` and asserts the mask JSON is applied; B-03 asserts every canonical mode string round-trips.

B-10 measurement script: `.audit/04_experiments/b10_small_roi_lct.py` (run with `app/env/Scripts/python.exe`, `NO_ALBUMENTATIONS_UPDATE=1`).
