# Changelog and incident notes

Dated notes on behaviour changes and the incidents behind them, in reverse order. The
full session record is [`SESSION_LOGS.md`](SESSION_LOGS.md); the running engineering
state lives outside the repository (`RECODE_STATUS.md` in the operator's `roop-keep`
folder). Entries before 2026-09-21 were moved here from the README on 2026-09-22.

## 2026-10-11

- **React UI API client: 15 s deadline by default (opt out with `timeout: 0`), errors that name the request, swallowed failures reported once, requests cancelled on unmount.**
  `api.js` had no deadline unless a call asked for one, so most of the 159 call sites could wait forever on a stalled backend and say nothing. Now every `getJSON` / `postJSON` / `postFile(s)` has 15 s unless it passes `timeout: 0`; uploads too (`xhr.timeout`). Which endpoints may outrun that is decided once, with a reason each, in `src/longRunning.js` (34: a swap start, previews, clip scans, uploads, ffmpeg joins, bulk deletes; judged by what the handler reaches, since a cold TensorRT build makes even a "preview" take minutes). **`scripts/api-callsites.mjs`** (`npm run lint:api-timeouts`, in `npm run check`, and `test_ui_api_timeouts.py`) enumerates every call site and fails on: a listed endpoint without `timeout: 0` (it measured **45 such sites** before they were fixed: 38 explicit, 7 computed-path wrappers now using `timeoutFor(method, path)`), an opt-out for an unlisted endpoint, a computed path that says nothing, a stale entry. A fixture tree proves each rule fires on the right line (and caught a bug in the scanner itself: a call named only inside a string literal was counted).
  **Errors:** every failure is an `ApiError` with `status`, `method`, `path`, `kind` (`http` / `network` / `timeout` / `parse`); the message ends `(HTTP 409 POST /api/swap)`. `useJobRecovery` decided "this backend is old" by regex-matching `404|not found` in the message; it checks `e.status !== 404` now. A caller's own abort stays an `AbortError`.
  **Swallowed failures:** 40 empty catches around backend calls (settings saves, profile/preset saves, panel data loads, status polls) now go through `src/failureLog.js`: each distinct failure once to the console, as a toast too unless it is a background poll, at most 2 toasts per 30 s, never an abort. A failed settings save used to vanish; it now says `Saving settings failed: disk full (HTTP 500 POST /api/settings)`. The ~55 empty catches around browser APIs (`localStorage`, `play()`, `ws.close()`, pointer capture) are left silent on purpose.
  **Unmount:** `useApi()` binds a component's calls to its lifetime; 27 components/hooks use it. GETs and the read-only POSTs (preview, preview upscale, runtime estimate, clip advisor, quality analysis, identity blend: `abortOnUnmount: true`) are aborted when the tab unmounts (checked in a browser: `net::ERR_ABORTED`, no toast). **Mutations and uploads are deliberately not aborted:** cancelling a POST client-side does not undo it, it only hides whether it happened. That change exposed a latent bug: Face Swap's preview treated *any* `AbortError` as "the model build took too long" (its own 15-minute killer), so a tab switch mid-preview would have toasted a false timeout and started the queued next preview; its deadline now marks itself.
  **Not covered:** raw `fetch()` (frame streams, workers, blob downloads: 7 sites) is outside `api.js` and has its own signals. Tests: `.render-check/render-api-client.mjs` (client, `ApiError`, failure log, unmount scope; fails when the default deadline is removed), `e2e/api-client.spec.js`, `test_ui_api_timeouts.py`.

- **React UI: 6,771 lines of unreachable code deleted, one ErrorBoundary, a README that matches the app, and a scan that keeps it so.**
  `src/components/{studio,preview,timeline,facebank,queue,telemetry}` (23 modules, added in one commit on 09-29, never imported by anything) are **deleted, not mounted**, with their four `.render-check` scripts (`test:preview`, `test:timeline`, `test:facebank`, `test:telemetry-queue`), whose passing was the only evidence the code worked: they imported the modules directly. Per folder: `preview` (WebGL renderer, 1,938 lines) duplicates `faceswap/InteractivePreview` + `player/`; `timeline` (2,020) duplicates `faceswap/Timeline`; `queue` (639) duplicates `faceswap/QueuePanel` + `useQueue`; `telemetry` (505) duplicates `LiveTelemetry` / `SystemTelemetryHud` on the same `/api/system/telemetry`; `facebank` (1,179, IndexedDB) duplicates `PersonGroups` + the backend face bank; `studio` (471) only composed the others and duplicates `useWorkspaceLayout`. Each is prop-driven with nothing producing its inputs (`clusters`, `identityTracks`, `curves`, `sceneCuts`, a binary-frame `wsUrl`), so mounting `#/studio` would have meant writing the adapters the V2 audit (2026-09-02) found nobody imported. Recoverable from `8c10e91`.
  **The scan:** `react-ui/scripts/unreachable-modules.mjs` (`npm run lint:unreachable`, in `npm run check`, and `test_ui_unreachable_modules.py` in the Python suite) walks static/dynamic imports, re-exports and `new URL(..., import.meta.url)` workers from `main.jsx`; before the deletion it reported exactly 23 modules / 6,771 lines, now 0 (121 of 121). No allowlist. A fixture-tree test proves it fires on an orphan and ignores imports that only appear in comments.
  **ErrorBoundary:** `main.jsx` had its own copy of the class; there is one now (`variant="app"` self-contained with inline styles, `variant="panel"` per tab), asserted by `.render-check/render-error-boundary.mjs` (a class attribute added to the app variant fails it). **Found on the way: Retry never recovered a failed tab chunk.** `React.lazy` memoizes its rejection, so after "Retry" no request was even made (measured); the old comment claiming otherwise was wrong. New `components/lazyPanel.js`: panels are `lazyPanel(load)` and the boundary calls `onReset` on Retry and when the failed tab unmounts, so returning to a healed tab loads it with no click (`e2e/error-boundary.spec.js`). The old `resetKey` path was already dead in this layout: the boundary sits inside `<motion.div key={tab}>`, so every tab gets a fresh instance.
  **`web_ui/` is kept and marked** (`web_ui/README.md`): it is the `face_engine` control UI (React 18 + TS), served by `face_engine/run.py` and the engine's server and covered by `face_engine/tests/test_web_ui_e2e.py`, not a leftover of `react-ui`. **`react-ui/README.md` rewritten:** it listed 4 of 9 tabs, ~12 of 131 endpoints (one, `/api/facemgr`, not a route), a dev server and port 8001 for a launcher that builds `dist/` and lets the backend serve it on a Pinokio-assigned port, and an out-of-date "not yet ported" list. The API section is generated from the code; `test_ui_api_surface.py` fails if the UI calls a route the backend does not register, or the README omits or invents one (all 128 HTTP + 3 WebSocket paths match today).

- **React UI, Face Swap: slider grid, 12px body text, 24px range hit area, one run control.**
  **Slider tracker.** `SliderTrackerBar` laid its cards out with a breakpoint ladder (4 columns at 1440px), which left ~81px for the label: 13 of 16 labels were `truncate`d ("Original / Enh..."). Now `grid-template-columns: repeat(auto-fit, minmax(220px, 1fr))` (2 columns at 1440 in the middle column, 3-4 wider), labels wrap to two lines then clamp, and the title carries the whole label plus its info text. 0 truncated labels at 1280/1440/1920.
  **Type scale.** `--text-mini` was 11px and is the caption size of ~150 call sites; it is 12px now (equal to `note`), not renamed. `nano` (9) and `micro` (10) are CHROME sizes (pill badge, `<kbd>`, 1-3 character tick, uppercase letter-spaced tag) and no longer used for labels, values, captions or button text: ~40 sites in Face Swap, PersonGroups, FileDrop, Timeline and the `Button` `xs` size moved to `text-note`. Body text under 12px on Face Swap at 1440: 108 nodes -> 0 (build stamp excepted, `data-small-ok`). `BiometricAngleHUD/*.tsx` carried 36 arbitrary `text-[8|9|10|11px]` classes that `test_ui_type_scale.py` never saw (it scanned `.js/.jsx` only); it scans `.tsx` now, requires every step under 12px to be exactly nano/micro, and forbids those sizes inside Field/Slider/Toggle/Button and the tracker card.
  **Range inputs** are a 24px-tall transparent hit box with the 6px rail drawn by the track pseudo-element (`--range-track`; the tracker paints its fill through it), thumb focus ring; a press 8px off the rail still moves the thumb. Global, so it applies to every slider in the app.
  **One run control.** The run bar's Start was disabled correctly; the floating dock carried a second Start that was never disabled, so with no media loaded it posted `/api/swap` the run bar had refused. The dock has no Start/Cancel now. The run bar's button turns into Cancel (with the header chip's confirm, because a double-click on Start would otherwise land on it) and the reason Start is unavailable is a sentence beside it ("Add a target video or image to start."), wired with `aria-describedby`; it was a truncated 14px aside before.
  **Dock.** Face Swap reserves `pb-28` under its content and a 104px `scroll-padding-bottom`, so the last row of a column can be scrolled clear and keyboard focus does not land underneath (WCAG 2.4.11). Regressions caught in screenshots and fixed: the Target Media toolbar squeezing its heading into four lines, and the Restoration Switcher's FPS chips wrapping once they were 12px. Face Swap tab stops 259 -> 258. `e2e/faceswap-layout.spec.js`: 7 tests (labels, 12px floor, hit area, single Start, no-media reason, dock clearance, scroll-padding).

- **React UI: the header nav no longer clips tabs. Five primary tabs, the rest under "More", icon-only below 1280px, one utility cluster (`react-ui/src/components/HeaderNav.jsx`).**
  Nine labelled tabs plus five loose tool buttons shared one `overflow-x-auto` row, so tabs were scrolled out of reach: `e2e/allowlist.json` recorded 6 clipped tabs at 1024px, 5 at 1280 and 3 at 1440 (0 at 1920); the allowlist is empty now, and Face Swap tab stops go 265 -> 259. Now Home / Face Swap / Batch Matrix / Outputs / Settings stay in the strip and
  Processing (while a run exists) / Face Manager / Editor / History live in a "More" disclosure (Enter/Space opens onto the list, arrows/Home/End move, Escape and click-away close). When the current tab is one of those, the trigger **becomes** that tab (its icon and name, plus a chevron),
  so the strip still says where you are. Below 1280px the tabs are icon-only (`aria-label` + a tooltip on hover and keyboard focus); Profile / HUD / Snapshots / Search / Zoom are one bordered cluster, icon-only below 1760px (the labelled cluster is ~310px wider; 1760 is where it still fits
  with the run chip showing and a More tab current - at 1600 it wrapped the header). The width test is `innerWidth / zoom`, not a CSS breakpoint, because the app zooms with CSS `zoom` on `<html>` and a media query cannot see it (125% on a 1440px window lays out like 1152px).
  The active tab is scrolled into view in the strip (a safety net: the strip only overflows on a phone-width window). `ALL_TABS`, `setTab`, the hash routes and `warmTab` (pointer-enter + focus on every strip tab and every More item) are untouched.
  **Known and not fixed:** with a run in flight the chip beside the brand adds ~186px, and at 1280-1366px (labelled tabs, icon-only cluster) that is wider than the row, so the header wraps to two rows - nothing is clipped, and 1024/1440/1920 stay on one row (tested).
  Keeping it on one row there means removing text from the chip (ETA) or the brand. HUD now uses the CPU glyph and Snapshots a camera: icon-only, they shared a gear with the Settings tab and a clock with History.
  A hover-only tooltip is dismissed by moving the pointer, pressing or scrolling, not by Escape: `test_ui_shortcut_keys.py` forbids a second window-level Escape handler (FaceSwap owns it), so Escape dismisses the tooltip of the control that has keyboard focus. The dropdown is opaque (`--card-bg` over `--bg-base`);
  `.glass-panel`'s translucent fill let the page chips show through the labels. `e2e/nav-visibility.spec.js` now covers the four widths (clipping, a wrapped header, label density), reachability of every More item, keyboard use, the hash and Back, zoom, tooltips, axe with the menu open, and the run chip.

- **React UI accessibility pass: zero axe violations on every tab and Batch strategy (was 65 nodes on 7 tabs, 16 more critical ones on Batch strategies 2-4); Face Swap tab stops 291 -> 265; text contrast measured and fixed.**
  `Toggle`'s decorative `aria-hidden` switch was a `motion.span` with `whileTap`, which framer gives `tabindex=0`: 26 invisible tab stops on Face Swap (and 10 on Settings), now CSS `group-active`.
  `Button` silently dropped every prop but six, so `<Button title=...>` icon buttons had no name and the Profiles modal's `type="submit"` Save never submitted; it forwards the rest now.
  **Space on any focused button on Face Swap did not activate it** (the global play/pause hotkey `preventDefault`ed it - Enter worked): Start Swapping could not be started with Space. Fixed; a keyboard-only
  run (Refresh Preview, then Start Swapping) is now a test. Face cards are a container with one select button and a SIBLING remove button (no nested interactive); names and labels on all four Batch strategies' controls, the Outputs table's
  checkboxes and the timeline markers, `h3` -> `h2` Section titles, `autoFocus` (6 sites) replaced by explicit focus-on-mount, and the confirm dialog now restores focus to its opener.
  **Contrast.** axe cannot judge text over this UI's gradient/glass (~1,040 nodes sat in `incomplete`), so `react-ui/e2e/contrast.js` composites the ancestor backgrounds itself (worst gradient stop). Measured:
  ~360 text nodes under AA on the default theme, from ~410 hard-coded `text-white/30..45`, white-on-crimson (3.83:1), accent text on tinted panels and a 70%-opacity toolbar. Now one muted token (`text-muted`),
  an accent-ink token (`text-accent`) and the computed accent-fill ink in every mode (it was light-themes-only): **default theme 298 -> 0** (3 tabs); across all 37 presets x 3 tabs **15,812 -> 4,430, no theme
  worse**. Not done: the light themes still have 311-760 (translucent-black scrims turn grey on a light page), Gruvbox/Catppuccin ~200-250, a few dark themes tens; they are per-theme ceilings in
  `allowlist.json` (`npm run test:e2e:themes`). **Regression caught on the way:** the first `--bg-base` change made the five AMOLED themes paint a WHITE page (their plain-colour `--bg-gradient` is only legal as
  the last background layer); only the theme sweep could see it, so it is kept as an opt-in spec. Lint now has `jsx-a11y` and `react/exhaustive-deps` as errors for new code (55 existing diagnostics in 19
  files listed per rule as legacy debt).

## 2026-10-10

- **React UI: a failed frame request is no longer retried forever (`react-ui/src/components/faceswap/retryPolicy.js`).** With a backend answering 500, `useThrottledFrameRequest` re-armed itself from
  `finally` every 150 ms (the wanted URL is still not the frame on screen after a FAILURE): **196 `/api/target/preview` requests in 30 s from an idle Face Swap tab** (found by the new
  `react-ui/e2e/idle-requests.spec.js`), and 7.5% of a core against 1.7% when frames load (`idle-cpu.spec.js`; 1.8% after). Failures are now counted per URL: exponential backoff with jitter
  (0.5 s, 1, 2, 4 s), give up on the 5th consecutive one, reset on a URL change or a manual retry; an AbortError is never counted. The hook returns `{error, retry}` and the stage shows
  "Frame unavailable - Retry" (over a stale swap or the empty placeholder) instead of silently keeping the previous picture. Audit of the other loops: `usePlaybackBuffer` had the same shape on its
  HTTP chunk path (a failed chunk was re-asked on the next animation frame, up to 60/s while playing) and now backs off the same way and STOPS playback with a toast after 5; `useJobRecovery`
  (fixed 15 s heartbeat, holds on failure), `frameSocket` / `useTelemetrySocket` / App's boot and offline retries (already exponential or fixed-slow), the grid-preview loader (one pass per
  dependency change) and `PreviewCanvas` (one decode per URL) are not loops and are unchanged. Not changed: `PreviewCanvas` still fetches the stage URL itself in parallel with the hook (one duplicate
  request per frame, success or failure).
- **RestoreFormer++ FP16 vs FP32 islands (`docs/perf/restorer_fp16_islands_2026-10-10.md`): target not reachable; shipped path unchanged; restorer canary added.** On 32 real crops (ORT CPU FP32 reference) the
  FP16 restorer loses 0.0033 SSIM, spread over the encoder and the VQ codebook lookup - not softmax or norms (FP32 on all of them: 0.99658 -> 0.99669). The only engine to clear 0.998 keeps the
  encoder + quantiser in FP32 (0.99816) at 31.6 ms vs 21.8 ms (69% speed); on the final composited face it recovers 0.0013 of a 0.010 SSIM loss (46.1 -> 46.5 dB, identity delta <= 0.0005). The
  shipped engine's 0.9975 is a lucky build: twelve fresh FP16 builds score 0.9964-0.9969. `Enhance_RestoreFormerPPlus` (and Restore Ultra) now run the aux canary against a CPU FP32 reference (SSIM floor 0.96, cached by
  engine hash, FP32 fallback); GPEN is deliberately not covered (its shipped TRT mixed engines sit at SSIM 0.84 from FP32 by design). New opt-in `ROOP_RESTORER_NATIVE_ENGINE` runs a native island engine.

- **Per-model TensorRT precision; XSeg, 2d106det and 1k3d68 now build FP32 (`docs/perf/trt_per_model_precision_2026-10-10.md`). Output changes: the mask and the landmark-refined kps move
  toward the FP32 reference.** On 500 real faces (d4/d1/d6/Love/s7) the shipped `mixed` engines miss the gates earlier briefs set: XSeg IoU mean 0.9936, min 0.81, 6 faces < 0.95;
  2d106det up to 1.2 px; 1k3d68 mean 1.2 px with the refined kps at 0.94 px mean, 3.6 px p95 (10.5% of the inter-ocular distance). The FP32 engines land at the engine-to-engine floor
  for +0.6 / +0.02 / +1.05 ms per call; w600k_r50 passes (cosine min 0.999916) and stays mixed. `precision_policy.PRECISION_OVERRIDES` / `ROOP_TRT_MODEL_PRECISION` map a model FILE stem to
  a precision; `providers_for` honours it (an explicit `requested=` still wins), and the buffalo_l bundle - which insightface builds from one provider chain - gets it through
  `face_util._per_model_precision` (precision step only; no shape profile, so no engine moves namespace). Engines are per precision on disk, so reverting is one table line and loses nothing.
  End-to-end fps was not A/B'd (expected ~1%, below what this rig resolves); the regression benchmark passes (300/300 swapped, face SSIM 0.985).
- **Startup canary for XSeg / w600k_r50 / 2d106det / 1k3d68, cached by engine hash (`roop/aux_canary.py`, `ROOP_AUX_CANARY=0` disables).** The swap canary's idea for the models whose output is read as
  ground truth: two fixed inputs through the live engine and a CUDA FP32 reference, metric by consumer, floors calibrated against 500 real faces and a wrong-face control
  (`docs/perf/aux_canary_calibration.md`). The verdict is keyed on the blake2b of the cached engine files, the ONNX digest, the build options and the floors, so a rebuilt engine, a driver /
  TensorRT / ORT upgrade or an edited floor is a miss and nothing is trusted across them; warm start 21 s vs 52 s cold, no reference session built. A failing engine is rebuilt on
  TensorRT FP32, then CUDA/CPU. It catches the swap-canary failure class (wrong picture, collapse, non-finite), not FP16 drift - that stays the fidelity harness's job.
- **XSeg's NHWC input costs one `Shuffle` layer: 0.0126 ms mixed / 0.0137 ms FP32 of a 2.5-3.2 ms call.** An NCHW variant was not built (ceiling 0.4-0.8% of one call). The live
  provider / precision of all four models is recorded by `tests/trt_precision_probe.py`; `_model_digest` is now memoised (it re-read the whole ONNX on every `providers_for` call).

## 2026-10-09

- **Opt-in static TensorRT profile for `retinaface_r50` (`docs/perf/r50_static_profile_2026-10-09.md`); defaults unchanged.** `ROOP_TRT_STATIC_PROFILE=1` pins the ORT
  detector to `1x3x640x640` (own cache namespace `_spstatic...`) instead of the 320-1280 / batch 1-8 band: each pooled instance's first inference costs +56 MiB
  instead of +1.17 GiB (-2.2 GiB at pool 2), latency neutral. Output differs from the band by FP16 tactic noise (the same distance from an FP32 render as the band
  itself), so the literal IoU 0.99 / 0.5 px gate is NOT met and it ships off. A pinned engine ignores `det_size` (warns once). Letterbox vs squash for
  `retinaface_r50_gpu` re-measured: squash stays (d1 recall 833 -> 764 matched, duplicates 19 -> 75 pairs); TensorRT/FP32 score ratio 1.000 in every geometry, so
  "letterbox suppresses scores under TensorRT" is not supported - the cause is scale.
- **`tools/quality_harness.py`: a quality harness with no simulated or constant metrics (`docs/perf/quality_harness_method.md`).** Reference = a full-FP32 render
  (`trt_precision=fp32`, `ROOP_SWAP_FP32=1`, verified from live `[Session]` records and a first-inference probe) of d1/d4/d6/Love/s7, 300 frames, lossless x264.
  Per candidate: identity cosine vs the source with AdaFace (independent of the pipeline's w600k), masked SSIM/PSNR on the composited face, keypoint drift, skin
  detail, identity jitter, yaw bins, XSeg mask IoU at the candidate's precision, detection recall (IoU >= 0.5), track counts and swap-audit counts. The run FAILS
  (exit 2) if two different swap networks (by content hash of the files actually loaded) share a model-dependent metric, if one network sits behind two names, or if
  the instrument cannot tell a swap from the untouched plate. Per model it logs the real provider, trt_fp16 and VRAM before/after the first inference
  (`tests/first_inference_probe.py`). `tools/benchmark_hyperswap_audit.py` is now a front end to it; the old file's quality values were typed in or random, so
  the Stage 5 report is marked INVALID. The other `tools/benchmark_*.py` declare their synthetic inputs (`tools/_synthetic_inputs.py`) in a banner and in their
  JSON. Full run (3 swappers x 5 clips, 69.5 min): hyperswap_1a at mixed precision is within noise of its FP32 reference (identity |d| <= 0.006, SSIM 0.93-0.96);
  inswapper_128 differs model-sized (SSIM 0.86-0.93, identity -0.04..+0.06, sign varies by clip), so the old "HyperSwap > InSwapper" claim is unsupported; XSeg
  mixed vs FP32 IoU mean 0.9918, min 0.858. 47 unit tests.
- **Parallel stabilization no longer swaps and restores the overlap frames twice (`docs/perf/stab_warmup.md`).** A block primes its filters by running
  the whole pipeline on the WU frames before it (the previous block's last WU frames) and discarding the picture: 144 of 744 frames on a 600-frame clip.
  With `temporal_detection` on, the live kps filter and the landmark smoother are not in the render loop (the tracking pre-pass smoothed the cached faces
  sequentially), so the swap net and the restorer are pure functions of (frame, face, M). `roop/stab_dedup.py` lets whichever block reaches an overlap
  frame first publish `(fake_frame, enhanced_frame, scale_factor, swap_model_mask)`; the other takes it and runs only its own mask / enhancer / HF
  filters. Every mask processor still runs on each visit - `process_mask` applies the block's `MaskStabilizer` inside itself, so nothing it returns is
  shared. Eligible only for `temporal_detection` with no live kps filter, no ordered temporal engine, `swap -> [restore_ultra | restoreformer++] ->
  mask*`, `warmup <= block`, per face no rotation/frontalization/temporal-quality; otherwise today's path, untouched. Bounded by `ROOP_STAB_DEDUP_MB`
  (384); `ROOP_STAB_DEDUP=0` turns it off. ABBA, live config, 600 frames, pinned chunk budget: **d4 +6.8%, s7 +9.6% fps, output file sha256 and decoded
  md5 identical in all four arms**, swap calls -136 / -126 (16.4% / 19.3% of all face inference), peak RSS unchanged. 31 unit tests.
- **Larger blocks (8x) are not worth it at the RAM this machine has, and nothing was changed.** At equal chunk RAM 8x is 9.6% (d4) and 3.7% (s7) SLOWER
  than 4x. Against a sequential render with the encoder taken out (x264 CRF 0, new `ROOP_BENCH_CODEC` / `ROOP_BENCH_CRF` env overrides in
  `two_face_video.py`) the mean difference falls monotonically with block size (0.0188 -> 0.0134 -> 0.0125 of 255 for 24/48/96-frame blocks), i.e.
  "closer, never further" holds, but the shipped blocks are already 0.019/255 away. The RAM-derived chunk budget the brief asked for already exists
  (`_default_stab_chunk_mb`, 75% share on a 32 GB desktop); its 40% cap would have shrunk it here.
- **A hevc_nvenc comparison is not a pixel comparison.** The same two renders differ by 0.65/255 encoded and 0.019/255 lossless: a rate-controlled
  encoder turns one sub-pixel difference into a whole-frame difference that persists. Pinned-geometry renders were bit-identical in every repeat
  (16 parallel, 3 sequential), so the "0.7142/255 noise floor" in AGENTS.md did not reproduce; it may belong to un-pinned geometries.
- **XSeg bound / CUDA-graph sessions (opt-in `ROOP_TRT_BOUND=xseg=graph`) and RF++ batching: measured, nothing kept on by default**
  (`docs/perf/trt_bound_sessions_2026-10-09.md`, commit 6dbe7be). The 10-06 "graph not bit-identical" finding was a harness bug.

## 2026-10-05

- **`lighting` stage: LCT bit-identical and 42% cheaper at 512 px (`docs/perf/stage_precision_verify_lighting_2026-10-05.md`).** `_color_transfer_lct`
  converted the whole target crop to LAB and read every 16th pixel, and clipped through fresh 3 MB arrays on ten threads. It now converts only the sampled
  pixels and clamps/truncates in place into a bounded per-thread buffer: 512 px 5.17 -> 3.01 ms, 256 px 1.06 -> 0.93; in a d4 render `lighting` 6.8 -> 5.6 ms/call
  with a byte-identical output (`490d9baa...`). `tests/test_color_transfer_lct_exact.py` keeps the old implementation as the reference (`array_equal`).
  Same report: **XSeg and RestoreFormer++ already run on TensorRT mixed** (one TRT node per run, no CUDA/CPU partition; Stage 0 ran provider cuda) - but
  against an FP32 reference the shipped mixed path misses the stated gate on real crops (XSeg IoU min 0.860, 17/120 faces < 0.995; RF++ PSNR min 31.2 dB,
  12/120 < 45 dB); FP32 passes for XSeg (+0.7 ms/call) and nearly for RF++ (+18 ms/call). Nothing changed pending a decision. Verification already
  detects through the GPU detector when `retinaface_r50_gpu` is active (451/451 main-pass executions, counters); pinned by `test_verify_detector_routing.py`.
- **Peak VRAM held under ~90% without changing a model or a precision (`docs/perf/vram_budget_2026-10-05.md`).** (1) After a replayed
  render's pre-pass the FaceAnalysis pool and every hybrid detector pool shrink to width 1 (`face_util.shrink_analysis_pools`, from
  `_release_replayed_analysis`; the ceiling ends with the pool and is cleared at every run start): -56 MiB on the live scrfd config,
  **-1,240 MiB on `retinaface_r50`** (peak 73.3% -> 63.2%). (2) `vram_governor.plan_job` now lowers pool widths - one context of one
  pool at a time, the one freeing most first - after the swap batch and before GPEN's look-changing resolution, holds the plan to
  `max(margin, 10% of the card)`, and publishes the widths to `TensorRTResourceManager.set_budget_caps`, which enforces them on every
  pool (explicit settings too) until the render ends; decided from NVML free memory, never a GPU name. Another process holding 5 GiB:
  peak **96.7% -> 76.9%**, no fps loss, pools 2/2/2 -> 1/1/1. (3) Windows, once per process at >= 95% VRAM: a warning naming NVIDIA
  Control Panel > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback for this `python.exe` (admission, the peak sampler and the
  `[VRAM]` stage log; written through `bar_write`). Output bit-identical in every arm (d4 600 frames `490d9baa...`). New harness flag
  `--governor` (`two_face_video.py`, `baseline_snapshot.py`): that harness bypasses the governor otherwise. Not measured: the 3060, or
  consecutive renders in one session.
- **Batched aux models for the tracking pre-pass: built, measured SLOWER, REMOVED.** recognition + `landmark_2d_106` +
  `landmark_3d_68` for every face of the frames waiting at once, on batch-relaxed ONNX copies (TensorRT profile 1..16) with GPU
  crops that reproduced `cv2.warpAffine` bit for bit. d4 pre-pass 60.9-66.6 fps per-face vs 49.8-50.3 batched FP16 / 44 FP32
  (`track_detect` 29-30 vs 39 / 44 ms; CPU crops neutral): the two detector workers give 1.36 faces and 1.00 frames per batch,
  nothing to amortise. Embedding cosine >= 0.9998; the 0.3 px landmark bar could not be met against the existing path, whose
  TensorRT-FP16 68-point output is itself a median 1.25 px (max 9.5) from FP32 truth. The code lived in `f11c8c2` as an opt-in
  (`ROOP_AUX_BATCH=1`) and was removed afterwards; `face_util` is back to the per-face aux loops. `docs/perf/aux_batch_2026-10-05.md`.

## 2026-10-04

- **New detector engine `retinaface_r50_gpu` (not the default).** face_engine's on-device RetinaFace R50 wired in as a hybrid engine
  (`roop/retinaface_gpu_engine.py`): padding, pyramid (same triggers, single-pass reuse), TensorRT AOT engine, decode and gate on the
  GPU; the app's `nms_keep` / `diou_nms` finish on the survivors; pool from `detector_pool_size`; the bound runner is logged. Preprocessing
  defaults to `squash` (reproduces the existing r50 to 4 faces in 3,412); `ROOP_R50_GPU_PREPROCESS=letterbox` is face_engine's geometry
  (loses 76/24/0/83 of the existing engine's faces and boxes an empty floor at 0.92). A/B vs scrfd (`docs/perf/gpu_engine_2026-10-04.md`):
  wrong-faceset 0 in all 20 renders; swapped d4 855 vs 799, d1 1243 vs 1147, d6 1456 = 1456, **Love 412 vs 506**; 131 of 3,264 baseline faces
  not matched at IoU 0.5 (the existing r50 loses the same ones); detector-only 4.8-5.2 ms at 720p/1080p (existing r50 10.8-10.9, scrfd 5.7-7.4),
  but pre-pass fps is not better. The AOT engine is stamped on file mtime, so `app/models/retinaface_r50.onnx` would silently run on ORT;
  the wrapper uses the registry copy. d9.mp4 no longer exists.
- **Multi-scale detector reuses its single pass (`face_detector.MultiScaleFaceDetector.detect`).** Only the retinaface /
  retinaface_r50 engines reach it (not scrfd, so no render on the current config changes). When the adaptive single pass
  triggers the pyramid, its result now fills the pyramid's scale-1.0 slot and only the other levels run: 4 -> 3 detector
  inferences per triggered frame. The scale passes run in the caller's thread when it holds a pooled FaceAnalysis lease
  (`face_detector.in_pool_worker`, set by `lease_face_analyser`), else on one persistent executor instead of one per call.
  `generate_scale_pyramid` already built every level once from the padded frame; trigger thresholds untouched.
  Measured on r50 (`docs/perf/pyramid_2026-10-04.md`): 1,468 frames incl. all of d6 4K and 80 synthetic close-ups, merged
  boxes/kps/scores 0.0 px from the old code (plain and pool-worker thread); d6 4.0 -> 3.0 inferences/frame; whole r50 renders
  of d6 old/new/new/old: no hang, identical md5 / swap audit / verdicts; function-level +10.8..+11.9% from the reuse, the
  pool-worker shortcut itself neutral. On r50 the pyramid merge is not redundant (it returns fewer boxes than the plain
  single pass on 213 of 488 d6 frames).
- **The rotated rescues run the aux models only on faces they keep (`face_util._rotated_pass`).** `_rescue_rotated` and
  the partial-miss rescue detected on a rotated frame with recognition + 106/68-point landmarks and discarded the aux
  work of every duplicate. They now detect with `aux=False` (and `unclamped=True`, so SCRFD geometry is what `fa.get()`
  gave), test duplicates on an un-rotated COPY of the coordinates, then run the aux models on the rotated frame from the
  rotated keypoints for survivors only and un-rotate those. Measured (`docs/perf/rescue_aux_2026-10-04.md`): 1,558 faces
  on 1,018 frames bit-identical to the old code (bbox, kps, embedding, landmarks; old-vs-old null identical too); whole
  renders of d4, Love and d1 give the baseline's decoded md5, swap audit and WRONG-FACESET counts exactly; aux calls
  d4 pre-pass -34%, Love -4%, d1 -10%, detector executions unchanged; d4 pre-pass 30.2 -> 34.3 fps (ABBA). Love's
  timing is not measurable (the machine slowed mid-sequence).
- **`_rescue_upscaled` stays on retinaface_r50 (proposed skip rejected on its own gate).** Skip-if-never-gains: on the 175
  frames where r50's first pass is empty in the baseline windows it returned a face on 13 (Love 10, d4 3) - mostly large
  partly-occluded kiss faces that scrfd misses, plus a few spurious boxes. `tests/probe_r50_upscale_gain.py`, decision
  recorded at the call site, `test_rotated_pass.py` pins it.
- **Baseline probes (measurement only, no behaviour change).** New `roop/baseline_probe.py`; see the
  `ROOP_PROFILE` row in `ENV_FLAGS.md` for what it prints. `[Session] model file provider trt_fp16 input
  requested` is logged once per distinct session at the five construction sites (buffalo_l via
  `face_util._build_face_analyser`, `retinaface._build_one`, `Mask_XSeg`, `FaceSwapInsightFace` incl. its
  fp32/no-TRT rebuilds, `Enhance_RestoreFormerPPlus`), reading the provider ORT actually BOUND and the TRT fp16
  option off the live session, and flags a bound provider that differs from the first one requested.
  `tests/baseline_snapshot.py` drives the four reference clips and writes `docs/perf/baseline_<date>.md`.
  Tests: `app/tests/test_baseline_probe.py`.
  First baseline (4070, `docs/perf/baseline_2026-10-04.md`): pinned to the first run's `ROOP_STAB_CHUNK_MB`, repeats
  are bit-identical; unpinned, free RAM changes the stabilizer geometry and the pixels; the first render of a clip
  runs 28-40% slower than a bit-identical repeat (cause not isolated); at 4K only 3 of 10 stabilizer workers fit
  the RAM budget; on SCRFD `bool('auto')` makes `has_multiscale` true so the close-up and padded rescues never
  run (counters, not code reading). Nothing was changed on the back of these.
- **"The fps collapses after a while" is the footage getting busier, not the render degrading (measured on
  a live 54,714-frame job and reproduced from fresh processes).** Job: 1276x716, hyperswap + Restore Ultra
  + XSeg, TensorRT/mixed, 10 stabilization workers, i9-14900K + RTX 4070. True fps per 3,600-frame part
  (from the `[Resume] part N written` times): 38 (frames 3.6-7.2k) -> 22 -> 19 -> 13 -> 10 -> 9 -> **7.8**
  (39.6-43.2k) -> **16.6** (43.2-46.8k) -> 10.7 -> 9.1; whole job 13.5 fps avg. It recovers, so it is not
  monotonic ageing. The same frames rendered by a FRESH backend (slice checkpoints through
  `/api/projects/<id>/resume`, 1,500 frames each) gave the same rate for the same position, so process age,
  leaked state, thermals and host memory are all excluded: frames 3.6-5.1k **33.6 / 33.0 / 35.4** fps,
  25.2-26.7k **17.4**, 43.2-44.7k **17.2 / 17.6 / 19.3**. Swapped faces per frame (swap audit, which
  includes the warm-up frames) 0.54 / 1.14 / 1.21, so swapped faces/s was **18.0 / 19.8 / 21.0** - flat,
  even slightly rising. A two-point fit gives about 8 ms per frame plus **~41 ms per swapped face** (10
  workers), a ceiling near 24 faces/s; fps = 1 / (that), so fps must fall as faces per frame rise. Same
  finding as 2026-09-29's 30 -> 16 fps (0.45 -> 1.15 faces/frame).
  What the live process looked like at 10 fps (py-spy `--gil`, 25 s, no pause): the **GIL was held in 95% of
  samples**, 2.4 of 32 cores busy, GPU 30% at 2835 MHz with no throttle reason, RSS a 13.6-15.7 GB
  sawtooth with a ~55 s period (one chunk of frames) and no growth, 0 pages in/out, pagefile 0.7 GB, thread
  and handle counts flat, and an idle backend back at 4.2 GB afterwards. The GIL profile is flat (~40 host
  lines at ~1%): Restore Ultra 25%, warm-up frames (re-run and discarded, 24 f blocks / 6 f warm-up) 22%,
  colour transfer 7.7%, the outcome-guard re-detection (`_verify_after`) 7.2%, masks 5.5%, stabilizer 5.4%.
  Nothing there scales with elapsed frames. Not found, therefore not "fixed": a leak, an O(frames) scan,
  throttling or paging. The lever is less host work per face (`phase-12`), not a fix to the drift.
  **Shipped:** the `[Pipeline]` line now ends with `| 1.20 faces/frame, 21.0 faces/s` (per window and in the
  closing `done` line), so a busy stretch reads as what it is. It counts faces painted into FINISHED frames
  (a per-thread tally in `_composite_faces`, read back in `update_progress`; warm-up frames are written off
  in `_process_block`), which is why it is ~20% under the audit's swapped count. Checked on the real path:
  0.43 faces/frame @ 35.4 fps = 15.2 faces/s early, 0.97 @ 19.3 = 18.7 faces/s late.
- **`python run.py --project X.roop --render` had never worked, and its first fix exposed a second silent
  failure.** (1) `render_project` imported `roop.globals as globals_` and then used the builtin `globals`:
  `AttributeError: 'builtin_function_or_method' object has no attribute 'output_path'` before any frame.
  (2) Past that, the render ran on CUDA / fp32 instead of the app's TensorRT / mixed, because the
  `MODEL_RUNTIME_INIT` phase that sets `roop.globals.execution_providers` runs in `ui/main.py` only.
  Restore Ultra's graph then failed on every frame (cuDNN frontend `GRAPH_EXECUTION_FAILED` on
  `/encoder/downsample/conv_3`), each failure wrote the ORIGINAL frame, and the run still ended "Done" at
  17 fps - a render that swaps nothing and reads fast. `render_project` now runs that phase with the config
  first; verified: `provider_active=tensorrt ... precision=mixed`, 0 failed frames, 35.57 fps and 15.2
  faces/s on frames 3.6-5.1k, the same as the backend. Tests: `test_project_io_nle.py`
  (`test_render_project_reaches_the_worker`). **Still open:** `--output <dir>` is not honoured by the
  headless path (the slice outputs landed in `app/output`).
- **Side finding, not chased: every job after the second in one backend starts with ~2.2 GB less free VRAM
  and the VRAM governor steps the swap batch 8 -> 4 -> 2 -> 1.** `[VramGovernor]` read 10,575 / 10,510 MB
  free before jobs 1 and 2, then 8,040 / 8,302 / 8,298 MB before jobs 3-5 (headroom 1.2-2.3 GB, "batch cap
  1"). fps was unchanged (33.0 vs 33.6 fps; 17.6 vs 17.2) - consistent with cross-frame batching being
  neutral on this GIL-bound path - so it is a mis-accounting to fix, not a slowdown measured here. Probably
  the same thing as the "job 0 ~9 fps, later jobs ~7.6" note in the entry below.

## 2026-10-03

- **More VRAM does not make the render faster (measured on the real backend, 2026-10-04).** The question
  was whether the 7.4 of 12 GB a render uses can be spent on speed. Real backend, `config.yaml` as shipped,
  `/api/swap` on s3.mp4 frames 0-200 (hyperswap + Restore Ultra), three consecutive renders per arm, NVML
  sampling; steady-state loop fps (jobs 1-2; job 0 reads ~9 in every arm) and VRAM peak:
  shipped pools 2/2/2, four arms: 7.57 / 7.55 / 7.66 / 7.61 / 7.56 / 7.34 fps, **7.1-7.3 GB**;
  pools 4/4/4: 6.70 / 6.46 fps, **11.7 GB** (-12%); pools 3/3/3: 7.35 then 4.77 fps, 9.6 GB (degrades
  across renders); `ROOP_CUDA_ARENA_STRATEGY=kNextPowerOfTwo`: 7.60 / 7.65 fps, 7.2 GB (no effect). The
  swap batch is already at its engine's 8 and the provider limit is already 10 GiB. GPU utilisation
  averaged ~30% over a job either way, so the card is not the constraint a bigger pool would relieve;
  it matches the earlier pool / enhancer-pool / detmask-pool results (monotonically worse past 2).
  The shipped 2/2/2 is the best point and the ~3 GB left free is the safety margin
  (`vram_safety_margin_gb: 3`), not waste. Two traps hit while measuring: a test-rig process sits at
  12.0 GB by itself (it keeps a second set of models resident), so it cannot answer a headroom question,
  and killing a python process does not stop a background shell loop that launches the next arm
  (overlapping renders produced 12 GB / 1 fps readings that were pure contention).
  Side observation, not chased: in every arm, job 0 of a fresh backend ran ~9 fps and the next
  renders in the same backend ~7.6 (-17%).

- **Every test failure fixed (full suite green); two real defects found on the way.**
  * **`torch.backends.cudnn.benchmark = True` removed from `roop/core.py`.** It made
    `tests/test_stage8_compositing.py::test_thread_safety` fail in every full run and pass alone, and
    it was a real defect: with it on, a ROI whose size changes every frame (a face that moves and
    scales) re-autotunes cuDNN on each new shape - the compositing engine's CUDA path measured 17.7
    ms/call off and 456 ms/call on single-threaded (485 at 8 threads, 7.5 off); for one repeated shape
    it bought nothing (17.4 vs 17.5 ms; GPEN 256 Pro GPU filter 1.6 vs 1.6). PyTorch's cuDNN benchmark
    cache is thread-local, so each worker autotuned alone and could pick a different algorithm: with
    `roop.core` imported (what a full run adds), 430 of 960 concurrent calls differed by one level on
    ~4 pixels from the single-thread result. `app/tests/test_cudnn_determinism.py` pins the flag and the
    concurrent determinism (the same check diverges in 44 of 128 calls with the old flag).
  * **`Enhance_GPEN256Pro` swallowed a model-lifecycle fault silently** (`except Exception: pass`), which
    `test_enhancer_guards` forbids; now reports through `swallowed()` like its siblings.
  * **Stale test expectations updated** (not code): `test_hardware_portability` still assumed the
    12 GB desktop's thread knee is 10, but Rule 4 (>= 11 GB and >= 16 cores) makes it 20 and the loader
    deliberately migrates an unstamped legacy value below the derived one; the "same machine" test now
    writes the user-pinned stamp the app writes, the "unstamped" test computes its expectation from the
    machine's own derivation, and the derive test covers 12 GB/24 cores -> 20 and 12 GB/12 cores -> 10.
    `face_engine` `test_zoo_declarations_are_complete` / `test_unavailable_model_fails_clearly` knew 25
    models and 6 swappers; the zoo has 32 entries and 8 swappers. The "unique filenames" assertion now checks what it
    meant (different models never share a file): four entries are deliberate ALIASES of one artifact
    (same file, URL, size and SHA-256).
  * **face_engine's environment, declared and verified.** 35 failures and 12 errors were
    `No module named 'av'` / `'kornia'`: `face_engine/requirements.txt` now lists both (installed with
    `--no-deps`; torch, numpy, opencv and onnxruntime unchanged). Tests that need a compiled
    TensorRT engine fell back to an in-process build (GPEN-512 403 s, xseg_3 509 s, hyperswap_1a 224 s on
    an RTX 4070) and a run looked hung for 20+ minutes; `tools/compile_engines.py --models all` builds all
    ten once (verified for fidelity), and the Ultra Restore end-to-end tests and the web UI test now
    SKIP with the command to run when the engine is absent (checked without building anything).
  * **`web_ui` did not build from a clean `npm ci`:** `vite.config.ts` uses `process` and `tsc -b`
    failed with TS2591 because `@types/node` was never a dependency and `tsconfig.json` lists `types`
    explicitly. Added `@types/node` 22.20.5 and `"node"` to `types`; build and the 26 vitest tests pass, and
    the browser end-to-end test (real Chromium, real server) passes.
  * Detector latency test (`test_1080p_latency_on_rtx_4070`): first RUN of it here, because it was skipped until the SCRFD engine existed. Median of one 200-iteration block measured 1.92-2.06 ms on an idle 4070 against a hard 2.0 ms, a coin flip (README 1.75 ms was another day); now best of three blocks against 2.2 ms (an engine optimised for 384x640 instead of 640x640 was only ~3% faster, so not changed).
  * Result: `face_engine` 283 passed / 2 xfailed (~6 min); app + root trees 4963 passed / 24 skipped / 2 xfailed, 0 failed.

- **`pinokio_batch_runner.py` had never rendered anything; fixed, and a failed video no longer kills
  the queue.** The Stage 9 brief asks for a 50-job batch through it. The first real run died on the
  first frame of the first video: `TypeError: QueueProgress.__init__() got multiple values for keyword
  argument 'total'`. `ProcessMgr` builds its progress bar as `ChunkedProgress(total=N, ...)`; the
  replacement class wrote `total=total or kwargs.pop("total", None)`, which short-circuits whenever the
  frame count is known, so `total` was passed twice. Nothing tested the runner or
  `roop/process_manager.py` (no test imported either), which is how it stayed broken. Now
  `make_worker_progress()` (module level, tested). Also: `IsolatedVideoBatch.run` raised on the first
  failed video and never started the rest, so a 50-video run died at video N; `--keep-going` /
  `run(keep_going=True)` records `{"status": "failed", "error": ...}` and continues (default
  unchanged; exit code 1 if any failed). `app/tests/test_batch_isolation.py`: real `spawn` children
  with stub workers - a fresh process per video, progress events reach the parent, stop on first
  failure by default, continue with `keep_going`, a child that dies silently (`os._exit`) is a failure
  and not a hang, stop requests kill the child, and the progress class takes the keywords
  `ProcessMgr` passes.
- **50-video batch through the runner, measured.** 50 distinct 24-frame 1080p clips (hyperswap + Restore
  Ultra, shipped config), `--keep-going`, memory sampled from outside every 2 s: **50/50 completed, 0
  failed, 2867 s.** The runner process: 24.2 MB for the first 30 jobs, 17.6 MB at the end (it never
  imports torch/onnxruntime); no python or ffmpeg process left afterwards. Each child peaks at about
  10.0 GB RSS (first ten jobs) vs 10.2 GB (last ten), +4.5 MB/job, which is not cumulative because
  every child is a fresh process (the child's peak follows that moment's free-RAM-derived stabilizer
  budget). System-wide free RAM during jobs drifted 11.1 -> 9.7 GB and GPU memory read 839 MB before and
  1455 MB after with nothing of ours alive: other applications (browsers, desktop); per-process GPU
  memory is not reported on this driver, so the GPU number is not attributed process by process.
- **Telemetry latency under a render (Stage 9 brief): the in-process design misses 25 ms, and the
  cause is model loading, not the frame loop.** `/ws/telemetry` answers a text `ping` with a `pong`;
  timed at 20 Hz against the real backend during three consecutive renders
  (`tools/probe_telemetry_latency.py`): idle p50 0.6 / p99 7.5 / max 7.6 ms; during the first ~25 s of
  a job (model load) p50 1.0 / p95 120 / **p99 1249 / max 5200 ms**; in the frame-processing phase p50
  0.9, p95 5-11, p99 13-97, max 145 ms. `py-spy dump` taken while a ping was outstanding put the GIL on
  `onnxruntime InferenceSession` creation (`_create_inference_session`) in every case, with
  `onnx.shape_inference`, `release_face_analyser` and a hardware-profile call alongside: C calls that
  hold the GIL for seconds freeze the event loop, the WebSocket and the progress sampler together.
  Only another PROCESS removes that. Not built: moving `/api/swap` into a worker process rewrites the
  parts of `api.py` that share the render's in-process state (the progress dict the sampler reads,
  live preview and `/ws/frames`, pause/stop, project checkpoints, `roop.globals` that routes mutate),
  and is a decision for the owner, not a side effect of a perf task. Already process-isolated:
  `pinokio_batch_runner.py` (a spawn child per video, no CUDA in the parent) and
  `distributed_render.py` (a process per GPU).
- **Shared-memory IPC and a lifecycle manager (Stage 9 brief): not needed as specified.** The runner's
  progress is a stdlib `multiprocessing` queue at <= 2 events/s; no measurement shows it as a cost, and
  the parent must not import torch, so `torch.multiprocessing` would be a regression. `release_resources`
  already runs at the start of every job (face analyser, `ProcessMgr`, caches, `gc.collect`,
  `torch.cuda.empty_cache`), and `gc.collect` already runs at hard cuts in the pre-pass and in the
  stabilizers. `empty_cache` at a cut would free nothing: quiescent torch allocation is 9.4 MB because
  inference runs through onnxruntime, whose sessions and TensorRT contexts are only released by
  tearing the session (process exit, for the batch path) down.

- **Landmark smoothing (Stage 8 brief): measured against the smoother that actually runs; a Kalman
  option added.** The brief assumed One Euro was the smoother. It is not the default: with
  `stabilize_landmarks` on (the default) the tracked pre-pass runs `AdaptiveLandmarkSmoother`
  (velocity-adaptive EMA, beta 0.35 still / 0.90 fast, coupled with the dense landmarks); One Euro
  and the "Smoothing method" select / min-cutoff / beta sliders were only consulted when that was
  off, so on the default path they changed nothing. Measured on one real face rendered with sensor
  noise through x264 (detector sigma 0.52 px), static and with a known path (0.4 Hz sway + a 60 px
  head turn in 8 frames), keypoint jitter removed above 2 Hz / 4 Hz, lag, head-turn peak error:
  shipped smoother -67.4 % / -73.8 %, 1.00 frame, 2.30 px; One Euro 0.1/0.1 -50.8 % / -58.0 %,
  0.50 frame, 1.65 px (raw detector 1.41 px). No One Euro min_cutoff / beta / d_cutoff gets 80%
  without a frame of lag and 2-3x the head-turn error (its speed estimate is a barely filtered
  derivative, so detector noise opens it on a still head). A constant-velocity Kalman filter with
  innovation-gated process noise (`KalmanKpsStabilizer`, `stabilize_method: kalman`) removes
  **-77.0 % / -81.0 %, no measurable lag, 1.79 px**, tracking error 1.01 px against 1.94 px for the
  shipped smoother. **That is on a still head only.** On real conversational footage (Weeds, Monica
  Bellucci; continuous single-face runs of 100-126 frames, heads moving, mouths talking) the >4 Hz
  band removed is 11.6 % / 6.4 % for Kalman against 12.2 % / 9.0 % for the shipped smoother, because
  there that band is partly real motion; so it is an opt-in, not the default (the brief's 80% holds
  for a still head with this filter, and for nothing on moving footage). Wired into the tracked
  pre-pass (dense landmarks follow the keypoint centroid shift) and the per-frame path, added to
  the React method select. Real render: same fps (3.45 vs 3.44), 98.7% of face frames swapped in
  both, A/V passes, output differs (frame PSNR 45.4). The stored detections are
  `app/tests/data/synthetic_face_jitter.npz`; `tools/gen_jitter_fixture.py` regenerates them.
- **GPU color transfer / multiband compositing (Stage 8 brief): measured, not built.** Reinhard
  (LAB mean/std) on torch CUDA: 1.2 ms at 256^2 and 1.4 ms at 512^2 with upload/download, against
  2.7 ms / 11.1 ms for the shipped single-threaded cv2 path (mean difference from cv2's 8-bit LAB
  0.9/255, max 5: not bit-identical). The existing `cuda_laplacian_pyramid_blend` (3 levels, already
  behind `composite_multiband`) is 1.7 / 3.8 / 9.1 ms at 400^2 / 640^2 / 1024^2 against 4.8 / 12.4 /
  32.0 ms for the numpy blend. So "under 3 ms" holds for a small ROI with transfers and for the
  resident case; it is not built because the render is GPU-bound and the stage shares are small
  (lighting 3.2% and blend 5.7% of thread time, earlier profile), and "entirely in VRAM" needs the
  frame on the GPU while decode and encode are host pipes. Not measured: the 3060 laptop.

- **Matte blur / erode / dilate ran over the WHOLE frame; now over the matte's support, bit-identically
  (`roop/mask_roi.py`).** The Stage 7 brief asked for the mask feather chain on the GPU. Instrumenting
  `cv2` through a real 1080p render found the real defect first: the paste matte is non-zero around one
  face, yet `blur_area` (3x3 blur, elliptical erode, a Gaussian ~10% of the face wide) and
  `create_landmark_mask` (hull dilate) ran on the full 1920x1080 uint8 frame: GaussianBlur k105-113
  ~77 ms/call, dilate 39-43 ~43 ms, erode 27-29 ~19 ms, ~224 CPU-ms per frame at the app's
  single-threaded cv2 (`tools/profile_cv2_mask_ops.py`). `mask_roi` runs the SAME cv2 call on the
  matte's bounding box padded by 2r+2 and writes into a zero frame; outside the box the input is 0 and
  blur/erode/dilate of 0 is 0, so it is exact: `app/tests/test_mask_roi.py` asserts `np.array_equal`
  against cv2 (blobs on every frame edge, thin bands, islands, single pixels, kernels to 111, ellipse
  and rect). Real renders: 246-frame 1080p, 8 counterbalanced arms ROI on/off, **all eight outputs
  byte-identical** (md5), and the perf suite's pre-change goldens still match bit for bit. **End-to-end
  fps is NEUTRAL** (4.69 vs 4.67; spread +-0.1) as the GPU-bound frame predicts; process CPU time -4.3%
  (230 vs 240 CPU-s per render); in-render the k~110 blur dropped 77 -> ~24 ms/call. `ROOP_MASK_ROI=0`
  restores the full-frame calls. Gain scales with how small the face is in the frame (3.0x at a 250 px
  face, 1.9x at 450 px).
- **GPU mask morphology (Stage 7 brief): measured, not built.** `tools/bench_mask_gpu_morphology.py`:
  torch `conv2d` Gaussian matches cv2 to 0.004 (the brief's 0.01 is met) and the whole chain is
  0.2-4 ms on the GPU against 2-130 ms of cv2, BUT `max_pool2d` erosion/dilation is a SQUARE; the
  shipped erode (`blur_area`) and hull dilate use `MORPH_ELLIPSE`, and the square differs from cv2's
  ellipse by up to 1.0 per pixel (whole chain 0.13), so the recipe cannot meet its own bound against the
  production masks (an ellipse needs per-row 1-D pools). It would also add GPU work to a render that
  waits on the GPU, and the numpy blend still needs the mask back on the host. The production XSeg is
  DFL XSeg 256 on TensorRT, ~2.5 ms/call (4.6 ms with its cv2 resize and 256 KB host I/O); the 15-25 ms
  the brief quotes was CPU matte work, now mostly removed above. Temporal mask reuse already exists as
  `XSeg3MaskCache` (2.5 deg / 1.8 px, close to the brief's 2 deg / 1.5 px) but is wired only inside the
  `Mask_XSeg3` engine; at ~2.5 ms/call there is little for it to save on the production engine.

- **Batched restorer acceleration (Stage 6 brief): measured, not built.** RestoreFormer++ (the network
  under Restore Ultra) on the 4070, TensorRT "mixed" FP16 through the app's own provider policy,
  GPU-resident IOBinding, real FFHQ-aligned crops, a dynamic-batch graph from the swapper's
  `_relax_batch_dim` with the repo's own TensorRT shape profile: **24.4 / 25.6 / 25.2 / 25.0 ms per
  face at B = 1 / 2 / 4 / 8** (second run 25.1 / 25.1 / 25.0 / 24.8; the fixed batch-1 production
  engine read 25.2 and 21.7 between runs, i.e. ~15% engine-to-engine spread). One 512 network already
  saturates the card, so the contract's -60% per-face latency is unreachable by batching (measured
  0%). SSIM of the shipped FP16 "mixed" output against TensorRT FP32 is 0.9960-0.9970 mean
  (min 0.9950) at B=1, already under the brief's 0.998, so a faster FP16 path cannot meet that gate
  either. The rest of the brief already exists or was already measured: FP16 "mixed" is the policy
  default; IOBinding is in use; the < 48 px bypass is `enhance_gate.py` (`enhance_min_face_px`,
  default 0 = off because it changes the look; set 48 for the brief's behaviour); GPU-resident
  pre/post scaling measured 1.1-40x slower and NEUTRAL end to end (see the restorer audit below).
  `tools/bench_restorer_batch.py` reproduces the table.

- **An untrimmed render lost its last video frame whenever the source audio ended before the video
  (`util_ffmpeg.restore_audio`).** A 90-frame 29.97 fps render came back with 89: the single-command
  branch (used when the trim starts at 0) ended with `-shortest`, and source audio routinely ends a
  fraction of a frame early (an AAC packet is 21.3 ms). The trimmed branch had been fixed for the same
  defect (b1.mp4, 120 -> 117); this one had not. It is now bounded by the render's own video length
  (`-t`), as the trimmed branch is. Found by the new `tests/test_performance_regression.py` A/V check;
  `app/tests/test_restore_audio_frame_count.py` (real ffmpeg, fails without the fix).
  `scripts/verify_roop_keep.py` also found no `ffprobe` on Pinokio's PATH and reported every output
  "unreadable"; it now falls back to the app's resolver.
- **Performance / fidelity regression suite (`tests/test_performance_regression.py`,
  [`docs/development/PERFORMANCE_REGRESSION.md`](development/PERFORMANCE_REGRESSION.md)).** Production
  renders of a still, 1080p and 4K, PSNR/SSIM/LPIPS vs a baseline, an identity "did it swap" gate, A/V
  timing and a leak soak. Measured finding: a render is not a pure function of its config - the
  stabilizer block size follows FREE RAM, and 12- vs 16-frame blocks differ by face SSIM 0.971 / LPIPS
  0.030 on one config. The rig pins `ROOP_STAB_CHUNK_MB`. The render tests carry a `perf` marker and are
  deselected from a bulk `pytest` run.

- **Startup canary for TensorRT swapper engines (`roop/swap_canary.py`, hooked in
  `FaceSwapInsightFace._canary_gate`).** An inswapper TensorRT FP16/"mixed" engine can build, warm up,
  report the TensorRT provider and run at full speed while emitting the WRONG picture for every face:
  inswapper unrolls InstanceNorm and the squared terms reach ~7e6, past FP16's 65504, and whether TRT
  keeps them in FP32 depends on tactic selection at build time. Measured on the RTX 4070 (TRT 10.9, ORT
  1.23.2, 14 real crops, SSIM vs CUDA FP32): production options 0.9973 min / identity unchanged;
  the same options without `trt_build_heuristics_enable`, or without the 2 GB workspace + sequential
  build, 0.749 / identity 0.762 -> 0.087. Nothing in `predictor.verify_and_warmup` can see it. The canary
  compares the engine to a transient CUDA FP32 session on two synthetic inputs (good >= 0.9958, corrupt
  <= 0.8653; floor 0.98), and on failure rebuilds on the TensorRT FP32 engine (15.9 ms, SSIM 0.99985,
  identity 0.7685 vs 0.7682), else CUDA/CPU. Verified end to end through the real `Initialize`: production
  engine passes (0.9974, 0.45 s), an injected corrupt engine is caught (0.40) and replaced; no false
  positives on inswapper / hyperswap / hififace / realswap (all >= 0.9974). `ROOP_SWAP_CANARY=0` disables.
  Also fixed: `_rebuild_without_trt` called `get_onnx_session_options` which was only imported locally
  inside `Initialize` (a latent `NameError` on the GHOST fallback), and the model-lifecycle log hard-coded
  `precision="fp32"` for every inswapper while production ran it mixed.
- **Stage 4: device-side alignment, inversion, masks and compositing in `CudaAffineBatch`
  (`roop/optimized_processor.py`) - dormant path, NOT wired into the production render.**
  `GpuFaceSwapProcessor` / `MemoryStreamingProcessor` / `vectorized_pipeline` already existed (batched crops,
  `[B, 512]` identity, `grid_sample` warp, in-VRAM paste) but had no callers and no tests. Added:
  `similarity_from_landmarks` (closed-form least-squares similarity on the GPU, == `estimate_norm`),
  `invert_affine` (batched closed form, == `cv2.invertAffineTransform` incl. its singular -> zeros),
  `gaussian_blur` (== `cv2.GaussianBlur`, incl. OpenCV's version-dependent fixed 1/3/5/7/9 kernels) and
  `box_mask` (Gaussian-feathered box matte, cached), an `occlusion_masks` / `occlusion_provider` hook so an
  XSeg tensor stays in VRAM, and `dynamic_batch_model_bytes` (a swap model that accepts PRE-BOUND batched
  outputs: `_relax_batch_dim` leaves 237 stale batch-1 `value_info` entries, so a bound `[B,3,128,128]` output
  was rejected). **Behaviour change in the dormant path:** `paste_faces` now feathers by default when the model
  emits no matte (it used to paste the whole square crop with a hard edge for inswapper); `feather=False` keeps
  the old behaviour. 39 tests (`app/tests/test_cuda_affine_batch.py`).
  Measured (RTX 4070, d1.mp4 1080p, 240 frames, 1.8 faces/frame, real inswapper_128 FP32 on the CUDA EP, real
  detections and source, each arm run in both orders): CPU/OpenCV one crop at a time 13.0 fps, 0.075 CPU-s/frame,
  GPU 46%; same with a batched model call 13.5 fps (model batching alone is +3%); GPU arm 25.5 fps (1.89x),
  0.042 CPU-s/frame (-44.5% vs CPU, -43.5% vs batched CPU), GPU util 91%; GPU-vs-CPU output SSIM 0.9998, mean
  difference 0.027 of 255, 3e-5 of pixels differ by more than 2 levels, max 9 levels. Parity: matrices ~1e-9
  (float64) / ~1e-6 relative (float32), Gaussian masks ~2e-6; the RESAMPLING cannot reach 1e-4 against OpenCV
  (OpenCV quantises sample coordinates to 1/32 px: bilinear max 5.5e-3 of full scale, 44% of pixels above 1e-4;
  5x worse under the global TF32 flag). Caveats: this is a sequential swap-stage harness, not the production
  render - production runs 20 threads and is GPU-bound with tracking, masks, stabilisation and enhancers, where
  the warp/blend is ~6-10% of host time, so the whole-render CPU drop is far below 40%; tensor-core activity was
  not measured (nvidia-smi does not expose it). ORT's CUDA EP runs convolutions in TF32 by default, which is why
  batched rows differ from batch-1 rows by 1.4e-2 (2.4e-5 with `use_tf32=0`).
- **Stage 1-3 optimisation prompts audited, nothing built** (decode/zero-disk, pre-pass keyframes +
  embedding cache, ORT FP16/IOBinding): the default render already streams through rawvideo pipes with
  no frame images (one 31 MB encoded segment is the only temp write), the pre-pass already runs N=8
  keyframes + ROI + scene-cut, and cached embeddings cannot reach cosine 0.995 (adjacent frames 0.913).
  IOBinding and the requested session options measure at zero gain; the ORT FP16 converters corrupt
  this graph even with every op blocked. See the memory notes `default-render-already-zero-frame-files`,
  `stage2-prepass-contract-unreachable`, `stage3-ort-contract-measured`.

## 2026-10-02

- **Pluggable recognition backend (additive; live matching NOT rewired).**
  `roop/recognition_registry.py` pins six recognisers (w600k_r50, AdaFace IR-101, Glint-R100 /
  antelopev2 - one file, MobileFaceNet, SFace) by URL + SHA-256, every one downloaded and
  hash-checked; MagFace / CosFace / GhostFaceNet are NOT registered because no ONNX export
  exists to pin. `roop/recognition_engine.py` builds a TensorRT > CUDA > DirectML/CoreML > CPU
  session and VERIFIES each tier after a real inference (ORT drops a provider silently, even
  during the first run), TensorRT runs behind a per-engine lock. `face_analyser` gains
  `set_recognition_model` (build -> publish -> release hot-swap), `extract_face_embedding`
  (the repo's `align_crop`, arcface_112_v2), `fuse_quality_weighted` and `IdentityBank`.
  Deliberately not routed into production: `face.embedding` also feeds the swapper, thresholds
  are per-model (only w600k/AdaFace have any), and the engine's embedding from the final
  `kps` agrees with `face.embedding` only to 0.937-0.995 cosine (insightface's own
  `norm_crop` gives the same, so the live vector comes from an earlier landmark estimate).
  Measured on the 4070: TensorRT FP16 vs CPU min cosine 0.99996+; EXHAUSTIVE cuDNN ~= HEURISTIC
  for these nets; a build/release cycle returns to the same +120 MiB (CUDA context), no growth.
- **All six recognition models compared on one video with a face swap (`tools/compare_recognition_swap.py`; report
  `docs/development/RECOGNIZER_VIDEO_COMPARISON.md`).** On `Monica Bellucci .mp4` (7,647 frames, 15,900 faces) each model
  decides which faces are "the main subject" and only those get harjot swapped in; a 3x2 grid video shows the six decisions
  with live speed figures. Quality: w600k, AdaFace and Glint-R100 are practically tied (decisions differ on 68-87 of
  15,900 faces); MobileFaceNet and SFace miss 6.5-11% of the swaps the majority agrees on, SFace flips twice as often.
  Speed (4070, TensorRT FP16, batch 1): 0.90 ms/face SFace to 2.25 ms Glint-R100/AdaFace, but detection dominates, so the
  pipeline-fps spread is only 34.0-37.6. No ground truth: the subject is a model-voted cluster of tracks and nobody is
  identified by face. Lessons recorded in the tool: pixel difference is the wrong yardstick for "was this swapped"
  (a swap keeps lighting, so a swapped face differs by ~7/255; use identity cosine to the source), and the real pipeline's
  "selected" mode swaps only the captured target (92% of faces had no track entry), so swapping every face needs
  `ProcessMgr.process_face` driven directly. The render writes to `*.part.mp4` and renames on success: an MP4 has no
  playable index until the end, and an earlier version overwrote a good file with a half-written one.
- **Every registered recogniser calibrated against w600k on 16 real clips: none wired, and a keypoint finding.**
  (Written for glintr100; `mobilefacenet` and `facerecognizersf` were added the same day: both clearly worse than
  w600k, AUC intervals below zero, EER 5.5-6.2% vs 3.1%; `antelopev2` is glintr100's file.)
  `tools/calibrate_recognition.py` (track-based same-person pairs, scene cuts break tracks, clip-level
  bootstrap) on 3,836 same / 1,183 different pairs. glintr100 was never better than w600k and was the lowest
  point estimate in every cut (AUC 0.9842 vs 0.9898 on identical raw-keypoint crops, interval below zero on all
  15 clips; borderline once 3 label-noisy clips are dropped); AdaFace is indistinguishable from w600k on identical
  crops. The pipeline's keypoint refinement (`_refine_kps_from_68`) runs AFTER buffalo_l embeds, so every
  recogniser that aligns from the final `face.kps` - AdaFace in production included - reads a worse crop:
  replacing refined with detector keypoints improves AUC for all three models (AdaFace +0.0052, CI +0.0013 to
  +0.0129) and cuts AdaFace's false accepts at w600k's false-reject rate from 19.0% to 8.5%. FIXED the same day:
  `face_util._stash_recognition_crops` builds the recognition crop from the detector keypoints just before the
  refinement (AdaFace only, separate key, released once embedded) and AdaFace prefers it; measured through the
  live path AdaFace's distances now equal the raw-keypoint ideal for all 5,019 pairs (AUC 0.9830 -> 0.9883, FAR
  19.0% -> 8.5%), regression benchmark PASS with and without AdaFace and pixel-identical to the unchanged code.
  The AdaFace thresholds were NOT re-calibrated (distances moved up to 0.48); that is still owed. Method,
  tables and caveats: `docs/development/RECOGNIZER_CALIBRATION.md`.
- **Read-only faceset embeddings in any recognition model's space (`roop/faceset_manager.py`); no
  loader or render calls it.** `load_and_validate_faceset(fsz, model_id)` returns a faceset's reference-face
  embeddings in `model_id`'s space and NEVER writes the archive. Why not the "migrate the archive when the
  backbone changes" design: a legacy `.fsz` stores only PNGs (no embeddings; `_ingest_faceset` re-detects
  and re-embeds on every load), and a V2 archive's cached vectors are w600k by design - the swapper's
  identity, which the temporal identity code pairs with the target's w600k vector - so rewriting them would
  change swap output. `default` returns V2's cached vectors (no detection, no engine) or the detector's own
  embedding; any other model detects the references, aligns with `align_crop` and uses the ACTIVE engine
  (it refuses rather than swap engines). Unusable faces are listed in `skipped`, never zero vectors; results
  are immutable and cached per (archive hash, model). `describe_faceset` reports the declared embedding
  space (`identity.embedding_model` if a future writer adds one; absent = `default`). Measured on 6 real
  archives: the default-space result equals the loader's own embeddings (3/3 identical); AdaFace
  re-embedding takes 1.2 s for 6 archives (16 ms cached); same-faceset cosine mean 0.70 vs 0.12 across
  people, extremes touching (min 0.438 / max 0.448), so a gate needs a calibrated threshold first. A test
  fails if any app module imports it.
- **Worker-process recipe for the recognition engine (`roop/worker_pool.py`); nothing in the render uses
  it.** The app's pipeline is one process with worker threads, and `face_engine` already has a spawn
  pool that builds its processor inside each worker and assigns GPUs round-robin, so no pool was added
  to either. This is the tested recipe for putting `RecognitionInferenceEngine` behind a
  `ProcessPoolExecutor`: the engine is NOT picklable (build it in the pool initializer), spawn only,
  round-robin GPU assignment through a shared counter, and CPU cores divided between workers (new
  `cpu_threads` engine option; the default was every physical core per session). Measured on the 4070
  with two spawned workers building a cold TensorRT cache at once: both initialised, a second run read
  the shared cache, embeddings matched a parent CPU reference at >= 0.99997 cosine. A failing
  initializer surfaces as `BrokenProcessPool`, not a hang.
- **The swapper's identity input is guarded and isolated from the tracking recogniser.** The swap
  path was already separate (it reads the 512-d w600k vector on `face.embedding`, computes the latent
  once per source face and caches it on the Face + a shared LRU), so no parallel swapper was built.
  Two real holes were closed: `HyperSwapSourceCache.get_latent` reshaped anything to (1, 512) and a
  ZERO embedding became a zero latent that was cached (a swap toward nobody, reported as success).
  `roop/swap_identity.validate_identity_embedding` now gates that path and the crossface-converter
  path (size, finite, non-zero; the message names the hazard). A shape check cannot tell a 512-d
  AdaFace/Glint vector from w600k, so that case is closed structurally:
  `tests/test_swap_identity_isolation.py` fails if any production file outside the recognition modules
  imports the tracking API (AST scan; `ALLOWED_RECOGNITION_IMPORTERS`). Not done, on purpose: a
  GPU-resident source-latent cache (the latent is 2 KB, already computed once; ORT io_binding was
  removed for CUDA error 999) and a separate `frame_swapper.py` (the draft skipped inswapper's emap).
- **Recognition verification + benchmark harness** (`tests/test_recognition_pipeline.py`).
  Correctness on models already on disk (a missing one is a visible skip, never a download; real
  aligned faces check same-vs-different separation, margin 0.10 = half the narrowest measured gap
  of 0.195), GPU/CPU equivalence, and real fallback (including ORT itself dropping CUDA). Benchmark:
  `python tests/test_recognition_pipeline.py --benchmark [--device auto] [--out t.md]`. VRAM is
  device-wide (`torch.cuda.memory_allocated` cannot see ORT's allocations and would print 0), warm-up
  includes a clock ramp, and a failed model is a FAILED row + non-zero exit. Mutation-checked: removing
  normalisation fails 13 tests; assuming the provider instead of reading it back fails the ORT-drop test.
- **Settings > Face recognition (React + API; the Gradio UI is untouched).**
  `roop/ui_recognition.py` (framework-neutral: tier advice from compute capability + the
  providers ORT offers, model catalogue, apply), `routes_recognition.py`
  (`GET /api/recognition/models`, `GET /api/recognition/current`, `POST /api/recognition/set`
  `{model_name, provider}`: 400 bad input, 409 while a render runs, 500 with the reason (traceback to
  the server log only) and nothing saved when the build fails) and `RecognitionPanel.jsx`, shown under
  Identity & tracking as "Identity API: embedding backend" (the existing "Recognition model" control
  above it is the live-matching `recognizer`). Applying also updates App's settings state
  (`recognitionSync.mergeSelection`): App autosaves the WHOLE settings object, so without it the next
  unrelated edit would post the stale selection back and undo the saved choice. Two new settings, `recognition_model` and
  `recognition_provider` (default `default` / `app` = follow the app's provider), persisted by
  `Settings` and READ by `face_analyser.get_recognition_engine()` on first use. The panel states
  that live swap matching is not switched by it. Tier advice does not use
  `runtime_optimizer.HardwareProfiler.profile()` (~15 s: ffmpeg, TensorRT builder); only the CPU
  tier suggests a different model (MobileFaceNet, measured 3.9 ms vs 41 ms), and only as a suggestion.

- **A CUDA session that came up CPU-only was invisible; now it raises.** The
  `predictor` assertion only fired when TensorRT was *requested*, so a session that asked
  for CUDA and silently got `CPUExecutionProvider` (missing cuDNN DLL, a CPU-only
  onnxruntime wheel shadowing onnxruntime-gpu, a rejected provider option) passed every
  check. `assert_session_providers` now raises `ProviderAssertionError` when CUDA or
  TensorRT was requested and no GPU provider is active; a session that requested no GPU
  provider (`cpu`, `force_cpu`, CPU-only models) can never trip it, and a TensorRT->CUDA
  drop is still only fatal where it was before. It is now also checked (a) after the
  warm-up inference, because ORT drops the EP *during the first run* (`verify_and_warmup`),
  and (b) on every session built through `backend_manager.build_session_with_fallback` -
  the detectors and `buffalo_l`, which never called the assertion. That check sits outside
  the `try` on purpose: a CPU-only session is not a build failure and must not be
  swallowed into the next fallback attempt. `ROOP_STRICT_PROVIDER=0` downgrades it to a
  warning + recorded degradation, as before.
- **Startup now says what is bound and what it costs.** One `[Provider] <tag>: active =
  ...` line per session and a `VRAM used / total (+delta since previous model)` line per
  loaded model (device-wide, read from the driver); `predictor.bound_sessions()` exposes
  both. `core` prints the resolved CUDA EP options once.
- **CUDA EP `gpu_mem_limit` defaults to 10 GiB** (an explicit `perf_gpu_mem_limit` /
  `ROOP_CUDA_MEM_LIMIT` still wins). Per-session arena ceiling: on the 12 GB desktop it
  keeps one session from pushing the card into WDDM shared-memory paging; on the 6 GB
  laptop it is above physical VRAM and changes nothing. Not scaled down for the laptop -
  a tighter cap there is untested.
- **Deliberately NOT changed (request said otherwise):** `cudnn_conv_algo_search` stays
  `HEURISTIC` (DEFAULT measured +55-241% slower per `cudnn_algo.py`; the CodeFormer
  family is lowered per model by a device probe); `inswapper_128`/swappers stay off a
  FP16 graph (FP16 overflow -> rainbow smudge, `precision_policy` `face_swap: fp16 =
  unsafe`; FP16 also costs identity 0.352 -> 0.407); detectors are not converted to an
  FP16 file (they already run FP16 under TensorRT `mixed`, and a landmark shift would
  break alignment). `ORT_ENABLE_ALL` is already the `get_onnx_session_options` default.

- **Frame dumps to disk: the render was already zero-disk; two helper caches were not.**
  The default render is a rawvideo pipe in (`video_stream`) and out
  (`NvencRawWriter`/`ffmpeg_writer`: NVENC `-preset p5`, `-pix_fmt yuv420p`, audio
  muxed) and the unified scheduler holds at most `_MAX_STREAM_INFLIGHT = 4` frames. What
  still wrote frames: (1) the SAM2 pre-pass wrote EVERY frame of the clip as a JPEG for
  SAM2 to decode again - now `SAM2FrameBuffer` builds SAM2's exact input tensor in RAM
  (bit-equal to `_load_img_as_tensor`, tested), injected into `init_state` through a
  scoped, always-restored patch of `sam2.sam2_video_predictor.load_video_frames`; (2) the
  scrub-preview fallback cached one JPEG per probed frame in `%TEMP%/roop_scrub_cache` -
  now an ffmpeg stdout pipe (lossless BMP) plus an 8-entry RAM LRU.
  NOT changed, by decision: Keep Frames (Frame Editor deliverable), per-frame-mask
  re-processing and the legacy `use_new_method=False` route still use the frame-folder
  path; removing them removes features. SAM2 still holds the whole clip as float32
  (~12.6 MB/frame at 1024^2), a limit of the library, not of disk.
- **Per-thread queue path capped at 64 live frames** (`frame_capped_queue_depth`,
  `ROOP_MAX_FRAMES_IN_FLIGHT`, 0 = off): live frames are `threads * (2*depth + 1)`; at
  20 threads depth 3 meant 140, now depth 1 = 60. Applies ONLY to the per-thread queue
  path (scheduler disabled, or stabilization off and scheduler not allowed); not
  A/B-measured. The stabilized parallel path (`_run_stab_parallel`) is deliberately NOT
  capped: it holds ~5 chunk copies sized by a RAM-share budget (`_default_stab_chunk_mb`,
  the 16 GB OOM fix and the 2-worker-round throughput tuning), which is hundreds of
  frames at 720p-1080p by design.
- **`--execution-batch-size N`** (run.py and core.py): sets `ROOP_BATCH_SWAP_MAX`, the
  cross-frame swap batch ceiling. Beats config `perf_batch_max`, and is released from the
  settings-owned set so a UI save cannot replace it mid-process. 1 = no batching; the
  effective size is still clamped to worker threads and the VRAM governor.
- **Batched swap + GPU warp: already done / already rejected, not redone.** Cross-frame
  batching has been live since 2026-08-15 (`SwapBatcher` -> `RunBatchMulti`, one source
  identity per crop, ceiling 8 on >= 11.5 GB). The swapper is `realswap` by default;
  `inswapper_128` is not what renders. Moving `cv2.warpAffine`/blend to torch CUDA was
  measured 1.1-40x SLOWER at real call sizes and reverted (bf96c1f), and the swap stage
  is at the GPU floor (~75 faces/s; deleting a whole network buys +4.8%), so more GPU
  work cannot help a GPU-bound pipeline.

- **Live pipeline monitor: FPS + queue state every 100 frames.** The decoupled pipeline
  asked for (decode / GPU worker / NVENC writer on bounded queues, off the GIL) already
  exists: ffmpeg decode (NVDEC where the codec allows, `nvdec_reader`) and NVENC encode
  run in child processes, a reader thread and a writer thread bound the workers, and a
  frame-lease semaphore is the real memory bound. It was NOT restructured: the one-owner
  stream (`scheduler.run`: one CUDA owner, three threads) is only used for streaming
  stabilization because TensorRT contexts are not shareable across threads and stateful
  filters cannot advance out of order, and the threaded worker path measured 3.4x faster
  (2026-09-24). What was missing was visibility while it runs - the scheduler's bottleneck
  verdict printed once, at the end, with diagnostics on. `roop/pipeline_monitor.py`
  now logs `[Pipeline] frames N | X fps (avg Y) | in a/b (empty p%) | out c/d (full q%) |
  verdict` per window (`ROOP_PIPELINE_LOG_EVERY`, default 100, 0 = off) and a closing
  `done:` summary. Queue state is sampled on every finished frame, and the verdict is the
  share of samples where the INPUT side was empty (`DECODE-STARVED`) or the OUTPUT side
  was full (`ENCODER-BOUND`); a full input queue is healthy (the reader should be ahead of
  a GPU-bound pipeline). It hooks `update_progress`, the one call every path makes per
  frame, and `_pipeline_queue_state` reads the queues of the path actually running - the
  per-thread lists exist on every render but are structural zeros on the stabilized and
  one-owner paths. It cannot raise into a render (probe and emit failures are swallowed).

- **Small-face restorer gate (`enhance_min_face_px` / `ROOP_ENHANCE_MIN_FACE_PX`, default
  0 = off).** The restorers are the one stage where the network itself is the cost (at its
  own floor; no pool, thread or host trick left - 09-28 A/B, `enhancer-pool-does-not-help`),
  so the only lever is running it less. A face whose shorter detected-box side is under the
  threshold in frame pixels skips the restorer and the swapped crop is pasted as-is - the
  same result as the "fast bilinear resize" alternative, since the paste warp is that
  resize, and the same `enhanced_frame is None` state "no enhancer selected" already
  produces, so every stage after it was already correct. Per-track hysteresis (resume only
  at 1.15x the limit) stops a face at the edge flipping every few frames, which would read
  as texture flicker. A UI slider ("Skip restorer below face size (px)"), live between
  renders; `[EnhanceGate] restorer skipped on N of M faces under T px` prints at the end of
  every render with the gate on. OFF by default because it changes how small faces look.
  Proven executing on the 4070 (regression clip, threshold forced to 2000): enhance stage 0
  calls, 408/408 skipped, swaps 300/300 intact, and the regression gate correctly flagged the
  look change (face SSIM 0.9638 < 0.970). Forced all-skip is an UPPER bound (main pass 14.6 ->
  32.2 fps), not what a real threshold buys: that clip's smallest face is 108 px, so 96
  would skip nothing there. The win is wide/crowd footage.
- **Restorer audit - not changed, with the evidence.** (1) FP16: the ONNX restorers already
  run TensorRT FP16 ("mixed") via `precision_policy`; GFPGAN (flat grey face), GPEN
  1024/2048 (NaN) and DMDNet (PyTorch, FP32-only) are marked unsafe on measured failures.
  (2) `torch.compile` / `autocast`: only DMDNet is PyTorch (already `inference_mode`), and
  `triton` is not installed here, so inductor cannot generate CUDA kernels. (3) Batching to
  `[B,3,512,512]`: one TensorRT context already saturates the card (1->6 contexts: 40.0 ->
  36.0 faces/s), GPEN is a StyleGAN with batch-1 baked tensors, and the exports are fixed
  batch-1. (4) Keeping crops on the GPU between swap and restore: moving cv2
  warp/blend onto torch CUDA measured 1.1-40x slower at real call sizes (bf96c1f), and the
  GPU-vs-CPU look filter was measured NEUTRAL end to end on a 600-frame counterbalanced A/B
  (GPEN 256 Pro 7.10 vs 7.19 fps; Restore Ultra's 12 ms CPU finish 7.99 vs 7.98), because a
  render costs about its restorer's network GPU time (+9.0 / +16.9 / +25.7 / +30.8 ms per
  frame for GPEN 256 / 256 Pro / UltraMax / Restore Ultra).

- **`tools/build_trt_engines.py` built engines the app never loads; new
  `tools/prebuild_engines.py` builds the ones it does.** The old tool compiles
  inswapper_128 / GPEN-512 / scrfd_2.5g into a flat `<repo>/models/trt_cache` with its own
  provider options; the app reads `app/models/trt_cache/<namespace>/` (GPU, sm, driver,
  CUDA/TRT/ORT versions, precision, tuning knobs) for the models in config.yaml
  (`hyperswap`+`hififace`, Restore Ultra, XSeg, buffalo_l...). An engine is keyed on that
  namespace and the session's options, so nothing it wrote could ever be hit - the same
  "reports success while not running" class as the rest of this file. It now logs a warning
  saying so (behaviour otherwise unchanged; automation may call it). The new tool brings the
  app up the way a render does (`init_pipeline` -> the processor loop `ProcessMgr.initialize`
  runs), runs one dummy inference per session (TensorRT builds on the FIRST inference, not at
  construction), and per stage prints the time, the engine-cache growth (>= 256 KB = COLD, built
  now; else warm), and whether each session is on TensorRT AFTER that pass. A stage with no
  inspectable session is reported UNVERIFIED rather than fine (the first draft said "every
  session is on TensorRT" while two of four stages had inspected none). Proven on the 4070: warm
  stack = 4 stages, 12 sessions verified, +0 MB, 26 s; a forced cold namespace
  (`ROOP_TRT_BUILDER_OPT_LEVEL=1`, analyser only) = 83.1 s, +173.6 MB, COLD, 5 sessions on
  TensorRT (that test namespace was deleted afterwards). Run with the app stopped:
  `python tools/prebuild_engines.py [--only analyser,swapper,mask,enhancer]`.
- **Not built, with the reasons.** (1) A hand-written native-`tensorrt` runner with `.engine`
  files: native TensorRT exists in `FaceSwapInsightFace` (`_native`) and `trt_quant.py`, and was
  measured NOT faster than ORT's TRT EP with IO binding (RestoreFormer++ 21.11 vs 20.44 ms;
  native FP16 numpy 8.9 vs 6.4 ms swapper); INT8 cost -0.026 identity and FP8 runs 0 FP8 layers on
  Ada/TRT 10.9 (`swap-int8-fp8-rejected`). Targeting sm89 and explicit profiles (swapper min 1 /
  opt 4 / max 8) are already how the app builds. (2) Compile in the background while the app
  renders on CUDA: an engine is keyed on the exact session options of each loader (graph
  optimization level, shape profile, precision-forced cache dirs), so a throwaway session only
  helps if it matches every one of ~40 construction sites, and it would contend for the GPU and
  VRAM with the render. The cache already matches on GPU + driver + compute capability by
  construction; run the prebuild once before starting the app instead.

- **Why the GPU sat at ~65% on a stabilized render, and a +9.6% fix for small-frame clips.**
  Diagnosed on `D:\k1.mp4` (480x854, 11,179 frames, live config: hyperswap + Restore Ultra +
  XSeg, all stabilizers on, 20 threads) which rendered 13.2 fps main pass (pre-pass 69.6 fps,
  end-to-end 10.7). Sampled at 1 Hz: system CPU 29%, busiest thread 62% (max 94%), GPU SM 65%,
  143 W of 200 W, memory bandwidth 43%, reader read-wait 0.1 s and writer stall 0 s over 1,500
  frames - no resource saturated and the GIL not pegged. The profiler shows 7.9 frames in flight
  against 12 workers: ~20-25% of worker time goes on the WARM-UP frames every block re-runs and
  discards (6 of every 30 frames in a 24-frame block, uncounted by `frame_total`), and ~12-17% on
  the join barrier at the end of every chunk (the fastest worker finishes at 50-90% of the slowest;
  `[STAB CHUNK]` imbalance 25% of processing wall on this clip's close-ups). The ceiling from GPU
  work alone is ~24 fps (swap 7 ms + mask 3 ms + restorer ~20 ms per face x 1.37 faces/frame), so
  ~90% GPU utilization would be ~21 fps - but utilization is a symptom, not a target: removing
  warm-up work removes GPU work too, so fps rises while SM utilization stays ~65%.
  `_stab_parallel_geometry` now picks an 8x-warm-up block (priming 25% -> 12.5%) automatically
  WHEN the RAM budget still holds two whole rounds of them (2 x workers blocks); otherwise the 4x
  block is kept, because a blanket 8x would make the shrink steps cut 1080p blocks below today's.
  New knob `ROOP_STAB_BLOCK_MULT` (2-16, explicit always wins; 4 restores the old behaviour).
  Counterbalanced A B C C B A on frames 3000-7600, 4,600 frames, identity unchanged on every arm
  (0.330, 38/38): default 18.77 fps (18.80 / 18.73, 0.4% apart) | `BLOCKS_PER_WORKER=4` 19.31 |
  `BLOCK_MULT=8` 20.58 (20.66 / 20.50). Default path re-run after the change: 20.12 fps, banner
  `24 blocks x 48f`; the 1080p regression clip keeps `12 blocks x 24f` and its output is identical
  to baseline (face SSIM 1.0). The whole k1 clip is NOT 20 fps: this slice is easier than average
  (the first 1,500-frame slice was 19.4, the clip average 13.2), and the per-face restorer is the
  dominant cost on its close-ups. Remaining known loss: the end-of-chunk join barrier (12-17%);
  a persistent pool that starts the next chunk's blocks before the current chunk joins would
  recover part of it but touches the pause/checkpoint/writer-drain machinery - not done here.

## 2026-09-29

- **3060 (sub-7 GB) temporal pre-pass: 72% of its wall clock was rescue detection.**
  Sampled the live render with py-spy (40 s, 2 people selected, 720p, 60,778 frames,
  ~10.7 fps, GPU 25% / 32 W): first detector call 28%, everything else rescue -
  partial-miss turns 33%, cardinal-turn ladder 21%, CLAHE 9%, 2x upscale 6%.
  Causes: (1) a ROI crop around ONE tracked face inherited `expected_count=2` from
  `TARGET_FACE_GROUP`, so every crop paid the three-turn partial-miss rescue for a
  person who was not in it, and on a ROI miss the ladder then ran again on the
  full-frame fallback; (2) on the full frame a person out of shot re-ran the whole
  ladder every frame. Fix (`face_util.small_card_prepass_active`, `RescueBackoff`,
  `rescue=` on `get_all_faces*`): on a sub-7 GB card, inline scan only, ROI crops
  skip the ladder and a futile rescue backs off to every 8th frame. The pooled
  4070 path is byte-for-byte unchanged. NOT yet measured end to end - see
  SESSION_LOGS 2026-09-29.

## 2026-09-27

- **Upper-lip colour and kiss flicker (Love.mp4, harjot on the woman, AdaFace).**
  1. `oral_cavity.py`: the landmark fallback took 106-point indices 66..71 as the
     inner mouth. 67, 68 and 71 are the OUTER upper-lip edge, so the hull was the
     upper-lip vermilion. It read a closed mouth as open on 118 of 120 sampled
     frames, and pasted the target's own upper lip, sharpened and darkened up to
     35%, over every swap (on by default). Now it uses the inner ring
     (65 66 62 70 69 | 57 60 54), and "open" means an inner gap ≥ 0.10 of the mouth
     width (70/120). Upper lip vs the original: 8.20 → 6.01 LAB, now equal to the
     lower lip's 6.04. The mismatch between the two lips is gone.
  2. `procmgr_tracking`: on AdaFace the track-assignment floor was the w600k 0.45
     rescaled to 0.30, which sits inside the target's own band (measured 0.06-0.44;
     bystanders 0.53+). That refused her 99-frame kiss track at 0.32. The floor is
     now calibrated on AdaFace (`ROOP_TRACK_ASSIGN_FLOOR_ADAFACE`, 0.45).
     Her profile in the kiss: swapped 47 → 66 of 108 faces. On/off transitions
     13 → 8. No face of the man swapped (every swapped face ≥ 0.8 inspected).
  3. Follow-up, verified on weeds.mp4: once the mask covered the real aperture,
     the restore pasted the original's blurry teeth, sharpened ×1.64, over the swap's
     clean teeth on every open mouth. The result was grey, mottled, doubled teeth. It
     now restores only when its own `detect_clamped_lip_artifact` says the swap
     collapsed the mouth. Across 174 open mouths on both clips, 0 had. Speech-driven
     (lip-sync) restores are unchanged.
  Still open (weeds.mp4): on a head turned ~120° away the detector puts keypoints on
  hair and cheek. Pose from those reads −8° to −68°, the outcome guard agrees with
  them, and a faint face is painted on the hair. Detector score, eye spacing and
  landmark plausibility all overlap real profiles, so there is no gate yet.
  Rejected: a second detector engine (SCRFD) as a partial-miss rescue. It added 151
  faces over 1550 frames and none was recognisable as her (see `face_util`).
  Still open: her face on the man's track in contact (39 frames); her profile
  fragments at 0.71-0.81, past what a frontal capture vouches for.

- **Flicker and "no swap when two faces are close": three causes, all fixed.** Measured on
  a 7647-frame, 135-cut, crowded single-person clip (Harjot on Monica Bellucci, live 4070
  config, AdaFace):
  1. `temporal_tracker.py`: the pre-pass COASTED (no detection, landmarks interpolated
     later) whenever any track was lost, e.g. a bystander walking out. That was 64% of
     frames, so most swaps were registered on guessed landmarks. Only the opt-in ROI cadence
     coasts now.
  2. `face_util._is_face_duplicate`: the 180° partial-miss rescue re-found the target with
     mirrored keypoints. Landmark divergence read that as a second person. The two boxes then
     marked each other's crop "shared", and the real face was refused with nobody beside it.
     Mirrored fits no longer count as divergence.
  3. `procmgr_tracking`: the track-assignment margin anchored on the person's best track in
     the whole clip, a frontal close-up in another shot. It refused 13 of the target's own
     crowd, profile and small-face tracks. The margin is anchored per shot now.

  | | before | after |
  |---|---:|---:|
  | target frames swapped | 87.8% | 96.0% |
  | on/off transitions | 114 | 49 |
  | refused "crop shared" | 195 | 21 |
  | swaps on interpolated landmarks | 3,119 | 50 |
  | end-to-end fps | 12.10 | 11.26 |

  The regression benchmark's golden-SSIM gate reads 0.9691 < 0.9700 with fix 1 alone. Fixes
  2 and 3 are bit-identical to HEAD there. On the synthetic clip, fix 1's only schedule
  change is that frames 1-3 are detected instead of coasted. The render changes by a
  constant ~1.4/255 inside the face from then on, with no colour or position shift. Its
  identity to the source is unchanged: 0.8174 vs 0.8168, paired +0.0006, better on 153 of
  300 frames. Registration error is identical (0.0240). Both beat the 09-25 golden (0.7883),
  so the baseline was re-recorded. `two_face_video.py` now applies settings through
  `apply_env`; its private copy never exported `ROOP_ADAFACE`, so the harness matched on
  w600k.

## 2026-09-26

- **Identity Blender: latent blends shipped; three of four attribute dials measured
  unwritable and disabled.** `roop/identity_algebra.py` blends up to four sources as
  `normalize(Σ wᵢzᵢ)` on the w600k unit sphere, each pose-matched to the target. It
  applies attribute offsets on the tangent plane, so `cos = 1/√(1+|t|²)` exactly, with a
  uniform-scaling identity guard (default cos ≥ 0.80). The hook sits in `ProcessMgr`
  directly after the V2 pose-embedding override. It keeps the raw norm for the
  converter-MLP swappers and skips image-source models and CSCS. The recipe arrives via
  `/api/identity/blend` or `identity_blend` in the preview/swap payload, which is also
  how queued jobs freeze it. The React dock has 4 drop slots, weight sliders, per-source
  share and cosine, and the dials.
  Directions come from `tools/fit_identity_directions.py`: 14,582 faces, 5,845 identities
  (LFW plus the local clips and facesets), ridge λ=1e-3, identity-disjoint folds. The
  local corpus alone (99 identities) gave sex AUC 0.60 at every λ; a planted-direction
  test recovers only cos 0.57 at 60 identities. Held-out scores: age r 0.57, sex AUC 0.78,
  jaw r 0.44, expression r 0.21. The held-out metric is Pearson r, not R², because R²
  rewards weak ridge while the direction gets *less* accurate (planted: R² 0.70 at cos
  0.80 vs R² 0.51 at cos 0.87).
  **Render validation** (`app/tests/identity_algebra_bench.py`, hyperswap + Restore
  Ultra, RTX 4070, paired faces re-measured by an independent buffalo_l):

  | arm | cos A | cos B | Δ rendered age | Δ jaw ratio |
  |---|---:|---:|---:|---:|
  | A100 (base) | 0.601 | 0.066 | – | – |
  | blend 75/25 | 0.566 | 0.184 | | |
  | blend 50/50 | 0.304 | 0.498 | | |
  | blend 25/75 | 0.169 | 0.616 | | |
  | B100 | 0.072 | 0.630 | | |
  | age dial ±30 y (fitted calibration, t=±0.17) | 0.60 / 0.59 | | +0.6 / −0.7 (noise ±2.5) | |
  | age step t=−0.75 / +0.75 | 0.55 / 0.47 | | **+6.3 / +3.7** (both older) | |
  | jaw step t=−0.75 / +0.75 | 0.52 / 0.48 | | | −0.011 / +0.009 |

  Age is not monotone, sex has no signed response, and expression is null, so those
  dials are disabled with the verdict shown (`roop/assets/identity_render_validation.json`).
  Jaw is monotone but weak; it stays enabled with ±1 = t 0.75. Blending is off until
  the user enables it. Not tried: other swap models, per-model directions, or a
  direction fitted in the swapper's own latent (hyperswap uses w600k directly, so a
  different space would need a different model).

## 2026-09-25

- **Swap-model INT8 / FP8 quantization, off by default (measured, not shipped).**
  `roop/trt_quant.py` builds a native TensorRT engine for the swap net, calibrated on
  500 real face crops, and runs it with zero-copy I/O: persistent torch CUDA tensors
  bound with `set_tensor_address`, and `execute_async_v3` on a torch stream. The tier
  is picked by compute capability: FP8 at 8.9 or above, INT8 from 8.0 to 8.6, FP16
  below that. The setting is `swap_quantization`: `off`, `auto`, `fp8`, `int8` or
  `fp16`. `tools/build_calibration_set.py` harvests the set: it wraps the swapper's own
  `_infer` during `live_swap`, so the crops went through the live preprocessing by
  construction, and it asserts the blob is exact. The 500 samples are stratified by
  pose, light, ITA skin tone and occlusion over 17 clips; two more clips are held out.
  Engines and calibration caches carry a manifest (model SHA, set SHA, TensorRT, GPU,
  capability, recipe). A mismatch recalibrates headlessly with a tqdm bar.
  Measured on the RTX 4070 against 186 held-out faces (`tests/quant_quality_bench.py`):

  | arm | id to source | vs FP16 | faces losing >0.02 | PSNR vs FP16 | GPU ms/call |
  |---|---:|---:|---:|---:|---:|
  | ORT + TRT mixed (shipped) | 0.6663 | 0.0000 | 0.0% | 62.8 | (6.37 wall) |
  | native FP16 | 0.6663 | - | - | - | 4.17 |
  | native INT8 (56 layers INT8) | 0.6403 | **-0.0260** | **59.1%** | 28.0 | 3.69 |
  | native FP8 Q/DQ | 0.6707 | +0.0043 | 2.2% | 44.0 | **14.48** |

  **INT8 fails the identity gate.** It saves 0.48 ms of GPU per face and costs 0.026
  of identity, the worst face 0.11. The drop is in every stratum, from -0.018 to
  -0.034. **FP8 does not execute on Ada with TensorRT 10.9**: 0 layers run in FP8, and
  the Q/DQ pairs become standalone fake-quant kernels, 3.5x slower than FP16. The
  builder now rejects any reduced-precision engine that runs no layer at its precision,
  and remembers the rejection. So `auto` on the 4070 says why in the log and stays on
  ORT. Native FP16 costs 8.9 ms per call wall time against ORT's 6.4: the H2D/D2H
  round trip from numpy outweighs the 2.2 ms of GPU it saves. No arm is a candidate, so
  no end-to-end fps A/B was run. The RTX 3060 (INT8 tier) is not measured.

- **Blink sync, Eye-gaze follow, and expression split by region.** The LivePortrait
  expression restorer now weights each keypoint separately. Expression strength drives
  every keypoint except the eyes. `expression_gaze_follow` drives the eye keypoints.
  `expression_blink_sync` uses LivePortrait's own eyelid-retargeting network
  (`stitching_eye.onnx`). That network is driven by lid openings measured with
  LivePortrait's 203-point landmarker (`landmark.onnx`). Both models are fetched on
  first use, and their SHA256 matches Hugging Face's published digest. A config without
  the gaze key derives it from strength and region, so it renders bit-identically to
  before; a unit test compares the output byte for byte. Measured with
  `tests/expression_eye_bench.py`: hyperswap + Restore Ultra + XSeg, TensorRT, 1014
  frames, 66 of them with the eyes closed. The harness is deterministic: a repeated arm
  matched to the last digit.

  | arm | lid err | blinks caught | false closes | pupil err | id to source |
  |---|---:|---:|---:|---:|---:|
  | swap only | 0.044 | 33% | 0.0% | 0.041 | 0.596 |
  | **blink sync** | **0.019** | **97%** | 0.1% | 0.043 | **0.590** |
  | gaze follow 1.0 | 0.043 | 100% | 1.5% | 0.052 | 0.573 |
  | expression 1.0, eyes 0 | 0.046 | 35% | 0.0% | 0.039 | 0.512 |
  | old strength 1.0 'all' | 0.028 | 100% | 0.2% | 0.049 | 0.492 |
  | gaze 1.0 + blink | 0.040 | 100% | 2.7% | 0.056 | 0.575 |

  - **The plain swap kept the eyes open on 67% of the target's blinks.** Blink sync
    fixes that at almost no identity cost, and a contact sheet across a squeezed blink
    agrees.
  - **Gaze follow makes eye direction worse, not better.** The dark-iris position error
    rises on wide-open eyes (0.043 → 0.055). The swap already keeps the target's eye
    direction, and re-rendering the eyes adds error. The control is kept, and its help
    text says this.
  - **Gaze + blink first double-counted the lids** (6.4% false closes). The eye network
    now receives the post-delta keypoints (`blink_retarget_state`), which brings it to
    2.7%. Blink alone is still the recommended setting.
  - **Expression restore at 1.0 costs identity** (0.596 → 0.49–0.51), because
    LivePortrait re-renders the whole face. This was true before the split too.
  - **The fake gaze retargeter is removed.** `roop/processors/frame/face_swapper.py` had
    a "LivePortrait neural gaze retargeter". Its model was a 257-byte ONNX file the code
    wrote itself: a single Gemm layer with an identity weight and zero bias. Its only
    caller was `swap_face`, which nothing calls.
- **Frontalization (`use_frontalization`) swapped every face UPSIDE DOWN. Fixed; still
  net-negative.** `face_frontalize.get_frontal_landmarks_from_pose` re-projected the
  frontal reference at rvec = 0. `_REF3D_68` is y-up with the nose at +z, and an OpenCV
  camera is y-down looking along +z, so that reference was the head upside down and
  facing away. The affine fit to it was a vertical mirror (M[1,1] = -1.19, det < 0). The
  swapper was handed an inverted face and returned no identity, and the inverse warp
  pasted a ghost of the original. The reference now faces the camera (rvec = (pi,0,0)), and
  `frontalize_crop` refuses any fit with det <= 0
  (`tests/test_face_frontalize_orientation.py` fails on the old code). The swap net's own
  mask is now defrontalized with the face (it was inert: `swap_model_mask_strength` is 0).
  `verify_swap` was ruled out first: ROOP_VERIFY_SWAP=0 gave the same numbers.
  `tests/frontalize_yaw_bench.py`, 218 frames from the angle clips, 4070, live config,
  null arms repeated exactly. Identity to the source by yaw band:

  | arm | 0-30 | 30-45 | 45-60 | 60-75 | 75-90 |
  |---|---:|---:|---:|---:|---:|
  | off | 0.639 | 0.472 | 0.530 | 0.515 | 0.350 |
  | front_30, mirrored (before) | 0.529 | 0.066 | 0.017 | 0.028 | 0.039 |
  | front_30, fixed | 0.596 | 0.404 | 0.371 | 0.218 | 0.183 |

  Still worse than off in every band. One global affine cannot undo an out-of-plane
  turn: it shears the face, and the reflected border shows as a seam on the far cheek.
  It does place the features better past 75 deg (eye error 0.47 to 0.27). It stays off
  by default, and the UI warning now carries these numbers.
- **Selected-mode renders swapped nothing (4bd577d, fixed in a76091a).** Commit 4bd577d
  built `allowed_source_indices` in `ProcessMgr.swap_faces` with
  `faces = getattr(src_data, 'faces', None)`, which overwrote the frame's detected faces
  with the source faceset's. The per-face loop then matched the source photo against the
  target, refused it, and never saw the real face. Every render in the default mode was
  untouched video from 2026-09-24 23:32 until the fix. The suite stayed green (3454
  passed). The new regression benchmark found it on its first real run: 0/300 frames
  swapped, 83% "faster". `tests/test_swap_faces_detected_list.py` now pins every
  assignment to `faces` in `swap_faces`.
- **Regression benchmark**: `run.py --benchmark --benchmark-mode regression`. It runs a
  real 300-frame 1080p render of the user's configuration and reports raw decode/encode
  fps, per-stage throughput, end-to-end fps, p99 frame latency, VRAM peak and GPU
  utilisation. It fails on dropped frames, a stage that never ran, lost swap coverage, or
  face-crop SSIM/PSNR below the per-machine golden render. FPS is advisory only. See
  [`development/REGRESSION_BENCHMARK.md`](development/REGRESSION_BENCHMARK.md) for the
  positive and null controls.
- **Model integrity at startup**: `app/model_manifest.json` pins size and SHA-256 for
  inswapper, RetinaFace, GPEN 512/256 and BiSeNet. Each hash was checked against the
  host's own published digest. `roop/model_integrity.py` verifies them at boot (cached
  against size and mtime) and fetches missing or corrupt ones with HTTP Range resume.
  `conditional_download` now keeps a failed `.part` for resume instead of deleting it.
  Progress is served at `GET /api/models/integrity` and shown in the React
  `ModelSplash`. The update health probe verifies but never downloads.
- **Portable runtime** (`portable/run.bat`, `portable/run.sh`, `portable/bootstrap.py`):
  it runs with no Pinokio, system Python or Conda. It installs uv, a uv-managed CPython
  3.10 and a venv under `portable/runtime/`; runs the install.js dependency steps (PyTorch
  2.7 cu128, ONNX Runtime GPU 1.23.2 and TensorRT 10.9 via `provision_runtime.py`); adds
  static FFmpeg 8.1; and opens the browser when the API answers. `--build-bundle` fills
  `portable/wheels` and `portable/vendor` for a no-network install; `provision_runtime.py`
  gained `ROOP_WHEELHOUSE`, `ROOP_OFFLINE` and `ROOP_PROVISION_RECORD` (all unset under
  Pinokio). Verified on the 4070: TensorRT provider active, all 5 manifest models
  verified, React served. Not yet run on Linux or macOS. See
  [`../portable/README.md`](../portable/README.md).

## 2026-09-24

- **VRAM governor, auto-tune, and six performance settings.** Asked for: a VRAM governor that
  budgets a job and steps it down below 1.5 GB free; a 100-frame CUDA/TensorRT x batch
  1/2/4/8 x NVENC p1-p7 auto-tune saving the fastest profile; and a React settings panel for
  them. Audited first: `render_guard` already refused low-VRAM renders, pools already sized
  from live VRAM, and `/api/benchmark` already existed -- but it runs `process_frame` one
  frame at a time on one thread in preview mode, so the batcher, the worker pool and the
  writer never execute there: a batch axis through it would have measured nothing.
  - `roop/vram_governor.py`: at render admission, budget = models x contexts x swap batch
    (session_pool's specs, scaled onto the 3060's measured 2346 MB) + NVDEC surfaces at the
    video's resolution + process overhead; below `vram_safety_margin_gb` it lowers the swap
    batch 8->4->2->1, then GPEN 2048->1024->512. A sampler records the real peak and learns
    peak/estimate per configuration (`vram_calibration.json`). The prior errs LOW on
    purpose (inert, like before, until it has learned): live 4070, b1 600 frames, estimate
    3814 MB vs measured 8248 MB (2.16x), no step-down; the learned ratio now applies.
  - Auto-tune (`roop/benchmark/autotune.py`, `routes_autotune.py`, Settings panel): every
    arm is a real trimmed render of the LAST render's normalized request through
    `_run_swap`. 100-frame screen (each arm twice, A..Z Z..A), swap-count guard, "ran as
    labelled" check (effective batch/provider), then the top 2 vs the current setting at 600
    frames A/B/B/A; saved only if it wins both pairs by more than the baseline's own spread
    and 3%. NVENC: best-quality preset encoding at >= 2x the confirmed render rate. Writes
    config.yaml; Revert restores. Live 4070 (b1.mp4, hyperswap/Restore Ultra/TRT, 10
    workers), 26 arms, 13/13 checks, 0 arms not as labelled: CUDA ~3.2 fps vs TensorRT ~6.5
    in screening; `tensorrt/b1` +5.4% vs noise 2.6% (both pairs), `b2` +1.1% inside noise;
    NVENC hevc p7 325 fps vs 17 needed (5.23 vs p5's 5.36 Mbps). One clip, one ABBA: the
    b1 result is this workload's, not a general rule.
  - New settings (Advanced performance): `vram_safety_margin_gb`, `perf_batch_max`
    (ROOP_BATCH_SWAP_MAX; 1 now really means no cross-frame batching -- it was floored to
    2), `perf_nvenc_preset`, `perf_gpu_affine` (the gate moved into `cuda_warp_affine`, so it
    reaches every caller), `perf_pinned_buffers` (new ROOP_PINNED_BUFFERS in buffer_pool),
    `temporal_step` (defaults 1, warns). These are `LIVE_ENV_SETTINGS`: a save re-exports
    them, so they apply on the next render, not after a restart.
  - TensorRT engine cache panel (`/api/trt_cache`): status, per-namespace size, "clear
    stale" (namespaces other than the one this process builds into -- ~3 GB of orphaned
    a0/a-1 namespaces on the 4070) and "clear all".
  - Not built: QuickSync / VideoToolbox decode (only NVDEC exists; the codec list already
    offers qsv/amf encoders the ffmpeg build has). `optimized_processor.VramGovernor` is
    unrelated (vectorized pipeline, not reached by renders).

- **Trimmed renders came out with the video starting late against the audio. Fixed.**
  `restore_audio` cut the source audio with an INPUT-side `-ss` and `-c:a copy`. For
  stream copy that seeks the file to the video keyframe BEFORE the trim point and keeps
  the audio from there with negative timestamps; `-avoid_negative_ts make_zero` then
  shifted every stream, so the video started (trim point - preceding keyframe) late:
  `b1.mp4` trimmed at frame 200 -> video `start_time` 3.788 s, audio 0, audio 8.78 s long
  for a 5.0 s render. Found through the new output compare view, whose two sides showed
  different scenes while the clocks agreed to 16 ms. A trimmed render now cuts its audio
  in an audio-only pass with an OUTPUT-side `-ss` (exact to the packet, still a stream
  copy), then muxes, bounded by the video's own duration rather than `-shortest` (which,
  against a packet-aligned audio cut, dropped the last 3 frames: 120 -> 117). Untrimmed
  renders keep the single command. Real render, b1 frames 200-320: video and audio both
  start at 0, 120/120 frames, audio within 17.5 ms (one AAC packet) of an exact source
  cut. `tests/test_restore_audio_trim_offset.py` fails on the old code (1.58 s late on a
  synthetic mid-GOP trim) and on a `-shortest` mux (99/100 frames). Present since at least
  2026-09-20; renders starting at frame 0 were never affected.
- **React UI: binary frame socket, off-thread canvas player, telemetry out of React state,
  hardened output player with an original-vs-result compare.** Asked for: binary WebSocket
  frame streaming into a WebGL/OffscreenCanvas `<FastCanvasPlayer />`, high-frequency
  telemetry kept out of React at <= 10 Hz, proper HTTP 206 with cache busting, and a WebGL
  split / side-by-side compare. Audited first: the scrub path was already binary
  (length-prefixed JPEG chunks, worker decode, an uncontrolled canvas) and telemetry
  already came over `/ws/telemetry`. What was actually wrong, and what changed:
  - **Timeline playback committed the whole Face Swap panel on every played frame**
    (`setBufferedSrc(blobUrl)` + `setFrame`), and kept firing random-access still requests
    at the single decoder its own stream was reading sequentially. Now frames go as JPEG
    bytes to a `<FastCanvasPlayer>` over the stage (worker `createImageBitmap` -> WebGL
    double-buffered textures, drawn on the display's clock); the playhead is written at
    <= 10 Hz; still requests stop while playing.
  - **New `/ws/frames`** (`app/routes_frames.py`): 20-byte little-endian header + JPEG.
    LIVE pushes each newly published render frame (and counts as a viewer, like a
    `/api/live_frame` poll); PLAY streams target frames under client-granted CREDIT, so a
    slow or hidden tab stops the decode. An accelerator only: the HTTP paths are unchanged
    and used whenever the socket is down. Does NOT raise the live preview's publish rate
    (`ROOP_LIVE_PREVIEW_MS` is still the knob; the render is GPU-bound).
  - **Found while measuring, pre-existing:** the loop-wrap prefetch scanned `[start,
    start + overflow]` even at overflow 0, so whenever the look-ahead was full it asked
    for the clip's first frame again (a decoder seek back to the start, evicted next
    tick). On the socket path: 164 streams / 1,715 frames for 290 played -> 1 stream /
    408 frames after the fix (`faceswap/playbackWindow.js`).
  - **Telemetry frames no longer re-render App and the mounted tab** (4 Hz for a whole
    render). Fast fields go to a Zustand store (`store/telemetryStore.js`); readouts
    subscribe through `<LiveText>` / `<LiveBar>` / `<LiveValue>` (direct DOM or leaf
    re-render, <= 10 Hz). `setProgress` runs only on structural edges. Processing's live
    readouts are leaf components, so its 250-line terminal re-renders on the poll only.
    The frame now also carries `fps_now` / `frame_ms` (3 s window) — end-to-end time per
    frame, not a model's inference latency.
  - **HTTP 206 was wrong for three requests a player sends**: `bytes=-N` (suffix) was
    served as the first N+1 bytes, a start past EOF got 206 instead of 416, an inverted
    range was "repaired". Now RFC 9110 (`routes_output.parse_byte_range`), plus ETag /
    Last-Modified / If-Range / 304, `Cache-Control: no-cache`, and no hand-written `*`
    CORS header. Output URLs are versioned by file identity (`?v=<size>-<mtime_ns>`)
    instead of `Date.now()`, which re-downloaded a finished render on every remount.
  - **Compare view**: `/api/output/source` serves the one target the latest output came
    from (the server records it; no path parameter). `OutputVideoPlayer` offers Result /
    Split / Side by side, drawn by `player/VideoCompareStage.jsx` (both videos as WebGL
    textures in one draw; the original follows the output's clock with rate-nudge sync,
    offset by `start_frame / fps` like the audio).

  Real-browser A/B, `app/tests/frame_transport_ab.py` (one backend, both clients via
  `vite preview`, headless Chromium with GPU, A/B/B/A, `b1.mp4` 720p 23.976 fps, 10 s of
  timeline playback on the RTX 4070 host):

  | client | React commits/s | main-thread script ms/s | main-thread task ms/s | playhead fps |
  |---|---:|---:|---:|---:|
  | before | 55.5, 50.8 | 183.9, 162.0 | 523.4, 477.4 | 24.00, 21.77 |
  | after | 16.4, 16.5 | 45.5, 54.5 | 226.6, 250.6 | 23.96, 23.96 |

  No long tasks in any arm at 720p. The decode and draw moved to a worker, so its cost is
  not in the main-thread columns by design. Not measured: 4K targets, the 3060.

- **Faces in contact were swapped with the NEIGHBOUR's geometry after autorotate. Fixed.**
  `process_face`'s autorotate re-detects in a cut padded 45% each side, took the
  LEFTMOST detection, and since 09-23 `_unrotate_face_to_parent` writes that detection's
  kps, bbox, landmarks and embedding over the target. With two faces in contact the cut
  holds both, so on `d2.mp4` the upside-down woman was aligned, swapped and
  enhancer-stabilized from keypoints on the upright woman's face: her own swap came out
  pale and smeared with a ghost ring, and the enhancer stabilizer (matching by centroid)
  blended her crop into the upright woman's track, painting a pale hard-edged patch across
  that cheek. The swap audit read 100% swapped / 0 wrong faceset throughout -- it counts
  intent, not where a face was pasted. Now `_match_rotated_face` picks the detection whose
  frame-space box overlaps the target's (IoU >= 0.3) or declines the rotation.
  Measured with the new `tests/diag_landmark_jitter.py` (independent SCRFD on source and
  render; `rel` = wobble of the pasted face against the head, % interocular, median/p95):

  | clip, arm | before | after |
  |---|---|---|
  | d2, enhancer stabilizer only | 5.30 / 29.9 | 3.45 / 21.1 |
  | d2, config.yaml (all stabilizers) | 5.01 / 30.6 | 3.36 / 21.3 |
  | d2, no stabilization | 3.44 / 21.0 | 3.50 / 19.6 |
  | d1, config.yaml | 16.71 / 58.7 | 13.90 / 46.4 |

  Swap coverage identical (d2 432/432 both people; d1 418/418 and 414/418). The request
  that led here -- detect every K frames with Kalman/LK tracking, EMA/One-Euro landmarks,
  CUDA warp/blend -- was declined: all exist or were measured and reverted (`bd71e12`
  cadence-2 flicker, `ROOP_TEMPORAL_STEP` 6x error on turned heads, `bf96c1f` GPU pre/post
  1.1-40x slower).
- **Unstabilized renders ran ALL inference on one thread — 3.4x slower. Fixed.**
  `ba607a3` (2026-09-02, "Optimize video render pipeline", never A/B'd) turned the unified
  scheduler's frame pipeline into a single CUDA owner: `run()` does `del workers`, and
  choosing that path also skipped building the cross-frame swap batcher. It was the
  default for every render without stabilization. RTX 4070, `d4.mp4` two-person, 600
  frames, live config (TensorRT, hyperswap, Restore Ultra, XSeg), `--threads 20`, ABBA:

  | arm | fps | path | not swapped |
  |---|---:|---|---|
  | stream (old default) | 3.88, 3.88 | one owner, `batch_mode=sequential` | 55 / 732 |
  | threaded | 13.38, 13.03 | 20 workers, xframe avg batch 1.58-1.68, max 8 | 55 / 732 |
  | after the fix, no env | 13.16 | threaded + batcher | 55 / 732 |

  0 wrong-faceset swaps in every arm. `frame_pipeline_allowed` now returns true only for
  streaming stabilization (`ROOP_STAB_STREAMING=1`, whose FIFO needs one in-order owner);
  `ROOP_SCHEDULER_FRAME_PIPELINE=1` still forces it. **Stabilized renders — the shipped
  config — were never affected**: parallel stabilization is chosen ahead of the stream
  (old 9.27 vs fixed 9.08 fps, same path). Every harness run with stabilization at its
  default OFF (e.g. `two_face_video.py`) since 09-02 measured the one-thread path: those
  absolute fps are ~3.4x low. The cross-frame batcher itself is healthy on hyperswap under
  TensorRT (no batch-2 fallback).
- **Requested and declined: blanket IOBinding-to-torch, FP16 everywhere, fixed batch
  profiles, a batch aggregator, an EP fallback chain.** All exist or were measured and
  rejected: GPU crop/warp 1.1-40x slower (`bf96c1f`); FP16 breaks inswapper, GFPGAN and
  GPEN and costs identity 0.352 -> 0.407; the live swapper and every restorer are static
  graphs (`trt_shape_profile.py`); `swap_batcher.py`; `core.py`'s TRT -> CUDA
  (`kSameAsRequested`, `ROOP_CUDA_MEM_LIMIT`) -> CPU chain.

- **VFR renders lost audio again — `detect_fps` reverted to the AVERAGE rate.** `52acd20`
  (2026-09-20) switched `detect_fps` to ffprobe's `r_frame_rate`. That is the timebase
  rate, not the playback rate: on a VFR fixture (4 s @30 + 4 s @15; `r=30/1`,
  `avg=2700/119`, 180 frames over 7.933 s) the render came out **6.000 s** and
  `restore_audio`'s `-shortest` cut **2 s of audio**. After the fix: 7.933 s video,
  7.924 s audio (the same 9 ms `-shortest` residual measured 2026-09-04). The rate now
  comes from `utilities._probe_frame_rate`: `avg_frame_rate`, unless it agrees with
  `r_frame_rate` to 1e-4, in which case the exact nominal (e.g. 24000/1001) is kept.
- **Writers stamp an exact rational `-r`.** `ffmpeg_path.frame_rate_arg` turns the
  pipeline's 6-decimal float back into `24000/1001` / `2700/119` for both
  `FFMPEG_VideoWriter` and `NVHardwareVideoWriter`.
- **Closing the NVDEC reader early no longer stalls 10 s.** Closing
  `NVHardwareVideoReader.read_frames()` before EOF ran `communicate()`, which drained
  the rest of the decode for up to 10 s, then killed FFmpeg and logged it as a decode
  failure (measured 10.10 s → 0.03 s). Production's `read()` + `release()` path was not
  affected.
- **Requested and declined: a PyAV / 3-process shared-memory I/O rewrite.** Measured on
  the 4070, `b1.mp4` 720p, 600 frames: production NVDEC pipe **440–461 fps**, NVENC
  writer **506–748 fps**, against a render of ~8–13 fps. Both codecs already run in
  FFmpeg child processes, off the GIL. The ceiling for any I/O rewrite is ~2–5% of wall
  clock, and the pipeline is GPU-bound. Audio in the encode pass was also not done: the
  default writer is the resumable `SegmentedVideoWriter`, whose parts are concatenated
  before the audio remux, and a per-segment AAC stream would gap at every join.

## 2026-09-22

- **Flicker while another, un-swapped face is in contact — measured, and the obvious
  fix REJECTED.** Second report the same day. The refusals here are enormous: on the
  reported clip's densest contact stretch, **1173 of 1485 detected faces (79%) went
  un-swapped**, 886 of them as `refused: crop shared with the face beside it` — and 406
  of those sat on a track the pre-pass had bound to the selected person. The cause is
  in `roop/face_contact.py`'s own table: when two faces touch the neighbour is *inside*
  this face's aligned recognition crop, and the distance to the person climbs 0.05 →
  0.63 on the *same* person as coverage grows. The gate stops measuring identity and
  starts measuring how close the other head is.

  A third claim tier was built — a face too contaminated to measure is claimed by the
  person its track is bound to — and measured three ways with
  `tests/diag_contact_identity.py`, which asks the output "is this face the source
  now?" and the plate "was this face the selected person?" (the only non-circular form
  of the question, since the target-side crop in these frames is the contaminated one):

  | variant | faces painted | of those, the WRONG person |
  |---|---|---|
  | track binding alone | 21 | **14** |
  | + distance cap 0.85, no gap-filled faces | 8 | 2 |
  | + claimed closest-first | 8 | 2 |

  Rejected and removed. During the contact the detector loses the occluded face, the
  neighbour's detection is associated to the bound track on position alone, and every
  rule that trusts the track paints through that error; a contaminated reading of the
  *wrong* face drags toward the target (0.98 on the plate, under 0.85 inside the
  pipeline, same face), so no absolute cap separates them, and on the failing frames
  the neighbour is the only candidate, so the relative comparison has nothing to
  compare against. Un-swapped is a bad frame; wrong-person is a worse one. The fix
  belongs in track association and detection recall, and the audit now counts the
  population it would have to reach (`of those, on an UNBOUND track that still looks
  like this person`: 178 and 274 in two of the four windows).

  Kept from the attempt: the instrumentation that settled it, and three diagnostics —
  `tests/diag_contact_scan.py` (where in a clip do faces actually share a crop; four
  windows picked from overlapping track *spans* contained none), `diag_contact_identity.py`
  and `diag_contact_panel.py`.

- **A pixel diff between two renders cannot attribute a change to a face.** The
  stabilizers carry state across frames, so an arm that swaps a face the other refused
  diverges on later frames and on pixels around the face — which made the change look,
  for an hour, as though it had been painted onto the neighbour. The same comparison
  with `--no-stabilize` put it squarely on the intended face. The null control is
  bit-identical for this config, so the difference was real; only its
  *location* was not readable that way.

- **The swapped face blinked on and off (Selected-person mode).** Reported on a
  246k-frame clip, one selected person, one faceset. The run's SWAP AUDIT said 24.4% of
  detected faces were never swapped, with "refused: over the identity threshold" the
  largest bucket — but nothing recorded *which* faces those were, so the bucket mixed the
  bystanders the run is right to pass over with the target herself on a hard frame.
  Instrumented first (the default path never fed the refusal-distance curve; only the
  identity-lock fallback did), then measured on the reported clip, 800-frame windows,
  the user's own config:

  | window | faces | refused over the gate | of those, on a track already bound to that person |
  |---|---|---|---|
  | 60000 | 996 | 190 | **178 (93.7%)**, median 1.04x the gate |
  | 140000 | 265 | 46 | 7 |
  | 180000 | 844 | 476 | 0 |
  | 220000 | 1376 | 779 | 0 |

  So in the flicker windows the refused faces *were* the selected person — swapped on the
  frames either side, refused on this one because a single frame's embedding is a noisy
  reading of a person. **Fix:** a face whose track the whole-clip pre-pass already bound
  to this person is held through such a frame (`roop/selected_routing.py`, second claim
  tier). The per-frame gate still decides who is who and always claims first; a hold needs
  the binding, a looser second gate (`ROOP_SELECTED_HOLD`, 0.95), and no other selected
  person fitting better. Window 60000: un-swapped faces **283 → 131 of 996 (28.4% → 13.2%)**.
  Windows 180000/220000, where the refusals are other people: **byte-identical**, zero held.
  The pre-pass already computed this binding on every run with `temporal_detection` on and
  threw it away unless "Lock face identities" was also on.

- **Three defects in the audit that hid the above.** A sub-count printed under whichever
  unrelated bucket its own number sorted next to ("of those, partly behind an object
  (masked, still swapped)" filed under a *refusal* line); a swap undone by the outcome
  check was still counted as a swap, so a window with 283 untouched faces reported 190;
  and more swaps than faces seen produced a negative total, which printed as no defect at
  all. All three now say what they mean, the last one loudly.

- **Consent and labelling.** Audit found no content or consent safeguard anywhere in
  `app/roop`. Added: a first-run screen that requires accepting NOTICE.md's intended-use
  terms (`/api/terms`, enforced on `/api/swap`; re-asked when the text changes); a
  default-on metadata tag on every output (`roop/synthetic_label.py`: MP4/MKV/WebM
  `comment` + `synthetic_media=true`, PNG text chunk, JPEG EXIF + comment, all without
  re-encoding); an opt-in visible watermark. Settings `synthetic_label`,
  `synthetic_watermark`, `synthetic_watermark_text`, `synthetic_label_text`,
  `intended_use_acknowledged`. Swap output pixels are unchanged unless the watermark
  is turned on.
- **Settings have one source.** `app/settings.py` now carries `UI_SETTINGS` (panel
  label/section) and `ENV_SETTINGS` (the `ROOP_*` mapping, applied by
  `settings.apply_env`); `tools/gen_settings.py` renders `app/settings.schema.json` and
  `react-ui/src/components/settingsCatalog.js` from it, and a test plus a CI step fail
  while they are stale. No setting's name or default changed; `apply_env` is proven
  equal to the old `run.py` block on every value shape. Found on the way:
  `perf_stab_chunk_mb` / `perf_stab_streaming` are read from `config.yaml` but have no
  `Settings` attribute or UI (schema marks them `config_only`), and the comparison bench
  had never exported the recognizer/priority flags (it now uses the shared mapping).
- **Repository restructure.** `AGENTS.md` is the single agent rule file; the other rule
  files point to it. README reorganised into what-it-is / install / run / config /
  troubleshooting / licence; benchmark phases, the update contract detail and incident
  notes moved into `docs/`. One-off root scripts moved: `scripts/clean.js`,
  `scripts/cleanup.py`, `scripts/fix_tensorrt.js`, `tools/diagnose_trt.py`,
  `tools/repair_venv_paths.py`, `tools/phase14-after-render.ps1`. Behaviour unchanged.
- **Reproducible installs, CI, pytest.** npm is the one JavaScript package manager (root
  `bun.lock` replaced by `package-lock.json`); Node `^20.19 || >=22.12` declared in both
  `package.json` files; `.github/workflows/ci.yml` runs react-ui lint/build, the root
  typecheck and the light-profile Python tests on Ubuntu and Windows; pytest is the test
  runner (`unittest discover` dropped the pytest-style tests and never ran `tests/`).
- **Filesystem boundary.** Every path, filename and upload endpoint goes through
  `app/safe_paths.py`: realpath + allowed roots, sanitized upload names, extension and
  magic-byte checks, size and count caps, streamed writes. `/api/file` and `/outputs/`
  previously followed symlinks out of the output folder; `/api/reveal` opened any path;
  uploads were unchecked; `/api/target/add_path` accepted UNC paths.
- **Network exposure.** The API answered any web page (CORS `*` with credentials) and
  `/ws/telemetry` had no Origin check; `server_share` never reached the API. Now: loopback
  by default, foreign Origins refused (403) on `/api` and `/ws`, and share mode binds all
  interfaces behind a per-launch bearer token shown in the console and the Pinokio
  sidebar (`app/api_access.py`).
- **Mock API server** moved to `react-ui/mock-server/`, announces itself
  (`X-Mock-Server: true`, `mock: true` in `/api/meta`), reads `PORT`.
- **Personal data scrubbed** from the tree (faceset names, machine paths, a committed face
  image); the README no longer claims the repository is private (it is public).

## 2026-09-21

- Two faces on the hardest two-person clip: the upright person was refused by a crop that
  did not exist (an inverted neighbour's reflected fit) and then by a junction phantom the
  bridge rule missed by 3%; 21.5% -> 0% frames not swapped.
- The backend could go deaf and stay "running": a Chromium pre-connect aborted mid-accept
  closed the listening socket on Windows (`roop/win_asyncio_compat.py` re-arms it).
- Two-people mapping: three traps in "Selected people" mode fixed; ROI cadence default back
  to 1 (2 interpolated every other swap).

## 2026-09-02 -- React UI 2.0 removed

React UI 2.0 was an experimental parallel client; it was removed on
2026-09-02 after every capability it uniquely had was migrated here and
verified. The audit and the per-feature decisions are in
`docs/development/UI_V1_V2_MIGRATION_AUDIT.md`.

## One server, one port -- why the backend serves the UI

This matters for portability. Serving the UI from `vite preview` put a Node
toolchain on the runtime path, and `vite preview` refuses to start when
`react-ui/dist` is missing. Because `dist/` is generated and never committed,
any build failure on another machine -- a Node older than Vite 8's
`^20.19 || >=22.12` requirement, a missing per-platform rolldown binary, a cold
npm cache -- took down the *server*, not just the build, and the app opened on a
Vite error instead of the UI. Node is now needed only to produce `dist/`, and a
failed build stops the launch with the real error rather than a broken page.

## 2026-08-23 -- the environment was a junction into another working copy

They must be **real local directories**, not links to another folder. Until
2026-08-23 all three were NTFS junctions into a different working copy on the
same machine, which meant deleting that folder would have taken this application
down and the project could not have been moved or handed to anyone.
`app/tests/test_standalone_install.py` fails if that ever comes back.

## Install-time repairs that replaced destructive resets

Starting or updating an older environment also repairs NumPy in place when a
previous install left NumPy 2.x behind. This avoids a destructive reset. If the
launcher reports a fatal dependency error before opening the UI, rerun **Update**
or **Install** so the environment repair can finish.

On a fresh install, `app/config.yaml` is optional: startup uses defaults until
you save settings. Older revisions incorrectly logged its absence as
`[Fallback] run.py:28 ... [Errno 2]`, which Pinokio could interpret as a startup
failure and terminate the shell. The startup fix treats only the missing file
as normal; unreadable or malformed settings are still reported. Pull the latest
code on the affected, stopped installation and click **Start**. Do not copy
another GPU's config, reset the app, or change a running installation's packages
to fix this particular message.
