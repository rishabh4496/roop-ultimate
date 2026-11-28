# React UI 2.0: a new client on 1.0's engine contract

2026-11-28. React UI 2.0 lives in `react-ui-v2/` and is served by the same backend at `/v2/`.
React UI 1.0 (`react-ui/`) is unchanged and is still what `start.js` launches.

Earlier the same day, 2.0 was a byte-for-byte copy of 1.0 with a visual layer on top (commits
`bdff209` and `7434b7a`). It was then rebuilt on request ("do not copy any visual of v1"). Every
screen, component and style in 2.0 is now new; none of 1.0's markup, classes or stylesheets is used.

## How it avoids the first 2.0's failure

The first React UI 2.0 was a rewrite that re-derived the engine contract. It was removed on
2026-09-02 (`docs/development/UI_V1_V2_MIGRATION_AUDIT.md`) because it could not create a job.
Its "complete endpoint parity" was adapter code that no screen imported. This rebuild avoids that
in three ways:

1. **The contract is imported, not re-derived.** 2.0 talks to the engine through React UI 1.0's
   **logic** modules, imported through the Vite alias `@engine` (pointing at `react-ui/src`), not
   copied. The selection payload is built by 1.0's own pure builders. That payload is where "the
   render swapped nobody" lives, so this is the part that matters most.

   | Module | Used for |
   |---|---|
   | `api.js` | HTTP client: deadlines, typed errors, uploads |
   | `useTelemetrySocket.js`, `usePipelineStatus.js`, `store/telemetryStore.js` | Live render numbers, and the TensorRT/canary status |
   | `failureLog.js` | Rate-limited failure toasts |
   | `components/faceswap/faceMapping.js`, `processingSelection.js`, `mouthCustody.js` | `face_mapping`, `selection_state`, `processing_selection`, mouth-custody fields |
   | `components/faceswap/useQueue.js` | Batch queue state, actions, and the ten-state job vocabulary |
   | `components/faceswap/pipelineRules.js`, `logRing.js` | Swapper grouping and labels; log severities |

   `vite.config.js` dedupes `react`, `react-dom` and `zustand` so the imported modules share 2.0's
   single copy of each.
2. **A test enforces this on what is actually reachable.** `app/tests/test_default_client_capability.py`
   walks the import graph from `react-ui-v2/src/main.jsx`, following the aliases, and reads routes
   only from modules reachable from there. Code that nothing imports cannot satisfy it. It checks:
   - 2.0 reaches the four job-intake routes plus the must-haves of the brief: preview, timeline frames
     and lanes, swap, pause, resume, stop, queue add and start, faceset library and load, model integrity.
   - No 1.0 `.jsx`, `.tsx` or `.css` file is reached, so no 1.0 visuals cross over.
   - 1.0 is still the client `start.js` promotes.
3. **The workflow was driven end to end** in a browser (see Verified).

## Screens

| Screen | What it does |
|---|---|
| **Studio** | Cast (source faces: drop, add, select, remove, clear, load from the library); media bin (upload or add by path, select, remove); **viewport**; **timeline**; render deck; people (capture, select, map each person to a source or "leave untouched", detection mode); pipeline (swapper, restorer, mask, detector, swap resolution, colour match, with an animated beam diagram); look (blend, identity distance, feather, sharpen, detection threshold). |
| **Batch** | The queue: run, pause, resume and stop the queue; per-job cancel, duplicate, retry and remove; drag to reorder; "Add current setup"; "Queue every target"; totals. |
| **Face Manager** | Faceset builder (images or a clip in, cut faces per frame, quality scores, prune below a threshold, remove, build `.fsz`, open `.fsz`); pose radar (yaw/pitch of every face, missing angles, dry-run then confirm auto-prune of blurry/duplicate faces); **Morph Lab** (pick identities, weight them, slerp or linear, de-identification against chosen identities with a max-similarity bound, live similarity readout, preview in the Studio, save to the library). |
| **Editor** | Transform (resolution, rotation, crop, output fps, replace the audio track and slip it ±500 ms) via `/api/extras/apply`; Enhance (every frame post-processor `/api/extras/frame_ops` lists); video → faceset (cut faces from chosen frames of a clip, build `.fsz`). |
| **History** | Every run (`/api/history`): frames, fps, duration, key models, outputs (open, show in folder, export through `/api/export/presets`), re-apply a run's settings, delete; tick two runs to see exactly which settings differ. |
| **Library** | Faceset library: search, load into the cast, rename in place, delete, import `.fsz`, save the selected cast face. |
| **Models** | Every model family from `/api/meta` as cards (swapper, restorer, two masks, detector, recognizer, restoration mode, swap resolution, colour match, SAM 2 size); execution provider and TensorRT precision; on-disk integrity of every manifest file. |
| **Outputs** | Rendered files, newest first. Videos play on hover; open or delete each; a marquee of the latest. |
| **Settings** | All 85 UI settings, rendered from `app/settings.schema.json` (generated from `app/settings.py`). Section tabs, search, and reset to default. Choices come from `/api/meta`. |

**Pause and Stop** are in the top bar on every screen while a render runs, and in the Studio's render deck.

**2.0 does not link into 1.0.** The dock's link to 1.0 and the top bar's "1.0" link were removed. A test now
scans 2.0's source and page for 1.0's look: its surface classes, its crimson `#E94560`, its typeface, and links
into a 1.0 screen.

### Precision tools (`src/studio/Tools.jsx`, `src/studio/Overlays.jsx`)
Face Swap's deep tools, rebuilt with 2.0's own UI on 1.0's pure models (`rotoModel`, `identityLock`,
`identityKeyframes`, `identityBlend`, `timeline/overrides`, `curves/curveModel` and `curvePieces`, through
`@engine`). Per-clip state is kept per browser, keyed by media id.
- **Mask brush:** paint or erase on the viewport in frame pixels. Sent as `imagemask`
  (`{"<source>": {exclude, canonical: false, ref_kps}}`), bound to the selected person's landmarks.
- **Roto pen:** exclude or include zones with feather, for the whole clip, the In/Out range or one frame.
  Sent as `roto_polygons`.
- **Identity pins:** with the Pins tab open, a face click locks that face to its source from that frame.
  Sent as `identity_pins`.
- **Identity blend:** weighted cast identities (slerp or linear) with keyframes. Synced to
  `/api/identity/blend` and sent as `identity_blend`.
- **Scene overrides and parameter curves:** previews apply them at the playhead. Start cuts the range into
  1.0's pieces and queues one job per piece (`add_batch` + `queue/start`).
- **Quad view** (original, swap, difference, mask + roto) and **Ref view** (a wipe against a finished
  render). These replace 1.0's A/B snapshot slots.
- **Timeline lanes** show scenes, curve keys and pins.

### The viewport (`src/studio/Viewport.jsx`)
- **Modes:** Swap (the `/api/preview` frame), Original (the untouched frame, for picking people) and
  Compare (original on the left, swap on the right, with a draggable wipe).
- **Faces:** every detected face gets a reticle with corners that draw themselves, and a tag
  showing the person and who they become. Click a face to make it a target person, or to select
  the person it already is.
- **Zoom and pan:** wheel or double-click to zoom up to 6×; drag to pan.
- **HUD:** frame, timecode and resolution; a status line (faces, swapped and refused counts, render
  running); a scanline and border beam while a preview renders.
- **Scrubbing** shows raw frames (a cheap JPEG GET) and asks for the swap once you let go, so
  dragging never queues swap inference.
- **During a render** the engine refuses previews with HTTP 409, and the viewport says so.

### The timeline (`src/studio/Timeline.jsx`)
- A filmstrip of real frames, with a hover thumbnail and timecode.
- A glowing playhead; drag anywhere to scrub.
- In/out handles, dragged on the strip or set with I and O, saved with `/api/target/set_frame`.
- Scene cuts and per-person presence lanes from `/api/target/timeline_tracks`.
- A ruler in timecode.
- Transport: first, previous, play over raw frames, next, last, and a frame field.
- Keys: ←/→ (with Shift for 10 frames), Home/End, Space.

## Visual system (`src/styles/app.css`, `src/fx/`)

- **Look:** black void, hairline rules, one hot accent with a cool counter-accent (five palettes:
  magenta, cyan, ember, aurora, mono), Geist and JetBrains Mono bundled for offline use.
- **3D horizon** (WebGL, `HorizonCanvas.jsx`). It holds one still frame while a render or TensorRT
  build runs, because the engine is GPU-bound.
- **Terminal dock:** the engine log plus UI events, with filters, CRT mode and a prompt. Ctrl+\`.
- **Boot sequence** showing the real startup steps.
- **Tab transitions:** depth, flip, portal, glitch and warp. Each ends at the identity, so fixed
  dialogs keep the window as their containing block.
- **FX panel:** every effect switchable, per browser.

The effect components in `src/fx/effects.jsx` were written in-house in the style of react-bits and
Magic UI: spotlight panel, border beam, shimmer button, number ticker, split text, meteors,
marquee, magnifying dock and animated pipeline beams. No code from either library is used. Two
reasons: react-bits is "MIT + Commons Clause", and this project is AGPL. Rules adopted from
ui-ux-pro-max:
- SVG icons (Lucide), never emoji.
- A pointer cursor on every clickable control.
- Visible focus rings.
- `prefers-reduced-motion` honoured everywhere.
- State never shown by colour alone.

### Research sources
[react-bits](https://github.com/DavidHDev/react-bits), [Magic UI](https://github.com/magicuidesign/magicui),
[ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill),
[dev.to: top UI component repositories](https://dev.to/dev_kiran/top-github-repositories-for-ui-components-dg4),
[React](https://github.com/react/react). omp.sh's CSS and JS were read for technique: the palette,
type and sunray keyframe, and the dithered horizon.

## Verified

- **`npm run test:render-checks` (react-ui-v2), 2 scripts:**
  - `render-studio.mjs` covers the payload builders:
    - selected-face and multi-person mapping, with automatic choices not forced to Skip;
    - the renamed keys the engine reads;
    - preview and queue bodies;
    - the readiness refusal for a render that would swap nobody.
  - `render-v2fx.mjs` covers the terminal feed, transitions and FX preferences.
- **Headless Chromium against the production build of 2.0, with `/api` on the mock backend:**
  24 of 27 checks pass. The full flow ran:
  - add a source;
  - select a target;
  - click a face (`use_face`);
  - Compare wipe;
  - scrub the timeline, then I sets the in point (`set_frame`);
  - Start (`/api/swap`), Pause, Resume and Stop, from the deck and the top bar;
  - Batch: add the current setup, run the queue, stop it;
  - Models lists 9 families;
  - Settings renders 85 controls and the search filters them;
  - the terminal opens.

  Also checked:
  - no horizontal overflow at 1280 px;
  - the screen transition ends at `transform: none`;
  - no hook errors (a single React copy).

  The three failures are mock gaps, not 2.0 defects:
  - The mock's `/api/faceset/library` returns `{libraries}`; the engine and 1.0 use `{entries}`.
  - The mock's integrity file list is empty.
  - The mock lists output files it does not serve.

  Screenshots were reviewed. They found:
  - invisible gradient headings (each letter of the split text is its own box);
  - a phone layout where the panels overlapped;
  - mock face boxes in `[x, y, w, h]` form. The engine sends insightface's `[x1, y1, x2, y2]`;
    2.0 now reads both.

  All three are fixed.
- **`app/tests/test_default_client_capability.py` (5)**, `test_backend_serves_ui_v2.py`,
  `test_backend_serves_ui.py`: 28 passed. Launcher, install, storage, client and update test files: 397 passed.

- **Live, against the real engine.** It ran at `http://127.0.0.1:42003/v2/`, served by the running backend,
  mid-render on a 54,190-frame job. A read-only tour (14 checks, clicking nothing that changes state) passed.
  The only POST was the Studio's preview, which the engine refuses during a render.
  - The render deck showed live telemetry (80 %, 43,448/54,190, 61.7 fps, ETA), with Pause and Stop.
  - It read the real 9-job queue, 34 facesets, 35 Morph Lab identities, 10 model families and 7 verified
    model files, 3 frame post-processors, 170 history runs and all settings.
  - Real data found a defect the mock could not: during the analysis stage the engine refuses frame
    decodes (503). The media thumbnails and timeline then showed broken images. They now show a
    placeholder and retry with back-off (`src/ui/FrameImg.jsx`).
- **New screens on the mock:**
  - Face Manager: builder, radar and Morph Lab render; a two-identity blend posts `morph/blend`.
  - Editor: all three tools render.
  - History: runs list, and a two-run compare shows the differing settings.
  - No link back into 1.0.

- **Actions against the real engine (idle, 2026-11-28): 16 of 16.**
  - A live swap preview (`1 face · 1 swapped` at frame 300).
  - In/Out set with the timeline's buttons.
  - Start, Pause, Resume and Stop, all sent to the engine.
  - A stopped run stays out of history; the engine records completed runs only.
  - A full 30-frame render finished `Done`, landed in history and wrote 30 frames.
  - Batch: add the current setup, then remove it.
  - The user's In/Out was restored.
  - No page errors.

  Three defects the mock could not show, all fixed:
  - **Blank app after a render.** `/api/progress.output` becomes an object once a render finishes, and the
    deck printed it raw (React error #31). It is now normalised. Each screen also sits in an error
    boundary.
  - **In point one frame late, in both clients.** The engine stores a 0-based seek plus an exclusive end;
    every client shows 1-based frames, and `set_frame` stored In raw. In 300 / Out 329 rendered 29 frames,
    from 301.
    - First fixed in 2.0's client (`d888e2d`): the same range rendered 30 frames, and output frame 0
      matched source frame 300 (mean abs diff 1.72, against 4.33 and 4.13 for its neighbours).
    - Then fixed at the API (`2617e3c`), so 1.0 is fixed too, with no change to 1.0. `set_frame` takes
      the frame on screen, and the target list reports `start_frame` 1-based. That matches the queue's
      `frame_start` and the mock. 2.0's `inOutOf()` and `engineFrameFor()` are now pass-throughs.
    - Covered by `app/tests/test_target_in_out_frames.py`. Checked on the restarted live backend: In 300 /
      Out 329 stores seek 299 and end 329 (30 frames) and reports 300-329. A render from seek 299 had
      already been measured to start on source frame 300.
  - **Idle GPU load.** Below.
- **GPU cost (RTX 4070).** Chromium ran on ANGLE/D3D11; the WebGL renderer string was checked, so this is
  not SwiftShader. `nvidia-smi` was sampled for 20 s per arm, in A/B/B/A order, with the engine idle.

  | Page state | GPU util before | after the fix |
  |---|---|---|
  | tab closed | 2.6 % | 1.9 % |
  | 2.0, horizon off | 12.1 % | 3.4 % |
  | 2.0, horizon balanced (the default) | 13.1 % | 5.4 % |
  | 2.0, pipeline beams live (during a render) | ~11.7 % | 6.3 % |

  - Bisecting by animation group: the four pipeline-beam SVG paths cost ~8 points. A `stroke-dashoffset`
    animation is repainted every frame, not composited, and it ran while idle. Idle beams now stand still,
    and live beams step at ~10 updates/s.
  - The horizon itself costs +0.6, +1.0 and +3.1 points (low, balanced, high).
  - Power stayed at 38–39 W in every arm (noise). The horizon freezes during a render by default.

## Not verified

- **RTX 3060:** not run. It needs the laptop.
- **GPU cost:** measured with a scratch script (`gpu_measure`), not a committed test.

## Browser tests

- **`react-ui-v2/e2e/`, `npm run test:e2e`:** 26 tests against the built app and 1.0's mock server,
  all passing. They replace the scratch `flow` script and cover:
  - the shell: every screen by dock and by hash, transitions that end, no link into 1.0, widths
    of 1280 and 420 px with no horizontal scroll, and the terminal;
  - the studio, the six precision tools (the next `/api/preview` must carry them), and
    Start/Pause/Resume/Stop;
  - a render cut into three queued pieces;
  - every other screen;
  - regressions: a `progress.output` object, and In/Out sending the frame on screen.
  The 3 mock gaps no longer fail: `e2e/fixtures.js` answers those three routes with the engine's
  shape.
- **`npm run test:e2e:live`:** the scratch `live_actions` script as a committed, opt-in test, updated
  to the 1-based API. It passed on the live engine: In 1200 / Out 1229 rendered 30 frames (History
  and an OpenCV decode agree), and the output's first frame matched source frame 1200 (diff 1.63,
  against 5.91 and 7.19 for frames 1199 and 1201).
