# React UI 2.0: feature audit of 1.0 and the new visual layer

2026-11-28. React UI 2.0 lives in `react-ui-v2/` and is served by the same backend at `/v2/`. It is
**1.0's client copied file for file, with a new visual layer on top.** React UI 1.0 (`react-ui/`)
was not changed and is still what `start.js` launches.

## Why this is not the first React UI 2.0 again

The first React UI 2.0 was a rewrite. It passed every acceptance row that was run against it but
could not add a source, capture a face or scrub a timeline, and its "complete endpoint parity"
turned out to be adapter wrappers that no screen imported. It was removed on 2026-09-02
(`docs/development/UI_V1_V2_MIGRATION_AUDIT.md`). Three rules keep this one from going the same way:

1. **Parity comes from copying, not rewriting.** All 381 git-tracked files of `react-ui/` were
   copied. Twelve files are *owned* by 2.0, meaning it edits them to mount the visual layer:

   | Owned file | What 2.0 changes |
   |---|---|
   | `src/App.jsx` | Mounts the horizon, grain, boot sequence and terminal dock; uses the transition styles; sends toasts and navigation to the terminal; adds FX entries to the command palette |
   | `src/main.jsx` | Imports the bundled fonts and `v2.css`; paints the Void theme from the first frame |
   | `src/themes.js` | "Default" becomes Void; 1.0's default is renamed "Obsidian"; adds four Void variants |
   | `src/components/shell/WorkstationHeader.jsx` | FX button, path crumb, decrypting title |
   | `src/components/shell/WorkstationRail.jsx` | Gliding active indicator, glowing brand mark, "2.0" tag |
   | `src/components/Gallery.jsx`, `ThemeGallery.jsx` | `v2-tilt` on the cards |
   | `index.html`, `package.json`, `package-lock.json`, `vite.config.js`, `README.md` | Title, fonts, `base: '/v2/'`, the `test:v2fx` check |

   Every other file is byte-identical to 1.0.
2. **Drift fails a test.** `scripts/sync_react_ui_v2.py --check` compares the two trees.
   `app/tests/test_default_client_capability.py::test_2_0_is_1_0_plus_a_visual_layer` runs it.
   - If a 1.0 file changes and the change is not copied into 2.0, the test fails.
   - If 1.0 changes a file that 2.0 owns, that file is flagged until it is merged by hand.
     `react-ui-v2/V1_BASE.json` records the hash of the 1.0 version each owned file was last
     merged from.
   - Running `python scripts/sync_react_ui_v2.py` copies 1.0's changes across and lists the owned
     files that need a merge.
3. **The capability floor applies to 2.0 too.**
   `test_2_0_can_take_in_media_and_pick_a_face` checks that 2.0 references the four routes no job
   can be created without. `test_1_0_is_still_what_start_js_promotes` checks that 1.0 is still
   the default client.

Everything in `src/v2fx/` is presentation only. It does not change an engine setting, it does not
change what any control does, and the terminal's commands are limited to navigation and
appearance. The FX preferences are stored per browser (`localStorage` key `roop_v2_fx`), so they
do not affect 1.0.

## Feature audit: what 1.0 has (and therefore 2.0 has)

Counted from the 1.0 source on 2026-11-28.

| Area | Features |
|---|---|
| Shell | Rail with 9 routes; Processing appears only while a job exists and carries a badge. Collapsible to icons; icon-only under 768 px. Hash routing, so a reload or Back/Forward returns to the same screen. |
| Header | Layout switcher (Standard / Max Canvas / Inspection Suite and saved layouts), undo-history popover, telemetry pill (GPU / VRAM / CPU, diagnostics), quality profiles, workspace snapshots, floating preview (PiP), command palette, UI zoom. The run chip appears while a job runs. |
| Home | Quick launch (source and target drop zones), counters (runs, last speed, outputs, frames), recent runs, latest outputs, resumable checkpoints. |
| Face Swap | 120 component files: recipes and custom presets, the 3-stage pipeline controls, pre-flight readiness with 1-click fixes, start/preview-render, the preview stage (fit modes, compare/wipe, zoom, brush and roto masks, face boxes, head-pose HUD), the multi-track timeline with keyframes and identity bindings, Face Bank and per-person overrides (roles, angle coverage), the modular pipeline flow (solo/bypass), curve editor, history A/B/C/D slots. |
| Batch Matrix | Job queue matrix, batch uploads, preset and timeline import, render slices. |
| Processing | Live progress by phase, canary precision, frame-bottleneck flamegraph, pipeline providers, pause/stop/cancel, Lite UI, desktop alerts, the processing terminal (log ring with 2,000 lines and severity filters), console mirror, throughput and GPU readouts. |
| Face Manager | Faceset library, Morph Lab, pose radar. |
| Editor (Extras) | Video frameset extractor and the media tools. |
| Outputs (Gallery) | Hover-scrub video cards, reuse as source or target, send to the reference monitor, load the settings a file was made with, multi-select delete and reveal. |
| History | Run history, re-apply settings. |
| Settings | 6 tabs: Execution Providers & TensorRT, CPU P/E-core & thread pools, Paths & Storage, Runtime Diagnostics & Benchmark Runner, Appearance & Pipeline Defaults, Keyboard Shortcuts. 85 catalogued settings, searchable from the panel and from the palette. Theme Gallery (38 themes) and Theme Studio (custom themes). 57 rebindable shortcut actions. |
| Global | Command palette (navigation, actions, canvas, layouts, profiles, provider, every setting, every theme, presets). Also: toasts, compile/canary banner, model-integrity splash, terms gate, confirm and prompt dialogs, reference monitor (`#monitor`), per-tab error boundary, reconnect banner, drag-and-drop anywhere, undo/redo of every edit, the telemetry WebSocket and the pipeline-status socket. |
| API | 137 distinct `/api/...` routes referenced by the client. |

The 2.0 copy runs 1.0's whole render-check suite (35 scripts, plus 2.0's own `test:v2fx`) and passes. The API
call-site lint (191 sites) passes, and so does the unreachable-module lint (263 of 263 modules reachable
from `main.jsx`).

## The visual layer (`react-ui-v2/src/v2fx/`)

| Effect | File | What it does | Cost and guards |
|---|---|---|---|
| **3D horizon** | `HorizonCanvas.jsx` | A planet limb ray-traced in one WebGL fragment shader. Ray-sphere intersection, so the latitude/longitude cage foreshortens and rotates in 3D. The atmosphere has three bands: a tight rim, a violet halo and a wide plum haze. Stars sit at three parallax depths and the dark side has sparse city lights. The camera follows the pointer, and the result is dithered with grain. Colours come from the theme (`--fx-sky/violet/plum/grid`). | Rendered at 0.35, 0.5 or 0.75 of the CSS size (24, 30 or 60 fps cap) and upscaled pixelated. **Holds one still frame while a render runs or a TensorRT build/canary is active** (the engine is GPU-bound; this is switchable). Stops while the page is hidden, draws a single frame under reduced motion, and is hidden on light themes. If WebGL is missing, 1.0's CSS background shows instead. |
| **Terminal dock** | `TerminalDock.jsx`, `termModel.js`, `termFeed.js` | `tail -f` of the engine's rolling log, merged by `seq` so each line prints once, interleaved with UI events (navigation, toasts, connection, renders, FX). Collapsed, it shows one line typing itself out plus a live ticker (phase, %, fps). Expanded, it can be resized, has filters (all/engine/ui/warn/error/perf), copy and clear, a CRT mode, and a prompt: `help`, `goto <tab>`, `tabs`, `palette`, `filter`, `fx <name> on\|off`, `transition <style>`, `clear`, with ↑/↓ history. | Ctrl+\` toggles it. It polls `/api/progress` every 4 s, only while open and idle; during a render it reads App's own poll. At most 1,000 engine lines are kept and 400 DOM rows rendered. |
| **Tab transitions** | `transitions.js` | Depth (receding Z with an X-tilt), Card flip (rotateY), Portal (circular iris from the rail), Glitch (skew and jitter slice-in), Warp (stretch along travel), None. | Every style **ends at the identity**: the transform, or the clip-path for Portal, is reset to `none` when the animation lands. This keeps `fixed` dialogs inside a tab positioned against the window. It is checked in `render-v2fx.mjs` and in the browser for all 8 routes. Changing the style does not remount the tab. |
| **Boot sequence** | `BootSequence.jsx` | A terminal that prints the real startup: bundle loaded, engine online with its version and commit, settings loaded with the key count. A step shows `[ ok ]` only when it has actually happened. | Runs once per browser session. Any key or click skips it. It gives up after 8 s, never blocks, and is skipped under reduced motion. |
| **Pointer surfaces** | `pointerFx.js` | A spotlight that follows the pointer inside every `.glass-panel`, and 3D tilt with a glare on `.v2-tilt` cards. | One rAF-throttled document listener. The spotlight is a background layer, with no pseudo-element and no forced `position`. Tilt applies a transform only while hovering. |
| **Title effects** | `ScrambleText.jsx`, `v2.css` | The page title decrypts in on navigation, with a sunray light sweep over the titles. | Screen readers get the final text. Reduced motion turns both off. |
| **FX panel** | `FxPanel.jsx`, `fxStore.js` | Header popover: every effect switchable; horizon quality; transition style with live previews; reset. Also in the command palette ("Visual effects"). | Per browser. |
| **Void design system** | `v2.css`, `themes.js` | Black void, hairline rules (8% and 14% white), a hot accent with a cool counter-accent, Geist and JetBrains Mono (bundled), sharper corners, cyan focus rings, a gradient progress bar with a glowing tip, film grain. Five Void themes: magenta (Default), Cyan, Ember, Aurora, Mono. All 38 of 1.0's themes still work and also get a matching horizon. | Fonts are bundled by Vite (offline rule). 1.0's tokens are repointed, so screens are not restyled one by one. |

### What omp.sh contributed

The site's CSS and JS were read for technique, not copied:

- **Palette.** An `#000` / `#0a0a0c` ink scale, an oklch magenta accent (70% .24 340) and an oklch cyan (81% .14 200).
- **Type.** Geist with JetBrains Mono.
- **Sunray.** The sweep keyframe: a 320% gradient clipped to text, every 9 s, with a brief glow.
- **Horizon.** A dithered planet-limb horizon, drawn there on a 2D canvas in 2×2 cells with sky, violet and plum bands at exponential falloffs. Here it is re-done as a real 3D shader with a rotating surface.
- **Marquee masks** that fade at the edges.
- **Hairline rules** and a `.terminal` mono block.

None of omp.sh's content, logos or branding is used.

### Online research and what was taken

- **React Three Fiber / drei shader backgrounds** ([Codrops: subtle shader background with R3F](https://tympanus.net/codrops/?p=81438); [Codrops: 3D product grid with R3F, 2026](https://tympanus.net/codrops/2026/02/24/from-flat-to-spatial-creating-a-3d-product-grid-with-react-three-fiber/); [pmndrs/drei](https://www.github.com/pmndrs/drei)). The technique was adopted: one full-screen fragment shader, animated per frame, kept off the main React tree. The library was not. Three.js plus R3F would add hundreds of KB for one quad and one draw call, so the shader is raw WebGL in about 300 lines.
- **View Transitions API** ([React `<ViewTransition>`](https://react.dev/reference/react/ViewTransition); [Next.js view transitions](https://nextjs.org/docs/app/guides/view-transitions)). React's component is still Canary-only on the React 19.2 this app ships, and it needs reduced motion handled by hand. 2.0 uses framer-motion, already a 1.0 dependency, with the same perspective, depth and iris vocabulary. Reduced motion is honoured globally by `MotionConfig`.
- **Terminal / CRT techniques** ([DEV: Retro CRT terminal in CSS + JS](https://dev.to/ekeijl/retro-crt-terminal-screen-in-css-js-4afh); [CSS-Tricks thread](https://css-tricks.com/?p=275830); [Carbon Five: vintage terminal effect](https://blog.carbonfive.com/vintage-terminal-effect-in-css3/)):
  - The typewriter is a `steps(N)` reveal, done with `clip-path`, so the line's layout is final from the first frame.
  - Scanlines are a 3 px repeating gradient with a rolling bar and a vignette, confined to the dock.
  - The cursor is a blinking block caret.

## How to use it

- **Pinokio sidebar:** "Open React UI 2.0", shown once `react-ui-v2/dist` exists.
- **URL:** `http://127.0.0.1:<port>/v2/`. In share mode use `/v2/?token=...`; the token cookie is shared with 1.0.
- **Effects:** the FX button in the header, or the command palette (search "visual effects").
- **Terminal:** Ctrl+\`, then `help`.

## Verified

- `app/tests/test_backend_serves_ui_v2.py` (11 tests) drives the real ASGI app. It covers:
  - `/v2/` and `/v2` redirecting;
  - hashed assets and fonts;
  - a missing asset returning 404 rather than HTML;
  - deep paths;
  - traversal;
  - `/api` not being shadowed;
  - `/` still serving 1.0.

  Passed together with `test_backend_serves_ui.py` and `test_api_access.py` (45 passed). The launcher, install, storage, client and update test files passed too (397).
- Headless Chromium against the production build of 2.0, with `/api` on the mock backend:
  - the boot sequence appears and leaves;
  - the Void theme and Geist (loaded from the bundle) are applied;
  - the horizon runs, holds a still frame while a mock render runs and resumes after it;
  - all 8 routes end their transition with `transform: none`;
  - the terminal opens with Ctrl+\`, `goto` navigates and an unknown command errors;
  - the FX panel opens;
  - the palette opens inside the viewport;
  - there is no horizontal overflow at 1280 px;
  - there were no console errors.

  Screenshots were checked by eye. They caught three defects that the assertions passed:
  - the titles were invisible: a `currentColor` sunray on transparent text;
  - the terminal let panel text bleed through it;
  - the FX popover opened under the rail.

  All three were fixed.
- `render-v2fx.mjs` (48 checks):
  - the engine-log merge (repeats print once, the oldest lines evict first, severities map to levels, filters partition correctly);
  - the UI feed;
  - transitions ending at the identity;
  - FX persistence and the fallback from a corrupt stored value.

## Not verified

- **Against the real backend.** The running backend has to restart to load the `/v2/` route. The engine half of the terminal is proven only by `render-v2fx.mjs`, because the mock sends no `log`.
- **GPU cost of the horizon.** Not measured against render fps. The still-frame-during-render rule exists so that no measurement is needed during a render, but the cost while idle or previewing is unmeasured on both GPUs.
- **RTX 3060** (6 GB, laptop): not run.
- **1.0's Playwright e2e suite** (`e2e/`) was not run against 2.0. The copy is byte-identical, but the specs point at `/`.
- **Light themes:** the horizon is hidden on them; the other effects were not screenshot-checked there.
- **Renders started from another tab or window:** the horizon freezes only when App knows a render is running. Against the mock, a render started outside this window was only seen after a visibility catch-up. That is 1.0's existing state handling, unchanged.
