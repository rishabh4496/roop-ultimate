# UI overhaul Stage 2: viewport shell and docking

The React shell uses a 38px top row across the entire window, a 48px expandable
navigation rail, and a bounded route surface. `#root` and the studio grid occupy
the viewport with document scrolling disabled. Existing UI zoom is compensated
so its scaled layout still reaches the window edges. All nine hash routes retain
direct loading, reload, and browser Back/Forward navigation.

`react-ui/src/components/shell/WorkspaceSplitLayout.jsx` owns the Face Swap docking
grid. Slots hold assets/facesets, the viewport, timeline, parameters/diagnostics,
and queue. Controls and their callbacks remain in FaceSwap; no render, matching,
provider, precision or per-machine look configuration changed.

`PanelResizeDivider` renders a one-pixel separator with a six-pixel pointer hit
area. It captures pointers (including touch/pen), compensates drag distance for
UI zoom, glows cyan on drag, supports arrow-key sizing and Home/Enter reset, and
resets on double-click. Sizes are bounded and saved through the existing
`roop_workspace` store. On smaller windows the side panes shrink to preserve the
center; the queue moves into the inspector when four columns will not fit.

The three primary presets are:

| Preset | Left | Center | Right | Timeline | Queue |
|---|---:|---|---:|---:|---|
| Standard NLE | 260px | Fluid preview | 340px | 260px | In inspector |
| Cinema Preview | 32px icon rail | Fluid preview | 32px icon rail | 260px | In expanded inspector |
| Ultra-Wide Dual Stage | 260px | Source A and live preview B | 340px | 260px | Separate 280px pane when space permits |

A fresh display wider than 2200px chooses Dual Stage. Saved choices take priority.
The layout buttons preserve the current route. Additional and user-saved layouts
remain available from the layout menu. `Cinema Preview` keeps the timeline.

The compact header uses the existing session snapshots for checkpoint selection.
Its dirty dot compares the current recipe with the selected snapshot (or the
initial recipe for an unsnapshotted working session). Selected checkpoint and
pane widths survive reload. This is a recipe snapshot selector; render checkpoint
validation/resume remains in the existing Persistent Projects panel.

The hardware strip reads GPU name, VRAM and live FPS/latency through the shared
telemetry store. The provider badge uses the effective provider reported by the
runtime; `UNKNOWN` means the backend has not reported one. No GPU identity or
memory capacity is hardcoded. The disclosure retains the detailed hardware
dialog, history, quality profiles, snapshots, floating preview, command palette,
and UI zoom. The action dock now has its own bottom row instead of covering the
timeline. Scrollbars inside studio panes use a 4px WebKit track and Firefox's
native thin track.

## Validation

`react-ui/e2e/workspace-shell.spec.js` drives all nine routes at 1920×1080,
3840×2160, 3440×1440 and 5120×1440. It asserts actual shell/header/main rectangles,
document dimensions, hash reload, preset geometry, pointer and keyboard resizing,
double-click reset, sidebar collapse, persisted widths and wide-screen startup.
Its raster test supplies a PNG through the real target-frame API URL, exercises
the production bitmap decoder and canvas, reads the painted RGBA pixel, and
checks independent inspector scrolling. Shared SVG thumbnails alone cannot
prove that bitmap decoding and canvas painting ran.

The existing deck and navigation tests continue to assert visible run controls,
viewport/timeline containment, absence of nested stage scrolling, proportional
wheel zoom, header fit, navigation, stored preferences, the hardware dialog, and
accessibility. Their preset names, initial collapsed rail, tools disclosure and
timeline-divider selector now reflect Stage 2.

Final commands and results are recorded in the dated CHANGELOG entry. Tests use
the mock API and headless Chromium; they are UI evidence, not a GPU throughput,
fidelity or physical multi-monitor validation claim.
