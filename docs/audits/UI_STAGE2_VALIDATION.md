# UI overhaul Stage 2 validation — 2026-11-18

Scope: the React shell and Face Swap docking layout. Starting revision:
`9612427` (Stage 1). No Python inference code, saved look configuration,
provider/precision policy, launcher or venv was changed.

## Verified

| Check | Result |
|---|---|
| Production Vite build | PASS; existing large-entry-chunk warning remains |
| oxlint | No errors; existing warnings remain |
| Reachable modules | PASS, 269/269 reachable |
| API callsite/deadline checks | PASS, 208 callsites |
| Offline bundle scan and scanner tests | PASS |
| `npm run test:render-checks` | PASS, including workspace state and Studio integration |
| Focused browser suite: workspace-shell, nav-visibility, faceswap-deck, a11y | 45 assertions/tests passed |
| Additional real-browser magnifier and body type-scale checks | 2 tests passed |
| All nine routes at 1920×1080, 3840×2160, 3440×1440, 5120×1440 | PASS, measured shell/header/main rectangles and document dimensions |
| Raster frame path | PASS, production decoder/canvas painted RGBA `[22,125,183,255]`; source image decoded |
| Checkpoint selector | PASS, saved recipe restored and dirty indicator changed with a real UI edit |
| Divider interactions | PASS, pointer drag, keyboard sizing, double-click reset, reload persistence and preset reset |
| Rail, route history and 125% UI zoom | PASS |

The browser fixture also emits an unhandled `ECONNRESET` from its WebSocket
socket during shutdown, producing exit 1 even on runs whose test assertions all
pass. Therefore the browser commands and `npm run check` are **not claimed
fully green**. The real application backend was not restarted to validate UI
changes during the user's active render.

## Broader checks and baseline failures

The final light-profile run used separate mock ports from the UI run:

```powershell
$env:ROOP_TEST_LIGHT = '1'
$env:E2E_PREVIEW_PORT = '4330'
$env:E2E_MOCK_PORT = '4331'
app/env/Scripts/python.exe -m pytest -m 'not gpu'
```

Result: **2,828 passed, 6 failed, 1,054 skipped, 206 deselected, 4 xfailed**.
Each of the six failures also failed in an isolated checkout of starting
revision `9612427` (same six files: 107 passed, 6 failed, 11 skipped):

- `test_detect_overrides`: expected five hybrid dispatch sites, found six.
- `test_exception_visibility`: `api.py` has nine silent handlers against an eight-handler allowance.
- `test_no_unmanaged_gpu_sessions`: `roop/ultramax_compile.py` contains an unaccounted session site.
- `test_phase_harness_contracts`: the unittest shim check rejects two pytest-style modules.
- `test_terminal_dashboard`: expects the retired `SWAP PROFILE` console heading.
- `test_ui_zoom_pan`: its structural lens assertion expects division inline even though coordinates are precomputed.

The broad `npm run check` browser sweep was stopped after 148 test decisions:
92 passed, 17 failed, 39 skipped. It did not finish the 361-test inventory and
is not an acceptance pass. During that sweep, newly exposed magnifier-border
alignment and preflight body-text sizing were corrected and their real-browser
checks passed separately. The mask brush (3 failures), face-exclusion controls
(2 failures), and playback retry counts (4 failures) were reproduced unchanged
in the starting checkout. Other failures from that interrupted sweep remain
unverified; this change does not close them or loosen any allowlist.

## Scope limits and repository checklist

- The user authorized commit and push with the recorded validation results and
  GPU-backed validation still pending.
- GPU-backed full pytest, render/fidelity/performance benchmarks and physical
  RTX 3060 validation were not run: a production render was already active,
  and the repository permits one render at a time. There is no throughput or
  fidelity claim for this CSS/React change.
- The applicable AGENTS sections were snapshotted: session records, companion
  policies, React-only UI work, setting ownership, tests, one render at a time,
  and launcher preflight/exit workflow.
- `PINOKIO_HOME` was resolved from the user's Pinokio config. Reference launcher:
  `<PINOKIO_HOME>/prototype/system/examples/mochi/start.js`; `PINOKIO.md` was
  consulted. No launcher API, capture regex or `local.set` block was edited,
  so the launcher-specific mutation checks are not applicable.
- All Face Swap controls remain: a component-count comparison against the
  starting source removes only the three old local resize dividers and adds
  the shared docking layout. Existing controls and callbacks occupy its slots.
- Generated test artifacts were moved out of the working tree. The venv,
  running backend, model caches and per-machine settings were left untouched.

Implementation and use: [UI_OVERHAUL_STAGE2.md](../development/UI_OVERHAUL_STAGE2.md).
