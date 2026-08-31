# Phase 8 — Target Expression Preservation

Date: 2026-08-31

## Status

Implemented as an opt-in, model-free temporal expression layer. Real-video
quality validation remains open because this checkout contains no supplied
expression clips or paired original/output renders. The implementation does
not change existing defaults or `.fsz` formats.

Enable it for a deliberate run with:

```powershell
$env:ROOP_TEMPORAL_EXPRESSION = "1"
```

Optional controls are `ROOP_TEMPORAL_EXPRESSION_ALPHA` (default `0.28`),
`ROOP_TEMPORAL_EXPRESSION_MOTION_ALPHA` (default `0.82`),
`ROOP_TEMPORAL_EXPRESSION_CLOSED_RATIO` (default `0.48`),
`ROOP_TEMPORAL_EXPRESSION_OPEN_RATIO` (default `0.70`),
`ROOP_TEMPORAL_EXPRESSION_EVENT_STRENGTH` (default `0.86`), and
`ROOP_TEMPORAL_EXPRESSION_CACHE_SIZE` (default `256`). These are environment
overrides; no saved user configuration is rewritten.

## Design

`app/roop/temporal_expression.py` measures the target's existing 106-point
landmarks. It keeps one bounded state record per track with:

- left/right eye openness and independent blink states;
- combined blink state (`open`, `closing`, `closed`, `opening`, `wink_left`, or
  `wink_right`);
- mouth openness and mouth aspect ratio;
- signed frame-to-frame eyebrow and jaw movement;
- expression confidence and calibrated per-eye/mouth event strengths.

Eye states have open/closed hysteresis and are updated independently, so a
wink or asymmetric blink cannot be converted into a symmetric source-eye
state. Small changes use a low-pass update; large changes receive a high
adaptive alpha and remain responsive. Low-confidence observations inherit
more history without suppressing a large genuine transition.

The ordered tracker stores the measurements on each target face. In the swap
worker, Phase 8 can restore only target eye pixels for an eye event and only
the target mouth polygon for a mouth/jaw/brow event. It does not blend a whole
face or output frame. Manual `restore_original_eyes` and
`restore_original_mouth` remain authoritative; usable lip-sync remains the
mouth owner. Existing enhancers, RealityUX, RealSwap, GPEN paths, TensorRT,
precision policies, source-bank/3D paths, detector alternatives, and
hardware-specific guards are untouched.

## Real-video benchmark contract

The harness compares the original target video and the corresponding rendered
output frame-by-frame, redetecting 106 landmarks and matching the output face
to the target face by overlap. It reports for each channel:

- mean absolute error;
- Pearson correlation;
- target/output dynamic range and range ratio;
- frame-to-frame delta correlation and delta MAE;
- paired and graded frame coverage.

Run all requested cases when paired files are available:

```powershell
env/Scripts/python.exe tests/phase8_expression_bench.py --scenario all --target-video path/to/original.mp4 --output-video path/to/swapped.mp4 --json output/phase8_expression.json
```

The acceptance cases are `slow_blink`, `fast_blink`, `asymmetric_blink`,
`wink`, `half_open_eyes`, `talking`, `smiling`, `mouth_wide_open`,
`teeth_visible`, `frowning`, and `fast_transitions`. The harness does not
generate substitute frames and returns `pending` when real input is absent.
Manual review is still required for eyelid contours, teeth, lip boundaries,
fast transitions, and identity texture after regional restoration.

## Risks and next step

The current event path uses target pixels as a lightweight expression carrier;
very large pose changes, severe occlusion, and landmark failure can reduce
confidence and should fall back to the established renderer. Brow movement is
derived from the repository's trusted 106→68 brow index convention. Measure
the paired clips on both the RTX 4070 and RTX 3060 profiles before changing
the flag default or adding another model. Record FPS, seconds/frame, VRAM,
RSS, CPU/GPU utilization, expression metrics, and visual findings.
