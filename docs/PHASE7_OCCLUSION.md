# Phase 7 - Temporal Occlusion and Interacting-Face Engine

Status: implemented, open for real-video validation (2026-08-31)

Phase 7 adds an opt-in causal layer around the existing mask processors and
`face_overlap` ownership fields. It does not add a segmentation model or alter
the default path.

## Runtime behavior

Enable with:

```powershell
$env:ROOP_TEMPORAL_OCCLUSION = "1"
```

Each track owns independent `face_mask`, `visible_face_mask`,
`occlusion_mask`, `previous_mask`, `predicted_mask`, and confidence state.
Tracking annotates interacting track IDs after gap filling and identity
assignment. `face_overlap` remains responsible for full-frame pixel ownership,
so an adjacent face is restored rather than used as another face's swap source.

The causal state machine has three paths:

- normal: run the configured mask engine;
- occlusion event: re-run that configured engine and retain trusted original
  pixels while the event enters or leaves;
- stable occlusion: propagate the cached track mask until refresh, motion, or
  appearance evidence requests re-analysis.

The transition on object exit is mask-only. The entire face/output is never
temporally blurred, preserving expression, blinking, and fine texture.

Optional environment controls:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `ROOP_OCCLUSION_EVENT_THRESHOLD` | `0.12` | restore-mask fraction that declares an event |
| `ROOP_OCCLUSION_INTERACTION_THRESHOLD` | `0.08` | neighboring-track interaction threshold |
| `ROOP_OCCLUSION_STABLE_FRAMES` | `3` | evidence frames before propagation |
| `ROOP_OCCLUSION_REFRESH_FRAMES` | `5` | maximum propagation interval before re-analysis |
| `ROOP_OCCLUSION_APPEARANCE_THRESHOLD` | `0.16` | compact crop change that reopens analysis |
| `ROOP_OCCLUSION_LEAVE_ALPHA` | `0.35` | release amount per leaving frame |
| `ROOP_OCCLUSION_ENTER_ALPHA` | `0.90` | trusted previous-mask retention on entry |

## Required real-video cases

Run from `app` after supplying a real clip and ROI. The harness records a
machine-readable report and deliberately remains pending until the production
renderer supplies actual mask/frame telemetry and manual visual review:

```powershell
env/Scripts/python.exe tests/phase7_occlusion_bench.py --video path/to/clip.mp4 --mask-dir path/to/hand_eye_masks --box 420,160,900,640 --scenario hand_eye --tag phase7_hand_eye_4070
```

Run the same command for `hand_cheek`, `hand_mouth`, `hair`, `glasses`,
`microphone`, `two_faces_touching`, `two_faces_crossing`, and
`partially_hidden`, ideally with separate clips that isolate each event.
Reports are written to `app/output/phase7_occlusion/<tag>/report.json`.

`--mask-dir` is a real per-frame restore-mask export (`.npy`, `.png`, `.jpg`,
or `.jpeg`) from the production crop. The harness measures raw/stabilized mask
temporal differences, propagation path counts, and mask-stage FPS, then emits
a CSV and montage for manual review. For every case also record end-to-end
seconds per frame/FPS, VRAM, GPU utilization, CPU use, and visual output review
of object preservation, smooth restoration, identity separation, and boundary
leakage. No result is valid if a synthetic fixture or an unrelated GPU is
substituted.

## Compatibility and risk

The feature is disabled by default, keeps existing `.fsz` formats untouched,
and reuses existing RealityUX, RealSwap, GPEN, TensorRT, precision, detector,
source-bank, 3D, SAM2, and provider paths. The causal state requires ordered
processing; when enabled the batch path forces one worker so masks cannot be
advanced out of order. Real RTX 4070/RTX 3060 measurements and visual cases
remain open because this checkout has no supplied video fixtures.
