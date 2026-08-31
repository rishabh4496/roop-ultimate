# Phase 6 Temporal Identity Consistency Engine

Date: 2026-08-31

## Implementation

`app/roop/temporal_identity.py` adds an opt-in, bounded
`TemporalIdentityStabilizer`. `ProcessMgr` creates a fresh instance for each
run and the existing ordered tracking replay feeds it before swap workers run.
The layer preserves the current track/source assignment authority and adds:

- per-track source identity, selected source-bank entry, source identity
  embedding, target embedding, pose, landmarks, alignment transform, swap and
  output confidence, previous canonical output, previous mask, and lighting;
- confidence-weighted landmark, target-embedding, pose, and lighting updates;
- persistent source-bank switching (`3` requests by default) with immediate
  but bounded re-estimation on a major yaw/pitch/roll transition;
- transition alpha across source-bank representation changes;
- mask-edge history with faster reveal when an occluder enters;
- low-frequency-only aligned-output blending, preserving current high-frequency
  detail and expression instead of blurring the whole face.

The runtime feature flag is `ROOP_TEMPORAL_IDENTITY=1`. It remains disabled by
default so existing renders and user configuration are unchanged. Other knobs
are environment overrides: `ROOP_TEMPORAL_IDENTITY_SWITCH_FRAMES`,
`ROOP_TEMPORAL_IDENTITY_TRANSITION_FRAMES`,
`ROOP_TEMPORAL_IDENTITY_GEOMETRY_ALPHA`,
`ROOP_TEMPORAL_IDENTITY_OUTPUT_STRENGTH`,
`ROOP_TEMPORAL_IDENTITY_MASK_STRENGTH`,
`ROOP_TEMPORAL_IDENTITY_MAJOR_YAW`,
`ROOP_TEMPORAL_IDENTITY_MAJOR_PITCH`,
`ROOP_TEMPORAL_IDENTITY_MAJOR_ROLL`, and
`ROOP_TEMPORAL_IDENTITY_CACHE_SIZE`.

Output history is stored as one bounded canonical crop per track. Crop blending
is bypassed on the parallel stabilization path because a previous-output state
cannot be updated safely out of frame order; precomputed geometry and source
decisions remain available there.

## Benchmark

Run from `app` with a real clip:

```powershell
$env:ROOP_TEMPORAL_IDENTITY = "1"
env/Scripts/python.exe tests/phase6_temporal_bench.py --video path/to/clip.mp4 --box 420,160,900,640 --scenario static --tag phase6_static_4070
env/Scripts/python.exe tests/phase6_temporal_bench.py --video path/to/clip.mp4 --box 420,160,900,640 --scenario talking --tag phase6_talking_4070
env/Scripts/python.exe tests/phase6_temporal_bench.py --video path/to/clip.mp4 --box 420,160,900,640 --scenario rapid_rotation --tag phase6_rotation_4070
env/Scripts/python.exe tests/phase6_temporal_bench.py --video path/to/clip.mp4 --box 420,160,900,640 --scenario blinking --tag phase6_blinking_4070
env/Scripts/python.exe tests/phase6_temporal_bench.py --video path/to/clip.mp4 --box 420,160,900,640 --scenario motion --tag phase6_motion_4070
env/Scripts/python.exe tests/phase6_temporal_bench.py --video path/to/clip.mp4 --box 420,160,900,640 --scenario lighting --tag phase6_lighting_4070
```

The harness reports mean and p95 raw/stabilized temporal crop deltas and
writes `temporal_delta.csv` plus `before_after_montage.png`. The montage is
marked `pending_manual_review`; a human must inspect blink preservation,
talking expression, rapid rotation lag/ghosting, motion, and lighting changes.
The current checkout has no video fixture, so no real temporal-delta or visual
quality result is claimed yet.

## Compatibility and risks

The layer is classical NumPy/OpenCV state only: it creates no model, ONNX,
TensorRT, CUDA, enhancer, or detector session. RealityUX, RealSwap, GPEN 256
Pro, GPEN Realistic, UltraMax, all enhancer paths, source-bank V1/V2 loading,
3D reconstruction, TensorRT/CUDA/FP16/FP32/mixed precision, detector
alternatives, and hardware guards remain available. V1 `.fsz` compatibility is
unchanged. A physical RTX 3060 run and real-video quality review remain open.
