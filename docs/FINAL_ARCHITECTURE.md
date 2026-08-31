# Roop Ultimate — final optimization architecture

This document describes the complete production pipeline as implemented at the
Phase 16 gate. It separates source-authoritative identity from target-
authoritative appearance, and keeps expensive work behind measured or
event-driven gates.

## Runtime and configuration

`settings.py` loads the saved configuration, while the API and preview paths
apply the same values before processing. `HardwareProfiler` records the actual
GPU, VRAM tier, CPU topology, codecs, provider versions, and precision
capabilities. `backend_manager` resolves an ordered provider chain and always
keeps CPU as the explicit last resort when available.

NVIDIA TensorRT namespaces include precision, GPU/SM, CUDA, display driver,
TensorRT, ONNX Runtime, and builder-tuning identity. Precision policy is
model-specific: unsafe FP16 paths fall back to FP32, mixed mode retains the
measured safeguards, and provider options are copied rather than mutated.
Small GPUs use the guarded single-context policy and bounded stabilization
budget; the RTX 4070 profile retains its measured pools and 12-worker policy.

## Processing flow

```text
input image/video
  -> decode / frame cache / optional NVDEC
  -> face detection and quality filtering
  -> temporal tracks, identity assignment, pose and expression analysis
  -> FaceSet lookup and pose-aware source selection
  -> alignment and swap crop
  -> target illumination/color adaptation
  -> identity-detail restoration (FaceSet V2, confidence weighted)
  -> adaptive enhancer selection or explicit manual enhancer
  -> semantic/occlusion/geometry mask construction
  -> adaptive temporal compositing and target-region blending
  -> event-driven temporal quality control
  -> encode / optional audio mux / output-integrity checks
```

### Detection, tracking, and geometry

Detection produces face boxes, landmarks, quality, pose, and optional 3D
geometry. Track assignment, gap filling, cross-shot stitching, and source
identity selection operate on bounded temporal history. Alignment uses the
canonical face template and retains the transform required to map source
representations into target geometry.

### FaceSet V2 and identity detail

FaceSet V2 preserves the legacy root-level PNG `.fsz` contract and adds
`metadata.json`. New archives carry schema/version, source entries, identity
embeddings, pose bank, appearance/expression descriptors, high-frequency
identity-detail representations, and checksums. Legacy archives remain
readable and can be migrated without losing their PNG references.

The detail path stores stable high-frequency candidates from the source
gallery. During processing, only confidence-supported details are aligned into
the target face. Low-strength compositing, target lighting, occlusion,
expression, and temporal stability gates prevent wholesale source-texture
pasting, camera-noise transfer, flicker, and artificial sharp points. Detail
restoration is protected from downstream enhancers and restorers.

### Target-conditioned appearance

The target face and surrounding region determine luminance, white balance,
color temperature, local contrast, shadow fraction, highlight rolloff, and
skin-region chroma. Temporal appearance state is smoothed so stable lighting
does not oscillate between warm, neutral, and blue frames. Normal, dark, and
very-dark tiers reduce restoration/sharpening and avoid hallucinated exposure
in low light.

### Enhancers

The manual paths remain independently selectable: RealityUX, RealSwap,
GPEN 256 Pro, GPEN Realistic, UltraMax, and the other registered restoration
paths. Adaptive mode evaluates face resolution, sharpness, blur, pose,
illumination, occlusion, confidence, temporal stability, and output quality;
it selects the least aggressive suitable path. FAST, BALANCED, REALISTIC, and
MAX QUALITY are preferences, not promises that a stronger model is always
better. Enhancers are not run sequentially unless explicitly selected.

### Masks and compositing

The compositor derives masks from face geometry, semantic regions, occlusion,
local contrast, and target lighting. Feathering is bounded and adaptive;
full-frame mattes remain authoritative while ROI warps reduce measured blend
cost. Target low-band color adaptation and generated high-band/detail
protection are applied without erasing facial boundaries. A temporal mask EMA
prevents hard/soft/hard edge oscillation around jaw, cheek, forehead, hair,
glasses, and hands.

### Temporal quality control

The optional controller compares each tracked face with short bounded history.
It detects identity drift, mask popping, brightness/color jumps, geometry
jumps, enhancer hallucination, detail loss, eye-state discontinuity,
jaw/expression discontinuity, and flicker. Normal frames pass through. A
latched anomaly triggers only a targeted correction—stable transform/source,
alignment, occlusion, color, enhancer strength, detail, or previous alpha—so
expensive correction is event-driven and real motion is preserved.

### Performance and resource ownership

The stage profiler observes detection, tracking, alignment, FaceSet lookup,
swap, expression, occlusion, detail, enhancement, lighting, masking,
blending, and encoding. Session pools, global GPU guards, worker-local locks,
fixed-shape opportunities, memory budgets, and queue boundaries remain owned
by their existing subsystems. The optimizer changes only automatic settings
with measured evidence; explicit user settings remain authoritative.

TensorRT/ONNX sessions are initialized in the process that uses them and are
not switched between providers or precision modes live. Fresh processes are
required for precision/provider benchmark arms. Output order, finite pixels,
frame count, audio, and duplicate/black/uniform frames are independently
checked by `phase16_integrity.py`.

## Validation architecture

`app/roop/final_quality_gate.py` defines the standardized 17-clip manifest,
four quality modes, five named component arms, required performance metrics,
required quality metrics, faceset checks, and winner rules. It refuses to
choose fastest/balanced/quality/night/angle/multi-face winners from incomplete
rows. `app/tests/phase16_final_quality_gate.py` writes the final JSON report.

The production gate is complete only when every required clip/configuration
row has real evidence, all important enhancers and modes have independent
output review, old and new `.fsz` checks pass, provider/hardware fallbacks are
verified on their applicable hosts, and prior-phase regression suites remain
green. A listed provider, a successful process, or a clean synthetic fixture
is not by itself a production-quality pass.
