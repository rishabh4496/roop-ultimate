# Phase Handoff — Phase 1 Verified

Date: 2026-08-31

## Current state

Phase 0 baseline verification and Phase 1 hardware/inference backend audit are
**COMPLETE / VERIFIED for the physically tested RTX 4070 path**. No application
code was changed in this session. The stable performance implementation is
`677385e49dddd9889be780d11fae52d8a07857fd`; the documentation checkpoint before
this update was `c2ba7224ab4be7edccaef0f09bd2f3dbb7140cca`.

The exact controlled baseline is 9.62 FPS on the RTX 4070 and 4.53 FPS mean on
the RTX 3060 automatic profile. Both recorded 600/600 output frames and zero
wrong-FaceSet applications, but the 3060 profile intentionally disables
TensorRT/enhancement and degrades masking to stay within its lower-VRAM policy.
The strict 3060 RSS target remains unmet at 3.734 GB.

## Safe next phase

Proceed to the realism/temporal-quality architecture and implementation phase.
Use the existing FaceSet, detector, landmark, 3D, tracking, swap, enhancer,
mask, compositor, scheduler, and I/O contracts. Do not reopen the performance
foundation or implement unrelated backend changes without new evidence.

## Preserve

Preserve the two-device capability policy, TRT/CUDA/FP16/FP32/mixed precision
fallbacks, detector alternatives and pooling, FaceSet/source-bank behavior,
3D reconstruction and landmark APIs, temporal tracking and frame ordering,
RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, UltraMax, all enhancers,
finite/wrong-FaceSet/output-integrity guards, and the custom 3060 look values
documented in the performance handoff.

Any change affecting GPU execution, precision, memory, concurrency, batching,
provider configuration, enhancer execution, or buffers requires separate RTX
3060 and RTX 4070 validation. Current session status: current GPU RTX 4070;
other GPU RTX 3060; GPU-sensitive change NO; current GPU test PASS; other GPU
test NOT TESTED.

Phase 1 live 4070 evidence: warm quick benchmark exit 0 in 177.7 s; detector
397.2 calls/s, RealSwap 205.5 calls/s, UltraMax 35.9 calls/s, XSeg 227.9
calls/s, and BiSeNet 55.9 calls/s. Cold build evidence was 199 s for the
swapper and 179 s for BiSeNet. The first cold pass had an XSeg invalid-
throughput harness anomaly; the cached pass succeeded. The cache-key probe
confirmed separate identities for workspace, partition, builder, auxiliary
stream, and CUDA-graph settings.

## Open evidence

P95 latency, application-managed transfer/synchronization timing, full manual
visual quality review, production-length leak soak, 3060 RSS reduction, 3060
DMDNet compatibility, and difficult-pose/occlusion/no-face recovery remain
open. These are explicitly follow-up items and are not reasons to alter the
frozen performance foundation during the Phase 1 audit.
