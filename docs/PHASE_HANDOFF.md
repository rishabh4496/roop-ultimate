# Phase Handoff — Phase 0 Verified

Date: 2026-08-31

## Current state

Phase 0 baseline verification is **COMPLETE / VERIFIED**. No application code
was changed in this session. The stable performance implementation is
`677385e49dddd9889be780d11fae52d8a07857fd`; the repository also contains the
handoff documentation commit `139e89125de032735a594b62f3e445f83548c691`.

The exact controlled baseline is 9.62 FPS on the RTX 4070 and 4.53 FPS mean on
the RTX 3060 automatic profile. Both recorded 600/600 output frames and zero
wrong-FaceSet applications, but the 3060 profile intentionally disables
TensorRT/enhancement and degrades masking to stay within its lower-VRAM policy.
The strict 3060 RSS target remains unmet at 3.734 GB.

## Safe next phase

Proceed to **Phase 1: realism architecture/dependency audit**. Map the
existing FaceSet, detector, landmark, 3D, tracking, swap, enhancer, mask,
compositor, scheduler, and I/O contracts to the realism plan before any Phase
3 implementation. Do not reopen Phase 0 or claim a new throughput result
without an order-balanced benchmark.

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

## Open evidence

P95 latency, application-managed transfer/synchronization timing, full manual
visual quality review, production-length leak soak, 3060 RSS reduction, 3060
DMDNet compatibility, and difficult-pose/occlusion/no-face recovery remain
open. These are explicitly follow-up items and are not reasons to alter the
frozen performance foundation during the Phase 1 audit.
