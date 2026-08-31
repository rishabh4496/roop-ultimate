# Phase 5: Pose-aware source selection

Phase 5 adds a classical, cached-aware geometry layer on top of FaceSet V2.
It is additive: V1 FaceSets, the existing source-bank selector, the existing
3D reconstruction path, and embedding-based swappers remain supported.

## Target pose record

For a tracked face, `roop.pose_source_selector.annotate_face_pose()` records a
detached `_pose_v5` object containing yaw, pitch, roll, absolute and relative
face scale, facial proportions, compact eye/mouth/smile expression values,
pose confidence, off-axis angle, perspective risk, inversion state, and an
availability flag. Temporal replay annotates faces after the existing tracking
smoothing and roll latch, so worker completion order cannot change the result.

Direct non-video calls use the same lightweight estimate when source-bank
selection is enabled. No inference session is created and no neural model is
run per frame.

## Source scoring

`FaceSet.select_pose_aware_reference()` scores each V2 source using:

- yaw, pitch, and circular roll distance;
- source identity/quality confidence;
- expression descriptor distance;
- target/source luminance, temperature, and skin-color statistics;
- facial proportions and relative scale.

The selected source index is retained by the temporal track. A small
hysteresis margin prevents source changes at pose-bank boundaries. The
environment variable `ROOP_POSE_SOURCE_SWITCH_MARGIN` can override the
default normalized margin of `0.035`.

The selector prefers a genuinely similar pose. It requests the existing 3D
crop correction when the pose is missing or low-confidence, the target is
between sparse bank poses, source pose coverage is insufficient, the view is
unusually rotated/inverted, perspective risk is high, or facial proportions
do not match. A good V2 match does not incur an additional image-source crop
warp. V1 selection retains the Phase 4 behavior.

## 3D safety changes

`roop.face_3d_recon` retains its public API and cached source-crop inputs. Its
new plan bounds yaw shear, applies substantially tighter pitch shear, rejects
unstable affine transforms, and only permits a horizontal flip when both
source and target are strongly off-axis on opposite sides. Frontal-to-lateral
conversion therefore does not mirror identity details, while opposite strong
views can still use the geometrically justified flip.

## Compatibility and operation

The feature is enabled only through the existing `use_source_bank` option;
user defaults are unchanged. Existing `use_3d_recon` remains opt-in and is
still restricted to image-source swappers, while embedding-based swappers are
not fed sheared/flipped crops. TensorRT/CUDA, FP16/FP32/mixed precision,
detector pools, hardware guards, enhancers, source-bank storage, and provider
fallbacks are unchanged.

The tests use deterministic synthetic pose metadata and landmark fixtures.
They validate source choice from 0 through profile, pitch/roll matching,
inversion, hysteresis, expression/lighting ties, low-confidence fallback,
and safe warp plans. Real-photo identity quality and physical GPU throughput
remain evaluation work rather than claims made by these unit tests.
