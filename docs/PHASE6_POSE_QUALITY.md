# Phase 6 Pose and Source-Bank Quality Evaluation

Date: 2026-08-31

## Purpose

This phase adds a repeatable real-photo evaluation harness for the existing
V2 pose-aware source selector and the existing `live_swap` path. It is
benchmark-only: no runtime default, user configuration, or `.fsz` archive is
changed.

## Measurement contract

- Load the existing V1 `ashna.fsz` source and `harjot.fsz` target archives with
  the established `angle_bench` ingestion path.
- Promote the loaded faces in memory to V2 metadata so legacy archives can be
  evaluated without a migration write.
- Report measured yaw, pitch, and roll values and explicitly flag missing
  profile, pitch, and inversion coverage.
- Evaluate source selection against the target pose with a 15-degree yaw and
  pitch matching tolerance, including the selector's 3D-fallback hint.
- Run the established swap/quality grader with source-bank disabled and
  enabled in both `off_on` and `on_off` orders. This makes initialization order
  visible and avoids presenting first-arm setup time as a causal speedup.
- Grade identity against the V2 cached global identity vector. The historical
  V1 first-face fallback remains unchanged for existing angle-bench callers.
- Record a physically absent requested GPU as `pending`; never substitute the
  other hardware profile.

## Reproduction

From `app`:

```powershell
env/Scripts/python.exe tests/phase6_pose_quality.py --target "RTX 4070" --provider auto --source ashna --target-faceset harjot --rolls 0,90,180 --tag phase6_4070_balanced
env/Scripts/python.exe tests/phase6_pose_quality.py --target "RTX 3060" --provider auto --source ashna --target-faceset harjot --rolls 0,90,180 --tag phase6_3060_pending
```

Artifacts are written to the ignored `app/output/phase6_pose_quality/` tree:
`report.json`, `selection.csv`, and one `quality.csv` per order and arm.

## Current evidence

The available V1 archives contain five real yaw plates, including profile,
with no real pitch or inversion plates. On the physical RTX 4070, the balanced
run completed 30 rows per source-bank arm with 30 detections (100%); elapsed
wall time was 24.900 seconds for source-bank off and 12.232 seconds for
source-bank on across both orders. These totals include two repeated arms and
must not be interpreted as a production FPS result or a causal source-bank
speedup. The selector produced 15 valid rows, 80% within the 15-degree pose
match tolerance, mean absolute yaw error 9.834707 degrees, mean absolute pitch
error 2.699181 degrees, and a 60% 3D-fallback hint rate.

The RTX 3060 command was correctly recorded as pending because no RTX 3060 is
physically present on the current machine. No real-photo quality improvement,
temporal metric, visual-review score, VRAM/RSS measurement, 3D-fallback arm, or
strong-pitch/inversion claim is made by this phase.

## Next measurement

Supply V2 or legacy photographs covering yaw 0/30/45/60/75/profile, upward and
downward pitch, roll, and inversion. Repeat on both required hardware profiles
with source selection and 3D fallback separately enabled, then add annotated
video metrics for identity switches, landmark/pose jitter, temporal stability,
detail retention, expression/lighting compatibility, FPS, VRAM, and RSS.
