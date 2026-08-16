# Geometry tests

```
cd app
env/Scripts/python -m unittest discover -s tests -t .    # Windows
python -m unittest discover -s tests -t .                # anywhere else
```

50 tests, ~2 s. No GPU, no models, no video, no network — pure numpy/cv2 over a
synthetic head projected at a known pose (`tests/facegeom.py`), using the
project's own 3-D reference face so the tests and the shipping pose code agree
on what a head is shaped like.

Plain `unittest`, so there is nothing to install. `pytest` also discovers these
if you happen to have it.

## What is covered, and why

These target the alignment/masking geometry, which is where the recurring bugs
have been: silent, pose-dependent, and invisible on frontal footage.

| File | Guards |
|------|--------|
| `test_pose_ratios.py` | `kps_pose_ratios` is **monotonic** in yaw to 90°, roll-invariant, and returns `(None, None)` rather than NaN on degenerate keypoints |
| `test_alignment.py` | `estimate_norm` is a **bit-exact no-op** when `yaw_align` is off (1350 size × template × pose combinations); the profile path is gated off mid-angles and kills the nod-coupled rotation swing |
| `test_landmark_mask.py` | `create_landmark_mask` is **bit-identical** on upright faces and **rotationally equivariant** on tilted ones; `_mask_crop_box` never returns a degenerate rectangle |
| `test_face_overlap.py` | Two swapped faces that touch get a **boundary**, not a smear: separated faces cost nothing, the sequential composite leaves no plate showing along the join, the boundary does not move with match order, and paint order is far-to-near and stable under bbox noise |
| `test_angle_handling.py` | The three shared angle layers. The polygon is placed by `swap_template_points`, **not** a hand-rolled `arcface_dst * size/112` — that guess is 53px out on a 512 crop and shipped once, because this test made the same guess. It **never trims the forehead** — which is what it was in fact doing, and *only* doing: split by region, 0.0% below the brow line at every pose and up to 8.1% above it, because the 106 landmarks stop at the eyebrows and the mask hull and the polygon extrapolate past them differently. Note the older property, "never trims a truly-visible landmark", passed at every pose while that was happening: **a safety property stated in terms of the evidence is blind where the evidence stops.** The trim is off entirely near frontal. The trim is applied **after** the feather (before it, shrinking the matte shrank the feather 29px→18px and *sharpened* the seam). The trim is **capped** so a wrong pose cannot take more than a quarter of the matte — measured against the real matte, because against the bare crop ellipse a normal trim scores 40% where against the real one it scores 1.9%. The fade is bounded, monotone, exactly off at 0, fades a 1024/2048 enhancer surface without ghosting, and **withdraws from the outside in**: at every fade the ceiling can reach, each of the five keypoints is either fully swapped or fully original and never a superposition, which is what a doubled eye or mouth is. Both paste call sites pass the polygon; the default path is bit-identical |
| `test_pose_shape.py` | **How wrong the 5-point pose solve is on a head that is not the reference head** — the measurement that decides whether the angle layers can be on by default. Nose protrusion carries most of the yaw signal in 5 points, so a nose 40% more prominent reads yaw 30° as 44.6° and one 40% flatter reads 45° as 16.5°. Pins that it is **systematic per person** (5× larger than 1px of keypoint jitter, so no temporal filter touches it) and that it is **big enough to saturate the engagement band** the three layers gate on |
| `test_track_stitch.py` | Tracklets that are **one person interrupted** are chained on geometry before any identity gate runs — the fix for the swap switching off for whole stretches when something crosses a turned face. Most of the file is about **refusing**, because the failure modes are asymmetric: a missed link costs what the pipeline already did, a wrong link hands one person's face to another. Refusal cases drive `_stitch_tracks` directly rather than through a scripted clip, or the scan's own association decides them first and the test measures the tracker instead of the rule it names |
| `test_settings_wiring.py` | A setting that reaches the UI actually reaches the render — no half-wired controls that silently do nothing |

`test_settings_wiring.py` parses `FaceSwap.jsx` and `api.py` as text rather than
importing them, which is how it can check across the Python/JavaScript boundary
with no browser and no running server. Its exception lists (`VIA_MASK_OFFSET_HELPER`,
`PREVIEW_ONLY`, `RUN_ONLY_TEMPORAL`, `RUN_ONLY_OUTPUT`) each carry the reason that
setting is legitimately absent from the other side — if you add to them, add the
reason too.

Two guarantees are load-bearing and easy to break by accident:

- **`yaw_align` off must be bit-exact.** `off` is the default and the A/B
  baseline, and the only way back to the pre-change output, so it has to stay a
  true no-op. The same applies to the other two angle layers: strength 0 and the
  trim disabled must both be bit-identical to not having them.
- **Upright faces must be bit-identical.** Frontal is the overwhelming majority
  of frames, so any mask change there is a regression in the common case.

## Refactor safety harness

`surface_snapshot.py` is a tool, not a test. It records everything a caller
could depend on — every module-level name, every class method, every signature,
and for `api.py` the full FastAPI route table including declaration order — so a
decomposition can be *proven* to be a pure move rather than asserted to be one:

```
env/Scripts/python tests/surface_snapshot.py before.json
# ...move code...
env/Scripts/python tests/surface_snapshot.py after.json
env/Scripts/python tests/surface_snapshot.py --diff before.json after.json
```

Route order matters: FastAPI matches in declaration order, so a reordering can
silently shadow a handler without changing any path. The diff reports that
case separately.

Use it on `api.py` and `ProcessMgr.py`, which have no behavioural tests. A clean
surface diff plus a green suite is what makes "I only moved code" checkable.

## Style

Assert **properties**, not magic numbers. The bug that motivated this suite was a
heuristic that looked fine at the angles anyone tested and collapsed to exactly
zero at 90° — a threshold test would have passed. `test_monotonic_in_yaw` checks
monotonicity across the whole range instead, which is the property that was
actually violated.

Where a test encodes a known limitation rather than a requirement
(`test_pitch_ratio_degenerates_at_profile`), it says so and explains what to do
if it ever starts failing.
