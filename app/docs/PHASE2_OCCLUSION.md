# Phase 2 — Occlusion: persistence, symmetry, foreground preservation

Status of each spec item, what was already present, what was actually built, and
what is still owed. Read `roop/tracker.py`'s module docstring for the design; this
file is the map and the honest ledger.

---

## 0. The finding that reframes the brief

**`roop/occlusion_mask.py` was a complete, correct, tested implementation of spec
item 3 that nothing in the codebase called.**

It contains `inject_occlusion_engine`, the algebra for
`M_composite = M_face_blend * (1 - M_occluder)` written out in full, and the
proof that appending a mask processor IS that product:

```
result_1 = plate * m1 + swap * (1 - m1)
result_2 = plate * m2 + result_1 * (1 - m2)
         = plate * (m2 + m1*(1 - m2)) + swap * (1 - m1) * (1 - m2)
```

Its own docstring said "`ProcessMgr.initialize` applies it". `ProcessMgr.initialize`
did not. `git grep inject_occlusion_engine` returned the definition and nothing else.

The shipped default is `mask_engine: RealityUX`, `mask_engine_2: None`, so on a
default render **nothing in the chain had been trained to find a hand, a mug or a
microphone in front of a face**. That is the direct cause of the reported symptom
"the hand gets painted over", and it was one call away from being fixed.

This is the failure mode this project keeps rediscovering — see the 2026-08-30
entry where four enhancers failed on 60 of 60 frames while the audit read 100%,
and the 2026-08-31 entry where the adaptive controller was unreachable. The suite
was green throughout, because every test called the function directly.
`tests/test_occlusion_wiring.py` now asserts the call sites exist, by AST walk.

---

## 1. Persistent multi-face tracking

### What already existed

| piece | where | state |
|---|---|---|
| Kalman/Hungarian tracker, 8-state constant velocity, IoU + ArcFace | `face_analyser.FaceTracker` | working, `max_age=30` already |
| whole-clip identity tracks, Re-ID, stitching | `procmgr_tracking._precompute_tracks` | working |
| detection scheduling / ROI planning | `temporal_tracker.TemporalFaceTracker` | working |
| gap fill across detection misses | `procmgr_tracking._build_temporal_faces` | working, `ROOP_TEMPORAL_GAP=10` |

### The actual gap

`FaceTracker.update()` **returned only the detections it was handed.** It predicted
in order to *score the assignment* and never to *maintain a face*. A frame with no
detection therefore produced no face, no matter how confident the track was one
frame earlier — the swap blinked off.

It was also only reached on the per-frame detection path (`_tfaces is None`), which
is **not** the shipped default (`temporal_detection: true`).

And `_build_temporal_faces`'s gap fill is an **interpolation between two real
anchors**, capped at 10 frames. It covers nothing when the gap is longer than that,
when `_bridgeable`/`_interp_collides` refuse it, or past a track's last observation
— which is exactly the population a long occlusion produces.

### What was built

`roop/tracker.py` — the tracker moved here from `face_analyser` (which re-exports
it, so every existing import is unchanged) and gained:

* `FaceTrack` trajectory state: embedding, bbox history, landmark history, pose,
  confidence, `velocity` off the Kalman state, `occlusion_state`, coast counters.
  Both histories are **bounded deques** — an unbounded one on a 60k-frame render is
  the host-RAM leak fixed on 2026-08-26.
* `coast()` — synthetic faces from `x_{t|t-1} = F x_{t-1}` for tracks with no
  detection, stamped `_coasted`, `_coast_age`, `_interpolated`, `occlusion_state`.
* `update_with_coasting()` — both, in one call.
* `MAX_COAST_FRAMES = 15` and `MAX_LOST_FRAMES = 30`, deliberately different:
  how long a *swap* may ride a prediction is not how long a *track* stays
  available for re-association.

### Where it runs

| path | site | shipped default? |
|---|---|---|
| whole-clip pre-pass | `procmgr_tracking._coast_track_gaps` | **yes** |
| per-frame detection | `ProcessMgr.swap_faces` → `_dispatch_face_tracker.coast` | fallback |
| reference module | `face_swapper.track_faces` / `process_frame_tracked` | not the render path |

**Ordering is the fix on the per-frame path.** The tracker call was previously
*after* the `if not faces: return` bail-out — i.e. unreachable on precisely the
frames an occlusion produces. It now runs before it.
`test_coasting_runs_before_the_no_face_return` fails if that is undone.

**In the pre-pass, interpolation still runs first and wins.** It has the closing
anchor, so it knows where the face actually went; prediction only ever sees frames
interpolation left empty. The pass runs the timeline forward and backward and each
hole takes whichever prediction reaches it younger, so a 20-frame occlusion is
covered from both ends and correctly left empty in the middle rather than having
one extrapolation pushed the whole way across.

### Why coasting is guarded rather than trusted

A coasted face is invented and carries the track's mean embedding, so it **passes
every downstream identity gate by construction** — the same property that makes
gap fill dangerous (`phantom-gapfill-swap`). A prediction that runs off the end of
a track paints a swapped face onto the background, which is worse than the flicker.

Every coasted face is therefore: capped at 15 consecutive frames; refused unless
the track has `>= 3` hits; refused once the predicted box leaves the frame by more
than half its area; refused if it lands on another track's real detection; and
carries a decaying `det_score`. It is stamped `_interpolated` **on purpose**, so
the existing swap audit counts it as gap-filled — a new flag would have made
coasted faces invisible to the report that exists to surface exactly them.

`ROOP_COAST_FRAMES=0` restores the previous output exactly.

---

## 2. Occlusion-aware landmark symmetry inpainting

`roop/tracker.py`, second half. Three functions and a state helper.

**The axis** is the nasal-bridge midline: eye midpoint → mouth midpoint, the same
axis `orientation.roll_from_face` uses. It is derived from the landmarks, not from
the detector box, so it survives roll.

**The pairing is derived, not hardcoded.** `derive_mirror_map` canonicalises a
complete observation onto the axis and matches each point to the one nearest its
own reflection. It is accepted only when the result is a proper involution
(`m[m[i]] == i`) with every residual inside a scale-free tolerance — so a bad
reference is **refused** rather than silently producing a permutation that mirrors
the chin onto the brow. This works for any landmark count; no 106-point index table
had to be guessed.

**Foreshortening is corrected** from the visible pairs' own half-width ratio, and
the whole repair is refused past `|yaw| > 55°`, where the far half is no longer a
planar reflection of the near one. 55 is set well inside where the geometry breaks
because `solve_pose_5pt` is measured 15–20° off per person.

**Visibility comes from the occluder mask** (`landmark_visibility`) — the only thing
in the pipeline that knows which landmarks are behind something. With no mask,
everything reads visible: this must never invent an occlusion, because inpainting a
landmark that was fine replaces a measurement with an estimate.

**`occlusion_state`** is stamped by `procmgr_masking._stamp_occlusion_state`, from
the occluder mask that stage already computed — no second inference, roughly a
`(106, 2)` matrix multiply. It is confined to the occluder family
(`mask_occluder`, `mask_xseg3`): a face-shape masker is HIGH on background too, so
reading it off RealityUX would mark every jaw and hairline landmark occluded and
put essentially every frame into the partial state. A flag that is always on
carries no information.

**Why the repair is published rather than applied in the render path.** By the time
any mask engine runs, alignment has already happened — the crop the mask lives in
was built from the very landmarks in question. Overwriting them there would change
nothing about that frame and would corrupt the track geometry every later frame is
smoothed against. So `process_mask` stamps `occlusion_state`,
`_occluded_landmark_frac` and `_landmarks_symmetric` for consumers that can use
them.

In `face_swapper.py`, where the swap is self-contained, the repair **is** applied:
the occluder mask moved from step 4 to step 1a so it exists *before* the blink and
inner-mouth retention read the landmarks. That ordering was the bug — under a hand,
the detector returns a guessed eye, and the blink logic measured an eye-aspect ratio
on a hallucination and pasted the target's eyelid through the hand.

---

## 3. Foreground occluder mask preservation

`ProcessMgr.initialize` now calls `inject_occlusion_engine`. On the shipped default
chain the processor dict becomes:

```
faceswap → ultramax → mask_realityux → mask_occluder
```

Order is execution order, and the occluder must run **last** for the product to come
out right. Injection is skipped when the chain already has an occlusion-aware engine
(`mask_occluder`, `mask_xseg3`), and when there is no `faceswap` at all — the preview
mask editor builds a mask-only chain with no swap to protect.

`ProcessOptions.disable_occlusion_injection` is a measurement escape hatch, not a
user setting (`enable_occlusion_mask` is). `tests/occlusion_ground_truth.py` sets it
on its deliberately-unprotected reference arm; without it, injection would protect
the very object that reference exists to leave unprotected and the metric would
collapse toward "no difference" while nothing had got worse.

---

## Measured, RTX 4070 / TensorRT / realswap + RealityUX + UltraMax

Read `[occl]`'s own caveat first: the occluder is **composited by the harness**, so
its mask is exact — but a flat synthetic patch is out of distribution for models
trained on real hands, hair and microphones. A neutral result here is weak evidence
either way, and this section does not pretend otherwise.

### 1. Wiring the occluder measured NEUTRAL on the synthetic occluder

`tests/occlusion_ground_truth.py --occluder texture --frames 20`, same clip, same
positions, peak face cover 40.8%:

| protection | injection OFF (= HEAD) | injection ON |
|---|---:|---:|
| worst | 18% | 20% |
| median | 40% | 42% |
| failing frames | 7 of 7 informative | 7 of 7 informative |

Per-frame: 57→59, 40→40, 42→42, 46→48, 18→20, 39→45, 23→35. **Inside the noise, and
not claimed as an improvement.** It reproduces the recorded Phase 10 baseline
(8–30% worst, 22–53% median) rather than moving it.

**The path was verified to execute**, and the first attempt to verify it was wrong:
a `tail -35` in the invoking shell pipeline had discarded the startup lines, so the
`[Occlusion]` banner appeared absent and the run briefly looked like a null
comparison. Re-run with the full log captured, it prints

```
[Occlusion] occlusion masking on: appended 'mask_occluder' (256x256 ONNX) after ['mask_realityux']
```

exactly once — correct, because `initialize` is called per frame and every call
after the first sees a dict that already contains the occluder.

### 2. Cost: 3.79 ms/face

Measured through `angle_bench.init_pipeline`, never a bare process (without the
app's init, TensorRT's DLLs are off PATH, ORT silently falls back to CPU, and a
4 ms model reports as 210 ms):

```
Mask_Occluder     3.79 ms/face
Mask_RealityUX   25.81 ms/face      <- already paid on the default chain
```

So it adds roughly 15% to a mask stage that was already running. This is **added
GPU work**, and by this project's standing rule that means it cannot make the
render clock faster; no throughput claim is made for it in either direction.

### 3. It does not damage clean faces

The occlusion harness only ever measures *inside* the composited object, so it is
structurally blind to the opposite failure: an engine that false-positives on clean
skin restores the original pixels there and quietly un-swaps part of every face in
the clip. Ten clean frames, occluder-in-chain vs not, inside the face box:

```
occluder-vs-no-occluder : mean 0.898/255  max 1.291   (noise floor 0.71/255)
swap retained           : 98.0%
```

Barely above the pixel noise floor measured on 2026-09-01, and 98% of the swap
survives. Acceptable.

### 4. Why it stays on by default anyway

Three reasons, none of them "the measurement was good":

* `roop.globals.enable_occlusion_mask` has defaulted to `True` all along, and
  `--disable-occlusion-mask` is already a documented CLI flag. The project's
  declared intent was always that this be on; it simply had no implementation
  behind it.
* The one test that says it is neutral is a test whose own docstring says it cannot
  see this — a synthetic patch is out of distribution for `face_occluder.onnx`.
* It is cheap (3.79 ms) and demonstrably harmless on clean frames.

It is switchable: `ROOP_OCCLUSION_MASK=0`, `--disable-occlusion-mask`, or
`ProcessOptions.disable_occlusion_injection` for a single arm.

---

## Verification

```
env/Scripts/python.exe -m unittest discover -s tests -t . -p "test_*.py"
env/Scripts/python.exe -m unittest tests.test_tracker_coasting -v
env/Scripts/python.exe -m unittest tests.test_landmark_symmetry -v
env/Scripts/python.exe -m unittest tests.test_occlusion_wiring -v
```

Rendered evidence, protected vs unprotected, exact per-pixel ground truth:

```
env/Scripts/python.exe tests/occlusion_ground_truth.py --occluder texture --frames 20
ROOP_OCCLUSION_MASK=0 env/Scripts/python.exe tests/occlusion_ground_truth.py --occluder texture --frames 20
```

Counterbalance anything end-to-end, at 600 frames, and read swap rate beside fps —
a setting that goes faster by finding fewer faces has not got faster.

---

## Still owed

1. **Real-occluder footage — the one that matters.** Everything measured above uses
   a *composited* occluder, out of distribution for every model in the chain. That
   is why wiring the occluder reads neutral, and it is why that neutral reading
   must not be turned into "the occluder does not help". Hands, glasses,
   microphones and hair on real footage is the test that would actually settle it,
   and it has not been run.
2. **A rendered A/B of coasting on footage with real dropouts.** The unit tests
   prove the mechanism; they do not prove a clip improved. `[Coast]` prints the
   count on every run, so silence means it did not fire.
3. **Nothing here is measured on the RTX 3060.** All of it is device-independent
   code, but the numbers are 4070-only.
4. **`_landmarks_symmetric` has no consumer yet** in the render path — it is
   published and unread. The obvious first consumer is the landmark hull in
   `procmgr_masking`, but changing the hull changes the seam, and that is a
   quality claim that needs a render behind it.
