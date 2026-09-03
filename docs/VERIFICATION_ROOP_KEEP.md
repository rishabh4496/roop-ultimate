# Verification sweep over the roop-keep corpus (2026-09-03, RTX 4070)

Harness `tests/verify_roop_keep.py`. Report:
`<roop-keep>/output/verification_report.{html,json}`, 18 diagnostic strips in
`output/diagnostic_frames/`, 8 rendered outputs.

Revision **2813da9f1** for every video -- the harness warns and voids the set
if the tree moves mid-batch. Window: first 600 frames per clip, every 5th
frame analysed, `--capture-budget 120`. Stack as `config.yaml` runs it:
realswap + UltraMax + RealityUX, TensorRT, `verify_swap` on.

    11 pass   12 fail   21 n/a   4 advisory

## Per video

| clip | angle | colour | blink | texture | tracking | occlusion | detected |
|---|---|---|---|---|---|---|---|
| `s1.mp4` | pass | pass | n/a | fail | n/a | n/a | 41/206 frames |
| `s2.mp4` | n/a | n/a | n/a | pass | n/a | n/a | 62/306 frames |
| `s3.mp4` | — | — | — | — | — | — | **CAPTURE FAILED** |
| `s4.mp4` | — | — | — | — | — | — | **CAPTURE FAILED** |
| `s5.mp4` | fail | pass | n/a | fail | n/a | advisory | 79/600 frames |
| `s6.mp4` | fail | pass | n/a | fail | n/a | advisory | 81/600 frames |
| `d1.mp4` | — | — | — | — | — | — | **CAPTURE FAILED** |
| `d2.mp4` | fail | n/a | n/a | fail | fail | advisory | 87/432 frames |
| `d3.mp4` | n/a | n/a | n/a | fail | pass | n/a | 79/394 frames |
| `d4.mp4` | fail | pass | n/a | pass | pass | advisory | 96/600 frames |
| `d5.mp4` | — | — | — | — | — | — | **CAPTURE FAILED** |
| `d6.mp4` | fail | pass | n/a | fail | pass | n/a | 98/488 frames |

**`n/a` means NOT TESTED and is never counted as a pass.** 21 n/a against 11
pass is the headline: most criteria had no qualifying population in a
600-frame window. `blink` is n/a on all 12 -- no clip's window contained a
target EAR below 0.20.

## THE REAL FINDING: lateral profile wrecks the swap on s5 and s6

| clip | >45 deg yaw | aspect change mean / max | identity cos on >45 deg |
|---|---|---|---|
| s5 | 45 of 79 (57 pct) | **11.5 / 26.1 pct** | **0.256** |
| s6 | 33 of 81 (41 pct) | **11.9 / 42.9 pct** | **0.359** |
| d2 | 0 of 165 | 5.1 / 47.0 pct | 0.224 |
| d4 | 2 of 138 | 8.1 / 34.2 pct | 0.608 |
| d6 | 1 of 210 | 1.1 / 13.2 pct | 0.043 |

Tolerance is 5 pct. A healthy swap on this project reads identity 0.41-0.45.

**s5 and s6 are the genuine defect.** Median yaw 54.1 and 21.3 deg, max 90.8
and 94.5 deg, aspect distortion averaging ~2.3x tolerance, identity collapsed
to 0.26-0.36. Visual: at frame 160 of s5 the paste covers the WHOLE HEAD
including hair, with a hard rectangular boundary and the jaw cut into a
floating fragment (`output/s5_face_crop.png`).

**d2, d4 and d6 are NOT the same thing, and the criterion overstates them.**
The gate is `|yaw|>45 OR |roll|>30`; those three have essentially no yaw past
45 deg, so their flags are ROLL-driven, and they fail only because any single
frame over tolerance fails the whole video. Do not read five clips as sharing
s5's defect.

Corpus-wide: **98 of 300 graded extreme-pose faces (33 pct) exceed the aspect
tolerance**, mean 6.5 pct.

### Why nothing refused these swaps

`verify_swap` is the outcome guard for exactly this case -- its own comment
calls it "the only thing standing between a head turned past 90 degrees and a
complete frontal face painted on its cheek". It is ON, and it passed
**392 of 392 faces on s5**.

`SWAP_MOVED_TOL = 1.0`, and the note beside it records that "the 80-88 degree
band -- real profiles, which swap acceptably -- sits at 0.340 and is
deliberately spared". That was calibrated on a studio yaw sweep. s5/s6 sit in
that band and do NOT swap acceptably, on 4K footage with glasses and a heavy
fringe.

**Hypothesis, not conclusion.** Confirming it needs a per-frame read of the
guard's own metrics on s5, which this sweep does not capture. Do that before
touching the threshold -- four gate changes in this project have been
implemented and reverted because the population was not in the band the
change targeted.

## Corpus gaps: 4 genuine, 1 self-inflicted

| clip | at budget 120 | verdict |
|---|---|---|
| s3, s4 | "no usable face was found anywhere in the clip" (s4 scanned 251 of 374 frames) | **genuine** |
| d1, d5 | "no scanned frame held the expected number of usable faces", no budget warning | **genuine** -- matches the recorded note that d1 is 0 pct gradeable because both people overlap in every frame |
| d6 | captured fine at 120 (242 frames, separation 0.889) after failing at 30 | **my budget choice**, not the clip |

A budget of 30 cost 42 pct of the corpus on the first attempt. The project
default is 90; this sweep used 120.

## Two defects found and fixed on the way

1. **Output frame rate was hardcoded** (`9b8c8fe`). `run_swap` passed
   `ProcessEntry(..., 30.0)` regardless of source while `trim()` preserves the
   source rate, so s5 (4096x2160, **25 fps**) came back as 600 frames at
   **30 fps** -- 20 pct fast, audio would drift. Per-frame grading pairs by
   index and is unaffected, which is why it survived; anything anyone WATCHES
   is affected. Hits every non-30fps clip through `two_face_video.py`.

2. **The harness hardcoded a drive letter** (`9b8c8fe`). The 3060 has no `G:`
   drive, and a missing DIRECTORY does not raise the way a missing file does --
   the sweep would iterate nothing and report a clean empty result. Caught by
   the existing `tests/test_fixture_paths.py`.

## Read the metrics with these limits

* **`texture` is near-vacuous in the FAIL direction.** Laplacian variance
  counts synthesised grain as texture, and `merger_grain_match` is 0.45, so
  the swap measures ~5.7x the plate on s1 while plate energy is healthy
  (median 212, min 67). Read `plate_energy_absolute` /
  `swap_energy_absolute`, not the ratio.
* **`occlusion` is ADVISORY and never gates.** The occluder is inferred
  (largest interior non-skin blob, features excluded). The first version fired
  on 20 of 20 faces of d4 with scores clustered 0.445-0.598 under a 0.60 floor
  -- it was detecting eyes, brows and lips, which the swap legitimately
  repaints. After recalibration: 9 of 60, scores 0.470-0.782. It still cannot
  separate a hand from hair. Sound measurement needs a composited occluder
  with a known mask.
* **EAR is on 106 landmarks, not 68.** `landmark_3d_68` is lazy and absent in
  a real render; the formula is unchanged.
* **The skin mask is geometric**, never an edge percentile -- that artefact
  once produced a "36 pct of plate" texture reading that was really ~155 pct.

## NEXT

1. **Per-frame `verify_swap` metrics on s5**, to confirm or kill the
   spared-band hypothesis before any threshold moves.
2. **s3/s4 need a manual `--capture` frame** or they stay unverified.
3. **d1/d5 cannot be auto-captured at all** -- both people overlap throughout.
   They need seeded targets, the intake gap already recorded for
   persistently-overlapping footage.
4. **Longer windows** for `blink` and `colour`: no qualifying population in
   600 frames on any clip.
5. **Nothing here is measured on the 3060.**
