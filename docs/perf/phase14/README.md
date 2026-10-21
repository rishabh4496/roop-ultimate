# Phase 14 — pre-pass detection interval, 1-euro landmark filter, trajectory cache (2026-10-21)

Everything below was measured on the RTX 4070 machine, on real clips, with the tools in this repository. The raw
numbers are the JSON files in this folder; `tables.md` is generated from them by
`python app/tests/probe_phase14_prepass.py report`, so no figure here is typed in by hand. Nothing was measured on the
RTX 3060.

| file | what |
|---|---|
| `interval_final_<clip>.json`, `tables.md` | one real pre-pass per K on a 900-frame window of each clip |
| `filter_final_<clip>.json` | the landmark filters on the tracks captured from those pre-passes |
| `flow_backends.json` | the point-tracking back ends, timed |
| `e2e_Weeds.json` | end-to-end renders (pre-pass, swap, enhance, encode) per arm |
| `interval_Weeds_explain.json` | which frames lost their source binding at K=4 and K=2 |
| `interval_*.json` (without `final`), `interval2_*.json`, `filter_Weeds.json` | earlier runs, kept: they are the evidence for the defects below |

Reproduce:

```
app\env\Scripts\python.exe app\tests\probe_phase14_prepass.py interval --clip <clip.mp4> --ks 1,1,2,4,8 --frames 900 --out docs\perf\phase14\interval_final_<clip>.json
app\env\Scripts\python.exe app\tests\probe_phase14_prepass.py filter <captured_tracks.npz> --fps 23.976
app\env\Scripts\python.exe app\tests\probe_phase14_prepass.py flow
app\env\Scripts\python.exe app\tests\ab_phase14_e2e.py --clip <clip.mp4> --frames 600 --arms base,base,euro,k4,k4,euro,base
app\env\Scripts\python.exe app\tests\probe_phase14_prepass.py report
```

The app must be stopped. `interval` stops each run the moment the pre-pass returns, so no swap pass is rendered.

## Verdicts

1. **`prepass_detect_interval` (K) works and is not a speedup you want by default.** It cuts the pre-pass by 11-14% (K=2),
   21-26% (K=4) and ~30% (K=8) and the faces it places by optical flow land within ~1% of an eye distance (median), 1.5-10x
   closer than linear interpolation. It also finds fewer faces and binds fewer of them to the source, and end to end it
   swaps fewer faces (below). The default stays 1; the slider (1-8) is opt-in and its text quotes these numbers.
2. **The zero-phase 1-euro filter works and is indistinguishable end to end.** On real tracks it matches or beats the
   shipped smoother on still jitter and removes its lag (shipped: 1-5% of an eye distance behind a fast head); on 600
   rendered frames its pixels differ from the baseline by exactly as much as two renders of the baseline differ from each
   other. The default stays `adaptive`; `one_euro` is an option.
3. **The point-tracking ladder:** the CPU `cv2` call takes 0.14-0.23 ms for the flow chain's real workload. A torch GPU
   Lucas-Kanade is built and verified but is ~90x slower there (kernel-launch bound), and the NVIDIA optical-flow ASIC rung
   cannot run on this machine (this OpenCV has no CUDA). `auto` therefore stays on the CPU call.
4. **The trajectory cache** (M, M_inv, pose per face per frame, memory-mapped, attached to the project checkpoint, read by
   the head-pose HUD) is built and tested. Stage 3 does **not** read its matrices, on purpose (see the module docstring).

## 1. Detection interval K

Reference = the second of two dense (K=1) runs; **the two K=1 runs are bit-identical** (the null control: same faces,
tracks and plan frames, pre-pass time within 0.7 s). "Frames matched to a source" is the Stage 2 plan: frames holding a
face bound to a source — the frames that can actually be swapped.

**Love** (900 frames, reference = the second dense run; the two dense runs are bit-identical)

| K | pre-pass | detector calls / frame | faces found | frames with a face | frames matched to a source | flow-placed error median / p95 (% eye dist.) | linear interpolation median | placed without flow: n, median |
|---|---|---|---|---|---|---|---|---|
| 1 | 23.8 s (+0%) | 1.00 | 1667 (+0%) | 826 / 826 | 621 (+0%) | - | - | - |
| 2 | 21.1 s (-11%) | 0.54 | 1583 (-5%) | 819 / 826 | 603 (-3%) | 1.22 / 11.1 | 3.94 | 193, 25.7 |
| 4 | 18.3 s (-23%) | 0.31 | 1497 (-10%) | 802 / 826 | 571 (-8%) | 1.15 / 9.2 | 6.51 | 315, 28.9 |
| 8 | 17.0 s (-29%) | 0.19 | 1341 (-20%) | 755 / 826 | 431 (-31%) | 1.28 / 9.8 | 12.67 | 306, 39.5 |

**MonicaBellucci** (900 frames, reference = the second dense run; the two dense runs are bit-identical)

| K | pre-pass | detector calls / frame | faces found | frames with a face | frames matched to a source | flow-placed error median / p95 (% eye dist.) | linear interpolation median | placed without flow: n, median |
|---|---|---|---|---|---|---|---|---|
| 1 | 24.4 s (+0%) | 1.00 | 2306 (+0%) | 900 / 900 | 872 (+0%) | - | - | - |
| 2 | 21.0 s (-14%) | 0.51 | 2258 (-2%) | 896 / 900 | 870 (-0%) | 1.07 / 11.8 | 3.34 | 223, 27.6 |
| 4 | 18.2 s (-26%) | 0.26 | 2149 (-7%) | 887 / 900 | 864 (-1%) | 1.20 / 14.1 | 4.48 | 379, 19.7 |
| 8 | 17.0 s (-30%) | 0.14 | 1980 (-14%) | 873 / 900 | 770 (-12%) | 1.41 / 11.7 | 6.36 | 427, 19.4 |

**Weeds** (900 frames, reference = the second dense run; the two dense runs are bit-identical)

| K | pre-pass | detector calls / frame | faces found | frames with a face | frames matched to a source | flow-placed error median / p95 (% eye dist.) | linear interpolation median | placed without flow: n, median |
|---|---|---|---|---|---|---|---|---|
| 1 | 17.1 s (+0%) | 1.00 | 1221 (+0%) | 900 / 900 | 719 (+0%) | - | - | - |
| 2 | 15.1 s (-12%) | 0.51 | 1225 (+0%) | 888 / 900 | 680 (-5%) | 0.80 / 13.1 | 1.22 | 88, 20.1 |
| 4 | 13.6 s (-21%) | 0.26 | 1187 (-3%) | 880 / 900 | 541 (-25%) | 0.89 / 14.8 | 1.80 | 125, 14.6 |
| 8 | 11.7 s (-31%) | 0.13 | 1115 (-9%) | 868 / 900 | 718 (-0%) | 1.14 / 17.8 | 3.64 | 165, 13.1 |

Reading it:

- **Time saved is real but small against the job.** The pre-pass is ~25% of a 900-frame render, so K=4's 21-26% of it is
  ~5-6% of the job.
- **Flow is accurate where it holds.** Median 0.8-1.4% of the eye distance for the frames flow placed, against 1.2-12.7%
  for straight linear interpolation between the same detections (what `temporal_step` ships). The p95 (9-18%) is the hard
  frames: fast turns and the first frames after a cut.
- **Where flow could not follow, the fallback is bad.** 13-40% of an eye distance at the median, 88-427 faces per run.
  Those are faces the detector would have seen at K=1.
- **Recall and source binding fall with K.** Faces found: 0% to -5% (K=2), -3% to -10% (K=4), -9% to -20% (K=8).
  Frames matched to a source: -0% to -5% (K=2), -1% to -25% (K=4), -0% to -31% (K=8). The loss is not monotonic in K on a
  single clip because a track's binding to the source is a gate on its mean embedding (and a rescue by 3 individually
  matching frames, `_TRACK_ASSIGN_MIN_OBS`): with fewer observations a hard-pose target can miss the rescue and lose a
  whole stretch. `interval_Weeds_explain.json` lists the stretches (K=4 on Weeds: frames 182-194, 287-351, 453-493,
  611-711, where the same people are still tracked but no longer bound).

## 2. Defects found on the way (each has a test that fails without the fix)

- **The interval was ignored under the pool's lag.** `TemporalFaceTracker` decides a frame's detection at plan time but a
  result is consumed up to pool_workers + 2 frames later, and consuming frame t-4 moved "the last detection" back to t-4;
  the next plan saw a gap of K and detected again. K=4 ran **900 detector calls over 900 frames with 0 coast frames**
  (Weeds, first sweep). Under K>1 the marks now only move forward (K=1 is untouched, bit for bit).
  `test_prepass_interval.py::test_the_old_scheduler_ignored_the_interval_under_lag` reproduces it.
- **The last K-1 frames of every track had no face.** Gap-fill bridges only *between* two detections, and a track's final
  detection has none after it. Optical flow, which kept stepping, now carries the face up to K-1 frames past the last
  detection, never past a cut and only while the chain stays consistent (`_flow_tail`). A pooled K=4 run lost the last
  three frames of the clip before this.
- **The flow chain waited for 3 observations**, i.e. 2K frames, so the first two gaps of every track were bridged by
  geometry alone. At K>1 one observation is enough (the chain's own forward-backward, rigid-fit and arrival checks keep a
  bad one out).
- **A track of K>=5 was declared lost between two key frames** (`max_misses` 3); it now grows with K (`2K-1`).
- **A claim of mine was wrong and a test caught it:** the closed-form matrices agree with `face_util.estimate_norm` to
  5e-5 px, not 1e-9. `estimate_norm` itself moves by 2.1e-5 between a float32 and a float64 call of the *same* keypoints
  (skimage keeps float32 intermediates). Docstring and test now say so.
- **`temporal_step` is a different thing.** It skips frames outright and interpolates (measured 6x worse landmark error on
  turned heads). `prepass_detect_interval` reads every frame's pixels; both settings remain and each says which it is.

## 3. Landmark filter

Synthetic fixture (`app/tests/data/synthetic_face_jitter.npz`: a known head path through the real detector, x264,
sensor noise), the same fixture that scores the shipped smoother and the Kalman option:

| smoother | static >2 Hz | static >4 Hz | sway lag | head-turn peak | rms |
|---|---|---|---|---|---|
| raw detector | | | | 1.41 px | 0.90 px |
| adaptive (shipped) | -67.4% | -73.8% | 1.00 f | 2.30 px | 1.94 px |
| Kalman (option) | -77.0% | -81.0% | 0.00 f | 1.79 px | 1.01 px |
| zero_phase EMA (Phase 7) | -72.8% | -76.7% | 0.00 f | 4.80 px | 1.16 px |
| **one_euro 0.6 / 25 / 3 (default)** | **-90.9%** | **-95.2%** | **0.00 f** | **2.59 px** | **0.83 px** |
| one_euro 0.8 / 50 / 3 (more open) | -82.1% | -89.3% | 0.00 f | 1.58 px | 0.75 px |

Real tracks, captured from the K=1 pre-pass of each clip (the fixture alone misleads: its best setting, beta 50, is no
better than the shipped filter on a still face in real footage, because detector noise in the speed estimate opens the
cutoff):

| clip | arm | tx px | ty px | eye-line deg | scale % | lag along motion (% eye dist.) | deviation from raw in fast motion (% eye dist.) | 106-landmark jitter px | speech kept |
|---|---|---|---|---|---|---|---|---|---|
| Weeds (150 still / 617 fast frames) | raw detector | 0.541 | 0.374 | 0.263 | 0.605 | - | - | 1.147 | 1.000 |
| Weeds | adaptive (shipped) | 0.285 | 0.220 | 0.093 | 0.194 | -1.12 | 1.62 | 0.513 | 0.973 |
| Weeds | zero_phase | 0.173 | 0.123 | 0.069 | 0.149 | -0.04 | 1.17 | 0.322 | 0.954 |
| Weeds | one_euro (0.6 / 25 / 3) | 0.236 | 0.163 | 0.087 | 0.179 | -0.02 | 1.26 | 0.408 | 0.960 |
| Love (17 still / 1099 fast frames) | raw detector | 0.554 | 0.611 | 0.488 | 0.805 | - | - | 1.558 | 1.000 |
| Love | adaptive (shipped) | 0.318 | 0.318 | 0.141 | 0.216 | -4.77 | 11.30 | 0.535 | 0.974 |
| Love | zero_phase | 0.226 | 0.207 | 0.116 | 0.188 | +0.87 | 7.84 | 0.435 | 0.969 |
| Love | one_euro (0.6 / 25 / 3) | 0.316 | 0.289 | 0.110 | 0.131 | -0.11 | 2.04 | 0.507 | 0.979 |
| Monica (180 still / 1363 fast frames) | raw detector | 0.436 | 0.470 | 0.277 | 0.461 | - | - | 1.054 | 1.000 |
| Monica | adaptive (shipped) | 0.245 | 0.279 | 0.164 | 0.198 | -3.12 | 4.47 | 0.508 | 0.961 |
| Monica | zero_phase | 0.131 | 0.138 | 0.081 | 0.126 | -0.04 | 2.40 | 0.294 | 0.943 |
| Monica | one_euro (0.6 / 25 / 3) | 0.161 | 0.178 | 0.095 | 0.128 | +0.02 | 1.60 | 0.314 | 0.950 |

Love has only 17 still frames: read its jitter columns as indicative. The three filters trade differently: zero_phase is
calmest on a still face but still smears a fast turn (7.8% off the raw track on Love); one_euro follows fast motion to
1.3-2.0% with no lag and sits between zero_phase and the shipped filter on jitter. Lower beta smooths a still face harder
and follows fast motion less well (0.6 / 12 / 3: still jitter about -20%, deviation in fast motion about x1.7).

The filter is a **parallel scan**, not a loop: the speed that sets the cutoff is read from the raw keypoints, so every
alpha is known before smoothing starts and the whole filter is a linear time-varying recurrence, done in ceil(log2 T)
vectorized passes on numpy or a torch tensor (all coordinates, runs and the 5 / 106 / 68 landmark sets at once; one
speed per frame shared by all of them, so the crop and its outline cannot disagree). CPU 0.1-12 ms for a typical
track; the GPU only wins past ~140k scalars (T x D) and is 14x faster at 60,000 frames (4.3 s vs 0.3 s). For a 300-frame
track either device costs milliseconds, so this is not a render-time item.

## 4. Point-tracking back ends (ms per call, median)

| crop px | points | cv (CPU) | torch (CPU) | torch (CUDA) |
|---|---|---|---|---|
| 96 | 40 | 0.14 | 15.9 | 17.4 |
| 192 | 70 | 0.23 | 26.5 | 20.3 |
| 192 | 300 | 0.30 | 83.4 | 22.1 |
| 384 | 1000 | 0.66 | 140.3 | 21.3 |
| 768 | 3000 | 2.34 | 295.4 | 21.5 |

The chain's real call is the 70-point, <=192 px row: 0.23 ms on the CPU. The torch rung is launch-bound (about 20 ms on
CUDA at any size here) and does not cross over below 3000 points at 768 px. It is verified (sub-pixel against ground truth,
agrees with OpenCV, refuses textureless and out-of-frame points, drives the full `FlowCoaster` chain) and left as
`ROOP_FLOW_BACKEND=torch`. **The NVOF rung is UNVERIFIED on hardware**: this environment's OpenCV has no CUDA
(`nvof_unavailable_reason()`), so only its dispatch and sampling are tested, against a fake engine.

## 5. End to end (Weeds, first 600 frames, realswap + GPEN Realistic, 20 threads)

Counterbalanced (base, base, one_euro, K=4, K=4, one_euro, base); `px diff` is the mean absolute pixel difference
(0-255) of each arm's output against the first baseline render, every 5th frame.

| arm | seconds | swapped faces | swap % | plan frames | px diff vs base |
|---|---|---|---|---|---|
| base | 70.0 / 68.1 / 73.4 (mean 70.5) | 469 / 469 / 469 | 59.0 | 473 | null: 0.069 / 0.049 |
| one_euro | 68.8 / 70.1 (mean 69.5) | 469 / 469 | 59.0 | 473 | 0.069 / 0.081 |
| K=4 | 64.1 / 65.5 (mean 64.8) | 424 / 424 | 53.3 | 439 | 0.554 / 0.563 |

K=4 is 8% faster and **swaps 9.6% fewer faces** ("a speedup that moves the swap rate down is not a speedup",
AGENTS.md). one_euro is within the 5% timing spread and its pixels are within the null difference of the baseline: no
end-to-end effect was measurable on this window.

## 6. K=1 is unchanged

The K=1 pre-pass of the pristine `HEAD` (a `git archive` copy) and of this tree, same clip and window: all 12 captured
arrays (faces, raw and smoothed keypoints, both landmark sets, flags, cuts, track ids) are **bit-identical**, in the same
16.9 s.

The regression benchmark (`python run.py --benchmark --benchmark-mode regression --benchmark-threads 20`, default
configuration: K=1, `adaptive`) **passes** on this tree: swap decided 300/300 and changed 300/300, face SSIM 0.9906
(min 0.9868) and face PSNR 46.07 dB against the golden render, 6.10 fps against the recorded baseline's 2.71.

## 7. Not done / unverified

- RTX 3060 (every number above is the 4070).
- The full Python suite is green except one test that was already red on `HEAD` in a full run:
  `test_gpu_filmstrip_and_scene_cuts::test_sparse_pre_scan_hardware_execution` fails after `test_7_boost_fixes`, which
  leaves `roop.globals.g_desired_face_analysis = 'all'` set (insightface then rejects the detector); it passes alone, and
  none of this phase's test files cause it.
- NVOF on an OFA; the torch rung was timed only on the 4070.
- One window per clip (900 frames; 600 end to end) and three clips; Love has few still frames.
- No render through `/api/swap` with a project (the checkpoint save/restore is tested against the real project record,
  and the pre-pass hand-off by source and test, not by a resumed render).
- The regression benchmark was run for the default configuration only (it passes); not with K>1 or `one_euro`.
- No fix for the identity loss at K>1 (it would mean changing the assignment gate, which this project has reverted five
  times on a measured population that was not the one the change targeted). The open follow-up this suggests: re-detect,
  in a second mini-pass, only the bridged frames whose flow chain was refused (15-30% of the bridged faces here).
