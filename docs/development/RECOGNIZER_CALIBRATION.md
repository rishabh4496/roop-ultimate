# Recognition model calibration: glintr100 vs w600k vs AdaFace (2026-10-03)

Question: should glintr100 (Glint-R100, registry key `glintr100`) be wired into live identity matching, and
what distance threshold would it need? Tool: `app/tools/calibrate_recognition.py`; raw summaries in
[`recognizer_calibration_2026-10.json`](recognizer_calibration_2026-10.json).

**Answer: no. glintr100 never beat the model in production, and was the weakest in every cut of the data.
The margin is small; the evidence is "never better, consistently a little worse", not "decisively worse".**
A byproduct matters more than the question: the pipeline's keypoint refinement degrades the crops that every
non-w600k recogniser is fed, including AdaFace in production (see "Keypoints").

## Method

* **Footage:** 16 clips from `roop-keep` (6 two-person `d1-d6`, 7 single-person `s1-s7`, `Love`, `Monica
  Bellucci`, `Weeds`); 5 windows x 30 samples per clip, one sample every 6 frames, faces under 48 px dropped.
* **Detector:** initialised the way the user's config runs it (`tests/angle_bench.init_pipeline`, `sync_config`:
  TensorRT, SCRFD, `refine_landmarks: true`).
* **Same person = a track.** Faces in consecutive samples are linked only when they are each other's best box
  (IoU >= 0.5) with no rival above 0.2; a scene cut or a new window breaks every link. Every pair inside a track
  counts (up to 60 per track), so pairs seconds apart are included. 3,836 pairs.
* **Different people = two faces in one frame** whose boxes are not duplicates (IoU <= 0.35, the pipeline's own
  duplicate rule; touching heads sit near 0.2 and stay in). 1,183 pairs, 61% of them from `Monica Bellucci`.
  Cross-clip pairs are never used: nothing proves two clips hold different people.
* **Models:** `default` is w600k as the pipeline produces it (detector-time alignment). Every other model embeds
  the same aligned crop (`align_crop`, `arcface_112_v2`) through `RecognitionInferenceEngine` on CUDA (FP32).
  `default@crop` is w600k on that same crop: the control that separates "the model" from "the alignment path".
* **Comparison is rank-based** (AUC, equal-error rate, false-accept rate at a fixed false-reject rate), because a
  distance scale is arbitrary. Intervals come from resampling whole **clips**: pairs inside a track are strongly
  correlated, so a pair-level interval would be far too narrow.

## Results

Production keypoints (`refine_landmarks: true`, the real condition):

| Model | AUC | EER | FAR at w600k's FRR (2.58%) | threshold for that FRR |
| :--- | ---: | ---: | ---: | ---: |
| `default` (own alignment) | 0.9905 | 3.13% | 5.16% | 0.748 |
| `default@crop` | 0.9859 | 3.63% | 19.95% | 0.836 |
| `adaface` | 0.9830 | 3.53% | 19.02% | 0.829 |
| `glintr100` | 0.9806 | 4.13% | 31.02% | 0.868 |

Detector's raw keypoints (`--raw-kps`):

| Model | AUC | EER | FAR at w600k's FRR (2.58%) | threshold for that FRR |
| :--- | ---: | ---: | ---: | ---: |
| `default` (own alignment) | 0.9905 | 3.13% | 5.16% | 0.748 |
| `default@crop` | 0.9898 | 3.32% | 8.88% | 0.748 |
| `adaface` | 0.9883 | 3.06% | 8.45% | 0.748 |
| `glintr100` | 0.9842 | 3.71% | 18.43% | 0.819 |

Against `default@crop` on identical crops (15 clips with pairs, 1000 clip resamples, 95% interval):

| Keypoints | Model | dAUC | dEER (pp) |
| :--- | :--- | :--- | :--- |
| raw | `glintr100` | -0.0055 (-0.0135, -0.0011) | +0.44 (-0.36, +1.34) |
| raw | `adaface` | -0.0015 (-0.0103, +0.0015) | -0.22 (-0.78, +0.24) |
| refined | `glintr100` | -0.0048 (-0.0130, -0.0006) | +0.51 (-0.00, +1.22) |
| refined | `adaface` | -0.0028 (-0.0138, +0.0015) | -0.08 (-0.44, +0.51) |

* **glintr100** ranks same-person pairs ahead of different-person pairs slightly worse than w600k: AUC interval
  entirely below zero, better in 0-2% of resamples. Its EER difference is within noise.
* **AdaFace** is not distinguishable from w600k on identical crops, in either metric.

### Sensitivity: dropping suspect clips

`Love`, `s1` and `s7` show a same-person 95th percentile of 0.84-0.92 for **every** model. A recogniser failure
would not hit all models equally, so this is label noise (a track linked across a cut or an occlusion), not model
error. Without those three clips every model improves to AUC 0.996-0.999 / EER 0.7-1.6%, i.e. that noise had
inflated all the EERs three- to four-fold. The ranking is unchanged but the margin gets thinner:

| Keypoints | `glintr100` vs `default@crop` (12 clips) | dAUC | dEER (pp) |
| :--- | :--- | :--- | :--- |
| raw | | -0.0025 (-0.0072, +0.0001) | +0.50 (+0.00, +1.16) |
| refined | | -0.0025 (-0.0060, -0.0001) | +0.52 (+0.00, +1.10) |

So with the noisy clips removed the AUC gap is borderline-significant at best. This is the honest summary:
glintr100 was never better in any cut, and no cut shows it clearly worse by a margin that would matter on its own.
It also costs ~7.6x the CPU time (313 vs 41 ms per face) and ~1.7x the TensorRT time (2.43 vs 1.44 ms) of w600k
(`tests/test_recognition_pipeline.py --benchmark`), so "not better" is enough to decline.

### If it were wired anyway

On the production (refined) crops a glintr100 distance threshold that keeps w600k's 2.58% false-reject rate is
**0.868**, at a 31% false-accept rate (w600k today: 5.2%). On raw-keypoint crops it is **0.819** at 18.4%. The
equal-error thresholds are 0.712 / 0.704 (all clips) and 0.625 / 0.626 (clean subset). None of this has been
checked against the ratio-rescaled gates (`recognizer_adaface.scale()`), which an earlier AdaFace calibration
showed can land inside the population they protect (commit 5fa4001: a ratio-rescaled floor refused the target's own
track); a per-track p0 check against the rescaled gates would be required before any wiring.

## Keypoints: refinement hurts recognition crops (applies to AdaFace in production)

With `refine_landmarks` on, the pipeline replaces the detector's 5 keypoints with ones derived from the 68
landmarks (`face_util._refine_kps_from_68`) **after** buffalo_l has already embedded on the detector's keypoints.
That is why `default` (5.2% FAR) beats `default@crop` (20% FAR) on identical footage: its crop never sees the
refined points. Every recogniser that aligns from the final `face.kps` reads the worse crop, including AdaFace
(`recognizer_adaface.face_embedding` aligns from `face.kps`).

Replacing refined keypoints with the detector's raw ones, same clips and same pairs, 1000 clip resamples:

| Model | AUC refined -> raw | dAUC (95% CI) | EER refined -> raw | dEER pp (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| `default@crop` | 0.9859 -> 0.9898 | +0.0039 (+0.0014, +0.0091) | 3.63% -> 3.32% | -0.35 (-1.21, -0.09) |
| `adaface` | 0.9830 -> 0.9883 | +0.0052 (+0.0013, +0.0129) | 3.53% -> 3.06% | -0.54 (-1.70, -0.05) |
| `glintr100` | 0.9806 -> 0.9842 | +0.0033 (+0.0006, +0.0077) | 4.13% -> 3.71% | -0.45 (-1.28, +0.00) |

At w600k's false-reject rate AdaFace's false-accept rate falls from 19.0% to 8.5%. **Not changed in this commit:**
the refined keypoints exist for swap-alignment stability and must stay for the swapper. The follow-up would be to
keep the detector's keypoints beside them (e.g. `kps_det`) and align recognition crops from those, then re-run
the regression benchmark and the per-track p0 table. That touches production matching, so it needs its own decision.

## What this does and does not show

* 16 clips, one machine, one detector configuration. Diverse (4K to 480x360, two-person, movie footage) but not a
  population; intervals describe clip-to-clip variation in THIS set.
* Same-person labels are inferred, not hand-checked, and carry noise (above). Different-person labels are
  reliable. Single-person clips contribute no different-person pairs.
* Models ran FP32 on CUDA. TensorRT FP16 moves embeddings by < 0.0001 cosine (Stage 2 measurement), far below
  these differences, but it was not re-measured here.
* Not measured: the RTX 3060, `mobilefacenet`, `facerecognizersf`, `antelopev2` (same file as glintr100).

## Reproduce

```
python tools/calibrate_recognition.py --models default,default@crop,adaface,glintr100 \
    --models-dir <dir with the models> --windows 5 --per-window 30 --step 6 \
    --clips <roop-keep>/double <roop-keep>/single <Love.mp4> <Monica.mp4> <Weeds.mp4> [--raw-kps] --out run.json
```
Run it when the GPU is otherwise idle. Unit tests for the analysis: `app/tests/test_calibrate_recognition.py`.
