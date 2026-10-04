# Rotated-rescue aux work, and the r50 upscale rescue (2026-10-04)

Two proposed changes to `face_util._detect_faces`, each gated on a measurement. Reference: `baseline_2026-10-04.md`
(code `f9aaf96`). Change under test: `de63f34`. RTX 4070, `config.yaml` live, `ROOP_PROFILE=1`, 10 threads.

## 1. Skip `_rescue_upscaled` on retinaface_r50 — NOT done (the gate failed)

The gate was: skip it only if it never gains a face. `tests/probe_r50_upscale_gain.py` (engine forced to
`retinaface_r50`, which none of the baseline renders used) collects frames where r50's first pass is empty, from the
baseline clips' own windows, and calls `_rescue_upscaled` on each. Only **175** such frames exist in the four windows
(d1 and d6 gave none: r50 finds a face on every frame there), so the 200 asked for could not be reached.

| clip | r50-empty frames | upscale returned a face | scrfd also sees a face on the frame |
|---|---:|---:|---:|
| d4 (0..600) | 133 | **3** | 14 |
| Love (1100..1700) | 42 | **10** | 16 |
| d1, d6 | 0 | - | - |
| **total** | **175** | **13 (7.4%)** | 30 |

The rescue gained a face on 13 frames, so by the stated rule it stays. What the gains are
(`r50_upscale_gain_2026-10-04_sheet.jpg`, boxes drawn on the 13 frames): six Love kiss frames (1317, 1325, 1326, 1340,
1379, 1384) are large (235-425 px), partly occluded or profile faces — real faces on contact footage, and scrfd
misses all but one of them; Love 1569 and d4 337 sit on a face; Love 1608/1617/1622 are dark hair/back-of-head
regions (ambiguous, probably spurious); d4 308/309 are tiny boxes in the frame's top-left corner (spurious). So it is
doing real work on kiss footage and some harm elsewhere; that is a different question from "never gains". Nothing
was changed; the reasoning and numbers are in the comment at the call site, and `test_rotated_pass.py` pins that it
still runs on r50. A 175-frame sample bounds a true zero-gain rate only to about 1.7% (rule of three), but the
observed rate is 7.4%, so the bound is moot.

## 2. Rotated rescues run the aux models only on survivors — done

`_rescue_rotated` and the partial-miss rescue detected on a rotated frame with the aux models (recognition, 106-point
and 68-point landmarks) and threw away the aux work of every duplicate. `face_util._rotated_pass` now:
detects with `aux=False`; tests duplicates on an un-rotated **copy** of the coordinates (same order, same `known`
list as before); runs the aux models on the **rotated frame from the rotated-space keypoints** for survivors only;
then un-rotates them. One addition to what was specified: detection passes `unclamped=True`, because SCRFD's
`aux=False` path clamps boxes and keypoints to the frame (via `_hybrid_detector_faces`) while the old `fa.get()` path
did not, and an edge face would otherwise get different geometry and a different crop.

### Equal to the baseline? Yes, bit for bit

**Function level** (`tests/ab_rescue_aux.py`): the old `face_util.py` (`git show f9aaf96`) and the new one run in one
process on the same real frames, `_detect_faces(frame, expected_count=2)`, order alternated per frame. 1,018 frames
(d4 and Love every 2nd frame, d1 all) → 1,558 faces compared in order:

| | frames | faces | different face count | max bbox / kps / det_score diff | min embedding cosine | max lm106 / lm68 diff | exactly equal |
|---|---:|---:|---:|---|---:|---|---:|
| **null: old vs old** (second call, another pooled context) | 1018 | 1558 | 0 | 0 / 0 / 0 | 1.0000000 | 0 / 0 | 1558 |
| **old vs new** | 1018 | 1558 | 0 | 0 / 0 / 0 | 1.0000000 | 0 / 0 | **1558** |

The acceptance bar was cosine >= 0.9999; the faces are identical, not just close, and the null control shows the
TensorRT contexts themselves add no noise on this path.

**Whole render** (`tests/ab_rescue_renders.py`): d4 and Love ABBA (old, new, new, old) after a discarded warm-up, d1
on the new code only; `ROOP_STAB_CHUNK_MB` pinned to the baseline's value so pixels are comparable.

| clip | decoded md5 | = recorded baseline md5 | swap audit | per-person verdicts incl. WRONG FACESET | rows.csv |
|---|---|---|---|---|---|
| d4 | identical in all 4 arms | yes (`b37f2bebe675`) | identical | identical (0 wrong) | identical |
| Love | identical in all 4 arms | yes (`87f08da47b22`) | identical | identical | identical |
| d1 | new arm only | yes (`bfb057030413`) | identical to baseline run | identical | identical |

### Fewer aux calls (model-level `get` counters; detector executions unchanged)

| scope | aux calls per model, old -> new | change |
|---|---|---|
| function level, 1,018 frames, expected_count=2 forced | 2718 -> 2233 | **-17.8%** (d4 -35%, Love -36%, d1 -3.5%) |
| whole render d4 (pre-pass) | 1271 -> 838 | **-34.1%** (detector executions 2805 = 2805) |
| whole render Love | 721 -> 691 | -4.2% |
| whole render d1 | 1608 (baseline run) -> 1451 (new) | -9.8% (no old arm this session) |

The old arm of the d4 render reproduces the baseline's 1271 exactly, so the counters line up across sessions. The
render-level saving is smaller than the function-level one on Love because the real pre-pass only passes
`expected_count` on some calls, whereas the harness forced 2 on every frame.

### Pre-pass speed

Pre-pass time is the driver's timestamps on the `temporal-prepass-start/complete` markers (600 frames).

| clip | old arms | new arms | pre-pass fps old -> new |
|---|---|---|---|
| d4 | 19.73 s, 19.96 s | 17.36 s, 17.58 s | **30.2 -> 34.3 (+13.6%)**; arms agree to 0.2 s, ABBA balanced |
| Love | 14.67 s, 24.56 s | 15.92 s, 24.70 s | **not measurable**: the machine slowed from ~15 s to ~25 s between arm 2 and 3, which swamps any difference (and the aux saving is only 4%) |
| d1 | - | 41.48 s (10.1 fps) | no old arm |

The frame loop is untouched by design: d4 15.23 -> 15.34 fps (+0.8%, noise). Per AGENTS.md the pre-pass is a
fraction of the render; this removes work in it, it does not change the loop that gates throughput.

### Limits

d6 was not re-rendered: its baseline counters show **no** partial-miss or rotated-rescue activity at all (its
pre-pass detections are `_upright_remeasure`, which this change does not touch), so there is nothing of this change
to compare. Only the main machine (RTX 4070) was measured. Love's timing was disturbed by a mid-run slowdown whose
cause was not isolated. The function-level comparison uses `expected_count=2` on every frame (more rescue activity
than a real render has), so its percentages are an upper bound for the saving on these clips.

Files: `rescue_aux_ab_2026-10-04.json`, `rescue_aux_renders_2026-10-04.json`, `r50_upscale_gain_2026-10-04.json`.
