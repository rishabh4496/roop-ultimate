# Per-model TensorRT precision for XSeg / w600k_r50 / 2d106det / 1k3d68 (2026-10-10)

Brief, four items: (1) what XSeg really runs under in a TensorRT render, and whether its NHWC input adds transposes; (2) mixed vs FP32 on 500 real
faces for xseg, w600k_r50, 2d106det, 1k3d68; (3) swap_canary-style startup checks cached by engine hash, and precision per model instead of the one global
`trt_precision`; (4) an NCHW XSeg variant, only if (1) shows transposes.

Hardware: RTX 4070 (MAIN), TensorRT 10.9.0.34, ORT 1.23.2, driver 616.56. The 3060 was not touched (TensorRT is not admitted there, so every precision arm is
inert and the canary is skipped).

## Result in one table

| model | shipped before | mixed vs FP32 (500 real faces) | TRT FP32 vs FP32 | FP32 cost / call | now |
|---|---|---|---|---|---|
| xseg | mixed | IoU mean 0.9936, **min 0.81**, 162 faces < 0.995, 6 < 0.95; boundary mean 0.16 px | IoU **min 0.9997**, 0 < 0.995 | +0.60 ms (2.23 -> 2.82) | **fp32** |
| 2d106det | mixed | mean 0.12 px, p95 0.61, max 1.25 (frame px); 48 faces > 0.3 px | 0.03 px (the engine-to-engine floor); shipped chain 0.000 | +0.02 ms (0.50 -> 0.52) | **fp32** |
| 1k3d68 | mixed | mean 1.19 px, p95 4.5, max 8.8; **refined kps** mean 0.94 px, p95 3.6 px = **10.5% of the inter-ocular distance** | 0.37 px (= floor); refined kps mean 0.29, p95 1.1 px = 3.0% IOD | +1.05 ms (0.72 -> 1.77) | **fp32** |
| w600k_r50 | mixed | cosine min **0.999916**, 0 faces < 0.999 (gate 0.999) | 1.000000 | +1.08 ms (0.95 -> 2.04) | mixed (unchanged) |

Gates are the ones earlier briefs set (`stage_precision_verify_lighting_2026-10-05.md`, `aux_batch_2026-10-05.md`): xseg IoU >= 0.995 on >= 99% of faces and none
< 0.95; embedding cosine >= 0.999 on every face; landmark p95 <= 0.3 px. Mixed fails the first and third by 2-12x and passes the second. Reference = CUDA EP, TF32
off, the shipped FP32 ONNX (asserted: no float16 initialisers). Full tables: `trt_precision_fidelity.md`; live-session confirmation:
`trt_precision_shipped_check.md`.

**The three FP32 moves cost 0.6 ms (xseg), 0.02 ms (2d106det) and 1.05 ms (1k3d68) per call. They are a decision, not a fact:** one table line each in
`precision_policy.PRECISION_OVERRIDES`, or `ROOP_TRT_MODEL_PRECISION="xseg:mixed"` for one run. The 10-05 brief left "move XSeg / RF++ to FP32?" to the
owner; this brief asked for per-model precision with these numbers in hand, so the three that fail their gate by a wide margin and cost little are moved, and RF++
(not in this brief) is untouched.

## Item 1 - what XSeg really runs under, and the NHWC input

`tests/trt_precision_probe.py` (writes `trt_precision_probe.json`, mixed; `trt_precision_probe.py shipped` writes `trt_precision_probe_shipped.json`). Facts come
from the sessions the app's own loaders build (`Mask_XSeg.Initialize`, the buffalo_l bundle), not from the request:

- Provider `TensorrtExecutionProvider` first, CUDA, CPU behind it. ORT profiling of the same chain: **exactly one `TensorrtExecutionProvider` node per run** for
  every one of the four models - the whole graph is one engine, no CUDA/CPU partition boundary.
- Under the old default: `trt_fp16_enable=1`, `trt_layer_norm_fp32_fallback=1`, heuristics 1, builder level 3, sequential build, 2 GiB workspace, cache
  `mixed_..._lnfp32_seq_heur_...`. A DETAILED-verbosity copy of the XSeg graph built with the same flags has 387 layers: output formats Half 393 / Float 10 / Int32 5.
  "Mixed" is therefore FP16 almost everywhere with ten FP32-format outputs (the input transpose and the fused LayerNorm-fallback groups), all inside one
  TensorRT engine - the deviation measured in item 2 is not a CUDA/CPU fallback. (Which FP16 layers cause the worst faces was not isolated.)
- **NHWC input:** XSeg's input is `xseg_input:0` `[N,256,256,3]`. The engine has **one** `Shuffle` layer for it, `Conv2D_18__6`, in both precisions:
  0.0126 ms of a 2.53 ms call (0.5%) mixed, 0.0137 ms of 3.17 ms (0.4%) FP32; plus the input `Reformat` feeding the first conv, 0.0111 ms (FP32). The other
  `Reformat` layers (55 mixed = 0.61 ms, 42 FP32 = 0.48 ms) are precision / layout copies between fused groups, not NHWC transposes. The output is `[N,256,256,1]`
  has a trailing `Conv2D_48__128` layer that times at 0.0 ms.
- Caveat: layer times are from the DETAILED diagnostic copy (the cached ORT engine is built at default verbosity and carries names only); wall time of that copy
  tracks the live engine (2.53 vs 2.23 ms mixed, 3.17 vs 2.82 FP32, different timing harness).

## Item 2 - mixed vs FP32 on 500 real faces

`tests/trt_precision_fidelity.py capture | evaluate | report`. 100 faces from each of d4, d1, d6, Love, s7; the three buffalo models' inputs are the exact blobs
production fed (a wrapper on `session.run` during `get_all_faces`), XSeg's are the aligned arcface-256 crops (and, second population, the unwarped bbox crops).
Metrics: XSeg IoU at 0.5, boundary distance both directions (mean / p95 / max px), the soft keep-mask the compositor really uses; cosine of the 512-d embeddings;
landmark error in **frame** px (crop error x (1.5 x face size / 192)) and, for 1k3d68, the five kps `_refine_kps_from_68` writes into `face.kps`.

- The mixed arm is deterministic (same inputs twice: max abs difference 0 for all four), so the errors are tactic/precision, not noise. A TF32 control
  (`app/output/trt_precision_fidelity/tf32_results.json`, not committed) shows the reference itself is exact: TRT FP32 vs CUDA FP32 with TF32 off is IoU min 0.99994, cosine min 0.9999998.
- **1k3d68 is the one that matters, as the brief said.** `refine_landmarks` replaces the 5 kps from it, and those kps align the crop for the swap net and the
  AdaFace crops. Mixed moves them by 0.94 px mean and 3.6 px at p95 (10.5% of the inter-ocular distance); FP32 moves them 0.29 px, which equals the
  engine-to-engine floor (TRT FP32 vs CUDA FP32), i.e. there is nothing left to recover.
- XSeg's mixed error is not mainly boundary jitter: the per-face mean boundary error is 0.16 px (p95 0.46 px), but the single worst pixel across the set is 124 px
  from the other mask - a missing or extra blob. 6 (aligned) / 8 (box) faces fall under IoU 0.95.
- 2d106det: the harness's FP32 arm went through the full `providers_for`, which attaches a dynamic-batch shape profile to a 'None'-batch graph (0.03 px). The
  shipped bundle seam applies the precision step only (below) and lands at 0.000 px on the same faces. Both are inside the gate.

## Item 3 - what was built

**Per-model precision** (`roop/precision_policy.py`).
- `PRECISION_OVERRIDES = {"xseg": "fp32", "2d106det": "fp32", "1k3d68": "fp32"}` keyed on the model FILE stem; no entry = follow `trt_precision`.
  `ROOP_TRT_MODEL_PRECISION="xseg:mixed;1k3d68:global"` wins over the table (`global` drops an entry). Swappers are refused (they have `ROOP_SWAP_FP32` and the
  swap canary); a typo is loud and inert. An explicit `requested=` to `providers_for` is still authoritative (the harnesses depend on it).
- XSeg goes through `providers_for` (the file path is known). The buffalo_l files do not: insightface builds all five from ONE chain, so
  `face_util._per_model_precision` wraps `model_zoo.get_model` for the duration of `FaceAnalysis(...)` (restored in `finally`; no change to site-packages) and calls
  `bundle_member_providers`, which applies **only** the precision step. It deliberately skips `_finalize` - that would also attach a shape profile to 2d106det /
  1k3d68 and move their engines to a new cache namespace, a different change from the one being made. An override can tighten (`fp32`) or equal the global
  setting; loosening a model under a global `fp32` would need the pre-global chain a bundle caller no longer has, so it is refused loudly.
- Engines stay per precision on disk: FP32 builds land in `..._masking_fp32` / `..._recognition_fp32` next to the mixed directory, so reverting loses nothing.
  `_model_digest` is now memoised on (path, size, mtime) - it re-read the whole file on every `providers_for` call, and 1k3d68 is 137 MB.
- Proof it executes (AGENTS: "count the stage, not the wrapper"): `aux_canary_calibration.py shipped` builds the live sessions through the loaders, reads
  `trt_fp16_enable` back and aborts if it disagrees with the table. Live: xseg 0, w600k_r50 1, 2d106det 0, 1k3d68 0, each with provider TensorRT.

**Startup canary cached by engine hash** (`roop/aux_canary.py`, wired in `Mask_XSeg.Initialize` and `face_util._guard_aux_models`).
- Two fixed synthetic inputs (the swap canary's "skin" and "noise", scaled as each model's own pre-processing scales a crop) go through the live session and
  through a transient CUDA FP32 session (TF32 off). Metrics by consumer: xseg `mean|sigmoid diff|` and IoU@0.5; w600k cosine; landmarks mean / max crop px after
  `Landmark.get`'s decode. Floors (`aux_canary_calibration.md`): xseg 0.08 / 0.60, cosine 0.99, 2d106det 1.5 / 5 px, 1k3d68 4 / 12 px.
- They are **not** the fidelity gates. They sit between the worst value of a healthy engine on the canary inputs and the value of a collapsed or wrong-face engine
  (e.g. XSeg healthy mixed 0.035 vs zeros 0.23-0.28; 2d106det healthy 0.09 px vs other-face median 13 px). A healthy mixed engine's worst real face (xseg IoU 0.81,
  1k3d68 mean 11 px) would NOT trip them - that drift is the fidelity harness's job; the canary is for the swap-canary failure class (builds, runs at speed, wrong
  picture).
- **Cache key:** sha256 of {ONNX digest, the model's floors, the build options that shape an engine, the cache-directory name (GPU / driver / CUDA / TRT / ORT
  versions), and the blake2b content hash of every cached `.engine` whose name carries the model's ONNX graph name}. ORT names engines
  `..._TRTKernel_graph_<graph name>_<hash>...` and several models share a graph name (`main_graph`, `mxnet_converted_model`), so the match is a superset: a rebuild
  of ANY engine in it re-checks (spurious, never unsafe). File hashes are memoised on (path, size, mtime). A verdict is stored in
  `models/runtime_profiles/aux_canary.json` (gitignored); delete it to force a re-check. A check that could not run is never cached; neither is anything when
  there is no engine cache to hash. A cold start (no engine yet) runs one canary feed first so the engine exists before it is hashed.
- **Measured:** first start after adoption 52 s init (the canary ran, plus the cold FP32 builds of 2d106det / 1k3d68, 12-16 s each); second start 21 s with every
  verdict served from the cache and no reference session built. The pooled second analyser re-checks on the very first start only, because new engines of the
  same graph name appear between the two checks.
- **On failure:** `guard` rebuilds the model on a TensorRT FP32 engine (the canonical-key FP32 cache `providers_for(..., 'fp32')` already uses), re-checks it,
  then falls to CUDA/CPU. Demonstrated live by giving the canary an impossible XSeg floor on a mixed engine: `FAILED ... running on the TensorRT FP32 engine
  instead`, and the mask still ran. FP32 engines are checked too (the cache makes it free after the first start; tactic selection can go wrong at any precision).
- Not done: the swap canary itself is still uncached (it builds its CUDA reference every start); `ROOP_AUX_CANARY=0` disables the new one.

## Item 4 - NCHW XSeg variant: NOT built

Item 1 does show a transpose - one `Shuffle` - but it costs 0.0126 ms mixed / 0.0137 ms FP32 (+0.0111 ms input reformat) of a 2.5-3.2 ms call: **a ceiling of
0.4-0.8% of one XSeg call, ~13-25 ms per 1000 calls**. The regression clip's mask stage logged 816 calls in 300 frames (XSeg plus the occluder after it), so
XSeg runs at most ~2.7 times per frame and the ceiling is under 0.07 ms of a ~140 ms frame.
That is below anything this rig resolves end to end (AGENTS: ~50% effects reliably, ~5% not at all), and it would cost model surgery plus a mask re-verification.
Rejected on the numbers; if the transposes ever show up larger (a different TRT version), `tests/trt_precision_probe.py` reports them in one run.

## Verification

- `test_trt_model_precision.py` (22), `test_aux_canary.py` (39): no GPU; fake sessions with the ORT surface the code touches, including: the table is exactly the
  measured one, an override changes only that model's precision and cache directory, the bundle seam never calls `_finalize`, a rebuilt / rewritten / different
  engine or model or option or floor is a cache miss, "could not check" is never a failure nor cached, FP32 failing falls to CUDA.
- Live: sessions read back per the table (above); canary OK on all four; warm start fully cached; forced failure rebuilt on FP32.
- `python run.py --benchmark --benchmark-mode regression` (300 frames, 1080p solo): **PASS**, swap decided 300/300 and changed 300/300, SWAP AUDIT 408/408
  swapped, face SSIM 0.985 / PSNR 41.7 dB. End to end 7.02 fps against the 2026-10-02 baseline's 7.30 - **that is a single 300-frame run against an old baseline,
  not an A/B, and is not a speed claim**.

## Not measured

- **End-to-end fps of the three FP32 moves.** Upper-bound cost ~1.6 ms per frame (xseg +0.6 ms x at most ~2.7 mask-stage calls; 2d106det ~0; 1k3d68 +1.05 ms per
  face only when it is in the per-face loop, i.e. not `lm68_lazy`) against ~140 ms frames: about 1%, below the ~5% this rig resolves, so a 600-frame ABBA would
  read noise. Per-call timings (GPU-resident IO, median of 200) are the measured part.
- The 3060 (TensorRT not admitted: precision arms inert, canary skipped).
- RF++ (`restoreformer_pp`) and the rest of the 10-05 pending decision - not in this brief.
- Occlusion robustness of the FP32 mask (no ground truth).

## Files

`roop/precision_policy.py` (table, parser, `bundle_member_providers`, digest memo) - `roop/aux_canary.py` (new) - `roop/face_util.py` (`_per_model_precision`,
`_guard_aux_models`) - `roop/processors/Mask_XSeg.py` (guard) - `tests/trt_precision_{probe,fidelity}.py`, `tests/aux_canary_calibration.py` (harnesses) -
`tests/test_trt_model_precision.py`, `tests/test_aux_canary.py` - `docs/perf/trt_precision_{fidelity,probe,probe_shipped,shipped_check}.*`,
`aux_canary_calibration.*`.
