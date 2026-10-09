# Is the batched hyperswap the same picture as B=1? (2026-10-10)

Test: `app/tests/test_swap_batch_equivalence.py` (GPU-gated; `python tests/test_swap_batch_equivalence.py --out f.json` prints the table).
Data: `docs/perf/swap_batch_equivalence_2026-10-10/`. RTX 4070, live `config.yaml` (realswap, `trt_precision` mixed, TRT 10.9, ORT 1.23.2).

## Method
16 REAL aligned crops (production alignment `canonicalize_face_alignment` at the swapper's template, `to_blob` with its mean/std) from d4 (5),
d1 (3), d6 (3), s7 (3), Love (2); every crop paired with its OWN source, 16 different library facesets, so every batch mixes identities.
Each (crop, source) is run at B=1 through the production swapper session (`Run`), batched through `RunBatchMulti` at B=2/4/8 (consecutive
groups), at B=1 on a CUDA-FP32 session of the same batch-relaxed graph, and at B=1 again (repeatability). The `secondary` (hififace) net is
cleared so the compared quantity is the hyperswap session. Per row against its B=1 result: max abs diff (model units, [-1,1]), SSIM (8-bit
picture, data_range 255, `swap_canary.ssim`) and identity = AdaFace cosine of the output to its source (AdaFace is not the pipeline's matcher),
reported as the batched-minus-B=1 delta. Warm-up runs on real crops, every batch shape twice, discarded.

**Proof the batch ran** (an identical result is also what a silent sequential fallback looks like): `_infer` is spied on and the batch dimension
of every real inference recorded: B=2 -> 8 calls of 2 rows, B=4 -> 4 calls of 4, B=8 -> 2 calls of 8, no other sizes. A first run without the
spy gave the same numbers but proved nothing, so it is not kept.

## Result (pass: every row SSIM >= 0.998 and |identity delta| <= 0.005)

| comparison | min SSIM | max abs diff | max abs identity delta | verdict |
|---|---:|---:|---:|---|
| B=1 vs B=1 again (control) | 1.00000 | 0.0000 | 0.0000 | bit-identical |
| **B=2 / 4 / 8 vs B=1, production TensorRT** | **1.00000** | **0.0000** | **0.0000** | **PASS** |
| B=1 (TensorRT mixed) vs CUDA-FP32 (control) | 0.99965 | 0.0177 | 0.0031 | the instrument resolves 1e-2 differences |
| CUDA-FP32 session, B=2 vs its own B=1 (information) | 0.99541 | 0.5294 | 0.0107 | **would FAIL** |
| CUDA-FP32 session, B=4 / B=8 vs its own B=1 | 0.99542 | 0.5296 / 0.5299 | 0.0106 / 0.0110 | **would FAIL** |

**The production path passes with zero difference, so the item-3 fix (port `decompose_instance_norm` into `_relax_batch_dim`) was NOT applied**:
the brief says fix only if it fails. Why bit-exact: not measured. A plausible reading is that one TensorRT engine (profile batch 1..8) uses the same
tactics for every row and InstanceNormalization reduces per (sample, channel), but that is a hypothesis.

## What this does NOT cover (a live exposure)
The ORT **CUDA** provider is wrong at batch > 1 on this generator, as `face_engine/utils/onnx_batch.py` documented on 2026-09-28: rows differ from
their own B=1 result by up to 0.53 model units. The TensorRT path does not have the problem, but batching (`ROOP_BATCH_SWAP`, on for any card with
>= 10 GB) is not tied to TensorRT. It reaches ORT CUDA when (a) a >= 10 GB card has no TensorRT, or (b) `swap_canary` rejects the TensorRT engine
and the swapper falls back to CUDA FP32 with batching still enabled. Neither case is tested or guarded today. Options if wanted: decompose the norm
only for non-TensorRT sessions, or force B=1 when the provider is not TensorRT. Not done: outside the brief's trigger.

## Not changed / not done
`FaceSwapInsightFace.warmup_batch` still warms with zero tensors (the test's warm-up uses real crops; production's does not). Item 4 (explicit FP32
islands for the norm sub-graphs) not started.
