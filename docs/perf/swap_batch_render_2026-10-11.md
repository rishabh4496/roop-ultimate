# Cross-frame swap batching, end to end: ON vs OFF (2026-10-11)

**Result: batching the swap stage does not speed the render up. On two-person footage it is 4-9% SLOWER on the 4070; on single-person footage it is inside the noise.** No code or default was changed.

Question (Stage 5 brief, option 2): does batching the `realswap` pair (hyperswap + the hififace eyelid band) move the render clock? The pair batching is already built and live - `FaceSwapInsightFace.RunBatchMulti` coalesces both nets, `ProcessMgr._make_swap_batcher` no longer excludes composites - so no new conversion was needed. (The comment above `RunBatch` still says a composite "declines the batched paths"; that is stale and wrong, `_batch_unsupported` is no longer set for a secondary.)

## Method
`app/tests/ab_swap_batch_render.py`, the production render path (`baseline_snapshot.run_one`, live `config.yaml` except `swap_model`, set in memory only), RTX 4070, `trt_precision mixed`, Restore Ultra, DFL XSeg, **8 threads (the live config)**, 600-frame window, one discarded warm-up per clip, then A B B A, `ROOP_STAB_CHUNK_MB` pinned to the warm-up's budget, `ROOP_PROFILE=1`.
- **A = ON** (shipped default; batcher `max_batch=8, wait=4.0ms`). **B = OFF** (`ROOP_BATCH_SWAP_MAX=1`, which logs "cross-frame batching OFF" and leaves the same batch-dynamic session built, so the arms differ in the batcher alone).
- **Each arm is proven from its own log:** A printed `cross-frame batching ON` and a batcher report; B printed `OFF` and no report. All arms ran as labelled.
- **Swap-rate guard:** `faces_seen` and swapped counts identical across every arm of a clip (d4 944 seen, 764 identity-lock + 37 per-frame; s7 652 seen, 652 swapped), so nothing got faster by swapping less. **Pair proof:** both arms logged `realswap: 665 of 665 faces composited with 'hififace' over the eye band`; the ON arm warmed batch-8 engines for both nets.

## Numbers (frame-loop fps)
| swap model | clip | ON (A, A) | OFF (B, B) | ON / OFF | A-vs-A | B-vs-B | avg batch |
|---|---|---|---|---|---|---|---|
| realswap | d4 (2 people) | 13.94, 13.97 | 15.04, 15.53 | **0.913** | 0.21% | 3.21% | 1.63 (max 8) |
| realswap | s7 (1 person) | 18.45, 19.16 | 19.46, 18.86 | 0.982 | 3.78% | 3.13% | 1.48 |
| hyperswap | d4 | 15.79, 15.81 | 16.59, 16.48 | **0.956** | 0.13% | 0.67% | 1.50 |

d4 effects (-8.7%, -4.5%) are several times the null spread of their own clip and agree in sign; s7 (-1.9%) is below its own noise (~3.5%) and is "not measurable".

## Reading
- **The batcher barely batches:** 1.3-1.6 crops per call at 8 threads, as in the 09-24 hyperswap note. Nothing to amortise, and what it adds is a single batcher thread that replaces the swapper pool's independent TensorRT contexts plus up to 4 ms of coalescing wait. That mechanism is a hypothesis - the A/B measures the effect, not the cause.
- **It agrees with the standing frame:** the swap stage is at the GPU floor (`FaceSwapInsightFace.Run`: realswap whole `Run()` 22 ms; deleting the second net entirely was +4.8% end to end), inswapper batching was ~+20% in a micro-bench, restorer batching is flat. This was the last unmeasured lead for the Stage 5 brief.
- The batched output is the same picture (`swap_batch_equivalence_2026-10-10.md`: SSIM 1.0, TensorRT), so the cost is speed only, not quality.

## Not measured (do not over-read)
- **Threads.** The live config is 8, so `max_batch` was capped at 8 and few crops overlap. `AGENTS.md` asks for `--threads 20` on bench runs; at 20 threads batches could be larger. Not run.
- **More clips/reps.** Two clips, two reps per arm; no d6/Love/d1. A d1 run was started and killed by a low-memory reaper after one arm (and used the whole clip, not 600 frames) - discarded, nothing quoted from it.
- **The 3060.** Batching is off there by design (sub-7 GB tier); not applicable and not measured.
- Pixel equality of the full render between arms (the swap-level equivalence report covers the net, not the composite).

## Decision for the owner
The shipped default (`perf_batch_swap` on, derived `max_batch` 8 on >= 11.5 GB cards) looks like a net loss on two-person footage at the live thread count. Options: (1) leave it and record this; (2) re-run at `--threads 20` and on d6/Love/d1 first, then flip the default for >= 10 GB cards (mind the provenance trap in `AGENTS.md`: a derived default that outlives its rule - saved configs keep the old value); (3) flip now. Not done.

## Reproduce
```
app/env/Scripts/python.exe tests/ab_swap_batch_render.py --swap-model realswap --clips d4,s7
app/env/Scripts/python.exe tests/ab_swap_batch_render.py --swap-model hyperswap --clips d4
```
Raw: `docs/perf/swap_batch_render_realswap_2026-10-11.json`, `swap_batch_render_hyperswap_2026-10-11.json`; arm logs in `app/output/ab_swap_batch_*` (gitignored).
