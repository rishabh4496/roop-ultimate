# TensorRT build heuristics, bound/graph static sessions, RF++ batching (2026-10-09)

Brief (A/B only, all opt-in, defaults unchanged). RTX 4070 12 GB, TensorRT **10.9.0.34**, ORT **1.23.2**,
live `app/config.yaml`: hyperswap, DFL XSeg, Restore Ultra, scrfd @ 512, `trt_precision: mixed`, threads 10,
pools 2/2/2. Continues the paused 10-06 session.

## Result

| item | outcome |
|---|---|
| 1. `ROOP_TRT_BUILD_HEURISTICS` | **No difference** (10-06, kept): off/on latency 0.999-1.039, never faster. Stopped. |
| 2. bound I/O + own stream | **Neutral** on XSeg, RF++, detector (1.00-1.04x). Not kept. |
| 2. + `trt_cuda_graph_enable` | **XSeg only**: bit-identical 40/40, stable 10k calls, 0.46-0.64x latency, 1.45-2.8x pooled calls/s. RF++ and the detector rejected (below). |
| 3. RF++ cross-frame batching | **Not built.** B>1 works and is bit-identical per face, but per-face cost is flat-to-worse. |
| ABBA renders, XSeg graph | **Not measurable** on d4 (+1.0%) or s7 (-1.7%); A-vs-A noise 3.0% / 7.3%. |

Net: one opt-in survived (`ROOP_TRT_BOUND=xseg=graph`), it is correct and stable, and it does not move a render.
This is the standing frame in AGENTS.md: XSeg is a few ms per face on a GPU-bound pipeline, so a faster XSeg
call redistributes thread time instead of removing GPU work.

## The 10-06 "graph is not bit-identical" finding was a harness bug

The 10-06 probe read XSeg graph as 0/40 bit-identical at 0.64x latency. It was returning **one constant
all-zero mask for every input** (12 distinct crops, 1 distinct output, mean 0.0), i.e. not running the model on
the input at all. Cause: `BoundStaticSession` learned a symbolic output shape with a plain `session.run` on the
graph-enabled session, and that run counts toward ORT's capture sequence, so the graph is captured around ORT's
own allocations and every later bound run replays it. Not stream or sync related (default stream, user stream and
full device syncs all froze identically); ORT-owned and torch-owned buffers both work once nothing but bound runs
touch the graph-enabled session. Fixed: the shape probe runs on a graph-off throwaway session.

Two more defects in the same module, found by the same re-run:

* **Declared output shapes are not trustworthy.** `det_10g` declares static outputs from its 640 export (12800
  anchors). ORT validates a pre-bound output against that, so at det size 512 (the live size) the run is refused
  (`Got: 8192 Expected: 12800`). Shapes now always come from a real run, and **bound/graph is unavailable for the
  detector at 512** (no preallocated-output binding without re-exporting the model). At 640 it works.
* **RF++ lists 15 graph outputs**, 14 of them feature maps up to 64 MB; the app binds `outputs[0]` only. Binding and
  copying all 15 made the first RF++ bound/graph read 3.3x slower (73 vs 22 ms; 37 ms of it was the numpy copy).
  `BoundStaticSession(outputs=[...])` now binds a subset. The first run's RF++ numbers are void.

## Item 2 numbers (`ab_trt_bound_sessions.py`, 300 interleaved calls, 40 real d4 crops)

Latency median ms (x = vs prod), identity = bit-identical to the production call path on all 40 inputs.

| model | prod | bound | graph | pooled x2 calls/s prod / bound / graph | 10k-call run (graph) |
|---|---|---|---|---|---|
| XSeg 1x256x256x3 | 2.34 / 3.38 | x1.04 | **x0.64 / x0.46** | 484 / 488 / 702 ; 265 / 278 / 738 | 0 re-check misses, 1.618 -> 1.597 ms, no VRAM drift |
| RF++ 1x3x512x512 | 21.7 | x1.00 | x0.93 | 51.3 / 49.7 / **46.6** | 3000 calls, stable (20.1 ms) |
| det_10g @640 | 2.12 | x0.95 | x0.83 | 506 / 565 / 686 | 10k calls, stable |
| det_10g @512 | - | unsupported | unsupported | - | - |

* Two runs of XSeg give two prod baselines (2.34 and 3.38 ms) because the card's state differs between runs; the
  ratios hold, the absolute numbers do not. Do not quote a ms figure from one run.
* RF++ graph saves ~1.4 ms of 21.7 single-threaded but is **slower pooled** (0.91x): the engine is the cost and
  the app runs it pooled. Rejected.
* The detector is only usable at 640; the config runs 512. Rejected as shipped.

## Item 3 (`tools/bench_restorer_batch.py`, now with a per-face bitwise check)

Production batch-1 20.36 ms/face. Relaxed-batch engine: B=1 21.32, B=2 23.28, B=4 22.71, B=8 22.54 ms/face
(call time scales linearly with B). Every face is **bit-identical** to the same engine at B=1 (1/1, 2/2, 4/4,
8/8, max abs diff 0.0). So a batcher modelled on `swap_batcher` would be correct and would gain nothing (-10.7%
best case). Not built. Matches 10-03.

## ABBA renders (`ab_trt_bound_render.py`; 600 frames, live config, `ROOP_STAB_CHUNK_MB` pinned)

A = shipped, B = `ROOP_TRT_BOUND=xseg=graph`. Discarded warm-up per clip. Frame-loop fps from `[Pipeline] done`.

| clip | A (arms 1, 4) | B (arms 2, 3) | B/A | A-vs-A | faces_seen | faces/s A / B |
|---|---|---|---|---|---|---|
| d4 | 9.60, 9.89 | 9.39, 10.30 | 1.010 | 3.0% | 963 all | 10.8, 11.1 / 10.5, 11.6 |
| s7 | 12.10, 11.25 | 11.44, 11.51 | 0.983 | 7.3% | 652 all | 10.6, 9.9 / 10.0, 10.1 |

**The picture is bit-identical too (added later the same day).** All four renders of each clip have the same file sha256 and
decoded-video md5, A and B alike (d4 `6650e371b02b`, s7 `891f687daa2c`; the same hashes reappear in
`stab_dedup_2026-10-09.json` from an independent session). So the XSeg CUDA graph changes nothing the viewer sees, only
how long the call takes, and a render at a pinned geometry is bit-reproducible on this machine.

Path proven: every B log carries `mask:xseg[graph] x2` (both pooled contexts bound), every A log `mask:xseg x2`;
same stabilizer path (parallel-blocks) and `faces_seen` in all arms; 0 failed frames. B-vs-B spread on d4 was 9%,
larger than the effect, so neither clip can resolve a difference under ~7%.

## What is in the tree (all opt-in, defaults unchanged)

* `ROOP_TRT_BUILD_HEURISTICS` (core.py): unset = shipped options and the SAME cache namespace (asserted by test).
* `ROOP_TRT_BOUND=xseg=bound|graph` (`roop/trt_bound_session.py`, wired in `Mask_XSeg` only). Falls back to the
  shipped path loudly if the provider is not TensorRT or the build fails. `rfpp=` / `det=` are parsed but not wired,
  deliberately: both lost.
* Harnesses `ab_trt_build_heuristics.py`, `ab_trt_bound_sessions.py`, `ab_trt_bound_render.py`; 11 unit tests in
  `test_trt_bound_session.py`.
* Pre-existing and not measured here: `ROOP_TRT_CUDA_GRAPH` (default 0) already sets `trt_cuda_graph_enable` for
  every TRT session through the ordinary call path. This work did not test it; the per-model graph above is a
  different thing (persistent bound buffers are what make a captured graph valid).

## Not tested / caveats

* Only the 4070. The 3060 does not admit TensorRT, so every arm there is inert (AGENTS.md).
* 10k-call stability is single-session synthetic repetition of 40 crops, not a 10k-frame render; the renders are
  600 frames. A graph that is correct for 40 crops over 10k calls is evidence, not proof across all content.
* The XSeg graph holds one captured graph per session; a pooled session reused across very different loads was not
  exercised beyond the two-thread pooled test and the 600-frame renders.
