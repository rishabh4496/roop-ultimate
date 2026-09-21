# Runtime audit — TensorRT / CUDA / ONNX Runtime vs. target selection

Date: 2026-09-20 (MAIN, RTX 4070 12 GB, ORT 1.23.2, TensorRT EP present).
Baseline: `d0b0d3c` plus the target-selection fixes of the preceding six
commits, all kept. The 41 selection tests (`test_target_selection_contract`,
`test_selected_target_mapping`, `test_selected_face_safety`,
`test_source_faceset_mapping`, `test_preview_render_selection_contract`,
`test_project_target_faces_serialization`, `test_target_path_resolution`)
were green before anything here was touched.

The rule under audit:

```text
runtime failure -> controlled processing error / fallback        (allowed)
runtime failure -> change face-selection semantics               (never)
```

## 1. Where the two layers meet

Selection is resolved ONCE, in `ProcessMgr.initialize`
(`roop/ProcessMgr.py` ~line 993): `target_selection` from the canonical
request, `selected_target_groups` via `selection_group_ids`. The per-frame
matcher (`swap_faces`, the `("selected", "selected_multi")` branch) reads only
those two attributes plus the identity distance. The runtime layer starts
strictly below that, at `process_face` -> `p.Run / RunBatch / RunBatchMulti`,
and receives an already-paired `(source_face, target_face)`.

Verified by `tests/test_runtime_selection_invariant.py`:

- none of `predictor.py`, `backend_manager.py`, `swap_batcher.py`,
  `session_pool.py`, `render_guard.py`, `precision_policy.py`,
  `trt_session_builder.py`, `gpu_preflight.py`, `trt_engine.py`,
  `trt_probe.py`, `ort_support.py`, `cudnn_algo.py`,
  `processors/FaceSwapInsightFace.py` names a selection symbol
  (`selection_state`, `swap_mode`, `selected_index`, `TARGET_FACE_GROUP`,
  `person_id`, `distance_threshold`, ...). There is no code path by which a
  fallback could rewrite one.
- the batch -> B=1 fallback (`_sequential_fallback`) replays every request
  with the SAME `(source, target)` pair, in order.

## 2. Item-by-item

| Item | Where | Finding |
|---|---|---|
| Active provider | `predictor.assert_session_providers` — reads `session.get_providers()`, the only reliable tell; ORT never raises on a dropped EP | OK. Strict by default (`ROOP_STRICT_PROVIDER=1`): a TensorRT request that came up on CUDA is a `ProviderAssertionError` at `Initialize`, i.e. a controlled render error before frame 0. Non-strict records a degradation. Fires only when TensorRT was *requested for that session*, so models `precision_policy` routes to CUDA/FP32 on purpose do not trip it. |
| Provider order | `backend_manager.canonical_provider_decision` + `_HIERARCHY` | OK. `auto`/`tensorrt` -> TRT>CUDA>CPU; `cuda` -> CUDA>CPU; `cpu` -> CPU. Live probe on this machine: all three chains resolve as requested, `degraded=False`, no recorded degradations. Cached per `(request, device)` for the process. |
| TensorRT availability | `gpu_preflight.run_gpu_preflight` builds a real TRT session, not a DLL listing | OK. `tensorrt_session_usable=True`, `active_provider=TensorrtExecutionProvider`, `failure_stage=None`. Tiering (sub-7 GB) sets workspace/pool/precision defaults; it never rejects TRT by card size. |
| CUDA fallback | `build_session_with_fallback` (construction) and `FaceSwapInsightFace._rebuild_without_trt` (run time) | OK and loud: `[Backend] ... session build FAILED, falling back to ...` / `[swap] '...' failed under TensorRT ... rebuilt on CUDA/CPU`. The run-time rebuild fires only on a **batch-1** failure (a batch>1 failure says nothing about batch-1 under TRT; that used to poison every later single-frame swap, 25x). Neither touches selection. |
| Model loading | `FaceSwapInsightFace.Initialize` | OK. Provider chain from `_swap_providers` (precision policy applied), `verify_and_warmup` asserts the provider and pays the engine build on a dummy tensor. A model switch releases and reloads. **Fixed here:** `Release()` now calls `predictor.forget(tag)` — the warm-up ledger was once-per-tag for the process, so a swapper reloaded under the same tag skipped its dummy pass and paid the engine load on frame 0 of the render. |
| Batch capability | `SWAP_MODELS[...]["batch_capable"]`, `_batch_unsupported` | OK. Declared per model (inswapper `False`, hyperswap family `True`); a composite (`realswap`) declines batching because its secondary must be coalesced in step. A first batch>1 failure sets `_batch_unsupported` for the rest of the run and says so. |
| B=1 vs batch | `ProcessMgr.process_face`: `xframe` (SwapBatcher, threads>1, non-unified scheduler) / `tile` (RunBatch over >1 pixel-boost tiles) / `sequential` (Run, B=1) | OK, with one visibility fix: with the unified scheduler ON (the production default) there is one inference owner and no cross-frame batcher, and with `subsample_size == model_output_size` there is one tile — so the shipped path is **B=1 per face** regardless of `perf_batch_swap`. The new banner names the path actually taken (`batch_mode=`), and `tile` is only claimed when >1 tile exists. |
| Provider-specific failures | `precision_policy` (FP16 smudge -> FP32 for inswapper; ESRGAN/RIFE/SAM TRT-unsupported; GFPGAN FP16 collapse), `_rebuild_without_trt` (GHOST shape verification) | OK. Every one is a runtime decision with a printed reason; none reaches `swap_faces`. An exception that escapes `process_face` propagates through `future.result()` and aborts the render (controlled error), it does not re-match the face. |
| Resource guards | `render_guard.check_render_gpu_headroom` (free-VRAM floor before `initialize`), small-card enhancer/decode policies, `_gpu_guard` per stage | OK. The VRAM floor is on *free* memory (1.0-2.5 GB), raises `RuntimeError` before model construction, never disqualifies TRT by total VRAM, and is skipped for CPU. |
| VRAM admission | `session_pool.TensorRTResourceManager` (pool sizing against live VRAM), `backend_manager.provider_admission` | OK. Admission explains itself (`admitted`, `is_sub_7gb_gpu`, `tensorrt_allowed`, `reason`). Explicit pools bypass the live guard (known, memory `explicit-pools-bypassed-vram-guard`). |
| Session reuse | `THREAD_LOCK_SWAPPER` singletons, `SessionPool` leases, `_io_bindings` keyed by `id(session)`, `_warmed`/`_asserted` tags | OK except the warm-up ledger above (fixed). `_rebuild_without_trt` clears `_io_bindings` and releases the old pool after outstanding leases. |

Two fallbacks that are *not* selection changes but do change the matching
mechanism, and print so:

- tracking pre-pass failure -> `_track_mode=False`, per-frame matching
  (`[Track] identity pre-pass failed ...`);
- temporal detection pre-pass failure -> per-frame detection
  (`[Temporal] detection pre-pass failed ...`).

Both still filter by `selected_target_groups`; the selected person is the same,
the per-frame assignment is what changes.

## 3. The banner

`roop/runtime_banner.py`, printed at `initialize` (`phase=init`) and again once
the video batch path is settled (`phase=video`):

```text
[Runtime] phase=video provider_active=tensorrt requested=tensorrt>cuda>cpu precision=mixed swap_model=hyperswap batch_mode=sequential target_selection=selected/selected selected_person=0 selected_groups=[0] persons=2 selection_invariant=OK
```

- `provider_active` is `session.get_providers()[0]` of the live swap session
  (`cuda(trt-rebuilt)` after a run-time TensorRT rebuild), never the request.
- `precision` is read off the provider options the session was built with.
- `batch_mode` is the dispatch path `process_face` will take.
- `selection_invariant` compares the selection frozen at `initialize` with the
  live one; a moved selection prints `VIOLATED(<fields>)`.

## 4. Cross-provider parity (`tests/provider_selection_parity.py`)

Renders `double/d1.mp4` in Selected-face mode under CPU, CUDA and TensorRT with
byte-identical selection input (targets captured once and serialized,
`selection_state` fixed) for person 0 and person 1 separately, and compares
`ProcessMgr._SWAP_LOG` — the settled per-frame `(bbox, source)` decision — 1:1
per frame. Each arm is its own process; the report carries each arm's
`[Runtime]` banner so `provider_active` proves where it ran.

### Results — 2026-09-20, RTX 4070

**Provider invariance (the core requirement).** `double/d1.mp4`, frames 0–119,
Selected-face mode, hyperswap, tracking on, one source (person_a), captured
targets frozen and shared:

| comparison | person 0 | person 1 |
|---|---|---|
| CPU vs CUDA | 119/119 identical | 119/119 identical |
| CPU vs TensorRT | 119/119 identical | 119/119 identical |

Every arm: `selection_invariant=OK`, 119/120 frames swapped (1 frame had no
detection at all — a detector miss, not a gate). `provider_active` from each
arm's banner confirmed `cpu` / `cuda` / `tensorrt` respectively; precision
`fp32` / `fp32` / `mixed`; `batch_mode=sequential` on all (unified scheduler +
single 256px tile → B=1 per face, the shipped path). Arm times 274 s (CPU),
42.6 s (CUDA), 47.1 s (TensorRT) — the runtime differs by ~6.4× while the
per-frame swap decision is byte-identical.

The selected target is identical across all three providers. **The runtime
layer does not change which face is selected.**

**Selection sensitivity — a methodology limit, not a runtime result.** Showing
that person 0 and person 1 select *different* faces (and that the difference is
identical across providers) needs a window with two faces both being swapped.
Two attempts could not produce one through this headless harness:

- `double/d2.mp4` / `double/d6.mp4`, selected-single-person mode: swap output is
  near-zero (0/200 composited, both providers, both persons). This is the
  **known open lead** "single-person fallback" on multi-face content
  (`roster-baseline-2026-08-18`), upstream of the runtime layer entirely — the
  matcher/tracking declines the faces before any provider runs. Per the audit
  brief, identity routing is NOT compensated for by changing thresholds, so it
  was left as-is. CUDA and TensorRT agreed even in this degenerate case
  (`0/0 identical`, `selection_invariant=OK`).
- d1's 0–119 window has one visible person, so either selection legitimately
  swaps the one face — the harness reports `NOT TESTED`, not a false pass.

Selection sensitivity itself is covered where it belongs — the selection layer's
own unit tests: `test_target_selection_contract` asserts *changing the selected
person changes only the eligible group* and *an invalid person index is
diagnostic and never redirected*. What THIS audit establishes is the runtime
claim: whatever the selection layer decides, CPU, CUDA and TensorRT execute it
identically (d1, 119/119, both persons). A future two-face sensitivity render
needs the full `processing_request`/`face_mapping` the API builds, which this
harness deliberately does not construct.
