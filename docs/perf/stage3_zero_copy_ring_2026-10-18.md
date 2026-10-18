# Stage 3 — zero-copy enhancer ring (2026-10-18, RTX 4070, TensorRT mixed, ORT 1.23.2)

Policy: `CUDA_EXECUTION_POLICY.md` "Stage 3". Code: `ZeroCopyBufferPool` / `ZeroCopySlot` in
`app/roop/optimized_processor.py`; wired in `Enhance_GPENRealistic`, `Enhance_RestoreFormerPPlus`,
`Enhance_UltraMax`. Tests: `test_zero_copy_buffer_pool.py`, `test_enhance_common_gpu.py`,
`test_enhancer_zero_copy.py`, `test_enhancer_ultramax.py`. Scratch harnesses (not in repo):
`probe_binding.py`, `probe_gpenr.py`, `probe_rfpp.py`, `probe_um.py`, `ab_ring.py`.

## What the brief assumed, and what was in the code

* "A single static OrtValue output reused across faces": **no such buffer existed.** Every enhancer bound
  its output as `bind_output(name, 'cuda')` (ORT allocates per run) or `CudaIOBinding.run_gpu` allocated a
  fresh `torch.empty` per call. The race the brief describes is created BY a shared output ring, which is why
  the ring is guarded by per-slot locks and events (tested, with a negative control and mutants).
* "Torch-owned output corrupts faces" (`enhance_common.ort_cuda_output_to_torch`) is true of CodeFormer's
  DYNAMIC output and false for a static one: GPEN-512 under TensorRT, input AND output bound to ring tensors,
  max |diff| 0.0 vs the host path on 4 crops. So static outputs are bound by pointer, dynamic ones stay ORT-owned.
* `Enhance_UltraMax._run_cuda_postprocess` was **off under TensorRT** (commit 8c10e91), i.e. on the 4070 UltraMax
  ran the 92.9 ms/face host path; and `test_model_output_is_never_bound_into_torch_owned_memory` passed
  vacuously there (its loop ran over an empty call list). Now asserts >= 1 `bind_output`.
* ORT's CUDA provider runs on its own stream, so `stream.wait_event(slot_event)` orders only the Torch side.
  The ORT side is a host wait on `slot.ready` before the run and `synchronize_outputs()` after.

## Output identity (real processors, 16 crops incl. 256 -> 512; host path vs ring path)

| Enhancer | differing subpixels | note |
|---|---|---|
| GPEN Realistic | 0 / 12.6 M | needed three exact ports: `convertScaleAbs` is a fused multiply-add (float64 -> float32), OpenCV 4.11 `BGR2GRAY` is 15-bit fixed point (3735/19235/9798), collapse test = population std |
| RestoreFormer++ | 0 / 12.6 M | |
| UltraMax (ring vs old CUDA path, CUDA EP) | 0 / 6.3 M | ring == old CUDA path exactly |
| UltraMax (host path vs CUDA chain) | 29% of subpixels, max 11, mean 0.29/255 | **pre-existing**: the CUDA post-ops are float, the host's are cv2 fixed-point; below the 0.71/255 render noise floor, not introduced here |

## Cost, isolated (ms per face, 1 thread) and throughput (8 threads on the real pool)

| Enhancer | host | ring | 8-thread faces/s host -> ring |
|---|---|---|---|
| GPEN Realistic | 26.4 | 20.7 (-22%) | 51.5 -> 51.6 |
| RestoreFormer++ (FP32 default) | 49.1 | 44.2 (-10%) | 24.0 -> 24.0 |
| UltraMax (TensorRT) | 40.0 | 30.1 (-25%) | 35.3 -> 36.5 |

Per-face savings do not become throughput: the render is GPU-bound (AGENTS "standing frame").

## Concurrency / hang evidence (UltraMax on TensorRT was the deadlock-prone case)

3,800 faces at 20 threads x 100 and 12 threads x 150 plus 8 x 30, live TRT engine, 0 hangs, 0 mismatches vs each
face's own first pass. GPEN Realistic / RF++: 4 arms x 8 threads x 40 each, 0 mismatches.

### Correction (found by the Stage 4 full-suite run, same day)

The stress numbers above compared each face with ITS OWN first pass, which a systematic race would
reproduce, and the UltraMax ring path had a real one: `enhance_common.ort_cuda_output_to_torch` copies the
ORT-owned output with `cudaMemcpy` device-to-device, which does not block the host and runs on the legacy
default stream that the ring's non-blocking worker streams do not wait for, so Torch sometimes read the
allocation before the copy landed. **~5% of dynamic-output calls returned stale data** (17/300 single-threaded,
115/2400 at 8 threads on a synthetic graph). It affected only the ORT-owned (dynamic) output path, i.e.
UltraMax on the ring (CUDA EP by default, TensorRT only when opted in); GPEN Realistic / RF++ bind a static
output by pointer and do not use that copy. Fixed with a `cudaStreamSynchronize(0)` after the copy (0 of 2,700
afterwards), regression test `test_the_ort_owned_copy_is_complete_before_torch_reads_it` (fails 19/250 with the
sync removed). Re-verified after the fix: UltraMax ring vs old CUDA path 0 differing subpixels; TensorRT ring,
12 threads x 50 faces against a single-thread reference, 0 mismatches. The end-to-end A/B below was rendered
BEFORE the fix; its swap counts matched but its ring-arm pixels carried the race, so treat its fps as
indicative only (the fix adds one stream-0 wait per face).

## End to end (single/s7.mp4, 600 frames, UltraMax + realswap + RealityUX, config.yaml, 20 threads, TRT mixed)

Counterbalanced H H R R H R R H after a discarded warm-up (harness elapsed incl. setup, so absolute fps is low):

    host 9.18 9.57 8.95 9.22  mean 9.23  spread 6.7%
    ring 9.23 9.21 9.37 9.55  mean 9.34  spread 3.7%     ring/host 1.012x (+1.2%) — NOT RESOLVABLE

Swap guard identical in all 9 renders (417 of 421 faces seen). Ring arms: `417 through the zero-copy ring`,
ring released at job end with `peak 12.0 MiB`. No fps speed-up is claimed.

## VRAM

512 ring = 4 slots x (3.1 MB in + 3.1 MB out) = 24 MiB fp32 (12 MiB fp16/fp16); 1024 ring would be 96 MiB, built only
if a 1024 enhancer runs. A ring that would leave < 1.5 GB free is cut to fewer slots.

## Not done / owner decisions

* **UltraMax on TensorRT stays opt-in (`ROOP_ULTRAMAX_TRT_RING=1`).** It is safe on every test above and 25% cheaper
  per face, but end to end it is +1.2% (noise) and the failure it guards against is a driver hang. Flip the default
  when you want the host CPU back; no code change beyond `_trt_ring_enabled`.
* **No end-to-end device-resident handoff.** The merger/compositor takes NumPy; one uint8 D2H per face remains.
* **3060 not measured** (TensorRT not admitted there; ring is 24 MiB, governor margin applies). 
* `CudaIOBinding.run_gpu` (batch swap path) was not changed; it already allocates a fresh output per call.
* Only GPEN Realistic, RF++ and UltraMax were wired. GPEN 256 Pro / GPEN / CodeFormer / GFPGAN still use their host paths.
