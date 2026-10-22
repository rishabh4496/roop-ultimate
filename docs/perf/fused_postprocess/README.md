# Fused Stage 3 post-processing kernel + L2 persistence (2026-10-22) — built and measured, NOT wired into the render

Code: `app/core/fused_postprocess.py` (Triton), `app/core/l2_persistence.py` (cuda-python). Tests: `tests/test_fused_postprocess.py`
(23: parity with the unfused maths, the maths against OpenCV, grain statistics, strided in-place composite, refusals),
`tests/test_l2_persistence.py` (7: the API calls and their cleanup). Probes: `tests/probe_fused_postprocess.py`,
`tests/probe_l2_persistence.py`. Raw: `benchmark.json`, `l2_persistence.json`.

## Corrections to the brief
* `torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")` is not available here: Inductor needs a C++ compiler and the
  Windows install has no `cl`. Triton was used (`triton-windows` 3.3.1, which bundles its own C stub compiler).
* `torch.cuda.get_device_properties(0)` has `L2_cache_size` only; there is no `persisting_l2_cache_max_size`. The persisting maximum
  is `cudaDeviceGetAttribute(cudaDevAttrMaxPersistingL2CacheSize)`: **24.75 MiB** of the 36 MiB L2 on this RTX 4070 (window max 128 MiB).
* There is no `cudaCtxSetLimit`. The runtime call is `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)` (driver: `cuCtxSetLimit`).
  `cudaStreamAttributeAccessPolicyWindow` is a header alias; cuda-python 12.9 exposes it as `cudaLaunchAttributeAccessPolicyWindow`.
* The driver granted **18 MiB** when 16 MiB was requested (it rounds up).
* The "TensorRT output tensors" are not on the device in production: ORT returns numpy, and the later stages are host-frame code.
  Hence not wired (below).

## What "single SRAM pass" means here, and its limit
A Gaussian is a stencil, and Triton exposes no shared memory; a scratch buffer in global memory would defeat the point. Each program
loads one haloed 32x32 tile of every input (reflect-indexed, so borders match OpenCV's `BORDER_REFLECT`), applies the Reinhard
colour transfer per pixel in registers, then blurs with a banded Gaussian **matrix** built from index arithmetic (`G @ X @ G` via
`tl.dot`, IEEE fp32), combines the bands, adds grain, feathers the mask, and composites into the destination (which may be a
strided VIEW of the frame). Nothing is written between stages. The order is the brief's: colour transfer BEFORE the blur.
* A 64x64 tile needs 161 KB of shared memory; Ada allows 99 KB. So the tile is 32x32, the halo at most 8 px (sigma <= 2.67), and a
  larger sigma is refused. At halo 6 the wrong warp count is 7x slower (8.97 ms vs 1.23 ms at B=4, 512x512), so it is chosen from the halo.
* Whole-image reductions cannot live in a tile kernel: the LAB statistics (`lab_stats`, `LabStatsEMA`) and the target's noise floor
  (`estimate_noise_sigma`, `grain_deficit`) are small torch reductions run first and handed in.
* Grain is Triton's counter-based `tl.randn`, so it is repeatable for a seed and independent of tile boundaries; it is verified
  statistically (sigma within 0.2, zero mean, identical on the three channels), not against another RNG.

## Numbers (RTX 4070; ms for the whole batch; `benchmark.json`)
| faces x size | fused | unfused torch (same maths) | fused incl. host upload + download | CPU chain, production functions |
|---|---|---|---|---|
| 1 x 256 | 0.13 | 2.81 (211 kernels) | 0.21 | 6.6 |
| 4 x 256 | 0.20 | 2.75 (213) | 0.39 | 25.1 |
| 1 x 512 | 0.36 | 3.46 (283) | 0.74 | 28.9 |
| 4 x 512 | 1.49 | 10.5 (287) | 2.00 | 186 |

Max difference from the unfused reference: **1 level** (uint8 rounding). The CPU arm is the real production functions (8-bit
`_color_transfer_rct`, `frequency_blend`, `grain_match`, a cv2-feathered composite), one thread; it prices the step, it is not a
pixel twin. LAB itself is checked against OpenCV: within 0.36 LAB units max (OpenCV's float Lab is a gamma LUT, its own approximation).

## L2 persistence: the calls work; the measured effect is small
`probe_l2_persistence.py` floods L2 with 1 GiB of copies on one stream, then times one read of the working set (CUDA events,
interleaved arms, 6 rounds; spreads 0-4%):

| working set | read after flood, no window | with a persisting window (hitRatio 1.0) | delta |
|---|---|---|---|
| 4 MiB | 15.4 us | 15.4 us | **0.0%** |
| 8 MiB | 26.6 us | 25.6 us | -3.8% |
| 16 MiB | 49.2 us | 43.0 us | -12.5% |

The driver accepts the window and grants the carve-out, but for the ROI working set this is meant for (`[4,3,512,512]` fp16 is 6.3 MB)
the effect is nil, and even at 16 MiB it is ~6 us per read against a post-processing step of 0.2-1.5 ms. Caveats: the "warm read, no
flood" control (18 / 45 / 49 us) was NOT faster than the flooded reads, so this probe does not show L2 residency directly (GPU clocks
idle without the flood); an earlier version of the probe, with the flood on a second stream, could not resolve anything (the time was
SMs taken by the other stream's copies). WDDM (this machine) vs TCC is unmeasured, as is the RTX 3060 (smaller L2).

## Not done
Wiring into the render (needs the swap / restore outputs on the device, a ROI-space resample step, and the GPU-bound-pipeline A/B:
moving host work onto the GPU adds GPU work); grain/LAB parity against production's 8-bit paths; fp16 intermediates; a full-frame
decode ring marked `cudaAccessPropertyStreaming` in a real pipeline (`mark_streaming` is tested for the call only); the RTX 3060.
The colour EMA the kernel's inputs allow was measured NEGATIVE (`docs/perf/color_ema/`): `LabStatsEMA(alpha=1.0)` is no smoothing.
