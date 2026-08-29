# Gate D — Intel hybrid CPU optimization

## Scope

Gate D adds capability-driven CPU profiling and controlled CPU-distribution
selection. It does not make i9-14900K behavior mandatory and does not claim a
CPU policy is globally optimal without end-to-end evidence on both required
GPUs.

The runtime now records:

- physical and logical processor counts;
- measured Windows CPU-set P/E logical indices when available;
- CPU name, current/max reported frequency, NumPy SIMD dispatch features;
- process-affinity support and topology source;
- CPU frequency and available CPU temperature samples during benchmark
  telemetry.

Windows hybrid topology uses `GetSystemCpuSetInformation`; the OS efficiency
class is treated as the topology signal, not the GPU or CPU brand string.
Linux capacity data and explicit `ROOP_CPU_P_INDICES` /
`ROOP_CPU_E_INDICES` overrides remain supported. If no reliable topology is
available, automatic scheduling remains in effect.

## Runtime policies

`ROOP_CPU_DISTRIBUTION` supports:

| Policy | Selection |
|---|---|
| `auto` | Existing bounded hardware/workload policy; no forced affinity |
| `p_only` | All measured P logical CPUs |
| `p_priority_e` | All measured P logical CPUs plus `ROOP_CPU_E_LIMIT` E CPUs (default one quarter) |
| `p_plus_e` | All measured P and E logical CPUs |

Explicit policies use the measured OS affinity indices. OpenCV optimized
dispatch remains enabled, while its internal pool, ORT intra/inter-op pools,
and FFmpeg threads stay independently bounded to avoid nested
oversubscription. CUDA/TensorRT resources remain selected by the GPU/workload
profile; CPU policy does not widen GPU contexts or streams.

On the current host the detected CPU profile is 24 physical / 32 logical,
with 8 P-core physical / 16 P logical and 16 E logical CPUs. Current/max
reported frequency was 3,200 MHz, NumPy exposed AVX/AVX2/FMA/SSE families,
and process affinity was available. This is a runtime observation, not a
hard-coded i9 assumption.

## Validation status

The code and targeted CPU/GPU optimizer tests pass. The complete repository
suite passes **1,430 tests, 1 skipped, 589 subtests**.

The first controlled RTX 4070 run used the real `d4.mp4` two-face workload,
GPEN 256 Pro, RealityUX, stabilization, tracking, and the existing NVENC/
FFmpeg pipeline. It reached provider/model preparation but did not finish its
120-frame render within the practical run window and was terminated. No FPS,
quality, or resource number from that partial run is accepted below.

### RTX 3060 — separate result table

| Metric | Result |
|---|---|
| Physical validation | **PENDING** — no RTX 3060 was available in this session |
| Baseline FPS | PENDING |
| P-only FPS | PENDING |
| P-priority + limited E FPS | PENDING |
| P+E FPS | PENDING |
| VRAM / CPU / GPU / stage throughput / latency | PENDING |
| Quality / stability | PENDING |
| Required rerun | `env\\Scripts\\python.exe tests\\gate_d_cpu_benchmark.py --target "RTX 3060" --end 120` |

### RTX 4070 — separate result table

| Metric | Result |
|---|---|
| Physical validation | Available; runtime detected NVIDIA GeForce RTX 4070, Ada, SM 8.9, 11.994 GiB |
| Baseline FPS | PENDING — controlled run timed out before completion |
| P-only FPS | PENDING — not started after baseline timeout |
| P-priority + limited E FPS | PENDING — not started after baseline timeout |
| P+E FPS | PENDING — not started after baseline timeout |
| VRAM / CPU / GPU / decode / inference / enhancement / encode / latency | PENDING for Gate D candidate matrix |
| Quality / stability | PENDING for Gate D candidate matrix |
| Required rerun | `env\\Scripts\\python.exe tests\\gate_d_cpu_benchmark.py --target "RTX 4070" --end 120 --timeout 1800` |

The previously locked Phase 2 4070 production baseline remains **9.62 FPS**
for the 600-frame workload, with 233.34 ms worker-time latency. It is a
historical baseline, not a Gate D CPU-policy result and is not used to claim a
Gate D improvement.

## Acceptance classification

| Change | RTX 3060 | RTX 4070 | Classification |
|---|---|---|---|
| Runtime CPU topology/frequency/SIMD/affinity detection | Logical validation only; physical benchmark pending | Runtime probe passed; E2E benchmark pending | **PENDING** |
| Explicit P/E policy selection and affinity | Logical validation only; physical benchmark pending | Runtime probe passed; E2E benchmark timed out | **PENDING** |
| OpenCV optimized dispatch with bounded internal threads | Logical test passed; physical benchmark pending | Code path exercised only during timed run | **PENDING** |

No optimization is classified as A–F until both physical target result rows
contain valid end-to-end measurements. A completed rerun must report one row
per policy and include FPS, peak/average VRAM, CPU/GPU utilization, decode,
inference, enhancement and encode throughput, latency, stability, and output
quality. Sustained CPU frequency and temperature are reported when the OS
exposes them; unavailable thermal/power fields remain explicitly unavailable.
