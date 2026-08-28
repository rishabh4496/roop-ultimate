# Phase 6 — CUDA Streams and CUDA Graphs

**Recorded:** 2026-08-28

This phase keeps CUDA execution hardware-adaptive. The RTX 3060 Laptop and
RTX 4070 Desktop are separate validation identities; no stream, graph, or
benchmark result is reused between them.

## Runtime policy

`roop.runtime_optimizer.CUDAGraphManager` exposes a bounded stream policy:

| Capability profile | Application streams | TensorRT auxiliary streams | Safe overlap rule |
|---|---:|---:|---|
| detected VRAM below 7 GB | 1 | 0 | serial; protects the RTX 3060 memory/RSS tier |
| detected VRAM at least 7 GB | at most 2 | at most 1 | only independent work with distinct contexts and no shared mutable buffer |

The policy uses detected VRAM/CUDA availability and workload independence,
not GPU model names. Explicit settings remain authoritative. The existing
TensorRT `-1` auxiliary-stream setting continues to delegate to TensorRT when
the user has not selected a bounded value; the new runtime profile records the
bounded recommendation without silently overriding that explicit setting.

## Candidate audit

| Candidate | Shape/dependency audit | Action |
|---|---|---|
| TensorRT auxiliary streams | TensorRT owns the internal streams. Contexts are distinct only when the existing session pool leases distinct sessions. | Accepted as a bounded policy/diagnostic recommendation; no extra application streams created. |
| LivePortrait appearance + motion | Independent model sessions can overlap. The appearance→warping device buffer has a real dependency. | Existing overlap retained; `synchronize_outputs()` remains mandatory before warping. A second motion context remains opt-in and VRAM-bounded. |
| GPEN 256 Pro GPU filter | Fixed candidate shapes are possible, but input uploads, output CPU transfer, and a low-texture branch dominate the small filter. | Graph runner implemented behind `ROOP_CUDA_GRAPH_FILTER=1`, but rejected for production after A/B. |
| UltraMax filter | `gain`, `sigma`, target size, input addresses, and host transfer vary per call. | Rejected; normal FP32 path retained. |
| Enhancement pipelines | Model output, color transfer, masks, stabilization, and paste buffers have ordered dependencies and mutable ownership. | Rejected; no unsafe cross-stage stream sharing. |
| Repeated face inference | Existing per-context `SessionPool` is the safe unit of concurrency. | Retained; no shared execution context or global synchronization added. |
| Upscaling tiles | Tile dimensions and paste destinations can change; the shared IO binding is not thread-safe. | Rejected for graphs/extra streams; existing parallel path uses independent sessions/plain runs. |

## CUDA Graph runner and invalidation

`CUDAGraphRunner` warms each static input, synchronizes once before capture,
captures one fixed execution path, and replays by copying into stable input
addresses. Replay does not add a device-wide synchronization. A runner is
owned by one worker thread because its static buffers are mutable.

The GPEN 256 Pro candidate key invalidates on:

- model/filter identity;
- input shape and batch dimensions;
- NumPy strides/dtype and CUDA device;
- target-size/configuration changes;
- runtime profile and TensorRT schedule identity;
- precision identity.

Release also drops the runner. Any capture or replay failure returns to the
existing FP32 GPU filter and then the established CPU fallback.

## RTX 4070 validation

Hardware/software: RTX 4070, SM 8.9, 11.99 GiB VRAM, CUDA 12.8,
TensorRT 10.9.0.34, ONNX Runtime 1.23.2, driver 610.88.

The GPEN 256 Pro filter was warmed, captured, and replayed:

| Arm | Mean filter time | Output result |
|---|---:|---|
| normal FP32 GPU filter | 1.67 ms | reference |
| captured/replayed graph | 2.06 ms | finite, max difference 0, including low-texture grain case |

Replay was approximately 23% slower because each call still needs host→device
input copies and a device→host output copy. It is therefore not enabled by
default and is not counted as an FPS gain.

The unchanged end-to-end quick benchmark measured:

- enhanced composite: **31.23 FPS**;
- heavy composite: **23.93 FPS**;
- stage samples: approximately **9.8–10.3 GB free VRAM**.

The provider-level `ROOP_TRT_CUDA_GRAPH=1` arm did not reach its first detector
steady-state result after approximately two minutes while building/capturing
the fresh graph cache. No FPS or quality pass is claimed for that arm; the
global TensorRT graph option remains opt-in and is not treated as a production
optimization.

## RTX 3060 validation

**PENDING — physical RTX 3060 Laptop was unavailable in this session.** No
RTX 4070 timing, graph cache, or stream result is copied to the RTX 3060.

Required exact test on that device:

1. Run the same capability probe and stream-policy report.
2. Run GPEN 256 Pro normal vs warm/capture/replay with the same input shapes.
3. Verify output difference, finite/collapse guards, capture invalidation, and
   graph failure fallback.
4. Run the identical real-video workload and record end-to-end FPS, latency,
   VRAM, RAM/RSS, GPU utilization, and synchronization/queue behavior.
5. Rerun the unresolved Phase 4 strict `<2.5 GB RSS` two-face gate.

The prior RTX 3060 Phase 4 result remains approximately 2.82–2.83 GB RSS and
blocked. This Phase 6 change does not erase that blocker.

## Decision

The accepted Phase 6 result is bounded, capability-driven stream policy plus
preservation of the already-safe LivePortrait/session-pool overlap. The graph
runner is implemented with complete invalidation and fallback semantics, but
the measured GPEN 256 Pro graph is rejected from the default runtime because
it reduces per-filter throughput and has no demonstrated end-to-end FPS gain.

## RTX 4070 stream A/B follow-up

The same warm `roop.bench --profile quick --no-apply` workload was run with
`ROOP_TRT_AUX_STREAMS=0` and `ROOP_TRT_AUX_STREAMS=1`, with global TensorRT and
GPEN filter graphs disabled. The runs used separate cache namespaces:

| Setting | GPU work/frame | Enhanced composite | Heavy composite | Standard composite | VRAM free during stages |
|---|---:|---:|---:|---:|---:|
| aux=0 | 50 ms | 32.58 FPS @ 12 threads | 29.11 FPS @ 12 threads | 70.95 FPS @ 32 threads | 9.8–10.3 GB |
| aux=1 | 51 ms | 29.57 FPS @ 12 threads | 24.08 FPS @ 12 threads | 66.59 FPS @ 32 threads | 9.6–10.3 GB |

Detector/recognition/landmark/swap/enhancer stage rates were similar, but aux=1
did not improve the composite and reduced the heavy workload. RealityUX DFL
XSeg reported invalid throughput in both arms while BiSeNet remained finite;
that benchmark issue is retained as an error row, not converted to a zero.
The 4070 decision is therefore to retain capability-driven bounded policy and
the user/configurable TensorRT setting, while not enabling aux=1 as the default.
No RTX 3060 conclusion is drawn from this A/B.
