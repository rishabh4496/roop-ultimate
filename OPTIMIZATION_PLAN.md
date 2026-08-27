# Performance Optimization Plan

## Scope

Optimize maximum stable real-world video throughput for the supported dual-device profiles without quality regression:

- RTX 4070 desktop, 12 GB VRAM, 32 GB RAM, i9-14900K.
- RTX 3060 Laptop, 6 GB VRAM, 16 GB RAM.

User-selected settings remain authoritative. Automatic tuning is allowed only for unset or `auto` settings, with bounded resources, bounded queues, CPU oversubscription protection, and safe CPU fallbacks.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Baseline, quality guards, hardware profiles, and benchmark foundations | Complete; repository history before `1cae1de` |
| 1 | End-to-end runtime audit and bottleneck map | Complete; recorded in repository audit commits |
| 2 | Centralized runtime optimizer and CPU scheduling integration | Complete; `d298fbf` |
| 3 | TensorRT engine/context resource management, reuse, warmup, pressure handling, and context-knee benchmarking | In progress in working tree |
| 4 | Further execution-path optimizations only after Phase 3 evidence | Not started |

## Phase 3 acceptance criteria

1. Model- and shape-aware resource estimates; no VRAM-only constant rule.
2. Explicit pool settings are never rewritten by automatic tuning.
3. Context counts are measured at 1/2/3/4 and selected at the throughput knee.
4. Active leases are never destroyed during execution; pressure reduces future admission safely.
5. TensorRT, mixed precision, FP32 fallback, and CUDA fallback behavior remain available.
6. Engine/provider paths remain reusable and no unnecessary engine recreation is introduced.
7. Tests pass for lifecycle, explicit overrides, pressure, portability, and benchmark behavior.
8. Real-video end-to-end results are recorded separately from synthetic stage benchmarks.

Phase 4 must not begin until the Phase 3 working tree is reviewed, committed, and validated on both hardware tiers.
