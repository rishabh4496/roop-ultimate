# ROOP-ULTIMATE SESSION HANDOFF

## PURPOSE

This is the file Codex reads first when resuming work in a new session.

The repository files and benchmark evidence are authoritative. Conversation history is not authoritative.

---

# RESUME PROTOCOL

Before doing anything:

1. Read `OPTIMIZATION_PLAN.md`.
2. Read `OPTIMIZATION_PROGRESS.md`.
3. Read `PERFORMANCE_BASELINE.md`.
4. Read this file.
5. Run `git status`.
6. Inspect the current diff.
7. Identify the first incomplete phase.
8. Continue from that exact phase.
9. Do not redo completed phases unless regression evidence requires it.

---

# CURRENT SESSION STATE

**Session:** 1

**Current phase:** PHASE 4 — TensorRT Engine Optimization

**Phase status:** RTX 4070 complete; physical RTX 3060 Laptop gate is blocked

**Last completed phase:** PHASE 3 implementation, checkpoint `c439e43`

**Immediate next action:**
Resume Phase 4 work on the physical RTX 4070 system. Preserve the documented
RTX 3060 Laptop RSS failure; do not start Phase 5 until the outstanding strict
laptop gate is explicitly resolved.

---

# LAST SESSION SUMMARY

Phase 4 was implemented and validated on the physical RTX 4070. The audit
preserved TensorRT cache/timing identity, precision namespaces, context and
session safety, IO binding, VRAM admission, and FP32 fallbacks. The benchmark
found model-specific knees rather than a universal context count: detector 6,
recognition/landmarks/masks/enhancer 2, and swapper 3. Heavy composite was
20.55 FPS at `trt_pool=3` versus 17.96 FPS at 6, so 6 was rejected.

The real-video run completed in 46.9 seconds at 5.12 end-to-end FPS and
195.4 ms/frame, with 7,984 MB peak and 4,764 MB sampled-average VRAM. CPU
telemetry was unavailable. This was throughput/resource evidence only because
the selected source did not match the fixture.

The 2026-08-28 physical RTX 3060 gate was attempted and is blocked. Hardware
was confirmed as an RTX 3060 Laptop with 6,144 MiB VRAM and 16 GB RAM. The
fresh benchmark exercised the TensorRT/CUDA stack and model-specific context
matrix. Auto low-VRAM policy selected `ROOP_TRT_POOL=0` and
`ROOP_DETMASK_POOL=0`; composite benchmarking was rejected after the 1.25 GB
VRAM reserve was no longer available. After the benchmark guard fix, DFL XSeg
reports valid throughput at 27.0 calls/s for one context.

The real d4.mp4 workload used the configured two faces, RealSwap, GPEN 256 Pro,
RealityUX, tracking, stabilization, and six requested threads. The sub-7GB
policy clamped execution to one worker and preserved the 1,536 MB stabilization
cap with adaptive 16-frame chunking. The completed 200-frame RealSwap run
reached final encoding with correct attribution, but sustained RSS was about
2.83 GB. A bare two-face RealSwap probe remained about 2.66 GB, and the full
GPEN/RealityUX path remained about 2.86 GB. The laptop requirement is strictly
below 2.5 GB RSS.

The latest revalidation used commit `8145c10`. Pinokio reached `online/ready`,
the bounded workload encoded 200/200 frames and produced 369 audit rows, and
RSS remained approximately 2.82–2.83 GB. The RTX 3060 gate therefore remains
blocked; the next session continues on the RTX 4070 system.

Full tests now run 1,346 cases with 1 skipped and all passing. The focused
runtime/benchmark/backend/stabilization sweep passes 89 tests; Python compile,
launcher syntax, and diff checks pass. The remaining issue is physical RSS,
not a software-test regression. Phase 5 was not started and FP32 safeguards
were not changed.

---

# WORK IN PROGRESS

Physical RTX 3060 Phase 0–4 gate is blocked only by the strict RSS ceiling. The
DFL XSeg measurement and prior test/catalog regressions are resolved.

---

# FILES CURRENTLY BEING MODIFIED

Validation fixes and low-VRAM runtime safety changes are present in
`app/roop/`, benchmark/test parity changes are in `app/tests/`, and the
settings catalog and handoff records are updated. The pre-existing
`.geminiignore` change is retained.

---

# TESTS CURRENTLY PASSING

Phase-focused sweep: 89/89 passed.

Full suite: 1,346/1,346 passed, 1 skipped.

---

# CURRENT PERFORMANCE

User-reported maximum:
**~20 FPS**

Controlled baseline:
**RTX 3060 gate did not complete; no valid end-to-end FPS result**

---

# KNOWN BLOCKERS

- The completed two-face RealSwap path is approximately 2.66–2.83 GB RSS on
  the 16 GB laptop, above the required 2.5 GB ceiling.
- Composite benchmark admission rejects the measured 6 GB configuration after
  the 1.25 GB VRAM reserve is consumed.

---

# NEXT SESSION INSTRUCTION

Continue Phase 4 work on the physical RTX 4070 system using the current
repository state. The RTX 3060 RSS gate remains the first incomplete acceptance
gate and must be explicitly resolved before Phase 5 precision work.

Do not start Phase 5 precision optimization, and do not change the existing
FP32 safeguards before the RTX 3060 gate is documented.

First establish:
- execution graph,
- exact bottleneck locations,
- current resource usage,
- current synchronization points,
- current model/session lifecycle,
- current video pipeline.

Then update `OPTIMIZATION_PROGRESS.md` and this file with findings and the exact next action.

---

# PHASE TRANSITION RULE

When a phase is complete:

CURRENT PHASE → COMPLETE
NEXT PHASE → IN PROGRESS

Record:
- files/functions changed,
- tests,
- benchmark,
- before/after FPS,
- resource metrics,
- regressions,
- Git checkpoint,
- next action.

Never leave the repository state ambiguous.
