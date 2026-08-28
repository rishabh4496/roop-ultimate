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

**Session:** 0

**Current phase:** PHASE 4 — TensorRT Engine Optimization

**Phase status:** RTX 4070 complete; physical RTX 3060 Laptop gate outstanding

**Last completed phase:** PHASE 3 implementation, checkpoint `c439e43`

**Immediate next action:**
Validate Phases 0 through 4 on the physical RTX 3060 Laptop, including the
Phase 4 model-specific 1/2/3/4/6 context matrix and real-video workload. Do
not start Phase 5 until this gate is complete.

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

No optimization session has been completed yet.

---

# WORK IN PROGRESS

None.

---

# FILES CURRENTLY BEING MODIFIED

None.

---

# TESTS CURRENTLY PASSING

Not established.

---

# CURRENT PERFORMANCE

User-reported maximum:
**~20 FPS**

Controlled baseline:
**Not yet established**

---

# KNOWN BLOCKERS

None yet. Phase 1 must identify the actual bottlenecks.

---

# NEXT SESSION INSTRUCTION

Run the physical RTX 3060 Laptop validation for Phases 0 through 4. This is
the first incomplete gate and must be completed before Phase 5 precision work.

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
