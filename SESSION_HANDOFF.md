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

**Current phase:** PHASE 1 — Repository Audit + Architecture Mapping

**Phase status:** NOT STARTED

**Last completed phase:** NONE

**Immediate next action:**
Perform the repository audit and map the complete video-processing execution path. Do not make optimization changes until the bottlenecks and responsible files/functions are documented.

---

# LAST SESSION SUMMARY

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

Start with Phase 1 only.

Do not jump directly to TensorRT, CUDA, CPU, RAM, Rubin, or scheduler implementation.

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
