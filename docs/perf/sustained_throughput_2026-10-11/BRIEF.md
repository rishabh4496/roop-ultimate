# Sustained-throughput investigation: shared brief (2026-10-11)

Branch `perf/sustained-throughput`, base commit `b3189690fcd1fb3049d6f7ee283fe6a1fcb76327` (clean tree at start).
Repo root: `G:\pinokio\api\roop-ultimate`. Python: `app\env\Scripts\python.exe` (3.10.20).
Clips and the running status log: `G:\pinokio\roop-keep\` (`RECODE_STATUS.md` newest section first, `Love.mp4`, `b1.mp4`, `Weeds.mp4`, `single\`, `double\`, `final\`).

## Reported symptom (NOT a guaranteed result)
Video face swap starts near 25 fps and decays toward 10 fps; GPU utilisation falls as it goes.

## Prior art you MUST read first (do not re-derive it blindly, but do re-test it)
- `docs/CHANGELOG.md` entry **2026-10-04, "The fps collapses after a while..."**: on a 54,714-frame job fps went 38 -> 7.8 -> 16.6 (NOT monotonic). Fresh-process re-renders of the same frames gave the same fps for the same position, so process age was excluded. Swapped faces/s was flat (18-21). Cost model: ~8 ms/frame + ~41 ms per swapped face, GIL held in 95% of py-spy samples, GPU ~30%. Conclusion then: footage got busier, not a leak. That is a hypothesis to CONFIRM OR REFUTE with new data, not a fact to inherit.
- `docs/CHANGELOG.md` 2026-10-04 side finding: jobs 3+ in one backend start with ~2.2 GB less free VRAM and the VramGovernor steps swap batch 8->4->2->1. Never chased. Candidate for a real across-job degradation.
- `docs/TROUBLESHOOTING.md` "GPU memory pressure": past ~95% VRAM the Windows driver spills to shared memory and fps collapses while utilisation still reads high.
- `docs/development/STAGE10_PIPELINE_CONCURRENCY.md`, `docs/ENV_FLAGS.md` (ROOP_PROFILE, ROOP_PROFILE_DETAIL, ROOP_RUNTIME_MONITOR), `docs/perf/stab_warmup.md`.
- `AGENTS.md` Part 1 benchmarking rules are binding (see below).

## Binding project rules (from AGENTS.md)
1. The pipeline is GPU-bound and host/GIL-bound in places; ROOP_PROFILE per-stage % is thread time SUMMED across workers, NOT a speedup budget. Only REMOVING work moves the render clock.
2. 600 frames minimum for any acceptance claim. 120 frames measures warm-up.
3. Counterbalance A/B (A/B then B/A). First arm pays the TensorRT engine build. Run a null control first (same config twice).
4. Read `app/config.yaml` live; pass swap model, mask engine, provider, detector, threads explicitly. `--threads 20` on bench runs. Use `tests/config_sync.py` / `init_pipeline(sync_config=True)`.
5. Read faces/s beside fps. fps falling with faces/s flat means busier footage, not a regression. Read SWAP RATE beside fps (a faster setting that finds fewer faces is not faster).
6. `faces_seen`: 679 = one sequential pass, >750 = parallel stabilizer on the locked fixture. Pin `ROOP_STAB_CHUNK_MB` for any fidelity or A/B comparison and read the `[Stabilize] parallel: ... blocks x Nf` line.
7. ONE render at a time. A render holds 12-15 GB RAM and most of the 12 GB VRAM. Anything else that loads models kills it.
8. Measure model cost only through `angle_bench.init_pipeline`, never a bare python process (TensorRT DLLs off PATH => silent CPU fallback).
9. "Reports success while not running" is this project's recurring failure class. Before believing "no effect", prove the code path executes. Count the stage, not the wrapper.
10. `app/settings.py` is the single source for a setting; new setting => `UI_SETTINGS`, `ENV_SETTINGS`, `python tools/gen_settings.py`, commit generated files. Do not edit with PowerShell Get-Content/Set-Content. Use Edit/apply_patch.
11. Never write a drive letter into a tool. Resolve PINOKIO_HOME.
12. Hardware: MAIN = RTX 4070 12GB, 31.7 GB RAM (this machine). The 3060 laptop is NOT available here; its look settings are deliberate; do not "realign". Anything touching it is UNVERIFIED.
13. Negative results are deliverables: record what was tried, numbers, and why rejected.

## Change control (from the mission)
- Do NOT commit, push, or force-push. Do NOT reset, clean, or delete user data, models, caches, clips, outputs.
- Static-analysis phase is READ-ONLY on source. Write only to `docs/perf/sustained_throughput_2026-10-11/` (your own report file) and to scratch under `%TEMP%\sust_tp\`.
- Only the lead edits source. Propose diffs in your report instead.
- No fabricated numbers, line numbers or profiler output. Every claim cites `file:line` read from the actual file, or a measurement with its raw artifact path. Say "unverified" when you did not run it.
- GPU use: only Agent B (Profiling) may run anything that loads models or touches the GPU, and only after announcing it to the lead. Everyone else is static/CPU-light (no model loads, no renders, no pytest that loads the GPU stack).

## Registers
- Issue register: `docs/perf/sustained_throughput_2026-10-11/ISSUES.md` (lead-owned; send findings to the lead, do not edit it).
- Report file per agent, named in your task.

## Report format (every agent)
Findings as a table: ID | file:function:lines | evidence (code quote <= 3 lines or artifact path) | mechanism | confidence (confirmed / strongly supported / plausible / rejected) | progressive-decline or constant-limit | proposed minimal fix | risk. Then a short "what I did NOT verify" list.
