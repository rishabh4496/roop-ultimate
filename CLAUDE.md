# Roop Ultimate — working guide

**Pinokio launcher rules are NOT repeated here.** They live in `G:\pinokio\CLAUDE.md`,
which Claude Code loads automatically from the parent directory, with the API reference
at `G:\pinokio\prototype\PINOKIO.md` and examples in `G:\pinokio\prototype\system\examples`.
This file used to carry a stale copy of that guide; it was removed on 2026-09-14, not lost.

Everything below is about the **face-swap app itself**, which is what work in this repo
is almost always about.

Full session history: **`docs/SESSION_LOGS.md`** (22 sessions, 2026-08-22 → 09-01,
verbatim). Read it when you need the evidence behind a rule below, or when a number you
are about to quote came from before that date.

---

## Start every session here

1. **Read `G:\pinokio\roop-keep\RECODE_STATUS.md`, newest section first.** It is the
   running state of a multi-session recode. The top section is current; older sections go
   stale — when a table and a prose summary disagree, trust the table.
   On the SECONDARY device (3060, under `C:\pinokio\`) there is no G: drive; `roop-keep`
   exists there with the clip folders but holds no `RECODE_STATUS.md`, so
   `docs/SESSION_LOGS.md` is the whole record. Do not conclude the state is unknown.
2. **Phase numbering stopped at PHASE 5 and the work moved past it.** Phase 4
   ("RealSwap": hyperswap base + a hififace eyelid/eyelash band) SHIPPED and is the live
   default (`swap_model: realswap`). Phase 5 ("UltraMax") shipped, was rebuilt lean, and
   had its clarity filter moved into the merger chain; the enhancer line continued past
   the phase framing into GPEN Realistic and GPEN 256 Pro. Phases 1 (detection) and 2
   (mask engines / RealityUX) are closed bar one checklist cell. Phase 3's headline ask
   (interacting faces) is characterized but unsolved.
3. **The newest sessions are not phase work** — validation campaigns on both GPUs. Read
   the last two entries of `docs/SESSION_LOGS.md` for the current ground, and
   `docs/PHASE_HANDOFF.md` / `docs/FINAL_VALIDATION_MATRIX.md` for next steps.

## Two machines, and a number from one is not a number from the other

| | MAIN | SECONDARY |
|---|---|---|
| GPU | RTX 4070 12GB | RTX 3060 Laptop 6GB |
| root | `G:\pinokio\` | `C:\pinokio\` |
| RAM | 31.7 GB | 15.8 GB |
| resolution at 600 frames | ~4-5% spread | ~3.3% spread |

`config.yaml` never syncs between them; code does. The 3060's look settings diverge from
the 4070's **deliberately** (tuned by eye for GPEN 256 Pro) — do not "realign" them.
TensorRT is not admitted on the 3060, so every precision arm there is inert. Mixed
precision is 4070-specific (+20% there, noise here).

---

## The standing frame: the pipeline is GPU-BOUND

Three stage-level wins in a row (stabilizer round scheduling, `temporal_detection`,
`det_size` 640->512) each measured well in isolation and NEUTRAL end to end. **A change
moves the render clock only if it REMOVES GPU work, not if it redistributes thread time.**

`ROOP_PROFILE`'s per-stage % is thread time SUMMED ACROSS WORKERS and is **not a speedup
budget** — detect read 42.4% and making it cheaper bought +1%.

Threads, contexts, queues and affinity are each now measured as dead ends (Gate E: 0.7%
across a 5x thread range, nothing saturated). **The only productive direction left is
removing work per face.**

## Benchmarking rules that are not optional

- **600 frames minimum for any acceptance claim.** 120 frames measures warm-up, proven on
  both GPUs *in opposite directions* (3060 +19.6% -> -0.5%; 4070 -10.7% -> +1.4%).
- **Counterbalance every end-to-end A/B (A/B then B/A).** The first arm pays the TensorRT
  engine build; without counterbalancing, two measured-neutral results read +21.8% and
  +9.8%. Use `tests/ab_temporal_detection.py --vary <globals key> --a <x> --b <y>`.
- **Run a null control first, every session** — the same config two or three times. The
  4070 resolves ~50% effects reliably and ~5% effects not at all; a 1% effect would need
  ~25 arms a side. Say "not measurable" rather than reporting what the noise produced.
- **Bench the models the user actually runs.** Read `app/config.yaml` live and pass swap
  model, mask engine, provider, detector and threads explicitly. Violated twice,
  invalidating whole sessions. `tests/config_sync.py` is now the one place this happens —
  `init_pipeline(sync_config=True)`.
- **`--threads 20`** on every bench run.
- **Report processing fps to the user roughly every 3 minutes** during a run.
- **One render at a time.** A render holds ~12-15 GB; anything else that loads models
  kills it.
- **Measure model cost through `angle_bench.init_pipeline`**, never a bare python process:
  without the app's init, TensorRT's DLLs are off PATH, ORT silently falls back to CPU,
  and a 4 ms model reports as 210 ms.
- **Read SWAP RATE beside fps.** A setting that goes faster by finding fewer faces has not
  got faster.
- **`faces_seen` is a free code-path discriminator** on the locked fixture: 679 = one
  sequential pass, >750 = parallel. Free RAM decides which path an arm takes
  (`_default_stab_chunk_mb` is `available * 0.40 / 6`), and the null drifted 2.9x
  mid-session on that alone. Record it beside fps, or a fallback is indistinguishable from
  a regression.

## The failure class this project keeps hitting

**Something reports success while not running.** Fifteen-plus instances now: four
enhancers failing on 60 of 60 frames at "100% swapped"; the adaptive enhancer restoring
nothing while posting the *fastest* row; `_build_temporal_faces` dedented out of its loop
so the default path stopped swapping and read **+47% faster**; the adaptive controller
wired to a writer production never uses; four benches running their "CodeFormer" arm with
no enhancer at all.

The instruments that keep missing it:

- **the swap audit counts INTENT**, over the faces it was handed — not outcome, and not
  the faces in the clip;
- **the return code and integrity sweeps pass** — an unswapped frame is a valid picture;
- **the test suite stays green** — it was green through every defect above.

So: **before believing "no effect", prove the code path executes.** A flag that is read, a
controller that is constructed and a hook that is defined are all consistent with never
running. Count the stage, not the wrapper.

The **pixel noise floor is 0.7142/255 mean, 22/255 max** between two renders of one
unchanged config (non-deterministic GPU reduction order; survives threads=1, CUDA instead
of TensorRT, and `PYTHONHASHSEED=0`). A delta at or below that is **not** evidence a
feature ran.

## The discipline that keeps being learned the hard way

- **Before tuning a gate, measure the distribution that gate actually reads.** Five gate
  changes have now been implemented and reverted because the population was not in the
  band the change targeted. Reasoning from a debug table to a failure is not evidence.
- **Verify a "pure refactor" is pure**: re-run a clip end to end and compare `rows.csv` by
  hash, not by eye. `tests/surface_snapshot.py` exists for this.
- **A negative result is a deliverable.** Record what was tried, the numbers, and why it
  was rejected — in the code at the site and in `RECODE_STATUS.md`. Days have been spent
  re-attempting ideas that were already measured and thrown away.
- **Ship the fix, not the flag.** A feature that defaults off is off for everyone; if a
  change is right, default it on and prove no regression.
- **Never write a drive letter into a tool.** Three harnesses hardcoded the other
  machine's `PINOKIO_HOME` and either died or silently skipped. Resolve it.
- **A derived default that outlives its rule is invisible.** `Settings.save()` writes
  derived values byte-identically to user-typed ones, so an improved formula could not
  reach any install that had ever saved. Provenance stamps (`_threads_auto`,
  `_threads_basis`) now carry the distinction; **bump `_THREAD_RULE` whenever
  `default_threads` changes**, or existing installs silently keep the old formula.

---

## Repo and UI

- **Commit and push to the GitHub fork without asking.** Run the full suite first:
  `env/Scripts/python.exe -m unittest discover -s tests -t . -p "test_*.py"` from `app/`.
  Current baseline: **2607 tests, 1 skipped, OK** in ~161 s (measured 2026-09-14). The
  session logs quote 1698 and a pair of `test_nvdec_reader` ffmpeg-spawn errors; both are
  stale.
- Pytest-style tests are invisible to unittest (`Ran 0 tests ... OK`) — use
  `tests/unittest_shim.py`.
- `G:\pinokio\roop-keep\` is NOT a git repo — `RECODE_STATUS.md` is saved by editing it,
  not by committing.
- The Gradio UI under `app/ui/` is **frozen**. All new UI work is the React app in
  `react-ui/`. `api.py` is a non-reloading uvicorn thread — the backend needs a restart
  for changes to take.
- A new setting must be registered in **three** places (`settings.py`, the panel, and
  `settingsCatalog.js`) and, if it drives a `ROOP_*` flag, mapped in
  `run.py::_apply_perf_env`. **Grep that something actually READS it** — a control bound
  to a value nothing consumes looks completely wired.
- Never edit source with PowerShell `Get-Content | Set-Content` (UTF-8 corruption). Use
  Edit, or Python.
- Stop the app before touching the venv — a running app locks ONNX/CUDA DLLs.

## Open, inherited

1. **Interacting faces** — Phase 3's headline ask, characterized but unsolved.
2. **Phase 3's RSS gate fails on the 3060** at 3.46-3.73 GB against `<2.5 GB`. Measured
   and explained (stabilization costs +886 MB, the enhancer +428 MB, and turning
   stabilization off loses 27% of swaps). Needs a decision, not a measurement.
3. **Real occluder and real night footage** — Phases 10 and 14 are both synthetic.
4. **Re-baseline everything through `two_face_video.py`** — 28 keys diverged before the
   config sync, so every pre-2026-09-01 absolute value is suspect (ratios survive).
5. **Twelve settings have no UI** — `identity_detail_strength`, `temporal_compositing_*`,
   `temporal_quality_*`. Deliberately not added; their own handoffs record them incomplete.
