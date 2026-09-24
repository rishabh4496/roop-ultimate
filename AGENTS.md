# Roop Ultimate — agent rules (single source of truth)

This is the one rule file. `CLAUDE.md`, `GEMINI.md`, `QWEN.md`, `.cursorrules`, `.clinerules`
and `.windsurfrules` only point here. **Part 1** is about the face-swap app itself, which is
what work in this repository almost always is; **Part 2** is the Pinokio launcher development
guide and the two hardware profiles every performance change must respect.

Full session history: **`docs/SESSION_LOGS.md`** (verbatim). Read it when you need the
evidence behind a rule below, or when a number you are about to quote came from before
the date on it. Dated behaviour changes: `docs/CHANGELOG.md`. Tests and CI:
`docs/development/CONTRIBUTING.md`.

---

# Part 1 — Working on the app

## Start every session here

1. **Read `<MEDIA_DIR>\RECODE_STATUS.md`, newest section first.** It is the
   running state of a multi-session recode. The top section is current; older sections go
   stale — when a table and a prose summary disagree, trust the table.
   On the SECONDARY device (3060) the main machine's drive does not exist; `roop-keep`
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
| root | `<PINOKIO_HOME>` (one drive) | `<PINOKIO_HOME>` (a different drive) |
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
- **After touching matching, selection or the frame loop, run the regression benchmark**
  (`python run.py --benchmark --benchmark-mode regression`, ~3 min on the 4070). It is the
  only check that fails when a render stops swapping: 4bd577d shipped a render that swapped
  nothing with the suite green. See `docs/development/REGRESSION_BENCHMARK.md`.
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
  `app/env/Scripts/python.exe -m pytest` from the repo root (pytest is the runner since
  2026-09-22; `unittest discover` drops the pytest-style tests). Light profile without the
  GPU stack: `ROOP_TEST_LIGHT=1 python -m pytest -m "not gpu"` (~20 s; what CI runs).
  Current baseline: **3458 passed, 6 skipped, 2 xfailed** over both trees (2026-09-25,
  pytest; 8-20 min). Any number quoted from the session logs (1698/2607, 3064, 3190) is stale.
- Pytest-style tests are invisible to unittest (`Ran 0 tests ... OK`) — another reason
  the runner is pytest; `tests/unittest_shim.py` is only for the legacy command.
- `<MEDIA_DIR>\` is NOT a git repo — `RECODE_STATUS.md` is saved by editing it,
  not by committing.
- The Gradio UI under `app/ui/` is **frozen**. All new UI work is the React app in
  `react-ui/`. `api.py` is a non-reloading uvicorn thread — the backend needs a restart
  for changes to take.
- **`app/settings.py` is the single source for a setting.** Adding one (since 2026-09-22):
  1. `Settings._load()`: `self.x = self.default_get(data, 'x', default)` and the entry in
     `save()` — the name and default live here and nowhere else.
  2. `UI_SETTINGS` (same file): `('x', 'Label', 'Section')` if the React panel exposes it.
     Then bind it in `Settings.jsx` with `bind('x')` / `bindToggle('x')` — the panel's
     controls are still hand-written JSX; `test_ui_settings_catalog.py` keeps the two in step.
  3. `ENV_SETTINGS` (same file): `('x', 'ROOP_X', kind)` if a `ROOP_*` flag reads it;
     `settings.apply_env` applies it in `run.py` and the comparison benches. Do NOT add a
     `_set('ROOP_...` line anywhere — `test_bench_perf_env.py` rejects a private copy.
  4. Run `python tools/gen_settings.py` and commit `app/settings.schema.json` and
     `react-ui/src/components/settingsCatalog.js` with the change. Both are GENERATED;
     `test_settings_schema.py` (and CI's `--check` step) fail while they are stale.
  5. **Grep that something actually READS the flag** — a control bound to a value nothing
     consumes looks completely wired (`test_identity_settings_wiring.py` checks the
     identity ones; the rule is general).
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

---

# Part 2 — Pinokio launcher development guide

The Pinokio API reference is `<PINOKIO_HOME>/prototype/PINOKIO.md` and the reference
launchers are `<PINOKIO_HOME>/prototype/system/examples`; resolve `PINOKIO_HOME` as
described below, never assume a drive letter.

## Non-Negotiable Execution Workflow

To guarantee every contribution follows this guide precisely, obey this checklist **before any edits** and **again before finalizing**. Do not skip or reorder.
1. **AGENTS Snapshot:** Re-open this file and write down (in your working notes or response draft) the exact sections relevant to the requested task. No work begins until this snapshot exists.
2. **Example Lock-in:** Identify the closest matching script in `<PINOKIO_HOME>\prototype\system\examples`. Record its path and keep it open while editing. Every launcher change must mirror that reference unless the user explicitly instructs otherwise.
3. **Pre-flight Checklist:** Convert the applicable rules from this document and `PINOKIO.md` at <PINOKIO_HOME>\prototype\PINOKIO.md into a task-specific checklist (install/start/reset/update structure, regex patterns, menu defaults, log checks, etc.). Confirm each item is ticked **before** making changes.
4. **Mid-task Verification:** Any time you touch a Pinokio script, cross-check the corresponding example line to ensure syntax and structure match. Document the reference (example path + line) in your reasoning.
5. **Exit Checklist:** Before responding to the user, revisit the pre-flight checklist and explicitly confirm every item is satisfied. If anything diverges from the example or these rules, fix it first.

If any step cannot be completed, stop immediately and ask the user how to proceed. These five steps are mandatory for every session.

### Critical Pattern Lock: Capturing Web UI URLs

When writing `start.js` (or any script that needs to surface a web URL for a server):

1. **Always copy the capture block from an example such as `system/examples/mochi/start.js`.**
```javascript
on: [{
  event: "/(http:\\/\\/[0-9.:]+)/",
  done: true
}]
```

2. **Set the local variable using the captured match exactly as below (The regex capture object is passed in as `input.event`, so need to use the index 1 inside the parenthesis):**
```javascript
{
  method: "local.set",
  params: {
    url: "{{input.event[1]}}"
  }
}
```

3. Always try to come up with the most generic regex.
4. During the exit checklist, explicitly confirm that the `url` local variable is set via `local.set` API by using the captured regex object as passed in as `input.event` from the previous `shell.run` step.

Deviation from this pattern requires written approval from the user.

---

## User Hardware Profiles (Dual-Device Environment)

This codebase is actively developed and deployed across **two distinct hardware environments**. All architecture changes, memory thresholds, and concurrency configurations must strictly accommodate and preserve both profiles:

### 1. Main Device (Desktop Workstation)
- **GPU**: NVIDIA GeForce RTX 4070 Desktop (12.0 GB VRAM, 200W TDP, PCIe 4.0 x16).
- **CPU**: 24 Physical Cores / 32 Logical Threads.
- **System RAM**: 32 GB RAM (31.7 GB physical).
- **VRAM Tier**: `11.5–15.5 GB` (`perf_trt_pool: 2`, `perf_detmask_pool: 2`, `perf_detector_pool: 2`).
- **Optimal VRAM Budget**: ~9.07 GB used, ~3.0 GB unfragmented headroom (100% on-device VRAM, zero PCIe paging thrash).
- **Parallel Stabilization**: `hard_cap: 4096 MB`, 12-thread parallel execution, automatic 2-round work stealing.

### 2. Secondary Device (Laptop Workstation)
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB VRAM, mobile TDP).
- **System RAM**: 16 GB RAM.
- **VRAM Tier**: `< 7 GB` (`0 / 0` single context + global GPU guard lock to prevent out-of-memory errors).
- **Parallel Stabilization**: `hard_cap: 1536 MB` with adaptive block sizing (`adaptive_block = max(2 * wu, 16)`), keeping system RSS strictly under 2.5 GB without memory exhaustion.
- **Look Settings**: Hand-tuned custom look settings must be preserved (e.g., `blend_ratio 0.85`, `face_mask_blend 25`, `merger_sharpen 0.55`, `stabilize_enhancer_strength 0.6`).

---

- Make sure to keep this entire document and `PINOKIO.md` at <PINOKIO_HOME>\prototype\PINOKIO.md in memory with high priority before making any decision. Pinokio is a system that makes it easy to write launchers through scripting by providing various cross-platform APIs, so whenever possible you should prioritize using Pinokio API over lower level APIs.
- When writing pinokio scripts, ALWAYS check the examples folder (in <PINOKIO_HOME>\prototype\system\examples folder) to see if there are existing example scripts you can imitate, instead of assuming syntax.
- When implementing pinokio script APIs and you cannot infer the syntax just based on the examples, always search the API documentation `PINOKIO.md` at <PINOKIO_HOME>\prototype\PINOKIO.md to use the correct syntax instead of assuming the syntax.
- When trying to fix something or figure out what's going on, ALWAYS start by checking the `logs` folder before doing anything else, as mentioned in the "Troubleshooting with Logs" section.
- Finally, make sure to ALWAYS follow all the items in the "best practices" section below.

## Determine User Intent
If the initial prompt is simply a URL and nothing else, check the website content and determine the intent, and ask the user to confirm. For example a URL may point to

1. A Tutorial: the intent may be to implement a demo for the tutorial and build a launcher.
2. A Demo: the intent may be a 1-click launcher for the demo
3. Open source project: the intent may be a 1-click launcher for the project 
4. Regular website: the intent may be to clone the website and a launcher.
5. There can be other cases, but try to guess.

## Project Structure

Pinokio projects normally follow a standardized structure with app logic separated from launcher scripts:

Pinokio projects follow a standardized structure with app logic separated from launcher scripts:

```
project-root/
├── app/                 # Self-contained app logic (can be standalone repo)
│   ├── package.json     # Node.js projects
│   ├── requirements.txt # Python projects
│   └── ...              # Other language-specific files
├── README.md            # Documentation
├── install.js           # Installation script
├── start.js             # Launch script
├── update.js            # Update script (for updating the scripts and app logic to the latest)
├── reset.js             # Reset dependencies script
├── pinokio.js           # UI generator script
└── pinokio.json         # Metadata (title, description, icon)
```

- Keep app code in `/app` folder only (never in root)
- Store all launcher files in project root (never in `/app`)
- `/app` folder should be self-contained and publishable


The only exceptions are serverless web apps---purely frontend only web applications that do NOT have a server component and connect to 3rd party API endpoints--in which case the folder structure looks like the following (No need for launcher scripts since the index.html will automatically launch. The only thing needed is the metadata file named pinokio.json):

```
project-root/
├── index.html           # The serverless web app entry point
├── ...
├── README.md            # Documentation
└── pinokio.json         # Metadata (title, description, icon)
```

IMPORTANT: ALWAYS try to follow the best practices in the examples folder (<PINOKIO_HOME>\prototype\system\examples) instead of trying to come up with your own structure. The examples have been optimized for the best user experience.

## Launcher Project Working Directory

- The project working directory for a script is always the same directory as the script location.
- For example, when you run `shell.run` API inside `pinokio/start.js`, the default path for shell execution is `pinokio`.
- If the launcher files are in the project root path, then the default path for shell execution is the project root.
- Therefore, it is important to specify the correct `path` attribute when running `shell.run` API commands.

Example: in the following project structure:

```
project-root/
├── pinokio/                 # Pinokio launcher folder
│    ├── start.js             # Launch script
│    ├── pinokio.js           # UI generator script
│    └── pinokio.json         # Metadata (title, description, icon)
└─── backend/
     ├── requirements.txt          # App dependencies
     └── app.py                    # App code
```

The `pinokio/start.js` should use the correct path `../backend` as the `path` attribute, as follows:

```
{
  run: [{
    ...
  }, {
    method: "shell.run",
    params: {
      message: "python app.py",
      venv: "env",
      path: "../backend"
    }
  }, {
    ...
  }]
}
```

## Development Workflow

### 1. Understanding the Project
- Check `SPEC.md` in project root. If the file exists, use that to learn about the project details (what and how to build)
- If no `SPEC.md` exists, build based on user requirements
### 2. Modifying Existing Launcher Projects
If we are starting with existing launcher script files, work with the existing files instead of coming up with your own.
- **Preserve existing functionality:** Only modify necessary parts
- **Don't touch working scripts:** Unless adding/updating specific commands
- **Follow existing conventions:** Match the style and structure already present
### 3. Try to adopt from examples as much as possible
- If starting from scratch, first determine what type of project you will be building, and then check the examples folder (<PINOKIO_HOME>\prototype\system\examples) to see if you can adopt them instead of coming up everything from scratch.
- Even if there are no relevant examples, check the examples to get inspiration for how you would structure the script files even if you have to write from scratch.
### 4. Writing from scratch as a last resort
If there are relevant examples to adopt from, write the scripts from scratch, but just make sure to follow the requirements in the next section.
### 5. Debugging
When the user reports something is not working, ALWAYS inspect the logs folder to get all the execution logs. For more info on how this works, check the "Troubleshooting with Logs" section below.

## Script Requirements

### 1. 1-click launchable
- The main purpose of Pinokio is to provide an easy interface to invoke commands, which may include launching servers, installing programs, etc. Make sure the final product provides ways to install, launch, reset, and update whatever is needed.

### 2. Write Documentation
- ALWAYS write a documentation. A documentation must be stored as `README.md` in the project root folder, along with the rest of the pinokio launcher script files. A documentation file must contain:
  - What the app does
  - How to use the app
  - API documentation for programmatically accessing the app's main features (Javascript, Python, and Curl)

## Types of launchers
## 1. Launching servers
- When an app requires launching a server, here are the commonly used scripts:
  - `install.js`: a script to install the app
  - `start.js`: a script to start the app
  - `reset.js`: a script to reset all the dependencies installed in the `install.js` step. used if the user wants to restart from scratch
  - `update.js`: a script to update the launcher AND the app in case there are new updates. Involves pulling in the relevant git repositories installed through `install.js` (often it's the script repo and some git repositories cloned through the install steps if any)
  - `pinokio.js`: the launcher script that ties all of the above scripts together by providing a UI that links to these scripts.
  - `pinokio.json`: For metadata

Here's a basic server launcher script example (`start.js`). Unless there's a special reason you need to use another pattern, this is the most recommended pattern. Use this or adopt it as needed, but NEVER try something else unless there's a good reason you should not take this approach:

```javascript
module.exports = {
  // By setting daemon: true, the script keeps running even after all items in the `run` array finishes running. Mandatory for launching servers, since otherwise the shells running the server process will get killed after the scripts finish running.
  daemon: true,
  run: [
    {
      // The "shell.run" API for running a shell session
      method: "shell.run",
      params: {
        // Edit 'venv' to customize the venv folder path
        venv: "env",
        // Edit 'env' to customize environment variables (see documentation)
        env: { },
        // Edit 'path' to customize the path to start the shell from
        path: "app",
        // Edit 'message' to customize the commands, or to run multiple commands
        message: [
          "python app.py",
        ],
        on: [{
          // The regular expression pattern to monitor.
          // Whenever each "event" pattern occurs in the shell terminal, the shell will return,
          // and the script will go onto the next step.
          // The regular expression match object will be passed on to the next step as `input.event`
          // Useful for capturing the URL at which the server is running (in case the server prints some message about where the server is running)
          "event": "/(http:\/\/\\S+)/", 

          // Use "done": true to move to the next step while keeping the shell alive.
          // Use "kill": true to move to the next step after killing the shell.
          "done": true
        }]
      }
    },
    {
      // This step sets the local variable 'url'.
      // This local variable will be used in pinokio.js to display the "Open WebUI" tab when the value is set.
      method: "local.set",
      params: {
        // the input.event is the regular expression match object from the previous step
        // In this example, since the pattern was "/(http:\/\/\\S+)/", input.event[1] will include the exact http url match caputred by the parenthesis.
        // Therefore setting the local variable 'url'
        url: "{{input.event[1]}}"
      }
    }
  ]
}
```

## 2. Launching serverless web apps

- In case of purely static web apps WITHOUT servers or backends (for example an HTML based app that connects to 3rd party servers--either remote or localhost), we do NOT need the launcher scripts.
- In these cases, simply include `index.html` in the project root folder and everything should automatically work. No need for any of the pinokio launcher scripts. (Do 
- You still need to include the metadata file so they show up properly on pinokio:
  - `pinokio.json`: For metadata

## 3. Launching quick scripts without web UI

- In many cases, we may not even need a web UI, but instead just a simple way to run scripts.
- This may include TUI (Terminal User Interface) apps, a simple launcher 
- In these cases, all we need is the launcher file `pinokio.js`, which may link to multiple scripts. In this case, there are no web apps (no serverless apsp, no servers), but instead just the default pinokio launcher UI that calls a bunch of scripts.
- Here are some examples:
  - A pinokio script to toggle the desktop theme between dark and light
    - Write some code (python or javascript or whatever)
    - Write a `toggle.js` pinokio script that executes the code
    - Write a `pinokio.js` launcher script to create a sidebar UI that displays the `toggle.js` so the user can simply click the "toggle" button to toggle back and forth between desktop themes
  - A pinokio script to fetch some file
    - Write some code (python or javascript or whatever)
    - Write a `fetch.js` pinokio script that executes the code
    - Write a `pinokio.js` launcher script to create a sidebar UI that displays the `fetch.js` so the user can simply click the "fetch" button to fetch some data.
- You still need to include the metadata file so they show up properly on pinokio:
  - `pinokio.json`: For metadata

## API

This section lists all the script APIs available on Pinokio. To learn the details of how they are used, you can:
1. Check the examples in the <PINOKIO_HOME>\prototype\system\examples folder
2. Read the `PINOKIO.md` at <PINOKIO_HOME>\prototype\PINOKIO.md further documentation on the full syntax

### Script API

These APIs can be used to describe each step in a pinokio script:
- shell.run: run shell commands
- input: accept user input
- filepicker: accept file upload
- fs.write: write to file
- fs.read: read from file
- fs.copy: copy files
- fs.download: download files
- fs.link: create a symbolic link (or junction on windows) for folders
- fs.open: open the system file explorer at a given path
- fs.cat: print file contents
- jump: jump to a specific step
- local.set: set local variables for the currently running script
- json.set: update a json file
- json.rm: remove keys from a json file
- json.get: get values from a json file
- log: print to the web terminal
- net: make network requests
- notify: display a notification
- script.download: download a script from a git uri
- script.start: start a script
- script.stop: stop a script
- script.return: return values if the current script was called by a caller script, so the caller script can utilize the return value as `input`
- web.open: open a url in web browser
- hf.download: huggingfac-cli download API
### Template variables
The following variables are accessible inside template expressions (example `{{args.command}` in scripts, resulting in dynamic behaviors of scripts:
- input: An input is a variable that gets passed from one RPC call to the next
- args: args is the parameter object that gets passed into the script (via pinokio.js `params`). Unlike `input` which takes the value passed in from the immediately previous step, `args` is a global value that is the same through out the entire script execution.
- local: local variable object that can be set with `local.set` API
- self: refers to the script file itself (which is JSON or JavaScript). For example if `start.js` that's currently running has `daemon: true` set, `{{self.daemon}}` will evaluate to true.
- uri: The current script uri
- port: The next available port. Very useful when you need to launch an app at a specific port without port conflicts.
- cwd: The current script execution folder path
- platform: The current operating system. May be one of the following: `darwin`, `win32`, `linux`
- arch: The current system architecture. May be one of the following: x32, x64, arm, arm64, s390, s390x, mipsel, ia32, mips, ppc, ppc64
- gpus: array of available GPUs on the machine (example: `['apple']`, `['nvidia']`)
- gpu: the first available GPU (example: `nvidia`)
- current: The current variable points to the index of the currently executing instruction within the run array.
- next: The next variable points to the index of the next instruction to be executed. (null if the current instruction is the final instruction in the run array)
- envs: You can access the environment variables of the currently running process with envs object.
- which: Check whether a command exists (example: `{{which('winget')}}`. Can be used in the `when` attribute of a script step to run commands or install first.
- exists: Check whether a file or folder exists at the specified relative path (example: `"when": "{{!exists('app')}}"`). Can be used with the `when` attribute to determine a path's existence and trigger custom logic. Use relative paths and it will resolve automatically to the current execution folder. 
- running: Check whether a script file is running (example: `"when": "{{!running('start.js')}}"`). Can be used with the `when` attribute to determine a path's existence and trigger custom logic. Use relative paths and it will resolve automatically to the current execution folder. 
- os: Pinokio exposes the node.js os module through the os variable.
- path: Pinokio exposes the node.js path module through the os variable (example: `{{path.resolve(...)}}`

## System Capabilities
### Package Management (Use in Order of Preference)
The following package managers come pre-installed with Pinokio, so whenever you need to install a 3rd party binary, remember that these are available. Also, you can assume these are available and include the following package manager commands in Pinokio scripts:
1. **UV** - For Python packages (preferred over pip)
2. **NPM** - For Node.js packages  
3. **Conda** - For cross-platform 3rd party binaries
4. **Brew** - Mac-only fallback when other options unavailable
5. **Git** - Full access to git is available.
**Important:** Include all install commands in the install script for reproducibility.
### HTTPS Proxy Support
- All HTTP servers automatically get HTTPS endpoints
- Convention: `http://localhost:<PORT>` → `https://<PORT>.localhost`
- Full proxy list available at: `http://localhost:2019/config/`
### Pterm Features:
- **Clipboard Access:** Read from or Write to system clipboard via pinokio Pterm CLI (`pterm clipboard` command.)
- **Notifications:** Send desktop alerts via pinokio pterm CLI (`pterm push` command.)
- **Script Testing:** Run launcher scripts via pinokio pterm CLI (`pterm start` command.)
- **File Selection:** Use built-in filepicker for user file/folder input (`pterm filepicker` command.)
- **Git Operations:** Clone repositories, push to GitHub
- **GitHub Integration:** Full GitHub CLI support (`gh` commands)

## Troubleshooting with Logs
Pinokio stores the logs for everything that happened in terminal at the following locations, so you can make use of them to determine what's going on:

### Log Structure
In case there is a `pinokio` folder in the project root folder, you should be able to find the logs folder here:

```
pinokio/
└── logs/   # Direct user interaction logs
    ├── api/     # Launcher script logs (install.js, start.js, etc.)
    ├── dev/     # AI coding tool logs (organized by tool)
    └── shell/   # Direct user interaction logs
```

Otherwise, the `logs` folder should be found at project root:

```
logs/
├── api/     # Launcher script logs (install.js, start.js, etc.)
├── dev/     # AI coding tool logs (organized by tool)
└── shell/   # Direct user interaction logs
```

### Log File Naming
- Unix timestamps for each session
- Special "latest" file contains most recent session logs
- **Default:** Use "latest" files for current issues
- **Historical:** Use timestamped files for pattern analysis and the full history.

## Best practices
### 0. Always reference the logs when debugging
- When the user asks to fix something, ALWAYS check the logs folder first to check what went wrong. Check the "Troubleshooting with Logs" section.
### 1. Shell commands for launching programs
- Launch flags related
  - Try as hard as possible to minimize launch flags and parameters when launching an app. For example, instead of `python app.py --port 8610`, try to do `python app.py` unless really necessary. The only exception is when the only way to launch the app is to specify the flags.
- Launch IP related
  - Always try to find a way to launch servers at 127.0.0.1 or localhost, often by specifying launch flags or using environment variables. Some apps launch apps at 0.0.0.0 by default but we do not want this.
- Launch Port related
  - In case the app itself automatically launches at the next available port by default (for example Gradio does this), do NOT specify port, since it's taken care of by the app itself. Always try to minimize the amount of code.
  - If the install instruction says to launch at a specific port, don't use the hardcoded port they suggest since there's a risk of port conflicts. Instead, use Pinokio's `{{port}}` template expression to automatically get the next available port.
  - For example, if the instruction says `python app.py --port 7860`, don't use that hardcoded port since there might be another app running at that port. Instead, automatically assign the next available port like this: `python app.py --port {{port}}`
  - Note that the `{{port}}` expression always returns the next immediately available port for each step, so if you have multiple steps in a script and use `{{port}}` in multiple steps, the value will be different. So if you want to launch at the next available port and then later reuse that port, you will need to first use `{{port}}` to get the next available port, and save the value in local variable using `local.set`, and then use the `{{local.<variable_name>}}` expression later.
### 2. shell.run API
- When writing `shell.run` API requests, always use relative paths (no absolute paths) for the `path` field. For example, if you need to run a command from `app` folder, the `path` attribute should simply be `app`, instead of its full absolute path.
### 2. Package managers
- When installing python packages, try best to use `uv` instead of `pip` even if the install instruction says to use pip. Instead of `pip install -r requirements.txt`, you can simply use `uv pip install -r requirements.txt` for example. Even if the project's own README says use pip or poetry, first check if there's a way to use uv instead.
- When you need to install some global package, try to use `conda` as much as possible. Even on macs, `brew` should be only used if there are no `conda` options.
### 3. Minimal Always
- If you are starting with existing script files, before modifying, creating, or removing any script files, first look at `pinokio.js` to understand which script files are actually used in the launcher. The only script files used are the ones mentioned in the `pinokio.js` file. The `pinokio.js` file is the file that constructs the UI dynamically.
- Do not create a redundant script file that does something that already exists. Instead modify the existing script file for the feature. For example, do not create an `install.json` file for installation if `install.js` already exists. Instead, modify the `install.js` file.
- Pinokio accepts both JSON and JS script files, so when determining whether a script for a specific purpose already exists, check both JSON and JS files mentioned in the `pinokio.js` file. Do not create script files for rendundant purpose.
- When building launchers for existing projects cloned from a repository, try to stay away from modifying the project folder (the `<PINOKIO_HOME>/api/roop-ultimate` folder), even if installations are failing. Instead, try to work around it by creating additional files in the launcher folder, and using those files IN ADDITION to the default project.
  - The only exception when you may need to make changes to the project folder is when the user explicitly wants to modify the existing project. Otherwise if the purpose is to simply write a launcher, the app logic folder should never be touched.
- When running shell commands, take full advantage of the Pinokio `shell.run` API, which provides features like `env`, `venv`, `input`, `path`, `sudo`, `on`, etc. which can greatly reduce the amount of script code.
  - Python apps: Always use virtual environments via `venv` attribute. This attribute automatically creates a venv or uses if it already exists.
### 4. Try to support Cross-platform as much as possible
- Use cross-platform shell commands only.
- This means, prefer to use commands that work on all platforms instead of the current platform.
- If there are no cross platform commands, use Pinokio's template expressions to conditionally use commands depending on `platform`, `arch`, etc.
- Also try to utilize Pinokio Pterm APIs for various cross-platform system features.
- If it is impossible to implement a cross platform solution (due to the nature of the project itself), set the `platform`, `arch`, and/or `gpu` attributes of the `pinokio.json` file to declare the limitation.
- Pinokio provides various APIs for cross-platform way of calling commonly used system functions, or lets you selectively run commands depending on `platform`, `arch`, etc.
### 5. Do not make assumptions about Pinokio API
- Do NOT make assumptions about which Pinokio APIs exist. Check the documentation.
- Do NOT make assumptions about the Pinokio API syntax. Follow the documentation.
### 6. Scripts must be able to replicate install and launch steps 100%
- The whole point of the scripts is for others to easily download and invoke them via Pinokio interface with one click. Therefore, do not assume the end user's system state, and make everything self-contained.
- When a 3rd party package needs to be installed, or a 3rd party repository needs to be downloaded, include them in the scripts.
### 7 Dynamic UI rendering
- The `pinokio.js` launcher script can change dynamically depending on the current state of the script execution. Which means, depending on what the file returns, it can determine what the sidebar looks like at any given moment of the script cycle.
  - `info.exists(relative_path)`: The `info.exists` can be used to check whether a relative path (relative to the script root path) exists. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `info.running(relative_path)`: The `info.running` can be used to check whether a script at a relative path is currently running (relative to the script root path) exists. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `info.local(relative_path)`: The `info.local` can be used to return all the local variables tied to a script that's currently running. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `default`: set the `default` attribute on any menu item for whichever menu needs to be selected by default at a given step. Some example scenarios:
    - during the install process, the `install.js` menu item needs to be set as the `default`, so it automatically executes the script
    - when launching the `start.js` menu item needs to be set as the `default`, so it automatically executes the script
    - after the app has launched, the `default` needs to be set on the web UI URL, so the user is sent to the actual app automatically.
  - Check the examples in the <PINOKIO_HOME>\prototype\system\examples folder to see how these are being used.
### 8. No need for stop scripts
- `pinokio.js` does NOT need a separate `stop` script. Every script that can be started can also be natively stopped through the Pinokio UI, therefore you do not need a separate stop script for start script
### 9. Writing launchers for existing projects
- When writing or modifying pinokio launcher scripts, figure out the install/launch steps by reading the project folder `app`.
- In most cases, the `README.md` file in the `<PINOKIO_HOME>/api/roop-ultimate` folder contains the instructions needed to install and run the app, but if not, figure out by scanning the rest of the project files.
- Install scripts should work for each specific operating system, so ignore Docker related instructions. Instead use install/launch instructions for each platform.
### 10. Don't use Docker unless really necessary
- Some projects suggest docker as installation options. But even in these cases, try to find "development" options to launch the app without relying on Docker, as much as possible. We do not need Docker since we can automatically install and launch apps specifically for the user's platform, since we can write scripts that run cross platform.
### 11. pinokio.json
- Do not touch the `version` field since the version is the script schema version and the one pre-set in `pinokio.js` must be used.
- `icon`: It's best if we have a user friendly icon to represent the app, so try to get an image and link it from `pinokio.json`.
  - If the git repository for the `<PINOKIO_HOME>/api/roop-ultimate` folder points to GitHub (for example https://github.com/<USERNAME>/<REPO_NAME>`, ask the user if they want to download the icon from GitHub, and if approved, get the `avatar_url` by fetching `https://api.github.com/users/<USERNAME>`, and then download the image to the root folder as `icon.png`, and set `icon.png` as the `icon` field of the `pinokio.json`. 
### 12. Gitignore
- When a launcher involves cloning 3rd party repositories, downloading files dynamically, or some files to be generated, these need to be included in the .gitignore file. This may include things like:
  - Cloning git repositories
  - Downloading files
  - Dynamically creating files during installation or running, such as Sqlite Databases, or environment variables, or anything specific to the user.
- Make sure these file paths are included in the .gitignore file, and if not, include them in .gitignore.

## AI Libraries (Pytorch, Xformers, Triton, Sageattention, etc.)
If the launcher has a dedicated built-in script named `torch.js`, it can be used as follows:

```
// install.js
module.exports = {
  run: [
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "venv",                // Edit this to customize the venv folder path
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          path: "app",
          venv: "venv",                // Edit this to customize the venv folder path
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
          // flashattention: true   // uncomment this line if your project requires flashattention
        }
      }
    },
  ]
}
```

The `torch.js` script also includes ways to install pytorch dependent libraries such as xformers, triton, sagetattention. If any of these libraries need to be installed, use the torch.js to install in order to install them cross platform.


## Quick Reference
### Essential Documentation
- **Pinokio Programming:** See `PINOKIO.md` at <PINOKIO_HOME>\prototype\PINOKIO.md → "Programming Pinokio" section
- **Dynamic Menus:** See `PINOKIO.md` at <PINOKIO_HOME>\prototype\PINOKIO.md → "Dynamic menu rendering" section  
- **CLI Commands:** See `PTERM.md` at <PINOKIO_HOME>\prototype\PTERM.md
### Common Patterns
- **Python Virtual Env:** `shell.run` with `venv` attribute
- **Cross-platform Commands:** Always test on multiple platforms
- **Error Handling:** Check logs/api for launcher issues
- **GitHub Operations:** Use `gh` CLI for advanced GitHub features
## Development Principles
1. **Minimize Shell Usage:** Leverage API parameters instead of raw commands
2. **Maintain Separation:** Keep app logic and launchers separate
3. **Follow Conventions:** Match existing project patterns
4. **Test Thoroughly:** Use CLI to verify launcher functionality
5. **Document Changes:** Update relevant metadata and documentation
