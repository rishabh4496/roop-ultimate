# Roop Ultimate

Face swapping for images and video, with a React user interface and a
one-click [Pinokio](https://pinokio.computer) launcher.

Runs on NVIDIA (CUDA / TensorRT), AMD (DirectML / ROCm), Apple Silicon and CPU.
This repository contains both the launcher and the full application.

> **Public source, no support.** This repository is public on GitHub and is
> licensed under the AGPL-3.0 (see [`LICENSE`](LICENSE) and
> [`NOTICE.md`](NOTICE.md)). There is no support channel and no guarantee that
> issues or pull requests will be answered.

> **Independent project.** Roop Ultimate is not affiliated with, endorsed by or
> connected to any other project. It derives from AGPL-3.0 code, which is
> credited in [`NOTICE.md`](NOTICE.md) as that licence requires — that is a
> statement of origin, not of any ongoing relationship. Do not send questions
> about this software to the upstream authors.

## What it is

Roop Ultimate swaps faces in images and video. You give it one or more source
faces (photos or a saved `.fsz` faceset), a target image or video, and it
renders the result locally on your GPU; nothing leaves the machine.

### Workstation features

- **Full-bleed media canvas:** sub-pixel coordinate mapping, persistent crossfading, split comparison wipe, alpha blend, diff map, a 3.5x magnifier loupe and a paint/erase mask brush.
- **Timeline:** filmstrip thumbnails, a measured timecode ruler, in/out trim points, chapter markers and 0.25x-4x playback.
- **3D head pose & tracking:** 5-point ArcFace landmark overlays with live `yaw`/`pitch`/`roll` readouts.
- **Face banking & person grouping:** rank-preserved target clustering, identity renaming, multi-angle galleries and a `.fsz` archive manager.
- **Batch matrix:** four strategies (one-to-many, grouped, per-file matrix, recipe matrix) with automatic video segment splitting.
- **Persistent projects:** every render writes a checkpoint of its exact inputs, settings, provider and hardware. Close the app, shut the machine down, come back, and a project whose inputs still validate can be loaded and resumed.
- **Queue:** the backend owns it, so it survives closing the tab and restarting. Ten job states, per-job progress, per-job cancel, drag reorder, duplicate, retry and clip joining.
- **Distributed video render (CLI):** GOP-aligned stream-copy slices are dispatched to isolated NVIDIA GPU workers; timestamped chunks are concat-copied and source audio is remuxed without a second video encode.
- **Diagnostics:** a live GPU/VRAM/CPU HUD, a structured runtime report with 14 named sections, a thread/pool benchmark runner, a standing environment-health card and a read-only update compatibility check.
- **Screens:** Home (`#/home`), Face Swap (`#/faceswap`), Batch Matrix (`#/batch`), Processing (`#/processing`), Face Manager (`#/facemgr`), Editor (`#/extras`), Outputs (`#/gallery`), History (`#/history`), Settings (`#/settings`). Each is a deep link, so a Pinokio tab switch returns you where you were.

### Offline

The client has no external URLs at all: fonts are self-hosted and nothing is
fetched from a CDN. Every processing feature runs against the loopback backend.
The single action that reaches the internet is the explicit
**Check compatibility** button in Settings; offline it reports UNVERIFIED and
nothing else changes.

### What is not in this repository

`app/env` (the virtual environment), `app/models` (model weights) and
`app/facesets` (your saved face libraries) are generated locally and are
gitignored — together they are around 49 GB on a working install. A fresh clone
has none of them; `install.js` creates the environment and the application
downloads weights on first use.

`app/tests/test_standalone_install.py` fails if any of them is a link into another
folder instead of a real directory (see [`docs/CHANGELOG.md`](docs/CHANGELOG.md), 2026-08-23).

### Models

The application downloads machine-learning models on first use. They are not
part of this project and are not covered by its licence; each has its own terms,
some of which prohibit commercial use. See [`NOTICE.md`](NOTICE.md).

## Install

### With Pinokio (recommended)

1. Open Pinokio and choose **Discover → Download from URL**.
2. Paste this repository's URL:
   ```
   https://github.com/rishabh4496/roop-ultimate.git
   ```
   The repository is public, so no git credentials are needed to clone it.
3. Click **Download**, then **Install**, then **Start**.

Pinokio detects the GPU and installs the matching PyTorch and ONNX Runtime
build. `install.js` is self-contained: it installs `app/requirements.txt`, the
React UI's lockfile dependencies with `npm ci`, builds the production UI,
installs PyTorch via `torch.js`, reasserts the InsightFace-compatible NumPy
version, and installs SAM 2. It does not clone or
download from any other project's repository. The installer writes its
completion marker only after every step succeeds, so an interrupted install
cannot expose a blank React screen as if the app were ready. If an older
installation already has `app/env` but Pinokio still shows a blank page,
choose **Install** once more to complete the frontend build.

TensorRT (`tensorrt-cu12`, `-libs`, `-bindings`) and `onnxruntime-gpu` are provisioned
automatically for NVIDIA profiles, and the initial configuration is seeded from
`default_config.yaml`. Install and startup problems -- NumPy left at 2.x by an older
install, a missing `config.yaml` message, TensorRT missing from a custom environment,
a failed NVIDIA runtime check -- are covered in
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md#install-and-startup).

### Manually

Toolchain: **Python 3.10**, **Node `^20.19 || >=22.12`** (declared in both
`package.json` files under `engines`; Vite 8 refuses older), and **npm** -- the
one JavaScript package manager this repository uses. Installs are reproducible
because both lockfiles are committed and every install path uses them:
`react-ui/package-lock.json` (the UI, what `install.js`, `update.js` and
`start_react.js` run `npm ci` against) and `package-lock.json` at the root (the
dev-only mock server and typecheck). Do not use bun or yarn here; they would
write a second, competing lockfile.

Windows (PowerShell):

```powershell
git clone https://github.com/rishabh4496/roop-ultimate.git
cd roop-ultimate\app
python -m venv env; .\env\Scripts\Activate.ps1
uv pip install -r requirements.txt
cd ..\react-ui; npm ci --no-audit --no-fund; npm run build
cd ..\app; python run.py
```

Linux / macOS:

```bash
git clone https://github.com/rishabh4496/roop-ultimate.git
cd roop-ultimate/app
python3 -m venv env && source env/bin/activate
uv pip install -r requirements.txt
cd ../react-ui && npm ci --no-audit --no-fund && npm run build
cd ../app && python run.py
```

### Updates

Pinokio's **Update** action runs a compatibility check before changing source and
applies only a manifest-gated, source-only fast-forward with a snapshot, health
check and rollback. It never reinstalls Python/Node dependencies, changes CUDA,
ONNX Runtime, TensorRT, FFmpeg or drivers, or replaces models. What it checks, what
it snapshots and how a failed update is recovered:
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md#updates) and the full contract in
[`docs/development/UPDATE_CONTRACT.md`](docs/development/UPDATE_CONTRACT.md).

### For development

Tests (pytest, the light profile CI runs), the CI workflow, the repository layout and
the rules for adding a setting are in
[`docs/development/CONTRIBUTING.md`](docs/development/CONTRIBUTING.md).

## Run

`start.js` launches the **React UI**, the production media workstation and the
only client. The legacy Gradio interface is preserved under `app/ui/` and can
be started with `start_legacy.js`.

### One server, one port

The launcher builds the React client (`npm run build`) and then starts the
backend, which serves that build itself. There is no Vite process at runtime:
the UI and the API share a single origin, so `/api` and the `/ws/telemetry`
socket need no proxy.

For UI development, `npm run dev` in `react-ui/` still gives HMR; the dev server
proxies `/api` and `/ws` to a backend started separately.

### Multi-GPU project render (CLI)

Save a `.roop` project in the UI, then run this with the UI/backend stopped:

```powershell
cd app
.\env\Scripts\python.exe -m roop.distributed_render --project C:\media\job.roop --output C:\media\finished.mp4 --chunk-seconds 10
```

The command detects all visible NVIDIA GPUs, gives each isolated worker one
GPU, and dispatches GOP chunks from a shared queue. It requires an H.264/HEVC
constant-frame-rate source, a trim starting/ending on keyframe boundaries, the
in-memory video method, and an H.264/HEVC output encoder in `config.yaml`.
Packet hashes and IDR NALs are verified after each stream-copy slice. It
rejects unsupported timeline automation rather than shifting it silently.
The final video is assembled with FFmpeg concat demuxer and `-c copy`; original
audio is stream-copied back once. Timestamped chunk MP4s, per-chunk H.264/HEVC
elementary streams, and worker logs remain in `finished.mp4.chunks/`. Raw
elementary streams carry no timestamps, so the MP4s—not the raw streams—are
used for the lossless concat. This is a CLI feature; the React queue is unchanged.

Why the UI is served by the backend rather than `vite preview` is recorded in
[`docs/CHANGELOG.md`](docs/CHANGELOG.md).

### Network access

The backend listens on `127.0.0.1` only. Even there, `/api` and `/ws` refuse
requests whose browser `Origin` is not this server or a local page (403), so a
web page from another site cannot drive it. **Public server (share)** in
Settings -> Server (or `--server_share`) makes it listen on every interface at
the next launch. That launch prints a banner with a random per-launch token and
the Pinokio sidebar shows it; every `/api` and `/ws` request, loopback included,
must carry it (`Authorization: Bearer`, `?token=`, or open the sidebar's
`/?token=...` link once and the UI keeps it in a cookie). Share mode is never
enabled silently: it is announced at startup and when the setting is saved. See
`app/api_access.py`.

### Mock API (development only)

`react-ui/mock-server/server.ts` is **not the real backend**. It is an Express
stand-in for `app/api.py` that lets the React UI be developed without a GPU or
the Python environment: swaps are simulated, the telemetry stream is invented
(fixed "RTX" numbers), and every preview is a generated SVG placeholder. No
launcher script, install step or build references it; `app/api.py` serves the
built UI in production.

```powershell
npm ci                 # repo root, from package-lock.json: express, ws, multer, tsx, typescript (dev-only)
npm run dev:mock       # http://localhost:3000; set PORT to change it
npm run typecheck      # tsc --noEmit over the mock server (CI runs this)
```

You can always tell it apart from the real backend: it prints a MOCK banner on
startup, every response carries `X-Mock-Server: true`, and `GET /api/meta`
returns `"mock": true`.

### HTTP API

The backend detects the active GPU and software stack at runtime. It records the
GPU name, architecture/compute capability, total and available VRAM, driver,
CUDA, TensorRT, ONNX Runtime, Tensor Core precision modes, and NVDEC/NVENC
codecs. Automatic pools, batching, streams, workers, and queue depth are
selected from that profile plus the current workload. Profile/cache identities
are isolated by hardware and workload, so moving between the RTX 3060 and RTX
4070 does not require editing a configuration file. A missing physical target
is reported as pending; its metrics are never copied from the other GPU.

Read the live profile with any HTTP client:

```javascript
const profile = await fetch(`${baseUrl}/api/system/hardware`).then(r => r.json())
console.log(profile.architecture, profile.vram_total_gb, profile.capabilities)
```

```python
import requests
profile = requests.get(f"{base_url}/api/system/hardware", timeout=10).json()
print(profile["gpu_name"], profile["vram_available_gb"])
```

```bash
curl "$BASE_URL/api/system/hardware"
```

The same profile is included in `/api/system/telemetry`. The benchmark endpoint
is `POST /api/settings/benchmark_threads` with `{"profile":"quick"}` or
`{"profile":"full"}`; poll `GET /api/settings/benchmark_status`. 
Benchmark reporting rules and the Python-only telemetry probe are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

#### Three-stage progress and busy previews

The Face Swap and Processing run bars show the same frames as a **3-step pill**: 1 Pre-Scan -> 2 Range Lock -> 3 Render & Mux
(`react-ui/src/components/faceswap/PipelinePill.jsx`, model in `pillModel.js`). It reads `/ws/telemetry` only - no REST call, so
the 15 s API deadline never applies to it - and shows the culled share (`skipped_frames_pct`) and the current fps beside step 3.
Stage 4 is not in `stages`, so step 3 reads "Muxing" between Stage 3 finishing and the report arriving.

Video renders always stream in memory with the 3-stage scan and a stream-copied audio track (Phase 12): Keep frames, Wait before
creating video, Skip audio, the Video method and the sparse pre-scan toggle are gone from the UI, `config.yaml` and the settings
schema. `roop/render_policy.py` holds the fixed choices; `skip_audio` and `sparse_auto_range` remain readable from a request
payload only (distributed chunk renders and the test harness).

A render runs up to three stages: `STAGE_1_SPARSE_PRESCAN` (every video; it is no
longer a setting - still images never reach it and an animated WebP is excluded),
`STAGE_2_ACTIVE_PREPASS` (tracking/detection over the
active ranges) and `STAGE_3_MAIN_SWAP`. Every `/ws/telemetry` frame carries
`stage` (the one running, or `null`) and `stages`, one object per stage:
`state` (`PENDING|RUNNING|DONE|SKIPPED|FAILED|RESTORED`), `stage_progress_pct`,
`active_ranges_count`, `verified_target_frames`, `skipped_frames_pct` and
`eta_seconds` (each `null` until known; `active_ranges_count` and
`skipped_frames_pct` are the Stage 1 verdict, `verified_target_frames` the frames
of the Stage 2 plan). An idle backend sends nothing.

`RESTORED` means the stage did not run this time: a resumed project loaded the
answer its interrupted run had already computed (see "Resuming a project" below).

While the backend builds a TensorRT engine or runs a startup canary (minutes on a
cold cache, with no frame counter) the same frames also carry `engine_state`
(`IDLE|COMPILING_TRT|RUNNING_CANARY`), `engine_label` (`"Compiling TensorRT
Engine..."` / `"Verifying TensorRT Engine..."`, `""` when idle), `engine_model`,
`engine_precision` and `engine_since` (epoch seconds; count up from it, nothing in
the frame ticks). The canaries announced are the swapper's, the small models' and
the restorers'. `WS /ws/v1/pipeline/status` carries the same stages with their
event history.

#### Resuming a project

A project resumes from its committed output segments. Since Phase 6 it also resumes
past Stage 1 and Stage 2: the moment each finishes, the project record stores the
Stage 1 active ranges (absolute frames) and the Stage 2 hand-off the swap pass
consumes (the gap-filled faces, track assignments, source map and shot cuts, in a
`<project id>.stage2.npz` beside the record, with its SHA-256 in the record). A
resumed job validates the cache against the target's SHA-256, the trim, every
source's SHA-256, the captured target faces, the settings, the model identity, the
application version and the `ROOP_*` environment; on any difference, or if the file
fails its digest, the stage runs as before and the log says why
(`[StageCache] not used: ...`). `/ws/telemetry` shows the skipped stages as
`RESTORED`. A finished project deletes its cache.

The cache is not used with the SAM 2 mask engine (its per-frame masks are not in
it), for animated-WebP targets, or for a pre-pass that was stopped early.

#### Source identity pre-flight (Stage 0)

Before the first frame is read, each source's identity is frozen for every head-pose
cell (the `.fsz` pose matrix's 3 x 3 yaw/pitch cells, with roll as a third key axis):
the pose-selected vector, and with an Identity Blender recipe the blend and the
tangent-plane dial offsets under the 0.80 identity guard. The swap pass then looks
the face up (`[Stage0] ...` lines in the log). The table answers only where it
equals the per-face selection exactly; otherwise it says why it declined and the
per-face path runs unchanged: image/CSCS swappers, `use_source_bank` with several
faces, an angle portfolio, the temporal quality engine, folder-upload banks, and
V1 `.fsz` archives with no recipe (every archive shipped in `app/facesets` is V1).
`ROOP_STAGE0=0` turns it off.

While Stage 1 or 2 owns the GPU, `GET /api/target/preview`, `/preview_grid` and
`/preview_seq` answer a frame that is not already cached with a fast
`503 {"busy": true, "stage": "...", "retry_after_ms": 1000}` plus `Retry-After`.
Wait that long and ask again; a cached frame is still served. Starting or
resuming a render (`POST /api/swap`, `/api/queue/start`, `/api/queue/resume`,
`/api/projects/{id}/resume`) returns without reading the target file: checkpoint
file checks on the request are size-only and the SHA-256 is completed or
verified by the job itself before it touches the files.

```javascript
const ws = new WebSocket(`${baseUrl.replace('http', 'ws')}/ws/telemetry`)
ws.onmessage = (e) => {
  const t = JSON.parse(e.data)
  const s = t.stage && t.stages[t.stage]
  if (s) console.log(t.stage, s.stage_progress_pct, s.eta_seconds, s.skipped_frames_pct)
}
const r = await fetch(`${baseUrl}/api/target/preview?index=0&frame=120`)
if (r.status === 503) setTimeout(retry, (await r.json()).retry_after_ms)
```

```python
import json, websockets, requests   # pip install websockets requests

async def watch(base_url):
    async with websockets.connect(base_url.replace("http", "ws") + "/ws/telemetry") as ws:
        async for message in ws:
            t = json.loads(message)
            if t.get("stage"):
                print(t["stage"], t["stages"][t["stage"]])

r = requests.get(f"{base_url}/api/target/preview", params={"index": 0, "frame": 120}, timeout=10)
if r.status_code == 503:
    print("busy:", r.json()["stage"], "retry in", r.json()["retry_after_ms"], "ms")
```

```bash
curl -i "$BASE_URL/api/target/preview?index=0&frame=120"   # 503 + Retry-After while a pre-pass runs
websocat "${BASE_URL/http/ws}/ws/telemetry"                 # frames with "stage" / "stages"
```

#### Multi-angle sources and timestamp-safe video

Use **Build one multi-angle faceset from a folder** in the React source panel,
or start the backend with `python run.py --source <image-or-folder>`. Each
reference is detected with the existing ArcFace analyser, normalized to 512
dimensions, clustered at cosine similarity 0.65, and combined as a
quality-weighted, L2-normalized identity vector. The retained reference yaw and
pitch values select (or blend at a pose boundary) the closest identity vector
during a swap. This adds no second detector or GPU session.

Programmatic folder ingestion is `POST /api/source/add-folder` as multipart
field `files`. The usual `POST /api/source/add` remains the one-image-per-source
operation. Example clients:

```javascript
const form = new FormData();
for (const file of folderFiles) form.append('files', file);
const source = await fetch(`${baseUrl}/api/source/add-folder`, { method: 'POST', body: form }).then(r => r.json());
```

```python
import requests
files = [('files', open(path, 'rb')) for path in reference_paths]
source = requests.post(f'{base_url}/api/source/add-folder', files=files, timeout=120).json()
```

```bash
curl -F "files=@front.jpg" -F "files=@left.jpg" -F "files=@right.jpg" "$BASE_URL/api/source/add-folder"
```

#### Identity Blender: latent blends and attribute dials

The **Identity Blender** panel (under the source gallery in Face Swap) blends up
to four source identities and shifts attributes of the result. Everything happens
on the unit ArcFace vector (buffalo_l / w600k_r50, 512-D) that the embedding
swappers consume:

- **Blend:** `z = normalize(Σ wᵢ·zᵢ)`. The weights are renormalised to sum to 1.
  Two different people sit at cosine ~0.0-0.2, so an un-normalised 50/50 mix is
  only ~0.7 long, and the normalisation is required. Each component's vector
  is pose-matched to the target when its faceset supports that (V2 cells or
  folder banks). The blend replaces the identity of every face assigned to any
  of the blended sources.
- **Attribute dials:** `z' = normalize(z + P_z(Σ αₖ·vₖ))`, where `P_z` projects
  onto the tangent plane at `z`. The result is exactly
  `cos(z', z) = 1/√(1+|t|²)`, which gives the **identity guard** a closed form:
  past the minimum cosine (default 0.80), the whole offset is scaled back
  uniformly, keeping its direction. **Age Shift** is in years, limited to ±30.
  **Femininity ↔ Masculinity**, **Feature Dominance** (jaw squareness) and
  **Expression Intensity** run from -1 to 1, where ±1 is two standard
  deviations of how real faces vary along that direction.
- **Directions** ship in `app/roop/assets/identity_directions.npz` and were
  fitted by `tools/fit_identity_directions.py`. They are ridge regressions of
  genderage age/sex and 68-landmark geometry on 14,582 faces (5,845 identities:
  LFW plus the local clips and facesets), orthonormalised with Gram-Schmidt,
  and scored on identity-disjoint held-out folds: age r 0.57, sex AUC 0.78,
  jaw r 0.44, expression r 0.21. The labels are model predictions, and ArcFace
  is trained to ignore expression.
- **Render validation.** A direction can *predict* a label and still not
  *write* it through the swapper. `app/tests/identity_algebra_bench.py`
  re-measures the swapped faces (hyperswap + Restore Ultra, 65–100 paired faces,
  RTX 4070). The verdicts are stored in
  `app/roop/assets/identity_render_validation.json`, and dials that fail are
  disabled with the reason shown:
  - **Age:** not monotone. Both directions read *older* (+6.3 / +3.7 years
    at a 0.75 step) while identity falls from 0.64 to 0.47–0.55. Disabled.
  - **Sex:** no signed response in the rendered face. Disabled.
  - **Expression:** null, because the swapper takes expression from the
    target. Disabled.
  - **Feature Dominance (jaw):** monotone but weak. The jaw ratio moves
    −0.011 to +0.009 over the full range, for −0.11 to −0.15 identity.
    Enabled, with ±1 set to the measured 0.75 step.

  The blend itself is verified on the render. Cosine of the swapped face to
  A/B was 0.60/0.07 at A100, 0.57/0.18 at 75/25, 0.30/0.50 at 50/50,
  0.17/0.62 at 25/75, and 0.07/0.63 at B100.

Blending applies to the embedding swappers only. Image-source models
(BlendSwap/UniFace) and CSCS compute identity from a crop in a different space,
so they are skipped. Programmatic access uses `GET`/`POST /api/identity/blend`.
A `/api/preview` or `/api/swap` payload can also carry `identity_blend`, and
the queue stores that payload, so queued jobs keep the recipe they were
queued with. Source ids are the `id` fields from `source_faces_info`.

```javascript
const recipe = {
  enabled: true,
  components: [{ source_id: idA, weight: 60 }, { source_id: idB, weight: 40 }],
  dials: { age: 10, gender: 0, jawline: 0.3, expression: 0 },
  min_cosine: 0.8,
};
const res = await fetch(`${baseUrl}/api/identity/blend`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(recipe),
}).then(r => r.json());
console.log(res.diagnostics.components, res.diagnostics.clamped);
```

```python
import requests
recipe = {"enabled": True,
          "components": [{"source_id": id_a, "weight": 60}, {"source_id": id_b, "weight": 40}],
          "dials": {"age": -15}}
res = requests.post(f"{base_url}/api/identity/blend", json=recipe, timeout=10).json()
print(res["diagnostics"]["cosine_to_anchor"], res["directions"]["heldout"])
```

```bash
curl -X POST "$BASE_URL/api/identity/blend" -H "Content-Type: application/json" \
  -d '{"enabled":true,"components":[{"source_id":"A","weight":60},{"source_id":"B","weight":40}],"dials":{"age":10}}'
```

Video processing forces constant frame rate and explicit generated video PTS;
audio is muxed with `-c:a copy` and the original audio bitstream when possible.
`python scripts/verify_roop_keep.py` records CFR, copied-audio codec/sample
rate, and output A/V duration drift for every rendered clip.

### Audio & Mouth Custody (lip-sync, dubbing, audio slip)

`POST /api/swap`, `POST /api/preview` and a project's `settings` accept three fields (a bad one is a `422` with the reason,
and nothing starts):

| Field | Meaning |
|---|---|
| `mouth_custody_mode` | `off` (the swap's own mouth, standard expression transfer), `source_mouth_restore` (the target's mouth is kept), `audio_driven_sync` (lip-sync from audio: Wav2Lip or MuseTalk, `lipsync_model`) |
| `override_audio_path` | An uploaded audio file (`POST /api/lipsync/audio/add` returns it) or a video with audio. It is the delivered audio in every mode, and what drives the mouth in `audio_driven_sync`. Without it the target video's own audio is used |
| `audio_offset_ms` | `-500` to `+500`; positive = the audio is heard later than the video. The mouth and the delivered track move together |

A request without `mouth_custody_mode` keeps its old meaning (`lipsync_enabled` / `restore_original_mouth`). A video with no
audio stream and no override track cannot drive a mouth: `audio_driven_sync` runs as `source_mouth_restore` for that clip
(`mouth_fallback` on `/ws/telemetry` says so). `lipsync_ms`, `mouth_mode`, `mouth_fallback` and `audio_offset_ms` are on every
telemetry frame; `lipsync_ms` is the Lip-Sync column of the terminal dashboard. In a multi-GPU project render the track is attached
once, to the joined video; each GPU worker reads it from its own chunk's start. The media editor (`#/extras`) takes
`override_audio_path` and `audio_offset_ms` too (no mouth mode: it runs no swap).

```javascript
const { path } = await upload('/api/lipsync/audio/add', dubFile)       // multipart, field "file"
await fetch(`${baseUrl}/api/swap`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...recipe, mouth_custody_mode: 'audio_driven_sync', override_audio_path: path, audio_offset_ms: -120 }) })
```

```python
import requests
path = requests.post(f"{base_url}/api/lipsync/audio/add", files={"file": open("dub.wav", "rb")}).json()["path"]
r = requests.post(f"{base_url}/api/swap", json={**recipe, "mouth_custody_mode": "audio_driven_sync",
                                                "override_audio_path": path, "audio_offset_ms": -120})
assert r.status_code != 422, r.json()["message"]
```

```bash
curl -F "file=@dub.wav" "$BASE_URL/api/lipsync/audio/add"
curl -X POST "$BASE_URL/api/swap" -H "Content-Type: application/json" -d '{"mouth_custody_mode":"source_mouth_restore"}'
```

### Batch: one fresh process per video

Use the root-level runner for a retained media folder (`<MEDIA_DIR>`, holding
`single/` and `double/` subfolders). It renders one video in a fresh Windows `spawn` process, then waits
for that worker to exit before starting the next video. This prevents CUDA,
ONNX Runtime, FFmpeg, and DirectShow state from accumulating across a batch.

```powershell
python pinokio_batch_runner.py --root <MEDIA_DIR> --single-faceset my_faceset --double-facesets my_faceset,other_faceset --dry-run
python pinokio_batch_runner.py --root <MEDIA_DIR> --single-faceset my_faceset --double-facesets my_faceset,other_faceset
```

The same three values can come from `ROOP_BATCH_ROOT`,
`ROOP_BATCH_SINGLE_FACESET` and `ROOP_BATCH_DOUBLE_FACESETS` instead of flags.
`single/*.mp4` is written to `single_results/` using the single faceset;
`double/*.mp4` is written to `double_results/` using the two double facesets.
Existing outputs are retained unless `--overwrite` is supplied. Progress
events include the frame index, FPS, and ETA; the parent writes them to
`<MEDIA_DIR>/pinokio_batch_runner.log`.

## Config

Settings live in `app/config.yaml` (per machine, never committed) and are edited from
the React **Settings** screen; on a fresh install the file is optional and startup uses
defaults until you save. `app/default_config.yaml` seeds a new install. The
`ROOP_*` environment variables below are read at startup; `docs/ENV_FLAGS.md` lists all of them.

### Online and offline operation

Normal processing, local previews, saved projects, checkpoints, and existing
models do not require Internet access after installation. Internet access is
used for installation, explicit application updates, and downloading a model
that is not already available locally. Optional startup pre-warming is skipped
when offline. See [`NETWORK_CONTRACT.md`](docs/development/NETWORK_CONTRACT.md)
for the audited dependency boundary and known limitations.

### Preview face alignment

Preview and export align faces to the selected swap model's five-point training
template at every angle. Profile estimates do not replace that template with
ear/chin anchors, which can displace eyes and mouths and produce doubled features.
Roll correction uses one image resample; video stabilization stays scoped to each
render's tracks. After an alignment update, restart the app and refresh the preview
to discard images cached by the previous version. Existing look settings are retained.

### Video performance pipeline

Video renders use bounded decode → inference → encode queues. FFmpeg provides
hardware decode/encode. Stabilized renders use parallel, ordered blocks with
warm-up overlap by default (`ROOP_STAB_STREAMING=0`); available RAM limits the
block width and chunk size. Face, mask, and enhancer smoothing all count as
stabilization when assigning workers. Turning off face-position smoothing alone
must not reduce enabled mask/enhancer smoothing to one worker.

`ROOP_STAB_STREAMING=1` selects the alternative continuous pipeline with one
CUDA owner and at most four in-flight full-resolution frames. It avoids block
warm-up recomputation, but can reduce inference concurrency. Both paths reset
temporal state at scene cuts. Progress FPS and ETA use a three-second completion-time
window, sampled every 500 ms with a fixed 0.15 EMA; only frames emitted by the
writer advance that meter.

UltraMax binds its ONNX CUDA output directly into a PyTorch CUDA allocation,
then completes chroma transfer and eye protection on that tensor. This also
works on ONNX Runtime builds without CUDA DLPack support. Providers that reject
the binding automatically retain the established CPU-compatible path; no UI,
CLI, or resume setting changes are required.

#### Strict TensorRT throughput harness

The opt-in strict path is exposed by `app/roop/trt_session_builder.py`,
`app/roop/optimized_prepass.py`, and `app/roop/optimized_processor.py`. It
registers the packaged TensorRT/CUDA DLLs on Windows, creates a TensorRT-only
ONNX Runtime session with explicit min/opt/max profiles and engine/timing
caches, batches faces across frames, performs affine sampling and compositing
on CUDA, and sends one rawvideo stream to NVENC. It raises if the session is not
TensorRT-first, if ORT adds CUDA as a fallback, or (via ORT's
`session.disable_cpu_ep_fallback`) if any node would run on the CPU; ORT's own
trailing `CPUExecutionProvider` list entry is expected and accepted. Run the
verifier from `app/` after installing matching TensorRT libraries:

```bash
python verify_trt_fps.py --model models/hififace_unofficial_256.onnx \
  --batch-size 8 --warmup 10 --iterations 100

# one engine (profile min 1 / opt 4 / max 8) measured at batch 1, 4 and 8:
python verify_trt_fps.py --model models/inswapper_128_dynamic.onnx --batch-sizes 1,4,8
```

A `--batch-sizes` sweep proves the dynamic profile (every size runs inside one
engine without a shape rejection), reports fps, per-item latency and the scaling
against the smallest batch, and fails (exit 3) if any output is NaN/Inf. Finite is
a tripwire for FP16 overflow, not proof of a correct picture: the swapper's FP16
failure is wrong colours with finite values, so build and measure the swapper FP32.

The strict path requires a dynamic-batch ONNX export. The shipped
`inswapper_128.onnx`, `scrfd_2.5g_kps.onnx` and SCRFD `det_10g.onnx` are batch-one
exports and are rejected (`StaticBatchError`) rather than repeatedly invoked at
batch one. Convert them with `tools/convert_dynamic_shapes.py`, which writes
`<name>_dynamic.onnx` only after proving a batch of N equals N single runs:

```bash
python tools/convert_dynamic_shapes.py app/models/inswapper_128.onnx --check-batches 1,4,8
```

Measured on the shipped files (ONNX Runtime, CPU, batches 1/4/8): `inswapper_128`,
`det_10g`, `scrfd_2.5g_kps` and `restoreformer_plus_plus` (image output only) convert
with a worst relative error of 0 to 4e-5. **GPEN 256/512 cannot be converted by graph
surgery**: batch 4 is rejected first at the encoder's flatten into the final Linear, and
with that fixed by hand batch 2 is rejected next in the generator's styled convolutions,
whose per-sample weights `[B,256,512,3,3]` are reshaped to `[256,512,3,3]` for one `Conv`
(batch 1 baked into the weights; undoing it needs a runtime `groups=B`, and ONNX
`Conv.group` is a compile-time attribute). The tool reports this and writes nothing. SCRFD's outputs fold the batch
into axis 0 *interleaved per cell* (`index = (cell * N + sample) * anchors + anchor`),
not batch-major; whatever decodes them must unfold that. TensorRT engine, profile, and
timing artifacts live under `app/models/trt_cache/`; decoded/intermediate video frames
are never written there.

#### Offline TensorRT engine builder

The explicit preparation command downloads the calibrated model set and builds
TensorRT engine and timing caches before a render:

```bash
python tools/build_trt_engines.py
# or, from the repository root:
npm run build:engines
```

It stages `inswapper_128.onnx`, `GPEN-BFR-512.onnx`, and
`scrfd_2.5g_kps.onnx` under `models/`, then creates machine-specific cache
artifacts under `models/trt_cache/`. Re-running with `--offline` performs no
network access and requires all three selected model files to already exist.
The builder uses a 2048 MiB workspace on the RTX 4070 tier and 1024 MiB on the
sub-7 GiB RTX 3060 laptop tier. Use `--model <name>` to build one registered
model, or set `ROOP_TRT_WORKSPACE_BYTES` only when deliberately overriding the
hardware default.

Models marked dynamic in the registry (inswapper min `target:1x3x128x128,source:1x512`
/ opt `4x` / max `8x`; SCRFD `1x3x640x640` / `4x` / `8x`) are first converted by
`tools/convert_dynamic_shapes.py` and built from the verified `_dynamic.onnx`. A model
that cannot be made batch-dynamic (GPEN) is built as static batch 1 and the summary line
says so. FP16 is the default; the swapper is built FP32 because inswapper overflows in
FP16 and TensorRT emits wrong colours without a warning (`ROOP_SWAP_FP16=1` opts in,
`ROOP_TRT_FP16=0` builds everything FP32). TensorRT builds an engine at the first
inference, not when the session is created, so the builder runs every batch size in
the profile, requires finite outputs, and fails if no `.engine` file was written. These
engines are a flat cache that the app does not load; `tools/prebuild_engines.py` builds
what the app uses.

The lower-level zero-pickle implementation is split into
`app/roop/hardware_streamer.py`, `app/roop/optimized_trt_engine.py`, and
`app/roop/vectorized_pipeline.py`. `SharedMemoryFrameRing` is a bounded
`multiprocessing.shared_memory` transport for fixed-size host frames and
metadata; it is not CUDA IPC, because Python shared memory is system RAM.
`NvdecFrameSource` therefore keeps decode on NVDEC until one explicit
`hwdownload` at that boundary, while persistent CUDA buffers, GPU affine
sampling, TensorRT I/O binding, and NVENC remain on device. Set
`ROOP_STRICT_NVDEC=1` and `ROOP_OPT_STRICT_TRT=1` to fail loudly if either
hardware path is unavailable.

### Synthetic-media labels and the intended-use screen

The first time the app opens on an install it shows the **Intended use** section of
[`NOTICE.md`](NOTICE.md) and asks you to accept it; rendering (`POST /api/swap`) is
refused until you do, and the screen returns if that text changes. Acceptance is
stored in `app/config.yaml` (`intended_use_acknowledged`).

Every rendered file is **tagged as synthetic media in its metadata by default**
(Settings → Output → *Label output as synthetic media*): MP4/MOV/MKV/WebM get the
container tags `comment` and `synthetic_media=true`, PNG a `Comment` text chunk, JPEG an
EXIF `ImageDescription` plus a comment segment. The tag is added to the finished file
without re-encoding, so pixels are untouched; `ffprobe`, `exiftool` and most asset
managers show it. An optional **visible watermark** (off by default, text configurable)
stamps a caption on every output frame instead.

What the labels do and do not guarantee: a metadata tag is a *label*, not a signature
or a content credential. Any re-encode, screenshot, or metadata editor removes it, and
nothing here can stop a recipient from doing that. The visible watermark survives
re-encoding but can be cropped. GIF and WebP output carry no tag (their containers offer
no comparable field here). The labels help you meet a duty to mark synthetic media;
they do not prove provenance, and they are not a substitute for consent.

### Storage review

The React Settings screen includes a Storage Manager backed by `GET /api/storage`.
It shows known application/Pinokio paths, category, size, classification reason,
regenerability, and current references. Only a single freshly revalidated
`SAFE_TO_DELETE` item can be explicitly confirmed through
`POST /api/storage/delete`; models, outputs, facesets, checkpoints, queue state,
active work, environments, and required dependencies remain protected. Unknown
drive-wide files and user-wide package caches are intentionally outside this
manager. See [`docs/development/STORAGE_CONTRACT.md`](docs/development/STORAGE_CONTRACT.md)
for the evidence and limitations.

## Troubleshooting

[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) covers install and startup
problems, the NVIDIA runtime compatibility check (`python run.py --diagnose-runtime`),
GPU memory pressure and pool settings, the terminal's structured runtime report, and
what the Update action does when a health check fails. Pinokio keeps every script's
output under `logs/` next to this file (`logs/api/` for the launcher scripts, `latest`
files for the most recent run); read those first.

Benchmark and validation procedures (Phases 6-16, autotuning, runtime monitoring) are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md); dated incident notes and behaviour changes in
[`docs/CHANGELOG.md`](docs/CHANGELOG.md).

## Licence and use

GNU Affero General Public License v3 — see [`LICENSE`](LICENSE), and
[`NOTICE.md`](NOTICE.md) for what that means for this repository and for
people you share it with.

Use this only on material you have the right to use, and only with the informed
consent of the people whose likenesses are involved. See the intended-use
section of [`NOTICE.md`](NOTICE.md).
## Real-time webcam mode

The dedicated low-latency webcam path, virtual camera output, optical-flow
tracking, scene-cut flushing, and optional delayed microphone passthrough are
documented in [`docs/LIVE_MODE.md`](docs/LIVE_MODE.md).
