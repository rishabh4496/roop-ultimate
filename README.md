# roop-unleashed

A deepfake face-swap application for images and videos with an easy-to-use Gradio web UI.
Supports NVIDIA (CUDA / TensorRT), AMD (DirectML / ROCm), Apple Silicon, and CPU.

This repository contains both the Pinokio launcher scripts and the full application code.

---

## Installation — Option 1: Pinokio (Recommended)

[Pinokio](https://pinokio.computer) automates the entire install in one click.

1. Open Pinokio and click **Discover** (or paste the repo URL directly).
2. Paste the repo URL:
   ```
   https://github.com/Adutchguy/roop-unleashed-wip.git
   ```
3. Click **Download**, then **Install**.
4. Once installed, click **Start** to launch the web UI.

Pinokio will automatically detect your GPU and install the correct PyTorch and ONNX Runtime variant.

---

## Installation — Option 2: Manual (GitHub)

### Prerequisites

| Requirement | Notes |
|---|---|
| Python **3.10** | Other versions may break onnxruntime |
| Git | For cloning |
| ffmpeg | Required for video processing — [download](https://ffmpeg.org/download.html) |
| CUDA 12.8 Toolkit | NVIDIA GPU users only |
| TensorRT | Optional — fastest NVIDIA inference |

### Step 1 — Clone the repository

```bash
git clone https://github.com/Adutchguy/roop-unleashed-wip.git
cd roop-unleashed-wip
```

### Step 2 — Create a virtual environment

```bash
cd app
python -m venv env
```

Activate it:

```bash
# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install PyTorch + ONNX Runtime

Choose the command set that matches your hardware:

#### NVIDIA GPU (CUDA 12.8)
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install onnxruntime-gpu==1.19.0
```

#### AMD GPU — Windows (DirectML)
```bash
pip install torch torch-directml torchvision torchaudio --force-reinstall
pip install onnxruntime-directml
```

#### AMD GPU — Linux (ROCm 6.3)
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall --no-deps
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl
```

#### Apple Silicon (M1 / M2 / M3)
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
pip install onnxruntime-silicon==1.16.3
```

#### CPU only
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
pip install onnxruntime==1.17.1
```

### Step 5 — (Optional) TensorRT acceleration — NVIDIA only

TensorRT provides the fastest inference on NVIDIA GPUs. Requires CUDA 12.x.

```bash
pip install tensorrt-cu12
```

Then open the Settings tab in the UI and change **Provider** to `tensorrt`.

> **Note:** On first use with TensorRT, each ONNX model is compiled to a TRT engine. This takes
> several minutes per model but is cached in `app/models/trt_cache/` — subsequent starts are instant.

### Step 6 — Run

```bash
python run.py
```

The Gradio UI opens at [http://127.0.0.1:7860](http://127.0.0.1:7860).

Models are downloaded automatically on first launch via the InsightFace model downloader.
Additional enhancement models (GFPGAN, GPEN, CodeFormer, etc.) can be downloaded from the
**Settings** tab inside the app.

---

## Usage

1. **Source Images / Facesets** — Upload one or more face images (or `.fsz` faceset files).
2. **Target File(s)** — Upload the image or video to apply the swap to.
3. Select a source face from the gallery and a target face (or use **All faces** mode).
4. Optionally enable **Face swap frames** preview to see the result before processing.
5. Click **▶ Start**.

### Key settings

| Setting | Description |
|---|---|
| Provider | `cuda` (default), `tensorrt` (fastest NVIDIA), `cpu` |
| Swap model | `inswapper` (default) · reswapper · hyperswap 1a/1b/1c · ghost 1/2/3 · simswap / simswap_512 · hififace · **blendswap** · **uniface** (each downloads on first use) |
| Detector engine | `scrfd` (default) or `yoloface` (better on steep profiles / occluded faces) |
| Color/lighting match | Match the swapped face's tone & lighting to the scene: `rct` (default) · `lct` (fixes hue casts) · `mkl` (fullest match) · `none` |
| Refine alignment (68-pt) | Derives alignment keypoints from the 68-point landmarks — steadier on angled faces |
| Rescue small faces | Retries detection on a 2× upscale when a frame has no face |
| Enhancer | Post-processing: GPEN (512/1024/2048), GFPGAN, CodeFormer, DMDNet, RestoreFormer++ |
| Restore original mouth | Composites the target's original mouth back over the swap |
| Video swapping method | **In-Memory** (fast, more RAM) or **Extract Frames** (large videos) |
| Subsample upscale | Internal face resolution: 128 → 256 → 512 px |

The **Editor** tab also offers AI post-processing on any image/video: Real-ESRGAN / LSDIR upscaling, DeOldify colorization, and stylize filters (cartoon, pencil, C64, …). Advanced performance knobs (TensorRT pool sizes, encoder preset, profiling) live under **Settings → Advanced performance** and apply after an app restart.

---

## Updating

### Pinokio
Click **Update** in the sidebar.

### Manual
```bash
git pull
pip install -r app/requirements.txt
```

---

## Freeing disk space

A full install is around 35 GB. Most of that is the model library and the
virtualenv, which have to stay — but a few GB is scratch and rebuildable cache.

### Pinokio
Click **Clean** in the sidebar (just under *Install*). It prints the size of each
item for your install, then asks which ones to remove:

| Item | What it is | Cost of removing it |
|------|-----------|---------------------|
| Uploaded media scratch | `app/temp/` — copies of files you dragged in | none, re-created on next upload |
| Stale TensorRT engine caches | engines for a precision mode you no longer use | none, the caches for your current setting are kept |
| Old run logs | everything but the newest 5 per folder and `latest` | loses older debugging history |
| Python bytecode caches | `__pycache__` under `app/` | a second or two on next start |
| Front-end build output | `react-ui/dist`, lint caches | none, unused at runtime |
| **ALL** TensorRT engine caches | every built engine | frees the most by far, but each model recompiles its engine once on next use — minutes of extra startup |

**Never touched:** `app/models/` (the model library), `app/env/` (the
virtualenv), `app/facesets/` (your saved facesets), and `app/output/` — your
rendered videos are never a cleanup target.

If some files can't be removed, the app is still running and holding them open;
stop it and run Clean again.

To shrink the virtualenv instead, use **Save Disk Space** — it hardlinks
duplicate library files rather than deleting anything.

### Manual
```bash
python cleanup.py --report                 # what could be freed, with sizes
python cleanup.py uploads trt-stale logs   # remove specific items
```
Targets: `uploads`, `pycache`, `trt-stale`, `trt-all`, `logs`, `build`.

---

## Resetting / Reinstalling

### Pinokio
Click **Reset** in the sidebar — this removes `app/` and lets you reinstall from scratch.

### Manual
Delete `app/env/` and recreate the virtual environment (Steps 2–4 above).

---

## Troubleshooting

**Uploads or preview appear to hang on first TensorRT use**
TRT is compiling ONNX models to engine files on first run. Let it complete — it will finish.
After the first run, engines are cached at `app/models/trt_cache/` and load instantly.

**Garbled / corrupt output with TensorRT**
Delete `app/models/trt_cache/` entirely and restart so engines recompile in FP32.

**`onnxruntime_providers_cuda.dll` Error 126 on Windows**
Use `onnxruntime-gpu==1.19.0` exactly. Newer versions have DLL dependency issues on Windows.

**ffmpeg not found**
Install ffmpeg and ensure it is on your system PATH. [Download here](https://ffmpeg.org/download.html).

**Video upload broken after server restart**
Gradio temp files from the previous session cause an asyncio event loop mismatch. Restart the
app fully (stop and start again from Pinokio).

---

## Project Structure

```
roop-unleashed-wip/
├── app/                    # Application code
│   ├── roop/               # Core processing (face swap, processors, utilities)
│   ├── ui/                 # Gradio web UI
│   ├── models/             # Downloaded model weights (gitignored)
│   ├── requirements.txt    # Python dependencies
│   └── run.py              # Entry point
├── install.js              # Pinokio install script
├── start.js                # Pinokio start script
├── update.js               # Pinokio update script
├── reset.js                # Pinokio reset script
├── torch.js                # Cross-platform PyTorch installer
├── pinokio.js              # Pinokio UI definition
└── README.md               # This file
```

---

## API Documentation

The backend (`app/api.py`) exposes FastAPI REST endpoints on `127.0.0.1`, bound to
`ROOP_API_PORT` — `8001` only when that variable is unset. **Under Pinokio the port is
assigned dynamically** (`start_react.js` passes `kernel.port()`), so read the actual
value from the launcher terminal rather than assuming `8001`; the examples below use
`$ROOP_API_PORT` for that reason.

### Endpoints
- `GET /api/state` — Full UI state: loaded source facesets, target files, selection.
- `POST /api/source/add` — Add source face image(s) (multipart upload).
- `POST /api/target/add` — Add target image/video file(s) (multipart upload).
- `POST /api/target/add_path` — Add target file(s) **already on this machine**, by
  path, with no upload and no second copy in `temp/`. Body `{"paths": [...]}` (a bare
  string is accepted too). Prefer this over `/api/target/add` for anything large: the
  server is on `127.0.0.1`, so uploading a multi-gigabyte clip to it transfers the
  whole file over a loopback socket and then writes it to disk again. Returns the
  usual target list plus `added` (absolute paths accepted) and `rejected`
  (`[{path, why}]` — a path that is not a file, or not a media type the pipeline can
  open). Adds nothing and leaves the current selection alone when every path is
  rejected.
- `POST /api/swap` — Start a job over the loaded sources/targets. Body is the run
  config (JSON object; every key is optional and falls back to the saved settings).
  Returns `409` if a job is already running, `400` if no target media or no source
  faces have been added.
- `GET /api/progress` — Current job status: `processing`, `paused`, `progress` (0-1),
  `desc`, `error`, plus the rolling `log`, output `parts` and `started_at`.
- `POST /api/stop` — Abort the running job and finalize a playable output video.
- `POST /api/pause` — Pause at the next frame boundary (no-op when idle).
- `POST /api/resume` — Resume a paused job.

There is no `POST /api/start`; starting a job is `POST /api/swap`, and it operates on
media already registered through `/api/source/add` and `/api/target/add`.

### Code Examples

#### cURL
```bash
API=http://127.0.0.1:${ROOP_API_PORT:-8001}

curl -F "files=@face.jpg"  "$API/api/source/add"
curl -F "files=@clip.mp4"  "$API/api/target/add"

# Or, for a file that is already on this machine — no transfer, no second copy:
curl -X POST "$API/api/target/add_path" -H 'Content-Type: application/json' \
     -d '{"paths":["/abs/path/clip.mp4"]}'
curl -X POST "$API/api/swap" -H 'Content-Type: application/json' \
     -d '{"enhancer":"GFPGAN","detection":"All faces"}'

curl "$API/api/progress"
curl -X POST "$API/api/stop"
```

#### JavaScript (Fetch)
```javascript
const API = `http://127.0.0.1:${8001}`;   // or the port shown in the launcher terminal

// Start a job over the already-loaded source faces and target media
await fetch(`${API}/api/swap`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ enhancer: 'GFPGAN', detection: 'All faces' }),
});

// Poll progress
const res = await fetch(`${API}/api/progress`);
const data = await res.json();
console.log(data.progress, data.desc);

// Stop the job (finalizes the video)
await fetch(`${API}/api/stop`, { method: 'POST' });
```

#### Python (Requests)
```python
import os, requests

API = f"http://127.0.0.1:{os.environ.get('ROOP_API_PORT', '8001')}"

with open("face.jpg", "rb") as f:
    requests.post(f"{API}/api/source/add", files={"files": f})

# The target is on the same machine as the server, so reference it rather than
# uploading it — a multi-gigabyte clip would otherwise cross a loopback socket
# and be written to disk a second time.
requests.post(f"{API}/api/target/add_path", json={"paths": [os.path.abspath("clip.mp4")]})

requests.post(f"{API}/api/swap", json={"enhancer": "GFPGAN", "detection": "All faces"})

print(requests.get(f"{API}/api/progress").json())
requests.post(f"{API}/api/stop")
```

---

## License

See [app/LICENSE](app/LICENSE).

