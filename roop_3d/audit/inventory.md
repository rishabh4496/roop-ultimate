# Roop Ultimate — Repository & Environment Inventory

**Audit Date:** 2026-10-30T14:43:00+05:30  
**Audit Directory:** `roop_3d/audit/`  
**Auditor Role:** Senior Repository Auditor & ML Systems Architect  

---

## 1. Environment & Source Control Baseline

| Property | Value | Evidence / Command |
|---|---|---|
| **Repository Root** | `G:\pinokio\api\roop-ultimate` | `Get-Location` |
| **Current Git Branch** | `main` | `git rev-parse --abbrev-ref HEAD` |
| **Commit SHA** | `71ebb65c5fa5f11a72e343e8fc4f285bf591b555` | `git rev-parse HEAD` |
| **Latest Commit Message** | `fix(watchdog): stop comparing the occluder mask against the face-mask engine's` | `git log -1 --oneline` |
| **Operating System** | `Windows 11 (AMD64, Version 10.0.26100)` | `platform.platform()` |
| **Python Version (Active venv)** | `Python 3.10.20` (`app/env/Scripts/python.exe`) | `python --version` |
| **Python Version (Miniforge)** | `Python 3.10.20` (`G:\pinokio\bin\miniforge\python.exe`) | `G:\pinokio\bin\miniforge\python.exe --version` |
| **Node.js Version** | `v24.18.0` (`G:\pinokio\bin\miniforge\node.exe`) | `node --version` |
| **FFmpeg Version** | `8.1.2` (`G:\pinokio\bin\miniforge\Library\bin\ffmpeg.exe`) | `ffmpeg -version` |
| **Working-Tree Status** | **Clean** (0 uncommitted modifications) | `git status --porcelain` (empty output) |

---

## 2. Top-Level Repository Directory Tree

The checked-out repository consists of the following primary top-level directory structure:

```
roop-ultimate/
├── .cache/                       # Local model cache (.cache/models) and TRT compiled engines
├── .git/                         # Git repository metadata
├── .githooks/                    # Custom repository hooks (e.g., pre-commit checks)
├── .github/                      # CI/CD workflows and GitHub templates
├── app/                          # Core Roop Ultimate application backend
│   ├── env/                      # Primary Python 3.10 virtual environment
│   ├── facesets/                 # Local FaceSet (.fsz) library
│   ├── models/                   # Local model weights and trt_cache
│   ├── roop/                     # Core computational algorithms, pipelines, enhancers
│   ├── ui/                       # Legacy Gradio UI (deprecated)
│   ├── api.py                    # FastAPI server entrypoint
│   ├── config.yaml               # Runtime configuration
│   └── settings.py               # Single source of truth for application settings
├── benchmarks/                   # Benchmark baselines and trace results
├── docs/                         # Architecture documentation, session logs, audit reports
├── face_engine/                  # Standalone face processing engine product
├── facesets/                     # Root FaceSet repository directory
├── models/                       # Root model storage (e.g. models/buffalo_l)
├── node_modules/                 # Root Node.js dependencies
├── react-ui/                     # Supported modern React frontend control surface
├── roop_3d/                      # [ISOLATED] Roop 3D architecture modules
│   └── audit/                    # [ISOLATED] Audit deliverables only
├── scripts/                      # Benchmark and verification scripts
├── tests/                        # Repository integration, precision, and regression test suites
├── tools/                        # Utility scripts, engine compilers, settings generators
├── web_ui/                       # Face engine dedicated web surface
├── AGENTS.md                     # Single source of truth for agent rules and policies
├── CUDA_EXECUTION_POLICY.md      # CUDA stream and memory concurrency policy
├── OPTIMIZATION_PLAN.md          # Multi-stage architectural roadmap
├── PERFORMANCE_BASELINE.md       # Measured hardware baselines
├── PRECISION_POLICY.md           # Model precision (FP32/FP16/TensorRT) rules
├── pinokio.js                    # Pinokio launcher configuration
├── run.py                        # Direct Python entrypoint
├── start.js                      # Root Pinokio entrypoint (delegates to start_react.js)
├── start_react.js                # Supported React launcher
└── requirements.txt              # Top-level Python dependency specification
```

---

## 3. Launchers & Entrypoints Inventory

1. **`start.js`**: Default Pinokio entrypoint. Automatically delegates execution to `start_react.js`.
2. **`start_react.js`**: Supported production launcher. Instantiates FastAPI backend on dynamic port and launches Vite React UI frontend.
3. **`start_legacy.js`**: Deprecated legacy launcher. Runs `app/legacy_ui_check.py` before starting Gradio UI.
4. **`pinokio.js`**: Pinokio service lifecycle definition, environment initialization, process control.
5. **`install.js` / `update.js` / `reset.js`**: Pinokio installation, dependency update, and clean-slate reset workflows.
6. **`pause.js` / `resume.js` / `stop.js` / `torch.js` / `link.js`**: Subordinate process lifecycle management scripts.
7. **`run.py`**: Direct command-line python execution entrypoint supporting `--benchmark`, `--headless`, and batch flags.
8. **`pinokio_batch_runner.py`**: Specialized runner for executing headless batch jobs across directories.

---

## 4. Configuration Files Inventory

1. **`app/config.yaml`**: Primary backend configuration; specifies `swap_model`, `mask_engine`, `face_detector`, `threads`, execution providers, and enhancer presets.
2. **`app/settings.py` & `app/settings.schema.json`**: Single source of truth for configuration keys. Enforces schema validation across UI and backend.
3. **`pinokio.json`**: Launcher metadata, title, icon, and menu hierarchy.
4. **`package.json` & `package-lock.json`**: Node.js dependencies for root scripts.
5. **`react-ui/package.json`**: React/Vite dependencies for the supported frontend.
6. **`web_ui/package.json`**: TypeScript/Vite dependencies for `face_engine`.
7. **`pyproject.toml` & `pytest.ini`**: Python package definitions, pytest test markers (`gpu`, `perf`), lint configurations.
8. **`requirements.txt`**: Python dependencies list.
9. **`update_manifest.json`**: File synchronization manifest for Pinokio updates.

---

## 5. FaceSet Readers & Writers

| Module Path | Core Functionality & Responsibilities | Key Classes / Functions |
|---|---|---|
| [`app/roop/FaceSet.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/FaceSet.py) | FaceSet container data structure; maintains in-memory face crop collections, 3D face bank (`face_3d`, `face_3d_bank`), pose cells, and embedding averages. | `FaceSet`, `AverageEmbeddings`, `pose_bin_embedding`, `get_pose_matched_embedding` |
| [`app/roop/faceset_v2.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/faceset_v2.py) | FaceSet V2 container format reader and writer. Encapsulates `metadata.json` schema (version 2), dermal patch frequency analysis, 3D pose matrix bins, and ZIP member I/O. | `read_faceset_archive`, `_zip_write`, `validate_metadata`, `_dermal_patch`, `pose_matrix_cell` |
| [`app/roop/faceset_manager.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/faceset_manager.py) | High-level library manager; scans FaceSet directories, computes SHA-256 hashes, maintains memory cache, and validates archive integrity. | `load_and_validate_faceset`, `clear_cache`, `describe_faceset`, `matches_stored` |
| [`app/routes_faceset.py`](file:///G:/pinokio/api/roop-ultimate/app/routes_faceset.py) | FastAPI REST API endpoints for FaceSet library operations. | `faceset_library_list`, `faceset_library_save`, `faceset_library_load`, `faceset_library_delete`, `faceset_library_import` |

---

## 6. Video I/O Modules

| Module Path | Core Architecture & Responsibilities | Key Implementations |
|---|---|---|
| [`app/roop/video_io_hw.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/video_io_hw.py) | Hardware-accelerated decoding and encoding via NVDEC/NVENC; spawns low-overhead FFmpeg worker pipes and manages zero-copy GPU buffers. | `HwVideoDecoder`, `HwVideoEncoder`, `probe_hw_support` |
| [`app/roop/video_io_optimizer.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/video_io_optimizer.py) | Video pipeline heuristics; dynamic thread pool calculation, buffer chunk sizing, batch streaming throughput optimizer. | `optimize_video_io`, `estimate_chunk_mb` |
| [`app/roop/video_stream.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/video_stream.py) | Asynchronous multi-threaded frame reader queue; decouples I/O disk reading from GPU inference workers. | `VideoStream`, `drain_stream`, `release_stream` |
| [`app/roop/capturer.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/capturer.py) | Video capture abstraction supporting both OpenCV `cv2.VideoCapture` and FFmpeg frame pipes with millisecond frame seeking. | `Capturer`, `get_frame_at`, `seek_frame` |
| [`app/roop/hardware_streamer.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/hardware_streamer.py) | Low-latency real-time video streaming pipeline for live web preview rendering. | `HardwareStreamer`, `stream_frames` |
| [`app/roop/distributed_render.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/distributed_render.py) | Block-based parallel stabilization (`hard_cap: 4096 MB` on 4070; `1536 MB` on 3060) with automatic 2-round work stealing. | `DistributedRender`, `stabilize_parallel` |
| [`app/roop/utilities.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/utilities.py) | Standard FFmpeg process management for frame extraction, video reassembly, audio demuxing, and constant frame rate filtering. | `extract_frames`, `create_video`, `detect_fps_fractional`, `cfr_video_filter`, `audio_sample_rate` |
| [`face_engine/media/capturer.py`](file:///G:/pinokio/api/roop-ultimate/face_engine/media/capturer.py) | Dedicated capture pipeline for `face_engine`. | `MediaCapturer` |
| [`face_engine/media/video_hw.py`](file:///G:/pinokio/api/roop-ultimate/face_engine/media/video_hw.py) | Dedicated hardware I/O routines for `face_engine`. | `NvdecReader`, `NvencWriter` |
