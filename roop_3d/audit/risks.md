# Roop 3D Architecture: Hardware Probe & Risk Register

**Audit Date:** 2026-10-30T14:43:00+05:30  
**Repository:** `G:\pinokio\api\roop-ultimate`  
**Commit SHA:** `71ebb65c5fa5f11a72e343e8fc4f285bf591b555` (Branch: `main`)  
**Auditor Role:** Senior Repository Auditor & ML Systems Architect  

---

## 1. Active Hardware & Execution Environment Probe

The active workstation has been inspected directly via live system and runtime probes (`nvidia-smi`, PyTorch, ONNX Runtime, and TensorRT):

| Component | Active Probed Specification | Verification Command / Source |
|---|---|---|
| **Host Operating System** | Windows 11 Home / Pro (AMD64, Version 10.0.26100) | `platform.platform()` |
| **Primary GPU** | **NVIDIA GeForce RTX 4070 Desktop** (AD104 Silicon, SM 8.9 Ada Lovelace) | `nvidia-smi` / `torch.cuda.get_device_name(0)` |
| **Total VRAM** | **12,282 MiB** (~12.0 GB Dedicated GDDR6X) | `torch.cuda.get_device_properties(0).total_memory` |
| **Free / Available VRAM** | **10,486 MiB** available (~1,796 MiB active OS/display allocation) | `nvidia-smi` live process memory probe |
| **Graphics Driver** | **621.01** (WDDM 3.2) | NVIDIA-SMI Driver Query |
| **CUDA Driver (UMD)** | **13.5** | NVIDIA-SMI UMD Version |
| **CUDA Runtime (PyTorch)**| **12.8** | `torch.version.cuda` (`2.7.0+cu128`) |
| **PyTorch Version** | **2.7.0+cu128** (`torch.cuda.is_available() == True`) | Active Python venv (`app/env/Scripts/python.exe`) |
| **ONNX Runtime** | **1.23.2** | `onnxruntime.__version__` |
| **ORT Execution Providers**| `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']` | `onnxruntime.get_available_providers()` |
| **TensorRT Library** | **10.9.0.34** | `import tensorrt; tensorrt.__version__` |

### Dual-Device Environment Context
Per [`AGENTS.md`](file:///G:/pinokio/api/roop-ultimate/AGENTS.md), this codebase must strictly support two target profiles:
1. **MAIN Device (Active Desktop Workstation):** RTX 4070 12 GB, 32 GB System RAM, high VRAM tier (`perf_trt_pool: 2`, `perf_detmask_pool: 2`), mixed precision TensorRT enabled (+20% throughput).
2. **SECONDARY Device (Laptop Workstation):** RTX 3060 Laptop 6 GB, 16 GB System RAM, low VRAM tier (`< 7 GB`), `0 / 0` single context, global GPU guard lock, hard-cap 1536 MB parallel stabilization. **TensorRT is NOT admitted on the 3060; all TensorRT optimizations are inert there.**

---

## 2. Risk Register & Mitigation Matrix

### Risk 1: 6 GB VRAM Pressure on Secondary Device (RTX 3060 Laptop)
* **Risk Severity:** HIGH
* **Vulnerability Description:**
  The 3D face pipeline introduces multiple concurrent deep neural networks: 3D face landmark extraction (`1k3d68`), LivePortrait motion extraction (112 MB) and warping SPADE generator (421 MB), identity swapping (Inswapper 128 at 554 MB or HyperSwap 256 at 402 MB), face restoration (GPEN 512 at 284 MB / CodeFormer at 376 MB), and mesh/blendshape deformation. Concurrently allocating these models into GPU memory easily exceeds 6.0 GB VRAM.
  When VRAM overflows on Windows WDDM, PyTorch and ONNX Runtime trigger PCIe paging to shared system RAM, collapsing frame rates from 25+ fps down to <1.5 fps or triggering catastrophic CUDA Out-Of-Memory (`CUDA error: out of memory`) faults.
* **Mitigations for Subsequent Stages:**
  1. **Strict Sequential Model Residency:** Models must not remain pinned in GPU VRAM across independent pipeline stages on low-memory profiles. Unload motion extractor and feature extractor memory before invoking heavy restorers.
  2. **Global GPU Guard Lock:** Preserve and honor the existing single-context guard lock (`_gpu_lock` / `CUDA_EXECUTION_POLICY.md`) on the secondary device.
  3. **Adaptive Block Sizing & Hard Cap:** Maintain the secondary hard cap at 1536 MB (`ROOP_STAB_CHUNK_MB`) and enforce batch size 1 for all 3D mesh and warping operations.
  4. **Fallback Enhancement Tiers:** Disallow 1024x1024 restoration passes (e.g. GPEN 1024) on the 3060; enforce GPEN 256 Pro or FP16 GPEN 512.

---

### Risk 2: Untrusted Archives & Deserialization Security
* **Risk Severity:** CRITICAL
* **Vulnerability Description:**
  - **PyTorch Pickle Deserialization:** Pretrained checkpoints (`.pth`, `.pt`, `.bin`) frequently utilize standard Python `pickle` deserialization under `torch.load()`. A malicious checkpoint loaded from an untrusted community source can contain arbitrary opcode execution in its `__reduce__` method, compromising the host system.
  - **FaceSet Archive Zip Exploits:** FaceSets (`.fsz`) are ZIP containers. Maliciously crafted `.fsz` files can attempt Zip Bomb attacks (expanding multi-gigabyte decompression streams in memory) or directory traversal (`../../app/run.py` overwriting core application code).
* **Mitigations for Subsequent Stages:**
  1. **Safe Serialization Standards:** Mandate `weights_only=True` on all `torch.load()` invocations across the 3D pipeline.
  2. **Format Conversion Gate:** Strictly disallow deploying raw `.pth` checkpoints in production pipelines; convert PyTorch models (e.g. Wav2Lip, DMDNet) to standardized `.onnx` or `.safetensors`.
  3. **Defensive Archive Unpacking:** In `routes_faceset.py` and `faceset_v2.py`, sanitize all ZIP entry names using `os.path.basename`, reject entries with path separators or `..`, and enforce a 50 MB decompression ceiling per FaceSet.

---

### Risk 3: Path Traversal & Unauthorized Filesystem Mutation
* **Risk Severity:** HIGH
* **Vulnerability Description:**
  The REST API in `app/api.py` and `app/routes_faceset.py` receives user-supplied filenames for FaceSets, video targets, and project outputs. If filenames containing traversal tokens (e.g. `../../windows/system32/` or `../../config.yaml`) are processed without canonicalization, arbitrary files outside the workspace may be read, corrupted, or deleted.
* **Mitigations for Subsequent Stages:**
  1. **Path Canonicalization:** Pass all incoming filesystem paths through `os.path.realpath(os.path.abspath(path))`.
  2. **Sandbox Boundary Assertions:** Verify that the canonical path strictly starts with the resolved prefix of allowed directories (`app/facesets/`, `roop_3d/facesets/`, `output/`).
  3. **Strict Slugification:** Enforce alphanumeric + underscore slugification (`_slugify_faceset_name`) on all user-supplied identifiers.

---

### Risk 4: Timestamp Drift & Audio-Video Desynchronization
* **Risk Severity:** MEDIUM
* **Vulnerability Description:**
  Audio-driven animation (Wav2Lip) and temporal 3D expression transfer require tight synchronization between audio frames (processed via STFT spectrogram windows every 20–25 ms) and video frames. Broadcast footage and consumer recordings commonly employ fractional frame rates (23.976 fps, 29.97 fps, 59.94 fps) or Variable Frame Rates (VFR). Naively incrementing integer frame indices or rounding FPS leads to cumulative timestamp drift, causing lipsync lag that worsens continuously over multi-minute sequences.
* **Mitigations for Subsequent Stages:**
  1. **Rational Frame Rate Probing:** Utilize exact fractional rational representations (`detect_fps_fractional`) rather than float or integer approximations.
  2. **Constant Frame Rate (CFR) Normalization:** Apply FFmpeg CFR transcoding (`cfr_video_filter`) before feeding frames to 3D motion and lipsync analyzers.
  3. **Absolute PTS Timeline Alignment:** Anchor audio windows and 3D blendshape keys to absolute milliseconds (PTS) rather than relative frame counters.

---

### Risk 5: Cloud Upload Consent & Biometric Privacy
* **Risk Severity:** CRITICAL (Regulatory / Legal / Ethical)
* **Vulnerability Description:**
  Facial geometry, 512-dimensional identity embeddings (ArcFace), 3D facial landmarks (`1k3d68`), and dermal frequency patches constitute sensitive biometric data subject to global privacy regulations (GDPR Article 9, Illinois BIPA, California CCPA). Any inadvertent transmission of facial imagery or extracted embeddings to third-party cloud APIs represents an existential privacy violation and breach of user trust.
* **Mitigations for Subsequent Stages:**
  1. **Zero-Telemetry Local Air-Gap:** All 3D reconstruction, landmark tracking, identity swapping, and restoration must execute 100% locally on-device. No telemetry, analytics, or remote API queries are permitted in the pipeline.
  2. **Explicit User Opt-In Barrier:** If any optional generative model (such as cloud-hosted FLUX.1 or remote diffusion services) is ever integrated, the application must require an explicit, un-bypassable user consent dialog confirming that external transmission will occur.
  3. **Secure In-Memory Retention:** Sensitive biometric intermediate buffers (embedding vectors, crop banks) must be purged from memory when sessions terminate.
