#!/usr/bin/env python3

import os
import sys
import shutil
import threading as _threading
import time as _time
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
import torch


import onnxruntime as ort
available_providers = ort.get_available_providers()
print("Available ONNX providers at startup:", available_providers)  # Debug

import pathlib
import argparse

from time import time
from roop.utilities import print_cuda_info
import roop.globals
import roop.metadata
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total, release_video
from roop.backend_manager import (resolve_provider_names, diagnostic_report,
                                  cache_namespace, trt_tuning_namespace)


clip_text = None

call_display_ui = None


def _remove_file_retry(path, attempts=5, delay=0.3):
    for i in range(attempts):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return  # already gone — nothing to do
        except PermissionError:
            if i < attempts - 1:
                _time.sleep(delay)
    print(f'[Warning] Could not delete temp file after {attempts} attempts: {path}')

process_mgr = None
_preview_process_mgr = None   # dedicated instance for live_swap — never shared with batch
# live_swap re-initializes the shared _preview_process_mgr on every call (releasing
# and rebuilding processors), so overlapping /api/preview requests — e.g. the
# enhancer comparison grid firing while a regular preview is in flight — release
# ONNX sessions out from under a running inference (NoneType io_binding crashes,
# NaN → black face output). All preview swaps must run one at a time.
_preview_lock = _threading.Lock()


# NOTE: upstream deleted the module-level `torch` name here on ROCm. That freed
# nothing (the module stays in sys.modules) and left every later `torch.` in this
# file raising NameError on an AMD/ROCm install — inert on NVIDIA, fatal there.
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    # Windows: also finalize the video if the console is closed via the X button
    # (that path never raises SIGINT, so the handler above wouldn't fire).
    install_console_close_handler()
    roop.globals.headless = False

    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--server_share', help='Public server', dest='server_share', action='store_true', default=False)
    program.add_argument('--cuda_device_id', help='Index of the cuda gpu to use', dest='cuda_device_id', type=int, default=0)
    roop.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    import onnxruntime
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)
    except Exception:
        pass

    # Resolve through one capability-aware hierarchy.  This prevents a CUDA or
    # TensorRT provider that is merely listed by ORT from becoming the sole
    # backend and failing later inside a model session.  CPU remains an ordered
    # fallback for every GPU hierarchy.
    resolved_names = resolve_provider_names(execution_providers,
                                            getattr(roop.globals, 'cuda_device_id', 0))
    requested_name = str(execution_providers[0] if execution_providers else '').lower()
    if ('tensorrt' in requested_name and
            not any('tensorrt' in str(p).lower() for p in resolved_names)):
        print('[Backend] sub-7GB GPU: TensorRT disabled by the laptop RSS '
              'safety policy; using CUDA/CPU providers. Set '
              'ROOP_ALLOW_TRT_SMALL_GPU=1 to override.')
    available = onnxruntime.get_available_providers()
    list_providers = [provider for provider in available
                      if provider in resolved_names]
    
    try:
        for i in range(len(list_providers)):
            if list_providers[i] == 'CUDAExecutionProvider':
                cuda_opts = {
                    'device_id': roop.globals.cuda_device_id,
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                    'arena_extend_strategy': os.environ.get('ROOP_CUDA_ARENA_STRATEGY', 'kSameAsRequested'),
                }
                cuda_mem_limit = os.environ.get('ROOP_CUDA_MEM_LIMIT')
                if cuda_mem_limit:
                    try:
                        cuda_opts['gpu_mem_limit'] = int(cuda_mem_limit)
                    except ValueError:
                        pass
                list_providers[i] = ('CUDAExecutionProvider', cuda_opts)
                torch.cuda.set_device(roop.globals.cuda_device_id)
            elif list_providers[i] == 'TensorrtExecutionProvider':
                trt_cache = str(pathlib.Path(__file__).parent.parent / 'models' / 'trt_cache')
                os.makedirs(trt_cache, exist_ok=True)
                trt_precision = getattr(roop.globals.CFG, 'trt_precision', 'mixed') if roop.globals.CFG else 'mixed'
                fp16_enable = trt_precision in ('fp16', 'mixed')
                # Include precision, GPU compute capability and ORT ABI in the
                # parent namespace. TensorRT's graph hash alone is not enough
                # to safely reuse an engine after a runtime/device change.
                # Build/runtime tuning knobs are part of the cache identity.
                # Without this suffix, a benchmark or a changed default can
                # silently keep loading an engine made with another schedule.
                try:
                    builder_opt = int(os.environ.get(
                        'ROOP_TRT_BUILDER_OPT_LEVEL',
                        '3'))
                except (TypeError, ValueError):
                    builder_opt = 3
                builder_opt = max(0, min(5, builder_opt))
                try:
                    auxiliary_streams = int(os.environ.get(
                        'ROOP_TRT_AUX_STREAMS', '-1'))
                except (TypeError, ValueError):
                    auxiliary_streams = -1
                # -1 delegates to TensorRT heuristics; 0 is the memory-saving
                # serial mode. Keep a conservative upper bound for pooled
                # contexts on the 6GB laptop.
                auxiliary_streams = max(-1, min(8, auxiliary_streams))
                cuda_graph_value = os.environ.get(
                    'ROOP_TRT_CUDA_GRAPH', '0').strip().lower()
                cuda_graph = cuda_graph_value in ('1', 'true', 'yes', 'on')
                cache_label = cache_namespace(trt_precision,
                                              roop.globals.cuda_device_id)
                # LayerNorm fallback changes TensorRT's graph partitioning and
                # therefore must not reuse engines built with the old setting.
                if trt_precision == 'mixed':
                    cache_label += '_lnfp32_seq_heur'
                cache_label += trt_tuning_namespace(builder_opt,
                                                     auxiliary_streams,
                                                     cuda_graph)
                precision_cache = os.path.join(trt_cache, cache_label)
                os.makedirs(precision_cache, exist_ok=True)

                # ── Engine-build tuning, scaled to the GPU ──────────────────
                try:
                    total_vram = torch.cuda.get_device_properties(roop.globals.cuda_device_id).total_memory
                except Exception:
                    total_vram = 0
                total_gb = total_vram / (1024 ** 3) if total_vram else 0
                
                env_mb = os.environ.get('ROOP_TRT_WORKSPACE_MB')
                if env_mb:
                    try:
                        workspace_size = int(float(env_mb) * 1024 * 1024)
                    except ValueError:
                        workspace_size = 0
                else:
                    env_frac = os.environ.get('ROOP_TRT_WORKSPACE_FRACTION')
                    if env_frac is not None:
                        try:
                            ws_frac = float(env_frac)
                        except ValueError:
                            ws_frac = 0.4
                    else:
                        if total_gb >= 15:
                            ws_frac = 0.5
                        elif total_gb >= 10:
                            ws_frac = 0.4
                        else:
                            ws_frac = 0.3
                    ws_frac = max(0.1, min(0.95, ws_frac))
                    calc_size = int(total_vram * ws_frac) if total_vram else 0
                    # Cap per-session workspace at 2GB max so pooled TRT sessions
                    # do not overcommit VRAM and trigger PCIe shared memory paging.
                    # Guarded like every other env read here: an unparseable
                    # value is a tuning typo, and letting it raise would take
                    # provider setup — and so the whole app — down at startup.
                    _default_cap = 2 * 1024 * 1024 * 1024
                    try:
                        max_cap = int(os.environ.get('ROOP_TRT_MAX_WORKSPACE_BYTES', _default_cap))
                    except (TypeError, ValueError):
                        max_cap = _default_cap
                    workspace_size = min(calc_size, max_cap) if calc_size > 0 else calc_size

                print(f"[TRT] device {total_gb:.1f}GB VRAM -> workspace limit "
                      f"({workspace_size / (1024**3):.1f}GB), partition_iters from env/default")
                try:
                    partition_iters = int(os.environ.get('ROOP_TRT_PARTITION_ITERATIONS', '2000'))
                except ValueError:
                    partition_iters = 2000

                trt_opts = {
                    'device_id': roop.globals.cuda_device_id,
                    'trt_fp16_enable': fp16_enable,
                    # CodeFormer/UltraMax contains LayerNorm + attention
                    # reductions that can overflow when TensorRT chooses
                    # FP16 kernels.  Keep the graph mixed, but explicitly
                    # retain those accuracy-sensitive reductions in FP32.
                    # This option is harmless for GPEN/RetinaFace graphs
                    # without LayerNorm and prevents enhancer smearing/flat
                    # output on mixed-precision builds.
                    'trt_layer_norm_fp32_fallback': trt_precision == 'mixed',
                    # Avoid concurrent tactic compilation stalls on large
                    # enhancer graphs during a fresh mixed-cache build.
                    'trt_force_sequential_engine_build': trt_precision == 'mixed',
                    # Heuristic tactic selection keeps cold-start builds
                    # bounded for the large GPEN/CodeFormer graphs. Runtime
                    # inference remains TensorRT mixed; this only changes how
                    # the first engine is searched and built.
                    'trt_build_heuristics_enable': trt_precision == 'mixed',
                    # ORT documents level 3 as the default-quality baseline;
                    # level 1 can leave performance on the table on large
                    # enhancer graphs. This is build-time only.
                    'trt_builder_optimization_level': builder_opt,
                    # CUDA graphs are deliberately opt-in: they require
                    # stable shapes and context lifetimes, and UltraMax has
                    # multiple pooled sessions on the desktop profile.
                    'trt_cuda_graph_enable': cuda_graph,
                    # -1 lets TensorRT choose; 0 can reduce memory pressure.
                    'trt_auxiliary_streams': auxiliary_streams,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': precision_cache,
                    'trt_max_partition_iterations': partition_iters,
                    'trt_context_memory_sharing_enable': True,
                    'trt_timing_cache_enable': True,
                    'trt_timing_cache_path': precision_cache,
                }
                if workspace_size > 0:
                    trt_opts['trt_max_workspace_size'] = workspace_size
                list_providers[i] = ('TensorrtExecutionProvider', trt_opts)
            elif list_providers[i] == 'DmlExecutionProvider':
                dml_opts = {
                    'device_id': roop.globals.cuda_device_id,
                }
                list_providers[i] = ('DmlExecutionProvider', dml_opts)
            elif list_providers[i] == 'ROCMExecutionProvider':
                rocm_opts = {
                    'device_id': roop.globals.cuda_device_id,
                    'arena_extend_strategy': 'kSameAsRequested',
                }
                list_providers[i] = ('ROCMExecutionProvider', rocm_opts)
            elif list_providers[i] == 'CoreMLExecutionProvider':
                coreml_opts = {
                    'coreml_flags': 0,
                }
                list_providers[i] = ('CoreMLExecutionProvider', coreml_opts)
    except Exception as exc:
        print(f"[Backend] provider option setup warning: {exc}")

    return list_providers
    
# Force GPU if available
# roop.globals.execution_providers = decode_execution_providers(['cuda'])
# print("Forced execution providers:", roop.globals.execution_providers)  # Debug

def suggest_max_memory() -> int:
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if platform.system().lower() == 'darwin':
            return max(2, int(total_ram_gb * 0.4))
        return max(4, min(64, int(total_ram_gb * 0.7)))
    except Exception:
        if platform.system().lower() == 'darwin':
            return 4
        return 16


def suggest_execution_providers() -> List[str]:
    return [p.replace('ExecutionProvider', '').lower()
            for p in resolve_provider_names(['auto'],
                                            getattr(roop.globals, 'cuda_device_id', 0))]


def suggest_execution_threads() -> int:
    # decode_execution_providers() wraps provider entries as (name, opts)
    # tuples, so compare against the extracted names, not the raw list.
    provider_names = [p[0] if isinstance(p, (list, tuple)) else p
                      for p in roop.globals.execution_providers]
    if 'DmlExecutionProvider' in provider_names or 'ROCMExecutionProvider' in provider_names:
        return 1
    if 'CoreMLExecutionProvider' in provider_names:
        try:
            import psutil
            cores = psutil.cpu_count(logical=False) or 4
            return max(1, min(4, cores // 2))
        except Exception:
            return 2

    suggested = 8
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or 4
        if any(p in provider_names for p in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']):
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(roop.globals.cuda_device_id).total_memory / (1024**3)
                suggested = int(min(max(2, cores - 1), max(2, vram_gb / 1.5)))
        elif 'CPUExecutionProvider' in provider_names:
            suggested = max(1, cores - 1)
    except Exception:
        pass
    
    return suggested


def limit_resources() -> None:
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def release_resources() -> None:
    import gc
    from roop.face_util import release_face_analyser
    from roop.capturer import clear_frame_cache, release_video
    global process_mgr, _preview_process_mgr

    release_face_analyser()
    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None
    with _preview_lock:
        if _preview_process_mgr is not None:
            _preview_process_mgr.release_resources()
            _preview_process_mgr = None

    clear_frame_cache()
    release_video()
    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                with torch.cuda.device(roop.globals.cuda_device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        except Exception:
            pass


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    # Pre-warm the model cache while online. Offline (auto-detected), we skip
    # this entirely and fall back to whatever is already on disk — the app still
    # boots, and a missing model is only reported when its feature is actually
    # used. required=False so even a single failed download online (host down,
    # transient error) warns instead of aborting startup.
    if util.is_online():
        _pre_warm = [
            ('../models', [
                'https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx',
                'https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx',
            ]),
            ('../models/CLIP', [
                'https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth',
            ]),
            ('../models/CodeFormer', [
                'https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx',
                # Half-precision export of the same graph — 189 MB against 377,
                # and 1.60x faster on CUDA (measured 162.9 -> 102.0 ms/call at
                # 512 on an RTX 4070). Downloaded with required=False like the
                # rest, so an offline install simply does not offer the tier.
                'https://huggingface.co/netrunner-exe/Face-Upscalers-onnx/resolve/main/codeformer.fp16.onnx',
            ]),
            ('../models/Frame', [
                'https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx',
                'https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx',
                'https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx',
                'https://huggingface.co/facefusion/models-3.3.0/resolve/main/ultra_sharp_2_x4.onnx',
                'https://huggingface.co/JackCui/facefusion/resolve/main/clear_reality_x4.onnx',
                'https://huggingface.co/wanesoft/faceswap_pack/resolve/main/span_kendata_x4.onnx',
                'https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Viso_Master_Models/realesr-general-x4v3.onnx',
                'https://huggingface.co/wanesoft/faceswap_pack/resolve/main/nomos8k_sc_x4.onnx',
                'https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife49_ensemble_True_scale_1_sim.onnx',
            ]),
        ]
        for subdir, urls in _pre_warm:
            util.conditional_download(util.resolve_relative_path(subdir), urls, required=False)
    else:
        update_status('Offline mode: skipping model pre-download. Using locally available models; any missing model will be reported only when you use the feature that needs it.')

    print_cuda_info()  # Debug CUDA during pre-check


    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    # Format terminal color dynamically based on message contents/keywords
    color_msg = message
    try:
        reset = "\033[0m"
        bold = "\033[1m"
        lower_msg = message.lower()
        if any(kw in lower_msg for kw in ["failed", "error", "stopped", "cannot", "warning"]):
            # Red color for warnings/errors/stops
            color_msg = f"\033[91m{bold}[ERROR] {message}{reset}"
        elif any(kw in lower_msg for kw in ["finished", "success", "completed", "took"]):
            # Green color for successes/completions
            color_msg = f"\033[92m{bold}[SUCCESS] {message}{reset}"
        elif any(kw in lower_msg for kw in ["creating", "extracting", "restoring", "building", "downloading", "processing"]):
            # Yellow/Amber for progression tasks
            color_msg = f"\033[93m{bold}[ACTION] {message}{reset}"
        else:
            # Cyan for standard tracking logs
            color_msg = f"\033[96m[STATUS] {message}{reset}"
    except Exception:
        pass

    print(color_msg)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if roop.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(roop.globals.source_path,  (False, 0))
        # roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        # faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        # roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        # if 'face_enhancer' in roop.globals.frame_processors:
        #     roop.globals.selected_enhancer = 'GFPGAN'
       
    # FIX: was batch_process_regular(None, False, None) — only 3 args for a 10-param function.
    # Headless mode is unsupported in this fork; log and fall through to UI launch.
    print('Headless batch processing is not implemented - falling through to UI.')


def get_processing_plugins(masking_engine, swap_model='inswapper'):
    """Build the processor dict for ProcessOptions.

    `masking_engine` may be one engine name or several. Several compose, and they
    compose the right way round for occlusion: each mask processor blends the
    swapped crop back toward the untouched one wherever it says "not face", so a
    pair of them restores the UNION of what either recognised. That matters
    because the engines are not redundant — they were trained on different data
    and miss different things, and "the mask failed when something came in front
    of the face" is usually one engine not knowing about that particular
    something rather than masking being off.

    Order in the dict is order of execution (dicts preserve insertion), and the
    mask stage runs after the swap and the enhancer because it is inserted last.
    """
    processors = {"faceswap": {"swap_model": swap_model}}

    if roop.globals.selected_enhancer == 'GFPGAN':
        processors.update({"gfpgan": {}})
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors.update({"codeformer": {}})
    elif roop.globals.selected_enhancer == 'Codeformer (fp16)':
        processors.update({"codeformer": {"fp16": True}})
    elif roop.globals.selected_enhancer == 'DMDNet':
        processors.update({"dmdnet": {}})
    elif roop.globals.selected_enhancer == 'GPEN 256':
        processors.update({"gpen": {"size": 256}})
    elif roop.globals.selected_enhancer == 'GPEN 256 Pro':
        # Upgraded GPEN 256: sharper, high dermal micro-textures, photoreal chrominance,
        # and lock-free pooled execution matching native GPEN 256 speed.
        processors.update({"gpen_256_pro": {}})
    elif roop.globals.selected_enhancer == 'GPEN 256 Ultra':
        processors.update({"gpen_256_pro": {}})
    elif roop.globals.selected_enhancer == 'GPEN Realistic':
        # GPEN-256's luminance with the swapper's colour, on a pooled, lean host
        # path. See Enhance_GPENRealistic: the "cartoonish" look people report
        # from GPEN-256 is a 2.96 chroma drift, not a detail deficit -- its
        # detail already beats CodeFormer-512 at a sixth of the cost.
        processors.update({"gpen_realistic": {}})
    elif roop.globals.selected_enhancer == 'GPEN':
        processors.update({"gpen": {"size": 512}})
    elif roop.globals.selected_enhancer == 'GPEN 1024':
        processors.update({"gpen": {"size": 1024}})
    elif roop.globals.selected_enhancer == 'GPEN 2048':
        processors.update({"gpen": {"size": 2048}})
    elif roop.globals.selected_enhancer == 'UltraMax':
        # codeformer.fp16.onnx -- the same weights as 'Codeformer (fp16)' -- on
        # a leaner host path, followed by a structure-gated texture restore. See
        # Enhance_UltraMax for the per-face cost table it is built from.
        processors.update({"ultramax": {}})
    elif roop.globals.selected_enhancer == 'Restoreformer++':
        processors.update({"restoreformer++": {}})
    elif roop.globals.selected_enhancer == 'KEEP (sidecar)':
        # Experimental: runs in sidecar_keep/.venv as a separate process
        # (dependency conflict with the main env); passes through unenhanced
        # when the sidecar isn't installed. See app/sidecar_keep/README.md.
        processors.update({"keep": {}})

    # A bare string stays a bare string for every existing caller (the Gradio
    # tab, virtualcam, the per-frame mask path) — none of them has to learn
    # about this.
    if isinstance(masking_engine, str):
        masking_engine = [masking_engine]
    for engine in (masking_engine or ()):
        if engine and engine not in processors:
            processors[engine] = {}

    return processors


def get_face_crop_from_frame(frame_bgr) -> str:
    """Return a base64 PNG data-URL of the canonical 512×512 aligned face crop from *frame_bgr*.

    Replicates the same autorotation pre-processing that ProcessMgr.process_face uses, so
    the crop shown in the Frame Editor mask modal exactly matches the coordinate space the
    processor operates in.  Returns empty string when no face is detected.
    """
    import base64 as _b64
    import cv2 as _cv2
    from roop.face_util import (get_first_face, align_crop, face_rotation_action,
                                rotation_improves_upright)

    if frame_bgr is None:
        return ""

    face = get_first_face(frame_bgr)
    if face is None or not hasattr(face, 'kps') or face.kps is None:
        return ""

    frame = frame_bgr.copy()
    if roop.globals.autorotate_faces:
        action = face_rotation_action(face, frame.shape[:2])
        if action is not None:
            x0, y0, x1, y1 = face.bbox.astype(int)
            offs = int(max(x1 - x0, y1 - y0) * 0.25)
            x0m = max(0, x0 - offs); y0m = max(0, y0 - offs)
            x1m = min(frame.shape[1], x1 + offs); y1m = min(frame.shape[0], y1 + offs)
            # Share the processor's own turn table rather than an if/else here:
            # the old `else` branch quietly turned clockwise for anything that
            # was not anticlockwise, so a third action would have shown the user
            # a crop the render never produces.
            cut = ProcessMgr.apply_rotation(frame[y0m:y1m, x0m:x1m], action)
            rotface = get_first_face(cut)
            if (rotface is not None and hasattr(rotface, 'kps') and rotface.kps is not None
                    and rotation_improves_upright(face, rotface)):
                face  = rotface
                frame = cut

    crop, _ = align_crop(frame, face.kps, 512)
    ok, buf = _cv2.imencode('.png', crop)
    if not ok:
        return ""
    return "data:image/png;base64," + _b64.b64encode(buf.tobytes()).decode('utf-8')


def live_swap(frame, options, input_facesets=None):
    """Swap a single frame. `input_facesets` overrides the loaded source
    facesets (the API passes a person-ordered remap); None = use them as-is."""
    global _preview_process_mgr

    if frame is None:
        return frame

    facesets = roop.globals.INPUT_FACESETS if input_facesets is None else input_facesets

    with _preview_lock:
        if _preview_process_mgr is None:
            _preview_process_mgr = ProcessMgr(None)
            _preview_process_mgr.is_preview = True

        _preview_process_mgr.initialize(facesets, roop.globals.TARGET_FACES, options)
        newframe = _preview_process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def _parse_per_frame_masks(json_str: str) -> dict:
    """Parse the JSON string from mask_per_frame_store.

    Supports two formats:
    - New: {"frame": {"facesetIdx": maskData, ...}, ...}
    - Old: {"frame": maskData, ...}  — backwards compat, wrapped as {"0": maskData}

    Returns {int_frame_num: {int_faceset_idx: maskData}}.
    """
    import json as _json
    if not json_str:
        return {}
    try:
        raw = _json.loads(json_str)
        if not isinstance(raw, dict):
            return {}
        result = {}
        for k, v in raw.items():
            if not k.isdigit() or not isinstance(v, dict):
                continue
            frame_num = int(k)
            # Detect old flat format: has 'exclude', 'include', or 'canonical' at top level
            is_old_flat = any(x in v for x in ('exclude', 'include', 'canonical'))
            if is_old_flat:
                result[frame_num] = {0: v}
            else:
                per_faceset = {int(fk): fv for fk, fv in v.items()
                               if fk.isdigit() and isinstance(fv, dict)}
                if per_faceset:
                    result[frame_num] = per_faceset
        return result
    except Exception:
        return {}


def _reprocess_custom_mask_frames(temp_frame_paths: list, orig_frame_paths: list,
                                   per_frame_masks: dict, masking_engine, new_clip_text: str,
                                   num_swap_steps: int, restore_original_mouth: bool,
                                   selected_index: int, use_3d_recon: bool,
                                   use_source_bank: bool = False,
                                   use_frontalization: bool = False,
                                   frontalization_threshold: float = 25.0,
                                   swap_model: str = 'inswapper') -> None:
    """Re-process frames that have a custom per-frame mask.

    Strategy:
    - temp_frame_paths contains the already-swapped frames (global-mask run).
    - orig_frame_paths are the pre-swap originals saved by save_original_frames().
    - For each frame number in per_frame_masks, re-run live_swap on the original
      with the custom mask and overwrite the corresponding temp frame.

    Frame numbers in per_frame_masks are 1-based to match the UI slider / JS.
    The temp / orig path lists are 0-based.
    """
    if not per_frame_masks or not orig_frame_paths:
        return

    import cv2 as _cv2
    import json as _json

    plugins = get_processing_plugins(masking_engine, swap_model=swap_model)

    # per_frame_masks: {int_frame_num: {int_faceset_idx: maskData}}
    for frame_num_1, faceset_masks in per_frame_masks.items():
        idx = frame_num_1 - 1          # convert 1-based → 0-based list index
        if idx < 0 or idx >= len(orig_frame_paths):
            continue
        orig_path = orig_frame_paths[idx]
        out_path  = temp_frame_paths[idx] if idx < len(temp_frame_paths) else orig_path

        orig_bgr = _cv2.imread(orig_path)
        if orig_bgr is None:
            print(f"[per-frame mask] could not read original {orig_path}")
            continue

        # Build combined per-faceset mask JSON: {"0": maskData, "1": maskData, ...}
        # ProcessMgr.initialize detects digit-string top-level keys as new format.
        combined_mask = {str(fi): fd for fi, fd in faceset_masks.items()
                         if isinstance(fd, dict)}
        mask_json_str = _json.dumps(combined_mask) if combined_mask else None

        options = ProcessOptions(
            plugins,
            roop.globals.distance_threshold,
            roop.globals.blend_ratio,
            roop.globals.face_swap_mode,
            selected_index,
            new_clip_text,
            mask_json_str,
            num_swap_steps,
            roop.globals.subsample_size,
            False,
            restore_original_mouth,
            use_3d_recon=use_3d_recon,
            use_source_bank=use_source_bank,
            use_frontalization=use_frontalization,
            frontalization_threshold=frontalization_threshold,
            swap_model=swap_model,
        )
        result = live_swap(orig_bgr, options)
        if result is not None:
            _cv2.imwrite(out_path, result)
            print(f"[per-frame mask] frame {frame_num_1} reprocessed → {os.path.basename(out_path)}")


def batch_process_regular(output_method, files:list[ProcessEntry], masking_engine:str, new_clip_text:str, use_new_method, imagemask, restore_original_mouth, num_swap_steps, progress, selected_index = 0, use_3d_recon=False, mask_per_frame_json="",
                          use_source_bank=False, use_frontalization=False,
                          frontalization_threshold=25.0, swap_model='inswapper',
                          stabilize_face=False, stabilize_method='one_euro', stabilize_min_cutoff=0.05, stabilize_beta=0.02,
                          stabilize_enhancer=False, stabilize_enhancer_strength=0.5,
                          stabilize_mask=False, stabilize_mask_strength=0.5,
                          input_facesets=None) -> None:
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    # imagemask is a JSON string produced by the canvas masking modal
    # (keys: "include" and/or "exclude", values: grayscale PNG data-URLs).
    # ProcessMgr.initialize decodes it into include_mask / exclude_mask arrays.
    # `input_facesets` lets the caller hand in a person-ordered remap of the
    # sources without mutating the global (see api.mapped_facesets).
    facesets = roop.globals.INPUT_FACESETS if input_facesets is None else input_facesets
    if len(facesets) <= selected_index:
        selected_index = 0
    options = ProcessOptions(get_processing_plugins(masking_engine, swap_model=swap_model),
                              roop.globals.distance_threshold, roop.globals.blend_ratio,
                              roop.globals.face_swap_mode, selected_index, new_clip_text, imagemask, num_swap_steps,
                              roop.globals.subsample_size, False, restore_original_mouth,
                              use_3d_recon=use_3d_recon,
                              use_source_bank=use_source_bank,
                              use_frontalization=use_frontalization,
                              frontalization_threshold=frontalization_threshold,
                              swap_model=swap_model,
                              stabilize_face=stabilize_face,
                              stabilize_method=stabilize_method,
                              stabilize_min_cutoff=stabilize_min_cutoff,
                              stabilize_beta=stabilize_beta,
                              stabilize_enhancer=stabilize_enhancer,
                              stabilize_enhancer_strength=stabilize_enhancer_strength,
                              stabilize_mask=stabilize_mask,
                              stabilize_mask_strength=stabilize_mask_strength)
    process_mgr.initialize(facesets, roop.globals.TARGET_FACES, options)

    # Stash per-frame mask map and batch options on globals so batch_process can access them
    roop.globals.mask_per_frame = _parse_per_frame_masks(mask_per_frame_json)
    roop.globals._batch_selected_index    = selected_index
    roop.globals._batch_clip_text         = new_clip_text
    roop.globals._batch_num_steps         = num_swap_steps
    roop.globals._batch_restore_mouth     = restore_original_mouth
    roop.globals._batch_use_3d_recon      = use_3d_recon
    roop.globals._batch_use_source_bank   = use_source_bank
    roop.globals._batch_use_frontalization= use_frontalization
    roop.globals._batch_front_threshold   = frontalization_threshold
    roop.globals._batch_swap_model        = swap_model

    batch_process(output_method, files, use_new_method)
    return

def batch_process_with_options(files:list[ProcessEntry], options, progress):
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    roop.globals.keep_frames = False
    roop.globals.wait_after_extraction = False
    roop.globals.skip_audio = False
    batch_process("Files", files, True)



# Set once the first run starts, so each NEW run clears the previous run's
# terminal output while the very first run keeps the startup logs visible.
_terminal_has_previous_run = False

def _clear_terminal_for_new_run() -> None:
    """Clear the terminal when a new processing run starts.

    Long runs print thousands of progress/profiling lines; without this each
    subsequent run piles onto the last and the terminal becomes unreadable.
    Uses ANSI escapes (clear scrollback + clear screen + cursor home), which
    xterm.js (Pinokio's terminal), Windows Terminal, and Unix terminals all
    honour. On classic Windows conhost, VT processing is enabled first.
    Defensive: a failure here must never break processing.
    """
    global _terminal_has_previous_run
    if not _terminal_has_previous_run:
        _terminal_has_previous_run = True
        return
    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                # 0x4 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(h, mode.value | 0x0004)
        print('\x1b[3J\x1b[2J\x1b[H', end='', flush=True)
    except Exception:
        pass


def _warn_single_worker_on_gpu() -> None:
    """One worker on a real GPU provider is always a misconfiguration, and it is
    the only silent one left.

    The dml/rocm override above announces itself. `CFG.max_threads == 1` with a
    working CUDA/TensorRT provider does not — nothing derives 1 (both
    `suggest_execution_threads` and `Settings.resolve_threads` floor at 2 and 4
    respectively) and no UI control writes it, so it can only come from a
    hand-edited config, and it then costs the whole pipeline its parallelism
    without a word.

    It reads to the user as a defect in whatever they happened to render when
    they first noticed. Measured on an RTX 4070, d1.mp4, realswap + RealityUX +
    UltraMax (tests/ab_face_count.py), the arm that gets blamed is always the
    one with more faces in frame:

                        1 thread     10 threads
        one face         3.36 fps      5.57 fps
        two faces        2.20 fps      4.64 fps

    Doubling the swapped faces is SUB-linear at both thread counts (faces/s
    goes UP: 3.36 -> 4.41 at one thread, 5.57 -> 9.27 at ten), so a two-face
    render is not paying a two-face penalty. It is simply the workload heavy
    enough to fall below usable once the threads are gone — which is why the
    same machine reads "fine on one face, 1-2 fps on two".
    """
    try:
        if int(roop.globals.execution_threads or 0) > 1:
            return
        names = [p[0] if isinstance(p, (list, tuple)) else p
                 for p in (roop.globals.execution_providers or [])]
        if not any(n in names for n in ('CUDAExecutionProvider',
                                        'TensorrtExecutionProvider')):
            return   # dml/rocm already said so above; CPU has its own reasons
        print("[Threads] WARNING: running ONE worker thread on a GPU provider. "
              "Nothing in the app derives 1 (the floors are 2 and 4) and no UI "
              "control writes it, so this is Max Threads set to 1 in "
              "config.yaml. Every stage loses its parallelism, and a render "
              "with two faces in frame drops roughly 1.5x below a one-face one "
              "-- which reads as 'two faces are broken' when it is the thread "
              "count. Set Max Threads to 7-10 (or turn Auto Thread Selection "
              "back on) and restart. See tests/diag_device.py.", flush=True)
    except Exception:
        pass


def batch_process(output_method, files:list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    _clear_terminal_for_new_run()

    roop.globals.processing = True
    # Marks the encode as live so a terminal Ctrl-C (destroy()) can wait for the
    # output video to be finalized before exiting. Cleared in end_processing.
    roop.globals.batch_active = True

    # Keep the GPU powered while the display is off so long runs don't freeze
    # (released in end_processing, which every exit path below goes through).
    from roop import keep_awake
    keep_awake.acquire()

    try:
        # limit threads for some providers. DirectML and ROCm are single-worker
        # (see suggest_execution_threads), and this OVERRIDES whatever the user
        # set -- so when it fires it has to say so. Silently discarding an
        # explicit Max Threads is the same defect class as a control bound to
        # something nothing reads: the setting looks wired and is not.
        max_threads = suggest_execution_threads()
        if max_threads == 1 and roop.globals.execution_threads != 1:
            print(f"[Threads] provider {roop.globals.execution_providers} runs "
                  f"single-worker: overriding Max Threads "
                  f"{roop.globals.execution_threads} -> 1. If you did not choose "
                  f"this provider, see the PROVIDER FALLBACK notice at startup "
                  f"(run tests/diag_device.py).", flush=True)
        if max_threads == 1:
            roop.globals.execution_threads = 1
        _warn_single_worker_on_gpu()

        imagefiles:list[ProcessEntry] = []
        videofiles:list[ProcessEntry] = []
           
        update_status('Sorting videos/images')


        for index, f in enumerate(files):
            fullname = f.filename
            if util.is_video(fullname) or util.has_extension(fullname, ['gif']) or util.is_animated_webp(fullname):
                destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'__temp.{roop.globals.CFG.output_video_format}')
                f.finalname = destination
                videofiles.append(f)

            elif util.has_image_extension(fullname):
                destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'.{roop.globals.CFG.output_image_format}')
                destination = util.replace_template(destination, index=index)
                pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
                f.finalname = destination
                imagefiles.append(f)



        if(len(imagefiles) > 0):
            update_status('Processing image(s)')
            origimages = []
            fakeimages = []
            for f in imagefiles:
                origimages.append(f.filename)
                fakeimages.append(f.finalname)

            process_mgr.run_batch(origimages, fakeimages, roop.globals.execution_threads)
            origimages.clear()
            fakeimages.clear()

        if(len(videofiles) > 0):
            # Warm-up: verify the video encoder can actually launch and encode
            # BEFORE the (often very long) analysis pass. Catches a blocked/broken
            # ffmpeg in seconds instead of silently hanging the frame pipe partway
            # through a render and wasting the whole analysis. Most common cause on
            # Windows: Smart App Control blocking an unsigned ffmpeg DLL.
            from roop.ffmpeg_writer import probe_encoder
            enc_ok, enc_msg = probe_encoder(roop.globals.video_encoder, roop.globals.video_quality)
            if not enc_ok:
                update_status(
                    f"Video encoder '{roop.globals.video_encoder}' is not working, aborting. "
                    f"{enc_msg}"
                )
                end_processing('Processing stopped: video encoder unavailable.')
                return

            for index,v in enumerate(videofiles):
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return
                fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
                if v.endframe == 0:
                    v.endframe = get_video_frame_total(v.filename)

                is_streaming_only = output_method == "Virtual Camera"
                if is_streaming_only == False:
                    update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')

                start_processing = time()
                _swaps_before = getattr(process_mgr, 'total_swaps', 0)
                _has_per_frame_masks = bool(getattr(roop.globals, 'mask_per_frame', {}))
                if (is_streaming_only == False and roop.globals.keep_frames) or not use_new_method or (is_streaming_only == False and _has_per_frame_masks):
                    util.create_temp(v.filename)
                    update_status('Extracting frames...')
                    extraction_ok = ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                    if not roop.globals.processing:
                        end_processing('Processing stopped!')
                        return

                    temp_frame_paths = util.get_temp_frame_paths(v.filename)
                    if not temp_frame_paths:
                        # Frame extraction produced no output — ffmpeg likely failed above.
                        # Log and skip this video rather than crashing on temp_frame_paths[0].
                        update_status(f'Frame extraction failed for {os.path.basename(v.filename)}, skipping...')
                        continue

                    # Save unswapped originals BEFORE run_batch overwrites them in-place.
                    # Needed for both keep_frames mode (Frame Editor) and per-frame mask re-processing.
                    per_frame_masks = getattr(roop.globals, 'mask_per_frame', {})
                    needs_originals = roop.globals.keep_frames or bool(per_frame_masks)
                    if needs_originals:
                        util.save_original_frames(v.filename)
                    process_mgr.run_batch(temp_frame_paths, temp_frame_paths, roop.globals.execution_threads)
                    if not roop.globals.processing:
                        end_processing('Processing stopped!')
                        return

                    # Re-process any frames that have custom per-frame masks.
                    if per_frame_masks:
                        update_status('Applying per-frame masks...')
                        orig_paths = util.get_temp_frame_paths_from_dir(util.get_frames_orig_path(v.filename))
                        _reprocess_custom_mask_frames(
                            temp_frame_paths, orig_paths, per_frame_masks,
                            masking_engine=None,
                            new_clip_text=getattr(roop.globals, '_batch_clip_text', ''),
                            num_swap_steps=getattr(roop.globals, '_batch_num_steps', 1),
                            restore_original_mouth=getattr(roop.globals, '_batch_restore_mouth', False),
                            selected_index=getattr(roop.globals, '_batch_selected_index', 0),
                            use_3d_recon=getattr(roop.globals, '_batch_use_3d_recon', False),
                            use_source_bank=getattr(roop.globals, '_batch_use_source_bank', False),
                            use_frontalization=getattr(roop.globals, '_batch_use_frontalization', False),
                            frontalization_threshold=getattr(roop.globals, '_batch_front_threshold', 25.0),
                            swap_model=getattr(roop.globals, '_batch_swap_model', 'inswapper'),
                        )

                    if roop.globals.wait_after_extraction and temp_frame_paths:
                        extract_path = os.path.dirname(temp_frame_paths[0])
                        util.open_folder(extract_path)
                        input("Press any key to continue...")
                        print("Resorting frames to create video")
                        util.sort_rename_frames(extract_path)                                    
                
                    ffmpeg.create_video(v.filename, v.finalname, fps)
                    if roop.globals.keep_frames:
                        util.move_frames_to_output(v.filename, fps=fps)
                    else:
                        util.delete_temp_frames(temp_frame_paths[0])
                        # If we saved originals only for per-frame mask re-processing (not keep_frames),
                        # clean them up now that the video has been compiled.
                        if per_frame_masks and not roop.globals.keep_frames:
                            orig_dir = util.get_frames_orig_path(v.filename)
                            if os.path.isdir(orig_dir):
                                import shutil as _shutil
                                _shutil.rmtree(orig_dir, ignore_errors=True)
                else:
                    if util.has_extension(v.filename, ['gif']) or util.is_animated_webp(v.filename):
                        skip_audio = True
                    else:
                        skip_audio = roop.globals.skip_audio
                    process_mgr.run_batch_inmem(output_method, v.filename, v.finalname, v.startframe, v.endframe, fps,roop.globals.execution_threads, skip_audio)
                
                # A Stop (React run-bar, Pinokio sidebar, Ctrl-C) must NOT skip the
                # finalization below. By the time run_batch_inmem returns, the writer
                # has already closed and merged its segments into the temp video, so
                # returning here left the user with a nameless, audio-less
                # `<name>__temp.mp4` sitting next to the `.seg####.mp4` parts — the
                # merge "not happening" from the UI's point of view. Instead, mark the
                # run stopped, fall through to mux audio + apply the output template,
                # and only then return.
                stopped = not roop.globals.processing

                video_file_name = v.finalname
                # Defined before the isfile() branch: the failure path below falls
                # through to the status line that references it.
                destination = ''
                if os.path.isfile(video_file_name):
                    if util.has_extension(v.filename, ['gif']) or util.is_animated_webp(v.filename):
                        gifname = util.get_destfilename_from_path(v.filename, roop.globals.output_path, '.gif')
                        destination = util.replace_template(gifname, index=index)
                        pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                        update_status('Creating final GIF')
                        # Pass fps explicitly so the GIF matches the original source
                        # timing — avoids a lossy re-detect from the intermediate MP4.
                        ffmpeg.create_gif_from_video(video_file_name, destination, target_fps=fps)
                        if os.path.isfile(destination):
                            _remove_file_retry(video_file_name)
                    else:
                        skip_audio = roop.globals.skip_audio
                        destination = util.replace_template(video_file_name, index=index)
                        pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                        if not skip_audio:
                            # Dubbing against an uploaded track: mux THAT file's audio
                            # instead of the original clip's. restore_audio only reads
                            # stream 1's audio (-map 1:a:0?), so any file works here.
                            audio_source = v.filename
                            if (getattr(roop.globals, 'lipsync_enabled', False)
                                    and getattr(roop.globals, 'lipsync_audio_source', 'original') == 'upload'
                                    and getattr(roop.globals, 'lipsync_audio_path', None)):
                                audio_source = roop.globals.lipsync_audio_path
                            ffmpeg.restore_audio(video_file_name, audio_source, v.startframe, v.endframe, destination)
                            if os.path.isfile(destination):
                                _remove_file_retry(video_file_name)
                        else:
                            shutil.move(video_file_name, destination)

                elif is_streaming_only == False and not stopped:
                    update_status(f'Failed processing {os.path.basename(v.finalname)}!')
                elapsed_time = time() - start_processing
                if stopped:
                    # Partial render: report what was actually saved, skip the runtime
                    # calibration (a truncated run would poison the estimate) and stop
                    # before the remaining queued videos.
                    if destination and os.path.isfile(destination):
                        update_status(f'\nStopped after {elapsed_time:.2f} secs — partial output saved as '
                                      f'{os.path.basename(destination)}')
                    end_processing('Processing stopped!')
                    return
                average_fps = (v.endframe - v.startframe) / elapsed_time
                update_status(f'\nProcessing {os.path.basename(destination or v.filename)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s')
                # Fold this run into the learned runtime estimator. Signature =
                # settings (stashed at run start) + measured face-density bucket
                # (avg faces/frame for THIS video). Guarded — never fatal.
                try:
                    from roop import runtime_calib
                    frames = v.endframe - v.startframe
                    base_sig = getattr(roop.globals, '_run_signature', None)
                    if base_sig:
                        swaps = getattr(process_mgr, 'total_swaps', 0) - _swaps_before
                        avg_faces = swaps / max(1, frames)
                        sig = runtime_calib.with_density(
                            base_sig, runtime_calib.density_bucket(avg_faces))
                        runtime_calib.record(sig, frames, elapsed_time * 1000.0)
                except Exception:
                    pass
                import gc
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        with torch.cuda.device(roop.globals.cuda_device_id):
                            torch.cuda.empty_cache()
                except Exception:
                    pass
        end_processing('Finished')
    finally:
        # Guarantee the run is marked finished even if an exception escaped
        # batch_process (e.g. the legacy Gradio caller has no finally), so a
        # later Ctrl-C / window-close never waits on a batch that is gone.
        keep_awake.release()
        roop.globals.batch_active = False


def end_processing(msg:str):
    from roop import keep_awake
    keep_awake.release()
    update_status(msg)
    roop.globals.target_folder_path = None
    release_resources()
    # Encode fully wound down (writers closed, output finalized). Clear last so a
    # terminal Ctrl-C waiting in destroy() only proceeds once the file is safe.
    roop.globals.batch_active = False


def finalize_active_batch(timeout: float = 120.0) -> bool:
    """Signal any in-progress batch to stop and wait (up to *timeout* seconds) for
    the encode thread to finalize the output video — i.e. close the ffmpeg writer
    so its trailer (moov atom) is written and the file stays playable.

    Shared by every abrupt-exit path (Ctrl-C via destroy(), and the Windows
    console-close handler). Returns True if the batch finished finalizing within
    the timeout. Safe to call when nothing is running (returns True immediately)."""
    if not roop.globals.batch_active:
        return True
    roop.globals.pause = False
    roop.globals.processing = False
    deadline = time() + timeout
    while roop.globals.batch_active and time() < deadline:
        _time.sleep(0.1)
    return not roop.globals.batch_active


def destroy() -> None:
    # Ctrl-C in the terminal lands here (SIGINT handler, see parse_args). If a
    # batch is mid-encode, do NOT hard-exit — that would kill the background
    # encode thread with ffmpeg's pipe still open, leaving a truncated/unplayable
    # output (no moov atom). Instead mirror the UI Stop: signal a graceful stop
    # and wait for the worker to finalize the output video before tearing down.
    if roop.globals.batch_active:
        print('\nStopping — finalizing output video, please wait...')
        if finalize_active_batch(timeout=120):
            print('Output video finalized.')
        else:
            print('Timed out waiting for finalize; exiting anyway.')
    if roop.globals.target_path:
        util.clean_temp(roop.globals.target_path)
    release_resources()
    sys.exit()


# Keeps the ctypes callback alive for the process lifetime (SetConsoleCtrlHandler
# stores only a raw pointer; letting Python GC it would crash on the next event).
_console_handler_ref = None


def install_console_close_handler() -> None:
    """Windows only: finalize the output video when the console window is closed
    with the X button (CTRL_CLOSE_EVENT) or on logoff/shutdown. The OS gives a
    close handler only a few seconds before force-killing the process, so the wait
    is short — enough to flush ffmpeg's trailer in the common case, which is all
    that's needed for the file to be playable. Ctrl-C / Ctrl-Break are left to the
    Python SIGINT handler (destroy) so they aren't handled twice."""
    global _console_handler_ref
    if sys.platform != 'win32' or _console_handler_ref is not None:
        return
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return

    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6
    HANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def _handler(ctrl_type):
        if ctrl_type in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
            if roop.globals.batch_active:
                try:
                    print('\nWindow closing — finalizing output video...')
                except Exception:
                    pass
                # Bounded to stay inside the OS kill window (~5s); the frames are
                # already encoded, so writing the trailer is fast in practice.
                finalize_active_batch(timeout=4.0)
            return True   # handled; the OS terminates the process afterwards
        return False      # Ctrl-C / Ctrl-Break → defer to the SIGINT handler

    try:
        _console_handler_ref = HANDLER_ROUTINE(_handler)
        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler_ref, True):
            _console_handler_ref = None
    except Exception:
        _console_handler_ref = None


def print_startup_banner() -> None:
    import psutil
    import platform
    import torch
    import onnxruntime as ort
    
    cfg = roop.globals.CFG
    if not cfg:
        return
        
    print("=" * 75)
    print("      ⚡ ROOP ULTIMATE - CORE INITIALIZATION GATEWAY ⚡")
    print("=" * 75)
    print(f"  [System Host] OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  [Environment] Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | ONNX Runtime: {ort.__version__}")
    
    # CPU Diagnostics
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    freq_str = f" @ {cpu_freq.current/1000:.2f}GHz" if cpu_freq else ""
    virtual_mem = psutil.virtual_memory()
    total_ram_gb = virtual_mem.total / (1024 ** 3)
    free_ram_gb = virtual_mem.available / (1024 ** 3)
    print(f"  [CPU Hardware] {physical_cores} Cores ({logical_cores} Threads){freq_str} | Total RAM: {total_ram_gb:.2f} GB (Available: {free_ram_gb:.2f} GB)")
    
    # GPU Diagnostics
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(roop.globals.cuda_device_id)
        try:
            free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info(roop.globals.cuda_device_id)
            total_vram_gb = total_vram_bytes / (1024 ** 3)
            free_vram_gb = free_vram_bytes / (1024 ** 3)
            vram_str = f"Total VRAM: {total_vram_gb:.2f} GB | Free VRAM: {free_vram_gb:.2f} GB"
        except Exception:
            vram_str = "VRAM: Detection Failed (driver context conflict)"
        print(f"  [GPU Hardware] Active CUDA Device: ID {roop.globals.cuda_device_id} - '{gpu_name}'")
        print(f"                 {vram_str}")
    else:
        print("  [GPU Hardware] Active CUDA Device: None (CPU Only)")
        
    backend = diagnostic_report(roop.globals.cuda_device_id,
                                getattr(cfg, 'provider', None))
    print(f"  [ONNX Backend] Available Providers: {backend['available_providers']}")
    print(f"  [ONNX Backend] AUTO validated chain: {backend['resolved']}")
    if backend['configured'] != 'auto':
        print(f"  [ONNX Backend] Requested override: {backend['configured']}")
    
    # Session Configuration Settings
    print("-" * 75)
    print(f"  [Active Configuration]")
    print(f"   - Max Processing Threads : {cfg.max_threads}")
    print(f"   - Memory Cap Limit (GB) : {cfg.memory_limit if cfg.memory_limit > 0 else 'Unlimited'}")
    print(f"   - Default Swap Model     : {cfg.swap_model}")
    print(f"   - Face Detection Grid    : {getattr(cfg, 'face_detector_size', '640')}px")
    print(f"   - Face Detector Threshold: {getattr(cfg, 'face_detector_threshold', 0.60):.2f}")
    print(f"   - Temp Folder Location   : {'System OS Temp' if cfg.use_os_temp_folder else 'Local Project Root'}")
    print("=" * 75)
    print("  Booting local FastAPI Swapping Gateway daemon thread...")
    print("=" * 75)


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings('config.yaml')
    roop.globals.cuda_device_id = roop.globals.startup_args.cuda_device_id
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    if roop.globals.startup_args.server_share:
        roop.globals.CFG.server_share = True
    print_startup_banner()
    main.run()
