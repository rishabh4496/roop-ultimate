import os
import yaml

# --- Make the TensorRT execution provider actually loadable on Windows ---
# onnxruntime advertises 'TensorrtExecutionProvider' as available even when its
# native runtime DLLs cannot be loaded. Loading onnxruntime_providers_tensorrt.dll
# requires BOTH of these to be resolvable at import time:
#   1) nvinfer_*.dll        -> shipped in the `tensorrt_libs` package folder
#   2) the CUDA/cuDNN runtime DLLs -> shipped inside torch's `lib` folder
# If either is missing, ORT fails with "LoadLibrary failed with error 126" and
# silently falls back to the CUDA provider (losing TensorRT acceleration).
#
# Since Python 3.8, Windows ignores PATH for dependent-DLL resolution and only
# searches directories registered via os.add_dll_directory(); the CUDA runtime
# additionally has to be *loaded* into the process, which importing torch does.
# settings is imported very early in every process (including spawned video
# workers), so doing this here fixes the silent CUDA fallback everywhere.
def _enable_tensorrt_runtime():
    dll_dirs = []
    try:
        import tensorrt
        trt_libs = os.path.join(os.path.dirname(os.path.dirname(tensorrt.__file__)), 'tensorrt_libs')
        if os.path.isdir(trt_libs):
            dll_dirs.append(trt_libs)
    except Exception:
        pass
    try:
        # Importing torch loads the CUDA/cuDNN runtime DLLs the TRT EP depends on.
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib):
            dll_dirs.append(torch_lib)
    except Exception:
        pass
    for d in dll_dirs:
        try:
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(d)
        except Exception:
            pass
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

_enable_tensorrt_runtime()


class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except:
            pass
        return value


    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except:
            data = None

        self.selected_theme = self.default_get(data, 'selected_theme', "Default")
        self.server_name = self.default_get(data, 'server_name', "")
        self.server_port = self.default_get(data, 'server_port', 0)
        self.server_share = self.default_get(data, 'server_share', False)
        self.output_image_format = self.default_get(data, 'output_image_format', 'png')
        self.output_video_format = self.default_get(data, 'output_video_format', 'mp4')
        self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx264')
        self.video_quality = self.default_get(data, 'video_quality', 14)
        self.clear_output = self.default_get(data, 'clear_output', True)
        # Dynamically scale threads to saturate GPU without OOM
        default_threads = 3
        try:
            self.provider = self.default_get(data, 'provider', 'cuda')
            if self.provider in ['cuda', 'tensorrt']:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    import psutil
                    cores = psutil.cpu_count(logical=False) or 4
                    # 1 thread per 1.5GB VRAM is safe for face swapping, bounded by CPU cores
                    default_threads = int(min(max(2, cores - 1), max(2, vram_gb / 1.5)))
        except Exception:
            pass

        saved_threads = self.default_get(data, 'max_threads', -1)
        # Auto-scale only when nothing is saved. A saved value — including 2 —
        # is a deliberate user choice and must stick across restarts.
        if saved_threads == -1:
            self.max_threads = default_threads
        else:
            self.max_threads = saved_threads

        # Prevent extreme CPU oversubscription by capping max_threads to logical CPU cores
        try:
            import psutil
            logical_cores = psutil.cpu_count(logical=True) or 4
            if self.max_threads > logical_cores:
                self.max_threads = logical_cores
        except Exception:
            pass

        self.auto_thread_selection = self.default_get(data, 'auto_thread_selection', True)
        self.benchmark_results = self.default_get(data, 'benchmark_results', {})
        
        self.memory_limit = self.default_get(data, 'memory_limit', 0)
        self.provider = self.default_get(data, 'provider', 'cuda')
        # TensorRT precision mode: 'fp32' | 'fp16' | 'mixed' (only used when provider == 'tensorrt')
        self.trt_precision = self.default_get(data, 'trt_precision', 'mixed')
        self.force_cpu = self.default_get(data, 'force_cpu', False)
        self.output_template = self.default_get(data, 'output_template', '{file}_{time}')
        # Faceset library folder: persistent, named .fsz facesets that survive
        # restarts so sources never need re-uploading. Blank = <app>/facesets.
        # Point it at a cloud-synced folder (OneDrive/Dropbox/Google Drive) to
        # sync your faceset library across devices.
        self.faceset_library_path = self.default_get(data, 'faceset_library_path', '')
        self.use_os_temp_folder = self.default_get(data, 'use_os_temp_folder', False)
        self.output_show_video = self.default_get(data, 'output_show_video', True)
        self.launch_browser = self.default_get(data, 'launch_browser', False)
        self.max_face_distance = self.default_get(data, 'max_face_distance', 0.75)
        # Faceswap session settings
        self.face_detection_mode = self.default_get(data, 'face_detection_mode', 'Selected face')
        # Face-detector input resolution: True = 640x640 (accurate, default),
        # False = 320x320 (~4x faster detection, may miss small/distant faces).
        self.default_det_size = self.default_get(data, 'default_det_size', True)
        self.face_detector_size = str(self.default_get(data, 'face_detector_size', '640' if self.default_det_size else '320'))
        self.face_detector_threshold = float(self.default_get(data, 'face_detector_threshold', 0.50))
        self.face_detector_nms = float(self.default_get(data, 'face_detector_nms', 0.3))
        self.sam2_model_size = self.default_get(data, 'sam2_model_size', 'tiny')
        self.track_identities = self.default_get(data, 'track_identities', True)
        self.num_swap_steps = self.default_get(data, 'num_swap_steps', 1)
        self.selected_enhancer = self.default_get(data, 'selected_enhancer', 'UltraMax')
        self.codeformer_fidelity = float(self.default_get(data, 'codeformer_fidelity', 0.55))
        self.subsample_upscale = self.default_get(data, 'subsample_upscale', '256px')
        self.upscale_after_swap = self.default_get(data, 'upscale_after_swap', False)
        self.upscale_model_after = self.default_get(data, 'upscale_model_after', 'fsr_x2')
        # Frame interpolation pass after the swap (and after any upscale):
        # 'off' | 'rife_2x' | 'rife_4x' | 'minterpolate_2x'
        self.interp_after_swap = self.default_get(data, 'interp_after_swap', 'off')
        self.blend_ratio = self.default_get(data, 'blend_ratio', 1.0)
        self.video_swapping_method = self.default_get(data, 'video_swapping_method', 'In-Memory processing')
        self.no_face_action = self.default_get(data, 'no_face_action', 'Use untouched original frame')
        self.vr_mode = self.default_get(data, 'vr_mode', False)
        self.autorotate_faces = self.default_get(data, 'autorotate_faces', True)
        self.skip_audio = self.default_get(data, 'skip_audio', False)
        self.keep_frames = self.default_get(data, 'keep_frames', False)
        self.wait_after_extraction = self.default_get(data, 'wait_after_extraction', False)
        self.output_method = self.default_get(data, 'output_method', 'File')
        self.mask_engine = self.default_get(data, 'mask_engine', 'RealityUX')
        # A second, independent occlusion engine. They compose as a union of
        # "not face", so this can only restore more of the original footage —
        # which is the answer to one engine not recognising the particular object
        # that came in front of the face. 'None' = one engine, as before.
        self.mask_engine_2 = self.default_get(data, 'mask_engine_2', 'None')
        self.mask_clip_text = self.default_get(data, 'mask_clip_text', 'cup,hands,hair,banana')
        # `sam2_model_size` and `track_identities` were also assigned in the
        # detection block above. Copy-paste duplicates, and not harmless: this
        # one ran LAST, so track_identities' real default was the False here
        # rather than the True up there, and editing the visible one changed
        # nothing. Both now live in the detection block only.
        self.show_mask_offsets = self.default_get(data, 'show_mask_offsets', False)
        self.restore_original_mouth = self.default_get(data, 'restore_original_mouth', False)
        self.mask_top = self.default_get(data, 'mask_top', 0.0)
        self.mask_bottom = self.default_get(data, 'mask_bottom', 0.0)
        self.mask_left = self.default_get(data, 'mask_left', 0.0)
        self.mask_right = self.default_get(data, 'mask_right', 0.0)
        self.face_mask_blend = self.default_get(data, 'face_mask_blend', 12.0)
        self.mouth_mask_blend = self.default_get(data, 'mouth_mask_blend', 10.0)
        self.mouth_top_scale = self.default_get(data, 'mouth_top_scale', 1.0)
        self.mouth_bottom_scale = self.default_get(data, 'mouth_bottom_scale', 1.0)
        self.mouth_left_scale = self.default_get(data, 'mouth_left_scale', 1.0)
        self.mouth_right_scale = self.default_get(data, 'mouth_right_scale', 1.0)
        # 3D source pose matching
        self.use_3d_recon = self.default_get(data, 'use_3d_recon', False)
        # Multi-angle source bank (Option 1)
        self.use_source_bank = self.default_get(data, 'use_source_bank', False)
        # Target frontalization (Option 2)
        self.use_frontalization = self.default_get(data, 'use_frontalization', False)
        self.frontalization_threshold = self.default_get(data, 'frontalization_threshold', 15.0)
        self.swap_model = self.default_get(data, 'swap_model', 'realswap')
        # One Euro temporal face stabilization (video)
        self.stabilize_face = self.default_get(data, 'stabilize_face', True)
        self.stabilize_method = self.default_get(data, 'stabilize_method', 'one_euro')
        self.stabilize_min_cutoff = self.default_get(data, 'stabilize_min_cutoff', 0.1)
        self.stabilize_beta = self.default_get(data, 'stabilize_beta', 0.1)
        self.stabilize_enhancer = self.default_get(data, 'stabilize_enhancer', True)
        self.stabilize_enhancer_strength = self.default_get(data, 'stabilize_enhancer_strength', 0.25)
        self.stabilize_mask = self.default_get(data, 'stabilize_mask', True)
        self.stabilize_mask_strength = self.default_get(data, 'stabilize_mask_strength', 0.5)
        # Skin-tone / lighting match of swapped crop → original: none|rct|lct|mkl
        self.color_transfer_mode = self.default_get(data, 'color_transfer_mode', 'lct')
        # Detection refinements
        self.refine_landmarks = self.default_get(data, 'refine_landmarks', True)
        # Swap-model face mask — only hififace/hyperswap emit one; the models that
        # do not are unaffected at any value.
        self.swap_model_mask_strength = self.default_get(data, 'swap_model_mask_strength', 0.0)
        # Jaw / chin reshape toward the source face shape
        self.jaw_reshape = self.default_get(data, 'jaw_reshape', False)
        self.jaw_reshape_strength = self.default_get(data, 'jaw_reshape_strength', 0.5)
        # Skin detail transfer strength (high-frequency texture from footage)
        self.detail_transfer_strength = self.default_get(data, 'detail_transfer_strength', 0.4)
        # Eye restore — the counterpart to restore_original_mouth
        self.restore_original_eyes = self.default_get(data, 'restore_original_eyes', False)
        self.eyes_blend_amount = self.default_get(data, 'eyes_blend_amount', 1.0)
        self.eyes_feather_blend = self.default_get(data, 'eyes_feather_blend', 25.0)
        self.eyes_size_factor = self.default_get(data, 'eyes_size_factor', 1.0)
        self.eyes_radius_x = self.default_get(data, 'eyes_radius_x', 1.0)
        self.eyes_radius_y = self.default_get(data, 'eyes_radius_y', 1.0)
        # Face Parser regions — which parsed parts count as the swap region
        self.parser_regions = self.default_get(data, 'parser_regions', ['skin', 'brows', 'eyes', 'nose', 'mouth'])
        self.parser_region_grow = self.default_get(data, 'parser_region_grow', {})
        # Enhancer alignment + a second colour pass after restoration
        self.enhancer_align = self.default_get(data, 'enhancer_align', False)
        self.color_match_after_enhance = self.default_get(data, 'color_match_after_enhance', True)
        # Lip-sync (MuseTalk) — see roop/globals.py. lipsync_audio_path is a
        # per-job temp upload reference, not a durable default.
        self.lipsync_enabled = self.default_get(data, 'lipsync_enabled', False)
        self.lipsync_audio_source = self.default_get(data, 'lipsync_audio_source', 'original')
        # DeepFaceLab merger post-ops — see roop/procmgr_merger.py. All neutral
        # by default; each is a bit-identical no-op at 0.
        self.merger_hist_match = self.default_get(data, 'merger_hist_match', 0.4)
        self.merger_sharpen = self.default_get(data, 'merger_sharpen', 0.35)
        self.merger_motion_blur = self.default_get(data, 'merger_motion_blur', 0.0)
        self.merger_grain_match = self.default_get(data, 'merger_grain_match', 0.45)
        self.merger_degrade = self.default_get(data, 'merger_degrade', 0.0)
        # 1.0 reproduces exactly what Enhance_UltraMax used to apply to its own
        # output, so moving the filter out here is behaviour-preserving for a
        # config already on UltraMax, and gives every other enhancer the same
        # look for one LAB round trip.
        self.merger_clarity = self.default_get(data, 'merger_clarity', 1.0)
        # Grow/shrink the pasted face about its own centre (DFL output_face_scale)
        self.output_face_scale = self.default_get(data, 'output_face_scale', 0.0)
        # Expression restorer (LivePortrait) — see roop.globals
        self.expression_restore_strength = self.default_get(data, 'expression_restore_strength', 0.0)
        self.expression_restore_region = self.default_get(data, 'expression_restore_region', 'all')
        self.rescue_small_faces = self.default_get(data, 'rescue_small_faces', True)
        self.detector_engine = self.default_get(data, 'detector_engine', 'retinaface_r50')
        # Temporal detection pre-pass (video anti-flicker): tracked detection with
        # gap-fill so the swap can't blink out on missed detections.
        self.temporal_detection = self.default_get(data, 'temporal_detection', True)
        # Advanced perf knobs (env-backed; 'auto' = leave launcher/auto-tune
        # behaviour untouched). Applied to os.environ at startup by run.py, so
        # changes take effect after an app restart.
        self.perf_trt_pool = self.default_get(data, 'perf_trt_pool', 'auto')
        # NVDEC GPU video decode (ffmpeg -hwaccel cuda pipe). auto = enabled
        # behind a per-file probe with automatic cv2 fallback; off disables.
        self.perf_nvdec = self.default_get(data, 'perf_nvdec', 'auto')
        self.perf_detmask_pool = self.default_get(data, 'perf_detmask_pool', 'auto')
        # Instances of the standalone detector, separate from the detect/mask
        # pool because a hybrid engine (retinaface/yoloface/yunet) brings its own
        # detector and only borrows buffalo_l's aux models — widening the detmask
        # pool alone parallelises the aux models and leaves the detector
        # single-file. 'auto' = follow the detmask pool, which is what
        # session_pool.detector_pool_size() already does.
        #
        # It got a setting of its own because the benchmark can now measure the
        # two separately and they do not agree: on an RTX 4070 the detector
        # scaled to 4 instances while the recognition/landmark models plateaued
        # at 2, and without this knob the only way to act on that was an env var.
        self.perf_detector_pool = self.default_get(data, 'perf_detector_pool', 'auto')
        # Expression restorer contexts. 'auto' is VRAM-tiered (0 below 11.5GB,
        # else 2). Worth raising to 3 only when the STAGE TIMING breakdown shows
        # 'expression' needing more concurrent threads than the pool has slots.
        self.perf_expr_pool = self.default_get(data, 'perf_expr_pool', 'auto')
        self.perf_encoder_preset = self.default_get(data, 'perf_encoder_preset', 'auto')
        self.perf_profile = self.default_get(data, 'perf_profile', 'auto')       # auto|on|off
        self.perf_batch_swap = self.default_get(data, 'perf_batch_swap', 'auto')  # auto|on|off

        # ── Identity & tracking behaviour ────────────────────────────────────
        # Features that shipped reachable only through a ROOP_* environment
        # variable, which means reachable only by someone editing a launcher.
        # Each maps to its flag in run.py::_apply_perf_env and each defaults to
        # 'auto' == "leave the environment alone", so the shipped behaviour is
        # unchanged and an explicit choice still wins. They are read at import
        # time, hence the "restart to apply" grouping in the panel.
        #
        # `recognizer` is the one that is genuinely a MODEL choice rather than a
        # switch: AdaFace is a second recognition model with its own distance
        # scale (roop/recognizer_adaface.py rescales every identity constant to
        # match), and it was previously unreachable from the UI entirely.
        self.recognizer = self.default_get(data, 'recognizer', 'default')     # default|adaface
        # Demarcation between two swapped faces that touch — the phase-3 work in
        # roop/face_overlap.py. Ships ON; exposed so it can be turned off.
        self.face_demarcate = self.default_get(data, 'face_demarcate', 'auto')  # auto|on|off
        # Chains track fragments back together across a cut or a brief occlusion.
        self.track_stitch = self.default_get(data, 'track_stitch', 'auto')      # auto|on|off
        # Re-detects the swapped result and undoes the swap if the face moved —
        # catches a frontal face painted onto a head pointing away.
        self.verify_swap = self.default_get(data, 'verify_swap', 'auto')        # auto|on|off
        # Re-measures a heavily rolled/inverted face on an uprighted frame
        # before anything reads its keypoints or embedding.
        self.upright_remeasure = self.default_get(data, 'upright_remeasure', 'auto')  # auto|on|off
        # Scheduling priority class while rendering. Only the names
        # keep_awake._PRIORITY_CLASSES knows are valid — it silently falls back
        # to 'high' for anything else. EcoQoS power throttling is opted out of
        # unconditionally there and is not a choice.
        self.process_priority = self.default_get(data, 'process_priority', 'auto')  # auto|high|above_normal|normal

        # ── Theme ────────────────────────────────────────────────────────────
        # User-authored themes, each a small recipe the UI expands into the full
        # CSS variable set (see react-ui/src/themeVars.js). Stored here rather
        # than in browser localStorage so they survive the Pinokio Run<->Dev
        # reload and travel with the config like every other preference.
        self.custom_themes = self.default_get(data, 'custom_themes', [])
        # When set, `selected_theme` is ignored and the theme follows the OS
        # light/dark signal, picking from this pair.
        self.theme_follow_system = self.default_get(data, 'theme_follow_system', False)
        self.theme_dark = self.default_get(data, 'theme_dark', 'Default')
        self.theme_light = self.default_get(data, 'theme_light', 'Glass Light')





    def save(self):
        data = {
            'selected_theme': self.selected_theme,
            'custom_themes': self.custom_themes,
            'theme_follow_system': self.theme_follow_system,
            'theme_dark': self.theme_dark,
            'theme_light': self.theme_light,
            'server_name': self.server_name,
            'server_port': self.server_port,
            'server_share': self.server_share,
            'output_image_format' : self.output_image_format,
            'output_video_format' : self.output_video_format,
            'output_video_codec' : self.output_video_codec,
            'video_quality' : self.video_quality,
            'clear_output' : self.clear_output,
            'max_threads' : self.max_threads,
            'memory_limit' : self.memory_limit,
            'provider' : self.provider,
            'trt_precision' : self.trt_precision,
            'force_cpu' : self.force_cpu,
            'output_template' : self.output_template,
            'faceset_library_path' : self.faceset_library_path,
            'use_os_temp_folder' : self.use_os_temp_folder,
            'output_show_video' : self.output_show_video,
            'launch_browser': self.launch_browser,
            'max_face_distance': self.max_face_distance,
            # Faceswap session settings
            'face_detection_mode': self.face_detection_mode,
            'default_det_size': self.default_det_size,
            'face_detector_size': self.face_detector_size,
            'face_detector_threshold': self.face_detector_threshold,
            'num_swap_steps': self.num_swap_steps,
            'selected_enhancer': self.selected_enhancer,
            'codeformer_fidelity': self.codeformer_fidelity,
            'subsample_upscale': self.subsample_upscale,
            'upscale_after_swap': self.upscale_after_swap,
            'upscale_model_after': self.upscale_model_after,
            'interp_after_swap': self.interp_after_swap,
            'blend_ratio': self.blend_ratio,
            'video_swapping_method': self.video_swapping_method,
            'no_face_action': self.no_face_action,
            'vr_mode': self.vr_mode,
            'autorotate_faces': self.autorotate_faces,
            'skip_audio': self.skip_audio,
            'keep_frames': self.keep_frames,
            'wait_after_extraction': self.wait_after_extraction,
            'output_method': self.output_method,
            'mask_engine': self.mask_engine,
            'mask_engine_2': self.mask_engine_2,
            'mask_clip_text': self.mask_clip_text,
            'sam2_model_size': self.sam2_model_size,
            'track_identities': self.track_identities,
            'show_mask_offsets': self.show_mask_offsets,
            'restore_original_mouth': self.restore_original_mouth,
            'mask_top': self.mask_top,
            'mask_bottom': self.mask_bottom,
            'mask_left': self.mask_left,
            'mask_right': self.mask_right,
            'face_mask_blend': self.face_mask_blend,
            'mouth_mask_blend': self.mouth_mask_blend,
            'mouth_top_scale': self.mouth_top_scale,
            'mouth_bottom_scale': self.mouth_bottom_scale,
            'mouth_left_scale': self.mouth_left_scale,
            'mouth_right_scale': self.mouth_right_scale,
            # 3D source pose matching
            'use_3d_recon': self.use_3d_recon,
            # Multi-angle source bank
            'use_source_bank': self.use_source_bank,
            # Target frontalization
            'use_frontalization': self.use_frontalization,
            'frontalization_threshold': self.frontalization_threshold,
            # Swap model
            'swap_model': self.swap_model,
            # One Euro temporal face stabilization
            'stabilize_face': self.stabilize_face,
            'stabilize_method': self.stabilize_method,
            'stabilize_min_cutoff': self.stabilize_min_cutoff,
            'stabilize_beta': self.stabilize_beta,
            'stabilize_enhancer': self.stabilize_enhancer,
            'stabilize_enhancer_strength': self.stabilize_enhancer_strength,
            'stabilize_mask': self.stabilize_mask,
            'stabilize_mask_strength': self.stabilize_mask_strength,
            'color_transfer_mode': self.color_transfer_mode,
            'refine_landmarks': self.refine_landmarks,
            'swap_model_mask_strength': self.swap_model_mask_strength,
            'jaw_reshape': self.jaw_reshape,
            'jaw_reshape_strength': self.jaw_reshape_strength,
            'detail_transfer_strength': self.detail_transfer_strength,
            'restore_original_eyes': self.restore_original_eyes,
            'eyes_blend_amount': self.eyes_blend_amount,
            'eyes_feather_blend': self.eyes_feather_blend,
            'eyes_size_factor': self.eyes_size_factor,
            'eyes_radius_x': self.eyes_radius_x,
            'eyes_radius_y': self.eyes_radius_y,
            'parser_regions': self.parser_regions,
            'parser_region_grow': self.parser_region_grow,
            'enhancer_align': self.enhancer_align,
            'color_match_after_enhance': self.color_match_after_enhance,
            'lipsync_enabled': self.lipsync_enabled,
            'lipsync_audio_source': self.lipsync_audio_source,
            'merger_hist_match': self.merger_hist_match,
            'merger_sharpen': self.merger_sharpen,
            'merger_motion_blur': self.merger_motion_blur,
            'merger_grain_match': self.merger_grain_match,
            'merger_degrade': self.merger_degrade,
            'merger_clarity': self.merger_clarity,
            'output_face_scale': self.output_face_scale,
            'expression_restore_strength': self.expression_restore_strength,
            'expression_restore_region': self.expression_restore_region,
            'rescue_small_faces': self.rescue_small_faces,
            'detector_engine': self.detector_engine,
            'face_detector_nms': self.face_detector_nms,
            'temporal_detection': self.temporal_detection,
            'perf_trt_pool': self.perf_trt_pool,
            'perf_nvdec': self.perf_nvdec,
            'perf_detmask_pool': self.perf_detmask_pool,
            'perf_detector_pool': self.perf_detector_pool,
            'perf_expr_pool': self.perf_expr_pool,
            'perf_encoder_preset': self.perf_encoder_preset,
            'perf_profile': self.perf_profile,
            'perf_batch_swap': self.perf_batch_swap,
            'recognizer': self.recognizer,
            'face_demarcate': self.face_demarcate,
            'track_stitch': self.track_stitch,
            'verify_swap': self.verify_swap,
            'upright_remeasure': self.upright_remeasure,
            'process_priority': self.process_priority,
            'auto_thread_selection': getattr(self, 'auto_thread_selection', True),
            'benchmark_results': getattr(self, 'benchmark_results', {}),
        }
        # Atomic write: dump to a temp file and replace. Writing config.yaml in
        # place means a crash mid-write truncates it, and load()'s fallback then
        # silently resets every setting to defaults.
        tmp_file = self.config_file + '.tmp'
        with open(tmp_file, 'w') as f:
            yaml.dump(data, f)
        os.replace(tmp_file, self.config_file)

    def resolve_threads(self, mode='standard') -> int:
        if not getattr(self, 'auto_thread_selection', True):
            return getattr(self, 'max_threads', 4)

        results = getattr(self, 'benchmark_results', {}) or {}
        if isinstance(results, dict) and results.get('best_threads'):
            mode_threads = results.get('best_threads', {}).get(mode)
            if mode_threads:
                return int(mode_threads)

        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                import psutil
                cores = psutil.cpu_count(logical=False) or 4
                logical = psutil.cpu_count(logical=True) or cores

                # TensorRT context sharing (trt_context_memory_sharing_enable=True)
                # allows worker threads to execute concurrently without duplicating weight memory.
                if vram_gb >= 15.5:
                    max_cap = min(logical, 16)
                elif vram_gb >= 11.5:
                    max_cap = min(logical, 16)
                elif vram_gb >= 7.5:
                    max_cap = min(logical, 12)
                else:
                    max_cap = min(cores, 8)

                if mode == 'heavy':
                    return int(min(max_cap, max(4, int(vram_gb / 1.5))))
                elif mode == 'enhanced':
                    return int(min(max_cap, max(6, int(vram_gb / 1.0))))
                else:
                    return int(min(max_cap, max(8, int(vram_gb / 0.75))))
        except Exception:
            pass
        return getattr(self, 'max_threads', 8)
