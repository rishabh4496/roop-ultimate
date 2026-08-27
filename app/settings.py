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


# Bumped whenever the `default_threads` formula below changes.
#
# WHY A VERSION EXISTS AT ALL. `save()` writes `max_threads` on every settings
# save, so the moment the app persists a value it derived, that value becomes
# indistinguishable from one the user typed -- and `_hw_get` only re-derives
# when the GPU changes, not when the RULE does. So an install keeps whatever
# formula was in force the first time it saved, forever.
#
# That is not hypothetical: it is why the RTX 3060 6GB this rule was rewritten
# FOR (see the `default_threads` comment, commit 0eda23b) was still running
# max_threads 4 on 2026-08-25, a full tier below the measured knee of 8, with
# the new rule in the source and unable to reach it. The card the fix was
# written for was the card the fix could not get to.
#
# Rule 1: min(cores - 1, vram_gb / 1.5)   -- the "workers cost VRAM" premise
# Rule 2: min(max(2, cores - 1), knee)    -- left one CPU core unused
# Rule 3: reach the measured knee when the CPU has that many physical cores;
#         retain one core of headroom on smaller CPUs.
_THREAD_RULE = 3


def detect_hardware():
    """What every performance default on this machine is derived from.

    Returns {'gpu', 'vram_gb', 'ram_gb'}, with empty/0 values when it cannot be
    determined — an unknown machine must fall through to the safest defaults,
    never to another machine's.
    """
    hw = {'gpu': '', 'vram_gb': 0.0, 'ram_gb': 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw['gpu'] = str(props.name)
            hw['vram_gb'] = round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        import psutil
        hw['ram_gb'] = round(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        pass
    return hw


def hardware_signature(hw=None):
    """A stable id for 'the machine the perf numbers were tuned on'.

    GPU model, VRAM and system RAM — the three inputs to every auto tier here
    and in session_pool. Deliberately NOT the driver version or CPU: those
    change without changing a single pool size, and a signature that churns
    would reset the user's settings for no reason.

    The CPU exclusion is narrower than it reads. It is true of the POOL sizes,
    which is what this signature guards, but the thread default has been bound
    to the core count since 0eda23b — `min(max(2, cores - 1), knee)`. Rather
    than reusing that former formula, the current policy reaches the measured
    GPU knee when enough physical cores exist, otherwise retaining one core of
    host headroom. The thread count carries its own basis stamp; see
    `_THREAD_RULE`.
    """
    hw = hw or detect_hardware()
    if not hw.get('gpu') and not hw.get('ram_gb'):
        return ''
    return f"{hw.get('gpu', '')}|{hw.get('vram_gb', 0.0)}|{hw.get('ram_gb', 0)}"


class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def __setattr__(self, name, value):
        """Record that `max_threads` was set by a person rather than derived.

        Every write path — the React settings panel (`POST /api/settings`), the
        Gradio settings tab, anything holding a CFG — assigns the attribute, so
        catching it here catches all of them without each one having to
        remember. `object.__setattr__` is used, so the value still lands in
        `__dict__` and `GET /api/settings` (which serialises `__dict__`) is
        unaffected.

        Only a CHANGE counts. The settings panel POSTs the whole object back on
        any unrelated save, so an untouched thread slider arrives here on nearly
        every write; treating that as a choice would re-pin the derived value
        and recreate the bug `_THREAD_RULE` exists to fix.
        """
        if name == 'max_threads' and not self.__dict__.get('_loading', False):
            try:
                new = int(value)
            except (TypeError, ValueError):
                new = None
            if new is not None and new != self.__dict__.get('max_threads'):
                object.__setattr__(self, '_threads_auto', False)
                object.__setattr__(self, '_threads_basis', 'user')
        object.__setattr__(self, name, value)

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except:
            pass
        return value

    def _hw_get(self, data, name, default):
        """default_get for values that are FACTS ABOUT THE GPU, not preferences.

        A pool size, a thread count or a benchmark result describes the card it
        was chosen on. Carrying one to a different card is how a value that is
        merely optimal on 12 GB becomes fatal on 6 GB: the RTX 4070 tier is
        detmask pool 2-4 and ~10 threads, and the same numbers on an RTX 3060
        6 GB drive TensorRT context thrashing at 2 fps (see
        session_pool._advisory_pool_size). So when the hardware signature
        changes, these revert to 'auto' and the new machine re-derives them.

        Preferences are NOT touched — theme, output template, swap model, mask
        engine, enhancer. Those are the user's choices and travel with them.
        """
        if getattr(self, '_hardware_changed', False):
            return default
        return self.default_get(data, name, default)


    def load(self):
        # load() assigns max_threads itself; __setattr__ must not read those
        # assignments as a user choice.
        object.__setattr__(self, '_loading', True)
        try:
            self._load()
        finally:
            object.__setattr__(self, '_loading', False)

        # Stamp a one-off thread migration back to disk straight away, so the
        # notice it printed is true: without this the decision is recomputed
        # from the same unstamped file on every launch and the message repeats
        # forever. Guarded on the file already existing because
        # `/api/settings/defaults` builds a throwaway Settings pointed at a path
        # that deliberately does not exist, and must not bring one into being.
        if getattr(self, '_threads_migrated', False):
            object.__setattr__(self, '_threads_migrated', False)
            try:
                if os.path.isfile(self.config_file):
                    self.save()
            except Exception:
                pass        # a read-only config is not worth failing startup for

    def _load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except:
            data = None

        # ── Hardware portability ─────────────────────────────────────────────
        # config.yaml is per-install and gitignored, but it does get copied, and
        # the same user runs this on more than one machine. Anything derived from
        # the GPU is therefore stamped with the GPU it was derived on, and reset
        # when that changes. Without this the FIRST render on the new machine is
        # the thing that discovers the mismatch, and on a small card it discovers
        # it as a thrash that looks like a hang.
        self.hardware = detect_hardware()
        self.hardware_signature = hardware_signature(self.hardware)
        _saved_sig = self.default_get(data, 'hardware_signature', '')
        self._hardware_changed = bool(_saved_sig) and bool(self.hardware_signature) \
            and _saved_sig != self.hardware_signature
        if self._hardware_changed:
            print(f"[Hardware] this config was tuned on '{_saved_sig}' but this "
                  f"machine is '{self.hardware_signature}'. Re-deriving the "
                  f"hardware-dependent settings (thread count, pool sizes, saved "
                  f"benchmark results) from THIS GPU; your model and output "
                  f"choices are untouched.")

        self.selected_theme = self.default_get(data, 'selected_theme', "Default")
        self.server_name = self.default_get(data, 'server_name', "")
        self.server_port = self.default_get(data, 'server_port', 0)
        self.server_share = self.default_get(data, 'server_share', False)
        self.output_image_format = self.default_get(data, 'output_image_format', 'png')
        self.output_video_format = self.default_get(data, 'output_video_format', 'mp4')
        self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx264')
        # NVENC/QSV/AMF are vendor-specific. Only reset when the new machine
        # cannot possibly have the encoder — an NVIDIA->NVIDIA move keeps it.
        if self._hardware_changed and any(t in str(self.output_video_codec)
                                          for t in ('nvenc', 'qsv', 'amf')):
            _vendor_ok = (('nvenc' in self.output_video_codec
                           and 'nvidia' in self.hardware.get('gpu', '').lower())
                          or ('nvenc' not in self.output_video_codec))
            if not _vendor_ok:
                print(f"[Hardware] '{self.output_video_codec}' is not available on "
                      f"'{self.hardware.get('gpu') or 'this machine'}' — falling "
                      f"back to libx264.")
                self.output_video_codec = 'libx264'
        self.video_quality = self.default_get(data, 'video_quality', 14)
        self.clear_output = self.default_get(data, 'clear_output', True)
        # Dynamically scale threads to saturate GPU without OOM
        default_threads = 3
        threads_basis = f"v{_THREAD_RULE}|unknown"
        try:
            self.provider = self.default_get(data, 'provider', 'cuda')
            if self.provider in ['cuda', 'tensorrt']:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    import psutil
                    cores = psutil.cpu_count(logical=False) or 4
                    # THREAD COUNT DOES NOT COST VRAM. This used to read
                    # `vram_gb / 1.5` -- one worker per 1.5GB -- and the premise
                    # is false: the models are held in per-model SessionPools
                    # whose size is set by session_pool, not by the number of
                    # workers, so a worker adds only its own frame buffers.
                    # Measured 2026-08-25 under the <7GB policy (pools 0/0),
                    # own VRAM across a whole render:
                    #
                    #     threads    4      6      8     10     12
                    #     VRAM    2317   2339   2374   2329   2369  MB
                    #     fps    14.4   17.5   18.1   17.8   17.5
                    #
                    # Flat to within noise, while the old rule derived FOUR
                    # threads on a 6GB card -- 20% below the knee at 8 -- and
                    # seven on this 12GB/24-core one, where 10 measures 22.2 fps
                    # against 8's 20.1. It was guarding a cost that is not there.
                    #
                    # So: bound by cores, capped at the measured knee for the
                    # tier. Below 7GB the pools are off (session_pool
                    # _auto_pool_defaults), which is where the knee sits at 8;
                    # above it the 12GB card measured 10, with 14 buying nothing.
                    #
                    # An 8-core / 6GB laptop used to get 7 workers because the
                    # generic formula always reserved one physical core. That
                    # leaves a measured worker unused (18.1 fps at 8 versus
                    # 17.5 at 6; VRAM remains ~2.3 GB in either case). Use the
                    # entire CPU only when it reaches the already-measured GPU
                    # knee. A smaller CPU still keeps one core for decoding,
                    # encoding and UI responsiveness.
                    knee = 8 if vram_gb < 7 else 10
                    usable_cores = cores if cores >= knee else max(2, cores - 1)
                    default_threads = int(min(usable_cores, knee))
                    threads_basis = f"v{_THREAD_RULE}|{cores}|{knee}"
        except Exception:
            pass

        # _hw_get: a saved thread count is a deliberate choice ON THAT CARD.
        # -1 on a new card means "auto-scale below", which is what we want.
        saved_threads = self._hw_get(data, 'max_threads', -1)
        saved_auto = self.default_get(data, '_threads_auto', None)
        saved_basis = self.default_get(data, '_threads_basis', '')
        self._threads_auto = True
        self._threads_basis = threads_basis

        if saved_threads == -1:
            # Nothing saved (or a new card): derive.
            self.max_threads = default_threads
        elif saved_auto is False:
            # The user set this number. It sticks — including a number lower
            # than the derived one, which is the whole point of recording who
            # chose it. See __setattr__ for how that is detected.
            self.max_threads = saved_threads
            self._threads_auto = False
            self._threads_basis = saved_basis or 'user'
        elif saved_auto is True and saved_basis == threads_basis:
            # App-derived under the rule and hardware still in force.
            self.max_threads = saved_threads
        elif saved_auto is True:
            self.max_threads = default_threads
            if saved_threads != default_threads:
                self._threads_migrated = True
                print(f"[Threads] max_threads {saved_threads} -> {default_threads}: "
                      f"that value was derived by this app under an older rule or "
                      f"core count ('{saved_basis or 'unknown'}' -> '{threads_basis}'), "
                      f"not chosen by you. Set it in Settings to pin it.")
        elif saved_threads < default_threads:
            # LEGACY config, written before provenance was recorded, carrying a
            # value BELOW what this machine now derives. Every such value was
            # written by save() echoing a derived default, and leaving it is the
            # bug this stamp exists to fix. Deliberately one-directional: a
            # saved value ABOVE the derived one is left alone, because raising
            # a number costs nothing measurable (thread count does not cost
            # VRAM — see above) while lowering one someone raised on purpose
            # would be exactly the silent downgrade this project keeps finding.
            self.max_threads = default_threads
            self._threads_migrated = True
            print(f"[Threads] max_threads {saved_threads} -> {default_threads}: "
                  f"the saved value predates this machine's measured thread knee "
                  f"and had no record of being chosen by you. Set it in Settings "
                  f"to pin any value you prefer; this migration runs once.")
        elif saved_threads == default_threads:
            # LEGACY and already exactly what this machine derives. Nothing to
            # correct — but the branch below would file it as a user's choice
            # and pin it, which is how the NEXT rule change would fail to reach
            # this machine. That is the bug this whole stamp exists to fix, so
            # a value indistinguishable from the derived one stays derived.
            # Costs nothing if it really was typed: re-deriving reproduces it.
            self.max_threads = saved_threads
        else:
            # LEGACY and ABOVE the derived value: somebody raised it on purpose.
            self.max_threads = saved_threads
            self._threads_auto = False
            self._threads_basis = 'user'

        # Prevent extreme CPU oversubscription by capping max_threads to logical CPU cores
        try:
            import psutil
            logical_cores = psutil.cpu_count(logical=True) or 4
            if self.max_threads > logical_cores:
                print(f"[Threads] max_threads {self.max_threads} -> {logical_cores}: "
                      f"capped to this machine's logical core count. This is one of "
                      f"the three ways a raised Max Threads can fail to take effect "
                      f"(see tests/diag_device.py).")
                self.max_threads = logical_cores
        except Exception:
            pass

        self.auto_thread_selection = self._hw_get(data, 'auto_thread_selection', True)
        # best_threads in here was measured on the OTHER machine; resolve_threads
        # prefers it over the VRAM tier, so it has to go with the rest.
        self.benchmark_results = self._hw_get(data, 'benchmark_results', {})
        
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
        self.perf_trt_pool = self._hw_get(data, 'perf_trt_pool', 'auto')
        # NVDEC GPU video decode (ffmpeg -hwaccel cuda pipe). auto = enabled
        # behind a per-file probe with automatic cv2 fallback; off disables.
        self.perf_nvdec = self.default_get(data, 'perf_nvdec', 'auto')
        # LEAVE THIS ON 'auto' UNLESS A COUNTERBALANCED A/B SAYS OTHERWISE.
        # It reads like a 'more is faster' knob and is not one. Measured
        # 2026-08-25 on an RTX 4070, whole render, 10 threads, GPEN 256 Pro,
        # identical 401 faces enhanced in every arm:
        #
        #     trt/detmask   2/2 (auto)  21.71 fps
        #                   2/4         18.44 fps   -15%
        #                   4/4         17.69 fps   -19%
        #                   6/6          5.90 fps   -73%
        #
        # A config carrying an explicit '4' cost 15% for as long as it was
        # set. Above the auto default session_pool now prints an advisory
        # (it still honours the value); see _advisory_pool_size for why the
        # failure mode is thrashing rather than an OOM or a hang.
        self.perf_detmask_pool = self._hw_get(data, 'perf_detmask_pool', 'auto')
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
        self.perf_detector_pool = self._hw_get(data, 'perf_detector_pool', 'auto')
        # Expression restorer contexts. 'auto' is VRAM-tiered (0 below 11.5GB,
        # else 2). Worth raising to 3 only when the STAGE TIMING breakdown shows
        # 'expression' needing more concurrent threads than the pool has slots.
        self.perf_expr_pool = self._hw_get(data, 'perf_expr_pool', 'auto')
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
            # Stamped so the next load can tell whether these numbers were
            # derived on THIS machine. See _hw_get.
            'hardware_signature': self.hardware_signature,
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
            # Provenance for the line above: did the app derive this number, and
            # under which rule + hardware? Without it a derived default is
            # indistinguishable from a user's choice on the next load, and an
            # improved rule can never reach an existing install. See _THREAD_RULE.
            '_threads_auto': getattr(self, '_threads_auto', True),
            '_threads_basis': getattr(self, '_threads_basis', ''),
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
            measured = results.get('settings_measured') or {}
            if not measured or self._benchmark_matches_settings(measured):
                mode_threads = results.get('best_threads', {}).get(mode)
                if mode_threads:
                    return int(mode_threads)
            else:
                print("[Auto Thread Selection] Ignoring benchmark thread result: "
                      "measured model/provider settings do not match this run.",
                      flush=True)

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

    def _benchmark_matches_settings(self, measured) -> bool:
        """Return whether a benchmark was measured for the active pipeline.

        Thread knees are workload-specific. Reusing a result measured with a
        different enhancer (for example GPEN 256 Pro for an UltraMax render)
        creates unstable or unnecessarily low throughput, especially in the
        heavy path. Older hand-written test fixtures without provenance remain
        accepted by ``resolve_threads``; real reports contain this map.
        """
        fields = (
            ('provider', 'provider'),
            ('swap_model', 'swap_model'),
            ('enhancer', 'selected_enhancer'),
            ('mask_engine', 'mask_engine'),
            ('detector_engine', 'detector_engine'),
            ('subsample_upscale', 'subsample_upscale'),
            ('perf_trt_pool', 'perf_trt_pool'),
            ('perf_detmask_pool', 'perf_detmask_pool'),
            ('perf_detector_pool', 'perf_detector_pool'),
            ('perf_batch_swap', 'perf_batch_swap'),
        )
        for measured_key, setting_key in fields:
            measured_value = measured.get(measured_key)
            current_value = getattr(self, setting_key, None)
            if measured_value not in (None, '') and current_value not in (None, ''):
                if str(measured_value).lower() != str(current_value).lower():
                    return False
        return True
