"""What a frame actually costs on THIS machine, measured on the pipeline's own models.

The point of this file is to answer, per device, the questions the perf settings
ask and nobody can answer from a spec sheet: how many worker threads, how wide
each TensorRT pool, TensorRT or CUDA, NVDEC or cv2, which encoder preset.

It replaces a benchmark that answered none of them. That one ran

    a = torch.randn((T, 3, 512, 512), fp16); torch.matmul(a, a.transpose(-1, -2))

in ONE thread, called `T` "threads", and reported `T / elapsed` as "FPS". `T` was
a batch dimension, so the number it maximised was batched-matmul efficiency —
which is why it read 18,698 "FPS" on an RTX 4070 whose real swap throughput is
single-digit fps, and why it rose monotonically to whatever the largest candidate
was. Nothing in it touched onnxruntime, TensorRT, a pool, a worker thread, or any
model this app runs, and its VRAM guard read `torch.cuda.max_memory_allocated()`,
which cannot see an onnxruntime allocation at all and therefore reported ~0.06 GB
while TensorRT held gigabytes. Its output was nevertheless wired straight into
`Settings.resolve_threads`, so it was setting `execution_threads` for real runs.

What this one measures instead:

  * the REAL models the current settings select — the selected swapper (at its
    real tile count for the selected pixel-boost size), the selected enhancer,
    the selected mask engine(s), the selected detector plus the aux models that
    run per face, the expression restorer;
  * built through the app's OWN provider rules, including the two places the app
    deliberately forces FP32 under TensorRT (swapper, GPEN) and the places it
    removes the TensorRT EP entirely (SAM masks, frame upscaler). Timing a model
    under a configuration the app never runs is flattering and useless;
  * concurrency the way the pipeline actually gets it — N independent sessions
    leased one-per-thread (`SessionPool`), which is the ONLY thing that makes
    TensorRT concurrent, against the global lock that serialises every stage
    that has no pool;
  * VRAM through `torch.cuda.mem_get_info`, which sees the whole device;
  * and a composite frame — detect, swap x tiles, enhance, mask, plus the real
    CPU-side align/paste — swept over thread counts, because the thread count
    that wins is the one where CPU work of one frame overlaps GPU work of
    another, and neither half alone can tell you where that is.

Two honest limits, stated here so they are not discovered later:

  * inputs are synthetic tensors of the models' declared shapes. Convolutional
    cost is data-independent, so per-call GPU time is right; what synthetic input
    cannot reproduce is detector POST-processing, which scales with how many
    faces are found. The detect stage is therefore modelled as detector net +
    `faces_per_frame` x (recognition + landmarks), which is where its GPU time
    goes, and the decode/NMS remainder is part of the measured CPU term.
  * the composite is a frame's WORK, not a full render: it excludes the tracking
    pre-pass, colour transfer, stabilisation and the writer. It is a thread-count
    experiment, and the thread count is what it is used for. Absolute fps from it
    will be higher than a real render's.

Standalone (prints the same report the UI renders, off the same config.yaml —
see `bootstrap_globals`, without which this path silently measures the DEFAULT
models instead of the selected ones):

    env/Scripts/python.exe -m roop.bench --profile full
"""

import os
import platform
import threading
import time
from queue import Queue

import numpy as np
import onnxruntime

import roop.globals


# ── profiles ─────────────────────────────────────────────────────────────────
# Each measurement is time-boxed rather than iteration-boxed: a 2048px enhancer
# and a 128px swapper differ by 40x per call, and a fixed iteration count either
# takes minutes on one or measures noise on the other.
PROFILES = {
    # Serial stage costs at the CURRENT pool sizes, then the thread sweep.
    # Answers "how many threads" and nothing else, and leaves the pools alone.
    'quick': {
        'measure_sec': 1.0,
        'warm_sec': 0.35,
        'reps': 1,
        'pool_levels': (1,),
        'sweep_pools': False,
        'provider_ab': False,
        'io': False,
        'batch_swap': False,
        'tile_upscale': False,
        'est_sec': 260,
    },
    # Everything: pool sweeps per stage plus an end-to-end check on the result,
    # TensorRT vs CUDA per stage, batched swap, encode/decode. This is the one
    # whose answers are worth saving.
    #
    # The estimate is dominated by SESSION BUILDS, not by measurements: loading a
    # cached TensorRT engine took 8-12 s per instance on the reference machine,
    # against 1.5 s to measure it. That is why the pool sweep stops as soon as a
    # level fails to improve, and why the levels are powers of two rather than
    # every integer.
    'full': {
        'measure_sec': 1.5,
        'warm_sec': 0.5,
        'reps': 2,
        # TensorRT context count is deliberately measured one-by-one through
        # six contexts. The 3-context point exposes a knee between 2 and 4;
        # the 6-context point catches utilization rising after throughput has
        # already collapsed from paging.
        'pool_levels': (1, 2, 3, 4, 6),
        'sweep_pools': True,
        'provider_ab': True,
        'io': True,
        'batch_swap': True,
        'tile_upscale': True,
        # Measured end to end on the reference machine at 350-400 s across
        # four runs. The estimate is what the progress bar counts against, so
        # it is set a little long rather than a little short — a bar that
        # reaches 100% and keeps going reads as a hang.
        'est_sec': 450,
    },
    # A sustained, multi-face workload for validating stability rather than
    # finding a short peak. It keeps the active model catalogue and provider,
    # but repeats the end-to-end composite so thermal/VRAM drift and scheduler
    # imbalance become visible. Two faces/frame is the minimum deliberate load;
    # the API still accepts a larger value for a heavier test.
    'stress': {
        'measure_sec': 12.0,
        'warm_sec': 2.0,
        'reps': 2,
        'composite_reps': 3,
        'pool_levels': (1, 2, 3, 4, 6),
        'sweep_pools': True,
        'provider_ab': False,
        'io': True,
        'batch_swap': True,
        'tile_upscale': True,
        'default_faces': 2.0,
        'min_faces': 2.0,
        'est_sec': 900,
    },
}

# Leave this much VRAM unallocated at all times. A pool level that would eat into
# it is not measured — an OOM here costs the benchmark, but a pool size RECOMMENDED
# on the strength of a measurement taken with 200MB to spare costs every later run
# (a 12GB card at 8/8 was measured thrashing from 11.8 fps to 0.5 and still falling).
VRAM_RESERVE_GB = 1.25

# A larger pool has to beat the smaller one by this much to be worth its VRAM.
# Below it the two are the same measurement twice.
POOL_GAIN = 0.04
# Two samples whose spread exceeds this are not stable enough for an automatic
# context knee. Raw samples remain in the report for diagnosis.
STABILITY_SPREAD_PCT = 20.0
# Same idea for threads. Higher than the pool margin, not lower, because the
# thread curve is the noisier of the two and its tail is nearly flat: a measured
# heavy curve ran 9.09 f/s at 12 threads and 9.58 at 32, so a 2% bar bought a
# near-tripling of the worker count for 5% — and the next run's noise could as
# easily have reversed it. At 5% a bigger thread count has to earn itself.
THREAD_GAIN = 0.05

_NP_DTYPE = {
    'tensor(float)': np.float32, 'tensor(float16)': np.float16,
    'tensor(double)': np.float64, 'tensor(int64)': np.int64,
    'tensor(int32)': np.int32, 'tensor(bool)': np.bool_,
    'tensor(uint8)': np.uint8,
}


class Cancelled(Exception):
    """Raised out of any phase when the caller's cancel predicate goes true."""


# ── device ───────────────────────────────────────────────────────────────────

def _torch():
    import torch
    return torch


def vram_free_total_gb():
    """(free, total) VRAM in GB, whole-device.

    `mem_get_info` rather than torch's allocator stats: every byte that matters
    here is allocated by onnxruntime and TensorRT, and the allocator stats are
    blind to those. This is the reading the old benchmark got wrong, and getting
    it wrong is what let it call a configuration safe.
    """
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return 0.0, 0.0
        free, total = torch.cuda.mem_get_info()
        return free / (1024 ** 3), total / (1024 ** 3)
    except Exception:
        return 0.0, 0.0


def probe_device():
    """Collect the same runtime identity used by profile/cache selection.

    Keeping this probe shared with ``RuntimeOptimizer`` is important: a
    benchmark result must be attributable to the exact GPU/software stack that
    will consume it.  Missing optional probes stay explicit rather than being
    filled with an assumed RTX 3060/4070 value.
    """
    info = {
        'gpu_name': '', 'total_vram_gb': 0.0, 'free_vram_gb': 0.0,
        'cpu_physical': 0, 'cpu_logical': 0, 'platform': platform.platform(),
        'provider': '', 'trt_precision': '', 'ort_providers': [],
        'active_providers': [], 'cuda_ok': False,
    }
    try:
        from roop.runtime_optimizer import HardwareProfiler
        hardware = HardwareProfiler().profile()
        info.update({
            'device_id': hardware.device_id,
            'gpu_name': hardware.gpu_name,
            'gpu_vendor': hardware.gpu_vendor,
            'architecture': hardware.architecture,
            'compute_capability': hardware.compute_capability,
            'vram_tier': hardware.vram_tier,
            'total_vram_gb': round(hardware.vram_total_gb, 2),
            'free_vram_gb': round(hardware.vram_available_gb, 2),
            'available_vram_gb': round(hardware.vram_available_gb, 2),
            'cuda_ok': hardware.cuda_available,
            'cuda_version': hardware.cuda_version,
            'driver_version': hardware.driver_version,
            'tensorrt_version': hardware.tensorrt_version,
            'onnxruntime_version': hardware.onnxruntime_version,
            'tensor_core_capabilities': list(hardware.tensor_core_capabilities),
            'fp16_supported': hardware.fp16_supported,
            'bf16_supported': hardware.bf16_supported,
            'int8_supported': hardware.int8_supported,
            'fp8_supported': hardware.fp8_supported,
            'nvdec_available': hardware.nvdec_available,
            'nvdec_codecs': list(hardware.nvdec_codecs),
            'nvenc_available': hardware.nvenc_available,
            'nvenc_codecs': list(hardware.nvenc_codecs),
            'cpu_physical': hardware.cpu_physical_cores,
            'cpu_logical': hardware.cpu_logical_cores,
            'ram_total_gb': hardware.ram_total_gb,
            'ram_available_gb': hardware.ram_available_gb,
            'platform': hardware.platform or info['platform'],
            'hardware_profile_key': hardware.as_dict().get('hardware_profile_key'),
        })
    except Exception:
        # Keep the legacy lightweight probe as a fallback for partial installs.
        # It still reports unknown fields as unknown; it never substitutes a
        # result from another GPU.
        pass
    try:
        import psutil
        info['cpu_physical'] = info['cpu_physical'] or psutil.cpu_count(logical=False) or 0
        info['cpu_logical'] = info['cpu_logical'] or psutil.cpu_count(logical=True) or 0
    except Exception:
        info['cpu_logical'] = os.cpu_count() or 4
        info['cpu_physical'] = max(1, info['cpu_logical'] // 2)
    try:
        torch = _torch()
        if torch.cuda.is_available() and not info['gpu_name']:
            props = torch.cuda.get_device_properties(0)
            info['gpu_name'] = props.name
            info['cuda_ok'] = True
    except Exception:
        pass
    free, total = vram_free_total_gb()
    info['free_vram_gb'] = info['free_vram_gb'] or round(free, 2)
    info['total_vram_gb'] = info['total_vram_gb'] or round(total, 2)
    info['available_vram_gb'] = info['free_vram_gb']
    # Stable identity for persisted benchmark/profile isolation.  Free VRAM is
    # retained for telemetry, but the key intentionally excludes it because it
    # changes during model loading and cannot identify a hardware target.
    if not info.get('hardware_profile_key'):
        try:
            from roop.hardware_validation import hardware_profile_key
            info['hardware_profile_key'] = hardware_profile_key(info)
        except Exception:
            info['hardware_profile_key'] = None
    try:
        info['ort_providers'] = list(onnxruntime.get_available_providers())
    except Exception:
        pass
    cfg = roop.globals.CFG
    info['provider'] = getattr(cfg, 'provider', '') if cfg else ''
    info['trt_precision'] = getattr(cfg, 'trt_precision', '') if cfg else ''
    info['active_providers'] = [_provider_name(p) for p in
                                (roop.globals.execution_providers or [])]
    info['trt_tuning'] = current_trt_tuning()
    info['cpu_tuning'] = {
        'opencv_kernel_threads': os.environ.get('ROOP_CV_THREADS', 'auto'),
        'ort_dynamic_block_base': os.environ.get('ROOP_ORT_DYNAMIC_BLOCK', '4'),
    }
    return info


def current_trt_tuning():
    """Return the effective TensorRT tuning knobs used by new sessions."""
    try:
        builder = int(os.environ.get('ROOP_TRT_BUILDER_OPT_LEVEL', '3'))
    except (TypeError, ValueError):
        builder = 3
    try:
        aux = int(os.environ.get('ROOP_TRT_AUX_STREAMS', '-1'))
    except (TypeError, ValueError):
        aux = -1
    graph = str(os.environ.get('ROOP_TRT_CUDA_GRAPH', '0')).strip().lower()
    return {
        'builder_optimization_level': max(0, min(5, builder)),
        'auxiliary_streams': max(-1, min(8, aux)),
        'cuda_graph': graph in ('1', 'true', 'yes', 'on'),
    }


def _provider_name(p):
    return p[0] if isinstance(p, (tuple, list)) else str(p)


def _is_trt(p):
    return 'tensorrt' in _provider_name(p).lower()


def _trt_active():
    return any(_is_trt(p) for p in (roop.globals.execution_providers or []))


def _without_trt(providers):
    out = [p for p in providers if not _is_trt(p)]
    return out or ['CUDAExecutionProvider', 'CPUExecutionProvider']


# ── stage catalogue ──────────────────────────────────────────────────────────

class Stage:
    """One model the pipeline runs, with everything needed to reproduce its cost.

    `pool_knob` is the setting that widens it. `None` means the stage has NO pool
    in the shipping code, which under TensorRT means it takes the global GPU lock
    and serialises against every other stage — that is a property worth reporting
    on its own, because it is invisible in a per-call number.
    """

    def __init__(self, key, label, path, providers, pool_knob, calls_per_frame,
                 shape_hint=None, note='', in_modes=(), batch_relax=False):
        self.key = key
        self.label = label
        self.path = path
        self.providers = providers
        self.pool_knob = pool_knob
        self.calls_per_frame = calls_per_frame
        self.shape_hint = shape_hint
        self.note = note
        self.in_modes = in_modes
        self.batch_relax = batch_relax
        # filled by measurement
        self.provider_used = ''
        self.ms_call = 0.0
        self.build_sec = 0.0
        self.vram_mb = 0.0
        self.scaling = []
        self.best_n = 1
        self.error = ''

    @property
    def pooled(self):
        return self.pool_knob is not None

    def as_dict(self):
        return {
            'key': self.key, 'label': self.label,
            'model': os.path.basename(self.path) if self.path else '',
            'provider': self.provider_used,
            'pool_knob': self.pool_knob or 'none (global GPU lock)',
            'pooled': self.pooled,
            'calls_per_frame': round(self.calls_per_frame, 3),
            'input_shape': list(self.shape_hint) if self.shape_hint else [],
            'ms_call': round(self.ms_call, 3),
            'ms_frame': round(self.ms_call * self.calls_per_frame, 3),
            'vram_per_instance_mb': round(self.vram_mb, 1),
            'build_sec': round(self.build_sec, 2),
            'scaling': self.scaling,
            'best_n': self.best_n,
            'note': self.note,
            'error': self.error,
        }


def _models_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


def _exists(rel):
    p = os.path.join(_models_dir(), rel)
    return p if os.path.exists(p) else None


def _enhancer_model(name):
    """(path, note) for the enhancer the settings name, or (None, why-not)."""
    if name == 'GPEN Realistic':
        # The processor supports a deliberate 256px preview tier as well as
        # its normal 512px path. Follow the active processor knob so the
        # benchmark measures the same graph and transfer volume.
        try:
            size = int(os.environ.get('ROOP_GPENR_SIZE', '512') or 512)
        except ValueError:
            size = 512
        if size == 256:
            return _exists('gpen_bfr_256.onnx'), 'GPEN Realistic 256 luminance with swapper chrominance'
        return _exists('GPEN-BFR-512.onnx'), 'GPEN Realistic 512 luminance with swapper chrominance'
    table = {
        'GFPGAN': ('GFPGANv1.4.onnx', 'default'),
        'Codeformer': (os.path.join('CodeFormer', 'CodeFormerv0.1.onnx'), 'has a fidelity input'),
        'Codeformer (fp16)': (os.path.join('CodeFormer', 'codeformer.fp16.onnx'), 'distinct FP16 graph'),
        'GPEN 256': ('gpen_bfr_256.onnx', 'FP32-forced under TRT'),
        'GPEN 256 Pro': ('gpen_bfr_256.onnx', 'Upgraded GPEN 256 (sharper, photoreal, high-texture)'),
        'GPEN': ('GPEN-BFR-512.onnx', 'FP32-forced under TRT'),
        'GPEN 1024': ('gpen_bfr_1024.onnx', 'FP32-forced under TRT'),
        'GPEN 2048': ('gpen_bfr_2048.onnx', 'FP32-forced under TRT'),
        'Restoreformer++': ('restoreformer_plus_plus.onnx', ''),
        # UltraMax is a lean CodeFormer path, not a GPEN network. Keep this
        # pointed at the model the processor actually opens so its benchmark
        # cannot report GPEN-512 timings under the UltraMax name.
        'UltraMax': (os.path.join('CodeFormer', 'codeformer.fp16.onnx'),
                     'CodeFormer fp16 with UltraMax lean host path'),
    }
    if name not in table:
        return None, f'{name} is not a single-file ONNX enhancer'
    rel, note = table[name]
    return _exists(rel), note


def _mask_models(engine_names):
    """[(label, relpath, no_trt)] for the selected mask engine(s)."""
    table = {
        'DFL XSeg': ('xseg.onnx', False),
        'Face Occluder v3 (XSeg-3)': ('xseg_3.onnx', False),
        'Face Occluder': ('face_occluder.onnx', False),
        'Face Parser (BiSeNet)': ('resnet18.onnx', False),
        'Segment Anything (FastSAM)': ('fastsam_s.onnx', True),
        # MobileSAM runs encoder AND decoder per call; the encoder dominates and
        # is the honest single-session stand-in for the pair.
        'Segment Anything (MobileSAM)': ('mobilesam_encoder.onnx', True),
    }
    out = []
    for name in engine_names:
        if name == 'RealityUX':
            # RealityUX is a live composite: Mask_RealityUX runs the XSeg
            # authoritative mask and the BiSeNet parser concurrently. Measure
            # both selected models so the benchmark does not silently replace
            # this user's mask pipeline with an unrelated workload.
            out.extend([
                ('RealityUX / DFL XSeg', 'xseg.onnx', False),
                ('RealityUX / BiSeNet', 'resnet18.onnx', False),
            ])
        elif name in table:
            rel, no_trt = table[name]
            out.append((name, rel, no_trt))
    return out


def _detector_model(engine):
    """(label, relpath, input size) of the detector the engine actually runs."""
    size = 640
    try:
        size = int(str(getattr(roop.globals, 'face_detector_size', None)
                       or getattr(roop.globals.CFG, 'face_detector_size', 640)))
    except Exception:
        size = 640
    table = {
        'scrfd': ('SCRFD det_10g', 'buffalo_l/det_10g.onnx'),
        'retinaface': ('RetinaFace 10g', 'retinaface_10g.onnx'),
        'retinaface_r50': ('RetinaFace r50', 'retinaface_r50.onnx'),
        'yoloface': ('YOLOFace 8n', 'yoloface_8n.onnx'),
    }
    label, rel = table.get(engine, table['scrfd'])
    return label, rel, size


def build_catalogue(faces_per_frame=1.0):
    """The stages the CURRENT settings will run, in per-frame execution order.

    Built from live settings rather than a fixed list so the answer is about the
    configuration the user renders with. A model that is selected but not on disk
    is skipped with a note instead of failing the run.
    """
    cfg = roop.globals.CFG
    prov = list(roop.globals.execution_providers or ['CPUExecutionProvider'])
    from roop.processors.FaceSwapInsightFace import SWAP_MODELS, _swap_providers
    from roop.processors.Enhance_GPEN import _fp32_trt_providers
    from roop import session_pool

    stages = []
    warnings = []
    F = max(0.1, float(faces_per_frame))

    # ── detect: the detector net once per frame, aux models once per face ────
    engine = getattr(cfg, 'detector_engine', 'scrfd')
    det_label, det_rel, det_size = _detector_model(engine)
    det_path = _exists(det_rel)
    if det_path:
        stages.append(Stage(
            'detect', f'Detector — {det_label} @{det_size}', det_path, prov,
            'detector_pool', 1.0, shape_hint=(det_size, det_size),
            note='one call per frame; hybrid engines pool via ROOP_DETECTOR_POOL',
            in_modes=('standard', 'enhanced', 'heavy')))
    else:
        warnings.append(f'detector model missing: {det_rel}')

    rec = _exists('buffalo_l/w600k_r50.onnx')
    if rec:
        stages.append(Stage(
            'recognize', 'Recognition — w600k_r50 @112', rec, prov,
            'detmask_pool', F, shape_hint=(112, 112),
            note='per face; feeds identity matching AND the swapper',
            in_modes=('standard', 'enhanced', 'heavy')))
    lm = _exists('buffalo_l/2d106det.onnx')
    if lm:
        stages.append(Stage(
            'landmark', 'Landmarks — 2d106det @192', lm, prov,
            'detmask_pool', F, shape_hint=(192, 192),
            note='per face; landmark_3d_68 is lazy and not counted here',
            in_modes=('standard', 'enhanced', 'heavy')))

    # ── swap: tiles per face is the number that matters, not ms/call ─────────
    swap_key = getattr(cfg, 'swap_model', 'inswapper')
    spec = SWAP_MODELS.get(swap_key) or SWAP_MODELS['inswapper']
    swap_path = _exists(spec['file'])
    try:
        subsample = int(str(getattr(cfg, 'subsample_upscale', '256px'))[:3])
    except Exception:
        subsample = 256
    out_size = int(spec.get('output_size', 128))
    tiles = max(1, (subsample // out_size) ** 2)
    if swap_path:
        stages.append(Stage(
            'swap', f'Swapper — {swap_key} @{out_size}', swap_path,
            _swap_providers(prov), 'trt_pool', F * tiles,
            note=f'{tiles} tile(s) per face at {subsample}px pixel boost; '
                 'FP32-forced under TRT (FP16 overflows to rainbow smudge)',
            in_modes=('standard', 'enhanced', 'heavy'), batch_relax=True))
    else:
        warnings.append(f'swap model missing: {spec["file"]}')

    # ── enhance ──────────────────────────────────────────────────────────────
    enh_name = getattr(cfg, 'selected_enhancer', 'None')
    if enh_name and enh_name != 'None':
        enh_path, enh_note = _enhancer_model(enh_name)
        if enh_path:
            is_gpen = enh_name.startswith('GPEN')
            # CodeFormer, RestoreFormer++ and UltraMax pool on ROOP_TRT_POOL; GPEN, GFPGAN
            # and DMDNet have no pool at all, so under TensorRT they hold the
            # global lock and serialise the whole pipeline behind one face.
            pooled_enh = (enh_name.startswith('Codeformer') or
                          enh_name in ('Restoreformer++', 'UltraMax',
                                       'GPEN Realistic', 'GPEN 256 Pro'))
            # The low-VRAM tier deliberately disables pools. A stage must not
            # be benchmarked as pooled there when the real processor will use
            # one shared session behind the GPU guard.
            pool_knob = ('trt_pool' if pooled_enh and
                         session_pool.pooling_enabled() else None)
            needs_fp32 = enh_name in ('GPEN 1024', 'GPEN 2048')
            stages.append(Stage(
                'enhance', f'Enhancer — {enh_name}', enh_path,
                _fp32_trt_providers(prov) if needs_fp32 else prov,
                pool_knob, F,
                note=enh_note or ('no pool — takes the global GPU lock'
                                  if pool_knob is None else ''),
                in_modes=('enhanced', 'heavy')))
        elif enh_note:
            warnings.append(f'enhancer not measured: {enh_note}')

    # ── mask ─────────────────────────────────────────────────────────────────
    engines = [getattr(cfg, 'mask_engine', 'None'),
               getattr(cfg, 'mask_engine_2', 'None')]
    engines = [e for e in engines if e and e != 'None']
    measured_any_mask = False
    for label, rel, no_trt in _mask_models(engines):
        path = _exists(rel)
        if not path:
            warnings.append(f'mask model missing: {rel}')
            continue
        measured_any_mask = True
        stages.append(Stage(
            f'mask_{os.path.splitext(os.path.basename(rel))[0]}',
            f'Mask — {label}', path,
            session_pool.providers_without_tensorrt(prov) if no_trt else prov,
            'detmask_pool', F,
            note='TRT EP removed for this model' if no_trt else '',
            in_modes=('heavy',)))
    if engines and not measured_any_mask:
        warnings.append('no selected mask engine is a single-file ONNX model '
                        '(Clip2Seg and SAM2 are not) — masking is not in the '
                        'heavy composite')

    # ── expression restore (heavy only, and only its dominant model) ─────────
    # Through `ensure_patched_model`, not the shipped file. warping_spade warps a
    # 5-D feature volume with GridSample, which TensorRT rejects outright
    # ("nbDims == 4") and which onnxruntime's own kernel also refuses, so timing
    # the raw file does not fail gracefully — it throws. The app rewrites those
    # nodes once into `warping_spade-trt.onnx` and runs that; so does this.
    warp = _exists('liveportrait/warping_spade.onnx')
    expression_strength = float(
        getattr(cfg, 'expression_restore_strength', 0.0) or 0.0)
    lipsync_enabled = bool(getattr(cfg, 'lipsync_enabled', False))
    if warp and (expression_strength > 0.0 or lipsync_enabled):
        try:
            from roop.gridsample5d import ensure_patched_model
            warp = ensure_patched_model(warp, verbose=False) or warp
        except Exception:
            pass
        stages.append(Stage(
            'expression', 'Expression — warping_spade', warp, prov,
            'expr_pool', F,
            note='68% of the LivePortrait restore; the 4 smaller models are not '
                 'counted, so the heavy mode understates this stage',
            in_modes=('heavy',)))

    return stages, warnings


# ── feeds ────────────────────────────────────────────────────────────────────

def make_feeds(sess, shape_hint=None, batch=1, seed=0):
    """Inputs from the session's declared signature, dtypes taken FROM the model.

    Only the session signature is consulted, never the raw ONNX graph: exports
    like GPEN declare every weight as a graph input, and onnxruntime already
    resolves those against their initialisers. Feeding them would upload 300MB of
    random weights per call and time the PCIe bus instead of the network.
    """
    rng = np.random.default_rng(seed)
    feeds = {}
    for inp in sess.get_inputs():
        shape = []
        for i, d in enumerate(inp.shape):
            if isinstance(d, str) or d is None or (isinstance(d, int) and d < 1):
                if i == 0:
                    shape.append(batch)
                elif shape_hint and len(inp.shape) == 4:
                    # NCHW vs NHWC: whichever axis is not the 3-channel one.
                    shape.append(shape_hint[0] if i in (2, 3) else 3)
                else:
                    shape.append(1)
            else:
                shape.append(int(d))
        # A 4-D input with a hint gets the hint on its spatial axes even when the
        # model declares them statically for batch 1 but was exported dynamic.
        dt = _NP_DTYPE.get(inp.type, np.float32)
        if not shape:
            feeds[inp.name] = np.array(0.5, dtype=dt)
            continue
        if np.issubdtype(dt, np.floating):
            arr = (rng.random(shape) * 2 - 1).astype(dt)
        else:
            arr = np.ones(shape, dtype=dt)
        feeds[inp.name] = arr
    return feeds


# ── session building ─────────────────────────────────────────────────────────

def _build_session(stage, providers=None):
    """One session for `stage`, built the way the app builds it.

    The swapper's graph transforms are reproduced exactly (frozen ConvTranspose
    reshape, batch relaxation under ROOP_BATCH_SWAP), because they change the
    graph hash and therefore which TensorRT engine gets loaded — mirroring them
    means this hits the engine the app already built instead of provoking a
    fresh multi-minute build for a benchmark.
    """
    providers = providers if providers is not None else stage.providers
    opts = onnxruntime.SessionOptions()
    opts.enable_cpu_mem_arena = False
    model_arg = stage.path
    if stage.batch_relax:
        try:
            import onnx
            from roop.processors.FaceSwapInsightFace import (
                _freeze_convtranspose_reshape, _relax_batch_dim, _BATCH_SWAP)
            m = onnx.load(stage.path)
            changed = _freeze_convtranspose_reshape(m)
            if _BATCH_SWAP:
                _relax_batch_dim(m)
                changed = True
            if changed:
                model_arg = m.SerializeToString()
        except Exception:
            model_arg = stage.path
    return onnxruntime.InferenceSession(model_arg, opts, providers=providers)


# ── measurement primitives ───────────────────────────────────────────────────

def _run_once(sess, feeds, out_names):
    sess.run(out_names, feeds)


def throughput(sessions, feeds_list, out_names, workers, seconds, warm,
               cancelled, serialize=None):
    """Calls/second with `workers` threads leasing one session each from `sessions`.

    This is the pipeline's own concurrency model: `SessionPool.lease` hands one
    TensorRT context to one thread for the duration of a call, and a stage with
    no pool instead takes the global lock — passed here as `serialize`. A pool of
    one and a global lock are the same measurement, which is exactly why the
    un-pooled stages are the interesting ones.
    """
    q = Queue()
    for s in sessions:
        q.put(s)
    counts = [0] * workers
    errors = []
    clock = _StartGate(workers, seconds)

    def call(sess, feeds):
        if serialize is not None:
            with serialize:
                _run_once(sess, feeds, out_names)
        else:
            _run_once(sess, feeds, out_names)

    def work(w):
        feeds = feeds_list[w % len(feeds_list)]
        # Warm on this thread with this thread's buffers: the first call on a
        # TensorRT context allocates its device memory, and counting that as
        # steady state makes a wide pool look worse than a narrow one purely
        # because it has more first calls.
        t_end = time.perf_counter() + warm
        try:
            while time.perf_counter() < t_end:
                sess = q.get()
                try:
                    call(sess, feeds)
                finally:
                    q.put(sess)
        except Exception as e:                      # noqa: BLE001 - reported, not raised
            errors.append(e)
        if not clock.ready():
            return
        n = 0
        while not clock.expired() and not errors:
            try:
                sess = q.get()
                try:
                    call(sess, feeds)
                finally:
                    q.put(sess)
            except Exception as e:                  # noqa: BLE001
                errors.append(e)
                break
            n += 1
        counts[w] = n

    threads = [threading.Thread(target=work, args=(i,), daemon=True)
               for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + warm + 180)
    if errors:
        raise errors[0]
    if cancelled():
        raise Cancelled()
    total = sum(counts)
    return total / clock.elapsed(), total


def best_of(reps, fn, *args, **kw):
    """Fastest of `reps` runs of a throughput measurement.

    The max, not the mean, because the noise here is one-sided: another process
    taking the GPU, a thermal dip, the OS scheduler — every one of them can only
    make a run slower than the hardware is capable of. Measured on the reference
    machine with the app idle in the background, the same stage came out 211 and
    275 calls/s in two runs minutes apart — a 30% spread, against a 4% margin
    used to decide whether a pool level earned its VRAM. Averaging that in makes
    the decision a coin toss; taking the best of two makes it a measurement.
    """
    best = 0.0
    for _ in range(max(1, reps)):
        got = fn(*args, **kw)
        if isinstance(got, tuple):
            got = got[0]
        best = max(best, got)
    return best


def _sampled_throughput(reps, fn, *args, **kw):
    """Return the existing best-of value plus samples for stability reporting."""
    samples = []
    for _ in range(max(1, int(reps))):
        got = fn(*args, **kw)
        if isinstance(got, tuple):
            got = got[0]
        samples.append(float(got))
    return max(samples), samples


class _StartGate:
    """Start the clock when the LAST worker has finished warming up.

    The obvious shape — main thread waits on a barrier, then sets the deadline —
    has a race that silently zeroes the measurement: workers are released by the
    same barrier and read the deadline before the main thread has written it, so
    they see 0.0, decide time is up and return having counted nothing. The
    deadline is therefore set by the barrier's own action, which runs before any
    party is released.
    """

    def __init__(self, parties, seconds):
        self._seconds = seconds
        self._t0 = 0.0
        self._stop = 0.0
        self._barrier = threading.Barrier(parties, action=self._start)

    def _start(self):
        self._t0 = time.perf_counter()
        self._stop = self._t0 + self._seconds

    def ready(self):
        """Block until every worker is warm. False = a worker died; give up."""
        try:
            self._barrier.wait(timeout=300)
            return True
        except threading.BrokenBarrierError:
            return False

    def expired(self):
        return time.perf_counter() >= self._stop

    def elapsed(self):
        return max(1e-6, time.perf_counter() - self._t0)


# ── phase 1/2: per-stage cost and pool scaling ───────────────────────────────

def measure_stage(stage, cfg, report, cancelled, sweep_pools, max_level):
    """Serial ms/call, then throughput at each affordable pool level.

    Sessions are built INCREMENTALLY and reused across levels — level 4 keeps the
    two sessions level 2 built — so a four-point sweep costs at most
    `max_level` session builds rather than rebuilding each candidate.
    """
    sessions = []
    feeds_list = []
    out_names = None
    levels = [n for n in cfg['pool_levels'] if n <= max_level] or [1]
    try:
        for li, target in enumerate(levels):
            while len(sessions) < target:
                if cancelled():
                    raise Cancelled()
                t0 = time.perf_counter()
                try:
                    sess = _build_session(stage)
                except Exception as e:              # noqa: BLE001
                    if not sessions:
                        stage.error = f'{type(e).__name__}: {str(e)[:120]}'
                        return
                    report(log=f'  {stage.label}: stopped at {len(sessions)} '
                               f'instance(s) — {type(e).__name__}')
                    target = len(sessions)
                    break
                build = time.perf_counter() - t0
                if not sessions:
                    stage.build_sec = build
                    stage.provider_used = sess.get_providers()[0].replace(
                        'ExecutionProvider', '')
                    out_names = [o.name for o in sess.get_outputs()]
                    if build > 25:
                        report(log=f'  {stage.label}: built a TensorRT engine '
                                   f'({build:.0f}s) — first run only, cached now')
                sessions.append(sess)
                feeds_list.append(make_feeds(sess, stage.shape_hint,
                                             seed=len(feeds_list)))
            if not sessions:
                return

            n = len(sessions)
            report(status=f'{stage.label} — {n} instance(s)',
                   stage=stage.label, threads=n)
            try:
                calls_s, samples = _sampled_throughput(
                    cfg.get('reps', 1), throughput, sessions, feeds_list,
                    out_names, n, cfg['measure_sec'], cfg['warm_sec'], cancelled)
            except Cancelled:
                raise
            except Exception as e:                  # noqa: BLE001
                # A model that will not RUN — a 5-D GridSample the provider
                # refuses, a shape the harness guessed wrong — is one stage's
                # problem. Recording it and carrying on is the difference between
                # a report with a gap in it and no report at all.
                stage.error = f'{type(e).__name__}: {str(e)[:120]}'
                report(log=f'  ! {stage.label} did not run: {stage.error}')
                return
            if not np.isfinite(calls_s) or calls_s <= 0:
                # Some providers can fail during the first inference without
                # raising, leaving the harness with a zero throughput value.
                # Do not turn that sentinel into a convincing 1e12 ms result.
                stage.error = f'invalid throughput: {calls_s!r}'
                report(log=f'  ! {stage.label} did not run: {stage.error}')
                return
            if n == 1:
                stage.ms_call = 1000.0 / max(1e-9, calls_s)
            base = stage.scaling[0]['calls_s'] if stage.scaling else calls_s
            free_now, _ = vram_free_total_gb()
            spread_pct = ((max(samples) - min(samples)) /
                          max(1e-9, calls_s) * 100.0)
            stage.scaling.append({
                'n': n,
                'calls_s': round(calls_s, 2),
                'runs_fps': [round(x, 2) for x in samples],
                'spread_pct': round(spread_pct, 1),
                'stable': spread_pct <= STABILITY_SPREAD_PCT,
                # Latency per call WITH n threads in flight, not the serial cost:
                # a pool trades latency for throughput and both belong in the row.
                'ms_latency': round(1000.0 * n / max(1e-9, calls_s), 3),
                'speedup': round(calls_s / max(1e-9, base), 3),
                'free_vram_gb': round(free_now, 2),
            })
            report(fps=round(calls_s, 1), vram=round(free_now, 2),
                   log=f'  {stage.label} x{n}: {calls_s:7.1f} calls/s '
                       f'({calls_s / max(1e-9, base):.2f}x, {free_now:.1f} GB free)')
            if not sweep_pools or not stage.pooled:
                break
            # Do not stop at the first plateau. Phase 4 needs the complete
            # 1/2/3/4/6 curve because a point can recover after a noisy sample and
            # because the selected knee is the smallest point within POOL_GAIN
            # of the best, not simply the first local maximum.
            # Cost per EXTRA instance, not cost of the first: the first session
            # also pays for the CUDA/TensorRT context, the engine deserialisation
            # and the arena, none of which repeat. Charging that to instance two
            # overstates a pool by hundreds of MB and stops the sweep early.
            stage.vram_mb = _per_instance_mb(stage.scaling)
            need = (levels[li + 1] - n) * stage.vram_mb / 1024.0 if li + 1 < len(levels) else 0
            if li + 1 < len(levels) and free_now - need < VRAM_RESERVE_GB:
                report(log=f'  {stage.label}: stopping the sweep at {n} — going to '
                           f'{levels[li + 1]} needs ~{need:.1f} GB and only '
                           f'{free_now:.1f} GB is free')
                break
        stage.vram_mb = _per_instance_mb(stage.scaling) or stage.vram_mb
        stable = [s for s in stage.scaling if s.get('stable', True)]
        chosen = stable or stage.scaling
        stage.best_n = _knee([s['calls_s'] for s in chosen],
                             [s['n'] for s in chosen], POOL_GAIN)
    finally:
        for s in sessions:
            del s
        sessions.clear()
        _release_vram()


def _per_instance_mb(scaling):
    """VRAM one EXTRA instance of a stage costs, from the widest span measured.

    Free VRAM is read whole-device, so this also absorbs anything else on the
    card moving during the sweep. It is a marginal cost and is used as one — to
    decide whether the next pool level fits, and to tell the user what a pool
    costs — never as an absolute allocation figure.
    """
    if len(scaling) < 2:
        return 0.0
    first, last = scaling[0], scaling[-1]
    span = last['n'] - first['n']
    if span <= 0:
        return 0.0
    freed = (first['free_vram_gb'] - last['free_vram_gb']) * 1024.0
    return max(0.0, freed / span)


def _knee(values, labels, gain):
    """Smallest label whose value is within `gain` of the best.

    Picking the argmax alone buys a fraction of a percent with a doubling of
    VRAM or threads. The knee is what the pool tiers in session_pool were chosen
    on and it is what this reproduces per device.
    """
    if not values:
        return labels[0] if labels else 1
    best = max(values)
    for v, n in zip(values, labels):
        if v >= best * (1.0 - gain):
            return n
    return labels[int(np.argmax(values))]


def _release_vram():
    try:
        torch = _torch()
        torch.cuda.empty_cache()
    except Exception:
        pass


# ── phase 3: the composite frame ─────────────────────────────────────────────

class _CpuFrameWork:
    """The per-frame CPU work a worker does around its GPU calls.

    Not a stand-in: `align_crop` is the pipeline's own function, and the paste is
    the same warpAffine-plus-feathered-blend at frame resolution that
    procmgr_masking performs. It is here because the optimal thread count is
    precisely the point where one frame's CPU work overlaps another's GPU work —
    measure only the GPU side and the sweep recommends too few threads.

    What it does NOT include is colour transfer, the mask derivations that run on
    CPU, stabilisation and the writer, so it is a floor on CPU cost, not the
    whole of it.
    """

    def __init__(self, width=1920, height=1080, crop=512, faces=1.0):
        import cv2
        self.cv2 = cv2
        rng = np.random.default_rng(7)
        self.frame = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        self.crop = crop
        self.faces = faces
        cx, cy, s = width * 0.5, height * 0.5, height * 0.18
        self.kps = np.array([
            [cx - 0.34 * s, cy - 0.38 * s], [cx + 0.34 * s, cy - 0.38 * s],
            [cx, cy + 0.10 * s], [cx - 0.37 * s, cy + 0.55 * s],
            [cx + 0.37 * s, cy + 0.55 * s]], dtype=np.float32)
        self.mask = None

        from roop.face_util import align_crop
        self._align_crop = align_crop
        m = np.zeros((crop, crop), dtype=np.float32)
        import cv2 as _cv
        _cv.ellipse(m, (crop // 2, crop // 2),
                    (int(crop * 0.42), int(crop * 0.50)), 0, 0, 360, 1.0, -1)
        self.mask = _cv.GaussianBlur(m, (0, 0), crop * 0.03)

    def frame_pass(self):
        align_crop = self._align_crop
        cv2 = self.cv2
        reps = max(1, int(round(self.faces)))
        out = self.frame.copy()
        for _ in range(reps):
            aligned, M = align_crop(self.frame, self.kps, self.crop)
            blob = (aligned[:, :, ::-1].astype(np.float32) / 255.0)
            blob = blob.transpose(2, 0, 1)[None]
            swapped = (blob[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            IM = cv2.invertAffineTransform(M)
            h, w = self.frame.shape[:2]
            back = cv2.warpAffine(swapped, IM, (w, h), borderMode=cv2.BORDER_REPLICATE)
            bmask = cv2.warpAffine(self.mask, IM, (w, h))[:, :, None]
            out = (back * bmask + out * (1 - bmask)).astype(np.uint8)
        return out


def measure_composite(stages, pools, modes, thread_levels, cfg, report, cancelled,
                      cpu_work, vram_out=None, metrics_out=None):
    """Frames/s at each thread count, for each workload mode.

    Each worker executes a whole frame's GPU call sequence in pipeline order,
    leasing from pools sized exactly as the recommendation, and taking a SHARED
    global lock for any stage that has no pool — which is the serialisation that
    decides where the thread curve stops rising. Under CUDA there is no global
    lock, matching `_gpu_guard`.

    All three modes share ONE set of pools, built once. Building them per mode
    would trebled the session builds and, worse, measure each mode against a
    different VRAM situation — the heavy mode holds every pool at once, so that
    is the allocation the standard mode has to be judged next to as well.
    """
    wanted = [s for s in stages
              if not s.error and any(m in s.in_modes for m in modes)]
    if not wanted:
        return {m: {} for m in modes}
    global_lock = threading.Lock() if _trt_active() else None

    built = {}
    keep = []
    curves = {m: {} for m in modes}
    try:
        for st in wanted:
            n = max(1, int(pools.get(st.pool_knob, 1))) if st.pooled else 1
            sessions, feeds, outs = [], [], None
            for i in range(n):
                if cancelled():
                    raise Cancelled()
                free, _ = vram_free_total_gb()
                if i and free < VRAM_RESERVE_GB:
                    report(log=f'  composite: capped {st.label} at {i} instance(s) '
                               f'({free:.1f} GB free)')
                    break
                try:
                    sess = _build_session(st)
                except Exception as e:              # noqa: BLE001
                    report(log=f'  composite: {st.label} stopped at {i} '
                               f'({type(e).__name__})')
                    break
                sessions.append(sess)
                feeds.append(make_feeds(sess, st.shape_hint, seed=i))
                outs = [o.name for o in sess.get_outputs()]
            if not sessions:
                # A stage with an empty pool would block the first worker that
                # reaches it forever, so it is dropped and said so — a composite
                # missing a stage is a wrong answer, not a hang.
                report(log=f'  ! composite: {st.label} has no session — excluded')
                continue
            q = Queue()
            for s in sessions:
                q.put(s)
            keep.append(sessions)
            built[st.key] = (st, q, feeds, outs)

        # ── the allocation does not happen at build time ─────────────────────
        # `InferenceSession(...)` returns before TensorRT has allocated its
        # execution-context memory — that lands on the first inference. So the
        # per-instance check in the build loop above cannot see the pools' real
        # cost, and on the reference machine a config it waved through went on to
        # occupy 11961 MiB of 12282 at 100% utilisation, where the driver pages
        # over PCIe and a frame that takes 100 ms takes minutes.
        #
        # Touching every session once makes every context allocate. Reading free
        # VRAM after THAT is the first honest reading, and if it is under the
        # reserve the sweep is abandoned rather than measured: the numbers from a
        # paging card are not slow, they are meaningless, and they take a long
        # time to collect.
        if built:
            _allocate_contexts(built)
            free, _ = vram_free_total_gb()
            # Report it out: this is the only reading that says what THIS pool
            # configuration really occupies, and the pool check needs it to know
            # what a wider one would have left over.
            if vram_out is not None:
                vram_out['free_gb'] = free
            if free < VRAM_RESERVE_GB:
                report(log=f'  ! composite: {free:.1f} GB free once the pools '
                           f'allocated — under the {VRAM_RESERVE_GB} GB reserve. '
                           'Not measuring a card that is about to page.')
                return {m: {} for m in modes}

        for mode in modes:
            active = {k: v for k, v in built.items() if mode in v[0].in_modes}
            if not active:
                continue
            best = 0.0
            stalls = 0
            for t in thread_levels:
                if cancelled():
                    raise Cancelled()
                report(status=f'{mode} composite — {t} thread(s)',
                       stage=f'{mode} frame', threads=t)
                runs = []
                for _ in range(max(1, int(cfg.get('composite_reps', 1)))):
                    runs.append(_composite_run(
                        active, t, cfg['measure_sec'] + 0.5,
                        cfg['warm_sec'], global_lock, cpu_work))
                # Sustained profiles use the median for recommendations. Keep
                # the distribution in the result so a short peak cannot hide
                # thermal throttling, VRAM paging, or scheduler variance.
                fps = float(np.median(runs))
                if metrics_out is not None:
                    metrics_out.setdefault(mode, {})[str(t)] = {
                        'runs_fps': [round(x, 2) for x in runs],
                        'median_fps': round(fps, 2),
                        'min_fps': round(min(runs), 2),
                        'max_fps': round(max(runs), 2),
                        'spread_pct': round(
                            (max(runs) - min(runs)) / max(1e-9, fps) * 100, 1),
                    }
                curves[mode][str(t)] = round(fps, 2)
                free, _ = vram_free_total_gb()
                report(fps=round(fps, 1), vram=round(free, 2),
                       log=f'  {mode:9s} @ {t:2d} threads: {fps:6.2f} frames/s')
                # Stop climbing after TWO consecutive levels that beat nothing.
                # One is not enough — the curve is noisy enough to dip and
                # recover (measured: enhanced went 7.95, 7.89, 9.35 over 4/6/8
                # threads), and stopping on the first dip would have reported the
                # optimum three levels early. Two in a row is a plateau.
                if fps > best * (1.0 + THREAD_GAIN):
                    best = fps
                    stalls = 0
                else:
                    stalls += 1
                    if stalls >= 2:
                        report(log=f'  {mode:9s}: plateaued past {t} threads')
                        break
    finally:
        for _k, (_st, q, _f, _o) in built.items():
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
        built.clear()
        keep.clear()
        _release_vram()
    return curves


def _one_frame(built, w, acc, global_lock, cpu_work):
    """One frame's worth of work for worker `w`, in pipeline order."""
    for key, (st, q, feeds, outs) in built.items():
        # Fractional calls per frame (a clip averaging 1.5 faces) are carried in
        # an accumulator rather than rounded, so over a measurement the stage
        # runs exactly as often as the frame rate says it should.
        acc[key] += st.calls_per_frame
        reps = int(acc[key])
        acc[key] -= reps
        fd = feeds[w % len(feeds)]
        for _ in range(reps):
            sess = q.get()
            try:
                if global_lock is not None and not st.pooled:
                    with global_lock:
                        sess.run(outs, fd)
                else:
                    sess.run(outs, fd)
            finally:
                q.put(sess)
    if cpu_work is not None:
        cpu_work.frame_pass()


def _allocate_contexts(built):
    """Run EVERY session in every pool once, so every context allocates.

    Not one frame: a single-threaded frame leases the same session out of a pool
    of eight over and over, so seven contexts would still be unallocated and the
    VRAM reading taken afterwards would be an eighth of the truth. The queue is
    drained so each session is touched exactly once, then refilled.
    """
    for _key, (_st, q, feeds, outs) in built.items():
        held = []
        while not q.empty():
            try:
                held.append(q.get_nowait())
            except Exception:
                break
        for i, sess in enumerate(held):
            try:
                sess.run(outs, feeds[i % len(feeds)])
            except Exception:                       # noqa: BLE001
                pass                                # the timed pass will report it
        for sess in held:
            q.put(sess)


def _composite_run(built, workers, seconds, warm, global_lock, cpu_work):
    counts = [0] * workers
    errors = []
    clock = _StartGate(workers, seconds)

    def work(w):
        acc = {k: 0.0 for k in built}
        t_end = time.perf_counter() + warm
        try:
            while time.perf_counter() < t_end:
                _one_frame(built, w, acc, global_lock, cpu_work)
        except Exception as e:                      # noqa: BLE001
            errors.append(e)
        if not clock.ready():
            return
        n = 0
        while not clock.expired() and not errors:
            try:
                _one_frame(built, w, acc, global_lock, cpu_work)
            except Exception as e:                  # noqa: BLE001
                errors.append(e)
                break
            n += 1
        counts[w] = n

    threads = [threading.Thread(target=work, args=(i,), daemon=True)
               for i in range(workers)]
    for t in threads:
        t.start()
    # A frame is tens of milliseconds; a minute of slack covers the first one on
    # a cold context and nothing else. Past that the card is paging, and joining
    # "eventually" would mean tearing down sessions that threads are still
    # calling into — so this is an error, not a slow result.
    deadline = seconds + warm + 60
    for t in threads:
        t.join(timeout=deadline)
    alive = [t for t in threads if t.is_alive()]
    if alive:
        raise RuntimeError(
            f'{len(alive)} of {workers} composite workers did not finish within '
            f'{deadline:.0f}s — the GPU is almost certainly out of VRAM and paging')
    if errors:
        raise errors[0]
    return sum(counts) / clock.elapsed()


# ── phase 4: TensorRT vs CUDA ────────────────────────────────────────────────

def measure_provider_ab(stages, cfg, report, cancelled):
    """Per-stage ms/call on the same models with the TensorRT EP removed.

    Worth measuring rather than assuming: TensorRT is the default here and is
    faster on the models it builds well, but it is also the reason every
    un-pooled stage serialises, it is not available on every install, and on
    some models the app removes it outright. A per-stage table says where the
    provider setting is actually earning its restart.
    """
    rows = []
    for st in stages:
        if st.error or not st.ms_call:
            continue
        if not any(_is_trt(p) for p in st.providers):
            continue                    # already CUDA/CPU — nothing to compare
        if cancelled():
            raise Cancelled()
        report(status=f'CUDA vs TensorRT — {st.label}', stage=st.label, threads=1)
        try:
            sess = _build_session(st, providers=_without_trt(st.providers))
            feeds = [make_feeds(sess, st.shape_hint)]
            outs = [o.name for o in sess.get_outputs()]
            # Same rep count as the TensorRT side it is compared against —
            # a best-of-2 number against a single run is not a comparison.
            calls_s = best_of(cfg.get('reps', 1), throughput, [sess], feeds, outs,
                              1, cfg['measure_sec'], cfg['warm_sec'], cancelled)
            cuda_ms = 1000.0 / max(1e-9, calls_s)
            # What ORT ACTUALLY ran, read AFTER the measurement — the timing of
            # this line is the entire guard.
            #
            # When the CUDA EP cannot initialise a node — RestoreFormer++ here
            # hits `CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED` on a Conv —
            # onnxruntime does not raise. It prints, drops the EP, rebuilds the
            # session CPU-only and retries, and the row then reports "CUDA
            # 1005 ms, TensorRT 18x faster" for a number that is a CPU time.
            #
            # The failure happens on the FIRST INFERENCE, not at construction,
            # so a check placed before the run reads the intention rather than
            # the outcome. Measured on this machine, same session object:
            #
            #     before run: ['CUDAExecutionProvider', 'CPUExecutionProvider']
            #     first run:  1948 ms      <- the fallback happens in here
            #     after run:  ['CPUExecutionProvider']
            #
            # The pre-run check therefore passed on the one stage it existed
            # for, and a full report went out with `stages_cuda_refused: 0` and
            # a 1005 ms CPU time averaged into the provider verdict. Same shape
            # as the TensorRT allocation that only appears once something has
            # actually run: nothing about an EP is settled until then.
            resolved = sess.get_providers()[0].replace('ExecutionProvider', '')
            fell_back = 'CUDA' not in resolved.upper()
            rows.append({
                'stage': st.label,
                'trt_ms': round(st.ms_call, 3),
                'cuda_ms': round(cuda_ms, 3),
                'trt_speedup': round(cuda_ms / max(1e-9, st.ms_call), 2),
                'calls_per_frame': round(st.calls_per_frame, 3),
                'cuda_provider': resolved,
                'cuda_fell_back': fell_back,
            })
            report(log=f'  {st.label}: TRT {st.ms_call:.2f} ms vs '
                       f'{resolved} {cuda_ms:.2f} ms '
                       f'({cuda_ms / max(1e-9, st.ms_call):.2f}x)'
                       + ('  [CUDA EP refused this model — that is a CPU time]'
                          if fell_back else ''))
            del sess
            _release_vram()
        except Cancelled:
            raise
        except Exception as e:                      # noqa: BLE001
            # Recorded, not skipped. "This model does not run on CUDA here" is
            # the strongest possible argument for the TensorRT setting, and a
            # silently missing row reads as "not measured".
            rows.append({
                'stage': st.label,
                'trt_ms': round(st.ms_call, 3),
                'cuda_ms': None,
                'trt_speedup': None,
                'calls_per_frame': round(st.calls_per_frame, 3),
                'error': f'{type(e).__name__}: {str(e)[:100]}',
            })
            report(log=f'  {st.label}: will not run on CUDA here — '
                       f'{type(e).__name__}')
    return rows


# ── phase 7: batched swap ────────────────────────────────────────────────────

def measure_batch_swap(stages, cfg, report, cancelled):
    """Measure safe swap batch candidates and return model-specific results.

    A batch dimension is not a proxy for concurrency: it competes with
    independent TensorRT contexts for VRAM.  Candidates are therefore timed
    one at a time on the app-shaped session, with batch 1, 2, 4, and 8 kept as
    separate rows.  A static export records a rejected row instead of silently
    treating it as a batch-one win.
    """
    swap = next((s for s in stages if s.key == 'swap' and not s.error), None)
    if swap is None:
        return {}
    report(status='Batched swap', stage=swap.label, threads=1)
    try:
        sess = _build_session(swap)
        outs = [o.name for o in sess.get_outputs()]
        rows = []
        for candidate in (1, 2, 4, 8):
            try:
                feeds = [make_feeds(sess, swap.shape_hint,
                                    batch=candidate)]
                free_before, total = vram_free_total_gb()
                calls = best_of(cfg.get('reps', 1), throughput, [sess], feeds,
                                outs, 1, cfg['measure_sec'], cfg['warm_sec'],
                                cancelled)
                free_after, _ = vram_free_total_gb()
                rows.append({
                    'batch_size': candidate,
                    'batches_s': round(calls, 3),
                    'items_s': round(calls * candidate, 1),
                    'batch_latency_ms': round(1000.0 / max(calls, 1e-9), 3),
                    'vram_free_before_gb': round(free_before, 3),
                    'vram_free_after_gb': round(free_after, 3),
                    'vram_used_after_gb': round(max(0.0, total - free_after), 3)
                    if total else None,
                })
                report(log=f'  swap batch {candidate}: '
                           f'{calls * candidate:.1f} items/s, '
                           f'{1000.0 / max(calls, 1e-9):.2f} ms/batch, '
                           f'free VRAM {free_after:.2f} GB')
            except Exception as e:                  # noqa: BLE001
                rows.append({
                    'batch_size': candidate,
                    'error': f'{type(e).__name__}: {str(e)[:100]}',
                })
                report(log=f'  swap batch {candidate}: rejected '
                           f'({type(e).__name__})')
        valid = [row for row in rows if 'items_s' in row]
        if valid:
            winner = max(valid, key=lambda row: row['items_s'])
            baseline = next(row for row in valid if row['batch_size'] == 1)
            gain = winner['items_s'] / max(baseline['items_s'], 1e-9) - 1.0
            out = {
                'candidates': rows,
                'recommended_batch_size': winner['batch_size'],
                'batch1_items_s': baseline['items_s'],
                'recommended_items_s': winner['items_s'],
                'gain_pct': round(gain * 100, 1),
                # Kept for older UI/report consumers.
                'batch1_faces_s': baseline['items_s'],
                'batch4_faces_s': next((r['items_s'] for r in valid
                                        if r['batch_size'] == 4), None),
                'recommend': 'on' if winner['batch_size'] > 1 and gain > 0.03
                             else 'off',
            }
        else:
            out = {'candidates': rows, 'recommend': 'off',
                   'error': 'all swap batch candidates rejected'}
        del sess
        _release_vram()
        return out
    except Cancelled:
        raise
    except Exception as e:                          # noqa: BLE001
        return {'error': f'{type(e).__name__}: {str(e)[:100]}'}


# ── phase 6: encode / decode ─────────────────────────────────────────────────

def measure_tile_upscale(cfg, report, cancelled):
    """Measure bounded Frame_Upscale tile batches on the selected device.

    This times the public frame path, so tile creation, overlap/crop merging,
    host copies, and model inference are included.  The batch-one output is a
    correctness reference for every candidate.
    """
    free, total = vram_free_total_gb()
    if not total:
        return {'status': 'skipped', 'reason': 'no CUDA device'}
    # Eight tiles are useful on the 12GB target.  On the 6GB tier this is not a
    # safe candidate by policy; record the guard instead of risking an OOM.
    candidates = (1, 2, 4, 8) if total >= 7.0 else (1, 2, 4)
    old_tile = os.environ.get('ROOP_UPSCALE_TILE')
    old_batch = os.environ.get('ROOP_UPSCALE_TILE_BATCH')
    old_runtime_batch = os.environ.get('ROOP_RUNTIME_UPSCALE_TILE_BATCH')
    proc = None
    try:
        from roop.processors.Frame_Upscale import Frame_Upscale
        report(status='Batched frame upscale', stage='Frame_Upscale', threads=1)
        os.environ['ROOP_UPSCALE_TILE'] = '128'
        os.environ.pop('ROOP_RUNTIME_UPSCALE_TILE_BATCH', None)
        proc = Frame_Upscale()
        proc.Initialize({'devicename': 'cuda', 'subtype': 'span_x4'})
        frame = np.empty((256, 256, 3), dtype=np.uint8)
        yy, xx = np.indices(frame.shape[:2])
        frame[:, :, 0] = (xx + 3 * yy) % 256
        frame[:, :, 1] = (2 * xx + yy) % 256
        frame[:, :, 2] = (xx + yy) % 256

        os.environ['ROOP_UPSCALE_TILE_BATCH'] = '1'
        reference = proc.RunThreadSafe(frame)
        rows = []
        for candidate in candidates:
            if cancelled():
                raise Cancelled()
            os.environ['ROOP_UPSCALE_TILE_BATCH'] = str(candidate)
            proc._tile_batch_unsupported = False
            try:
                proc.RunThreadSafe(frame)
                deadline = time.perf_counter() + max(0.5, cfg['measure_sec'])
                calls = 0
                started = time.perf_counter()
                last = None
                while time.perf_counter() < deadline or calls < 2:
                    if cancelled():
                        raise Cancelled()
                    last = proc.RunThreadSafe(frame)
                    calls += 1
                elapsed = max(time.perf_counter() - started, 1e-9)
                max_diff = int(np.max(np.abs(last.astype(np.int16) -
                                              reference.astype(np.int16))))
                free_after, total_after = vram_free_total_gb()
                rows.append({
                    'batch_size': candidate,
                    'effective_batch_size': 1 if proc._tile_batch_unsupported
                    else candidate,
                    'supported': not proc._tile_batch_unsupported,
                    'tiles_per_frame': len(proc.create_tile_frames(
                        frame, (128, 8, 2))[0]),
                    'fps': round(calls / elapsed, 3),
                    'latency_ms': round(1000.0 * elapsed / calls, 3),
                    'max_abs_diff_vs_batch1': max_diff,
                    'vram_free_before_gb': round(free, 3),
                    'vram_free_after_gb': round(free_after, 3),
                    'vram_used_after_gb': round(max(0.0, total_after - free_after), 3)
                    if total_after else None,
                })
                report(log=f"  upscale tile batch {candidate}: "
                           f"{calls / elapsed:.2f} frame/s, "
                           f"{1000.0 * elapsed / calls:.2f} ms/frame, "
                           f"effective={rows[-1]['effective_batch_size']}, "
                           f"max diff={max_diff}, free VRAM {free_after:.2f} GB")
            except Exception as e:                  # noqa: BLE001
                rows.append({'batch_size': candidate,
                             'error': f'{type(e).__name__}: {str(e)[:120]}'})
                report(log=f'  upscale tile batch {candidate}: rejected '
                           f'({type(e).__name__})')
        valid = [r for r in rows if 'fps' in r and r['supported']]
        winner = max(valid, key=lambda r: r['fps']) if valid else None
        result = {'model': 'span_x4', 'tile_size': 128,
                  'candidates': rows,
                  'candidate_8': 'not_safe_on_sub7GB' if total < 7.0 else 'tested'}
        if winner:
            result.update({
                'recommended_batch_size': winner['batch_size'],
                'recommended_fps': winner['fps'],
                'batch1_fps': next((r['fps'] for r in valid
                                    if r['batch_size'] == 1), None),
            })
        else:
            result['error'] = 'no supported tile batch candidate'
        return result
    except Cancelled:
        raise
    except Exception as e:                          # noqa: BLE001
        return {'error': f'{type(e).__name__}: {str(e)[:120]}'}
    finally:
        if proc is not None:
            try:
                proc.Release()
            except Exception:
                pass
        _release_vram()
        if old_tile is None:
            os.environ.pop('ROOP_UPSCALE_TILE', None)
        else:
            os.environ['ROOP_UPSCALE_TILE'] = old_tile
        if old_batch is None:
            os.environ.pop('ROOP_UPSCALE_TILE_BATCH', None)
        else:
            os.environ['ROOP_UPSCALE_TILE_BATCH'] = old_batch
        if old_runtime_batch is None:
            os.environ.pop('ROOP_RUNTIME_UPSCALE_TILE_BATCH', None)
        else:
            os.environ['ROOP_RUNTIME_UPSCALE_TILE_BATCH'] = old_runtime_batch


def _bundled_ffmpeg():
    """Pinokio's own ffmpeg, for when this is not running in a Pinokio shell.

    Inside the app `shutil.which` finds it, because Pinokio puts it on the PATH
    of the shells it launches — `roop.core` relies on exactly that. A benchmark
    started from an ordinary terminal (or `python -m roop.bench`) does not
    inherit it, and the encode/decode phase would then silently report nothing.
    Derived from the app's own location rather than hardcoded, since the layout
    is fixed: <PINOKIO_HOME>/api/<launcher>/app.
    """
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    home = os.path.abspath(os.path.join(here, '..', '..', '..'))
    for rel in (os.path.join('bin', 'ffmpeg-env', 'Library', 'bin'),
                os.path.join('bin', 'ffmpeg-env', 'bin')):
        cand = os.path.join(home, rel, 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
        if os.path.exists(cand):
            return cand
    return None



def measure_io(report, cancelled, seconds_of_video=2.0, width=1920, height=1080,
               fps=30):
    """Encoder presets and NVDEC, on a clip generated for the purpose.

    The render's wall clock is decode + swap + encode, and the first and last are
    the two settings (`perf_encoder_preset`, `perf_nvdec`) that no amount of GPU
    benchmarking touches. Synthetic frames are a moving gradient rather than
    noise: noise is incompressible and makes every encoder look equally slow,
    which is the one comparison this must not get wrong.
    """
    import shutil
    import subprocess
    import tempfile
    import cv2

    out = {'encode': [], 'decode': [], 'notes': []}
    ff = shutil.which('ffmpeg') or _bundled_ffmpeg()
    if not ff:
        out['notes'].append('ffmpeg not found — encode/decode not measured')
        return out

    n_frames = int(seconds_of_video * fps)
    tmp = tempfile.mkdtemp(prefix='roop_bench_')
    raw = os.path.join(tmp, 'raw.mp4')

    # Materialise the frames ONCE, before any clock starts. Generating them
    # inside the feed loop would put numpy's cost inside the encoder's number and
    # make a fast encoder indistinguishable from a slow one.
    xs = np.linspace(0, 255, width, dtype=np.float32)
    base = np.repeat(xs[None, :], height, axis=0)
    payload = []
    for i in range(n_frames):
        f = np.empty((height, width, 3), dtype=np.uint8)
        shift = (base + i * 3.0) % 255.0
        f[:, :, 0] = shift
        f[:, :, 1] = np.roll(shift, i * 5, axis=1)
        f[:, :, 2] = 255 - shift
        y0 = (i * 7) % max(1, height - 200)
        f[y0:y0 + 200, 100:400] = 240
        payload.append(f.tobytes())

    def encode(codec, preset_args, path):
        cmd = [ff, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', codec]
        cmd += preset_args + [path]
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for buf in payload:
                proc.stdin.write(buf)
            proc.stdin.close()
            proc.wait(timeout=300)
        except Exception:
            proc.kill()
            return None
        if proc.returncode != 0:
            return None
        return n_frames / max(1e-6, time.perf_counter() - t0)

    # libx265 is in here because it is the app's DEFAULT video codec: measuring
    # only x264 would rank presets for an encoder most renders do not use. The
    # medium/faster pair per encoder is the `perf_encoder_preset` question, and
    # at a fixed CRF the two are visually equivalent — the preset trades encoder
    # search time for bitrate, not quality, so the file size column is the cost.
    trials = [
        ('libx265 medium', 'libx265', ['-preset', 'medium', '-crf', '18']),
        ('libx265 faster', 'libx265', ['-preset', 'faster', '-crf', '18']),
        ('libx264 medium', 'libx264', ['-preset', 'medium', '-crf', '18']),
        ('libx264 faster', 'libx264', ['-preset', 'faster', '-crf', '18']),
        ('h264_nvenc p5', 'h264_nvenc', ['-preset', 'p5', '-cq', '20']),
        ('hevc_nvenc p5', 'hevc_nvenc', ['-preset', 'p5', '-cq', '20']),
    ]
    for label, codec, args in trials:
        if cancelled():
            raise Cancelled()
        report(status=f'Encoder — {label}', stage='encode')
        path = os.path.join(tmp, f'{codec}_{len(out["encode"])}.mp4')
        got = encode(codec, args, path)
        if got is None:
            out['notes'].append(f'{label}: not available on this build')
            continue
        out['encode'].append({'name': label, 'fps': round(got, 1),
                              'mb': round(os.path.getsize(path) / 1e6, 2)})
        report(fps=round(got, 1), log=f'  encode {label}: {got:.1f} fps')
        if not os.path.exists(raw):
            try:
                import shutil as _sh
                _sh.copyfile(path, raw)
            except Exception:
                pass

    payload.clear()          # ~370 MB of raw frames; the decode half does not need them

    # ── decode: cv2 against ffmpeg -hwaccel cuda ─────────────────────────────
    # On a LONG clip, and that is the whole point. Measured over the 60-frame
    # encode clip instead, cv2 read 218 fps against NVDEC's 132 and the obvious
    # conclusion — turn NVDEC off — was an artefact: spawning ffmpeg costs a few
    # hundred milliseconds, which is most of a two-second clip and none of a
    # real render, where the process is started once for the whole video. So the
    # clip is looped out to something long enough for startup to stop mattering
    # and each reader gets one pass over it.
    long_path = os.path.join(tmp, 'long.mp4')
    long_frames = n_frames
    if os.path.exists(raw):
        loops = max(1, int(round(600 / max(1, n_frames))))
        try:
            subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-y',
                            '-stream_loop', str(loops - 1), '-i', raw,
                            '-c', 'copy', long_path],
                           check=True, timeout=120)
            long_frames = n_frames * loops
        except Exception:
            long_path = raw

    def _passes(fn):
        t0 = time.perf_counter()
        got = fn()
        return got / max(1e-6, time.perf_counter() - t0)

    if os.path.exists(long_path):
        raw = long_path
        n_frames = long_frames
        report(status='Decode — cv2', stage='decode')

        def _cv2_pass():
            cap = cv2.VideoCapture(raw)
            n = 0
            while True:
                ok, _f = cap.read()
                if not ok:
                    break
                n += 1
            cap.release()
            return n

        cv2_fps = _passes(_cv2_pass)
        if cv2_fps:
            out['decode'].append({'name': 'cv2 (CPU)', 'fps': round(cv2_fps, 1)})
            report(fps=round(cv2_fps, 1), log=f'  decode cv2: {cv2_fps:.1f} fps')

        report(status='Decode — NVDEC', stage='decode')

        def _nvdec_pass(pix_fmt, prefetch_depth):
            from roop.nvdec_reader import FFmpegVideoReader
            reader = FFmpegVideoReader(raw, width, height, fps,
                                        hwaccel='cuda', pix_fmt=pix_fmt,
                                        prefetch_depth=prefetch_depth)
            got = 0
            try:
                while True:
                    ok, _frame = reader.read()
                    if not ok:
                        break
                    got += 1
            finally:
                reader.release()
            return got

        # The first arm preserves the old host-BGR boundary and measures the
        # cost of NVDEC without the reader's asynchronous buffer.  The second
        # arm is the production adaptive path: safe 8-bit 4:2:0 input uses a
        # smaller NV12 download plus CPU BGR conversion and a bounded prefetch
        # queue; other source formats retain the BGR fallback.
        for name, pix_fmt, prefetch_depth in (
                ('NVDEC (GPU, sync BGR)', 'bgr24', 0),
                ('NVDEC (GPU, adaptive buffered)', 'auto', None)):
            try:
                nv_fps = _passes(lambda: _nvdec_pass(pix_fmt, prefetch_depth))
                if nv_fps:
                    out['decode'].append({
                        'name': name, 'fps': round(nv_fps, 1),
                        'pix_fmt': pix_fmt,
                        'prefetch_depth': prefetch_depth,
                    })
                    report(fps=round(nv_fps, 1),
                           log=f'  decode {name}: {nv_fps:.1f} fps')
                else:
                    out['notes'].append(f'{name} produced no frames')
            except Exception as exc:
                out['notes'].append(f'{name} failed: {type(exc).__name__}')

    try:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass
    return out


# ── recommendation ───────────────────────────────────────────────────────────

# The encoder has to keep up with the pipeline, not win a drag race, and it is
# asked to do so with the CPU that the worker threads are already using. The
# measured figure comes from a lone ffmpeg with the whole machine and a 1080p
# synthetic clip; a real render gives it a fraction of the cores and possibly 4x
# the pixels. So a candidate has to beat the frame rate by this much before it
# counts as keeping up. Set from the two knowns rather than tuned: ~2x for the
# CPU it will not get, and the resolution unknown on top.
ENCODER_HEADROOM = 2.0

# Codecs the app will actually offer. Mirrors `_VIDEO_CODECS` in api.py, which is
# the authority — a recommendation outside that list would set a value the
# dropdown cannot display. Pinned by a test rather than imported, because
# importing api.py from here would boot the whole FastAPI app in the CLI.
APP_CODECS = ('libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc')


def _recommend_encoder(rows, threads, curves, current_codec=''):
    """The cheapest encoder that can keep up — not the fastest one measured.

    Encoding is NOT a separate pass: `ProcessMgr.write_frames_thread` streams
    frames into a live `FFMPEG_VideoWriter` while the swap is running. So an
    encoder that is already faster than the pipeline produces frames buys
    nothing in wall clock, and the fastest encoder measured here is the one that
    costs the most per frame stored — on the reference machine `hevc_nvenc p5`
    ran 170 f/s against libx265 medium's 42.5, for a file **15x larger** on the
    same 60 frames (1.5 MB against 0.10 MB).

    Ranking by fps alone therefore recommends a large quality-and-size cost for
    a speedup the render cannot use. What the report is really being asked is
    "can the writer keep up, and if not, what is the cheapest thing that can".

    So the PRESET moves first and the codec only if it has to. A faster
    libx264/libx265 preset at a fixed CRF is quality-neutral by construction —
    rate control holds perceptual quality constant, so the preset trades encoder
    search time for a slightly larger file and nothing else — which makes the
    output size an honest cost within one codec. Across codecs it is not: CRF 18
    does not mean the same thing to x264 and to x265, so "libx264 medium is
    0.20 MB against libx265 faster's 0.40" compares two different qualities and
    must not decide anything. The configured codec is therefore kept whenever
    any of its presets clears the bar.

    Only when no preset of the configured codec can keep up does the codec move,
    and then to the fastest thing measured — usually NVENC, which is also the
    right answer there for a second reason: it runs on the GPU's dedicated
    encode engine, off the cores the worker threads are competing for.
    """
    ok = [r for r in rows if r.get('fps')]
    if not ok:
        return {}
    fastest = max(ok, key=lambda r: r['fps'])
    # The rate the writer has to absorb: the highest sustained frame rate any
    # mode achieves at its recommended thread count. The heavy mode is slower
    # and cannot be the binding constraint.
    need = 0.0
    for mode, n in (threads or {}).items():
        try:
            need = max(need, float((curves.get(mode) or {}).get(str(n), 0) or 0))
        except (TypeError, ValueError):
            continue
    if need <= 0:
        return {'encoder': fastest['name'], 'encoder_fastest': fastest['name'],
                'encoder_reason': 'no thread curve to size the encoder against'}

    bar = need * ENCODER_HEADROOM
    same = [r for r in ok if r['name'].split()[0] == current_codec and r['fps'] >= bar]
    if same:
        # Cheapest preset of the codec already configured. `mb` missing means the
        # size could not be read; treating it as infinite keeps it from winning
        # on absent data.
        choice = min(same, key=lambda r: (r.get('mb') if r.get('mb') else float('inf')))
        reason = (f'{choice["name"]} keeps up with {need:.1f} f/s '
                  f'({choice["fps"]:.0f} f/s, {ENCODER_HEADROOM:g}x headroom) at the '
                  f'smallest output for this codec')
        if choice['name'] != fastest['name']:
            reason += (f'; {fastest["name"]} is faster but the render cannot use '
                       f'it — encoding overlaps the swap')
    else:
        choice = fastest
        reason = (f'no {current_codec or "configured"} preset clears {bar:.0f} f/s '
                  f'({ENCODER_HEADROOM:g}x the {need:.1f} f/s the pipeline produces), '
                  f'so the encoder would hold the render back')
    return {'encoder': choice['name'],
            'encoder_fastest': fastest['name'],
            'encoder_reason': reason}


def recommend(stages, device, pools, curves, provider_rows, io_res, batch_res):
    """Turn the measurements into the settings the app reads.

    Every value here is the knee of a curve that was measured on this device, and
    the report carries the curve next to it so a recommendation can be argued
    with rather than trusted.
    """
    logical = device.get('cpu_logical') or 8
    threads = {}
    for mode, curve in curves.items():
        if not curve:
            continue
        labels = sorted(int(k) for k in curve)
        values = [curve[str(n)] for n in labels]
        threads[mode] = min(_knee(values, labels, THREAD_GAIN), logical)
    for mode in ('standard', 'enhanced', 'heavy'):
        threads.setdefault(mode, min(4, logical))

    # Keep the model-level knees beside the shared UI knobs.  A single setting
    # is still used for legacy callers, but runtime SessionPool sizing can use
    # the exact stage curve when a model is loaded alongside other engines.
    rec = {
        'threads': threads,
        'pools': dict(pools),
        'context_knees': {
            st.key: int(st.best_n) for st in stages
            if st.pooled and not st.error and st.scaling
        },
    }

    # Rows where the CUDA EP was refused and onnxruntime silently ran the model
    # on CPU are excluded from the totals: a CPU time in the CUDA column would
    # inflate "TensorRT is N% faster" into a comparison with no CUDA in it. They
    # are still shown in the table, flagged, because "this model will not run on
    # the CUDA provider on this install" is itself the answer to the provider
    # question.
    timed = [r for r in provider_rows if r.get('cuda_ms') and not r.get('cuda_fell_back')]
    if timed:
        trt = sum(r['trt_ms'] * r['calls_per_frame'] for r in timed)
        cuda = sum(r['cuda_ms'] * r['calls_per_frame'] for r in timed)
        rec['provider'] = {
            'tensorrt_ms_frame': round(trt, 2),
            'cuda_ms_frame': round(cuda, 2),
            'recommend': 'tensorrt' if trt < cuda else 'cuda',
            'margin_pct': round((cuda / max(1e-9, trt) - 1) * 100, 1),
            'stages_compared': len(timed),
            'stages_cuda_refused': len(provider_rows) - len(timed),
        }
    elif provider_rows:
        # Every row refused. Emitting nothing here would drop the strongest
        # possible answer to the provider question — "the CUDA EP will not run
        # a single one of your stages on this install" — and it would read in
        # the UI as "the provider was not measured".
        rec['provider'] = {
            'tensorrt_ms_frame': round(sum(r['trt_ms'] * r['calls_per_frame']
                                           for r in provider_rows), 2),
            'cuda_ms_frame': None,
            'recommend': 'tensorrt',
            'margin_pct': None,
            'stages_compared': 0,
            'stages_cuda_refused': len(provider_rows),
        }

    if io_res and io_res.get('encode'):
        rec.update(_recommend_encoder(
            io_res['encode'], threads, curves,
            current_codec=str(getattr(roop.globals.CFG, 'output_video_codec', '')
                              or '') if roop.globals.CFG else ''))
    if io_res and len(io_res.get('decode', [])) == 2:
        cpu = next((d['fps'] for d in io_res['decode'] if 'cv2' in d['name']), 0)
        gpu = next((d['fps'] for d in io_res['decode'] if 'NVDEC' in d['name']), 0)
        # Only a COLLAPSE moves this setting, and the numbers are reported either
        # way. Two reasons the raw ratio cannot decide it:
        #
        #   * the clip is synthetic. A smooth gradient at CRF 18 is a fraction of
        #     a megabyte and trivially cheap for a CPU decoder — measured here
        #     cv2 272 fps against NVDEC 215 — where real high-bitrate footage is
        #     the case NVDEC exists for. The bench cannot make that clip without
        #     shipping one;
        #   * frames per second is not the whole benefit. NVDEC runs on a
        #     dedicated engine, so its throughput is throughput the CPU is NOT
        #     spending, and the CPU is what the worker threads are competing for.
        #
        # `auto` already probes per file and falls back on its own, so the
        # default is left alone unless the GPU path is less than half as fast,
        # which would mean something is actually wrong with it.
        if gpu and gpu < cpu * 0.5:
            rec['nvdec'] = 'off'
        io_res.setdefault('notes', []).append(
            'Decode figures are for a synthetic low-bitrate clip, which favours '
            'the CPU reader; NVDEC also offloads work the worker threads want. '
            'perf_nvdec is left on auto unless the GPU path collapses.')
    if batch_res and batch_res.get('recommend'):
        rec['batch_swap'] = batch_res['recommend']

    # The un-pooled stages are the headline finding on a TensorRT install, so say
    # so rather than leaving it in a table cell.
    serialised = [s.label for s in stages
                  if not s.pooled and not s.error and s.ms_call]
    if serialised and _trt_active():
        rec['serialised_stages'] = serialised
    return rec


def _validate_pools(stages, pools, curves, cfg, report, cancelled, cpu_work,
                    vram_at_knee):
    """Re-run the heavy frame with the COSTLIEST stage's pool one step wider.

    One knob, not all of them, and the measurement is why. Widening every pool at
    once on the reference machine (RTX 4070, 12 GB) took the four knobs from
    4/2/4/2 to 8/4/8/4 — forty TensorRT contexts — and drove the card to
    11961 MiB of 12282 used at 100% utilisation, where the driver starts paging
    over PCIe and throughput collapses. That is the same failure the pool tiers
    in `session_pool` exist to avoid, and a benchmark that walks into it both
    takes minutes to escape and risks RECOMMENDING it.

    It was also uninterpretable: four knobs moved, one number came back. So the
    knob tested is the one belonging to the stage with the largest share of the
    frame, since that is where another context can pay for itself, and the rest
    stay at the knee the per-stage sweep found.
    """
    heavy = curves.get('heavy') or {}
    if not heavy:
        return {}
    best_t = max(heavy, key=lambda k: heavy[k])
    base_fps = heavy[best_t]

    # Cost of the knob, summed over every stage that shares it. Recognition,
    # landmarks and the mask engine all lease from detmask_pool, so a step there
    # allocates a slot in each of them — taking the max instead of the sum is
    # what made the headroom check pass when it should not have.
    cost_mb, share_ms = {}, {}
    for st in stages:
        if not st.pooled or st.error:
            continue
        cost_mb[st.pool_knob] = cost_mb.get(st.pool_knob, 0.0) + max(st.vram_mb, 64.0)
        share_ms[st.pool_knob] = share_ms.get(st.pool_knob, 0.0) + st.ms_call * st.calls_per_frame
    if not share_ms:
        return {}

    knob = max(share_ms, key=lambda k: share_ms[k])
    cur = pools.get(knob, 1) or 1
    per_step_gb = cost_mb.get(knob, 512.0) / 1024.0
    # Headroom is what was free WHILE THE KNEE CONFIG WAS LOADED, not what is
    # free now that it has been released. The wider config is the knee config
    # plus a delta, so comparing the delta against an empty card is short by the
    # whole base allocation — measured, that waved a 4->8 step through against
    # 9.9 GB "free" when the knee itself was using 6 of the 12, and the card went
    # to 11899 MiB of 12282 and started paging over PCIe.
    headroom = (vram_at_knee or {}).get('free_gb')
    if headroom is None:
        headroom, _ = vram_free_total_gb()
    # Prefer a doubling, but fall back to a single extra slot rather than
    # reporting "no headroom" for a step that was simply too big. On a 12GB card
    # trt_pool 4->8 costs ~4.7 GB and does not fit; 4->5 costs ~1.2 GB and does,
    # and the answer to "is one more worth it" is the useful one.
    step = cur
    for cand in (min(8, cur * 2), cur + 1):
        if cand > cur and cand <= 8 and headroom - (cand - cur) * per_step_gb >= VRAM_RESERVE_GB:
            step = cand
            break
    if step == cur:
        need_gb = per_step_gb
        report(log=f'  {knob} {cur}->{cur + 1} would need ~{need_gb:.1f} GB on top '
                   f'of the {headroom:.1f} GB left with the current pools — '
                   f'keeping {cur}')
        return {'winner': 'knee', 'knob': knob,
                'reason': 'no VRAM headroom' if cur < 8 else 'already at the cap',
                'needed_gb': round(need_gb, 2), 'free_gb': round(headroom, 2)}

    need_gb = (step - cur) * per_step_gb
    wider = dict(pools)
    wider[knob] = step
    report(status=f'Pool check — {knob} {cur}→{step}', phase='threads',
           log=f'  {knob} carries {share_ms[knob]:.0f} ms of the frame — trying '
               f'{cur}->{step} (~{need_gb:.1f} GB more)')
    alt = measure_composite(stages, wider, ('heavy',), [int(best_t)], cfg,
                            report, cancelled, cpu_work)
    alt_fps = (alt.get('heavy') or {}).get(best_t, 0.0)
    won = alt_fps > base_fps * (1.0 + POOL_GAIN)
    report(log=f'  pool check @ {best_t} threads: {knob}={cur} gave {base_fps:.2f} '
               f'fps, {knob}={step} gave {alt_fps:.2f} fps -> keeping '
               f'{step if won else cur}')
    return {
        'winner': 'wider' if won else 'knee',
        'knob': knob, 'threads': int(best_t),
        'knee_pools': dict(pools), 'knee_fps': base_fps,
        'wider_pools': wider, 'wider_fps': alt_fps,
        'wider_curve': alt if won else {},
    }


def _summary(device, rec, stages):
    per_frame = sum(s.ms_call * s.calls_per_frame for s in stages if not s.error)
    parts = [
        f"{device.get('gpu_name') or 'GPU'} "
        f"({device.get('total_vram_gb')} GB), "
        f"{device.get('cpu_physical')}c/{device.get('cpu_logical')}t",
        f"{per_frame:.0f} ms of GPU work per frame",
        'threads ' + '/'.join(f"{m[:3]} {rec['threads'].get(m, '-')}"
                              for m in ('standard', 'enhanced', 'heavy')),
        'pools ' + '/'.join(f"{k.replace('_pool', '')} {v}"
                            for k, v in rec['pools'].items()),
    ]
    if rec.get('provider'):
        parts.append(f"provider {rec['provider']['recommend']}")
    return '; '.join(parts)


# ── the run ──────────────────────────────────────────────────────────────────

def run_benchmark(profile='full', faces_per_frame=1.0, report=None,
                  cancelled=None, apply_to_config=True):
    """Measure this device and return the report the UI renders / config stores.

    `report(**kw)` receives progress: `status`, `log`, `phase`, `pct`, `stage`,
    `threads`, `fps`, `vram`. `cancelled()` is polled between measurements.
    """
    cfg_profile = PROFILES.get(profile) or PROFILES['full']
    report = report or (lambda **kw: None)
    cancelled = cancelled or (lambda: False)
    t_start = time.perf_counter()
    # Errors only. A benchmark builds dozens of sessions, and onnxruntime's
    # TensorRT logger emits several paragraphs of warnings per build; left on,
    # they bury the benchmark's own output in the console it shares with the app.
    try:
        onnxruntime.set_default_logger_severity(3)
    except Exception:
        pass

    device = probe_device()
    report(phase='probe', pct=1,
           status=f"{device['gpu_name'] or 'CPU'} — {device['total_vram_gb']} GB",
           log=f"{device['gpu_name'] or 'no CUDA device'}, "
               f"{device['total_vram_gb']} GB VRAM ({device['free_vram_gb']} GB free), "
               f"{device['cpu_physical']} cores / {device['cpu_logical']} threads")
    report(log=f"providers: {', '.join(device['active_providers']) or 'none'}"
               f" (config: {device['provider']}/{device['trt_precision']})")

    stages, warnings = build_catalogue(faces_per_frame)
    for w in warnings:
        report(log=f'  ! {w}')
    if not stages:
        return {'status': 'error', 'message': 'no measurable models found',
                'warnings': warnings}
    report(log=f'{len(stages)} stage(s) from the current settings: '
               + ', '.join(s.label for s in stages))

    # A pool cannot be wider than the threads that would lease from it.
    # Context autotuning has a bounded candidate set through six. The live VRAM
    # reserve stops a candidate before allocation if it cannot fit; six is a
    # measurement point, not a recommendation.
    max_level = max(1, min(6, device.get('cpu_logical') or 6))

    try:
        # ── per-stage cost + pool scaling ────────────────────────────────────
        for i, st in enumerate(stages):
            report(phase='stages',
                   pct=3 + int(45 * i / max(1, len(stages))),
                   status=f'Measuring {st.label}')
            measure_stage(st, cfg_profile, report, cancelled,
                          cfg_profile['sweep_pools'], max_level)
            if st.error:
                report(log=f'  ! {st.label}: {st.error}')

        # ── pool sizes ───────────────────────────────────────────────────────
        from roop import session_pool
        pools = {
            'trt_pool': session_pool.pool_size() or 1,
            'detmask_pool': session_pool.detmask_pool_size() or 1,
            'detector_pool': session_pool.detector_pool_size() or 1,
            'expr_pool': session_pool.expression_pool_size() or 1,
        }
        if cfg_profile['sweep_pools']:
            measured = {}
            for st in stages:
                if st.error or not st.pooled:
                    continue
                measured.setdefault(st.pool_knob, []).append(st.best_n)
            for knob, vals in measured.items():
                # The knob is shared by several stages (detect/mask; swapper and
                # the pooled enhancers), so it has to satisfy the widest of them.
                pools[knob] = int(max(vals))
            report(log='pool knees: ' + ', '.join(f'{k}={v}' for k, v in pools.items()))

        # ── composite thread sweep ───────────────────────────────────────────
        cpu_work = _CpuFrameWork(faces=faces_per_frame)
        logical = max(2, device.get('cpu_logical') or 8)
        levels = sorted({t for t in (1, 2, 4, 6, 8, 12, 16, logical)
                         if t <= logical})
        report(phase='threads', pct=50, status='Thread sweep')
        vram_at_knee = {}
        composite_metrics = {}
        curves = measure_composite(stages, pools,
                                   ('standard', 'enhanced', 'heavy'), levels,
                                   cfg_profile, report, cancelled, cpu_work,
                                   vram_out=vram_at_knee,
                                   metrics_out=composite_metrics)

        # ── does a wider pool actually help the whole frame? ─────────────────
        # The per-stage sweep above measures a stage against ITSELF, and that is
        # not the whole of what a pool does: under TensorRT an un-pooled stage
        # takes the GLOBAL lock, so widening one pool changes how much every
        # OTHER stage has to wait. A stage that plateaus alone can therefore
        # still be worth widening in a frame, and the only way to know is to run
        # the frame. One alternative config, at the thread count the sweep liked,
        # is cheap enough to be worth the certainty.
        pool_ab = {}
        if cfg_profile['sweep_pools'] and curves.get('heavy'):
            pool_ab = _validate_pools(stages, pools, curves, cfg_profile,
                                      report, cancelled, cpu_work, vram_at_knee)
            if pool_ab.get('winner') == 'wider':
                pools = pool_ab['wider_pools']
                report(log=f'  wider pools won end-to-end: {pools}')
                # The winning number stays in `pool_ab` and does NOT go into the
                # thread curve. It was measured at one thread count under a
                # different pool config, and dropping it in beside the rest
                # makes a curve whose points are not comparable — the thread
                # recommendation would then be reading a pool change as a thread
                # effect (measured: it lifted the 32-thread point from 9.58 to
                # 10.12 and nothing else moved).

        # ── provider A/B, batch swap, encode/decode ──────────────────────────
        provider_rows = []
        if cfg_profile['provider_ab'] and _trt_active():
            report(phase='provider', pct=86, status='TensorRT vs CUDA')
            provider_rows = measure_provider_ab(stages, cfg_profile, report, cancelled)

        batch_res = {}
        if cfg_profile['batch_swap']:
            report(phase='batch', pct=92, status='Batched swap')
            batch_res = measure_batch_swap(stages, cfg_profile, report, cancelled)

        tile_upscale_res = {}
        if cfg_profile.get('tile_upscale'):
            report(phase='batch', pct=93, status='Batched frame upscale')
            tile_upscale_res = measure_tile_upscale(cfg_profile, report, cancelled)

        io_res = {}
        if cfg_profile['io']:
            report(phase='io', pct=94, status='Encode / decode')
            io_res = measure_io(report, cancelled)

    except Cancelled:
        raise

    rec = recommend(stages, device, pools, curves, provider_rows, io_res, batch_res)
    duration = time.perf_counter() - t_start
    from roop.phase11_matrix import create_matrix

    result = {
        'status': 'success',
        'version': 2,
        'profile': profile,
        'ran_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_sec': round(duration, 1),
        'device': device,
        'hardware_profile_key': device.get('hardware_profile_key'),
        'validation_target': os.environ.get('ROOP_VALIDATION_TARGET', 'detected'),
        'gpu_name': device['gpu_name'],            # kept for the old UI shape
        'total_vram_gb': device['total_vram_gb'],
        'faces_per_frame': faces_per_frame,
        'workload': {
            'kind': 'sustained' if cfg_profile.get('composite_reps', 1) > 1
                    else 'calibration',
            'faces_per_frame': faces_per_frame,
            'measure_sec': cfg_profile['measure_sec'],
            'warm_sec': cfg_profile['warm_sec'],
            'composite_reps': cfg_profile.get('composite_reps', 1),
        },
        'composite_metrics': composite_metrics,
        'settings_measured': {
            'provider': getattr(roop.globals.CFG, 'provider', ''),
            'trt_precision': getattr(roop.globals.CFG, 'trt_precision', ''),
            'swap_model': getattr(roop.globals.CFG, 'swap_model', ''),
            'enhancer': getattr(roop.globals.CFG, 'selected_enhancer', ''),
            'mask_engine': getattr(roop.globals.CFG, 'mask_engine', ''),
            'mask_engine_2': getattr(roop.globals.CFG, 'mask_engine_2', ''),
            'detector_engine': getattr(roop.globals.CFG, 'detector_engine', ''),
            'face_detector_size': getattr(roop.globals, 'face_detector_size',
                                          getattr(roop.globals.CFG, 'face_detector_size', '')),
            'subsample_upscale': getattr(roop.globals.CFG, 'subsample_upscale', ''),
            'perf_trt_pool': getattr(roop.globals.CFG, 'perf_trt_pool', ''),
            'perf_detmask_pool': getattr(roop.globals.CFG, 'perf_detmask_pool', ''),
            'perf_detector_pool': getattr(roop.globals.CFG, 'perf_detector_pool', ''),
            'perf_batch_swap': getattr(roop.globals.CFG, 'perf_batch_swap', ''),
            'trt_tuning': current_trt_tuning(),
            'cpu_tuning': {
                'opencv_kernel_threads': os.environ.get('ROOP_CV_THREADS', 'auto'),
                'ort_dynamic_block_base': os.environ.get('ROOP_ORT_DYNAMIC_BLOCK', '4'),
            },
        },
        'stages': [s.as_dict() for s in stages],
        'pools': rec['pools'],
        'context_knees': rec.get('context_knees', {}),
        'threads': rec['threads'],
        # `Settings.resolve_threads` reads these two names; keeping them means the
        # thread recommendation reaches real runs without touching that path.
        'best_threads': rec['threads'],
        'fps_map': curves,
        'thread_curve': curves,
        'provider_ab': provider_rows,
        'pool_ab': pool_ab,
        'provider': rec.get('provider'),
        'batch_swap': batch_res,
        'tile_upscale': tile_upscale_res,
        'io': io_res,
        # Phase 11 keeps one row per source-discovered enhancer/upscaler.  The
        # selected stage can contribute measured inference timing; every other
        # row stays explicitly pending until that exact model/path is run.
        'enhancer_matrix': create_matrix(device, _phase11_matrix_measurements(stages)),
        'recommend': rec,
        'warnings': warnings,
    }
    # Keep the final report shape dual-target even when only one physical GPU
    # is attached. The caller may set ROOP_VALIDATION_TARGET to the target
    # being measured; no target is inferred for runtime configuration and an
    # unlabelled run leaves both target tables pending.
    from roop.hardware_validation import build_dual_target_report
    validation_target = result['validation_target']
    validation_records = {}
    if validation_target in ('RTX 3060', 'RTX 4070'):
        validation_records[validation_target] = {
            'status': 'measured_partial',
            'hardware': device,
            'hardware_profile_key': device.get('hardware_profile_key'),
            'notes': (
                'This benchmark records the detected device and stage/composite '
                'measurements. Supply a controlled baseline/final run before '
                'claiming a complete acceptance result.'
            ),
        }
    result['dual_target_validation'] = build_dual_target_report(validation_records)
    result['summary'] = _summary(device, rec, stages)

    if apply_to_config and roop.globals.CFG:
        result['applied'] = apply_recommendation(result)

    report(phase='done', pct=100, status='Benchmark complete',
           log=result['summary'])
    return result


def apply_recommendation(result):
    """Write what can be applied, and name what needs a restart.

    Thread counts take effect on the next run (`resolve_threads` reads them per
    job). The pool sizes are env-backed and read once at process start by
    `run.py`, so they are saved to config and reported as pending — claiming
    otherwise would have the user measure a change that has not happened.
    """
    cfg = roop.globals.CFG
    applied, pending = {}, {}
    cfg.benchmark_results = result
    rec = result.get('recommend') or {}

    pool_keys = {
        'trt_pool': 'perf_trt_pool',
        'detmask_pool': 'perf_detmask_pool',
        'detector_pool': 'perf_detector_pool',
        'expr_pool': 'perf_expr_pool',
    }
    # A calibration run may measure wider candidate pools, but those
    # candidates are not an admission decision.  On a detected sub-7GB GPU,
    # persist the tier's auto policy (0/0 pooled contexts and one detector)
    # rather than turning a benchmark result into a restart-time OOM risk.
    small_gpu = False
    try:
        import torch
        small_gpu = bool(torch.cuda.is_available() and
                         torch.cuda.get_device_properties(0).total_memory /
                         (1024 ** 3) < 7.0)
    except Exception:
        pass
    for knob, key in pool_keys.items():
        val = (rec.get('pools') or {}).get(knob)
        if val is None:
            continue
        current = str(getattr(cfg, key, 'auto') or 'auto').strip().lower()
        if small_gpu:
            if current not in ('', 'auto'):
                setattr(cfg, key, 'auto')
                pending.setdefault('safety_resets', {})[key] = {
                    'from': current, 'to': 'auto'}
            continue
        # Any non-auto value is an operator choice, including a value written by
        # an earlier benchmark. Automatic tuning may report a recommendation but
        # must not silently replace that explicit choice.
        if current not in ('', 'auto'):
            pending.setdefault('explicit_overrides', {})[key] = current
            continue
        if current != str(val).strip().lower():
            setattr(cfg, key, str(val))
            pending[key] = str(val)
    # For these two, `auto` already resolves to `on` in run.py, so writing 'on'
    # over 'auto' changes nothing except to mark the setting as drifted from its
    # default in the UI — and a benchmark that agrees with the default should
    # leave the default alone.
    for key, value in (('perf_batch_swap', rec.get('batch_swap')),
                       ('perf_nvdec', rec.get('nvdec'))):
        if not value or not hasattr(cfg, key):
            continue
        current = str(getattr(cfg, key))
        if current == value or (current == 'auto' and value == 'on'):
            continue
        setattr(cfg, key, value)
        pending[key] = value

    # The encoder, which until now was measured and then only printed — the one
    # number in the report that nothing acted on. Two different lifetimes here,
    # and conflating them would misreport one of them:
    #
    #   * the CODEC is re-read from config on every run (`_run_swap` assigns
    #     roop.globals.video_encoder from CFG), so it is live immediately;
    #   * the PRESET goes out as ROOP_ENCODER_PRESET, which run.py exports once
    #     at startup, so it needs a restart like the pools.
    enc = str(rec.get('encoder') or '').split()
    if enc:
        codec, preset = enc[0], (enc[1] if len(enc) > 1 else '')
        if codec in APP_CODECS and str(getattr(cfg, 'output_video_codec', '')) != codec:
            cfg.output_video_codec = codec
            applied['output_video_codec'] = codec
        # ROOP_ENCODER_PRESET is x264-style and `ffmpeg_writer` consults it for
        # libx264/libx265 only — NVENC's preset is a separate knob
        # (ROOP_NVENC_PRESET, p1-p7) whose p5 default is the only value measured
        # here, so there is nothing to write for it. Writing 'p5' into
        # perf_encoder_preset would fail that validator and silently become
        # 'faster'.
        if preset and codec in ('libx264', 'libx265') and \
                str(getattr(cfg, 'perf_encoder_preset', 'auto')) != preset:
            cfg.perf_encoder_preset = preset
            pending['perf_encoder_preset'] = preset

    applied['best_threads'] = rec.get('threads')
    try:
        cfg.save()
    except Exception as e:                          # noqa: BLE001
        return {'error': f'could not save config: {e}'}
    return {'applied_now': applied, 'pending_restart': pending}


# ── standalone ───────────────────────────────────────────────────────────────

def _print_report(res):
    print('\n' + '=' * 78)
    print(res.get('summary', ''))
    print('=' * 78)
    print(f"\n{'stage':34s} {'provider':9s} {'ms/call':>8s} {'x/frame':>8s} "
          f"{'ms/frame':>9s} {'pool':>5s} {'MB':>7s}")
    for s in res['stages']:
        if s['error']:
            print(f"{s['label']:34s} ERROR {s['error']}")
            continue
        print(f"{s['label'][:34]:34s} {s['provider'][:9]:9s} {s['ms_call']:8.2f} "
              f"{s['calls_per_frame']:8.2f} {s['ms_frame']:9.2f} "
              f"{s['best_n']:5d} {s['vram_per_instance_mb']:7.0f}")
        for row in s['scaling']:
            print(f"      x{row['n']}: {row['calls_s']:8.1f} calls/s "
                  f"({row['speedup']:.2f}x)  {row['free_vram_gb']:.1f} GB free")
    print('\nthread curve (frames/s of GPU+CPU frame work):')
    for mode, curve in (res.get('thread_curve') or {}).items():
        row = '  '.join(f"{k}t={v}" for k, v in sorted(curve.items(),
                                                       key=lambda kv: int(kv[0])))
        print(f"  {mode:9s} {row}")
    if res.get('provider_ab'):
        print('\nTensorRT vs CUDA:')
        for r in res['provider_ab']:
            # `cuda_ms` is None on a row whose CUDA session would not run at all,
            # which the phase records deliberately rather than dropping — a
            # `:7.2f` on it raises out of the report AFTER a full measured run.
            if r.get('cuda_ms') is None:
                print(f"  {r['stage'][:34]:34s} TRT {r['trt_ms']:7.2f}  "
                      f"CUDA       —  will not run on CUDA here"
                      + (f" ({r['error']})" if r.get('error') else ''))
                continue
            # The provider is printed as RESOLVED, not as requested. A row that
            # silently ran on CPU reads as a spectacular TensorRT win otherwise
            # (17.83x on this machine), and it is the one row that must not be
            # believed.
            label = r.get('cuda_provider') or 'CUDA'
            print(f"  {r['stage'][:34]:34s} TRT {r['trt_ms']:7.2f}  "
                  f"{label:4s} {r['cuda_ms']:7.2f}  {r['trt_speedup']:.2f}x"
                  + ('   [CUDA refused this model — CPU time, excluded from the '
                     'verdict]' if r.get('cuda_fell_back') else ''))
    if res.get('io'):
        for k in ('encode', 'decode'):
            for r in res['io'].get(k, []):
                print(f"  {k} {r['name']:20s} {r['fps']:8.1f} fps"
                      + (f"  {r['mb']:6.2f} MB" if r.get('mb') else ''))
        for note in res['io'].get('notes', []):
            print(f"  note: {note}")
    if res.get('pool_ab'):
        print('\npool check:', res['pool_ab'].get('knob'),
              res['pool_ab'].get('reason') or
              f"{res['pool_ab'].get('knee_fps')} -> {res['pool_ab'].get('wider_fps')} fps, "
              f"kept {res['pool_ab'].get('winner')}")
    tile = res.get('tile_upscale') or {}
    if tile.get('candidates'):
        print('\nFrame_Upscale tile batches:')
        for row in tile['candidates']:
            if row.get('error'):
                print(f"  batch {row['batch_size']}: rejected ({row['error']})")
            else:
                print(f"  batch {row['batch_size']}: {row['fps']:.2f} frame/s, "
                      f"{row['latency_ms']:.2f} ms, "
                      f"effective {row['effective_batch_size']}, "
                      f"max diff {row['max_abs_diff_vs_batch1']}")
        if tile.get('recommended_batch_size'):
            print(f"  recommended: {tile['recommended_batch_size']} "
                  f"({tile['recommended_fps']:.2f} frame/s)")
    for w in res.get('warnings', []):
        print('  ! ' + w)
    print('\nrecommend:', res.get('recommend'))


def _phase11_matrix_measurements(stages):
    """Translate measured enhancer stage rows into the Phase 11 schema.

    The normal benchmark measures the selected live configuration only.  Keep
    all other inventory rows pending rather than copying a result between
    models or hardware profiles; a Phase 11 run can fill them independently.
    """
    from roop.enhancer_inventory import entries

    by_label = {item['label'].lower(): item['id'] for item in entries()}
    measurements = {}
    for stage in stages:
        if not stage.label.startswith('Enhancer') or stage.error:
            continue
        label = stage.label.split('—', 1)[-1].strip()
        key = by_label.get(label.lower())
        if key is None and label.lower() == 'gpen realistic':
            try:
                size = int(os.environ.get('ROOP_GPENR_SIZE', '512') or 512)
            except (TypeError, ValueError):
                size = 512
            key = by_label.get(f'gpen realistic {size}'.lower())
        if key is None or not stage.ms_call:
            continue
        measurements[key] = {
            'FPS': round(1000.0 / max(stage.ms_call, 1e-9), 3),
            'Latency': round(stage.ms_call, 3),
            'VRAM': round(stage.vram_mb, 1),
            'CPU': None,
            'status': 'measured_inference_stage',
            'Notes': (f"selected configuration; inference-stage timing only; "
                      f"full pre/post quality and host utilization remain in "
                      f"the Phase 11 audit")
        }
    return measurements


def bootstrap_globals():
    """Load config.yaml into `roop.globals` when nothing else has.

    Every stage in the catalogue is chosen off `roop.globals.CFG` and built with
    `roop.globals.execution_providers`. Under the UI both are already populated —
    `core.run` assigns CFG and `ui.main` resolves the provider — so the benchmark
    the panel starts measures what the user renders with.

    Imported standalone, neither is set. CFG then falls back to a fresh Settings
    (its DEFAULTS: inswapper, SCRFD, no enhancer) and the provider list is empty,
    which is read as CPU. Measured here before the fix: the CLI reported
    "providers: CUDAExecutionProvider, CPUExecutionProvider (config: /)" and
    benchmarked `inswapper @128` + `SCRFD det_10g` while config.yaml selected
    tensorrt, hyperswap and RestoreFormer++ — a full report, internally
    consistent, about a configuration this machine never runs. That is the same
    trap the file's opening docstring is about, one layer up: not a fake
    workload this time, but real models that are the wrong ones.

    `CFG is None` is the signal, and the only reliable one: `execution_providers`
    is NOT empty when unresolved — `roop/globals.py` initialises it to
    `['CUDAExecutionProvider', 'CPUExecutionProvider']`, so a truthiness check
    there passes on a process that has configured nothing, which is how the CPU/CUDA
    pair above got reported as if it were a setting. CFG genuinely defaults to None
    and is assigned in exactly one place (`core.run`), so it separates "the app
    booted" from "this module was imported" cleanly.

    Returns True when it bootstrapped, False when the app had already done it.
    """
    if roop.globals.CFG:
        return False                    # the app booted; its state wins

    from settings import Settings
    from roop.core import decode_execution_providers

    # Same filename core.run uses, resolved against the app directory rather
    # than the shell's cwd — `python -m roop.bench` is legitimately run from
    # either app/ or the repo root, and a relative miss here is silent: it
    # yields a defaults Settings, which is precisely the failure above.
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'config.yaml')
    roop.globals.CFG = Settings(cfg_path if os.path.exists(cfg_path)
                                else 'config.yaml')

    # Provider resolution mirrors ui/main.py, including its TensorRT->CUDA
    # degrade: a report of TensorRT stage times from a machine without the
    # runtime is advice for a different machine. CUDA is appended behind
    # TensorRT for the same reason the app appends it — so a TRT node that
    # fails to build lands on CUDA rather than falling through to CPU.
    provider = getattr(roop.globals.CFG, 'provider', 'cuda') or 'cuda'
    providers = [provider]
    if provider == 'tensorrt':
        try:
            import tensorrt            # noqa: F401
        except ImportError:
            print('TensorRT runtime libraries not found - '
                  'falling back to CUDA provider.')
            providers = ['cuda']
        else:
            providers.append('cuda')
    roop.globals.execution_providers = decode_execution_providers(providers)

    if not roop.globals.execution_threads:
        roop.globals.execution_threads = getattr(
            roop.globals.CFG, 'max_threads', 0) or 0
    return True


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--profile', default='full', choices=sorted(PROFILES))
    ap.add_argument('--faces', type=float, default=1.0,
                    help='faces per frame the composite assumes')
    ap.add_argument('--no-apply', action='store_true',
                    help='measure without writing config.yaml')
    ap.add_argument('--target', choices=('RTX 3060', 'RTX 4070'),
                    help='physical validation target label for the report')
    args = ap.parse_args(argv)
    if args.target:
        os.environ['ROOP_VALIDATION_TARGET'] = args.target
    bootstrap_globals()

    def report(**kw):
        if kw.get('log'):
            print(kw['log'], flush=True)
        elif kw.get('status'):
            print(f"-- {kw['status']}", flush=True)

    res = run_benchmark(args.profile, args.faces, report=report,
                        apply_to_config=not args.no_apply)
    _print_report(res)
    return res


if __name__ == '__main__':
    main()
