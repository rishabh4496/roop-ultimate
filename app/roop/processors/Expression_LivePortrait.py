"""Expression restorer — put the TARGET's expression back onto the swapped face.

Why this exists
---------------
Identity swappers take a crop that carries the expression and an identity vector,
and the generator regresses toward the mean expression of its training set. The
further an expression is from that mean — a wide laugh, crying, a grimace — the
harder it is compressed. Face restorers then push further toward a clean,
neutral face prior. Nothing in that chain is trying to preserve *how open the
mouth is*, so swapped faces come out stiff.

LivePortrait's implicit-keypoint representation separates a face into a canonical
shape, a head pose, and an *expression deformation*. That last term is exactly
what the swap flattened, and it can be read off the original target frame and
re-applied.

Expression-only transfer
------------------------
The usual LivePortrait use is animating a still from a different driving video,
which needs pose transfer, relative-motion handling and a first-frame reference.
None of that applies here: the "source" (swapped crop) and the "driving"
(original target crop) are the SAME frame in the SAME aligned crop space, so the
head pose, scale and translation already match. Only the expression differs.

So instead of replacing the pose, keep it and substitute just the expression:

    x_s     = scale_s * (kp_s @ R_s + exp_s)   + t_s
    x_drive = scale_s * (kp_s @ R_s + exp_mix) + t_s
    exp_mix = exp_s + strength * (exp_d - exp_s)

which reduces to

    x_drive = x_s + scale_s * strength * (exp_d - exp_s)

The rotation cancels out of the *difference*, so the transfer cannot introduce
pose drift even if the head-pose estimate is noisy — the only thing moving is the
expression delta. strength = 0 returns x_s exactly, i.e. a no-op.

Models (~537 MB, downloaded on first use) are the FasterLivePortrait ONNX
exports, which suit this project's onnxruntime/TensorRT stack.
"""

import concurrent.futures
import contextlib
import os
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.gridsample5d import ensure_patched_model
from roop.session_pool import SessionPool, expression_pool_size
from roop.precision_policy import providers_for
from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download

_BASE = "https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/"

MODELS = {
    "appearance": {"file": "appearance_feature_extractor.onnx", "url": _BASE + "appearance_feature_extractor.onnx"},
    "motion":     {"file": "motion_extractor.onnx",             "url": _BASE + "motion_extractor.onnx"},
    "warping":    {"file": "warping_spade.onnx",                "url": _BASE + "warping_spade.onnx"},
    "stitching":  {"file": "stitching.onnx",                    "url": _BASE + "stitching.onnx"},
}

# LivePortrait's implicit keypoints are not semantic, but these index groups are
# the ones its own eye/lip retargeting modules drive, so they are what "eyes
# only" / "lips only" restrict the expression delta to.
LIP_INDICES = (6, 12, 14, 17, 19, 20)
EYE_INDICES = (11, 13, 15, 16, 18)

INPUT_SIZE = 256          # LivePortrait's native crop size
NUM_BINS = 66             # head-pose regression bins


# ── sub-stage timing (ROOP_EXPR_PROFILE=1) ───────────────────────────────────
# One restored face costs five session runs plus CPU conversion on both ends,
# and ROOP_PROFILE only reports the stage as a single 'expression' total. That
# total says the stage is expensive; it does not say WHICH of the five to
# attack, and the five are very differently sized (warping_spade is 421 MB and
# emits 512x512, motion_extractor is 112 MB and runs twice, stitching is 182 KB).
# So this keeps its own breakdown, off by default and free when off.
_EXPR_PROFILE = os.environ.get('ROOP_EXPR_PROFILE', '0') == '1'
_REPORT_EVERY = 200                     # faces between printed summaries

_stage_times = defaultdict(float)
_stage_calls = defaultdict(int)
_stage_lock = threading.Lock()
_faces_done = 0


def _say(msg):
    """Terminal output that will not shred the progress bar.

    Everything this module prints happens INSIDE a render: the restorer is built
    lazily on the first face (ProcessMgr._expression_restorer), and the failure
    paths fire per face. A bare print() lands in the middle of tqdm's rewritten
    line and terminates it, turning one bar into a line per frame. Imported
    lazily so the processor stays importable without ProcessMgr's runtime, and
    degrading to print() if that ever fails — a status line has no business
    killing a render."""
    try:
        from roop.procmgr_runtime import bar_write
        bar_write(msg)
    except Exception:
        print(msg)


@contextlib.contextmanager
def _substage(name):
    if not _EXPR_PROFILE:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _stage_lock:
            _stage_times[name] += dt
            _stage_calls[name] += 1


def _maybe_report():
    """Print the breakdown every _REPORT_EVERY faces.

    Printed through bar_write, never print(): a bare print() mid-render
    terminates tqdm's line and shreds the progress bar.
    """
    global _faces_done
    if not _EXPR_PROFILE:
        return
    with _stage_lock:
        _faces_done += 1
        if _faces_done % _REPORT_EVERY:
            return
        snapshot = [(k, _stage_times[k], _stage_calls[k]) for k in _stage_times]
        faces = _faces_done
    try:
        from roop.procmgr_runtime import bar_write
    except Exception:
        return
    total = sum(t for _, t, _ in snapshot) or 1.0
    lines = [f"[Expression] per-stage cost over {faces} restored faces "
             f"(summed across worker threads):"]
    for k, t, c in sorted(snapshot, key=lambda x: -x[1]):
        lines.append(f"    {k:12s} {t:7.2f}s {100 * t / total:5.1f}%  "
                     f"{c:6d} calls  {1000 * t / max(c, 1):7.2f} ms/call")
    if _parallel_mode():
        lines.append("    (appearance/motion overlap, so their times sum above "
                     "the stage's wall clock)")
    bar_write("\n".join(lines))


# ── overlapping the independent calls (ROOP_EXPR_PARALLEL) ───────────────────
# appearance(swapped), motion(swapped) and motion(target) read nothing from each
# other — only stitching and warping depend on their results. Run sequentially
# they cost the sum; issued together they cost roughly the longest of them,
# because none of these three saturates a modern GPU at batch 1 (256x256 in,
# and the appearance extractor is a 3.3 MB model).
#
#   0  sequential — the original order.
#   1  overlap appearance with motion(swapped). The two are DIFFERENT models
#      with different sessions, hence different TensorRT contexts, so running
#      them from two threads is safe (see session_pool's module docstring); the
#      two motion calls still share one session and stay in series. Costs no
#      extra VRAM at all, which is why it is the default.
#   2  also build a SECOND motion session so motion(target) overlaps too, making
#      the front half cost max(...) instead of a sum. Costs one extra
#      motion_extractor engine (~112 MB model), so it is opt-in — on a card
#      already running the swapper and detect/mask pools there may be no room.
#
# Keep the expectations honest: measured on an RTX 4070 this is worth 1-2%, not
# the 14% the arithmetic suggests. The three calls are only 7.1 ms of a 34 ms
# face (appearance 2.2, motion 2.5 + 2.4) and they do not overlap cleanly,
# because warping_spade has already left the GPU with little idle capacity to
# fill. It is free and it is bit-identical, so it stays on by default — but the
# stage's cost is warping, and this does not touch warping.
def _parallel_mode() -> int:
    raw = os.environ.get('ROOP_EXPR_PARALLEL', '1')
    try:
        return max(0, min(2, int(raw)))
    except ValueError:
        return 1


# The appearance extractor's output is a (1,32,16,64,64) float32 volume — 8 MB —
# and warping_spade is its only consumer. Left to onnxruntime's default binding
# it is copied device->host into a fresh numpy array and then straight back
# host->device, a 16 MB PCIe round trip per restored face for data that never
# leaves the pipeline. io_binding hands the second session the first session's
# device pointer instead. Bit-identical: same tensor, one less pair of copies.
# Also small: ~1% on a 4070, since 16 MB over PCIe 4.0 is well under a
# millisecond and the copies were already overlapping compute.
# ROOP_EXPR_IOBINDING=0 forces the plain path (it is also disabled automatically
# if either session is not on a GPU, or if the chained run ever throws).
def _iobinding_enabled() -> bool:
    return os.environ.get('ROOP_EXPR_IOBINDING', '1') != '0'


def _provider_names(providers):
    return [str(p[0] if isinstance(p, (tuple, list)) else p).lower() for p in providers]


def _has_gpu_provider(providers):
    """True if anything in the list can execute the rewritten graph on a GPU."""
    return any('tensorrt' in n or 'cuda' in n for n in _provider_names(providers))


def warping_providers(providers):
    """Providers that can execute the STOCK warping module.

    This is the fallback path. Normally roop.gridsample5d rewrites the two 5-D
    GridSample nodes into ops every backend can build, and the session just uses
    the standard provider list; this function is what remains for when that
    rewrite is disabled or fails.

    warping_spade warps a 5-D feature volume (1,32,16,64,64) with GridSample.
    onnxruntime's **CUDA** GridSample kernel is 4-D only, so with CUDA in the
    list the node is assigned to it and the run dies with "Only 4-D tensor is
    supported" — measured, not assumed. TensorRT executes it fine, and the CPU
    kernel handles 5-D but takes ~1.9 s per face, which is unusable for video and
    only tolerable for a single preview frame.

    So: keep TensorRT if present, otherwise fall back to CPU, and never leave
    CUDA to claim the node. The other three models are ordinary 4-D networks and
    keep the normal provider list.

    (FasterLivePortrait also ships warping_spade-fix.onnx, which replaces the
    node with a custom `GridSample3D` op. That needs a plugin library this
    project does not carry — the model fails to even load — so it is not used.)
    """
    kept = [p for p in providers
            if 'cuda' not in str(p[0] if isinstance(p, (tuple, list)) else p).lower()]
    if not any('tensorrt' in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
               for p in kept):
        return ['CPUExecutionProvider']
    if not any('cpu' in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
               for p in kept):
        kept.append('CPUExecutionProvider')
    return kept


@contextlib.contextmanager
def _quiet_trt_parser():
    """Silence the TensorRT parser errors this model provokes on purpose.

    TensorRT cannot build the two GridSample nodes either — its builder asserts
    `addGridSample: input.getDimensions().nbDims == 4` and the parser reports
    INVALID_NODE for /dense_motion_network/GridSample and /GridSample. That looks
    exactly like a crash in the terminal but is NOT a failure: onnxruntime
    partitions those two nodes to the CPU provider and runs everything else on
    TensorRT. Measured here: 0.25 s warm for the mixed partition versus 1.93 s on
    CPU alone, so the GPU still does nearly all the work.

    SessionOptions.log_severity_level does NOT suppress them — they come from the
    TensorRT logger bridged through onnxruntime's GLOBAL logger, not the
    session's — so the global severity has to be raised instead. The window is
    kept as narrow as possible: session construction only, which is where the
    parse happens, and it is restored afterwards even on failure. onnxruntime
    exposes no getter for the current severity, so it is restored to ORT's own
    default of 2 (warning).
    """
    onnxruntime.set_default_logger_severity(4)      # fatal only
    try:
        yield
    finally:
        onnxruntime.set_default_logger_severity(2)  # ORT's default: warning


# ── pure geometry (no models, unit-tested) ───────────────────────────────────

def headpose_pred_to_degree(pred):
    """The pose heads emit a 66-bin distribution, not an angle: expected bin
    index, rescaled to degrees. Matches LivePortrait's own conversion."""
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, NUM_BINS)
    pred = pred - pred.max(axis=1, keepdims=True)          # stabilised softmax
    e = np.exp(pred)
    prob = e / e.sum(axis=1, keepdims=True)
    idx = np.arange(NUM_BINS, dtype=np.float32)
    return (prob * idx).sum(axis=1) * 3.0 - 97.5


def get_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """Row-vector rotation matrix, so keypoints compose as `kp @ R`."""
    p, y, r = (np.radians(np.asarray(a, dtype=np.float32).reshape(-1))
               for a in (pitch_deg, yaw_deg, roll_deg))
    n = p.shape[0]
    zeros, ones = np.zeros(n, np.float32), np.ones(n, np.float32)
    rot_x = np.stack([ones, zeros, zeros,
                      zeros, np.cos(p), -np.sin(p),
                      zeros, np.sin(p), np.cos(p)], 1).reshape(n, 3, 3)
    rot_y = np.stack([np.cos(y), zeros, np.sin(y),
                      zeros, ones, zeros,
                      -np.sin(y), zeros, np.cos(y)], 1).reshape(n, 3, 3)
    rot_z = np.stack([np.cos(r), -np.sin(r), zeros,
                      np.sin(r), np.cos(r), zeros,
                      zeros, zeros, ones], 1).reshape(n, 3, 3)
    return np.transpose(rot_z @ rot_y @ rot_x, (0, 2, 1))


def transform_keypoint(kp, exp, scale, t, rot):
    """kp/exp: (B,N,3); scale: (B,1); t: (B,3); rot: (B,3,3) -> (B,N,3).
    The z translation is dropped, as in LivePortrait."""
    kp = np.asarray(kp, np.float32).reshape(kp.shape[0], -1, 3)
    exp = np.asarray(exp, np.float32).reshape(kp.shape[0], -1, 3)
    t = np.asarray(t, np.float32).reshape(kp.shape[0], 3).copy()
    t[:, 2] = 0.0
    out = (kp @ rot) + exp
    out = out * np.asarray(scale, np.float32).reshape(-1, 1, 1)
    return out + t[:, None, :]


def blend_expression(exp_source, exp_driving, strength, region="all"):
    """Move the source expression toward the driving one by `strength`.

    strength 0 returns the source expression unchanged (bit-exact no-op), 1.0
    adopts the target's expression fully, and values above 1 exaggerate past it
    — the equivalent of a micro-expression boost, for expressions the swap
    compressed rather than removed."""
    src = np.asarray(exp_source, np.float32)
    dst = np.asarray(exp_driving, np.float32)
    if strength == 0.0:
        return src.copy()
    delta = (dst - src) * float(strength)
    if region != "all":
        idx = LIP_INDICES if region == "lips" else EYE_INDICES
        gate = np.zeros_like(delta)
        flat = delta.reshape(delta.shape[0], -1, 3)
        gflat = gate.reshape(gate.shape[0], -1, 3)
        for i in idx:
            if i < flat.shape[1]:
                gflat[:, i, :] = flat[:, i, :]
        delta = gate
    return src + delta


def driving_keypoints(x_source, scale_source, exp_source, exp_driving,
                      strength, region="all"):
    """x_s + scale * (exp_mix - exp_s), the reduced form derived in the module
    docstring. Rotation and translation cancel, so this cannot move the head."""
    exp_mix = blend_expression(exp_source, exp_driving, strength, region)
    delta = (exp_mix - np.asarray(exp_source, np.float32))
    delta = delta.reshape(x_source.shape)
    return x_source + delta * np.asarray(scale_source, np.float32).reshape(-1, 1, 1)


def concat_feat(kp_source, kp_driving):
    """Stitching input: both keypoint sets flattened and concatenated."""
    b = kp_source.shape[0]
    return np.concatenate([kp_source.reshape(b, -1), kp_driving.reshape(b, -1)],
                          axis=1).astype(np.float32)


def _to_input(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE),
                                  interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None]


class Expression_LivePortrait:
    plugin_options: dict = None
    processorname = 'expression_liveportrait'
    type = 'expression'

    def __init__(self):
        self.sessions = {}
        self.devicename = 'cpu'
        self._lock = threading.Lock()
        self.pool = None
        self._exec = None          # overlaps the independent front-half calls
        self._chain = False        # feature_3d stays on the device (io_binding)
        self._device_id = 0

    @property
    def pooled(self) -> bool:
        """Whether this restorer owns independent contexts per worker thread."""
        return self.pool is not None

    @property
    def self_excluding(self) -> bool:
        """Whether the stage can skip ProcessMgr's GLOBAL GPU lock.

        It can. The restorer already excludes concurrent use of its own
        TensorRT contexts — a slot lease when pooled, self._lock when not — so
        no context is ever entered twice at once, which is the only thing the
        global lock is needed for. What the global lock adds on top is
        serialisation against UNRELATED stages, and that is not a safety
        property: distinct contexts running in parallel is exactly what the
        swapper and detect/mask pools already do all run long (see
        session_pool's module docstring).

        Holding it was therefore pure cost. Expression is the most expensive
        stage in a run that enables it, and every millisecond it held the global
        lock also stalled the unpooled enhancers (GFPGAN/GPEN/CodeFormer), which
        do still take it — so an enhancer thread and an expression thread, doing
        unrelated work on separate contexts, were taking turns for no reason.

        This does NOT make two expression calls concurrent; that still needs
        ROOP_EXPR_POOL. It stops the stage from blocking, and being blocked by,
        the rest of the pipeline. ROOP_EXPR_GLOBAL_LOCK=1 restores the old
        serialisation if a card ever misbehaves.
        """
        return os.environ.get('ROOP_EXPR_GLOBAL_LOCK', '0') != '1'

    @contextlib.contextmanager
    def _leased(self):
        """The session set to use for one call, and the right exclusion for it.

        Pooled: take a slot from the queue — no global serialisation, because no
        two threads can hold the same slot. Unpooled: the single set behind its
        own lock, since onnxruntime sessions are not thread-safe.
        """
        if self.pool is not None:
            with self.pool.lease() as sessions:
                yield sessions
        else:
            with self._lock:
                yield self.sessions

    # ── lifecycle ────────────────────────────────────────────────────────────

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None and \
                self.plugin_options.get("devicename") != plugin_options.get("devicename"):
            self.Release()
        self.plugin_options = plugin_options
        self.devicename = str(plugin_options.get("devicename", "cpu")).replace('mps', 'cpu')
        if self.sessions:
            return
        model_dir = resolve_relative_path('../models/liveportrait')
        os.makedirs(model_dir, exist_ok=True)
        conditional_download(model_dir, [m["url"] for m in MODELS.values()])
        providers = roop.globals.execution_providers

        # The stock warping module cannot run entirely on the GPU: its two 5-D
        # GridSample nodes are rejected by TensorRT and by onnxruntime's CUDA
        # kernel alike, so they land on the CPU and split the rest into nine
        # separate TensorRT subgraphs. Rewriting them into TRT-native gathers
        # collapses that to one engine: measured 237.8ms -> 34.0ms per restored
        # face. ROOP_EXPR_PATCH_GRIDSAMPLE=0 keeps the stock model and its split.
        # Only worth it on a GPU: measured on CPU the expansion is ~9% SLOWER
        # than onnxruntime's own kernel (2.66s vs 2.44s), because eight gathers
        # beat one fused kernel only when there are cores to spread them over.
        warp_path = os.path.join(model_dir, MODELS["warping"]["file"])
        patched_path = warp_path
        if os.environ.get('ROOP_EXPR_PATCH_GRIDSAMPLE', '1') != '0' and \
                _has_gpu_provider(providers):
            patched_path = ensure_patched_model(warp_path)
        is_patched = patched_path != warp_path

        if is_patched:
            # Nothing 5-D survives the rewrite, so the normal provider list
            # applies — including CUDA, which the stock model has to exclude.
            warp_providers = providers
            _say("[Expression] Warping module: fully on the GPU — the two 5-D "
                  "GridSample nodes were rewritten into TRT-native ops, so there "
                  "is no CPU partition.")
        else:
            warp_providers = warping_providers(providers)
            if warp_providers == ['CPUExecutionProvider']:
                _say("[Expression] No TensorRT provider — the warping module runs on CPU "
                      "(~1.9s per face). Fine for a preview, far too slow for video.")
            else:
                _say("[Expression] Warping module: TensorRT with the two GridSample nodes "
                      "on CPU (~0.25s per face). TensorRT parser warnings about "
                      "'addGridSample ... nbDims == 4' are expected and silenced.")

        parallel = _parallel_mode()

        from roop.utilities import get_onnx_session_options
        sess_opts = get_onnx_session_options()

        def build_sessions(_slot=0):
            built = {}
            for key, spec in MODELS.items():
                path = os.path.join(model_dir, spec["file"])
                if key == "warping":
                    # The parser errors only exist on the unpatched path; keep
                    # real errors visible once the rewrite removed their cause.
                    ctx = contextlib.nullcontext() if is_patched else _quiet_trt_parser()
                    session_providers, _precision = providers_for(
                        f'liveportrait:{key}', warp_providers, patched_path)
                    with ctx:
                        built[key] = onnxruntime.InferenceSession(
                            patched_path, sess_opts, providers=session_providers)
                else:
                    session_providers, _precision = providers_for(
                        f'liveportrait:{key}', providers, path)
                    built[key] = onnxruntime.InferenceSession(
                        path, sess_opts, providers=session_providers)
            if parallel >= 2:
                # A second, independent context for the same model — the two
                # motion calls can only overlap if they stop sharing one.
                motion_path = os.path.join(model_dir, MODELS["motion"]["file"])
                motion_providers, _precision = providers_for(
                    'liveportrait:motion2', providers, motion_path)
                built["motion2"] = onnxruntime.InferenceSession(
                    motion_path, sess_opts, providers=motion_providers)
            return built

        # One set of sessions per pool slot. Each slot owns its own TensorRT
        # engine + execution context, which is what makes concurrent use safe —
        # the contexts themselves are not thread-safe, only distinct ones are.
        size = expression_pool_size(
            model_key='expression:liveportrait', input_shape=(1, 3, 256, 256))
        if size >= 2:
            _say(f"[Expression] Session pool: {size} instances "
                  f"(~{size}x the VRAM of one restorer; ROOP_EXPR_POOL={size}).")
            self.pool = SessionPool(build_sessions, size,
                                    model_key='expression:liveportrait',
                                    input_shape=(1, 3, 256, 256))
            self.sessions = self.pool._items[0]
        else:
            self.sessions = build_sessions()

        self._device_id = int(getattr(roop.globals, 'cuda_device_id', 0) or 0)

        # Chaining the 8 MB feature volume needs BOTH ends on the same device;
        # the warping session can legitimately be CPU-only (no TensorRT and no
        # GridSample rewrite), in which case there is no device copy to save.
        self._chain = bool(
            _iobinding_enabled()
            and _has_gpu_provider(self.sessions["appearance"].get_providers())
            and _has_gpu_provider(self.sessions["warping"].get_providers()))
        if self._chain:
            _say("[Expression] feature_3d stays on the GPU between the "
                  "appearance extractor and the warping module "
                  "(ROOP_EXPR_IOBINDING=0 to disable).")

        if parallel >= 1:
            # Two helper threads per concurrent caller: appearance and, at
            # level 2, motion(target). The caller thread runs motion(swapped)
            # itself, so it is never idle waiting on its own work.
            self._exec = concurrent.futures.ThreadPoolExecutor(
                max_workers=2 * max(1, size),
                thread_name_prefix="expr")
            _say(f"[Expression] Overlapping the independent front-half calls "
                  f"(ROOP_EXPR_PARALLEL={parallel}"
                  f"{'; second motion session built' if parallel >= 2 else ''}).")

    def Release(self):
        if self._exec is not None:
            self._exec.shutdown(wait=True)
            self._exec = None
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        self.sessions = {}
        self.plugin_options = None

    # ── inference helpers ────────────────────────────────────────────────────

    def _run(self, sessions, key, feeds):
        """Bind inputs BY NAME, never by position.

        warping_spade declares its inputs as (feature_3d, kp_driving, kp_source)
        — driving before source, which is the reverse of how they read. Zipping a
        positional list against the session's input order silently swapped them,
        applying the expression backwards. Naming them makes that class of
        mistake impossible and self-documenting at the call site."""
        sess = sessions[key]
        expected = {i.name for i in sess.get_inputs()}
        missing = expected - set(feeds)
        if missing:
            raise KeyError(f"{key}: model expects inputs {sorted(expected)}, "
                           f"missing {sorted(missing)}")
        return sess.run(None, {k: v for k, v in feeds.items() if k in expected})

    def _motion(self, sessions, img, key="motion"):
        """-> (pitch, yaw, roll, t, exp, scale, kp).

        Verified against the FasterLivePortrait export: outputs are declared in
        exactly this order, with exp and kp flattened to (1, 63) = 21 points.

        `key` selects which motion session to use: the second one exists only at
        ROOP_EXPR_PARALLEL=2, and holds identical weights, so which one produced
        a result cannot change it."""
        out = self._run(sessions, key, {"img": img})
        pitch, yaw, roll, t, exp, scale, kp = out[:7]
        return (headpose_pred_to_degree(pitch), headpose_pred_to_degree(yaw),
                headpose_pred_to_degree(roll),
                np.asarray(t, np.float32).reshape(1, 3),
                np.asarray(exp, np.float32).reshape(1, -1, 3),
                np.asarray(scale, np.float32).reshape(1, 1),
                np.asarray(kp, np.float32).reshape(1, -1, 3))

    def _appearance(self, sessions, src_in):
        """-> feature_3d, as a numpy array or as an on-device OrtValue.

        The chained form leaves the 8 MB volume in GPU memory for warping to
        pick up. synchronize_outputs() is not optional: the two sessions run on
        different CUDA streams, and without it warping can start reading the
        buffer before the extractor's stream has finished writing it — a race
        that would show up as corrupted faces, not as an error."""
        if not self._chain:
            return self._run(sessions, "appearance", {"img": src_in})[0]
        sess = sessions["appearance"]
        io = sess.io_binding()
        io.bind_cpu_input("img", src_in)
        io.bind_output(sess.get_outputs()[0].name, "cuda", self._device_id)
        sess.run_with_iobinding(io)
        io.synchronize_outputs()
        return io.get_outputs()[0]

    def _warp(self, sessions, feature, x_s, x_d):
        """The final generator pass. Accepts feature_3d in either form."""
        if isinstance(feature, np.ndarray):
            return self._run(sessions, "warping", {"feature_3d": feature,
                                                   "kp_source": x_s,
                                                   "kp_driving": x_d})[0]
        sess = sessions["warping"]
        expected = {i.name for i in sess.get_inputs()}
        missing = {"feature_3d", "kp_source", "kp_driving"} - expected
        if missing:
            raise KeyError(f"warping: model expects inputs {sorted(expected)}, "
                           f"missing {sorted(missing)}")
        io = sess.io_binding()
        io.bind_ortvalue_input("feature_3d", feature)
        # bind_cpu_input takes the array as-is; the keypoint maths produces
        # views and slices, and a non-contiguous buffer would be read wrong.
        io.bind_cpu_input("kp_source", np.ascontiguousarray(x_s, np.float32))
        io.bind_cpu_input("kp_driving", np.ascontiguousarray(x_d, np.float32))
        io.bind_output(sess.get_outputs()[0].name, "cpu")
        sess.run_with_iobinding(io)
        return io.copy_outputs_to_cpu()[0]

    def _front_half(self, sessions, src_in, drv_in):
        """appearance(swapped), motion(swapped), motion(target).

        Mutually independent, so they are issued together when
        ROOP_EXPR_PARALLEL allows it. Every future is joined before returning,
        including on the error path: leaving a helper thread inside a session
        after the lease has gone back to the pool would hand that session's
        TensorRT context to a second thread, which is exactly the corruption the
        pool exists to prevent."""
        if self._exec is None:
            with _substage('appearance'):
                feature = self._appearance(sessions, src_in)
            with _substage('motion_src'):
                m_s = self._motion(sessions, src_in)
            with _substage('motion_drv'):
                m_d = self._motion(sessions, drv_in)
            return feature, m_s, m_d

        def _app():
            with _substage('appearance'):
                return self._appearance(sessions, src_in)

        def _drv():
            with _substage('motion_drv'):
                return self._motion(sessions, drv_in, key="motion2")

        futures = []
        try:
            f_app = self._exec.submit(_app)
            futures.append(f_app)
            f_drv = None
            if "motion2" in sessions:
                f_drv = self._exec.submit(_drv)
                futures.append(f_drv)
            with _substage('motion_src'):
                m_s = self._motion(sessions, src_in)
            if f_drv is not None:
                m_d = f_drv.result()
            else:
                with _substage('motion_drv'):
                    m_d = self._motion(sessions, drv_in)
            return f_app.result(), m_s, m_d
        finally:
            concurrent.futures.wait(futures)

    # ── main entry ───────────────────────────────────────────────────────────

    # The stage is split into prepare / infer / finish so the caller can hold the
    # global GPU lock around `infer` ALONE. Only the five session runs need that
    # lock; the crop->tensor conversion and the 512x512 float->uint8->BGR->resize
    # on the way back are pure CPU, and while they sat inside the lock every other
    # worker thread queued behind them for nothing. The swap path already splits
    # this way (prepare_crop_frame / p.Run / normalize_swap_frame in ProcessMgr).
    # Run() below still composes all three, so single-shot callers are unchanged.

    def prepare(self, swapped_bgr: Frame, driving_bgr: Frame):
        """Both crops -> network inputs. CPU only; safe outside the GPU lock."""
        if swapped_bgr is None or driving_bgr is None:
            return None
        with _substage('cpu_pre'):
            return _to_input(swapped_bgr), _to_input(driving_bgr)

    def infer(self, prepared, strength: float = 1.0, region: str = "all",
              use_stitching: bool = True):
        """The five session runs. Returns the raw network output, or None.

        This is the only part that touches the GPU, so it is the only part the
        caller has to serialise."""
        if prepared is None or strength == 0.0:
            return None
        try:
            return self._infer_once(prepared, strength, region, use_stitching)
        except Exception as e:
            # io_binding is the one part here that depends on the provider
            # honouring device-to-device chaining. If it ever throws, drop to the
            # plain path for the rest of the session and retry this face rather
            # than losing its expression — the fallback is what the code did
            # before chaining existed, so it cannot fail for the same reason.
            if self._chain:
                self._chain = False
                _say(f"[Expression] device-chained feature_3d failed ({e}); "
                      f"falling back to host round-trip for the rest of the run.")
                try:
                    return self._infer_once(prepared, strength, region, use_stitching)
                except Exception as e2:
                    e = e2
            _say(f"[Expression] LivePortrait restore failed: {e}")
            return None

    def _infer_once(self, prepared, strength, region, use_stitching):
        src_in, drv_in = prepared
        with self._leased() as sessions:
            feature, m_s, m_d = self._front_half(sessions, src_in, drv_in)
            p_s, y_s, r_s, t_s, exp_s, scale_s, kp_s = m_s
            exp_d = m_d[4]

            rot_s = get_rotation_matrix(p_s, y_s, r_s)
            x_s = transform_keypoint(kp_s, exp_s, scale_s, t_s, rot_s)
            x_d = driving_keypoints(x_s, scale_s, exp_s, exp_d, strength, region)

            if use_stitching and "stitching" in sessions:
                with _substage('stitching'):
                    delta = np.asarray(
                        self._run(sessions, "stitching",
                                  {"input": concat_feat(x_s, x_d)})[0],
                        np.float32)
                n = x_d.shape[1]
                if delta.size >= n * 3 + 2:
                    flat = delta.reshape(-1)
                    x_d = x_d + flat[:n * 3].reshape(1, n, 3)
                    x_d[:, :, :2] += flat[n * 3:n * 3 + 2].reshape(1, 1, 2)

            with _substage('warping'):
                return self._warp(sessions, feature, x_s, x_d)

    def finish(self, raw, swapped_bgr: Frame) -> Frame:
        """Network output -> BGR crop the size of `swapped_bgr`. CPU only.

        `raw` of None means infer() declined or failed, so the untouched input is
        returned: an expression tweak must never cost a frame."""
        if raw is None:
            return swapped_bgr
        try:
            with _substage('cpu_post'):
                h, w = swapped_bgr.shape[:2]
                img = np.asarray(raw, np.float32)
                if img.ndim == 4:
                    img = img[0]
                if img.shape[0] in (1, 3):            # CHW -> HWC
                    img = np.transpose(img, (1, 2, 0))
                if not np.isfinite(img).all():
                    return swapped_bgr
                img = np.clip(img, 0.0, 1.0) * 255.0
                bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                if bgr.shape[:2] != (h, w):
                    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)
            return bgr
        except Exception as e:
            _say(f"[Expression] LivePortrait restore failed: {e}")
            return swapped_bgr
        finally:
            _maybe_report()

    def Run(self, swapped_bgr: Frame, driving_bgr: Frame, strength: float = 1.0,
            region: str = "all", use_stitching: bool = True) -> Frame:
        """Re-impose `driving_bgr`'s expression onto `swapped_bgr`.

        Both must be the SAME aligned crop of the same frame — the swapped face
        and the original target face. Returns a crop of the input's size. Any
        failure returns the input unchanged: an expression tweak must never cost
        a frame."""
        if strength == 0.0 or swapped_bgr is None or driving_bgr is None:
            return swapped_bgr
        prepared = self.prepare(swapped_bgr, driving_bgr)
        raw = self.infer(prepared, strength, region, use_stitching)
        return self.finish(raw, swapped_bgr)
