import os
import threading

import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool


# ── Per-model swap contract ───────────────────────────────────────────────────
# Every swapper here consumes the buffalo_l / w600k_r50 ArcFace identity
# embedding, but each differs in resolution, alignment template, input
# normalization, identity projection and output range. ProcessMgr reads the
# published attributes (model_output_size / model_mean / model_standard_deviation
# / model_denormalize / model_template) to drive align/normalize. Model files
# download lazily into app/models/ on the first run that selects them.
#
# "embedding" selects how the 512-d identity vector is prepared (specs match
# FaceFusion's prepare_source_embedding):
#   normed_emap    — normed_embedding @ emap, re-normalized (inswapper family)
#   normed         — normed_embedding used directly (hyperswap)
#   converted_raw  — RAW embedding through the crossface converter, no norm (ghost)
#   converted_norm — RAW embedding through the converter, then normalized
#                    (simswap / hififace)
# "template" is the 5-point alignment the model was trained on ('arcface' =
# the existing insightface alignment; others live in face_util.WARP_TEMPLATES).
_FF30 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
_FF31 = "https://huggingface.co/facefusion/models-3.1.0/resolve/main/"
_FF33 = "https://huggingface.co/facefusion/models-3.3.0/resolve/main/"
_FF34 = "https://huggingface.co/facefusion/models-3.4.0/resolve/main/"

SWAP_MODELS = {
    # inswapper_128 — the original. arcface align, [0,1] input, identity =
    # normed_embedding @ emap, [0,1] output.
    "inswapper": {
        "file": "inswapper_128.onnx",
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
        "output_size": 128,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "normed_emap",
        "template": "arcface",
    },
    # ReSwapper 256 — open reproduction of inswapper at 2x resolution. Same
    # identity pipeline (emap), near drop-in.
    "reswapper": {
        "file": "reswapper_256.onnx",
        "url": "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/reswapper_256.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "normed_emap",
        "template": "arcface",
    },
    # HyperSwap 1a/1b/1c (FaceFusion) — 256px. [-1,1] input (mean/std 0.5),
    # identity = normed_embedding directly (NO emap), de-normalized output.
    # The model emits (image, mask); we use the image output. The three
    # checkpoints trade identity likeness vs blending differently — offered
    # side by side for A/B.
    "hyperswap": {
        "file": "hyperswap_1a_256.onnx",
        "url": _FF33 + "hyperswap_1a_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "normed",
        "template": "arcface",
    },
    "hyperswap_1b": {
        "file": "hyperswap_1b_256.onnx",
        "url": _FF33 + "hyperswap_1b_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "normed",
        "template": "arcface",
    },
    "hyperswap_1c": {
        "file": "hyperswap_1c_256.onnx",
        "url": _FF33 + "hyperswap_1c_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "normed",
        "template": "arcface",
    },
    # Ghost 1/2/3 (sberbank GHOST, FaceFusion export) — 256px, arcface_112_v1
    # alignment, [-1,1] in/out, identity = RAW embedding through the crossface
    # converter (un-normalized). Different generator depths; 3 = newest.
    "ghost_1": {
        "file": "ghost_1_256.onnx",
        "url": _FF30 + "ghost_1_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_raw",
        "template": "arcface_112_v1",
        "converter_file": "crossface_ghost.onnx",
        "converter_url": _FF34 + "crossface_ghost.onnx",
    },
    "ghost_2": {
        "file": "ghost_2_256.onnx",
        "url": _FF30 + "ghost_2_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_raw",
        "template": "arcface_112_v1",
        "converter_file": "crossface_ghost.onnx",
        "converter_url": _FF34 + "crossface_ghost.onnx",
    },
    "ghost_3": {
        "file": "ghost_3_256.onnx",
        "url": _FF30 + "ghost_3_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_raw",
        "template": "arcface_112_v1",
        "converter_file": "crossface_ghost.onnx",
        "converter_url": _FF34 + "crossface_ghost.onnx",
    },
    # SimSwap 256 — arcface_112_v1 alignment, ImageNet mean/std input, [0,1]
    # output (no denorm), identity = converted + normalized embedding.
    "simswap": {
        "file": "simswap_256.onnx",
        "url": _FF30 + "simswap_256.onnx",
        "output_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "standard_deviation": [0.229, 0.224, 0.225],
        "denormalize": False,
        "embedding": "converted_norm",
        "template": "arcface_112_v1",
        "converter_file": "crossface_simswap.onnx",
        "converter_url": _FF34 + "crossface_simswap.onnx",
    },
    # SimSwap 512 (unofficial) — native 512px decoder, plain [0,1] in/out.
    "simswap_512": {
        "file": "simswap_unofficial_512.onnx",
        "url": _FF30 + "simswap_unofficial_512.onnx",
        "output_size": 512,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "converted_norm",
        "template": "arcface_112_v1",
        "converter_file": "crossface_simswap.onnx",
        "converter_url": _FF34 + "crossface_simswap.onnx",
    },
    # BlendSwap 256 — image-source swapper (NOT embedding-based). The source is
    # an aligned face IMAGE crop (arcface_112_v2 @ 112), the target crop uses
    # the ffhq_512 template. [0,1] in/out. Model inputs named 'source'/'target'.
    "blendswap": {
        "file": "blendswap_256.onnx",
        "url": _FF30 + "blendswap_256.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "image",
        "template": "ffhq_512",
        "source_crop_key": "_src_crop_arcface_112_v2",
    },
    # UniFace 256 — image-source swapper. Source = ffhq_512 @ 256 image crop,
    # target = ffhq_512 template, [-1,1] in, de-normalized out.
    "uniface": {
        "file": "uniface_256.onnx",
        "url": _FF30 + "uniface_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "image",
        "template": "ffhq_512",
        "source_crop_key": "_src_crop_ffhq_256",
    },
    # HifiFace (unofficial) — 256px, mtcnn_512 alignment, [-1,1] in/out,
    # identity = converted + normalized embedding.
    "hififace": {
        "file": "hififace_unofficial_256.onnx",
        "url": _FF31 + "hififace_unofficial_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "converted_norm",
        "template": "mtcnn_512",
        "converter_file": "crossface_hififace.onnx",
        "converter_url": _FF34 + "crossface_hififace.onnx",
        # See "verify_tol" below. hififace's clean swaps sit further from the
        # outcome guard's threshold than any other model's here, so it can
        # afford a tighter one.
        "verify_tol": 0.65,
    },
    # RealSwap — hyperswap and hififace as ONE swapper. `secondary` names a
    # second model from this same table; the processor loads both and mixes
    # their outputs (see _mix_outputs). The published contract is the PRIMARY's,
    # so ProcessMgr aligns, normalizes and de-normalizes exactly as it would for
    # hyperswap alone, and the secondary is re-warped into that crop space
    # internally. The pairing is cheap precisely because these two agree on
    # everything but alignment and identity projection: both are 256px, both
    # take [-1,1] in and out, and both emit their own face mask.
    "realswap": {
        "file": "hyperswap_1a_256.onnx",
        "url": _FF33 + "hyperswap_1a_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "normed",
        "template": "arcface",
        "secondary": "hififace",
    },
}


# ── Per-model outcome-guard tolerance ────────────────────────────────────────
# face_util.swap_moved_the_face rejects a swap that put the face somewhere the
# plate's face was not, by measuring keypoint displacement in interocular units.
# Its threshold (face_util.SWAP_MOVED_TOL = 1.0) is one number for every model,
# and that is one number too few: how far a CLEAN swap moves the keypoints is a
# property of the swapper. Measured over a 107-frame full-turn sweep, with the
# guard disabled so every frame's reading survives:
#
#                    clean swaps          frames the swap wrecks
#   hififace         max 0.42             from 0.61
#   hyperswap        max 0.79             from 0.79
#   inswapper        max 0.79             from 0.60   (distributions OVERLAP)
#
# hififace's clean band stops at 0.42 where the others run to 0.79, so 1.0 sits
# 2.4x above its worst honest frame and lets three wrecked ones through.
# hyperswap and inswapper are left alone deliberately: hyperswap's two bands are
# 0.004 apart and inswapper's overlap outright, so no threshold separates them
# and lowering either would start discarding real profiles to catch nothing.
#
# 0.65 is not a clean separator either — inside hififace's own band the readings
# interleave (0.61 wrecked, 0.71 clean, 0.88 wrecked) — so it is a stated trade,
# not a fitted boundary: it discards two frames that paint a face onto the back
# of a turned head, and costs one legitimate 88-degree profile, which reverts to
# the plate. That is the same trade the guard itself was introduced under.
#
# CALIBRATED ON ONE CLIP (a synthetic head, one source identity, 3 frames in the
# decision band). Re-measure on real footage before trusting the exact value;
# the mechanism is the durable part, the constant is not.
def verify_tol_for(swap_processor):
    """The outcome guard's tolerance for the loaded swap model, or None to use
    the global default. Reads the published attribute rather than the spec so a
    processor that never loaded a model cannot force a threshold."""
    return getattr(swap_processor, 'model_verify_tol', None)


# Opt-in batched swap (ROOP_BATCH_SWAP=1): runs multiple face crops through one
# inference call instead of one-at-a-time, to better saturate the GPU. The stock
# swap ONNX is fixed batch-1; we relax the input/output batch dim to symbolic so
# the session (and TensorRT engine) accept batches. Verified to produce output
# numerically identical to per-crop runs.
_BATCH_SWAP = os.environ.get('ROOP_BATCH_SWAP', '0') == '1'


def _relax_batch_dim(model):
    """Mutate `model` in place, giving every graph input/output a symbolic
    batch dimension so the session accepts batches > 1."""
    for t in list(model.graph.input) + list(model.graph.output):
        dims = t.type.tensor_type.shape.dim
        if len(dims):
            dims[0].dim_param = 'N'
            dims[0].ClearField('dim_value')


def _freeze_convtranspose_reshape(model):
    """Mutate `model` in place so TensorRT can build its ConvTranspose layers.

    GHOST's generator reshapes the identity vector with a [1,-1,1,1] target and
    feeds it straight into a ConvTranspose. onnxruntime resolves the -1 to 512,
    but TensorRT keeps that inferred channel dimension dynamic and refuses to
    build the deconvolution ('IDeconvolutionLayer number of channels in `input`
    tensor must not be dynamic' → INVALID_NODE). Bake the -1 into its
    statically-inferable value for any Reshape feeding a ConvTranspose. This is
    numerically a no-op — the graph inputs are fixed-shape, so every internal
    shape is already static — verified bit-identical on CPU. Returns True if it
    changed anything (inswapper and the other emap swappers match nothing)."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return False
    static_shapes = {}
    for vi in (list(inferred.graph.value_info) + list(inferred.graph.input)
               + list(inferred.graph.output)):
        dims = vi.type.tensor_type.shape.dim
        static_shapes[vi.name] = [
            (d.dim_value if (d.dim_param == '' and d.dim_value > 0) else None)
            for d in dims]
    convt_inputs = {n.input[0] for n in model.graph.node
                    if n.op_type == 'ConvTranspose' and n.input}
    inits = {i.name: i for i in model.graph.initializer}
    changed = False
    for node in model.graph.node:
        if node.op_type != 'Reshape' or node.output[0] not in convt_inputs:
            continue
        out_shape = static_shapes.get(node.output[0])
        if not out_shape or any(v is None for v in out_shape):
            continue
        shape_init = inits.get(node.input[1])
        if shape_init is None:
            continue
        arr = onnx.numpy_helper.to_array(shape_init)
        if -1 not in arr.tolist() or len(arr) != len(out_shape):
            continue
        shape_init.CopyFrom(onnx.numpy_helper.from_array(
            np.array(out_shape, dtype=arr.dtype), shape_init.name))
        changed = True
    return changed


def _swap_providers(providers):
    """Return a copy of `providers` with the TensorRT provider forced to FP32.

    inswapper_128 (and the other emap swappers) have layers that overflow in
    FP16, producing rainbow/smudge artifacts when the global precision mode is
    'mixed'/'fp16'. The swapper is tiny (128-256px), so full precision costs
    almost nothing while fixing the corruption; detection and enhancers stay on
    FP16 where they're stable and fast. Opt back into an FP16 swapper with
    ROOP_SWAP_FP16=1 (not recommended)."""
    if os.environ.get('ROOP_SWAP_FP16', '0') == '1':
        return providers
    patched = []
    for p in providers:
        if isinstance(p, (tuple, list)) and len(p) == 2 and 'tensorrt' in str(p[0]).lower():
            name, opts = p[0], dict(p[1])
            opts['trt_fp16_enable'] = False
            # Separate engine cache so the FP32 swap engine never collides with
            # the FP16 engines TensorRT builds for the other models.
            cache = opts.get('trt_engine_cache_path')
            if cache:
                fp32_cache = cache + '_swap_fp32'
                os.makedirs(fp32_cache, exist_ok=True)
                opts['trt_engine_cache_path'] = fp32_cache
            patched.append((name, opts))
        else:
            patched.append(p)
    return patched


class FaceSwapInsightFace():
    processorname = 'faceswap'
    type = 'swap'

    def __init__(self):
        self.plugin_options = None
        self.model_swap_insightface = None
        self.emap = None
        self.converter = None            # crossface embedding converter session
        self.converter_input = "input"
        self.embedding_mode = "normed_emap"
        self.source_crop_key = None      # image-source models: which pre-warped crop
        self.image_input_name = "target"
        self.embed_input_name = "source"
        self.loaded_model_key = None
        self.devicename = None
        self.pool = None        # SessionPool of extra sessions (TRT multi-context)
        self._swap_providers = None   # providers used to build the swap session(s)
        self._model_arg = None        # onnx path or serialized bytes handed to ORT
        self._trt_disabled = False    # set once we fall back off TensorRT for this model
        self._batch_unsupported = False   # set once RunBatch/RunBatchMulti fail for this model
        # Contract consumed by ProcessMgr — defaults match inswapper_128.
        self.model_output_size = 128
        self.model_mean = [0.0, 0.0, 0.0]
        self.model_standard_deviation = [1.0, 1.0, 1.0]
        self.model_denormalize = False
        self.model_template = "arcface"
        # Outcome-guard tolerance for this model; None = face_util's default.
        # See verify_tol_for / SWAP_MODELS above.
        self.model_verify_tol = None
        # Some swappers emit their own face mask as a SECOND graph output —
        # hififace and hyperswap both do. It says where the net actually
        # synthesised a face, which the paste matte cannot know: the matte is an
        # ellipse intersected with a landmark hull whose forehead extension runs
        # 60% above the brows and therefore into the HAIR. Measured against the
        # model's own verdict, 15-27% of the matte is territory hififace says is
        # not face on a frontal head, and 31% on a profile.
        #
        # The mask lands here, per thread, because the swap runs on N workers and
        # `Run` cannot change its return type without touching every caller
        # (RunBatch, RunBatchMulti, swap_batcher, ProcessMgr). ProcessMgr reads it
        # immediately after its own Run call, on the same thread.
        self.model_has_mask = False
        self._mask_tls = threading.local()
        # A second swapper this one routes to (RealSwap). None for every
        # single-net model, which is every model but `realswap`.
        self.secondary = None
        self.secondary_template = None
        # {track_id: routed_to_secondary}. Shared by every worker thread, hence
        # the lock: the routing latch is a property of the TRACK, so the frames
        # of one track are read and written from whichever workers happen to
        # take them.
        self._route_latch = {}
        self._route_lock = threading.Lock()
        # How much of a run actually used the second net. Without this a clean
        # result on real footage is ambiguous — "the routing is safe" and "the
        # routing never fired" look identical from the audit.
        self._routed_faces = 0
        self._seen_faces = 0

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options

        # Every run, not just the runs that load a model: track ids restart per
        # video, and the model is already loaded on the second one, so clearing
        # this inside the load block below would leave the previous clip's
        # latches deciding the opening frames of the next.
        with self._route_lock:
            self._route_latch.clear()
            self._routed_faces = 0
            self._seen_faces = 0

        swap_model = plugin_options.get("swap_model", "inswapper")
        if swap_model not in SWAP_MODELS:
            swap_model = "inswapper"
        spec = SWAP_MODELS[swap_model]

        # Reload when the user switched to a different swap model.
        if self.model_swap_insightface is not None and self.loaded_model_key != swap_model:
            self.Release()

        if self.model_swap_insightface is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [spec["url"]])
            model_path = os.path.join(model_dir, spec["file"])

            self.embedding_mode = spec.get("embedding", "normed_emap")
            self.source_crop_key = spec.get("source_crop_key")
            if self.embedding_mode == "normed_emap":
                graph = onnx.load(model_path).graph
                self.emap = self._find_emap(graph)
            else:
                self.emap = None

            # crossface embedding converter (ghost / simswap / hififace): a tiny
            # MLP that maps the buffalo_l ArcFace embedding into the identity
            # space the swap net was trained with. CPU is plenty for it — the
            # result is cached per source face, so it runs once per face per
            # model, not per frame.
            if spec.get("converter_url"):
                conditional_download(model_dir, [spec["converter_url"]])
                self.converter = onnxruntime.InferenceSession(
                    os.path.join(model_dir, spec["converter_file"]),
                    None, providers=["CPUExecutionProvider"])
                self.converter_input = self.converter.get_inputs()[0].name
            else:
                self.converter = None

            self.devicename = plugin_options["devicename"].replace('mps', 'cpu')

            swap_providers = _swap_providers(roop.globals.execution_providers)
            # Load once and apply the transforms this session needs. Freezing the
            # ConvTranspose reshape channel makes TensorRT-incompatible exports
            # (GHOST) buildable; batch relaxation is opt-in. If neither applies we
            # hand onnxruntime the path so it can memory-map the file directly.
            model_arg = model_path
            _model = onnx.load(model_path)
            _changed = _freeze_convtranspose_reshape(_model)
            if _BATCH_SWAP:
                _relax_batch_dim(_model)
                _changed = True
            if _changed:
                model_arg = _model.SerializeToString()

            # Remember what we built with so a run-time TensorRT failure can
            # rebuild this exact model on CUDA/CPU (see _rebuild_without_trt).
            self._swap_providers = swap_providers
            self._model_arg = model_arg
            self._trt_disabled = False
            self._batch_unsupported = False

            def _build(_i=0):
                sess_options = onnxruntime.SessionOptions()
                sess_options.enable_cpu_mem_arena = False
                return onnxruntime.InferenceSession(
                    model_arg, sess_options, providers=swap_providers)

            self.model_swap_insightface = _build()

            # Resolve input tensor names by rank instead of assuming names:
            # rank-4 = the image (NCHW), rank-2 = the identity embedding.
            for inp in self.model_swap_insightface.get_inputs():
                rank = len(inp.shape)
                if rank == 4:
                    self.image_input_name = inp.name
                elif rank == 2:
                    self.embed_input_name = inp.name
            # Image-source models (BlendSwap/UniFace) have TWO rank-4 inputs, so
            # rank alone can't tell target from source — resolve by the names the
            # models expose ('target' / 'source').
            if self.embedding_mode == "image":
                names = [inp.name for inp in self.model_swap_insightface.get_inputs()]
                self.image_input_name = "target" if "target" in names else self.image_input_name
                self.embed_input_name = "source" if "source" in names else self.embed_input_name

            # Optional TensorRT multi-context pool: the primary session plus
            # (N-1) independent extras so up to N worker threads can swap
            # concurrently instead of serialising behind the global GPU lock.
            if session_pool.pooling_enabled():
                n = session_pool.pool_size()
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model_swap_insightface] + extras): _e[i], n)

            # Publish the per-model contract ProcessMgr reads.
            self.model_output_size = spec["output_size"]
            self.model_mean = spec["mean"]
            self.model_standard_deviation = spec["standard_deviation"]
            self.model_denormalize = spec["denormalize"]
            self.model_template = spec.get("template", "arcface")
            # Absent for every model but hififace, which is the point: a model
            # without a measured value keeps face_util's shared default rather
            # than inheriting whichever one was loaded before it.
            self.model_verify_tol = spec.get("verify_tol")
            # Read from the GRAPH, not from the spec table: whether a net emits a
            # mask is a property of the file, and a hand-kept flag would be one
            # more thing to get wrong when a model is added.
            self.model_has_mask = len(self.model_swap_insightface.get_outputs()) > 1
            self.loaded_model_key = swap_model

            # ── RealSwap's second net ────────────────────────────────────────
            # Loaded through this same class rather than a bespoke loader, so it
            # inherits the TensorRT fallback, the session pool, the crossface
            # converter and the mask stashing without any of it being written
            # twice. The nested instance's own spec has no "secondary", so this
            # cannot recurse.
            secondary_key = spec.get("secondary")
            if secondary_key and secondary_key in SWAP_MODELS:
                sub = FaceSwapInsightFace()
                sub_options = dict(plugin_options)
                sub_options["swap_model"] = secondary_key
                sub.Initialize(sub_options)
                self.secondary = sub
                self.secondary_template = SWAP_MODELS[secondary_key].get(
                    "template", "arcface")
                # Batching would have to coalesce two nets' crops in step, and
                # the primary here (hyperswap) cannot batch anyway — its export
                # has an internal reshape baked to batch=1. Declaring it up
                # front means RunBatch/RunBatchMulti route straight to the
                # sequential path, which mixes correctly, instead of each run
                # paying for one doomed inference to discover the same thing.
                self._batch_unsupported = True

    @staticmethod
    def _find_emap(graph):
        """Locate the 512x512 identity-projection matrix (emap) embedded in the onnx."""
        for init in reversed(graph.initializer):
            arr = onnx.numpy_helper.to_array(init)
            if arr.ndim == 2 and arr.shape == (512, 512):
                return arr
        # Fallback: inswapper_128 stores emap as the last initializer.
        return onnx.numpy_helper.to_array(graph.initializer[-1])

    def _compute_latent(self, source_face: Face) -> np.ndarray:
        """Prepare the (1, 512) identity vector per the loaded model's contract.
        Converter results are cached on the Face object (keyed by model) so the
        crossface MLP runs once per source face, not once per frame."""
        mode = self.embedding_mode
        if mode == "normed":
            return source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        if mode in ("converted_raw", "converted_norm"):
            cache_key = f"_latent_{self.loaded_model_key}"
            cached = source_face.get(cache_key) if hasattr(source_face, 'get') else None
            if cached is not None:
                return cached
            emb = np.asarray(source_face.embedding, dtype=np.float32).reshape(-1, 512)
            converted = self.converter.run(None, {self.converter_input: emb})[0]
            converted = converted.ravel()
            if mode == "converted_norm":
                converted = converted / np.linalg.norm(converted)
            latent = converted.reshape(1, -1).astype(np.float32)
            try:
                source_face[cache_key] = latent
            except Exception:
                pass
            return latent
        # Default: inswapper-family normed_embedding @ emap.
        latent = source_face.normed_embedding.reshape((1, -1)).astype(np.float32)
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
        return latent

    def _prepare_source_crop(self, source_face: Face) -> np.ndarray:
        """Image-source models (BlendSwap/UniFace): build the (1,3,H,W) source
        blob from the pre-warped aligned source crop (BGR→RGB, /255 — no
        mean/std, matching FaceFusion's prepare_source_frame). Cached per source
        face + model. Returns None when the crop is absent (source ingested
        before crops were attached — caller falls back to a no-op swap)."""
        cache_key = f"_srcblob_{self.loaded_model_key}"
        cached = source_face.get(cache_key) if hasattr(source_face, 'get') else None
        if cached is not None:
            return cached
        crop = source_face.get(self.source_crop_key) if hasattr(source_face, 'get') else None
        if crop is None:
            return None
        blob = crop[:, :, ::-1] / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        try:
            source_face[cache_key] = blob
        except Exception:
            pass
        return blob

    def _compute_source_input(self, source_face: Face):
        """Return whatever the loaded model feeds into its source input: an
        image blob for image-source models, else the identity latent vector."""
        if self.embedding_mode == "image":
            return self._prepare_source_crop(source_face)
        return self._compute_latent(source_face)

    @staticmethod
    def _is_trt(p):
        name = p[0] if isinstance(p, (tuple, list)) else p
        return 'tensorrt' in str(name).lower()

    def _rebuild_without_trt(self) -> bool:
        """Rebuild the swap session(s) with the TensorRT provider stripped out,
        keeping CUDA/CPU. Some torch_jit swap exports (notably GHOST) build a
        TensorRT engine that then fails shape verification at run time
        ('OrtValue shape verification failed. Current shape:{1,1024,2,2}
        Requested shape:{1,512,1,1}') because TRT mis-fuses the identity
        Reshape → ConvTranspose. The swap net is tiny (128-256px), so CUDA EP
        costs almost nothing. Returns False (→ caller re-raises) when TRT was
        already gone, so a genuine non-TRT error is not swallowed."""
        if self._trt_disabled or not self._swap_providers:
            return False
        providers = [p for p in self._swap_providers if not self._is_trt(p)]
        if len(providers) == len(self._swap_providers):
            return False   # no TRT provider to strip — can't help, re-raise
        def _build(_i=0):
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            return onnxruntime.InferenceSession(
                self._model_arg, sess_options, providers=providers)
        self.model_swap_insightface = _build()
        if self.pool is not None:
            n = session_pool.pool_size()
            extras = [_build(i) for i in range(n - 1)]
            self.pool = session_pool.SessionPool(
                lambda i, _e=([self.model_swap_insightface] + extras): _e[i], n)
        self._trt_disabled = True
        print(f"[swap] '{self.loaded_model_key}' failed under TensorRT "
              f"(shape verification); rebuilt on CUDA/CPU for this model.")
        return True

    def _stash_masks(self, ort_outs, count=1):
        """Keep this call's mask output(s) for the calling thread to collect.

        `take_masks` below is the only reader, and it clears as it reads, so a
        model WITHOUT a mask cannot serve a stale one left behind by a previous
        model in the same session.
        """
        masks = None
        if len(ort_outs) > 1 and ort_outs[1] is not None:
            m = np.asarray(ort_outs[1])
            # (B,1,H,W) -> list of (H,W); some exports drop the channel axis.
            if m.ndim == 4:
                masks = [m[i, 0] for i in range(m.shape[0])]
            elif m.ndim == 3:
                masks = [m[i] for i in range(m.shape[0])]
        self._mask_tls.masks = masks if masks and len(masks) >= count else None

    def take_masks(self):
        """The mask(s) from this thread's most recent inference, or None. Clears,
        so each swap's mask is consumed exactly once."""
        masks = getattr(self._mask_tls, 'masks', None)
        self._mask_tls.masks = None
        return masks

    def _republish_masks(self, masks):
        """Publish a batch's worth of masks that were collected one call at a
        time, so a sequential fallback keeps the batched path's contract: the
        caller does ONE take_masks() and expects one mask per crop.

        A partial set is published as None rather than as a short list — the
        caller pairs masks to crops by position, so a gap would misattribute
        every mask after it to the wrong face.
        """
        self._mask_tls.masks = (
            masks if masks and all(m is not None for m in masks) else None
        )

    def _infer(self, feed):
        """Run the swap net, transparently falling back off a broken TensorRT
        engine to CUDA/CPU the first time a SINGLE-FRAME (batch=1) call fails
        (see _rebuild_without_trt). Shared by Run / RunBatch / RunBatchMulti.

        A batch>1 failure must NOT trigger this: for a model whose export has
        an internal reshape baked to batch=1 (e.g. hyperswap), batch>1 failing
        says nothing about whether batch=1 works under TensorRT — it does,
        measured at ~24ms/call vs ~600ms/call once wrongly disabled here (a
        25x regression that used to hit every single-frame swap for the rest
        of the run). RunBatch/RunBatchMulti already have their own fallback
        to sequential Run() calls for exactly this case, and each of those
        goes through _infer() again with batch=1 — so re-raising immediately
        here just lets that fallback's own single-frame calls get a fair,
        unpoisoned shot at TensorRT instead of inheriting a CUDA-only session
        that a batch-shape problem, not a real TRT failure, forced onto them.
        """
        is_batch1 = feed[self.image_input_name].shape[0] <= 1
        try:
            if self.pool is not None:
                with self.pool.lease() as sess:
                    return sess.run(None, feed)
            return self.model_swap_insightface.run(None, feed)
        except Exception:
            if not is_batch1 or not self._rebuild_without_trt():
                raise
            if self.pool is not None:
                with self.pool.lease() as sess:
                    return sess.run(None, feed)
            return self.model_swap_insightface.run(None, feed)

    # ── RealSwap: two nets, one crop ─────────────────────────────────────────

    @staticmethod
    def _crop_to_crop(kps, size, src_template, dst_template):
        """The affine carrying a crop aligned on `src_template` into
        `dst_template`'s crop space, for THIS face: M_dst @ inv(M_src).

        Computed per face because that is exact and costs nothing — two umeyama
        fits on 5 points, against a face that takes ~210 ms end to end.

        It is NOT, however, meaningfully face-dependent, and the comment here
        used to claim it was. The reasoning was that estimate_norm is a
        least-squares SIMILARITY fit, so it lands the keypoints near the
        template rather than on it, and a profile fits a frontal 5-point
        template far worse than a frontal head does — so the two crops' relation
        should inherit that residual. Measured, it does not: between a frontal
        and a profile keypoint set the matrix moves by 1.1e-05 on a 256px crop,
        because the residual enters the composition only at second order. So
        this could equally be a cached constant per (size, template pair); it is
        left per-face because exactness is free here and a cache is one more
        thing to invalidate.
        """
        from roop.face_util import estimate_norm
        pts = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        src = np.vstack([estimate_norm(pts, size, src_template), [0, 0, 1]])
        dst = np.vstack([estimate_norm(pts, size, dst_template), [0, 0, 1]])
        return (dst @ np.linalg.inv(src))[:2].astype(np.float32)

    @staticmethod
    def _warp_chw(chw, M, size):
        """Resample a [3,H,W] float tensor through a 2x3 affine.

        LANCZOS4, not the obvious INTER_LINEAR, because a routed face is
        resampled TWICE — into the secondary's template space and back — and
        bilinear is far more destructive over a round trip than it looks.
        Measured over 12 real profile crops, detail retained after the pair of
        warps (Laplacian variance against the un-warped crop):

            INTER_LINEAR    35.8%      <- throws away nearly two thirds
            INTER_CUBIC     71.4%
            INTER_LANCZOS4  81.6%

        This was a live quality bug, not a micro-optimisation: bilinear here
        softened every routed face before the second net even saw it, and on
        contact/profile footage two thirds of faces route. It showed up as
        "the face is blurred with no definite landmarks", and it was visible in
        the sweep's own `ghost` column all along — realswap read 0.931/0.917 at
        profile against hyperswap's 0.945/0.944 and native hififace's
        0.955/0.942, i.e. softer than BOTH of its parents, which a derived crop
        has no business being.

        BORDER_REPLICATE for the same reason align_crop uses it: the two
        templates do not frame the face identically, so one crop's edge maps
        outside the other's, and a black wedge there is a hard edge the net
        never saw in training.
        """
        img = np.ascontiguousarray(np.asarray(chw).transpose(1, 2, 0))
        out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(out.transpose(2, 0, 1))

    def _renormalize(self, blob, src, dst):
        """Re-express an input blob prepared for `src`'s mean/std in `dst`'s.

        A no-op for the shipped pairing — hyperswap and hififace both take
        [-1,1] — but the pairing is a table entry, and a mismatched one would
        otherwise show up as a colour shift in half the face rather than as an
        error.
        """
        s_mean = np.asarray(src.model_mean, dtype=np.float32).reshape(3, 1, 1)
        s_std = np.asarray(src.model_standard_deviation, dtype=np.float32).reshape(3, 1, 1)
        d_mean = np.asarray(dst.model_mean, dtype=np.float32).reshape(3, 1, 1)
        d_std = np.asarray(dst.model_standard_deviation, dtype=np.float32).reshape(3, 1, 1)
        if np.array_equal(s_mean, d_mean) and np.array_equal(s_std, d_std):
            return blob
        return ((blob * s_std + s_mean) - d_mean) / d_std

    # Pose band that routes a face to the secondary net, with hysteresis: enter
    # at ENTER degrees of |yaw|, and do not come back until below EXIT.
    #
    # Measured, 1310 paired frames per model over yaw -90/-45/0/+45/+90 (see the
    # yaw-sweep table in RECODE_STATUS.md). At yaw 0 and +/-45 hyperswap wins
    # identity on 100% of frames and NEITHER model has a tail — max bulk shift
    # 0.010 interocular units against a 0.002 measurement floor. There is nothing
    # to fix there and routing away would only cost identity.
    #
    # At +/-90 the failure is a tail, not a mean: hyperswap displaces the face by
    # more than 0.10 on 13-17% of frames, up to 0.59 and 0.947 — a whole face
    # moved by an eye-to-eye distance — where hififace does so on 6%, to a max of
    # 0.375. And the failures NEST: of hyperswap's 35/45 blow-ups per band,
    # hififace shares 16/14 and fixes 19/31, while introducing exactly ONE of its
    # own in each. Identity at profile is 0.35 vs 0.23, which is the difference
    # between poor and poor, against the difference between an intact face and a
    # destroyed one.
    #
    # These fire on ~2/3 of the faces in the `double/` clips, which looked like a
    # broken gate and is not one. The two-person clips are people facing EACH
    # OTHER, so both heads are near-profile to camera; d1's solver readings have
    # a MEDIAN |yaw| of 85 deg and 61.7% of faces at or above 80.
    #
    # Checked against a signal the solver has no part in — eye separation as a
    # fraction of face width, which collapses when a head turns and one eye goes
    # behind the nose. Calibrated on plates of known pose (true frontal 0.460,
    # true 90 deg 0.047), it tracks the solver monotonically on real footage:
    #
    #   solver |yaw|      0-30   30-50   50-70   70-80   80-95
    #   d1  eye-sep/w    0.496   0.381   0.237   0.183   0.091
    #   d6  eye-sep/w    0.353   0.306   0.270   0.184   0.121
    #
    # So the faces this routes really are turned that far, and the routing rate
    # is a property of the material, not a misfire.
    #
    # The one caveat left: the 80-95 band reads 0.09-0.12 where a TRUE 90 deg
    # face reads 0.047, so that band likely holds faces nearer 70-85 than 90 —
    # consistent with the 15-20 deg per-person error in
    # pose-solve-head-shape-limit. The sweep only proved hififace better at the
    # plates' 85-87, so the lower part of this band is routed on an
    # extrapolation. Narrowing it needs footage at a KNOWN intermediate yaw,
    # which the five-plate sweep cannot provide.
    _SECONDARY_ENTER_YAW = 80.0
    _SECONDARY_EXIT_YAW = 70.0

    def _route_to_secondary(self, target_face: Face) -> bool:
        """True when this face should be swapped by the secondary net instead.

        Latched along the TRACK, not decided per frame. A bare threshold makes
        a head hovering near it alternate between two models with different
        identity strength from frame to frame, which is requirement 11's exact
        failure mode — and the fix the project already uses everywhere else
        (roll latch, non-frontal router) is hysteresis along the track, NOT a
        cross-fade between the two outputs. A cross-fade here would be the
        uniform blend of two differently-shaped faces that `angle-handling-
        three-layers` measured and ruled out: it doubles every feature.

        Without tracking there is no track to latch on, so this degrades to the
        bare threshold rather than pretending to be stable.
        """
        from roop.face_util import solve_pose_5pt
        pose = solve_pose_5pt(getattr(target_face, 'kps', None))
        if pose is None:
            return False
        yaw = abs(float(pose[0]))

        tid = target_face.get('_track_id') if hasattr(target_face, 'get') else None
        with self._route_lock:
            if tid is None:
                latched = yaw >= self._SECONDARY_ENTER_YAW
            else:
                latched = self._route_latch.get(tid, False)
                latched = (yaw >= self._SECONDARY_EXIT_YAW if latched
                           else yaw >= self._SECONDARY_ENTER_YAW)
                self._route_latch[tid] = latched
            self._seen_faces += 1
            self._routed_faces += int(latched)
            return latched

    def route_summary(self):
        """One line on how many faces this run sent to the second net, or None
        for a single-net model. Read by the benches so a clean result can be
        told apart from an idle router."""
        if self.secondary is None or not self._seen_faces:
            return None
        pct = 100.0 * self._routed_faces / self._seen_faces
        return (f"[swap] {self.loaded_model_key}: {self._routed_faces} of "
                f"{self._seen_faces} faces ({pct:.1f}%) routed to "
                f"'{self.secondary.loaded_model_key}'")

    def _run_secondary(self, source_face: Face, target_face: Face, temp_frame):
        """The second net's swap of the same face, resampled back into the
        PRIMARY's crop space and normalization, or None when it is unavailable.

        A failure here is downgraded to "RealSwap runs as its primary alone" for
        the rest of the run rather than killing the render: the routing is a
        quality feature, and a face is better swapped by one net than not at all.
        """
        sec = self.secondary
        kps = getattr(target_face, 'kps', None)
        if sec is None or kps is None:
            return None
        size = int(temp_frame.shape[-1])
        try:
            to_b = self._crop_to_crop(kps, size, self.model_template,
                                      self.secondary_template)
            blob = self._renormalize(temp_frame, self, sec)
            blob_b = self._warp_chw(blob[0], to_b, size)[np.newaxis]
            out_b = sec.Run(source_face, target_face, blob_b)
            back = cv2.invertAffineTransform(to_b)
            # This face was swapped by the secondary, so the mask that describes
            # it is the SECONDARY's — carried into the primary's crop space,
            # because that is the space ProcessMgr pastes through. Draining it is
            # not optional either way: left stashed on the sub-processor it would
            # be served to a later face on the same thread.
            sec_masks = sec.take_masks()
            if sec_masks and sec_masks[0] is not None:
                m = np.asarray(sec_masks[0], dtype=np.float32)
                self._mask_tls.masks = [cv2.warpAffine(
                    m, back, (size, size), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)]
            else:
                self._mask_tls.masks = None
            out_b = np.asarray(out_b, dtype=np.float32)
            if bool(sec.model_denormalize) != bool(self.model_denormalize):
                out_b = (out_b + 1.0) / 2.0 if sec.model_denormalize else out_b * 2.0 - 1.0
            return self._warp_chw(out_b, back, size)
        except Exception as e:
            print(f"[swap] realswap: secondary net '{sec.loaded_model_key}' failed "
                  f"({e!r}); running as '{self.loaded_model_key}' alone for the "
                  f"rest of this run.")
            self.secondary = None
            return None

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        # RealSwap routes each face to ONE of its two nets and runs only that
        # one. Running both and combining them was the original framing, and the
        # measurement retired it: there is no region where the second net helps a
        # face the first one swaps well (at yaw 0/+/-45 hyperswap wins identity
        # on 100% of frames and neither model has a tail), and cross-dissolving
        # two differently-shaped faces doubles the features anyway. Routing also
        # means a face costs ONE net, so the pairing is free on the ~95% of faces
        # that never reach the profile band.
        if getattr(self, 'secondary', None) is not None and self._route_to_secondary(target_face):
            routed = self._run_secondary(source_face, target_face, temp_frame)
            if routed is not None:
                return routed
            # Secondary unavailable — fall through and swap with the primary.

        latent = self._compute_source_input(source_face)
        if latent is None:
            # Image-source model but no source crop available → return the target
            # crop unchanged (no swap) rather than crashing.
            return temp_frame[0]
        # Use the standard run() API rather than io_binding.  io_binding with
        # bind_output() (no device_type) leaves output placement to TensorRT,
        # which registers a device type that copy_outputs_to_cpu() has no
        # transfer path for. run() handles all device transfers internally and
        # works correctly across CPU, CUDA, and TensorRT execution providers.
        feed = {self.image_input_name: temp_frame, self.embed_input_name: latent}
        # _infer leases an independent pool session (own TensorRT context) when
        # pooling is on, and falls back off a broken TRT engine transparently.
        ort_outs = self._infer(feed)
        # Some models (hififace, HyperSwap) emit (image, mask). The image is
        # output [0]; the mask is kept for this thread rather than dropped — see
        # `model_has_mask` and `take_masks`.
        self._stash_masks(ort_outs, 1)
        return ort_outs[0][0]

    def _sequential_fallback(self, requests: list) -> list:
        """requests = list of (source_face, target_face, blob). Runs each crop
        through Run() one at a time — the numerically-identical, always-safe
        path RunBatch/RunBatchMulti fall back to when batching doesn't work."""
        results = []
        masks = []
        for src, tgt, blob in requests:
            results.append(self.Run(src, tgt, blob))
            # Each Run stashes its OWN mask into the same single-slot
            # thread-local, so it has to be drained here — otherwise only
            # the last crop's mask survives the loop and the caller, which
            # expects one mask per crop, silently loses the swap model's
            # face mask (or fails to reassemble it) on exactly the path
            # this fallback exists to rescue.
            m = self.take_masks()
            masks.append(m[0] if m else None)
        self._republish_masks(masks)
        return results

    def RunBatch(self, source_face: Face, target_face: Face, temp_frames: list) -> list:
        """Batched equivalent of Run: temp_frames is a list of [1,3,H,W]
        preprocessed crops sharing the same source identity. Returns a list of
        [3,H,W] outputs, one per crop — numerically identical to calling Run on
        each, but in a single inference (better GPU utilization). Requires the
        session to be batch-dynamic (ROOP_BATCH_SWAP=1)."""
        if self._batch_unsupported:
            return self._sequential_fallback(
                [(source_face, target_face, t) for t in temp_frames])
        latent = self._compute_source_input(source_face)
        if latent is None:
            # Image-source model with no source crop → no-op (return the input
            # target crops unchanged), matching Run's fallback.
            return [t[0] for t in temp_frames]
        img_batch = np.concatenate(temp_frames, axis=0).astype(np.float32)   # [B,3,H,W]
        latent_batch = np.repeat(latent, img_batch.shape[0], axis=0)         # [B,512] or [B,3,Hs,Ws]
        feed = {self.image_input_name: img_batch, self.embed_input_name: latent_batch}
        try:
            ort_outs = self._infer(feed)
            out = ort_outs[0]   # [B,3,H,W]
            self._stash_masks(ort_outs, out.shape[0])
            return [out[i] for i in range(out.shape[0])]
        except Exception as batch_err:
            # If batch inference fails (e.g. TRT shape restriction or a model
            # whose graph has an internal reshape baked to batch=1 — some
            # exports can't be made batch-dynamic just by relaxing the graph's
            # declared input/output shapes), fall back gracefully to running
            # single face swaps sequentially. This is a property of the loaded
            # MODEL, not a transient condition, so remember it and stop
            # attempting the batched path for the rest of this model's
            # lifetime — otherwise every remaining frame pays for a doomed
            # inference call (and a matching TensorRT/CUDA error) before
            # falling back anyway.
            self._batch_unsupported = True
            print(f"[swap] '{self.loaded_model_key}' does not support batched inference "
                  f"({batch_err!r}); disabling batching for the rest of this run "
                  f"(falling back to sequential single-frame swaps).")
            return self._sequential_fallback(
                [(source_face, target_face, t) for t in temp_frames])

    def RunBatchMulti(self, requests: list) -> list:
        """Like RunBatch but each crop carries its OWN source identity (for
        cross-frame coalescing where different faces batch together).
        requests = list of (source_face, target_face, blob[1,3,H,W]); the
        target_face is unused by the swap net. Returns a list of [3,H,W]."""
        if self._batch_unsupported:
            return self._sequential_fallback(requests)
        latents = [self._compute_source_input(src) for src, _tgt, _blob in requests]
        if any(l is None for l in latents):
            # Image-source model with a crop-less source → no-op passthrough.
            return [r[2][0] for r in requests]
        latent_batch = np.concatenate(latents, axis=0)                       # [B,512]
        img_batch = np.concatenate([r[2] for r in requests], axis=0).astype(np.float32)  # [B,3,H,W]
        feed = {self.image_input_name: img_batch, self.embed_input_name: latent_batch}
        try:
            ort_outs = self._infer(feed)
            out = ort_outs[0]
            self._stash_masks(ort_outs, out.shape[0])
            return [out[i] for i in range(out.shape[0])]
        except Exception as batch_err:
            # See RunBatch above: a model-level incompatibility, not transient.
            self._batch_unsupported = True
            print(f"[swap] '{self.loaded_model_key}' does not support batched inference "
                  f"({batch_err!r}); disabling batching for the rest of this run "
                  f"(falling back to sequential single-frame swaps).")
            return self._sequential_fallback(requests)

    def Release(self):
        summary = self.route_summary()
        if summary:
            print(summary)
        if self.secondary is not None:
            self.secondary.Release()
            self.secondary = None
            self.secondary_template = None
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model_swap_insightface
        self.model_swap_insightface = None
        self.emap = None
        self.converter = None
        self.loaded_model_key = None
        self.model_has_mask = False
        self.model_verify_tol = None
