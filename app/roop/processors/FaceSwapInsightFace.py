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
#   cscs_dual      — CSCS's OWN recognizer plus its id adapter, summed. Not a
#                    crossface conversion of buffalo_l's vector: these two nets
#                    take an IMAGE (the ffhq-aligned source crop at 112) and
#                    produce their own 512-d spaces, which CSCS was trained on.
#                    (simswap / hififace)
# "template" is the 5-point alignment the model was trained on ('arcface' =
# the existing insightface alignment; others live in face_util.WARP_TEMPLATES).
_FF30 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
_FF31 = "https://huggingface.co/facefusion/models-3.1.0/resolve/main/"
_FF33 = "https://huggingface.co/facefusion/models-3.3.0/resolve/main/"
_FF34 = "https://huggingface.co/facefusion/models-3.4.0/resolve/main/"
# VisoMaster's asset release (GPL-3.0 project; see NOTICE.md).
_VISO = "https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/"

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
    # InStyleSwapper256 A/B/C (VisoMaster) — inswapper architecture at 256 with
    # its own emap, so the identity path is the plain inswapper one. Three
    # checkpoints trained differently; offered side by side like hyperswap's.
    #
    # BOTH of the non-obvious values here were MEASURED, not read off a doc:
    #   * normalization — [0,1] in / no denorm. The [-1,1] arms return NaN, so
    #     this is not a preference, it is the only one that runs.
    #   * template — ffhq_512, NOT the arcface the upstream docs imply. Scored by
    #     identity transfer (cosine of the re-detected output against the source
    #     minus against the target), which is comparable across templates where a
    #     self-swap MAE is not: a different template is a different crop, so MAE
    #     compares two arms against two different references.
    #         reswapper (control, known arcface)  arcface +0.4243  ffhq +0.3819
    #         InStyle A                           arcface +0.6309  ffhq +0.6803
    #         InStyle B                           arcface +0.5804  ffhq +0.6522
    #         InStyle C                           arcface +0.6008  ffhq +0.6584
    #     The control lands on its own known-correct template, which is what
    #     makes the other three rows worth anything. n=3 pairs, single frames —
    #     enough to pick the template, NOT a quality claim against other models.
    "instyleswapper_a": {
        "file": "InStyleSwapper256_Version_A.fp16.onnx",
        "url": _VISO + "InStyleSwapper256_Version_A.fp16.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "normed_emap",
        "template": "ffhq_512",
    },
    "instyleswapper_b": {
        "file": "InStyleSwapper256_Version_B.fp16.onnx",
        "url": _VISO + "InStyleSwapper256_Version_B.fp16.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "normed_emap",
        "template": "ffhq_512",
    },
    "instyleswapper_c": {
        "file": "InStyleSwapper256_Version_C.fp16.onnx",
        "url": _VISO + "InStyleSwapper256_Version_C.fp16.onnx",
        "output_size": 256,
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "denormalize": False,
        "embedding": "normed_emap",
        "template": "ffhq_512",
    },
    # CSCS (VisoMaster) — ffhq alignment, [-1,1] in/out, and the reason it is
    # here: it does NOT reuse buffalo_l's embedding. It ships its own recognizer
    # AND an id adapter, both taking the 112 ffhq source crop, and SUMS the two
    # L2-normalized results. Upstream describes that as being aimed at difficult
    # head poses, which is this project's standing open problem — but that is
    # their claim, unmeasured here. It is offered, not defaulted.
    #
    # Costs 1.26 GB across three files, which is why it is worth knowing it is
    # opt-in: on a 6GB card that is a real fraction of the budget.
    "cscs": {
        "file": "cscs_256.onnx",
        "url": _VISO + "cscs_256.onnx",
        "output_size": 256,
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "denormalize": True,
        "embedding": "cscs_dual",
        "template": "ffhq_512",
        "recognizer_file": "cscs_arcface_model.onnx",
        "recognizer_url": _VISO + "cscs_arcface_model.onnx",
        "id_adapter_file": "cscs_id_adapter.onnx",
        "id_adapter_url": _VISO + "cscs_id_adapter.onnx",
        "source_crop_key": "_src_crop_ffhq_112",
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

    # (size, template) -> feathered eye-band mask. Face-independent (the crop is
    # the template), so it is built once for the whole run.
    _EYE_MASK_CACHE = {}

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
        # The ORIGINAL frame and the primary's crop affine, published per-thread
        # by ProcessMgr for the face about to be swapped. Only realswap reads it,
        # to crop its secondary net straight from the plate instead of from the
        # primary's already-resampled crop. None means "not available for this
        # face" and the derived-crop path is taken instead.
        self._plate_tls = threading.local()
        # How each composited face got its secondary crop. See mix_summary.
        self._plate_crops = 0
        self._derived_crops = 0
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
        # result on real footage is ambiguous — "the second net helped" and "the
        # second net never ran" look identical from the audit.
        self._mixed_faces = 0
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
            self._mixed_faces = 0
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

            # CSCS's own identity pair. Separate from `converter` on purpose:
            # a crossface converter maps an EMBEDDING we already have, these two
            # take an IMAGE and produce the space CSCS was trained on. CPU is
            # plenty — the result is cached per source face, so they run once per
            # face per model, not per frame.
            if spec.get("recognizer_url"):
                conditional_download(model_dir, [spec["recognizer_url"],
                                                 spec["id_adapter_url"]])
                self.cscs_rec = onnxruntime.InferenceSession(
                    os.path.join(model_dir, spec["recognizer_file"]),
                    None, providers=["CPUExecutionProvider"])
                self.cscs_id = onnxruntime.InferenceSession(
                    os.path.join(model_dir, spec["id_adapter_file"]),
                    None, providers=["CPUExecutionProvider"])
            else:
                self.cscs_rec = None
                self.cscs_id = None

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
            #
            # COUNTING the outputs is not enough, and CSCS is why: its export
            # leaks nine internal attribute tensors alongside the image, the
            # first of which is (1, 1024, 2, 2). Under a count test that reads as
            # "has a mask" and the pipeline then composites a 1024-channel
            # feature map. A mask is a SINGLE-CHANNEL map the size of the output,
            # so ask for that shape.
            self.model_has_mask = self._graph_emits_mask(
                self.model_swap_insightface, spec["output_size"])
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
    def _graph_emits_mask(session, output_size):
        """True when output[1] is a single-channel map matching the image.

        Accepts (1,1,S,S) and (1,S,S), and tolerates symbolic dims — an export
        with dynamic spatial axes still declares its CHANNEL count, which is the
        part that separates a mask from a leaked feature tensor.
        """
        outs = session.get_outputs()
        if len(outs) < 2:
            return False
        shape = list(outs[1].shape or [])
        if len(shape) == 4:
            chan, spatial = shape[1], shape[2:]
        elif len(shape) == 3:
            chan, spatial = 1, shape[1:]
        else:
            return False
        if chan != 1:
            return False
        return all((not isinstance(d, int)) or d == output_size for d in spatial)

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
        if mode == "cscs_dual":
            cache_key = f"_latent_{self.loaded_model_key}"
            cached = source_face.get(cache_key) if hasattr(source_face, 'get') else None
            if cached is not None:
                return cached
            crop = source_face.get(self.source_crop_key) if hasattr(source_face, 'get') else None
            if crop is None:
                # Source ingested before the crops were attached. Falling back to
                # buffalo_l's vector would be WORSE than useless — it is a
                # different space, so it would produce a confident swap toward
                # nobody. Signal "no identity" and let the caller skip.
                return None
            blob = (crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
            blob = blob.transpose(2, 0, 1)[None].astype(np.float32)
            def _emb(sess):
                v = sess.run(None, {sess.get_inputs()[0].name: blob})[0].reshape(-1)
                return v / (np.linalg.norm(v) or 1.0)
            # Summed, and deliberately NOT re-normalized afterwards: the sum's
            # magnitude is what carries the appearance/identity balance CSCS was
            # trained with, and normalizing it away measurably flattens it.
            latent = (_emb(self.cscs_rec) + _emb(self.cscs_id)).reshape(1, -1).astype(np.float32)
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

    def set_plate_context(self, plate, M, usable):
        """Publish the plate and the primary's crop affine for the next Run.

        `usable` is ProcessMgr's call, not this processor's, because ProcessMgr
        is what knows whether the primary's crop is still a plain align_crop of
        the plate. Two cases where it is not, and where a plate-derived
        secondary crop would therefore sit in a different place from the base it
        is composited onto:

          * pixel boost > 1 -- the crop is imploded into INTERLEAVED SUBSAMPLE
            tiles, and a tile has no plate-space equivalent;
          * frontalization -- the crop has been warped to a frontal pose after
            alignment, so it is no longer align_crop(plate, kps).

        Both default off, so the shipped configuration takes the fast path.
        """
        self._plate_tls.ctx = (plate, M) if usable else None

    def clear_plate_context(self):
        """Drop the published plate. Not optional: left set, it would be served
        to the NEXT face on this thread, which is the same trap the swap mask
        has (see take_masks) -- and a stale plate would put another face's
        eyelids on this one."""
        self._plate_tls.ctx = None

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
        c_min, c_max = float(np.min(chw)), float(np.max(chw))
        out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
        if c_min < c_max:
            np.clip(out, c_min, c_max, out=out)
        return np.ascontiguousarray(out.transpose(2, 0, 1))

    @staticmethod
    def _prepare_blob(crop, model):
        """A BGR uint8 crop -> the model's input blob.

        Mirrors procmgr_tiling.prepare_crop_frame exactly (BGR->RGB, /255,
        mean/std, HWC->CHW, batch axis). Duplicated rather than imported because
        that lives on the ProcessMgr mixin and this runs inside the processor;
        if the two ever diverge the secondary net silently receives a
        differently-scaled image, so keep them in step.
        """
        x = np.asarray(crop)[:, :, ::-1] / 255.0
        x = (x - np.asarray(model.model_mean, dtype=np.float64)) /             np.asarray(model.model_standard_deviation, dtype=np.float64)
        return np.expand_dims(x.transpose(2, 0, 1), axis=0).astype(np.float32)

    @staticmethod
    def _compose_affine(outer, inner):
        """The 2x3 affine equivalent to applying `inner` and then `outer`."""
        a = np.vstack([np.asarray(outer, dtype=np.float64), [0, 0, 1]])
        b = np.vstack([np.asarray(inner, dtype=np.float64), [0, 0, 1]])
        return (a @ b)[:2].astype(np.float32)

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

    # ── The composition rule ─────────────────────────────────────────────────
    # hyperswap is the base for the whole face; hififace contributes ONLY the
    # eyelid / eyelash band around each eye.
    #
    # This is the user's own brief, given by feature rather than by pose:
    # hyperswap is the better model for faithfulness to the faceset and for the
    # nose, eye interior, mouth, chin and cheeks; hififace is the better one for
    # eyelids, eyelashes and expression. So the base is hyperswap and hififace
    # fills the ~15-20% it is weak at.
    #
    # It replaces a pose ROUTER that sent whole faces to hififace past ~80 deg of
    # yaw. That was measured on real contact footage (d1, 170 routed faces) and
    # is not coming back: identity to the faceset fell to 0.168 against
    # hyperswap's 0.424 -- 0.26 of identity for no gain in sharpness (90.2 vs
    # 93.6) -- and it read on screen as "the face is not visible, no definitive
    # landmark". The synthetic sweep had predicted only 0.35 -> 0.23; real
    # footage was more than twice as bad.
    #
    # Why a region composite is allowed here when a uniform blend is not: an
    # identity swap MOVES the features, so cross-dissolving two differently
    # shaped WHOLE faces superimposes two sets of them and there is no strength
    # at which that does not double (angle-handling-three-layers). Both crops
    # here are aligned from the SAME keypoints, so each eye lands in the same
    # place in both, and the blend is confined to a band whose feathered edge
    # falls on skin -- brow ridge above, cheek below -- where there is no feature
    # to double. `ghost` in the angle sweep is the check: above 1.0 means the two
    # faces are superimposing.
    # The band is an ANNULUS, not a disc: the lid margins and lashes are
    # hififace's, the eye APERTURE inside them stays hyperswap's. That is the
    # brief read exactly -- hififace for "eyelids, eyelashes, expression",
    # hyperswap for "nose, EYES, mouth, chin, cheeks" -- and it is also what the
    # measurement points to. A disc over the whole eye cost a mean 0.041 of
    # identity over the six non-profile cells of the yaw sweep, against a total
    # hyperswap->hififace gap of 0.118 in those same cells: 16.6% of the crop
    # AREA bought 34% of the available identity gap, about twice its share,
    # because ArcFace draws more identity per pixel from the periocular
    # interior than from anywhere else on a face. Keeping the aperture is the
    # attempt to put that back.
    # GRADED (yaw sweep, 1310 paired frames, 786 of them non-profile):
    #
    #                  hyperswap  hififace   disc  annulus
    #   id_source         0.7934    0.6751 0.7525   0.7645   +0.0120 vs disc
    #   eyes drift        0.0282    0.0247 0.0254   0.0245   -0.0009
    #   ghost             0.9530    0.9002 0.9172   0.9151   -0.0021
    #
    # Identity is higher on 95.7% of frames (t=+48.5) and the eye geometry did
    # NOT spring back -- it improved, to just under hififace's own 0.0247
    # (t=-9.4). Nose and mouth are unmoved. So the annulus is kept.
    #
    # But the reasoning that produced it did NOT hold, and that is the useful
    # part: cutting 61% of the band's AREA recovered only 29% of the identity
    # the disc gave up. Had the aperture been the identity-dense region, it
    # would have recovered most of it. The cost is spread across the band, if
    # anything weighted toward the lid ring that is still the secondary's. Do
    # not shrink this band further expecting identity back at that rate.
    #
    # Unfixed by this, and now the composite's largest single cost: `ghost` is
    # 0.038 below hyperswap and the annulus did not improve it (-0.0021).
    # Less secondary area bought no sharpness. One untested explanation is that
    # an annulus has TWO feathered edges rather than one, the inner running
    # straight across the eye. Measure before acting on that.
    # Radii are fractions of the interocular distance. The aperture is far wider
    # than it is tall -- an eye is a slit, not a disc -- so the inner ellipse
    # needs its own x and y factors; a single one either swallows the lid or
    # leaves the iris exposed.
    # ── LASH BAND (2026-08-21) ───────────────────────────────────────────────
    # The user's requirement, given directly: RealSwap's identity must match
    # hyperswap's, and only the EYELASHES come from hififace. That is a
    # different shape from the lid ring above, and it inverts the trade.
    #
    # The lid ring handed hififace a wide band of periocular SKIN, which is
    # where ArcFace reads identity most densely, and it only ever gave the
    # lashes half strength once opacity came down to 0.5. So it paid identity
    # for skin it did not need and still did not own the lashes outright.
    #
    # The band is now the LASH LINE only: a thin ring hugging the eye aperture,
    # at FULL opacity so the lashes really are hififace's, with the lid, brow,
    # socket and every other periocular pixel left to the base model. Radii are
    # fractions of the interocular distance; the aperture is a slit, far wider
    # than it is tall, so x and y are separate.
    # SIZED TO THE USER'S SPLIT: hififace 15%, hyperswap 85%. Measured against a
    # face oval fitted to this template (which is 36.1% of the crop -- the rest
    # is hair and background, so "% of the crop" understates the split by ~2.8x
    # and is the wrong number to quote):
    #
    #     outer_x  outer_y   % of FACE   % of crop
    #        0.40     0.26      11.52%       4.16%
    #        0.42     0.28      13.89%       5.08%
    #        0.44     0.30    * 15.16% *     5.60%
    #        0.45     0.32      17.24%       6.45%
    #
    # 0.44 horizontally is also the practical ceiling: the eye centres sit at
    # +-0.5 sep, so anything wider runs the two ellipses into each other across
    # the NOSE BRIDGE, which is the base model's outright.
    #
    # The aperture stays punched out at full size. That is the brief read
    # literally -- hififace for "eyelids, eyelashes, expression", hyperswap for
    # "nose, EYES, mouth, chin, cheeks" -- so the eye INTERIOR is the base
    # model's and the ring of lid and lash around it is the secondary's.
    _EYE_APERTURE_X = 0.225   # the eye opening itself -- stays the BASE model's
    _EYE_APERTURE_Y = 0.066
    _OUTER_X = 0.44           # outer edge of the lid+lash ring
    _OUTER_Y = 0.30
    _EYE_LIFT = 0.03          # upper lashes are longer than lower
    _EYE_FEATHER = 0.05       # small: the old 0.16 was wider than the ring
                              # itself and blurred it into a uniform blend.
    # Peak opacity of the band. 1.0 = the lid ring is wholly the secondary's,
    # which is what shipped. Lower values keep some of the base model inside the
    # ring itself, which is a DIFFERENT lever from shrinking the band: the
    # annulus cut 61% of the band's AREA and recovered only 29% of the identity
    # the disc gave up, because the cost is not at the aperture -- it is spread
    # through the ring, which area-shrinking never touched. Opacity attacks the
    # ring directly.
    #
    # MEASURED AND SET TO 0.5 (2026-08-21). The curve, yaw sweep at production
    # settings, non-profile, as a fraction of the full band's effect:
    #
    #   alpha    id_source    eyes     ghost    id cost  eye gain  ghost cost
    #   0.00      0.7897     0.0279   0.9514        0%        0%          0%
    #   0.25      0.7898     0.0272   0.9338        0%       19%         47%
    #   0.50      0.7851     0.0261   0.9213       15%       49%         80%
    #   1.00      0.7597     0.0242   0.9139      100%      100%        100%
    #
    # Identity cost is strongly CONVEX in opacity -- nearly all of it lives in
    # the top half of the range -- while eye gain is roughly linear. That is why
    # opacity works where AREA did not: the annulus cut 61% of the band's area
    # and recovered only 29% of the identity, because the cost is not at the
    # aperture. It is peak secondary content, and this is the lever for it.
    #
    # Confirmed on real footage, which is what the value is set on: d5, 6158
    # paired rows, identity distance to the faceset 0.5308 -> 0.4804 against the
    # full band (t=-88.3, better on 91.0% of frames), closing 65% of the gap to
    # plain hyperswap (+0.0776 -> +0.0272). Note the sweep predicted 85%: it
    # over-promises, as it has every time hififace is involved, so treat its
    # deltas as a direction and a ranking, never as the size of the effect.
    #
    # 0.25 was rejected despite being FREE on identity in the sweep: it keeps
    # only 19% of the eyelid gain, and the eyelid gain is the entire reason the
    # user chose to keep this band.
    #
    # Not an env knob. A knob defaulting to the old behaviour is the old
    # behaviour for everyone.
    _EYE_ALPHA = float(os.environ.get('ROOP_REALSWAP_BAND_ALPHA', '1.0') or '1.0')

    # How much secondary (hififace) goes into the BASE -- every pixel, including
    # the identity-dense skin the band deliberately avoids.
    #
    # This is the knob the block above says not to reach for. 0.15 was added on
    # 2026-08-22 with no measurement, against a comment recording that hififace
    # OUTSIDE the band cost 0.26 of faceset identity on real contact footage,
    # and it is a geometric blend of two nets whose eyes and mouth do not sit in
    # the same place (hififace's 3D shape branch), which is a soft double edge
    # by construction rather than a softer version of one face.
    #
    # Measured 2026-08-23 and REVERTED to 0. A/B on d1 under the production
    # stack, graded from the pipeline's own decision, identity distance of the
    # OUTPUT to the source faceset (lower is better), paired per frame because
    # both arms graded the same rows:
    #
    #     4702 paired frames
    #     0.00 beats 0.15 on 3185 of them (67.7%)
    #     mean delta -0.00654, median -0.00580, paired t = -30.5
    #     person 0: better on 70.8% of frames ; person 1: 64.1%
    #
    # Small per frame (~1.5-2% of the distance) and overwhelmingly consistent,
    # in exactly the direction the _EYE_ALPHA block above already recorded from
    # d5: hififace outside the band costs faceset identity. The 0.15 was added
    # on 2026-08-22 with no measurement against that comment.
    #
    # The user's brief -- "80-85% hyperswap + 15-20% hififace for eyelids,
    # lashes, expression" -- is served by the BAND, which is untouched and still
    # 100% hififace at _EYE_ALPHA=1.0. The ratio in the brief describes which
    # REGION comes from which net, not a global alpha over identity-dense skin;
    # reading it as a global blend is what produced the 0.15.
    #
    # NOTE this measures IDENTITY only. If a nonzero base is ever wanted for a
    # perceived-texture reason, measure that axis explicitly -- do not restore
    # it on the strength of this number being small.
    _BASE_MIX = float(os.environ.get('ROOP_REALSWAP_BASE_MIX', '0.0') or '0.0')

    @classmethod
    def _eye_region_mask(cls, size, template='arcface'):
        """A feathered [H,W] mask over both eyelid bands, in crop space.

        Face-independent, so it is built once per (size, template) and cached:
        the crop IS the template, so every aligned face puts its eyes on the
        same two points by construction. That is also what makes the composite
        safe -- the band is anchored to the alignment, not to a per-face
        landmark fit that could wander.
        """
        key = (int(size), str(template), float(cls._EYE_ALPHA))
        cached = cls._EYE_MASK_CACHE.get(key)
        if cached is not None:
            return cached
        from roop.face_util import swap_template_points
        pts = np.asarray(swap_template_points(int(size), template), dtype=np.float32)
        left, right = pts[0], pts[1]
        sep = float(np.linalg.norm(right - left)) or (size * 0.27)
        ax, ay = cls._EYE_APERTURE_X * sep, cls._EYE_APERTURE_Y * sep
        ox, oy = cls._OUTER_X * sep, cls._OUTER_Y * sep
        m = np.zeros((int(size), int(size)), np.float32)
        # Outer edge of the lash zone, lifted very slightly: upper lashes are
        # longer than lower ones.
        for eye in (left, right):
            c = (int(round(eye[0])), int(round(eye[1] - cls._EYE_LIFT * sep)))
            cv2.ellipse(m, c, (max(1, int(round(ox))), max(1, int(round(oy)))),
                        0, 0, 360, 1.0, -1)
        # Punch the aperture back out, centred on the eye itself. What is left
        # is the ring AT the lid margin, which is where lashes grow -- not the
        # lid, socket or brow, which are identity-dense skin and stay the base
        # model's.
        for eye in (left, right):
            c = (int(round(eye[0])), int(round(eye[1])))
            cv2.ellipse(m, c, (max(1, int(round(ax))), max(1, int(round(ay)))),
                        0, 0, 360, 0.0, -1)
        k = int(cls._EYE_FEATHER * sep) | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        # After the feather, so the edge profile is unchanged and only the peak
        # moves -- scaling before the blur would do the same thing here, but the
        # intent is "how much secondary at most", not "a smaller shape".
        a = float(cls._EYE_ALPHA)
        if a != 1.0:
            m = m * max(0.0, min(1.0, a))
        cls._EYE_MASK_CACHE[key] = m
        return m

    def _mix_outputs(self, primary, secondary, size, target_face=None):
        """The base mix everywhere, hififace alone inside the eyelid band.

        `_BASE_MIX` is how much secondary goes into the base -- i.e. outside the
        eye band, over identity-dense skin. It was 0 by construction until
        2026-08-22 (the band WAS the whole feature) and is measured, not
        assumed: see the constant.
        """
        m = self._eye_region_mask(size, self.model_template)
        # If extreme yaw (profile pose), attenuate far-eye eyelid mask to maintain profile silhouette integrity
        kps = getattr(target_face, 'kps', None) if target_face is not None else None
        if kps is not None and len(kps) >= 3:
            try:
                (lex, _), (rex, _), (nx, _) = kps[0], kps[1], kps[2]
                yaw = float(np.log((abs(nx - lex) + 1e-5) / (abs(rex - nx) + 1e-5)))
                if abs(yaw) > 0.65:
                    m = m.copy()
                    mid_x = int(size * 0.5)
                    if yaw > 0.65:
                        m[:, mid_x:] *= max(0.0, 1.0 - (yaw - 0.65) * 1.5)
                    else:
                        m[:, :mid_x] *= max(0.0, 1.0 - (-yaw - 0.65) * 1.5)
            except Exception:
                pass
        with self._route_lock:
            self._seen_faces += 1
            self._mixed_faces += 1
        b = float(self._BASE_MIX)
        base = primary if b <= 0.0 else primary * (1.0 - b) + secondary * b
        return base * (1.0 - m) + secondary * m

    def mix_summary(self):
        """One line on how many faces were composited, or None for a single-net
        model. A two-path feature is unreadable without it: "the second net
        helped" and "the second net never ran" look identical from the audit."""
        if self.secondary is None or not self._seen_faces:
            return None
        cov = float(self._eye_region_mask(self.model_output_size,
                                          self.model_template).mean())
        # The crop-source split is reported because a two-path feature is
        # unreadable without it. This processor has already been bitten twice:
        # the pose router sent 69% of a clip's faces to the wrong net while the
        # audit read 100%/100%, and the batched paths dropped the composite
        # entirely while mix_summary printed nothing at all. A silent fallback
        # to the derived crop would look exactly like the fix working.
        n = self._plate_crops + self._derived_crops
        src = ""
        if n:
            src = (f"; {100.0 * self._plate_crops / n:.0f}% cropped from the "
                   f"plate ({self._derived_crops} of {n} fell back to the "
                   f"derived crop)")
        return (f"[swap] {self.loaded_model_key}: {self._mixed_faces} of "
                f"{self._seen_faces} faces composited with "
                f"'{self.secondary.loaded_model_key}' over the eye band "
                f"({100.0 * cov:.1f}% of the crop, opacity "
                f"{self._EYE_ALPHA:g}), base mix {self._BASE_MIX:g}{src}")

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
            ctx = getattr(self._plate_tls, 'ctx', None)
            if ctx is None and target_face is not None:
                ctx = (getattr(target_face, 'plate_ctx', None) if hasattr(target_face, 'plate_ctx')
                       else target_face.get('plate_ctx') if isinstance(target_face, dict) else None)
            with self._route_lock:
                if ctx is not None:
                    self._plate_crops += 1
                else:
                    self._derived_crops += 1
            if ctx is not None:
                # FIRST-GENERATION CROP. The secondary's pixels used to survive
                # THREE resamples: plate -> the primary's arcface crop, that crop
                # -> the secondary's template, and back again. So the second net
                # never saw plate detail at all -- its INPUT was already a
                # resampled image, and LANCZOS4 (9cd9263) reduced that damage
                # without being able to undo it.
                #
                # Cropping from the plate makes it two, and the one removed is
                # the one UPSTREAM of the net, which is the one that matters:
                # hififace now gets exactly the kind of crop it was trained on.
                from roop.face_util import align_crop
                plate, M_a = ctx
                crop_b, M_b = align_crop(plate, kps, size,
                                         mode=self.secondary_template)
                blob_b = self._prepare_blob(crop_b, sec)
                # Secondary crop -> primary crop, composed through plate space:
                # a point is M_a . (M_b^-1 . p). One warp on the output, same as
                # before -- the saving is entirely on the input side.
                back = self._compose_affine(M_a, cv2.invertAffineTransform(M_b))
            else:
                # Pixel boost or frontalization: the primary's crop is not a
                # plain align_crop of the plate, so derive from it as before.
                to_b = self._crop_to_crop(kps, size, self.model_template,
                                          self.secondary_template)
                blob = self._renormalize(temp_frame, self, sec)
                blob_b = self._warp_chw(blob[0], to_b, size)[np.newaxis]
                back = cv2.invertAffineTransform(to_b)
            out_b = sec.Run(source_face, target_face, blob_b)
            # Drain the secondary's mask and DISCARD it. Draining is not
            # optional: left stashed on the sub-processor it would be served to a
            # later face on the same thread. Discarding is the composite's
            # contract — the published mask says where the pasted face is, and
            # under the composite that face is the PRIMARY's everywhere but the
            # ~6% eye band, so the primary's mask (already stashed by Run before
            # this call) is the one that describes it.
            #
            # This block used to overwrite the primary's mask with the
            # secondary's. That was correct under the pose ROUTER it was written
            # for (a95dd42), where the whole face really was the secondary's, and
            # it was left behind when the router became a region composite
            # (aeff9df) — whose own comment in `Run` already states the primary's
            # mask is the one published. Two ways it was wrong: hififace's
            # verdict on where the face is was trimming the paste matte of a
            # face hyperswap had painted, and a secondary that emitted no mask
            # nulled the primary's too, dropping the matte trim for that face
            # entirely.
            sec.take_masks()
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
        # RealSwap runs BOTH of its nets on EVERY face and composites them by
        # region (see "The composition rule" above): hyperswap is the base, and
        # hififace supplies only the eyelid annulus. So a face costs two nets,
        # +4.0 ms, about +1.9% of the ~210 ms swap+mask+enhance budget.
        #
        # This replaced a pose ROUTER that ran one net per face, picked by yaw.
        # The router is measured and rejected, not merely superseded: on real
        # contact footage it cost 0.26 of faceset identity for no gain in
        # sharpness. Do not reach for it again.
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
        out = ort_outs[0][0]

        # RealSwap: composite the second net's eyelid band over this face. The
        # primary's mask is the one already stashed and the one published, since
        # the base of the composite -- and every pixel outside the eye band -- is
        # the primary's.
        if getattr(self, 'secondary', None) is not None:
            other = self._run_secondary(source_face, target_face, temp_frame)
            if other is not None:
                out = self._mix_outputs(out, other, int(temp_frame.shape[-1]), target_face=target_face)
        return out

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

    # ── Why a composite model declines the batched paths ─────────────────────
    # Both batched entry points below run the PRIMARY net over a batch and
    # return its output directly; neither calls `_run_secondary`. For every
    # single-net model that is exactly right, but for `realswap` it silently
    # returns hyperswap alone — the eye band, which is the entire reason the
    # model exists, never happens, and nothing says so: `mix_summary` counts
    # only faces that reached `_mix_outputs`, so it reports "0 of 0" and prints
    # nothing at all.
    #
    # CORRECTION (2026-08-21, same day): this was first written as a live bug
    # and it is NOT one. `Load` already sets `_batch_unsupported = True`
    # whenever a secondary is configured — it has since a95dd42, for a different
    # stated reason (the primary's export has an internal reshape baked to
    # batch=1, so batching it is doomed anyway) — and both methods below check
    # that flag first. So a composite has always taken the sequential fallback,
    # and the eye band has never been silently dropped.
    #
    # The guard is kept because it says WHY in terms of the composite rather
    # than relying on a flag that means "this model cannot batch", which is a
    # different proposition that could be fixed one day and would then re-open
    # the hole. It is belt-and-braces, not a fix, and it should not be cited as
    # one.
    #
    # If batching a composite is ever wanted, the work is real: the secondary's
    # crops have to be coalesced in step with the primary's, which changes both
    # methods and the batcher's contract.
    def RunBatch(self, source_face: Face, target_face: Face, temp_frames: list) -> list:
        """Batched equivalent of Run: temp_frames is a list of [1,3,H,W]
        preprocessed crops sharing the same source identity. Returns a list of
        [3,H,W] outputs, one per crop — numerically identical to calling Run on
        each, but in a single inference (better GPU utilization). Requires the
        session to be batch-dynamic (ROOP_BATCH_SWAP=1)."""
        # getattr, not attribute access: `Run` guards the same way, because a
        # subclass that drives these methods over a stub session (the batch
        # fallback tests) never runs __init__.
        if self._batch_unsupported or getattr(self, 'secondary', None) is not None:
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
        # getattr, not attribute access: `Run` guards the same way, because a
        # subclass that drives these methods over a stub session (the batch
        # fallback tests) never runs __init__.
        if self._batch_unsupported or getattr(self, 'secondary', None) is not None:
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
        summary = self.mix_summary()
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
