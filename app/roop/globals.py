import os
from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
startup_args = None

cuda_device_id = 0
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
codeformer_fidelity = 0.5
subsample_size = 256
# AI upscale folded into the swap pipeline (runs as the last frame processor,
# strictly after the face is swapped/enhanced, producing one output file).
upscale_after_swap = True
upscale_model_after = 'esrganx2'
face_swap_mode = 'DFL XSeg'
blend_ratio = 0.80
distance_threshold = 1
default_det_size = True
face_detector_size = '640'
face_detector_threshold = 0.50  # Lowered from 0.60: insightface's SCRFD was missing faces at slight angles
face_detector_nms = 0.40
sam2_model_size = 'tiny'   # SAM2 tracked-mask checkpoint: tiny|small|base_plus|large
track_identities = False   # video: lock each tracked person to one source (anti identity-flip)
# Skin-tone / lighting match of the swapped crop to the original crop.
# 'rct' = LAB mean/std (Reinhard, legacy default), 'lct' = LAB covariance
# whitening (handles color casts RCT can't), 'mkl' = Monge-Kantorovitch linear
# (fuller distribution match), 'none' = off.
color_transfer_mode = 'rct'
# Target-conditioned appearance: the target crop supplies low-frequency
# illumination, scene cast, shadows, highlights, and skin chroma. Disabled by
# default so existing renders/custom looks remain bit-identical until enabled.
target_conditioned_appearance = False
target_conditioned_appearance_strength = 0.75
target_conditioned_appearance_temporal_alpha = 0.30
target_conditioned_appearance_cache_size = 256
# Alignment refinement: derive the 5 arcface keypoints from the 68-point
# landmarks (more stable at angles than the detector's raw 5 kps).
refine_landmarks = False
# Swap-model face mask (hififace / hyperswap emit one as a second graph output).
# Percent: how strongly to trim the paste matte to where the NET says it drew a
# face. Unlike the two above this is not pose-gated — the model's verdict is
# derived from the actual image. Off by default because it is new and unproven on
# real footage, not because it is suspected: turning it up is the first thing to
# try when a hififace/hyperswap paste reaches into hair. See
# procmgr_masking._model_mask_matte.
swap_model_mask_strength = 0.0
# Discard a swap that moved the face somewhere the plate's face was not. Past
# ~90 deg of yaw the detector claims a face on a head pointing away and the
# swapper paints a frontal one onto its cheek; no pose test separates that from
# a real profile, but the outcome does. See face_util.swap_moved_the_face.
# Costs one DETECTOR-ONLY re-detect, and only for faces turned or rolled far
# enough for the failure to be reachable — as shipped it was a full face
# analysis on every swapped face, which is 11.7 ms of an ~210 ms per-face budget
# spent mostly on an embedding nothing reads. See ProcessMgr._verify_worth_it
# and ROOP_VERIFY_MIN_OFFAXIS. ROOP_VERIFY_SWAP=0 turns the guard off entirely.
verify_swap = os.environ.get('ROOP_VERIFY_SWAP', '1') != '0'
# Jaw / chin reshape: warp the target's lower-face silhouette toward the SOURCE
# person's jaw/chin shape after the swap (identity swappers keep the target's
# geometry). strength 0..1 = amount of the shape difference applied.
jaw_reshape = False
jaw_reshape_strength = 0.5
# Skin detail transfer: inject the original footage's high-frequency texture
# (pores, grain) onto the swapped/enhanced face. 0 = off (no-op), 0..1 = amount.
detail_transfer_strength = 0.0
# Persistent source identity detail from FaceSet V2. Unlike the setting above,
# this never reads the target crop's texture and is confidence/visibility gated.
# 0 = off (bit-identical legacy path), 0..1 = controlled restoration strength.
identity_detail_strength = 0.0
# Eye restore: composite the TARGET's own eyes back over the swapped result.
# See ProcessMgr.apply_eyes_area. Radii are fractions of interocular distance;
# feather is a percentage of the eye radius, so none of these need a per-clip
# retune as the face changes size on screen.
restore_original_eyes = False
eyes_blend_amount = 1.0     # 0..1 how much of the plate's eyes comes back
eyes_feather_blend = 25.0   # 0..100 edge softness, % of the eye radius
eyes_size_factor = 1.0      # overall scale of both ellipses
eyes_radius_x = 1.0         # width only
eyes_radius_y = 1.0         # height only
# Face Parser mask regions — see roop/processors/Mask_FaceParser.py.
# None means "the default set", which is the inner face; a list of group names
# overrides it. parser_region_grow dilates a group before the union, in pixels
# of the 512x512 parse.
parser_regions = None
parser_region_grow = None
# Re-warp the swap crop into the enhancer's own training alignment before
# restoring, and back afterwards. See Enhance_CodeFormer.model_template for the
# measured mismatch. Off = the crop is handed over as-is, which is what every
# render before this did.
enhancer_align = False
# Run the colour transfer a SECOND time, on the enhancer's output — the first
# pass runs before it and the restorer regrades what it is given.
color_match_after_enhance = False
# ── DeepFaceLab merger post-ops (roop/procmgr_merger.py) ─────────────────────
# Cheap CPU passes over the merged crop, applied just before paste-back. Every
# one is a bit-identical no-op at the value below, and the chain short-circuits
# when they all are, so the defaults cost one attribute read each.
#   hist_match  0..1  match the plate's per-channel histogram (a 4th colour
#                     family: rct/lct/mkl match moments, this matches shape)
#   sharpen    -1..1  signed unsharp; negative softens (the denoise direction)
#   motion_blur 0..1  directional blur, axis measured off the plate's own smear
#   grain_match 0..1  add noise matched to the plate's measured noise floor —
#                     the swap comes back cleaner than the footage around it
#   degrade     0..1  bicubic down-and-up, to match a soft/compressed plate
merger_hist_match = 0.0
merger_sharpen = 0.0
merger_motion_blur = 0.0
merger_grain_match = 0.0
merger_degrade = 0.0
merger_clarity = 0.0
# Grow/shrink the pasted face about its own centre (DFL's output_face_scale).
# Identity swappers keep the TARGET's head size, so this is the only lever when
# the source person's face is visibly narrower or broader. 0 = off (no-op).
output_face_scale = 0.0
# Expression restorer (LivePortrait): put the TARGET's expression back onto the
# swapped face. Swappers regress toward their training set's mean expression, so
# laughing/crying/grimacing get flattened; this reads the expression deformation
# off the original frame and re-applies it. 0 = off (bit-exact no-op), 1 = adopt
# the target's expression fully, >1 exaggerates past it. Region limits the
# transfer to 'all' | 'lips' | 'eyes'.
expression_restore_strength = 0.0
expression_restore_region = 'all'
# Lip-sync (MuseTalk): regenerate the mouth region to match a driving audio
# track, post-composite (same slot as restore_original_mouth — the two are
# mutually exclusive, see ProcessMgr.process_face). audio_source picks what
# drives it: 'original' re-syncs to the target video's own audio (repairs
# drift the swap itself introduces), 'upload' dubs against
# lipsync_audio_path instead. Off by default (bit-exact no-op).
lipsync_enabled = False
lipsync_audio_source = 'original'
lipsync_audio_path = None
# Small-face rescue: when a frame yields no detections, retry on a 2x upscale
# so tiny/distant faces get picked up (without raising the global det size).
rescue_small_faces = False
# Face detector engine: 'scrfd' (insightface default), 'yoloface' (better on
# steep profiles / occluded faces) or 'retinaface' (highest recall on hard
# poses/lighting). The alternates reuse buffalo_l's aux models for identity +
# landmarks.
detector_engine = 'scrfd'
# Temporal detection (video anti-flicker): pre-pass detects+tracks every frame,
# gap-fills short detection misses (<= ROOP_TEMPORAL_GAP frames) and smooths
# kps/lm106/bbox per track when stabilization is on; the swap pass then reads
# the cached faces instead of re-detecting, so the swap can't blink out.
temporal_detection = False

no_face_action = 1

processing = False
# True for the whole lifetime of a batch_process run (start → end_processing),
# independent of the `processing` stop-signal. `processing` is cleared the moment
# a stop is requested (UI Stop or Ctrl-C), but the background thread still needs a
# moment to wind down and finalize the output video. `batch_active` stays True
# until that teardown completes, so a terminal Ctrl-C can wait for the video to be
# finalized (ffmpeg moov atom written) before exiting — same as the UI Stop path.
batch_active = False
# When True, the per-frame processing loops block (without aborting) so the
# user can pause and later resume from the exact same spot. Stop clears both
# `processing` and `pause`, so an abort always wins over a pause.
pause = False

g_current_face_analysis = None
g_desired_face_analysis = None
# True when landmark_3d_68 is loaded for AUTOROTATE ALONE, in which case it is
# kept out of the per-face analysis loop and run on demand — see
# face_util._lm68_should_measure and ProcessMgr.initialize.
lm68_lazy = False

FACE_ENHANCER = 'GPEN'

INPUT_FACESETS = []
TARGET_FACES = []
# Parallel to TARGET_FACES: the person/group id each target face belongs to.
# Multiple angles of the same person share a group id; each group maps (by rank)
# to one source faceset. Enables multi-angle target tracking (anti-flicker).
TARGET_FACE_GROUP: List[int] = []
# Optional human-friendly names keyed by RAW group id (e.g. {0: "Bride"}).
# Ranks shift when groups merge/split, but the raw group id is stable, so we key
# names by raw id and resolve to per-rank names in the API payload.
TARGET_FACE_NAMES: dict = {}


IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None

use_3d_recon = False
