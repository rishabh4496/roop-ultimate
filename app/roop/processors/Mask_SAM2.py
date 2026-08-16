import os
import numpy as np
import cv2
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download


# Segment Anything 2 (SAM2) as a TEMPORALLY-TRACKED face-mask engine. Unlike the
# per-frame engines (MobileSAM/FastSAM/XSeg/BiSeNet) which mask each crop
# independently, SAM2 tracks the face as a video object across the whole clip with
# its memory attention, producing temporally-consistent masks → far less mask
# flicker frame-to-frame.
#
# This does NOT fit the stateless per-crop Run(crop) contract, so it works in two
# phases, orchestrated by ProcessMgr:
#   1. precompute(): a sequential pre-pass over the trimmed frames. We detect the
#      faces on frame 0, hand SAM2 a box per face, and propagate to get a
#      full-frame mask for every frame, cached in self.precomputed (downscaled).
#   2. get_crop_mask(): during the (still multi-threaded) swap, ProcessMgr warps
#      the cached full-frame mask into the aligned-crop space via the same affine
#      M used to make the crop. Read-only, so the swap stays parallel.
#
# Video-only: on stills there is nothing to track, so it no-ops (full-crop swap).
_CKPTS = {
    'tiny':      ('sam2.1_hiera_tiny.pt',      'configs/sam2.1/sam2.1_hiera_t.yaml',
                  'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'),
    'small':     ('sam2.1_hiera_small.pt',     'configs/sam2.1/sam2.1_hiera_s.yaml',
                  'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'),
    'base_plus': ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml',
                  'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt'),
    'large':     ('sam2.1_hiera_large.pt',     'configs/sam2.1/sam2.1_hiera_l.yaml',
                  'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'),
}
_MASK_MAXSIDE = 640   # cap cached full-frame mask resolution to bound memory


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class Mask_SAM2():
    plugin_options: dict = None

    processorname = 'mask_sam2'
    type = 'mask'

    def __init__(self):
        self.predictor = None
        self._loaded_size = None
        self.device = 'cpu'
        self._orig_hw = None
        # {frame_idx (0-based within trim): uint8 mask at <=_MASK_MAXSIDE}
        self.precomputed = None

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        size = getattr(roop.globals, 'sam2_model_size', 'tiny') or 'tiny'
        if size not in _CKPTS:
            size = 'tiny'
        if self.predictor is None or self._loaded_size != size:
            self._load(size)

    def _load(self, size):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        fname, cfg, url = _CKPTS[size]
        model_dir = resolve_relative_path('../models/sam2')
        conditional_download(model_dir, [url])
        ckpt = os.path.join(model_dir, fname)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.predictor = build_sam2_video_predictor(cfg, ckpt, device=self.device)
        self._loaded_size = size
        print(f'[SAM2] loaded {size} on {self.device}')

    def precompute(self, frames_dir, boxes, orig_hw):
        """Track the faces (given as frame-0 xyxy boxes) across the JPEG frames in
        *frames_dir* (named %06d.jpg from 0). Fills self.precomputed with one
        downscaled full-frame mask per frame index. *orig_hw* is the (H, W) of the
        original frames, used to upscale the cached mask back before warping."""
        import torch
        self.precomputed = {}
        self._orig_hw = orig_hw
        if not boxes:
            return
        autocast = torch.autocast(self.device, dtype=torch.bfloat16) if self.device == 'cuda' \
            else _nullcontext()
        with torch.inference_mode(), autocast:
            state = self.predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            for obj_id, box in enumerate(boxes):
                self.predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=obj_id, box=np.asarray(box, np.float32))
            total_frames = len(os.listdir(frames_dir))
            for fidx, _obj_ids, logits in self.predictor.propagate_in_video(state):
                # logits: (num_objs, 1, H, W). Union the objects → one region mask.
                m = (logits > 0.0).any(dim=0)[0].detach().cpu().numpy().astype(np.uint8)
                h, w = m.shape
                scale = _MASK_MAXSIDE / max(h, w)
                if scale < 1.0:
                    m = cv2.resize(m, (max(1, int(w * scale)), max(1, int(h * scale))),
                                   interpolation=cv2.INTER_NEAREST)
                self.precomputed[int(fidx)] = m
                if fidx % 50 == 0 or fidx == total_frames - 1:
                    pct = (fidx + 1) / total_frames * 100 if total_frames > 0 else 0.0
                    print(f'[SAM2] mask propagation: frame {fidx + 1}/{total_frames} ({pct:.1f}%)')
        print(f'[SAM2] precomputed masks for {len(self.precomputed)} frames')

    def get_crop_mask(self, frame_idx, M, crop_shape):
        """Return the roop-convention mask (1.0 = keep ORIGINAL) for the aligned
        crop, by upscaling the cached full-frame mask to the original size and
        warping it into crop space via M. Falls back to all-zeros (swap the whole
        crop, i.e. no extra masking) when this frame wasn't tracked — e.g. stills,
        or a face absent from frame 0."""
        ch, cw = crop_shape[:2]
        full = None
        if self.precomputed is not None and frame_idx is not None:
            full = self.precomputed.get(int(frame_idx))
        if full is None or M is None or self._orig_hw is None:
            return np.zeros((ch, cw), np.float32)
        H, W = self._orig_hw
        if full.shape[:2] != (H, W):
            full = cv2.resize(full, (W, H), interpolation=cv2.INTER_LINEAR)
        # M maps ORIGINAL-frame coords → crop coords (same matrix align_crop used).
        crop = cv2.warpAffine(full.astype(np.float32), M, (cw, ch), flags=cv2.INTER_LINEAR)
        face = (crop > 0.5).astype(np.float32)
        face = cv2.GaussianBlur(face, (0, 0), sigmaX=3)
        face = np.clip(face, 0.0, 1.0)
        return (1.0 - face).astype(np.float32)

    def Run(self, img1, keywords: str) -> Frame:
        # Not used in the SAM2 path (ProcessMgr calls get_crop_mask instead). Safe
        # no-op so a stray call masks nothing rather than blanking the swap.
        return np.zeros(img1.shape[:2], np.float32)

    def Release(self):
        self.precomputed = None
        self.predictor = None
        self._loaded_size = None
        self._orig_hw = None
