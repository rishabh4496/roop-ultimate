import os
import threading
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


_SAM2_IMG_MEAN = (0.485, 0.456, 0.406)
_SAM2_IMG_STD = (0.229, 0.224, 0.225)
_loader_patch_lock = threading.Lock()


class SAM2FrameBuffer:
    """The clip's frames, preprocessed for SAM2 and held in RAM.

    SAM2's video predictor only accepts a folder of JPEGs, which used to mean
    encoding every frame of the clip to disk and having SAM2 decode them all
    again. This builds the exact tensor SAM2's own loader would build
    (``load_video_frames_from_jpg_images``: PIL RGB resize to image_size^2,
    /255, then mean/std) straight from the decoded frames, minus the lossy JPEG
    round trip. The tensor is the same size SAM2 allocates itself
    (N x 3 x S x S float32, ~12.6 MB a frame at 1024), so this adds no memory
    and removes the disk traffic.

    Pass the trimmed frame count as *capacity* so the tensor is allocated once;
    without it the frames are collected and stacked at the end.
    """

    def __init__(self, image_size, capacity=0):
        import torch
        self.image_size = int(image_size)
        self.count = 0
        self.height = 0
        self.width = 0
        self._capacity = max(0, int(capacity))
        self._images = (torch.zeros(self._capacity, 3, self.image_size,
                                    self.image_size, dtype=torch.float32)
                        if self._capacity else None)
        self._overflow = []

    def add(self, bgr_frame):
        """Append one BGR uint8 frame (what cv2 hands back)."""
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # Same calls, same default resample, as sam2.utils.misc._load_img_as_tensor
        img_pil = Image.fromarray(rgb)
        arr = np.array(img_pil.convert('RGB').resize((self.image_size, self.image_size)))
        tensor = torch.from_numpy(arr / 255.0).permute(2, 0, 1)
        if self.count == 0:
            self.width, self.height = img_pil.size
        if self._images is not None and self.count < self._capacity:
            self._images[self.count] = tensor
        else:
            self._overflow.append(tensor.to(torch.float32))
        self.count += 1

    def finalize(self):
        """Normalised (images, video_height, video_width), as SAM2's loader returns."""
        import torch
        if self.count == 0:
            raise RuntimeError('SAM2FrameBuffer is empty')
        parts = []
        if self._images is not None:
            parts.append(self._images[:min(self.count, self._capacity)])
        if self._overflow:
            parts.append(torch.stack(self._overflow))
            self._overflow = []
        images = parts[0] if len(parts) == 1 else torch.cat(parts)
        self._images = None
        mean = torch.tensor(_SAM2_IMG_MEAN, dtype=torch.float32)[:, None, None]
        std = torch.tensor(_SAM2_IMG_STD, dtype=torch.float32)[:, None, None]
        images -= mean
        images /= std
        return images, self.height, self.width


class _in_memory_frames:
    """Make SAM2's ``init_state`` read *buffer* instead of a JPEG folder.

    ``init_state`` calls the module-level ``load_video_frames`` with no hook for
    anything but a path, so it is swapped for the duration of the call (under a
    lock, and always restored). The loader is only ever used by init_state.
    """

    def __init__(self, buffer):
        self.buffer = buffer

    def __enter__(self):
        import sam2.sam2_video_predictor as svp
        _loader_patch_lock.acquire()
        self._svp = svp
        self._original = svp.load_video_frames
        buffer = self.buffer

        def _load(video_path=None, image_size=None, offload_video_to_cpu=True,
                  compute_device=None, **_ignored):
            if image_size is not None and int(image_size) != buffer.image_size:
                raise RuntimeError(
                    'SAM2FrameBuffer was built for image_size=%d but the predictor '
                    'wants %d' % (buffer.image_size, int(image_size)))
            images, height, width = buffer.finalize()
            if not offload_video_to_cpu and compute_device is not None:
                images = images.to(compute_device)
            return images, height, width

        svp.load_video_frames = _load
        return self

    def __exit__(self, *exc):
        try:
            self._svp.load_video_frames = self._original
        finally:
            _loader_patch_lock.release()
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

    def new_frame_buffer(self, capacity=0):
        """An empty in-RAM frame buffer sized for this predictor's input."""
        return SAM2FrameBuffer(self.predictor.image_size, capacity)

    def precompute(self, frames, boxes, orig_hw):
        """Track the faces (given as frame-0 xyxy boxes) across the clip. Fills
        self.precomputed with one downscaled full-frame mask per frame index.
        *orig_hw* is the (H, W) of the original frames, used to upscale the cached
        mask back before warping.

        *frames* is a SAM2FrameBuffer (the frames already preprocessed in RAM, no
        files written) or, for callers that still have one, a directory of
        %06d.jpg frames numbered from 0."""
        import torch
        self.precomputed = {}
        self._orig_hw = orig_hw
        if not boxes:
            return
        in_memory = isinstance(frames, SAM2FrameBuffer)
        autocast = torch.autocast(self.device, dtype=torch.bfloat16) if self.device == 'cuda' \
            else _nullcontext()
        with torch.inference_mode(), autocast:
            loader = _in_memory_frames(frames) if in_memory else _nullcontext()
            with loader:
                state = self.predictor.init_state(
                    video_path=None if in_memory else frames,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
            for obj_id, box in enumerate(boxes):
                self.predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=obj_id, box=np.asarray(box, np.float32))
            total_frames = frames.count if in_memory else len(os.listdir(frames))
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
