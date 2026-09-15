"""Restore Ultra — RestoreFormer++ with forced FFHQ alignment and ultra clarity.

WHAT THIS ADDS OVER `Restoreformer++`. Nothing in the network: this opens the
same `restoreformer_plus_plus.onnx` weights through the same pooled io-binding
session path as `Enhance_RestoreFormerPPlus`, inheriting its provider policy,
its multi-context SessionPool and its non-finite output guard unchanged. The two
differences are what the network is HANDED and what happens to what it returns:

1. FORCED ALIGNMENT (`force_align = True`). RestoreFormer++ learned its prior on
   FFHQ-aligned 512 faces, but an enhancer receives whatever crop the SWAPPER's
   template produced. `enhancer_align` makes that correction opt-in globally;
   this profile declares it mandatory, because the eye-clarity stage below is
   keyed to the FFHQ template's own eye coordinates and is only anatomically
   correct on a crop that is genuinely in that space. See ProcessMgr's
   enhancer-alignment block for the affine round-trip.

2. THE ULTRA FINISH (`enhance_restore_ultra`). A gentler set of constants than
   GPEN Ultimate's (bilateral sigma 18 / knee 10 / sharpen sigma 0.8 against
   22 / 12 / 1.0), because RestoreFormer++ already returns more micro-contrast
   than GPEN and the same strengths would read as over-sharpened. Every stage is
   bounded against its own input's local 3x3 min/max envelope, so no overshoot
   exists at any radius for a feather to spread into a halo ring. Pure CPU
   post-processing: no second inference, no extra engine, no extra VRAM.

Exactly ONE inference and ONE finish per face.
"""

import threading

import cv2

from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
from roop.processors.enhance_common import enhance_restore_ultra
from roop.typing import Face, FaceSet, Frame


class Enhance_RestoreUltra(Enhance_RestoreFormerPPlus):
    """An ultra-high-definition fidelity profile over the RestoreFormer++ weights.

    Features:
    - Forced FFHQ alignment based on the original target face keypoints,
      guaranteeing exact anatomical feature registration with original faces.
    - Dedicated eye clarity boost with anti-halo bounding, producing
      crystal-clear, expressive eyes with rich iris definition and natural
      catchlights (zero halo rings).
    - Edge-preserving fine-line sharpening for eyelashes, eyebrows and lip
      borders without amplifying noise or creating plastic artifacts.
    - Pooled multi-context TensorRT execution for concurrent worker throughput,
      inherited from Enhance_RestoreFormerPPlus.
    """

    processorname = 'restore_ultra'
    # Mandatory, not opt-in: the eye stage is keyed to the FFHQ template's own
    # eye coordinates. See the module docstring.
    force_align = True
    # FFHQ-trained — see Enhance_CodeFormer.model_template.
    model_template = 'ffhq_512'
    # A DISTINCT lock and session from Enhance_RestoreFormerPPlus's. Both are
    # CLASS attributes on the base, so without these two declarations this
    # profile and plain Restoreformer++ would share one session -- and whichever
    # was selected second would silently reuse the first one's state and its
    # lock. Re-declaring them keeps each subclass's session and mutex its own.
    _session_lock = threading.Lock()
    model_restoreformerpplus = None
    plugin_options = None
    pool = None

    def Run(self, source_faceset: FaceSet, target_face: Face,
            temp_frame: Frame) -> Frame:
        # The reference is the crop as HANDED IN -- the already-swapped,
        # already-realigned face. Taking it from the target plate instead would
        # inject the ORIGINAL identity's texture back over the swap.
        reference = temp_frame
        result, scale_factor = super().Run(source_faceset, target_face,
                                          temp_frame)
        if result is None or reference is None:
            return result, scale_factor
        # `sized()` can return a buffer larger than the input crop, so bring the
        # reference to the result's geometry before any per-pixel operation.
        if reference.shape[:2] != result.shape[:2]:
            reference = cv2.resize(reference,
                                   (result.shape[1], result.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)
        result = enhance_restore_ultra(result, reference,
                                       target_face=target_face)
        return result, scale_factor
