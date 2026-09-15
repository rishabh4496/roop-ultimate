"""GPEN Ultimate — GPEN-512 with forced FFHQ alignment and an anti-halo finish.

WHAT THIS ADDS OVER `GPEN`. Nothing in the network: this opens the same
`GPEN-BFR-512.onnx` weights through the same pooled, io-binding session path as
`Enhance_GPEN`, so its VRAM footprint, provider policy, FP32/FP16 handling and
non-finite guard are inherited unchanged. The two differences are both about
what the network is HANDED and what happens to what it returns:

1. FORCED ALIGNMENT (`force_align = True`). GPEN learned its prior on
   FFHQ-aligned 512 faces, but the crop an enhancer receives is whatever the
   SWAPPER's template produced — off by ~17% in scale and 31 px in eye height
   for the inswapper family. `enhancer_align` makes that correction opt-in for
   every restorer; this profile declares it mandatory, because both of its
   finishing stages are keyed to the FFHQ template's own eye coordinates and are
   only anatomically correct on a crop that is actually in that space. See
   ProcessMgr's enhancer-alignment block for the two affines and why the ring
   outside the swap crop is filled from the PLATE rather than replicated.

2. THE ULTIMATE FINISH (`enhance_gpen_ultimate`). One bilateral detail
   injection, one eye-clarity pass and one anti-halo sharpen, all bounded
   against their own input's local 3x3 min/max envelope so no overshoot exists
   for a feather to spread into a halo ring. Pure CPU post-processing on the
   returned uint8 crop: no second inference, no extra engine, no extra VRAM.

Exactly ONE inference and ONE finish run per face — the finish is applied here,
in `Run`, and never inside the shared base class, so selecting plain `GPEN`
remains bit-identical to what it was before this profile existed.
"""

import threading

import cv2

from roop.processors.Enhance_GPEN import Enhance_GPEN
from roop.processors.enhance_common import enhance_gpen_ultimate
from roop.typing import Face, FaceSet, Frame


class Enhance_GPENUltimate(Enhance_GPEN):
    """An ultimate quality profile built on GPEN-512 with forced FFHQ alignment.

    Features:
    - Forced alignment with the original face geometry via target keypoints.
    - Native 512px resolution (matching the swap crop, avoiding 256px blur).
    - Dedicated eye clarity boost with anti-halo bounding (zero halos around
      the eyes).
    - Bilateral edge-preserving detail transfer and anti-halo sharpening for
      crisp skin texture.
    - Pooled multi-context TensorRT inference, inherited from Enhance_GPEN.
    """

    processorname = 'gpen_ultimate'
    # Mandatory, not opt-in: the finish below is keyed to the FFHQ template's
    # own eye coordinates. See the module docstring.
    force_align = True
    # FFHQ-trained — see Enhance_CodeFormer.model_template.
    model_template = 'ffhq_512'
    # A DISTINCT lock from Enhance_GPEN's. `_session_lock` is a class attribute,
    # so inheriting it would make this profile and plain GPEN serialise against
    # each other's sessions for no reason -- they hold separate sessions, and
    # each lock only has to protect its own.
    _session_lock = threading.Lock()

    def Initialize(self, plugin_options: dict):
        options = dict(plugin_options)
        # 512 is the classic GPEN weight and matches the swap crop size, so the
        # paste needs no resampling. A caller may still request another tier.
        options.setdefault("size", 512)
        super().Initialize(options)

    def Run(self, source_faceset: FaceSet, target_face: Face,
            temp_frame: Frame) -> Frame:
        # The reference is the crop as HANDED IN -- i.e. the already-swapped,
        # already-realigned face. Taking it from the target plate instead would
        # inject the ORIGINAL identity's texture back over the swap.
        reference = temp_frame
        result, scale_factor = super().Run(source_faceset, target_face,
                                          temp_frame)
        if result is None or reference is None:
            return result, scale_factor
        # `sized()` may have returned a buffer LARGER than the input crop (the
        # 1024/2048 tiers report scale_factor 2/4), so the reference has to be
        # brought up to the result's geometry before any per-pixel operation.
        if reference.shape[:2] != result.shape[:2]:
            reference = cv2.resize(reference,
                                   (result.shape[1], result.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)
        result = enhance_gpen_ultimate(result, reference,
                                       target_face=target_face)
        return result, scale_factor
