"""Stage 16 — Hardware-Adaptive Quality Profiles.

Each profile is a frozen, benchmark-derived descriptor covering the full
pipeline: detector, swap model, enhancer, mask, temporal stabilization,
codec modes and scheduling hints.  The AUTO resolver picks and adjusts a
base profile from six measured signals so that the right trade-off is made
automatically for each hardware class.

Benchmark sources
-----------------
All numeric constants come from the ``docs/SESSION_LOGS.md`` and
``docs/PHASE_HANDOFF.md`` validation campaigns:

* RTX 4070 12 GB (main device) — 18–31 fps at 1280×720, 77–100% GPU,
  130–140 W, realswap + GPEN 256 Pro.  TRT mixed precision: +20% over FP32.
  NVDEC: 300+ fps decode.  SCRFD 2.5g: <3 ms; SCRFD 10g / RetinaFace: 8–12 ms.
  Tracking_frequency=2: up to 40% faster detector phase on static footage.
  Batch size 8→16 cross-frame on 4070, halved when VRAM is tight.

* RTX 3060 Laptop 6 GB (secondary device) — single-context (pool 0/0), RSS
  strict < 2.5 GB, CPU decoder, no TRT CUDA graphs, no enhancer by policy.
  Pre-pass fix lifted 21.4→30.1 fps (+40%) in the swap phase alone.

"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

from roop.degrade import swallowed as _swallowed

logger = logging.getLogger("roop.quality_profiles")

# ---------------------------------------------------------------------------
# Optional heavyweight imports — psutil / torch not required at import time
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil  # type: ignore
except Exception as exc:  # noqa: BLE001
    _swallowed("quality_profiles.import_psutil", exc, "VRAM/RAM queries will be skipped")
    _psutil = None  # type: ignore[assignment]

try:
    import torch as _torch  # type: ignore
except Exception as exc:  # noqa: BLE001
    _swallowed("quality_profiles.import_torch", exc, "GPU name/VRAM queries will use fallback")
    _torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityProfile:
    """A complete, benchmark-derived pipeline quality descriptor.

    All 15 pipeline fields are required.  ``benchmark_basis`` is a free-text
    string documenting which measured run(s) the values were derived from.
    ``thread_count`` is a scheduler hint only — it is intentionally excluded
    from ``to_settings_patch()`` because it is not a direct config key.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    name: str

    # ── Detection ─────────────────────────────────────────────────────────
    detector: str                   # "scrfd_2.5g" | "scrfd_10g" | "retinaface_r50"
    detector_resolution: int        # pixels, e.g. 320 / 512 / 640
    tracking_frequency: int         # 1 = every frame, 2 = every 2nd …

    # ── Alignment ─────────────────────────────────────────────────────────
    alignment_quality: str          # "draft" | "standard" | "precise" | "ultra"

    # ── Swap ──────────────────────────────────────────────────────────────
    swap_model: str                 # "hyperswap" | "realswap"
    swap_precision: str             # "fp16" | "mixed" | "fp32"

    # ── Enhancement ───────────────────────────────────────────────────────
    enhancer: str                   # "none" | "gpen_256" | "gpen_256_pro" | …
    enhancer_strength: float        # 0.0 – 1.0  (stabilize_enhancer_strength)

    # ── Masking ───────────────────────────────────────────────────────────
    xseg_mode: str                  # "none" | "fast" | "quality"
    mask_quality: str               # "draft" | "standard" | "precise"

    # ── Temporal / scheduling ─────────────────────────────────────────────
    temporal_stabilization: bool
    batch_size: int                 # cross-frame batch ceiling
    thread_count: int               # worker thread hint (NOT written to settings)

    # ── I/O codec ─────────────────────────────────────────────────────────
    decode_mode: str                # "nvdec" | "cpu" | "auto"
    encode_mode: str                # "nvenc_hevc" | "nvenc_h264" | "libx264" | …

    # ── Provenance ────────────────────────────────────────────────────────
    benchmark_basis: str = ""       # free-text, non-empty for all shipped profiles

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict with all 15 pipeline fields (plus metadata)."""
        return asdict(self)

    def to_settings_patch(self) -> Dict[str, Any]:
        """Map profile fields to the settings keys consumed by app/settings.py.

        ``thread_count`` is intentionally excluded — it is a scheduler hint,
        not a direct ``config.yaml`` key.  ``encode_mode='auto'`` is also
        excluded so that an existing codec selection is left unchanged.
        """
        patch: Dict[str, Any] = {}

        # Detection
        patch["detector_engine"] = self.detector
        patch["face_detector_size"] = str(self.detector_resolution)
        patch["temporal_step"] = self.tracking_frequency

        # Alignment
        _ALIGN = {
            "draft":    {"enhancer_align": False},
            "standard": {"enhancer_align": True,  "use_frontalization": False},
            "precise":  {"enhancer_align": True,  "use_frontalization": True},
            "ultra":    {"enhancer_align": True,  "use_frontalization": True,
                         "use_3d_recon": True},
        }
        patch.update(_ALIGN.get(self.alignment_quality, {}))

        # Swap
        patch["swap_model"] = self.swap_model
        patch["trt_precision"] = self.swap_precision

        # Enhancer
        _ENH_MAP = {
            "none":          "None",
            "gpen_256":      "GPEN 256",
            "gpen_256_pro":  "GPEN 256 Pro",
            "gpen_512":      "GPEN 512",
            "codeformer":    "CodeFormer",
            "restore_ultra": "Restore Ultra",
        }
        patch["selected_enhancer"] = _ENH_MAP.get(self.enhancer, "None")
        patch["stabilize_enhancer_strength"] = self.enhancer_strength

        # Mask
        _XSEG_MAP = {
            "none":    "None",
            "fast":    "DFL XSeg",
            "quality": "RealityUX",
        }
        patch["mask_engine"] = _XSEG_MAP.get(self.xseg_mode, "None")
        _MQ_MAP = {
            "draft":    35,
            "standard": 25,
            "precise":  12,
        }
        patch["face_mask_blend"] = _MQ_MAP.get(self.mask_quality, 25)

        # Temporal
        patch["temporal_detection"] = self.temporal_stabilization

        # Batch
        patch["perf_batch_max"] = self.batch_size

        # Decode
        _DEC_MAP = {
            "nvdec": "on",
            "cpu":   "off",
            "auto":  "auto",
        }
        patch["perf_nvdec"] = _DEC_MAP.get(self.decode_mode, "auto")

        # Encode — leave unchanged for "auto"
        _ENC_MAP = {
            "nvenc_hevc": "hevc_nvenc",
            "nvenc_h264": "h264_nvenc",
            "libx264":    "libx264",
            "libx265":    "libx265",
        }
        if self.encode_mode in _ENC_MAP:
            patch["output_video_codec"] = _ENC_MAP[self.encode_mode]

        return patch


# ---------------------------------------------------------------------------
# The four shipped profiles
# ---------------------------------------------------------------------------

#: Maximum throughput, quality traded for speed.
#: Basis: SCRFD 2.5g <3 ms measured; tracking_frequency=2 → up to 40% faster
#: detector phase on static footage; no enhancer = largest single cost removed;
#: FP16 for maximum throughput; NVDEC 300+ fps decode.
FAST_PROFILE = QualityProfile(
    name="FAST",
    detector="scrfd_2.5g",
    detector_resolution=320,
    tracking_frequency=2,
    alignment_quality="draft",
    swap_model="hyperswap",
    swap_precision="fp16",
    enhancer="none",
    enhancer_strength=0.0,
    xseg_mode="none",
    mask_quality="draft",
    temporal_stabilization=False,
    batch_size=16,
    thread_count=20,
    decode_mode="nvdec",
    encode_mode="nvenc_hevc",
    benchmark_basis=(
        "SESSION_LOGS.md — 4070 detector fast-path: SCRFD 2.5g <3 ms; "
        "tracking_frequency=2 → up to 40% faster detector phase; "
        "enhancer removal = single largest frame-time reduction; "
        "FP16 throughput; NVDEC 300+ fps decode."
    ),
)

#: Sweet-spot default — matches the live config.yaml at time of Stage 16.
#: Basis: GPEN 256 = live config default; stabilize_enhancer_strength=0.6 from
#: config; DFL XSeg (mask_engine) = live default; mixed precision +20% on 4070;
#: SCRFD 10g 8–12 ms, good accuracy; detector_resolution 512 neutral vs 640.
BALANCED_PROFILE = QualityProfile(
    name="BALANCED",
    detector="scrfd_10g",
    detector_resolution=512,
    tracking_frequency=1,
    alignment_quality="standard",
    swap_model="hyperswap",
    swap_precision="mixed",
    enhancer="gpen_256",
    enhancer_strength=0.6,
    xseg_mode="fast",
    mask_quality="standard",
    temporal_stabilization=True,
    batch_size=8,
    thread_count=12,
    decode_mode="nvdec",
    encode_mode="nvenc_hevc",
    benchmark_basis=(
        "SESSION_LOGS.md — live config.yaml at Stage 16: "
        "GPEN 256, stabilize_enhancer_strength=0.6, DFL XSeg mask; "
        "TRT mixed precision +20% on 4070; SCRFD 10g 8–12 ms; "
        "detector_resolution 512 neutral vs 640 (measured)."
    ),
)

#: High visual fidelity — measured live production configuration on 4070.
#: Basis: realswap + GPEN 256 Pro + RealityUX → 77–100% GPU, 130–140 W,
#: 18–31 fps at 1280×720 on RTX 4070; RetinaFace for superior angle recall;
#: restore_ultra_detail_weight=0.75 from config.
QUALITY_PROFILE = QualityProfile(
    name="QUALITY",
    detector="retinaface_r50",
    detector_resolution=512,
    tracking_frequency=1,
    alignment_quality="precise",
    swap_model="realswap",
    swap_precision="mixed",
    enhancer="gpen_256_pro",
    enhancer_strength=0.75,
    xseg_mode="quality",
    mask_quality="precise",
    temporal_stabilization=True,
    batch_size=4,
    thread_count=12,
    decode_mode="nvdec",
    encode_mode="nvenc_hevc",
    benchmark_basis=(
        "SESSION_LOGS.md — realswap + GPEN 256 Pro + RealityUX: "
        "77–100% GPU, 130–140 W, 18–31 fps at 1280×720 on RTX 4070; "
        "RetinaFace: 8–12 ms, best recall at extreme angles; "
        "enhancer_strength=0.75 from restore_ultra_detail_weight in config."
    ),
)

#: Maximum studio quality — every quality lever at maximum.
#: Basis: ULTRA extends QUALITY with Restore Ultra (highest detail recovery),
#: detector at 640 (maximum precision), alignment_quality='ultra' (+3D recon),
#: enhancer_strength=1.0.
ULTRA_PROFILE = QualityProfile(
    name="ULTRA",
    detector="retinaface_r50",
    detector_resolution=640,
    tracking_frequency=1,
    alignment_quality="ultra",
    swap_model="realswap",
    swap_precision="mixed",
    enhancer="restore_ultra",
    enhancer_strength=1.0,
    xseg_mode="quality",
    mask_quality="precise",
    temporal_stabilization=True,
    batch_size=4,
    thread_count=12,
    decode_mode="nvdec",
    encode_mode="nvenc_hevc",
    benchmark_basis=(
        "SESSION_LOGS.md — Restore Ultra = selected_enhancer 'Restore Ultra' "
        "from config; detector 640 = maximum precision; "
        "all quality levers at maximum (3D recon, RealityUX, realswap)."
    ),
)


#: Registry of all named profiles — key is upper-cased profile name.
ALL_PROFILES: Dict[str, QualityProfile] = {
    "FAST":     FAST_PROFILE,
    "BALANCED": BALANCED_PROFILE,
    "QUALITY":  QUALITY_PROFILE,
    "ULTRA":    ULTRA_PROFILE,
}


def get_profile(name: str) -> QualityProfile:
    """Return a named profile, case-insensitively.

    Raises
    ------
    KeyError
        If *name* does not match any registered profile.
    """
    key = name.upper()
    if key not in ALL_PROFILES:
        raise KeyError(
            f"Unknown quality profile {name!r}. "
            f"Valid names: {sorted(ALL_PROFILES)}"
        )
    return ALL_PROFILES[key]


# ---------------------------------------------------------------------------
# AUTO resolver
# ---------------------------------------------------------------------------

@dataclass
class AutoProfileContext:
    """Six measured signals used by the AUTO resolver.

    All fields have safe defaults so callers can supply only what is known.
    """

    gpu_name: str = ""                      # GPU model string, e.g. "RTX 3060"
    vram_mb: float = 0.0                    # VRAM in MB; 0 = unknown / CPU-only
    input_resolution: Tuple[int, int] = (0, 0)   # (width, height)
    face_count: int = 1                     # faces per frame estimate
    swap_model: str = ""                    # currently selected swap model
    quality_hint: float = 0.5              # 0.0 draft … 1.0 archival

    # Convenience properties
    @property
    def is_cpu_only(self) -> bool:
        """True when no GPU is available (vram_mb == 0 and no CUDA)."""
        return self.vram_mb <= 0.0

    @property
    def is_laptop_tier(self) -> bool:
        """True for sub-7168 MB VRAM (3060 laptop tier and below)."""
        return 0.0 < self.vram_mb < 7168.0

    @property
    def is_high_end(self) -> bool:
        """True for ≥11520 MB VRAM (4070 tier and above)."""
        return self.vram_mb >= 11520.0

    @property
    def is_4k(self) -> bool:
        """True when the input is UHD 4K or wider."""
        w, h = self.input_resolution
        return w >= 3840 and h >= 2160

    @property
    def is_crowd(self) -> bool:
        """True when five or more faces appear per frame."""
        return self.face_count >= 5


def _build_auto_context() -> AutoProfileContext:
    """Try to read live GPU / RAM info and build a context object."""
    ctx = AutoProfileContext()

    try:
        if _torch is not None and _torch.cuda.is_available():
            ctx.vram_mb = _torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            ctx.gpu_name = _torch.cuda.get_device_properties(0).name
    except Exception as exc:  # noqa: BLE001
        _swallowed("quality_profiles.auto_ctx_gpu", exc, "GPU properties unavailable")

    return ctx


def resolve_auto_profile(ctx: Optional[AutoProfileContext] = None) -> QualityProfile:
    """Select and adjust a base profile from the six measured signals.

    Adjustment order (later rules override earlier ones where they conflict):

    1. Default base → BALANCED
    2. quality_hint ≥ 0.80 → QUALITY base
    3. quality_hint ≥ 0.95 → ULTRA base
    4. 4K input → detector_resolution=640, tracking_frequency=1,
       batch_size halved
    5. Crowd (face_count ≥ 5) → detector='scrfd_10g',
       xseg_mode='quality', temporal_stabilization=True
    6. Sub-7168 MB VRAM (3060 tier) → force enhancer='none',
       batch_size ≤ 2, decode_mode='cpu'
    7. CPU-only → FAST base, force decode_mode='cpu',
       encode_mode='libx264'

    The 3060 RSS safety rule (enhancer='none') is the only HARD constraint —
    it overrides every other quality lever.
    """
    if ctx is None:
        ctx = _build_auto_context()

    # ── 1. Default base ────────────────────────────────────────────────────
    base: QualityProfile = BALANCED_PROFILE

    # ── 2/3. Quality hint ─────────────────────────────────────────────────
    if ctx.quality_hint >= 0.95:
        base = ULTRA_PROFILE
    elif ctx.quality_hint >= 0.80:
        base = QUALITY_PROFILE

    # ── 7. CPU-only override (highest priority for base selection) ─────────
    if ctx.is_cpu_only:
        base = FAST_PROFILE

    # Work on a mutable dict so we can layer adjustments
    fields: Dict[str, Any] = asdict(base)
    fields.pop("name")          # replaced below
    fields.pop("benchmark_basis")

    # ── 4. 4K input ───────────────────────────────────────────────────────
    if ctx.is_4k:
        fields["detector_resolution"] = 640
        fields["tracking_frequency"] = 1
        fields["batch_size"] = max(1, fields["batch_size"] // 2)

    # ── 5. Crowd ──────────────────────────────────────────────────────────
    if ctx.is_crowd:
        fields["detector"] = "scrfd_10g"
        fields["xseg_mode"] = "quality"
        fields["temporal_stabilization"] = True

    # ── 6. 3060 / sub-7 GB VRAM ─────────────────────────────────────────
    if ctx.is_laptop_tier:
        fields["enhancer"] = "none"          # HARD rule: never enhance on 3060
        fields["enhancer_strength"] = 0.0
        fields["batch_size"] = min(fields["batch_size"], 2)
        fields["decode_mode"] = "cpu"

    # ── 7 (continued). CPU-only ───────────────────────────────────────────
    if ctx.is_cpu_only:
        fields["decode_mode"] = "cpu"
        fields["encode_mode"] = "libx264"
        fields["enhancer"] = "none"
        fields["enhancer_strength"] = 0.0

    basis_parts = [base.benchmark_basis, "AUTO resolver applied"]
    if ctx.is_cpu_only:
        basis_parts.append("CPU-only: decode=cpu, encode=libx264, enhancer=none")
    if ctx.is_laptop_tier:
        basis_parts.append("3060 RSS safety: enhancer forced none, batch≤2, decode=cpu")
    if ctx.is_4k:
        basis_parts.append("4K: detector_resolution=640, batch halved")
    if ctx.is_crowd:
        basis_parts.append("Crowd: detector=scrfd_10g, xseg=quality, temporal=True")

    return QualityProfile(
        name="AUTO",
        benchmark_basis="; ".join(basis_parts),
        **fields,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "QualityProfile",
    "AutoProfileContext",
    "FAST_PROFILE",
    "BALANCED_PROFILE",
    "QUALITY_PROFILE",
    "ULTRA_PROFILE",
    "ALL_PROFILES",
    "get_profile",
    "resolve_auto_profile",
]
