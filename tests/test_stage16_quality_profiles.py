"""Tests for Stage 16 — Hardware-Adaptive Quality Profiles.

Covers:
  1.  All 4 profiles present in ALL_PROFILES with correct names
  2.  Each profile has all 15 required fields
  3.  Field values within valid ranges
  4.  to_dict() returns dict with all 15 keys
  5.  to_settings_patch() includes expected keys
  6.  get_profile() is case-insensitive
  7.  get_profile('FAST') returns FAST_PROFILE
  8.  get_profile('invalid') raises KeyError
  9.  AutoProfileContext properties
  10. AUTO: CPU-only → decode_mode='cpu', encode_mode in ('libx264','libx265')
  11. AUTO: 3060 laptop → enhancer='none' ALWAYS
  12. AUTO: 4K → detector_resolution=640
  13. AUTO: crowd (≥5) → detector='scrfd_10g'
  14. AUTO: quality_hint=0.9 → QUALITY base (enhancer is gpen_256_pro or better)
  15. AUTO: quality_hint=0.97 → ULTRA base (enhancer is restore_ultra)
  16. AUTO: default 4070 → BALANCED base (enhancer is gpen_256)
  17. All 4 profiles are mutually distinct
  18. benchmark_basis is non-empty string for all 4 profiles
  19. FAST tracking_frequency > BALANCED tracking_frequency
  20. 3060 resolve: batch_size ≤ 4
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running directly or via pytest from the repo root
# ---------------------------------------------------------------------------
APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.quality_profiles import (  # noqa: E402
    ALL_PROFILES,
    BALANCED_PROFILE,
    FAST_PROFILE,
    QUALITY_PROFILE,
    ULTRA_PROFILE,
    AutoProfileContext,
    QualityProfile,
    get_profile,
    resolve_auto_profile,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS = {
    "detector",
    "detector_resolution",
    "tracking_frequency",
    "alignment_quality",
    "swap_model",
    "swap_precision",
    "enhancer",
    "enhancer_strength",
    "xseg_mode",
    "mask_quality",
    "temporal_stabilization",
    "batch_size",
    "thread_count",
    "decode_mode",
    "encode_mode",
}

_VALID_DETECTORS = {"scrfd_2.5g", "scrfd_10g", "retinaface_r50"}
_VALID_ALIGNMENT = {"draft", "standard", "precise", "ultra"}
_VALID_SWAP_MODELS = {"hyperswap", "realswap"}
_VALID_SWAP_PRECISION = {"fp16", "mixed", "fp32"}
_VALID_ENHANCERS = {"none", "gpen_256", "gpen_256_pro", "gpen_512", "codeformer", "restore_ultra"}
_VALID_XSEG = {"none", "fast", "quality"}
_VALID_MASK_Q = {"draft", "standard", "precise"}
_VALID_DECODE = {"nvdec", "cpu", "auto"}
_VALID_ENCODE = {"nvenc_hevc", "nvenc_h264", "libx264", "libx265", "auto"}

# Enhancer quality ordering for comparison tests
_ENH_RANK = {
    "none": 0,
    "gpen_256": 1,
    "gpen_256_pro": 2,
    "gpen_512": 3,
    "codeformer": 4,
    "restore_ultra": 5,
}

_FOUR_PROFILES = [FAST_PROFILE, BALANCED_PROFILE, QUALITY_PROFILE, ULTRA_PROFILE]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _profile_dict(p: QualityProfile) -> dict:
    d = p.to_dict()
    # Remove metadata keys not in the 15 required pipeline fields
    return {k: v for k, v in d.items() if k in _REQUIRED_FIELDS}


# ===========================================================================
# 1. All 4 profiles in ALL_PROFILES with correct names
# ===========================================================================

class TestProfileRegistry:
    def test_all_four_names_present(self):
        for name in ("FAST", "BALANCED", "QUALITY", "ULTRA"):
            assert name in ALL_PROFILES, f"{name} missing from ALL_PROFILES"

    def test_profile_name_matches_key(self):
        for key, profile in ALL_PROFILES.items():
            assert profile.name == key, f"Profile name mismatch: key={key}, name={profile.name}"

    def test_exactly_four_profiles(self):
        assert len(ALL_PROFILES) == 4


# ===========================================================================
# 2. Each profile has all 15 required fields
# ===========================================================================

class TestProfileCompleteness:
    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_has_all_15_fields(self, profile: QualityProfile):
        d = profile.to_dict()
        missing = _REQUIRED_FIELDS - set(d)
        assert not missing, f"{profile.name} is missing fields: {missing}"


# ===========================================================================
# 3. Field values within valid ranges
# ===========================================================================

class TestFieldRanges:
    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_detector_valid(self, profile):
        assert profile.detector in _VALID_DETECTORS, \
            f"{profile.name}.detector={profile.detector!r} not in {_VALID_DETECTORS}"

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_detector_resolution_positive(self, profile):
        assert profile.detector_resolution > 0

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_tracking_frequency_positive(self, profile):
        assert profile.tracking_frequency >= 1

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_alignment_quality_valid(self, profile):
        assert profile.alignment_quality in _VALID_ALIGNMENT

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_swap_model_valid(self, profile):
        assert profile.swap_model in _VALID_SWAP_MODELS

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_swap_precision_valid(self, profile):
        assert profile.swap_precision in _VALID_SWAP_PRECISION

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_enhancer_valid(self, profile):
        assert profile.enhancer in _VALID_ENHANCERS

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_enhancer_strength_range(self, profile):
        assert 0.0 <= profile.enhancer_strength <= 1.0

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_xseg_mode_valid(self, profile):
        assert profile.xseg_mode in _VALID_XSEG

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_mask_quality_valid(self, profile):
        assert profile.mask_quality in _VALID_MASK_Q

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_batch_size_positive(self, profile):
        assert profile.batch_size >= 1

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_thread_count_positive(self, profile):
        assert profile.thread_count >= 1

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_decode_mode_valid(self, profile):
        assert profile.decode_mode in _VALID_DECODE

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_encode_mode_valid(self, profile):
        assert profile.encode_mode in _VALID_ENCODE

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_temporal_stabilization_is_bool(self, profile):
        assert isinstance(profile.temporal_stabilization, bool)


# ===========================================================================
# 4. to_dict() returns dict with all 15 keys
# ===========================================================================

class TestToDict:
    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_to_dict_has_all_required_keys(self, profile):
        d = profile.to_dict()
        missing = _REQUIRED_FIELDS - set(d)
        assert not missing, f"to_dict() missing fields: {missing}"

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_to_dict_returns_dict(self, profile):
        assert isinstance(profile.to_dict(), dict)


# ===========================================================================
# 5. to_settings_patch() includes expected keys
# ===========================================================================

class TestToSettingsPatch:
    _REQUIRED_PATCH_KEYS = {
        "detector_engine",
        "face_detector_size",
        "temporal_step",
        "swap_model",
        "trt_precision",
        "selected_enhancer",
        "stabilize_enhancer_strength",
        "mask_engine",
        "face_mask_blend",
        "temporal_detection",
        "perf_batch_max",
        "perf_nvdec",
    }

    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_patch_has_required_keys(self, profile):
        patch = profile.to_settings_patch()
        missing = self._REQUIRED_PATCH_KEYS - set(patch)
        assert not missing, f"{profile.name}.to_settings_patch() missing: {missing}"

    def test_face_detector_size_is_string(self):
        patch = BALANCED_PROFILE.to_settings_patch()
        assert isinstance(patch["face_detector_size"], str)

    def test_detector_engine_mapped(self):
        patch = FAST_PROFILE.to_settings_patch()
        assert patch["detector_engine"] == "scrfd_2.5g"

    def test_enhancer_none_maps_to_None_string(self):
        patch = FAST_PROFILE.to_settings_patch()
        assert patch["selected_enhancer"] == "None"

    def test_enhancer_gpen_256_maps_correctly(self):
        patch = BALANCED_PROFILE.to_settings_patch()
        assert patch["selected_enhancer"] == "GPEN 256"

    def test_enhancer_gpen_256_pro_maps_correctly(self):
        patch = QUALITY_PROFILE.to_settings_patch()
        assert patch["selected_enhancer"] == "GPEN 256 Pro"

    def test_enhancer_restore_ultra_maps_correctly(self):
        patch = ULTRA_PROFILE.to_settings_patch()
        assert patch["selected_enhancer"] == "Restore Ultra"

    def test_xseg_none_maps_to_None_string(self):
        patch = FAST_PROFILE.to_settings_patch()
        assert patch["mask_engine"] == "None"

    def test_xseg_fast_maps_to_dfl_xseg(self):
        patch = BALANCED_PROFILE.to_settings_patch()
        assert patch["mask_engine"] == "DFL XSeg"

    def test_xseg_quality_maps_to_realityux(self):
        patch = QUALITY_PROFILE.to_settings_patch()
        assert patch["mask_engine"] == "RealityUX"

    def test_nvdec_on_maps_correctly(self):
        patch = FAST_PROFILE.to_settings_patch()
        assert patch["perf_nvdec"] == "on"

    def test_nvdec_cpu_maps_to_off(self):
        # Build a profile with decode_mode='cpu'
        p = QualityProfile(
            name="TEST",
            detector="scrfd_2.5g",
            detector_resolution=320,
            tracking_frequency=1,
            alignment_quality="draft",
            swap_model="hyperswap",
            swap_precision="fp16",
            enhancer="none",
            enhancer_strength=0.0,
            xseg_mode="none",
            mask_quality="draft",
            temporal_stabilization=False,
            batch_size=2,
            thread_count=4,
            decode_mode="cpu",
            encode_mode="libx264",
        )
        patch = p.to_settings_patch()
        assert patch["perf_nvdec"] == "off"
        assert patch["output_video_codec"] == "libx264"

    def test_encode_auto_not_in_patch(self):
        p = QualityProfile(
            name="TEST",
            detector="scrfd_2.5g",
            detector_resolution=320,
            tracking_frequency=1,
            alignment_quality="draft",
            swap_model="hyperswap",
            swap_precision="fp16",
            enhancer="none",
            enhancer_strength=0.0,
            xseg_mode="none",
            mask_quality="draft",
            temporal_stabilization=False,
            batch_size=2,
            thread_count=4,
            decode_mode="cpu",
            encode_mode="auto",
        )
        patch = p.to_settings_patch()
        assert "output_video_codec" not in patch

    def test_thread_count_not_in_patch(self):
        """thread_count is a scheduler hint and must NOT appear in the patch."""
        for profile in _FOUR_PROFILES:
            patch = profile.to_settings_patch()
            assert "thread_count" not in patch, \
                f"{profile.name}.to_settings_patch() should not contain thread_count"

    def test_mask_quality_draft_maps_to_35(self):
        patch = FAST_PROFILE.to_settings_patch()
        assert patch["face_mask_blend"] == 35

    def test_mask_quality_standard_maps_to_25(self):
        patch = BALANCED_PROFILE.to_settings_patch()
        assert patch["face_mask_blend"] == 25

    def test_mask_quality_precise_maps_to_12(self):
        patch = QUALITY_PROFILE.to_settings_patch()
        assert patch["face_mask_blend"] == 12

    def test_alignment_draft_no_enhancer_align(self):
        patch = FAST_PROFILE.to_settings_patch()
        assert patch.get("enhancer_align") is False

    def test_alignment_ultra_sets_3d_recon(self):
        patch = ULTRA_PROFILE.to_settings_patch()
        assert patch.get("use_3d_recon") is True


# ===========================================================================
# 6/7. get_profile — case-insensitive lookup
# ===========================================================================

class TestGetProfile:
    def test_lowercase_fast(self):
        assert get_profile("fast") is FAST_PROFILE

    def test_uppercase_fast(self):
        assert get_profile("FAST") is FAST_PROFILE

    def test_mixed_case(self):
        assert get_profile("BaLaNcEd") is BALANCED_PROFILE

    def test_quality_lookup(self):
        assert get_profile("quality") is QUALITY_PROFILE

    def test_ultra_lookup(self):
        assert get_profile("ultra") is ULTRA_PROFILE

    def test_invalid_raises_key_error(self):
        with pytest.raises(KeyError):
            get_profile("invalid")

    def test_empty_string_raises(self):
        with pytest.raises(KeyError):
            get_profile("")


# ===========================================================================
# 9. AutoProfileContext properties
# ===========================================================================

class TestAutoProfileContext:
    def test_cpu_only_when_vram_zero(self):
        ctx = AutoProfileContext(vram_mb=0.0)
        assert ctx.is_cpu_only is True

    def test_cpu_only_when_vram_negative(self):
        ctx = AutoProfileContext(vram_mb=-1.0)
        assert ctx.is_cpu_only is True

    def test_not_cpu_only_with_vram(self):
        ctx = AutoProfileContext(vram_mb=6000.0)
        assert ctx.is_cpu_only is False

    def test_laptop_tier_low_vram(self):
        ctx = AutoProfileContext(vram_mb=6000.0)
        assert ctx.is_laptop_tier is True

    def test_laptop_tier_boundary(self):
        ctx = AutoProfileContext(vram_mb=7167.9)
        assert ctx.is_laptop_tier is True

    def test_not_laptop_at_7168(self):
        ctx = AutoProfileContext(vram_mb=7168.0)
        assert ctx.is_laptop_tier is False

    def test_high_end_at_11520(self):
        ctx = AutoProfileContext(vram_mb=11520.0)
        assert ctx.is_high_end is True

    def test_not_high_end_below_threshold(self):
        ctx = AutoProfileContext(vram_mb=8000.0)
        assert ctx.is_high_end is False

    def test_4k_detection(self):
        ctx = AutoProfileContext(input_resolution=(3840, 2160))
        assert ctx.is_4k is True

    def test_not_4k_at_1080p(self):
        ctx = AutoProfileContext(input_resolution=(1920, 1080))
        assert ctx.is_4k is False

    def test_crowd_at_5_faces(self):
        ctx = AutoProfileContext(face_count=5)
        assert ctx.is_crowd is True

    def test_not_crowd_at_4(self):
        ctx = AutoProfileContext(face_count=4)
        assert ctx.is_crowd is False


# ===========================================================================
# 10–16. resolve_auto_profile scenarios
# ===========================================================================

class TestResolveAutoProfile:
    # 10. CPU-only
    def test_cpu_only_decode_cpu(self):
        ctx = AutoProfileContext(vram_mb=0.0)
        p = resolve_auto_profile(ctx)
        assert p.decode_mode == "cpu"

    def test_cpu_only_encode_is_libx264_or_libx265(self):
        ctx = AutoProfileContext(vram_mb=0.0)
        p = resolve_auto_profile(ctx)
        assert p.encode_mode in ("libx264", "libx265")

    def test_cpu_only_enhancer_is_none(self):
        ctx = AutoProfileContext(vram_mb=0.0)
        p = resolve_auto_profile(ctx)
        assert p.enhancer == "none"

    # 11. 3060 laptop — enhancer ALWAYS none
    def test_3060_enhancer_none(self):
        ctx = AutoProfileContext(vram_mb=6000.0)
        p = resolve_auto_profile(ctx)
        assert p.enhancer == "none", "3060 RSS safety: enhancer must be 'none'"

    def test_3060_high_quality_hint_still_none(self):
        """Even quality_hint=0.99 must not enable enhancer on 3060."""
        ctx = AutoProfileContext(vram_mb=6000.0, quality_hint=0.99)
        p = resolve_auto_profile(ctx)
        assert p.enhancer == "none"

    def test_3060_decode_mode_cpu(self):
        ctx = AutoProfileContext(vram_mb=6000.0)
        p = resolve_auto_profile(ctx)
        assert p.decode_mode == "cpu"

    # 12. 4K → detector_resolution=640
    def test_4k_detector_resolution(self):
        ctx = AutoProfileContext(
            vram_mb=12000.0,
            input_resolution=(3840, 2160),
        )
        p = resolve_auto_profile(ctx)
        assert p.detector_resolution == 640

    def test_4k_tracking_frequency_one(self):
        ctx = AutoProfileContext(
            vram_mb=12000.0,
            input_resolution=(3840, 2160),
        )
        p = resolve_auto_profile(ctx)
        assert p.tracking_frequency == 1

    # 13. Crowd
    def test_crowd_detector_scrfd_10g(self):
        ctx = AutoProfileContext(vram_mb=12000.0, face_count=5)
        p = resolve_auto_profile(ctx)
        assert p.detector == "scrfd_10g"

    def test_crowd_xseg_quality(self):
        ctx = AutoProfileContext(vram_mb=12000.0, face_count=5)
        p = resolve_auto_profile(ctx)
        assert p.xseg_mode == "quality"

    def test_crowd_temporal_stabilization_true(self):
        ctx = AutoProfileContext(vram_mb=12000.0, face_count=5)
        p = resolve_auto_profile(ctx)
        assert p.temporal_stabilization is True

    # 14. quality_hint=0.9 → QUALITY base (gpen_256_pro or better)
    def test_quality_hint_09_enhancer_quality_or_better(self):
        ctx = AutoProfileContext(vram_mb=12000.0, quality_hint=0.9)
        p = resolve_auto_profile(ctx)
        rank = _ENH_RANK.get(p.enhancer, 0)
        assert rank >= _ENH_RANK["gpen_256_pro"], \
            f"quality_hint=0.9 should give gpen_256_pro or better, got {p.enhancer!r}"

    # 15. quality_hint=0.97 → ULTRA base (restore_ultra)
    def test_quality_hint_097_restore_ultra(self):
        ctx = AutoProfileContext(vram_mb=12000.0, quality_hint=0.97)
        p = resolve_auto_profile(ctx)
        assert p.enhancer == "restore_ultra", \
            f"quality_hint=0.97 should give restore_ultra, got {p.enhancer!r}"

    # 16. Default (4070) → BALANCED (gpen_256)
    def test_default_4070_enhancer_gpen_256(self):
        ctx = AutoProfileContext(vram_mb=12288.0, quality_hint=0.5)
        p = resolve_auto_profile(ctx)
        assert p.enhancer == "gpen_256", \
            f"Default 4070 context should give gpen_256, got {p.enhancer!r}"

    # 20. 3060 batch_size ≤ 4
    def test_3060_batch_size_le_4(self):
        ctx = AutoProfileContext(vram_mb=6000.0)
        p = resolve_auto_profile(ctx)
        assert p.batch_size <= 4


# ===========================================================================
# 17. All 4 profiles are mutually distinct
# ===========================================================================

class TestProfileDistinctness:
    def test_all_profiles_distinct(self):
        pairs = [
            (a, b)
            for i, a in enumerate(_FOUR_PROFILES)
            for b in _FOUR_PROFILES[i + 1:]
        ]
        for a, b in pairs:
            a_d = {k: v for k, v in a.to_dict().items() if k in _REQUIRED_FIELDS}
            b_d = {k: v for k, v in b.to_dict().items() if k in _REQUIRED_FIELDS}
            assert a_d != b_d, \
                f"Profiles {a.name} and {b.name} are identical on all 15 fields"


# ===========================================================================
# 18. benchmark_basis is non-empty for all 4 profiles
# ===========================================================================

class TestBenchmarkBasis:
    @pytest.mark.parametrize("profile", _FOUR_PROFILES, ids=lambda p: p.name)
    def test_benchmark_basis_nonempty(self, profile):
        assert isinstance(profile.benchmark_basis, str)
        assert len(profile.benchmark_basis) > 0, \
            f"{profile.name}.benchmark_basis is empty"


# ===========================================================================
# 19. FAST tracking_frequency > BALANCED tracking_frequency
# ===========================================================================

class TestFastTrackingFrequency:
    def test_fast_tracking_frequency_greater_than_balanced(self):
        assert FAST_PROFILE.tracking_frequency > BALANCED_PROFILE.tracking_frequency, (
            f"FAST.tracking_frequency={FAST_PROFILE.tracking_frequency} should be "
            f"> BALANCED.tracking_frequency={BALANCED_PROFILE.tracking_frequency}"
        )

    def test_fast_tracking_frequency_is_2(self):
        assert FAST_PROFILE.tracking_frequency == 2

    def test_balanced_tracking_frequency_is_1(self):
        assert BALANCED_PROFILE.tracking_frequency == 1


# ===========================================================================
# Extra: Specific benchmark-derived values
# ===========================================================================

class TestBenchmarkValues:
    """Spot-check that the benchmark-derived numeric constants are correct."""

    def test_fast_batch_size_is_16(self):
        assert FAST_PROFILE.batch_size == 16

    def test_fast_enhancer_none(self):
        assert FAST_PROFILE.enhancer == "none"

    def test_fast_detector_scrfd_25g(self):
        assert FAST_PROFILE.detector == "scrfd_2.5g"

    def test_balanced_enhancer_strength_06(self):
        assert BALANCED_PROFILE.enhancer_strength == pytest.approx(0.6)

    def test_balanced_detector_scrfd_10g(self):
        assert BALANCED_PROFILE.detector == "scrfd_10g"

    def test_balanced_detector_resolution_512(self):
        assert BALANCED_PROFILE.detector_resolution == 512

    def test_quality_enhancer_gpen_256_pro(self):
        assert QUALITY_PROFILE.enhancer == "gpen_256_pro"

    def test_quality_swap_model_realswap(self):
        assert QUALITY_PROFILE.swap_model == "realswap"

    def test_quality_detector_retinaface(self):
        assert QUALITY_PROFILE.detector == "retinaface_r50"

    def test_ultra_detector_resolution_640(self):
        assert ULTRA_PROFILE.detector_resolution == 640

    def test_ultra_enhancer_restore_ultra(self):
        assert ULTRA_PROFILE.enhancer == "restore_ultra"

    def test_ultra_enhancer_strength_10(self):
        assert ULTRA_PROFILE.enhancer_strength == pytest.approx(1.0)
