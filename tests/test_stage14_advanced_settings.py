"""Tests for Stage 14 — Advanced Settings Redesign.

Covers:
1. All 8 setting groups (QUALITY, PERFORMANCE, GPU, VIDEO, DETECTION, TRACKING, RESTORATION, EXPERT).
2. Complete metadata presence on every catalogued setting (description, default, valid range, hardware, quality, performance impact).
3. Audit and identification of settings that:
   - do nothing
   - duplicate another setting
   - conflict with another setting
   - expose unsafe combinations
4. Sanitization and conflict resolution engine.
5. The 5 pipeline presets: FAST, BALANCED, QUALITY, ULTRA, AUTO.
6. Dynamic AUTO resolver factoring in:
   - GPU model & architecture
   - VRAM size & headroom
   - System RAM
   - Target face resolution
   - Face count (single vs crowded/interacting)
   - Enhancer model
   - Detector model
   - Video resolution (720p, 1080p, 4K)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure app path is available
ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app"
if str(APP_PATH) not in sys.path:
    sys.path.insert(0, str(APP_PATH))

from roop.advanced_settings_manager import (
    ADVANCED_SETTINGS_CATALOG,
    AdvancedSettingMetadata,
    AutoSettingsContext,
    PRESET_CONFIGS,
    PresetMode,
    SettingConflictWarning,
    SettingGroup,
    apply_preset,
    audit_advanced_settings,
    resolve_auto_settings,
    sanitize_settings,
)


class TestSettingGroupsAndMetadata(unittest.TestCase):
    """Test that all 8 required groups exist and every setting has complete metadata."""

    def test_eight_groups_exist(self):
        expected_groups = {
            "QUALITY",
            "PERFORMANCE",
            "GPU",
            "VIDEO",
            "DETECTION",
            "TRACKING",
            "RESTORATION",
            "EXPERT",
        }
        actual_groups = {g.value for g in SettingGroup}
        self.assertEqual(actual_groups, expected_groups)

    def test_catalog_covers_all_groups(self):
        groups_in_catalog = {meta.group for meta in ADVANCED_SETTINGS_CATALOG.values()}
        self.assertEqual(len(groups_in_catalog), 8)

    def test_every_setting_has_full_metadata(self):
        for key, meta in ADVANCED_SETTINGS_CATALOG.items():
            self.assertEqual(meta.key, key)
            self.assertTrue(len(meta.label) > 0, f"Empty label for {key}")
            self.assertIsInstance(meta.group, SettingGroup)
            self.assertTrue(len(meta.description) > 10, f"Short description for {key}")
            self.assertIsNotNone(meta.default, f"Missing default for {key}")
            self.assertTrue(meta.valid_range, f"Missing valid_range for {key}")
            self.assertTrue(len(meta.hardware_impact) > 5, f"Missing hardware_impact for {key}")
            self.assertTrue(len(meta.quality_impact) > 5, f"Missing quality_impact for {key}")
            self.assertTrue(len(meta.performance_impact) > 5, f"Missing performance_impact for {key}")

    def test_serialization_to_dict(self):
        meta = ADVANCED_SETTINGS_CATALOG["video_quality"]
        d = meta.to_dict()
        self.assertEqual(d["key"], "video_quality")
        self.assertEqual(d["group"], "QUALITY")
        self.assertIn("valid_range", d)


class TestAuditAndConflictResolution(unittest.TestCase):
    """Test auditing of no-op, duplicate, conflicting, and unsafe settings."""

    def test_do_nothing_setting_flagged(self):
        # cpu_ort_inter_threads on CUDA provider does nothing
        cfg = {"provider": "cuda", "cpu_ort_inter_threads": 4}
        warnings = audit_advanced_settings(cfg)
        types = [w.issue_type for w in warnings if w.key == "cpu_ort_inter_threads"]
        self.assertIn("DO_NOTHING", types)

    def test_duplicate_setting_flagged(self):
        # perf_detmask_pool duplicates perf_detector_pool with diverging sizes
        cfg = {"perf_detmask_pool": "4", "perf_detector_pool": "2"}
        warnings = audit_advanced_settings(cfg)
        types = [w.issue_type for w in warnings if w.key == "perf_detmask_pool"]
        self.assertIn("DUPLICATE", types)

    def test_conflict_force_cpu_flagged(self):
        # force_cpu=True with provider="tensorrt"
        cfg = {"provider": "tensorrt", "force_cpu": True}
        warnings = audit_advanced_settings(cfg)
        types = [w.issue_type for w in warnings if w.key == "force_cpu"]
        self.assertIn("CONFLICT", types)

    def test_conflict_batch_swap_off_with_max_flagged(self):
        # perf_batch_swap="off" with perf_batch_max=8
        cfg = {"perf_batch_swap": "off", "perf_batch_max": 8}
        warnings = audit_advanced_settings(cfg)
        types = [w.issue_type for w in warnings if w.key == "perf_batch_max"]
        self.assertIn("CONFLICT", types)

    def test_unsafe_cuda_graph_on_laptop_flagged(self):
        # trt_cuda_graph on RTX 3060 Laptop (6GB VRAM)
        cfg = {"trt_cuda_graph": True}
        warnings = audit_advanced_settings(cfg, gpu_vram_gb=6.0, is_laptop_or_sub7gb=True)
        types = [w.issue_type for w in warnings if w.key == "trt_cuda_graph"]
        self.assertIn("UNSAFE", types)

    def test_unsafe_trt_pool_on_laptop_flagged(self):
        # perf_trt_pool=2 on RTX 3060 Laptop
        cfg = {"perf_trt_pool": "2"}
        warnings = audit_advanced_settings(cfg, gpu_vram_gb=6.0, is_laptop_or_sub7gb=True)
        types = [w.issue_type for w in warnings if w.key == "perf_trt_pool"]
        self.assertIn("UNSAFE", types)

    def test_sanitize_settings_resolves_all_issues(self):
        dirty_cfg = {
            "provider": "tensorrt",
            "force_cpu": True,
            "perf_batch_swap": "off",
            "perf_batch_max": 8,
            "trt_cuda_graph": True,
            "perf_trt_pool": "2",
            "perf_detmask_pool": "4",
            "perf_detector_pool": "2",
            "vram_safety_margin_gb": 0.2,
        }
        sanitized = sanitize_settings(dirty_cfg, gpu_vram_gb=6.0, is_laptop_or_sub7gb=True)

        self.assertFalse(sanitized["force_cpu"])
        self.assertEqual(sanitized["perf_batch_max"], 1)
        self.assertFalse(sanitized["trt_cuda_graph"])
        self.assertEqual(sanitized["perf_trt_pool"], "0")
        self.assertEqual(sanitized["perf_detmask_pool"], "2")
        self.assertGreaterEqual(sanitized["vram_safety_margin_gb"], 0.5)


class TestFivePresets(unittest.TestCase):
    """Test the FAST, BALANCED, QUALITY, ULTRA, and AUTO presets."""

    def test_all_five_presets_exist(self):
        expected_modes = {"AUTO", "FAST", "BALANCED", "QUALITY", "ULTRA"}
        actual_modes = {p.value for p in PresetMode}
        self.assertEqual(actual_modes, expected_modes)

    def test_fast_preset(self):
        applied = apply_preset({}, PresetMode.FAST)
        self.assertEqual(applied["trt_precision"], "fp16")
        self.assertEqual(applied["detector_engine"], "scrfd_2.5g")
        self.assertEqual(applied["temporal_step"], 2)
        self.assertEqual(applied["enhancer_model"], "none")
        self.assertEqual(applied["video_quality"], 22)

    def test_balanced_preset(self):
        applied = apply_preset({}, PresetMode.BALANCED)
        self.assertEqual(applied["trt_precision"], "mixed")
        self.assertEqual(applied["detector_engine"], "retinaface_r50")
        self.assertEqual(applied["temporal_step"], 1)
        self.assertEqual(applied["enhancer_model"], "gpen_256")
        self.assertEqual(applied["video_quality"], 18)

    def test_quality_preset(self):
        applied = apply_preset({}, PresetMode.QUALITY)
        self.assertEqual(applied["detector_engine"], "scrfd_10g")
        self.assertEqual(applied["enhancer_model"], "gpen_512")
        self.assertEqual(applied["track_stitch"], "on")
        self.assertEqual(applied["face_demarcate"], "on")
        self.assertEqual(applied["video_quality"], 16)

    def test_ultra_preset(self):
        applied = apply_preset({}, PresetMode.ULTRA)
        self.assertEqual(applied["detector_engine"], "scrfd_10g")
        self.assertEqual(applied["enhancer_model"], "gpen_512")
        self.assertTrue(applied["light_harmonizer"])
        self.assertTrue(applied["expression_blink_sync"])
        self.assertEqual(applied["output_face_scale"], 1.25)
        self.assertEqual(applied["video_quality"], 14)


class TestDynamicAutoResolver(unittest.TestCase):
    """Test AUTO dynamic selection based on the 8 factors:
    GPU, VRAM, RAM, resolution, face count, enhancer, detector, video resolution.
    """

    def test_auto_on_rtx_4070_desktop(self):
        ctx = AutoSettingsContext(
            gpu_name="NVIDIA GeForce RTX 4070",
            vram_gb=12.0,
            ram_gb=32.0,
            target_resolution=256,
            face_count=1,
            enhancer="gpen_256",
            detector="retinaface_r50",
            video_resolution=(1920, 1080),
        )
        resolved = resolve_auto_settings(ctx)

        self.assertEqual(resolved["provider"], "tensorrt")
        self.assertEqual(resolved["trt_precision"], "mixed")
        self.assertEqual(resolved["perf_trt_pool"], "2")
        self.assertEqual(resolved["perf_batch_max"], 16)
        self.assertGreaterEqual(resolved["max_threads"], 16)
        self.assertEqual(resolved["vram_safety_margin_gb"], 2.5)
        self.assertEqual(resolved["video_quality"], 16)

    def test_auto_on_rtx_3060_laptop(self):
        ctx = AutoSettingsContext(
            gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU",
            vram_gb=6.0,
            ram_gb=16.0,
            target_resolution=256,
            face_count=1,
            enhancer="gpen_256",
            detector="retinaface_r50",
            video_resolution=(1920, 1080),
        )
        resolved = resolve_auto_settings(ctx)

        self.assertEqual(resolved["provider"], "tensorrt")
        self.assertEqual(resolved["perf_trt_pool"], "0")  # Strictly 0/0 single context
        self.assertFalse(resolved["trt_cuda_graph"])  # Disabled on laptop
        self.assertLessEqual(resolved["perf_batch_max"], 4)
        self.assertLessEqual(resolved["max_threads"], 8)
        self.assertEqual(resolved["vram_safety_margin_gb"], 1.0)

    def test_auto_on_cpu_fallback(self):
        ctx = AutoSettingsContext(
            gpu_name="CPU",
            vram_gb=0.0,
            ram_gb=16.0,
            target_resolution=256,
            face_count=1,
            enhancer="none",
            detector="retinaface_r50",
            video_resolution=(1280, 720),
        )
        resolved = resolve_auto_settings(ctx)

        self.assertEqual(resolved["provider"], "cpu")
        self.assertEqual(resolved["trt_precision"], "fp32")
        self.assertEqual(resolved["perf_batch_swap"], "off")
        self.assertEqual(resolved["perf_nvdec"], "off")
        self.assertEqual(resolved["output_video_codec"], "libx264")

    def test_auto_on_crowded_scene_multiple_faces(self):
        # Multiple faces (e.g. 4 faces in an interaction scene)
        ctx = AutoSettingsContext(
            gpu_name="NVIDIA GeForce RTX 4070",
            vram_gb=12.0,
            ram_gb=32.0,
            target_resolution=256,
            face_count=4,
            enhancer="gpen_256",
            detector="scrfd_10g",
            video_resolution=(1920, 1080),
        )
        resolved = resolve_auto_settings(ctx)

        self.assertEqual(resolved["face_demarcate"], "on")
        self.assertEqual(resolved["track_stitch"], "on")
        self.assertEqual(resolved["verify_swap"], "on")
        self.assertEqual(resolved["temporal_step"], 1)

    def test_auto_on_4k_video_with_heavy_enhancer(self):
        # 4K resolution (3840x2160) with heavy GPEN-512
        ctx = AutoSettingsContext(
            gpu_name="NVIDIA GeForce RTX 4070",
            vram_gb=12.0,
            ram_gb=32.0,
            target_resolution=512,
            face_count=2,
            enhancer="gpen_512",
            detector="scrfd_10g",
            video_resolution=(3840, 2160),
        )
        resolved = resolve_auto_settings(ctx)

        self.assertEqual(resolved["detector_engine"], "scrfd_10g")
        self.assertEqual(resolved["enhancer_model"], "gpen_512")
        self.assertLessEqual(resolved["perf_batch_max"], 8)  # Clamped to prevent 4K VRAM OOM
        self.assertEqual(resolved["perf_nvenc_preset"], "p4")  # Throughput preset for 4K
        self.assertTrue(resolved["color_match_after_enhance"])


if __name__ == "__main__":
    unittest.main()
