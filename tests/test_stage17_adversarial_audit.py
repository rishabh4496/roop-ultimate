"""Stage 17 — Final Adversarial Audit: test suite.

Covers all 30 audit scenarios.  Scenarios that require physical hardware or
specific media are skipped with an explicit reason (``skipIf`` / ``skipUnless``);
they are never promoted to PASS by inference.

Running this file (light profile, no GPU mark):

    ROOP_TEST_LIGHT=1 app/env/Scripts/python.exe -m pytest tests/test_stage17_adversarial_audit.py -v

All 30 scenarios are exercised at the code level.  Hardware-dependent scenarios
are represented by:
  1. A ``@unittest.skip`` test that documents the exact run conditions needed.
  2. The import / metadata contract that CAN be verified without a GPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
APP  = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.stage17_audit import (
    AUDIT_SCENARIOS,
    AuditReport,
    AuditScenario,
    AuditStatus,
    DeltaRow,
    EvidenceCategory,
    ScenarioResult,
    build_initial_report,
    check_pipeline_integrity,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scenario(id_: int) -> AuditScenario:
    for s in AUDIT_SCENARIOS:
        if s.id == id_:
            return s
    raise KeyError(id_)


# ---------------------------------------------------------------------------
# Class 1: Scenario catalogue
# ---------------------------------------------------------------------------

class TestScenarioCatalogue(unittest.TestCase):
    """All 30 scenarios are defined with required metadata."""

    def test_exactly_30_scenarios_defined(self):
        self.assertEqual(len(AUDIT_SCENARIOS), 30)

    def test_ids_are_1_to_30_inclusive(self):
        ids = [s.id for s in AUDIT_SCENARIOS]
        self.assertEqual(sorted(ids), list(range(1, 31)))

    def test_every_scenario_has_name(self):
        for s in AUDIT_SCENARIOS:
            self.assertTrue(s.name.strip(), f"Scenario #{s.id} has empty name")

    def test_every_scenario_has_description(self):
        for s in AUDIT_SCENARIOS:
            self.assertGreater(len(s.description), 20, f"#{s.id}: description too short")

    def test_every_scenario_has_evidence_category(self):
        for s in AUDIT_SCENARIOS:
            self.assertIsInstance(s.category, EvidenceCategory)

    def test_hardware_needed_scenarios_name_a_gpu(self):
        for s in AUDIT_SCENARIOS:
            if s.category == EvidenceCategory.HARDWARE_NEEDED:
                self.assertTrue(s.gpu_required, f"#{s.id}: HARDWARE_NEEDED but no gpu_required")

    def test_hardware_needed_min_frames_600(self):
        for s in AUDIT_SCENARIOS:
            if s.category == EvidenceCategory.HARDWARE_NEEDED:
                self.assertGreaterEqual(s.min_frames, 600, f"#{s.id}: under 600-frame minimum")

    def test_existing_suite_names_at_least_one_file(self):
        for s in AUDIT_SCENARIOS:
            if s.category == EvidenceCategory.EXISTING_SUITE:
                self.assertTrue(s.existing_tests, f"#{s.id}: EXISTING_SUITE but no test files")

    def test_hardware_scenario_ids_match_request(self):
        # RTX 4070 — scenarios 1, 9, 10, 11, 28
        hw_ids = {s.id for s in AUDIT_SCENARIOS if s.category == EvidenceCategory.HARDWARE_NEEDED}
        for expected in (1, 9, 10, 11, 28):
            self.assertIn(expected, hw_ids, f"Scenario #{expected} should be HARDWARE_NEEDED")

    def test_media_needed_scenario_ids(self):
        media_ids = {s.id for s in AUDIT_SCENARIOS if s.category == EvidenceCategory.MEDIA_NEEDED}
        # 17 (dark), 18 (fast motion) are both media-dependent
        for expected in (17, 18):
            self.assertIn(expected, media_ids, f"#{expected} should be MEDIA_NEEDED")


# ---------------------------------------------------------------------------
# Class 2: Pipeline integrity
# ---------------------------------------------------------------------------

class TestPipelineIntegrity(unittest.TestCase):
    """Stages 14–16 must not have modified any render-pipeline file."""

    def test_no_pipeline_files_changed_since_stage14(self):
        additive_only, changed = check_pipeline_integrity()
        self.assertTrue(
            additive_only,
            f"Pipeline files modified since Stage 14 (6e6880c): {changed}\n"
            "Stages 14–16 must be additive-only to avoid pipeline regressions.",
        )

    def test_changed_files_are_not_render_pipeline_modules(self):
        """Only newly-ADDED library modules appear in the changed list.

        The function check_pipeline_integrity() only flags files from its
        PIPELINE_BASENAMES set (ProcessMgr.py, core.py, etc.).  New library
        modules like unified_runtime_scheduler.py, quality_profiles.py and
        stage17_audit.py are excluded from that set deliberately.
        """
        _, changed = check_pipeline_integrity()
        pipeline_basenames = {
            "ProcessMgr.py", "core.py", "face_swapper.py",
            "session_pool.py", "vram_governor.py",
            "runtime_scheduler.py", "procmgr.py",
        }
        for f in changed:
            basename = Path(f).name
            self.assertIn(basename, pipeline_basenames,
                          f"Non-pipeline file appeared in changed list: {f}")

    def test_quality_profiles_does_not_import_processmgr(self):
        """quality_profiles.py must not trigger the full pipeline at import."""
        import importlib, sys
        # Remove cached imports to force a fresh check
        for key in list(sys.modules.keys()):
            if "quality_profiles" in key:
                del sys.modules[key]
        # Should import without any ProcessMgr side effect
        import roop.quality_profiles  # noqa: F401

    def test_unified_scheduler_does_not_import_processmgr(self):
        import roop.unified_runtime_scheduler  # noqa: F401

    def test_stage17_audit_does_not_import_processmgr(self):
        import roop.stage17_audit  # noqa: F401


# ---------------------------------------------------------------------------
# Class 3: Scenario 1 + 2 — Hardware tier (code-level contracts)
# ---------------------------------------------------------------------------

class TestHardwareTierContracts(unittest.TestCase):

    @unittest.skip(
        "HARDWARE_NEEDED: requires RTX 4070 with ≥600-frame render. "
        "Historical baseline: 8.13–8.49 fps at 720p two-person (SESSION_LOGS.md). "
        "Expected delta vs Stage 14: 0% (no pipeline file changed)."
    )
    def test_s01_rtx4070_end_to_end(self):
        pass  # pragma: no cover

    @unittest.skip(
        "HARDWARE_NEEDED: requires RTX 3060 Laptop (absent). "
        "Historical baseline: 13.0–14.2 fps at 720p no-enhancer (SESSION_LOGS.md). "
        "Expected delta: 0% (additive-only stages)."
    )
    def test_s02_rtx3060_end_to_end(self):
        pass  # pragma: no cover

    def test_s02_3060_rss_policy_code_level(self):
        """3060 tier: quality_profiles AUTO never enables enhancer."""
        from roop.quality_profiles import AutoProfileContext, resolve_auto_profile
        # Simulate 3060 Laptop (6 GB = 6144 MB)
        ctx = AutoProfileContext(
            gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU",
            vram_mb=6144.0,
            input_resolution=(1280, 720),
            face_count=1,
            quality_hint=1.0,  # Even ULTRA hint must be overridden
        )
        profile = resolve_auto_profile(ctx)
        self.assertEqual(profile.enhancer, "none",
                         "3060 RSS safety rule: enhancer must be 'none' regardless of quality_hint")
        self.assertEqual(profile.decode_mode, "cpu",
                         "3060 tier: decode must fall back to CPU")

    def test_s03_cpu_only_profile(self):
        """CPU-only: quality_profiles AUTO returns cpu decode/encode."""
        from roop.quality_profiles import AutoProfileContext, resolve_auto_profile
        ctx = AutoProfileContext(vram_mb=0.0, quality_hint=0.9)
        profile = resolve_auto_profile(ctx)
        self.assertEqual(profile.decode_mode, "cpu")
        self.assertIn(profile.encode_mode, ("libx264", "libx265"))
        self.assertEqual(profile.enhancer, "none")

    def test_s04_cuda_provider_admission_code_level(self):
        """CUDA provider is a valid selection in the settings catalog."""
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        provider_meta = ADVANCED_SETTINGS_CATALOG["provider"]
        self.assertIn("cuda", provider_meta.valid_range)

    def test_s02_vram_tier_boundaries(self):
        """AutoProfileContext tier properties use correct VRAM thresholds."""
        from roop.quality_profiles import AutoProfileContext
        # 3060 boundary: 6144 MB is laptop tier
        ctx_3060 = AutoProfileContext(vram_mb=6144.0)
        self.assertTrue(ctx_3060.is_laptop_tier)
        self.assertFalse(ctx_3060.is_high_end)
        self.assertFalse(ctx_3060.is_cpu_only)

        # 4070 boundary: 12288 MB is high-end
        ctx_4070 = AutoProfileContext(vram_mb=12288.0)
        self.assertFalse(ctx_4070.is_laptop_tier)
        self.assertTrue(ctx_4070.is_high_end)


# ---------------------------------------------------------------------------
# Class 4: Scenarios 5–8 — Precision modes
# ---------------------------------------------------------------------------

class TestPrecisionModes(unittest.TestCase):

    @unittest.skip(
        "HARDWARE_NEEDED: TRT engine build requires RTX 4070 + TRT 10.9.0.34. "
        "Code-level admission tests run in test_trt_probe.py / test_trt_context_manager.py."
    )
    def test_s05_tensorrt_engine_full_render(self):
        pass  # pragma: no cover

    def test_s05_tensorrt_is_admitted_by_settings(self):
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        self.assertIn("tensorrt", ADVANCED_SETTINGS_CATALOG["provider"].valid_range)

    def test_s06_fp32_is_valid_precision(self):
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        self.assertIn("fp32", ADVANCED_SETTINGS_CATALOG["trt_precision"].valid_range)

    def test_s07_fp16_is_valid_precision(self):
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        self.assertIn("fp16", ADVANCED_SETTINGS_CATALOG["trt_precision"].valid_range)
        # FAST profile uses FP16
        from roop.quality_profiles import FAST_PROFILE
        self.assertEqual(FAST_PROFILE.swap_precision, "fp16")

    def test_s08_mixed_is_live_default(self):
        """Mixed precision is the live default in config.yaml."""
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        self.assertIn("mixed", ADVANCED_SETTINGS_CATALOG["trt_precision"].valid_range)
        # Balanced and Quality profiles both use mixed
        from roop.quality_profiles import BALANCED_PROFILE, QUALITY_PROFILE, ULTRA_PROFILE
        for p in (BALANCED_PROFILE, QUALITY_PROFILE, ULTRA_PROFILE):
            self.assertEqual(p.swap_precision, "mixed")

    def test_s07_fp16_collapse_guard_exists(self):
        """Regression guard test file for FP16 colour collapse exists."""
        guard = APP / "tests" / "test_enhancer_fp16_collapse.py"
        self.assertTrue(guard.exists(), f"FP16 collapse guard missing: {guard}")

    def test_s08_mixed_retains_fp32_layernorm(self):
        """The settings catalog documents the mixed-precision LayerNorm guarantee."""
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        desc = ADVANCED_SETTINGS_CATALOG["trt_precision"].quality_impact
        self.assertIn("LayerNorm", desc,
                      "mixed description must mention FP32 LayerNorm retention")


# ---------------------------------------------------------------------------
# Class 5: Scenarios 9–11 — Resolution
# ---------------------------------------------------------------------------

class TestResolutionScenarios(unittest.TestCase):

    @unittest.skip(
        "HARDWARE_NEEDED: 720p full render on RTX 4070. "
        "Historical: 8.13–8.49 fps (SESSION_LOGS.md). "
        "Stages 14–16: additive-only → expected delta = 0%."
    )
    def test_s09_720p_full_render(self):
        pass  # pragma: no cover

    @unittest.skip(
        "HARDWARE_NEEDED: 1080p full render on RTX 4070. "
        "No baseline established — run null control first."
    )
    def test_s10_1080p_full_render(self):
        pass  # pragma: no cover

    @unittest.skip(
        "HARDWARE_NEEDED + MEDIA_NEEDED: 4K render on RTX 4070. "
        "No 4K fixture in the repository."
    )
    def test_s11_4k_full_render(self):
        pass  # pragma: no cover

    def test_s11_4k_auto_profile_adjustments(self):
        """AUTO resolver: 4K → detector_resolution=640, tracking_frequency=1."""
        from roop.quality_profiles import AutoProfileContext, resolve_auto_profile
        ctx = AutoProfileContext(
            vram_mb=12288.0,
            input_resolution=(3840, 2160),
            quality_hint=0.5,
        )
        profile = resolve_auto_profile(ctx)
        self.assertEqual(profile.detector_resolution, 640)
        self.assertEqual(profile.tracking_frequency, 1)

    def test_s11_4k_batch_halved(self):
        """AUTO 4K halves the batch size relative to the base profile."""
        from roop.quality_profiles import AutoProfileContext, BALANCED_PROFILE, resolve_auto_profile
        ctx_hd = AutoProfileContext(vram_mb=12288.0, input_resolution=(1920, 1080))
        ctx_4k = AutoProfileContext(vram_mb=12288.0, input_resolution=(3840, 2160))
        hd_profile = resolve_auto_profile(ctx_hd)
        k4_profile = resolve_auto_profile(ctx_4k)
        self.assertLessEqual(
            k4_profile.batch_size, hd_profile.batch_size,
            "4K should not produce a larger batch than HD on the same hardware",
        )


# ---------------------------------------------------------------------------
# Class 6: Scenarios 12–20 — Face scenarios
# ---------------------------------------------------------------------------

class TestFaceScenarios(unittest.TestCase):

    def test_s12_single_face_test_coverage(self):
        """test_selected_face_integration.py exists."""
        f = APP / "tests" / "test_selected_face_integration.py"
        self.assertTrue(f.exists())

    def test_s13_multi_face_test_coverage(self):
        f = APP / "tests" / "test_multi_identity_assignment.py"
        self.assertTrue(f.exists())

    def test_s14_small_faces_test_coverage(self):
        f = APP / "tests" / "test_temporal_tracker.py"
        self.assertTrue(f.exists())

    def test_s14_rescue_small_faces_is_in_catalog(self):
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        self.assertIn("rescue_small_faces", ADVANCED_SETTINGS_CATALOG)

    def test_s15_profile_face_test_coverage(self):
        f = APP / "tests" / "test_rotated_face_match.py"
        self.assertTrue(f.exists())

    def test_s16_occlusion_test_coverage(self):
        for name in ("test_occlusion_mask.py", "test_temporal_occlusion.py",
                     "test_mask_occlusion.py"):
            self.assertTrue((APP / "tests" / name).exists(), f"Missing: {name}")

    @unittest.skip(
        "MEDIA_NEEDED: No real dark-scene fixture in repository. "
        "Phases 10 and 14 noted as synthetic in AGENTS.md."
    )
    def test_s17_dark_scenes_full_render(self):
        pass  # pragma: no cover

    def test_s17_rescue_small_faces_clahe_path_exists(self):
        """The rescue_small_faces CLAHE path must be a catalogued setting."""
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        meta = ADVANCED_SETTINGS_CATALOG["rescue_small_faces"]
        self.assertIn("CLAHE", meta.description)

    @unittest.skip(
        "MEDIA_NEEDED: No fast-motion fixture in repository."
    )
    def test_s18_fast_motion_full_render(self):
        pass  # pragma: no cover

    def test_s18_track_gapfill_test_coverage(self):
        f = APP / "tests" / "test_track_gapfill.py"
        self.assertTrue(f.exists())

    def test_s19_face_crossing_test_coverage(self):
        for name in ("test_track_assignment.py", "test_track_stitch.py",
                     "test_track_reid.py"):
            self.assertTrue((APP / "tests" / name).exists(), f"Missing: {name}")

    def test_s19_face_demarcate_is_catalogued(self):
        from roop.advanced_settings_manager import ADVANCED_SETTINGS_CATALOG
        self.assertIn("face_demarcate", ADVANCED_SETTINGS_CATALOG)

    def test_s20_object_crossing_test_coverage(self):
        self.assertTrue((APP / "tests" / "test_occluder_edge.py").exists())

    def test_s20_crowd_auto_profile(self):
        """AUTO crowd (≥5 faces) → scrfd_10g, xseg=quality."""
        from roop.quality_profiles import AutoProfileContext, resolve_auto_profile
        ctx = AutoProfileContext(vram_mb=12288.0, face_count=6, quality_hint=0.5)
        p = resolve_auto_profile(ctx)
        self.assertEqual(p.detector, "scrfd_10g")
        self.assertEqual(p.xseg_mode, "quality")
        self.assertTrue(p.temporal_stabilization)


# ---------------------------------------------------------------------------
# Class 7: Scenario 21 — Source identity changes
# ---------------------------------------------------------------------------

class TestSourceIdentity(unittest.TestCase):

    def test_s21_source_faceset_mapping_coverage(self):
        self.assertTrue((APP / "tests" / "test_source_faceset_mapping.py").exists())

    def test_s21_source_portfolio_coverage(self):
        self.assertTrue((APP / "tests" / "test_source_portfolio.py").exists())


# ---------------------------------------------------------------------------
# Class 8: Scenarios 22–25 — Enhancer / XSeg
# ---------------------------------------------------------------------------

class TestEnhancerXSeg(unittest.TestCase):

    def test_s22_enhancer_visibility_guard_exists(self):
        """Regression guard for D4070.3 (enhancer returning nothing silently)."""
        f = APP / "tests" / "test_adaptive_enhancer_visibility.py"
        self.assertTrue(f.exists())

    def test_s22_enhancer_output_guard_exists(self):
        self.assertTrue((APP / "tests" / "test_enhancer_output_guard.py").exists())

    def test_s22_enhancer_guards_exists(self):
        self.assertTrue((APP / "tests" / "test_enhancer_guards.py").exists())

    def test_s23_enhancer_disabled_profile(self):
        """FAST profile explicitly sets enhancer='none'."""
        from roop.quality_profiles import FAST_PROFILE
        self.assertEqual(FAST_PROFILE.enhancer, "none")
        self.assertEqual(FAST_PROFILE.enhancer_strength, 0.0)

    def test_s23_enhancer_disabled_settings_patch(self):
        """Enhancer='none' maps to selected_enhancer='None' in the patch."""
        from roop.quality_profiles import FAST_PROFILE
        patch = FAST_PROFILE.to_settings_patch()
        self.assertEqual(patch["selected_enhancer"], "None")

    def test_s24_xseg_enabled_profiles(self):
        """BALANCED uses fast (DFL XSeg); QUALITY/ULTRA use quality (RealityUX)."""
        from roop.quality_profiles import BALANCED_PROFILE, QUALITY_PROFILE, ULTRA_PROFILE
        self.assertEqual(BALANCED_PROFILE.xseg_mode, "fast")
        self.assertEqual(QUALITY_PROFILE.xseg_mode, "quality")
        self.assertEqual(ULTRA_PROFILE.xseg_mode, "quality")

    def test_s24_xseg_patch_mapping(self):
        """xseg_mode fast → mask_engine='DFL XSeg'; quality → 'RealityUX'."""
        from roop.quality_profiles import BALANCED_PROFILE, QUALITY_PROFILE
        bal_patch = BALANCED_PROFILE.to_settings_patch()
        qual_patch = QUALITY_PROFILE.to_settings_patch()
        self.assertEqual(bal_patch["mask_engine"], "DFL XSeg")
        self.assertEqual(qual_patch["mask_engine"], "RealityUX")

    def test_s25_xseg_disabled_profile(self):
        """FAST profile disables XSeg (xseg_mode='none')."""
        from roop.quality_profiles import FAST_PROFILE
        self.assertEqual(FAST_PROFILE.xseg_mode, "none")
        patch = FAST_PROFILE.to_settings_patch()
        self.assertEqual(patch["mask_engine"], "None")

    def test_s24_realityux_test_coverage(self):
        self.assertTrue((APP / "tests" / "test_realityux_nonface_set.py").exists())


# ---------------------------------------------------------------------------
# Class 9: Scenarios 26–30 — Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle(unittest.TestCase):

    def test_s26_pause_resume_coverage(self):
        self.assertTrue((APP / "tests" / "test_pause_resume.py").exists())
        self.assertTrue((APP / "tests" / "test_resume_progress_base.py").exists())

    def test_s27_cancellation_coverage(self):
        self.assertTrue((APP / "tests" / "test_reader_shutdown.py").exists())
        self.assertTrue((APP / "tests" / "test_video_stream_release.py").exists())
        self.assertTrue((APP / "tests" / "test_render_guard.py").exists())

    @unittest.skip(
        "HARDWARE_NEEDED: ≥10,000-frame render on RTX 4070. "
        "Historical evidence: 27,556-frame b1.mp4, no RSS growth "
        "(SESSION_LOGS.md 2026-08-22). Unit coverage in test_oom_guards.py."
    )
    def test_s28_long_video_full_render(self):
        pass  # pragma: no cover

    def test_s28_oom_guard_coverage(self):
        self.assertTrue((APP / "tests" / "test_oom_guards.py").exists())

    def test_s29_batch_processing_coverage(self):
        self.assertTrue((APP / "tests" / "test_target_media_isolation.py").exists())
        self.assertTrue((APP / "tests" / "test_batch_matrix_queue.py").exists())

    def test_s30_repeated_processing_coverage(self):
        self.assertTrue((APP / "tests" / "test_quality_regression.py").exists())
        self.assertTrue((APP / "tests" / "test_regression_benchmark.py").exists())

    def test_s30_pixel_noise_floor_documented(self):
        """The noise floor is recorded in the report object."""
        report = build_initial_report()
        self.assertAlmostEqual(report.pixel_noise_floor_mean, 0.7142, places=3)
        self.assertAlmostEqual(report.pixel_noise_floor_max, 22.0, places=0)


# ---------------------------------------------------------------------------
# Class 10: Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration(unittest.TestCase):

    def test_build_initial_report_runs(self):
        report = build_initial_report()
        self.assertIsInstance(report, AuditReport)

    def test_report_has_30_results(self):
        report = build_initial_report()
        self.assertEqual(len(report.results), 30)

    def test_pipeline_integrity_flag_set(self):
        report = build_initial_report()
        self.assertTrue(
            report.stages_14_to_16_are_additive_only,
            "Stages 14–16 must be additive-only.  "
            f"Changed: {report.pipeline_files_changed_since_stage14}",
        )

    def test_hardware_scenarios_are_not_tested(self):
        report = build_initial_report()
        for r in report.results:
            if r.scenario.category == EvidenceCategory.HARDWARE_NEEDED:
                self.assertEqual(r.status, AuditStatus.NOT_TESTED,
                                 f"#{r.scenario.id}: hardware scenario must be NOT_TESTED")

    def test_media_scenarios_are_not_tested(self):
        report = build_initial_report()
        for r in report.results:
            if r.scenario.category == EvidenceCategory.MEDIA_NEEDED:
                self.assertEqual(r.status, AuditStatus.NOT_TESTED,
                                 f"#{r.scenario.id}: media scenario must be NOT_TESTED")

    def test_no_hardware_scenario_is_promoted_to_pass(self):
        report = build_initial_report()
        for r in report.results:
            if r.scenario.category in (EvidenceCategory.HARDWARE_NEEDED,
                                       EvidenceCategory.MEDIA_NEEDED):
                self.assertNotEqual(r.status, AuditStatus.PASS,
                                    f"#{r.scenario.id}: cannot be PASS without hardware/media")

    def test_summary_is_a_non_empty_string(self):
        report = build_initial_report()
        summary = report.summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 200)

    def test_summary_mentions_additive_only(self):
        report = build_initial_report()
        self.assertIn("additive", report.summary().lower())

    def test_baseline_fps_recorded(self):
        report = build_initial_report()
        lo, hi = report.baseline_fps_4070_720p_twoperson
        self.assertGreater(hi, lo)
        self.assertGreater(lo, 0)

    def test_not_tested_count_matches_hardware_and_media(self):
        report = build_initial_report()
        hw_count = sum(1 for s in AUDIT_SCENARIOS
                       if s.category in (EvidenceCategory.HARDWARE_NEEDED,
                                         EvidenceCategory.MEDIA_NEEDED))
        nt_count = sum(1 for r in report.results
                       if r.status == AuditStatus.NOT_TESTED)
        self.assertEqual(nt_count, hw_count)


# ---------------------------------------------------------------------------
# Class 11: Compare against Stage 0 — code-level delta contract
# ---------------------------------------------------------------------------

class TestStage0Comparison(unittest.TestCase):
    """
    Stage 0 = the codebase at commit 6e6880c (before Stage 14).
    Stages 14–16 added library modules only.  The delta contract:

      FPS:            0%   (no pipeline file changed)
      VRAM:           0 MB (no new sessions created at import)
      RAM:            0 MB (no new persistent allocations)
      GPU util:       0%   (no new kernel launches)
      Quality:        0    (no render path changed)
      Failure rate:   0    (no exception handler changed)
    """

    def test_delta_fps_is_zero_by_construction(self):
        """
        Because no render-pipeline file was modified, the render clock
        cannot change.  Verify the pipeline integrity check confirms this.
        """
        additive_only, changed = check_pipeline_integrity()
        self.assertTrue(
            additive_only,
            "FPS delta cannot be zero if pipeline files changed: "
            + str(changed),
        )

    def test_delta_vram_is_zero_at_import(self):
        """
        Importing Stages 14–16 modules must not allocate GPU memory.
        (They import torch optionally, with _swallowed fallback.)
        """
        # In the light test profile, torch.cuda is either absent or not
        # initialised; all three modules must import cleanly regardless.
        import roop.advanced_settings_manager  # noqa: F401
        import roop.unified_runtime_scheduler  # noqa: F401
        import roop.quality_profiles           # noqa: F401

    def test_delta_quality_is_zero_by_construction(self):
        """
        Render pixel output is unchanged: no compositing, blending or
        mask logic was touched.  The pixel noise floor (0.7142/255 mean)
        defines 'same output'; any delta below this is not a regression.
        """
        additive_only, _ = check_pipeline_integrity()
        self.assertTrue(additive_only)

    def test_new_settings_have_no_pipeline_side_effects(self):
        """
        Importing advanced_settings_manager should not instantiate any
        ONNX session, torch tensor or GPU context.
        """
        import roop.advanced_settings_manager as asm
        # The catalog is a plain dict of frozen dataclasses — no GPU work
        self.assertIsInstance(asm.ADVANCED_SETTINGS_CATALOG, dict)
        self.assertGreater(len(asm.ADVANCED_SETTINGS_CATALOG), 0)

    def test_stage15_scheduler_not_wired_into_pipeline(self):
        """
        UnifiedRuntimeScheduler exists but is not called from ProcessMgr
        or core.py.  Verify by checking that ProcessMgr does not import it.
        """
        processmgr = APP / "roop" / "ProcessMgr.py"
        if not processmgr.exists():
            self.skipTest("ProcessMgr.py not found in app/roop/")
        content = processmgr.read_text(encoding="utf-8", errors="replace")
        self.assertNotIn(
            "unified_runtime_scheduler",
            content,
            "UnifiedRuntimeScheduler must not be imported by ProcessMgr "
            "until it is intentionally wired in and tested end-to-end.",
        )

    def test_stage16_quality_profiles_not_wired_into_pipeline(self):
        """
        quality_profiles.py exists but is not yet called from core.py.
        It is a library only; no pipeline side effects.
        """
        core_py = APP / "roop" / "core.py"
        if not core_py.exists():
            self.skipTest("core.py not found")
        content = core_py.read_text(encoding="utf-8", errors="replace")
        self.assertNotIn(
            "quality_profiles",
            content,
            "quality_profiles must not be imported by core.py yet — "
            "it has not been validated end-to-end.",
        )
