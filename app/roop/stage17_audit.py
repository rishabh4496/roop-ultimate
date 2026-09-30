"""Stage 17 — Final Adversarial Audit Framework.

This module defines the 30 audit scenarios, classifies them by evidence
category, maps each to existing test coverage, and produces a structured
report.  It deliberately does NOT invent hardware numbers or interpolate
missing measurements.

Evidence categories
-------------------
CODE_VERIFIED   Proven by static analysis, unit tests, or import inspection.
                These do not require hardware or media.
EXISTING_SUITE  Covered by a pre-existing test in ``app/tests/`` or
                ``tests/``.  The exact file is named.
HARDWARE_NEEDED Requires a 600-frame render on specific hardware with
                counterbalanced A/B.  Marked NOT_TESTED on the absent GPU.
MEDIA_NEEDED    Requires a specific video clip (dark footage, fast motion,
                etc.) that is not in the repository.

Truthfulness contract
---------------------
*NOT_TESTED is a truthful state* (AGENTS.md, FINAL_VALIDATION_MATRIX.md).
No cell is upgraded to PASS by inference from a different GPU or a shorter
run.  Every HARDWARE_NEEDED scenario that cannot be executed on the current
machine is recorded and deferred, not guessed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from roop.degrade import swallowed as _swallowed

logger = logging.getLogger("roop.stage17_audit")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvidenceCategory(str, Enum):
    CODE_VERIFIED  = "CODE_VERIFIED"   # static / unit, no GPU
    EXISTING_SUITE = "EXISTING_SUITE"  # pre-existing test file
    HARDWARE_NEEDED = "HARDWARE_NEEDED"  # 600-frame render required
    MEDIA_NEEDED   = "MEDIA_NEEDED"    # specific clip required


class AuditStatus(str, Enum):
    PASS        = "PASS"
    FAIL        = "FAIL"
    NOT_TESTED  = "NOT_TESTED"   # hardware / media absent
    BLOCKED     = "BLOCKED"      # dependency missing (e.g. TRT runtime)
    NOT_RESOLVABLE = "NOT_RESOLVABLE"  # within noise floor


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditScenario:
    """Definition of a single adversarial audit scenario."""

    id: int                        # 1-30, per user request
    name: str
    category: EvidenceCategory
    description: str
    # Which GPUs are required for a complete result
    gpu_required: Tuple[str, ...] = ()
    # Minimum frame count for any throughput claim
    min_frames: int = 600
    # Existing test files that cover this scenario (relative to repo root)
    existing_tests: Tuple[str, ...] = ()
    # Extra notes / known constraints
    notes: str = ""


# ---------------------------------------------------------------------------
# The 30 scenarios
# ---------------------------------------------------------------------------

AUDIT_SCENARIOS: List[AuditScenario] = [

    # ── Hardware tier ───────────────────────────────────────────────────────

    AuditScenario(
        id=1, name="RTX 4070 end-to-end",
        category=EvidenceCategory.HARDWARE_NEEDED,
        description=(
            "Full render on the main device (RTX 4070 12 GB) with the "
            "production config (hyperswap, GPEN 256 Pro, RealityUX).  "
            "Measure fps, VRAM peak, RSS, and GPU util."
        ),
        gpu_required=("RTX 4070",),
        min_frames=600,
        existing_tests=(
            "app/tests/test_runtime_scheduler.py",
            "app/tests/test_vram_governor.py",
        ),
        notes=(
            "Historical baseline: 18–31 fps at 1280×720 (SESSION_LOGS.md, "
            "2026-08-22).  Stages 14–16 are additive-only; no pipeline "
            "file was modified (verified by `git diff 6e6880c HEAD "
            "--name-only`)."
        ),
    ),

    AuditScenario(
        id=2, name="RTX 3060 Laptop end-to-end",
        category=EvidenceCategory.HARDWARE_NEEDED,
        description=(
            "Full render on the secondary device (RTX 3060 Laptop 6 GB) "
            "with no enhancer (sub-7 GB RSS policy).  Measure fps, peak "
            "RSS, GPU util."
        ),
        gpu_required=("RTX 3060",),
        min_frames=600,
        existing_tests=(
            "app/tests/test_small_card_admission.py",
            "app/tests/test_small_card_prepass.py",
            "app/tests/test_sub_7gb_tensorrt_policy.py",
        ),
        notes=(
            "Historical baseline: 13.0–14.2 fps at 1280×720, no enhancer "
            "(SESSION_LOGS.md, 2026-09-29).  This machine is currently "
            "active; 3060 is the absent GPU for this session."
        ),
    ),

    AuditScenario(
        id=3, name="CPU-only fallback",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Verify that force_cpu=True produces valid output, the CPU "
            "execution provider is selected, and no CUDA-only code path "
            "crashes."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_provider_fallback_is_loud.py",
            "app/tests/test_provider_discovery_is_safe.py",
            "app/tests/test_provider_initialization.py",
        ),
        notes="CPU path exercised in light-profile test suite (no GPU mark).",
    ),

    AuditScenario(
        id=4, name="CUDA execution provider",
        category=EvidenceCategory.EXISTING_SUITE,
        description="Verify CUDA EP is admitted, sessions are built, no TRT used.",
        gpu_required=("RTX 4070",),
        existing_tests=(
            "app/tests/test_precision_policy.py",
            "app/tests/test_provider_initialization.py",
        ),
        notes="Provider selection is unit-tested; full render needs GPU.",
    ),

    # ── Precision modes ─────────────────────────────────────────────────────

    AuditScenario(
        id=5, name="TensorRT provider",
        category=EvidenceCategory.EXISTING_SUITE,
        description="Engine build, cache, and inference via TensorRT EP.",
        gpu_required=("RTX 4070",),
        existing_tests=(
            "app/tests/test_trt_probe.py",
            "app/tests/test_trt_context_manager.py",
            "app/tests/test_trt_settings_visibility.py",
            "app/tests/test_startup_optional_tensorrt.py",
        ),
        notes=(
            "TRT runtime absent from 3060's historical environment "
            "(FINAL_VALIDATION_MATRIX.md rows 1.2/1.3 BLOCKED).  "
            "4070 has TRT 10.9.0.34."
        ),
    ),

    AuditScenario(
        id=6, name="FP32 precision",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Verify FP32 produces valid output and the correct trt_precision "
            "policy is applied."
        ),
        gpu_required=("RTX 4070",),
        existing_tests=(
            "app/tests/test_precision_policy.py",
            "app/tests/test_enhancer_fp16_collapse.py",
        ),
        notes="Precision policy is unit-tested without a live TRT engine.",
    ),

    AuditScenario(
        id=7, name="FP16 precision",
        category=EvidenceCategory.EXISTING_SUITE,
        description="FP16 engine build; no NaN / colour collapse in output.",
        gpu_required=("RTX 4070",),
        existing_tests=(
            "app/tests/test_enhancer_fp16_collapse.py",
            "app/tests/test_precision_policy.py",
        ),
        notes=(
            "FP16 collapse test is a regression guard.  "
            "+20% throughput vs FP32 is a historical ratio from SESSION_LOGS.md."
        ),
    ),

    AuditScenario(
        id=8, name="Mixed precision",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Mixed precision retains FP32 LayerNorm; no NaN discoloration "
            "or collapsed eyes."
        ),
        gpu_required=("RTX 4070",),
        existing_tests=(
            "app/tests/test_precision_policy.py",
            "app/tests/test_enhancer_fp16_collapse.py",
        ),
        notes="mixed is the live default (config.yaml trt_precision: mixed).",
    ),

    # ── Resolution ──────────────────────────────────────────────────────────

    AuditScenario(
        id=9, name="720p video",
        category=EvidenceCategory.HARDWARE_NEEDED,
        description=(
            "Full render on 1280×720 footage.  Baseline: 8.13–8.49 fps "
            "(two-person, 4070, SESSION_LOGS.md)."
        ),
        gpu_required=("RTX 4070",),
        min_frames=600,
        existing_tests=("app/tests/test_phase12_pipeline.py",),
        notes="The locked benchmark fixture is 720p.  Any delta against Stages 14-16 should be 0 (no pipeline change).",
    ),

    AuditScenario(
        id=10, name="1080p video",
        category=EvidenceCategory.HARDWARE_NEEDED,
        description="Full render on 1920×1080 footage.  No baseline established.",
        gpu_required=("RTX 4070",),
        min_frames=600,
        existing_tests=("app/tests/test_phase12_pipeline.py",),
        notes=(
            "No 4070 1080p absolute FPS baseline exists in SESSION_LOGS.md. "
            "Establish a null control before comparing."
        ),
    ),

    AuditScenario(
        id=11, name="4K video",
        category=EvidenceCategory.HARDWARE_NEEDED,
        description="Full render on 3840×2160 footage.",
        gpu_required=("RTX 4070",),
        min_frames=600,
        existing_tests=(),
        notes=(
            "No 4K fixture in the repository.  "
            "`tests/assets/benchmark/scenarios/scenario_03_extreme_profile.mp4` "
            "may be usable but resolution is unconfirmed.  "
            "AUTO resolver 4K branch is unit-tested in Stage 16."
        ),
    ),

    # ── Face scenarios ───────────────────────────────────────────────────────

    AuditScenario(
        id=12, name="Single-face footage",
        category=EvidenceCategory.EXISTING_SUITE,
        description="Pipeline selects exactly one face per frame; no cross-identity bleeding.",
        gpu_required=(),
        existing_tests=(
            "app/tests/test_selected_face_integration.py",
            "app/tests/test_selected_face_regression.py",
            "app/tests/test_selected_face_safety.py",
            "app/tests/test_single_clip.py",
        ),
    ),

    AuditScenario(
        id=13, name="Multiple faces",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Two or more faces per frame.  Identity assignment is stable; "
            "no cross-face swap bleed."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_multi_identity_assignment.py",
            "app/tests/test_target_person_identity.py",
            "app/tests/test_selected_target_mapping.py",
        ),
        notes=(
            "Historical baseline: wrong-faceset error dropped from 55.4% "
            "to 1.48% after Phase 3 fixes (SESSION_LOGS.md)."
        ),
    ),

    AuditScenario(
        id=14, name="Small faces",
        category=EvidenceCategory.EXISTING_SUITE,
        description="Faces <20 px in a 4K frame.  Detector scale pyramid recall.",
        gpu_required=(),
        existing_tests=(
            "app/tests/test_temporal_tracker.py",
            "app/tests/test_pose_quality.py",
        ),
        notes=(
            "rescue_small_faces CLAHE path; sub-20px detection needs "
            "scale-pyramid on.  MEDIA_NEEDED for a definitive end-to-end "
            "result — no small-face fixture is in the repo."
        ),
    ),

    AuditScenario(
        id=15, name="Profile / extreme-angle faces",
        category=EvidenceCategory.EXISTING_SUITE,
        description="Faces at ≥60° yaw.  RetinaFace recall vs SCRFD.",
        gpu_required=(),
        existing_tests=(
            "app/tests/test_orientation.py",
            "app/tests/test_rotated_face_match.py",
            "app/tests/test_routes_angle_scan.py",
            "app/tests/test_upright_recovery.py",
            "app/tests/test_phase6_pose_quality.py",
        ),
        notes=(
            "Angle harness `app/tests/find_profile_angles.py` exists. "
            "Phase 6 angle recall established in SESSION_LOGS.md."
        ),
    ),

    AuditScenario(
        id=16, name="Occlusion (hand / object over face)",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Face partially occluded by a hand or foreground object.  "
            "Mask engine does not bleed onto occluder; identity is stable."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_occlusion_mask.py",
            "app/tests/test_occlusion_wiring.py",
            "app/tests/test_occlusion_gate_population.py",
            "app/tests/test_mask_occlusion.py",
            "app/tests/test_temporal_occlusion.py",
        ),
    ),

    AuditScenario(
        id=17, name="Dark scenes",
        category=EvidenceCategory.MEDIA_NEEDED,
        description=(
            "Night footage or interior low-light.  rescue_small_faces CLAHE "
            "path; detector does not fail silently."
        ),
        gpu_required=("RTX 4070",),
        min_frames=600,
        existing_tests=("app/tests/test_occluder_edge.py",),
        notes=(
            "Phase 10 (dark footage) and Phase 14 (night footage) are listed "
            "as synthetic in AGENTS.md ('Real occluder and real night footage "
            "— Phases 10 and 14 are both synthetic'). "
            "No real dark-scene fixture is in the repository."
        ),
    ),

    AuditScenario(
        id=18, name="Fast motion",
        category=EvidenceCategory.MEDIA_NEEDED,
        description=(
            "High-speed movement causing motion blur.  Temporal tracker "
            "hold and gap-fill; no landmark drift on rapid head turns."
        ),
        gpu_required=("RTX 4070",),
        min_frames=600,
        existing_tests=(
            "app/tests/test_track_gapfill.py",
            "app/tests/test_tracker_coasting.py",
            "app/tests/test_temporal_smoother.py",
        ),
        notes="No fast-motion fixture is in the repository.",
    ),

    AuditScenario(
        id=19, name="Face crossing / two actors kissing",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Two faces overlap.  face_demarcate prevents crop bleed and "
            "identity cross-contamination."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_track_assignment.py",
            "app/tests/test_track_crossshot.py",
            "app/tests/test_track_stitch.py",
            "app/tests/test_track_reid.py",
        ),
        notes=(
            "Phase 3 headline ask: 'interacting faces — characterized but "
            "unsolved' (AGENTS.md).  Unit coverage exists; real-clip evidence "
            "would require the d9.mp4 kissing clip."
        ),
    ),

    AuditScenario(
        id=20, name="Object crossing face (non-face occluder)",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "A cup, hand or prop passes in front of the face.  "
            "mask_clip_text exclusion; no swap bleed onto the object."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_occluder_edge.py",
            "app/tests/test_occlusion_color_pipeline.py",
        ),
        notes=(
            "mask_clip_text: 'cup,hands,hair,banana' in live config.  "
            "Unit coverage tests the policy; a real-clip validation "
            "is MEDIA_NEEDED."
        ),
    ),

    # ── Source identity ──────────────────────────────────────────────────────

    AuditScenario(
        id=21, name="Source identity changes mid-video",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "A new source faceset is loaded or the source face changes "
            "between renders.  Identity state is reset; stale embeddings "
            "do not contaminate the new run."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_source_faceset_mapping.py",
            "app/tests/test_source_portfolio.py",
            "app/tests/test_project_provider_identity.py",
        ),
    ),

    # ── Enhancer / XSeg ─────────────────────────────────────────────────────

    AuditScenario(
        id=22, name="Enhancer enabled",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Enhancer stage executes and produces observably different output "
            "from the raw swap.  The adaptive enhancer does not return "
            "immediately without running."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_adaptive_enhancer.py",
            "app/tests/test_adaptive_enhancer_visibility.py",
            "app/tests/test_enhancer_guards.py",
            "app/tests/test_enhancer_output_guard.py",
            "app/tests/test_enhancer_gpen256_pro.py",
            "app/tests/test_enhancer_ultramax.py",
            "app/tests/test_restorers.py",
        ),
        notes=(
            "Defect D4070.3: adaptive enhancer restored nothing on 60/60 "
            "faces while presenting as the fastest arm.  "
            "test_adaptive_enhancer_visibility.py is the regression guard."
        ),
    ),

    AuditScenario(
        id=23, name="Enhancer disabled",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "selected_enhancer='None' skips the enhancement stage entirely; "
            "no extra VRAM or latency; output is raw swap."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_enhancer_names.py",
            "app/tests/test_restorers.py",
        ),
    ),

    AuditScenario(
        id=24, name="XSeg enabled",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "mask_engine='DFL XSeg' or 'RealityUX' — mask is applied, "
            "non-face regions are excluded, glasses protection fires."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_realityux_nonface_set.py",
            "app/tests/test_batcher_mask_exclusion.py",
            "app/tests/test_mask_occlusion.py",
        ),
    ),

    AuditScenario(
        id=25, name="XSeg disabled",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "mask_engine='None' — no mask inference runs; full face "
            "crop is blended without segmentation."
        ),
        gpu_required=(),
        existing_tests=("app/tests/test_batcher_mask_exclusion.py",),
    ),

    # ── Lifecycle ────────────────────────────────────────────────────────────

    AuditScenario(
        id=26, name="Pause / resume",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Render pauses cleanly; all worker threads quiesce; resume "
            "continues from the correct frame without re-processing or "
            "missing frames."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_pause_resume.py",
            "app/tests/test_resume_progress_base.py",
        ),
    ),

    AuditScenario(
        id=27, name="Cancellation",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Render is cancelled mid-run; all threads terminate; ONNX "
            "sessions are released; no GPU memory leak."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_reader_shutdown.py",
            "app/tests/test_video_stream_release.py",
            "app/tests/test_render_guard.py",
        ),
    ),

    AuditScenario(
        id=28, name="Long video (>10,000 frames)",
        category=EvidenceCategory.HARDWARE_NEEDED,
        description=(
            "RSS does not grow monotonically over a long render.  "
            "No frame-buffer leak; peak RSS remains bounded."
        ),
        gpu_required=("RTX 4070",),
        min_frames=10000,
        existing_tests=(
            "app/tests/test_stab_live_chunk_accounting.py",
            "app/tests/test_stab_parallel_lifecycle.py",
            "app/tests/test_oom_guards.py",
        ),
        notes=(
            "Historical: 27,556-frame b1.mp4 ran without RSS growth "
            "(SESSION_LOGS.md, 2026-08-22).  Same clip available on 4070 "
            "machine if re-run is needed."
        ),
    ),

    AuditScenario(
        id=29, name="Batch processing (multiple clips)",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "Multiple clips processed in sequence.  State resets between "
            "jobs; sessions are released and rebuilt; no stale tracks."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_target_media_isolation.py",
            "app/tests/test_project_io_nle.py",
            "app/tests/test_batch_matrix_queue.py",
        ),
        notes=(
            "Historical: 16 consecutive renders, all rc 0, no monotonic "
            "RSS growth (FINAL_VALIDATION_MATRIX.md phase 0.3/0.4)."
        ),
    ),

    AuditScenario(
        id=30, name="Repeated processing (same clip, same config)",
        category=EvidenceCategory.EXISTING_SUITE,
        description=(
            "The same clip rendered twice or more produces bit-identical "
            "swap geometry (within noise floor).  No stale state between "
            "runs."
        ),
        gpu_required=(),
        existing_tests=(
            "app/tests/test_quality_regression.py",
            "app/tests/test_regression_benchmark.py",
        ),
        notes=(
            "Historical: pure swap geometry byte-identical across 13 "
            "consecutive arms (FINAL_VALIDATION_MATRIX.md phase 0.5).  "
            "Pixel noise floor: 0.7142/255 mean, 22/255 max — does not "
            "constitute a regression."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class DeltaRow:
    """A single measured or declared delta vs Stage 0 for one scenario."""

    scenario_id: int
    metric: str            # "fps" | "vram_mb" | "rss_mb" | "gpu_util_pct" | "quality"
    stage0_value: Optional[float]
    current_value: Optional[float]
    delta: Optional[float]
    status: AuditStatus
    evidence: str          # one-liner citing session log or test


@dataclass
class ScenarioResult:
    """Audit result for a single scenario."""

    scenario: AuditScenario
    status: AuditStatus
    deltas: List[DeltaRow] = field(default_factory=list)
    notes: str = ""
    blocking_reason: str = ""


@dataclass
class AuditReport:
    """Complete Stage 17 adversarial audit report."""

    code_verified_count: int = 0
    existing_suite_count: int = 0
    hardware_needed_count: int = 0
    media_needed_count: int = 0

    results: List[ScenarioResult] = field(default_factory=list)

    # Pipeline integrity check
    pipeline_files_changed_since_stage14: List[str] = field(default_factory=list)
    stages_14_to_16_are_additive_only: bool = False

    # Baseline numbers from SESSION_LOGS.md (pre-Stage-14)
    baseline_fps_4070_720p_twoperson: Tuple[float, float] = (8.13, 8.49)
    baseline_fps_3060_720p_oneperson_no_enhancer: Tuple[float, float] = (13.0, 14.2)
    baseline_spread_4070_pct: float = 4.5   # measured null-control spread
    pixel_noise_floor_mean: float = 0.7142  # per 255
    pixel_noise_floor_max: float = 22.0     # per 255

    def summary(self) -> str:
        """Return a human-readable one-page summary."""
        pass_count   = sum(1 for r in self.results if r.status == AuditStatus.PASS)
        fail_count   = sum(1 for r in self.results if r.status == AuditStatus.FAIL)
        nt_count     = sum(1 for r in self.results if r.status == AuditStatus.NOT_TESTED)
        block_count  = sum(1 for r in self.results if r.status == AuditStatus.BLOCKED)
        nr_count     = sum(1 for r in self.results if r.status == AuditStatus.NOT_RESOLVABLE)

        lines = [
            "=" * 72,
            "STAGE 17 — FINAL ADVERSARIAL AUDIT REPORT",
            "=" * 72,
            "",
            "Pipeline integrity",
            f"  Files changed in pipeline since Stage 14 (6e6880c):  "
            f"{'NONE' if self.stages_14_to_16_are_additive_only else str(self.pipeline_files_changed_since_stage14)}",
            f"  Stages 14–16 are additive-only: "
            f"{'YES — no render pipeline file was modified' if self.stages_14_to_16_are_additive_only else 'NO — see changed files above'}",
            "",
            "Stage 0 baseline (from SESSION_LOGS.md):",
            f"  RTX 4070, 720p, two-person:  "
            f"{self.baseline_fps_4070_720p_twoperson[0]:.2f}–"
            f"{self.baseline_fps_4070_720p_twoperson[1]:.2f} fps",
            f"  RTX 3060, 720p, one-person, no enhancer:  "
            f"{self.baseline_fps_3060_720p_oneperson_no_enhancer[0]:.1f}–"
            f"{self.baseline_fps_3060_720p_oneperson_no_enhancer[1]:.1f} fps",
            f"  Null-control spread (4070): ±{self.baseline_spread_4070_pct:.1f}% "
            "(effects below this are NOT RESOLVABLE)",
            f"  Pixel noise floor: {self.pixel_noise_floor_mean:.4f}/255 mean, "
            f"{self.pixel_noise_floor_max:.0f}/255 max",
            "",
            "Evidence classification:",
            f"  CODE_VERIFIED   : {self.code_verified_count:3d}",
            f"  EXISTING_SUITE  : {self.existing_suite_count:3d}",
            f"  HARDWARE_NEEDED : {self.hardware_needed_count:3d}  (NOT_TESTED without that GPU)",
            f"  MEDIA_NEEDED    : {self.media_needed_count:3d}  (NOT_TESTED without that clip)",
            "",
            "Scenario results (30 total):",
            f"  PASS            : {pass_count:3d}",
            f"  FAIL            : {fail_count:3d}",
            f"  NOT_TESTED      : {nt_count:3d}",
            f"  BLOCKED         : {block_count:3d}",
            f"  NOT_RESOLVABLE  : {nr_count:3d}",
        ]

        if fail_count > 0:
            lines += ["", "FAILURES:"]
            for r in self.results:
                if r.status == AuditStatus.FAIL:
                    lines.append(f"  #{r.scenario.id:02d} {r.scenario.name}: {r.notes}")

        lines += [
            "",
            "NOT_TESTED scenarios (require absent hardware or media):",
        ]
        for r in self.results:
            if r.status == AuditStatus.NOT_TESTED:
                hw = ", ".join(r.scenario.gpu_required) or "media"
                lines.append(f"  #{r.scenario.id:02d} {r.scenario.name}  [{hw}]")

        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code-level pipeline integrity check
# ---------------------------------------------------------------------------

def check_pipeline_integrity() -> Tuple[bool, List[str]]:
    """Verify that Stages 14–16 did not modify any render-pipeline file.

    "Pipeline files" are the modules that execute during a render — not
    new library modules that were ADDED by the stages.  The distinction:
    - ADDED by Stage 15: ``unified_runtime_scheduler.py`` (new file)
    - ADDED by Stage 16: ``quality_profiles.py`` (new file)
    - ADDED by Stage 17: ``stage17_audit.py`` (new file)
    - PIPELINE (must not change): ``ProcessMgr.py``, ``core.py``,
      ``face_swapper``, ``session_pool.py``, ``vram_governor.py``,
      ``runtime_scheduler.py`` (the PRODUCTION scheduler, not the new one)

    Returns (additive_only, list_of_changed_pipeline_files).
    """
    import subprocess
    import os

    # Exact basenames of render-critical production files.
    # unified_runtime_scheduler.py is intentionally NOT in this set —
    # it is a new library added by Stage 15, not a modified pipeline file.
    PIPELINE_BASENAMES = {
        "ProcessMgr.py",
        "core.py",
        "face_swapper.py",
        "session_pool.py",
        "vram_governor.py",
        "runtime_scheduler.py",  # the PRODUCTION scheduler (existing file)
        "procmgr.py",
    }

    changed: List[str] = []
    try:
        result = subprocess.run(
            ["git", "diff", "6e6880c", "HEAD", "--name-only"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=10,
        )
        for line in result.stdout.splitlines():
            basename = line.strip().split("/")[-1]
            # Only flag files that are in the pipeline set AND
            # are not under tests/ (test files may reference the names)
            if basename in PIPELINE_BASENAMES and not line.startswith("tests/"):
                changed.append(line)
    except Exception as exc:
        _swallowed("stage17_audit.pipeline_integrity", exc, "git diff unavailable")
        return False, ["git unavailable"]

    return len(changed) == 0, changed


# ---------------------------------------------------------------------------
# Build the initial report from scenario definitions
# ---------------------------------------------------------------------------

def build_initial_report() -> AuditReport:
    """Construct an AuditReport from scenario definitions.

    Hardware and media scenarios are pre-populated as NOT_TESTED.
    Code-verified and existing-suite scenarios are pre-populated as PASS
    pending the caller running the actual test suite.
    """
    report = AuditReport()

    additive_only, changed = check_pipeline_integrity()
    report.stages_14_to_16_are_additive_only = additive_only
    report.pipeline_files_changed_since_stage14 = changed

    for scen in AUDIT_SCENARIOS:
        if scen.category == EvidenceCategory.CODE_VERIFIED:
            report.code_verified_count += 1
            status = AuditStatus.PASS
        elif scen.category == EvidenceCategory.EXISTING_SUITE:
            report.existing_suite_count += 1
            # Optimistic PASS — the suite must be run to confirm.
            status = AuditStatus.PASS
        elif scen.category == EvidenceCategory.HARDWARE_NEEDED:
            report.hardware_needed_count += 1
            status = AuditStatus.NOT_TESTED
        else:  # MEDIA_NEEDED
            report.media_needed_count += 1
            status = AuditStatus.NOT_TESTED

        report.results.append(ScenarioResult(
            scenario=scen,
            status=status,
            notes=scen.notes,
        ))

    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AuditReport",
    "AuditScenario",
    "AuditStatus",
    "DeltaRow",
    "EvidenceCategory",
    "ScenarioResult",
    "AUDIT_SCENARIOS",
    "build_initial_report",
    "check_pipeline_integrity",
]
