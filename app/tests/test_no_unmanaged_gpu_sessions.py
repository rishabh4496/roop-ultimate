"""Every ONNX Runtime session built in the app goes through roop.strict_session,
or is on this list with the reason it must not.

A raw ``onnxruntime.InferenceSession(path, providers=[CUDA, CPU])`` is the
construction that lets a failed GPU provider degrade to a working CPU session
without a word. This test is a ratchet: a NEW raw site fails it until the author
either uses ``create_strictly_accelerated_session`` or adds a line below saying
why not. A site that disappears fails it too, so the list cannot go stale.

Scope: application code (``roop/`` and the top-level modules). Tests, tools and
the vendored env are out of scope.
"""
from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"env", "tests", "tools", "__pycache__", "node_modules", "models",
             "output", "scratch"}

# path (relative to app/) -> (number of raw InferenceSession(...) calls, reason)
UNMANAGED = {
    # -- the factory itself --------------------------------------------------
    "roop/strict_session.py": (2, "the factory: the GPU build and the by-configuration build"),

    # -- canary / probe sessions: they exist to compare against, or to ask a question of, the device
    "roop/aux_canary.py": (2, "reference + rebuild sessions the startup canary owns; its tiers end on CPU on purpose"),
    "roop/face_util.py": (1, "rebuild lambda handed to aux_canary.guard"),
    "roop/processors/Enhance_RestoreFormerPPlus.py": (1, "rebuild lambda handed to aux_canary.guard"),
    "roop/processors/Mask_XSeg.py": (2, "ROOP_TRT_BOUND input-name probe + rebuild lambda handed to aux_canary.guard"),
    "roop/processors/FaceSwapInsightFace.py": (4, "3 CPU-by-design converters/recognisers + the swap canary's reference session"),
    "roop/cudnn_algo.py": (1, "device probe: runs one conv model to see whether the cuDNN frontend fails"),
    "roop/gpu_preflight.py": (2, "the preflight probes that DECIDE admission; a tiny model on [TRT,CUDA,CPU] and [CUDA,CPU]"),
    "roop/face_analyser.py": (1, "OpenVINO usability probe"),

    # -- own ladders that already verify the built session ---------------------
    "roop/recognition_engine.py": (1, "tiered engine: warm-up inference + provider post-check per tier, final tier CPU by design"),
    "roop/processors/frame/inference_engine.py": (2, "_check_finite diagnostic + OptimizedInferenceSession (own tiers, ProviderFallbackError under strict)"),
    "roop/processors/frame/onnx_swappers.py": (2, "the non-CUDA/TensorRT branch by construction + a CPU converter"),
    "roop/optimized_processor.py": (1, "create_onnx_session: candidate ladder that rejects GPU-requested sessions with no active GPU provider; no bare-CPU candidate after a GPU request"),
    "roop/trt_session_builder.py": (2, "TensorRT-only strict builder (own assert, no fallback) + a CPU metadata read"),
    "roop/trt_bound_session.py": (2, "TensorRT bound/graph sessions: own provider check, opt-in ROOP_TRT_BOUND"),
    "roop/trt_engine.py": (1, "TensorrtInferenceSession: checks active_providers[0] itself"),
    "roop/trt_quant.py": (1, "offline calibration run"),
    "core/trt_precision_surgeon.py": (2, "TRT precision surgeon probe sessions"),
    "roop/retinaface.py": (2, "CPU default in the constructor + a build_session_with_fallback lambda (detectors keep its printed, recorded CPU rung: get_all_faces swallows detector exceptions, so a raise there would give a silently unswapped render)"),
    "update_health.py": (1, "checks the chain it asked for against session.get_providers() itself"),

    # -- measurement tools --------------------------------------------------
    "benchmark_speed.py": (1, "benchmark harness: builds the session it measures with the providers under test"),
    "roop/bench.py": (1, "stage benchmark: builds with the providers under test"),
    "roop/inference_optimizer.py": (1, "mode comparison that includes a CPU mode on purpose"),
}

# Modules whose GPU sessions were moved onto the factory. Each must still use it.
MIGRATED = [
    "roop/processors/Enhance_CodeFormer.py", "roop/processors/Enhance_GFPGAN.py",
    "roop/processors/Enhance_GPEN.py", "roop/processors/Enhance_GPEN256Pro.py",
    "roop/processors/Enhance_GPENRealistic.py", "roop/processors/Enhance_RestoreFormerPPlus.py",
    "roop/processors/Enhance_UltraMax.py", "roop/processors/Mask_FaceParser.py",
    "roop/processors/Mask_FastSAM.py", "roop/processors/Mask_MobileSAM.py",
    "roop/processors/Mask_Occluder.py", "roop/processors/Mask_XSeg.py",
    "roop/processors/Mask_XSeg3.py", "roop/processors/Frame_Masking.py",
    "roop/processors/Frame_Upscale.py", "roop/processors/Frame_Colorizer.py",
    "roop/processors/Lipsync_Wav2Lip.py", "roop/processors/FaceSwapInsightFace.py",
    "roop/processors/Expression_LivePortrait.py", "roop/processors/frame/face_swapper.py",
    "roop/processors/frame/onnx_swappers.py", "roop/rife.py", "roop/yoloface.py",
    "roop/recognizer_adaface.py",
]


def _raw_sites() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in APP.rglob("*.py"):
        relative = path.relative_to(APP)
        if any(part in SKIP_DIRS for part in relative.parts[:-1]):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "InferenceSession")
                or (isinstance(node.func, ast.Name) and node.func.id == "InferenceSession")))
        if count:
            found[relative.as_posix()] = count
    return found


def test_every_raw_session_site_is_accounted_for():
    found = _raw_sites()
    allowed = {path: count for path, (count, _reason) in UNMANAGED.items()}
    new = {p: c for p, c in found.items() if c > allowed.get(p, 0)}
    stale = {p: (allowed[p], found.get(p, 0)) for p in allowed if found.get(p, 0) < allowed[p]}
    assert not new, (
        "raw InferenceSession(...) call(s) added: %s.\n"
        "Use roop.strict_session.create_strictly_accelerated_session(path, providers=..., "
        "tag=...), or add the file to UNMANAGED in this test with the reason it must not."
        % new)
    assert not stale, (
        "UNMANAGED lists sites that no longer exist (allowed, found): %s. "
        "Lower the count or delete the line." % stale)


def test_every_unmanaged_entry_has_a_reason():
    assert all(len(reason.strip()) > 15 for _count, reason in UNMANAGED.values())


def test_migrated_modules_use_the_factory_and_have_no_raw_gpu_site_left():
    problems = []
    for relative in MIGRATED:
        source = (APP / relative).read_text(encoding="utf-8")
        if "create_strictly_accelerated_session" not in source:
            problems.append(f"{relative}: does not use create_strictly_accelerated_session")
    assert not problems, "\n".join(problems)


def load_tests(loader, tests, pattern):
    from app.tests.unittest_shim import load_tests_for
    return load_tests_for(globals())

