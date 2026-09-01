"""Target-aware, single-path enhancer orchestration.

The adaptive path is deliberately a wrapper around the existing enhancer
processors.  It never chains restorers, never replaces a manually selected
processor, and does not import model code until a decision actually needs it.
The wrapper owns only the decision/cache policy; inference, precision, model
guards, and output validation remain in the established processor classes.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
import importlib
import os
from threading import RLock
from typing import Any

import cv2
import numpy as np


PROFILES = ("FAST", "BALANCED", "REALISTIC", "MAX QUALITY")
NONE = "none"

_CANDIDATES = {
    "gpen_256_pro": ("roop.processors.Enhance_GPEN256Pro", "Enhance_GPEN256Pro", {}),
    "gpen_realistic": ("roop.processors.Enhance_GPENRealistic", "Enhance_GPENRealistic", {}),
    "ultramax": ("roop.processors.Enhance_UltraMax", "Enhance_UltraMax", {}),
}

# The profile is a preference, not a promise that a strong model is better.
# Quality gates can return NONE before these preferences are consulted.
PROFILE_CANDIDATES = {
    "FAST": {"light": "gpen_256_pro", "strong": "gpen_256_pro"},
    "BALANCED": {"light": "gpen_256_pro", "strong": "gpen_realistic"},
    "REALISTIC": {"light": "gpen_realistic", "strong": "ultramax"},
    "MAX QUALITY": {"light": "gpen_realistic", "strong": "ultramax"},
}


def _clamp(value, low=0.0, high=1.0):
    return max(float(low), min(float(high), float(value)))


def _number(value, fallback=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def _face_value(face, key, default=None):
    try:
        if hasattr(face, "get"):
            return face.get(key, default)
    except Exception:
        pass
    return getattr(face, key, default)


def _sharpness(crop):
    try:
        gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_BGR2GRAY)
        if gray.shape[:2] != (160, 160):
            gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        return _clamp(variance / 350.0), variance
    except Exception:
        return 0.0, 0.0


def _resolution(face, crop):
    try:
        box = np.asarray(face.bbox, dtype=np.float32)
        side = float(min(box[2] - box[0], box[3] - box[1]))
    except Exception:
        side = float(min(np.asarray(crop).shape[:2])) if crop is not None else 0.0
    return _clamp((side - 48.0) / 208.0), side


def _pose(face):
    yaw = _number(_face_value(face, "_adaptive_yaw", 0.0), 0.0)
    pitch = _number(_face_value(face, "_adaptive_pitch", 0.0), 0.0)
    if not yaw and not pitch:
        # A cheap fallback using the detector keypoints.  ProcessMgr publishes
        # the canonical pose when it has already solved it, so this is mainly
        # for preview/tests and does not add a second pose model inference.
        try:
            kps = np.asarray(face.kps, dtype=np.float32)
            iod = max(float(np.linalg.norm(kps[1] - kps[0])), 1e-3)
            eye_mid = (kps[0] + kps[1]) * 0.5
            yaw = abs(float(kps[2, 0] - eye_mid[0])) / iod * 45.0
        except Exception:
            pass
    offaxis = max(abs(yaw), abs(pitch))
    return _clamp(1.0 - offaxis / 65.0), float(yaw), float(pitch)


def _appearance_quality(appearance):
    if not appearance:
        return 0.5, "NORMAL"
    tier = str(appearance.get("tier", "NORMAL")).upper()
    mean_luma = _number(appearance.get("mean_luma", appearance.get("p50", 0.45)), 0.45)
    # Illumination is a suitability signal, not a reason to brighten the face.
    # Dark tiers are explicitly retained for the low-strength/no-hallucination
    # policy below.
    return _clamp(mean_luma / 0.48), tier


def evaluate_face_frame(face, crop, appearance=None, previous=None,
                        output_quality=0.5, occlusion=0.0,
                        identity_detail_required=None):
    """Return all selector inputs for one aligned target face.

    Every value is bounded and serialisable so the same contract can be used by
    tests, telemetry, and the live wrapper. ``output_quality`` is the previous
    candidate's observed quality; it cannot be known before inference.
    """
    sharp, sharp_var = _sharpness(crop)
    resolution, face_px = _resolution(face, crop)
    pose, yaw, pitch = _pose(face)
    illumination, tier = _appearance_quality(appearance)
    confidence = _clamp(_number(_face_value(face, "det_score", 0.0), 0.0))
    motion = abs(_number(_face_value(face, "_temporal_motion", 0.0), 0.0))
    temporal = _clamp(1.0 - motion / 45.0)
    if previous:
        # A crop signature is cheap and catches unstable alignment even when a
        # tracker did not publish a motion value.
        try:
            current = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
            prior = _number(previous.get("luma"), current)
            temporal *= _clamp(1.0 - abs(current - prior) / 80.0)
        except Exception:
            pass
    quality = (0.24 * resolution + 0.24 * sharp + 0.14 * pose
               + 0.14 * illumination + 0.12 * confidence
               + 0.12 * temporal)
    if identity_detail_required is None:
        detail_required = _number(os.environ.get(
            "ROOP_IDENTITY_DETAIL_STRENGTH", "0"), 0.0) > 0.0
    else:
        detail_required = bool(identity_detail_required)
    return {
        "resolution": round(resolution, 6),
        "face_px": round(face_px, 3),
        "sharpness": round(sharp, 6),
        "sharpness_var": round(sharp_var, 3),
        "blur": round(1.0 - sharp, 6),
        "pose": round(pose, 6),
        "yaw": round(yaw, 3),
        "pitch": round(pitch, 3),
        "illumination": round(illumination, 6),
        "low_light_tier": tier,
        "occlusion": round(_clamp(occlusion), 6),
        "confidence": round(confidence, 6),
        "temporal_stability": round(temporal, 6),
        "output_quality": round(_clamp(output_quality), 6),
        "quality": round(_clamp(quality), 6),
        "identity_detail_required": bool(detail_required),
        "luma": round(_number(appearance.get("p50", 0.45) if appearance else 0.45), 6),
    }


def choose_enhancer(metrics, profile="BALANCED", current=None,
                    available=None, small_card=False):
    """Select exactly one path: ``none`` or one existing restorer.

    Geometry, confidence, occlusion, and temporal instability veto aggressive
    restoration. Very dark frames also veto strong restoration. ``current`` is
    retained in the hysteresis band so a stable shot cannot alternate models.
    """
    profile = str(profile or "BALANCED").upper()
    if profile not in PROFILES:
        profile = "BALANCED"
    available = set(available or _CANDIDATES)
    q = _clamp(metrics.get("quality", 0.0))
    pose = _clamp(metrics.get("pose", 0.0))
    temporal = _clamp(metrics.get("temporal_stability", 0.0))
    confidence = _clamp(metrics.get("confidence", 0.0))
    occlusion = _clamp(metrics.get("occlusion", 0.0))
    tier = str(metrics.get("low_light_tier", "NORMAL")).upper()
    high_cut = {"FAST": 0.63, "BALANCED": 0.68, "REALISTIC": 0.72,
                "MAX QUALITY": 0.76}[profile]
    if small_card:
        return NONE, "small-card-safety"
    if pose < 0.18:
        return NONE, "extreme-angle-geometry-first"
    if confidence < 0.28 or occlusion > 0.72:
        return NONE, "low-confidence-or-occluded"
    if temporal < 0.28:
        return current or NONE, "temporal-instability-hold"
    if q >= high_cut:
        # Once a candidate is active, require a small quality margin before
        # dropping it. This is decision hysteresis, not output blending: it
        # keeps a stable shot from alternating between a restorer and null.
        if current and current != NONE and q < high_cut + 0.06 and tier == "NORMAL":
            return current, "path-hysteresis"
        return NONE, "high-quality-face-minimal-enhancement"
    if tier == "VERY_DARK":
        return NONE, "very-dark-no-hallucination"
    if current and current != NONE and q > high_cut - 0.06 and tier == "NORMAL":
        return current, "path-hysteresis"
    pref = PROFILE_CANDIDATES[profile]
    if tier == "DARK" or q < 0.38:
        wanted = pref["light"]
        reason = "dark-or-very-low-quality-light-restoration"
    else:
        wanted = pref["strong"]
        reason = "moderate-quality-restoration"
    if (metrics.get("output_quality", 0.5) < 0.40 and tier == "NORMAL"
            and pose >= 0.40):
        wanted = pref["strong"]
        reason = "observed-output-quality-restoration"
    if metrics.get("identity_detail_required") and tier != "NORMAL":
        wanted = pref["light"]
        reason = "identity-detail-protection"
    if wanted not in available:
        wanted = next((x for x in (pref["light"], pref["strong"])
                       if x in available), NONE)
        reason += "-fallback"
    return wanted, reason


def output_quality(result, source):
    """Measure finite/range/detail retention without claiming visual quality."""
    if result is None:
        return 0.0
    try:
        out = np.asarray(result)
        if out.size == 0 or not np.isfinite(out).all():
            return 0.0
        src = np.asarray(source)
        spread = float(cv2.meanStdDev(out)[1].mean())
        src_spread = max(float(cv2.meanStdDev(src)[1].mean()), 1.0)
        detail = _clamp(spread / (src_spread * 1.8))
        return _clamp(0.65 + 0.35 * detail)
    except Exception:
        return 0.0


@dataclass
class Decision:
    path: str
    reason: str
    metrics: dict


class AdaptiveEnhancer:
    """Lazy, bounded wrapper around the existing face-restoration processors."""

    processorname = "adaptive_enhancer"
    type = "enhance"
    # ProcessMgr performs one optional template alignment for the wrapper; all
    # built-in adaptive candidates use the same FFHQ alignment contract.
    model_template = "ffhq_512"
    self_excluding = True

    def __init__(self):
        self.plugin_options = None
        self.profile = "BALANCED"
        self._loaded = OrderedDict()
        self._active = defaultdict(int)
        self._lock = RLock()
        self._previous = {}
        self._last_path = {}
        self._last_quality = {}
        self._decisions = []
        self._max_loaded = 2
        self._small_card = False
        self._fallback_seen = {}

    def Initialize(self, plugin_options: dict):
        if self._loaded:
            self.Release()
        self.plugin_options = dict(plugin_options or {})
        self.profile = str(self.plugin_options.get("adaptive_profile", "BALANCED")).upper()
        if self.profile not in PROFILES:
            self.profile = "BALANCED"
        try:
            vram = float(self.plugin_options.get("vram_gb", 0.0) or 0.0)
        except (TypeError, ValueError):
            vram = 0.0
        self._small_card = 0.0 < vram < 7.0
        default_max = 1 if self._small_card else 2
        try:
            self._max_loaded = max(1, min(3, int(os.environ.get(
                "ROOP_ADAPTIVE_MAX_LOADED", default_max))))
        except (TypeError, ValueError):
            self._max_loaded = default_max
        print(f"[AdaptiveEnhancer] profile={self.profile} max_loaded={self._max_loaded} "
              f"small_card={self._small_card}; one candidate per face", flush=True)

    def _build(self, name):
        module_name, class_name, extra = _CANDIDATES[name]
        cls = getattr(importlib.import_module(module_name), class_name)
        proc = cls()
        opts = dict(self.plugin_options or {})
        opts.update(extra)
        proc.Initialize(opts)
        return proc

    def _candidate(self, name):
        with self._lock:
            proc = self._loaded.get(name)
            if proc is None:
                proc = self._build(name)
                self._loaded[name] = proc
            self._loaded.move_to_end(name)
            self._active[name] += 1
            return proc

    def _trim(self):
        with self._lock:
            while len(self._loaded) > self._max_loaded:
                name, proc = next(iter(self._loaded.items()))
                if self._active[name]:
                    self._loaded.move_to_end(name)
                    if all(self._active[n] for n in self._loaded):
                        break
                    continue
                self._loaded.pop(name, None)
                try:
                    proc.Release()
                except Exception as exc:
                    print(f"[AdaptiveEnhancer] release {name} skipped: {exc}", flush=True)

    def _report_fallback(self, path, kind, exc):
        """Announce a fall back to the unenhanced frame -- once per cause.

        Both call sites sit on the PER-FACE path, so an unbounded print here is
        two lines per face: 120,000 of them on a 60,000-frame two-face render,
        which is its own failure. Bounded the way face_util bounds swallowed
        detector failures. The running total is reported at Release, because
        "it fell back 4,000 times" is the number that matters and a single
        first-occurrence line does not carry it.
        """
        sig = (path, kind, type(exc).__name__, str(exc)[:200])
        with self._lock:
            seen = sig in self._fallback_seen
            self._fallback_seen[sig] = self._fallback_seen.get(sig, 0) + 1
        if seen:
            return
        print(f"[AdaptiveEnhancer] candidate {path} {kind}; using unenhanced "
              f"frame: {type(exc).__name__}: {exc}\n"
              f"[AdaptiveEnhancer] Further occurrences of this cause are "
              f"counted, not printed; the total is reported at Release.",
              flush=True)

    def fallback_counts(self):
        """Per-cause fallback totals, for harness and test reporting."""
        with self._lock:
            return dict(self._fallback_seen)

    def Run(self, source_faceset, target_face, temp_frame):
        metrics = _face_value(target_face, "_adaptive_metrics", None) or {}
        if not metrics:
            metrics = evaluate_face_frame(target_face, temp_frame)
        track = _face_value(target_face, "_track_id", None)
        key = track if track is not None else "preview"
        metrics = dict(metrics)
        metrics["output_quality"] = self._last_quality.get(
            key, metrics.get("output_quality", 0.5))
        current = self._last_path.get(key)
        path, reason = choose_enhancer(
            metrics, self.profile, current=current, small_card=self._small_card)
        self._decisions.append(Decision(path, reason, dict(metrics)))
        if len(self._decisions) > 400:
            del self._decisions[:-400]
        if path == NONE:
            self._last_path[key] = NONE
            self._previous[key] = {"luma": metrics.get("luma", 0.45)}
            return None, 0
        try:
            proc = self._candidate(path)
        except Exception as exc:
            self._report_fallback(path, "unavailable", exc)
            self._last_path[key] = NONE
            return None, 0
        try:
            result, scale = proc.Run(source_faceset, target_face, temp_frame)
            quality = output_quality(result, temp_frame)
            self._last_quality[key] = quality
            self._last_path[key] = path
            self._previous[key] = {"luma": metrics.get("luma", 0.45)}
            try:
                target_face["_adaptive_output_quality"] = quality
                target_face["_adaptive_path"] = path
            except Exception:
                pass
            return result, scale
        except Exception as exc:
            self._report_fallback(path, "failed", exc)
            self._last_path[key] = NONE
            return None, 0
        finally:
            with self._lock:
                self._active[path] = max(0, self._active[path] - 1)
            self._trim()

    def telemetry(self):
        with self._lock:
            counts, reasons, scores = {}, {}, []
            for item in self._decisions:
                counts[item.path] = counts.get(item.path, 0) + 1
                reasons[item.reason] = reasons.get(item.reason, 0) + 1
                # An ABSENT quality is not a quality of zero. Defaulting it to
                # 0.0 would pull the reported band down and make a
                # high-scoring population read as a low-scoring one -- which
                # inverts the conclusion this band exists to support.
                try:
                    raw = item.metrics.get("quality")
                except AttributeError:
                    raw = None
                if raw is not None:
                    try:
                        scores.append(float(raw))
                    except (TypeError, ValueError):
                        pass
            band = None
            if scores:
                ordered = sorted(scores)
                band = {"n": len(ordered),
                        "min": round(ordered[0], 4),
                        "p50": round(ordered[len(ordered) // 2], 4),
                        "max": round(ordered[-1], 4)}
            return {
                "profile": self.profile,
                "max_loaded": self._max_loaded,
                "small_card": self._small_card,
                "decisions": counts,
                # The REASON each decision was taken, and the quality score that
                # drove it. Without these a run is only reportable as
                # `decisions={'none': 60}`, which says the selector enhanced
                # nothing and not why -- and `last_quality` is empty in exactly
                # that case, because it is only written after a candidate runs.
                # So the one number that explains the behaviour was absent
                # precisely when it was needed.
                "reasons": reasons,
                "quality_band": band,
                "fallbacks": {"%s %s: %s: %s" % k: v
                              for k, v in self._fallback_seen.items()},
                "last_quality": dict(self._last_quality),
                "loaded": list(self._loaded),
                "recent": [asdict(x) for x in self._decisions[-20:]],
            }

    def Release(self):
        summary = self.telemetry()
        if summary["decisions"]:
            print(f"[AdaptiveEnhancer] decisions={summary['decisions']} "
                  f"reasons={summary['reasons']} "
                  f"quality={summary['quality_band']} "
                  f"last_quality={summary['last_quality']}", flush=True)
            # An adaptive selector that restored nothing is a legitimate
            # outcome of its own policy and is also indistinguishable, in every
            # other instrument, from the enhancer never having run: the render
            # returns 0, the swap audit reads 100%, an `enhance` stage is
            # counted (the wrapper is still called), and the arm comes out as
            # the FASTEST in an enhancer sweep. Measured on this 4070: BALANCED
            # on the locked d4 fixture chose `none` for 60 of 60 faces at
            # 0.0 ms/face and 1.95 fps against 1.87 for `--enhancer None`.
            if set(summary["decisions"]) <= {NONE}:
                print("[AdaptiveEnhancer] NO FACE WAS ENHANCED in this run. "
                      "The output is the unenhanced swap. The reason counts "
                      "above say which gate refused; if it is "
                      "'high-quality-face-minimal-enhancement', every face "
                      "scored at or above this profile's cut and the profile "
                      "considers them good enough. Select a restorer by name "
                      "to enhance unconditionally.", flush=True)
        # The total, not just the first occurrence. A selector that fell back to
        # the unenhanced frame on most faces is indistinguishable from one that
        # chose NONE on purpose unless this is said out loud -- and both produce
        # a run that returns 0 with a swap audit reading 100%.
        if summary["fallbacks"]:
            total = sum(summary["fallbacks"].values())
            print(f"[AdaptiveEnhancer] FELL BACK to the unenhanced frame "
                  f"{total} time(s): {summary['fallbacks']}", flush=True)
        with self._lock:
            loaded = list(self._loaded.values())
            self._loaded.clear()
            self._previous.clear()
            self._last_path.clear()
            self._last_quality.clear()
        for proc in loaded:
            try:
                proc.Release()
            except Exception as exc:
                print(f"[AdaptiveEnhancer] candidate release skipped: {exc}", flush=True)


__all__ = [
    "AdaptiveEnhancer", "Decision", "NONE", "PROFILES", "PROFILE_CANDIDATES",
    "choose_enhancer", "evaluate_face_frame", "output_quality",
]
