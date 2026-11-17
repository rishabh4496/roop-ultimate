"""Timeline automation (keyframes + track bindings), the frame-accurate preview contract, the waveform route, and
segment retention until the join is verified.

What each group proves, and what it does NOT:

* ``Keyframe*`` / ``Binding*``      the pure arithmetic and validation, against an independent per-frame oracle.
* ``RealPaste*``                    the keyed numbers reach PIXELS: the real ``MaskingMixin.paste_upscale`` is driven
                                    with and without a timeline, and a neutral value is BIT-identical to no timeline.
                                    The yaw>25 swap-parse is switched off (it would load BiSeNet); everything else is
                                    the production paste.
* ``GateAtSwapFaces``               the real ``ProcessMgr.swap_faces`` with ``process_face`` replaced by a recorder
                                    (the harness ``test_swap_count_is_what_was_painted`` uses): a bound track outside
                                    its ``active_ranges`` is not painted, an unbound one is.
* ``PreviewFrameApi``               the real ASGI app, a real encoded video, a real decode. ``live_swap`` and the
                                    detector are replaced where the test says so; the decode, the request handling,
                                    the encoding, the headers and the WebSocket are not.
* ``SegmentRetention``              REAL ffmpeg/ffprobe: a join that silently dropped frames is refused and the
                                    parts survive.

NOT covered here (said so in docs/CHANGELOG.md): no full render was run, so ``procmgr_tracking``'s binding hook and
the stage-cache fingerprint are shown by their unit-level behaviour and call sites, not by a pre-pass; and a
one-frame preview does not apply a binding's ``faceset_path`` source override (a render-time decision).
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from types import MappingProxyType
from unittest import mock

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop import timeline_state as ts  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_timeline():
    ts.clear_active()
    yield
    ts.clear_active()


def _kf(frame, interpolation="linear", **values):
    return {"frame": frame, "interpolation": interpolation, **values}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# keyframes
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _oracle(keyframes, frame):
    """Independent, deliberately naive evaluation: scan, no bisect, no precomputed arrays."""
    out = {}
    for name in ts.PARAMETERS:
        pts = sorted((k["frame"], k[name], k.get("interpolation", "linear")) for k in keyframes if k.get(name) is not None)
        if not pts:
            continue
        if frame <= pts[0][0]:
            out[name] = pts[0][1]
            continue
        if frame >= pts[-1][0]:
            out[name] = pts[-1][1]
            continue
        for (f0, v0, interp), (f1, v1, _) in zip(pts, pts[1:]):
            if f0 <= frame < f1:
                if frame == f0 or interp == "hold":
                    out[name] = v0
                else:
                    t = (frame - f0) / (f1 - f0)
                    t = t * t * (3 - 2 * t) if interp == "ease_in_out" else t
                    out[name] = v0 + (v1 - v0) * t
                break
    return out


class TestKeyframeEvaluation:
    def test_matches_the_oracle_on_random_timelines(self):
        rng = np.random.default_rng(11)
        for _ in range(60):
            frames = sorted(rng.choice(np.arange(0, 400), size=int(rng.integers(1, 9)), replace=False).tolist())
            kfs = []
            for f in frames:
                row = {"frame": f, "interpolation": str(rng.choice(ts.INTERPOLATIONS))}
                for name, (lo, hi) in ts.PARAMETER_RANGES.items():
                    if rng.random() < 0.6:
                        row[name] = float(rng.uniform(lo, hi))
                kfs.append(row)
            curves = ts.KeyframeCurves(kfs)
            for frame in range(-3, 410, 7):
                got, want = curves.evaluate(frame), _oracle(kfs, frame)
                assert got.keys() == want.keys()
                for k in want:
                    assert got[k] == pytest.approx(want[k], abs=1e-9), (frame, k, kfs)

    def test_exact_at_keyframes_and_held_outside(self):
        kfs = [_kf(10, swap_opacity=0.2, mask_feather=5.0), _kf(50, swap_opacity=0.9, mask_feather=40.0)]
        assert ts.evaluate_keyframes(kfs, 10) == {"swap_opacity": 0.2, "mask_feather": 5.0}
        assert ts.evaluate_keyframes(kfs, 50) == {"swap_opacity": 0.9, "mask_feather": 40.0}
        assert ts.evaluate_keyframes(kfs, 0) == {"swap_opacity": 0.2, "mask_feather": 5.0}      # before the first
        assert ts.evaluate_keyframes(kfs, 9999) == {"swap_opacity": 0.9, "mask_feather": 40.0}  # after the last

    def test_each_interpolation_means_what_it_says(self):
        base = [_kf(0, swap_opacity=0.0), _kf(10, swap_opacity=1.0)]
        assert ts.evaluate_keyframes(base, 5)["swap_opacity"] == pytest.approx(0.5)
        ease = [_kf(0, "ease_in_out", swap_opacity=0.0), _kf(10, swap_opacity=1.0)]
        assert ts.evaluate_keyframes(ease, 5)["swap_opacity"] == pytest.approx(0.5)
        assert ts.evaluate_keyframes(ease, 2)["swap_opacity"] < 0.2 < 0.5      # slow start: below the linear 0.2
        hold = [_kf(0, "hold", swap_opacity=0.25), _kf(10, swap_opacity=1.0)]
        assert ts.evaluate_keyframes(hold, 9)["swap_opacity"] == 0.25
        assert ts.evaluate_keyframes(hold, 10)["swap_opacity"] == 1.0

    def test_a_parameter_is_interpolated_only_between_ITS_keys(self):
        kfs = [_kf(0, swap_opacity=0.0, mask_feather=10.0), _kf(10, swap_opacity=1.0), _kf(20, mask_feather=30.0)]
        assert ts.evaluate_keyframes(kfs, 5)["swap_opacity"] == pytest.approx(0.5)
        assert ts.evaluate_keyframes(kfs, 10)["mask_feather"] == pytest.approx(20.0)       # 10 -> 30 across 0..20

    def test_unkeyed_parameters_are_absent_and_no_keyframes_is_empty(self):
        assert ts.evaluate_keyframes([], 5) == {}
        assert set(ts.evaluate_keyframes([_kf(3, color_match_strength=0.5)], 3)) == {"color_match_strength"}

    def test_order_and_object_form_do_not_matter(self):
        a = [_kf(0, swap_opacity=0.0), _kf(10, swap_opacity=1.0)]
        objs = [ts.ParameterKeyframe.from_dict(k) for k in reversed(a)]
        assert ts.evaluate_keyframes(a, 4) == ts.evaluate_keyframes(objs, 4)

    @pytest.mark.parametrize("bad", [
        {"keyframes": [_kf(1, swap_opacity=1.5)]},                 # out of range: rejected, never clamped
        {"keyframes": [_kf(1, swap_opacity=float("nan"))]},
        {"keyframes": [_kf(1, occlusion_sensitivity=-0.1)]},
        {"keyframes": [_kf(1, "cubic", swap_opacity=0.5)]},        # unknown interpolation
        {"keyframes": [_kf(-1, swap_opacity=0.5)]},                # negative frame
        {"keyframes": [_kf(True, swap_opacity=0.5)]},              # a bool is not a frame number
        {"keyframes": [_kf(1.5, swap_opacity=0.5)]},
        {"keyframes": [_kf(4, swap_opacity=0.1), _kf(4, swap_opacity=0.2)]},   # two keys, one frame: ambiguous
        {"keyframes": [{"frame": 1, "swap_opacty": 0.5}]},         # a typo must not be silently ignored
        {"keyframes": [{"swap_opacity": 0.5}]},                    # no frame
        {"bindings": [{"track_id": 1}, {"track_id": 1}]},          # a track bound twice
        {"bindings": [{"track_id": 1, "active_ranges": [[5, 2]]}]},
        {"bindings": [{"track_id": 1, "active_ranges": [[0]]}]},
        {"bindings": [{"track_id": 1, "locked": "yes"}]},
        {"bindings": [{"track_id": 1, "thumbnail_b64": "not base64!!"}]},
        {"bindings": [{"track_id": 1, "faceset_path": "a\x00b.fsz"}]},
        {"version": 99},
        {"unexpected": 1},
    ])
    def test_invalid_timelines_are_rejected_with_a_named_field(self, bad):
        with pytest.raises(ts.TimelineError):
            ts.TimelineState.from_dict(bad)

    def test_round_trip_is_lossless_and_sorted(self):
        state = ts.TimelineState.from_dict({
            "keyframes": [_kf(30, "hold", swap_opacity=0.4), _kf(2, mask_feather=12.0)],
            "bindings": [{"track_id": 3, "label": "Ann", "faceset_path": "C:/x/ann.fsz",
                          "thumbnail_b64": base64.b64encode(b"\xff\xd8thumb").decode(),
                          "active_ranges": [[40, 50], [0, 10], [11, 20]], "locked": True}]})
        again = ts.TimelineState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert again == state
        assert [k.frame for k in state.keyframes] == [2, 30]
        assert state.bindings[0].active_ranges == ((0, 20), (40, 50))      # touching ranges merged, sorted

    def test_a_request_with_no_timeline_changes_nothing(self):
        assert ts.TimelineState.from_payload({"frame_idx": 3}) is None
        assert ts.TimelineState.from_payload(None) is None
        flat = ts.TimelineState.from_payload({"keyframes": [_kf(1, swap_opacity=0.5)], "track_bindings": [{"track_id": 2}]})
        assert len(flat.keyframes) == 1 and flat.bindings[0].track_id == 2
        nested = ts.TimelineState.from_payload({"timeline": {"keyframes": [_kf(9, swap_opacity=0.1)]}, "keyframes": []})
        assert nested.keyframes[0].frame == 9                                # the nested form wins


class TestBindings:
    def test_active_ranges_are_inclusive_and_empty_means_everywhere(self):
        b = ts.FaceTrackBinding.from_dict({"track_id": 1, "active_ranges": [[10, 12], [20, 20]]})
        assert [f for f in range(0, 30) if b.is_active(f)] == [10, 11, 12, 20]
        assert all(ts.FaceTrackBinding.from_dict({"track_id": 2}).is_active(f) for f in (0, 5, 10 ** 6))

    ENTRIES = [{"index": 0, "id": os.path.abspath("a.fsz"), "path": os.path.abspath("a.fsz"), "faces": 3, "name": "a.fsz"},
               {"index": 1, "id": os.path.abspath("b.fsz"), "path": os.path.abspath("b.fsz"), "faces": 2, "name": "b.fsz"},
               {"index": 2, "id": os.path.abspath("empty.fsz"), "path": os.path.abspath("empty.fsz"), "faces": 0,
                "name": "empty.fsz"}]

    def _state(self, *rows):
        return ts.TimelineState.from_dict({"bindings": list(rows)})

    def test_locked_overrides_and_unlocked_only_fills(self):
        emb = np.ones(3)
        source = {1: (0, emb), 2: (0, emb), 3: (None, emb)}
        state = self._state(
            {"track_id": 1, "faceset_path": "b.fsz", "locked": True},       # matcher said 0, user locked 1
            {"track_id": 2, "faceset_path": "b.fsz", "locked": False},      # matcher already chose: left alone
            {"track_id": 3, "faceset_path": "b.fsz", "locked": False})      # matcher chose nothing: filled
        out, report = ts.apply_bindings_to_source_map(source, state, self.ENTRIES)
        assert [out[t][0] for t in (1, 2, 3)] == [1, 0, 1]
        assert out[1][1] is emb                                              # the embedding is kept
        assert [(r["track_id"], r["applied"], r["changed"]) for r in report] == [(1, True, True), (2, False, False),
                                                                                  (3, True, True)]
        assert "not locked" in report[1]["reason"]
        assert source[1][0] == 0                                             # the input mapping is not mutated

    def test_what_cannot_be_applied_is_reported_never_guessed(self):
        source = {1: (0, None)}
        state = self._state({"track_id": 9, "faceset_path": "a.fsz", "locked": True},       # no such track
                            {"track_id": 1, "faceset_path": "nowhere.fsz", "locked": True})  # not in the gallery
        out, report = ts.apply_bindings_to_source_map(source, state, self.ENTRIES)
        assert out == source
        assert not any(r["applied"] for r in report)
        assert "no such track" in report[0]["reason"] and report[1]["reason"]

    def test_an_empty_faceset_is_refused(self):
        _, report = ts.apply_bindings_to_source_map({1: (0, None)}, self._state(
            {"track_id": 1, "faceset_path": "empty.fsz", "locked": True}), self.ENTRIES)
        assert not report[0]["applied"] and "empty" in report[0]["reason"]

    def test_a_frozen_map_stays_frozen(self):
        # multi_face_tracker.locked_source only reads a MappingProxyType: losing the type silently turns identity lock off.
        out, _ = ts.apply_bindings_to_source_map(MappingProxyType({1: (0, None)}), self._state(
            {"track_id": 1, "faceset_path": "b.fsz", "locked": True}), self.ENTRIES)
        assert isinstance(out, MappingProxyType) and out[1][0] == 1

    def test_the_cache_fingerprint_tracks_what_decides_the_assignment_only(self):
        a = self._state({"track_id": 1, "faceset_path": "b.fsz", "locked": True, "active_ranges": [[0, 5]]})
        b = self._state({"track_id": 1, "faceset_path": "b.fsz", "locked": True, "active_ranges": [[0, 99]]})
        c = self._state({"track_id": 1, "faceset_path": "b.fsz", "locked": False})
        d = self._state({"track_id": 1, "faceset_path": "a.fsz", "locked": True})
        assert ts.bindings_fingerprint(a) == ts.bindings_fingerprint(b)       # ranges gate pasting, not the assignment
        assert len({json.dumps(ts.bindings_fingerprint(s)) for s in (a, c, d)}) == 3
        assert ts.bindings_fingerprint(None) == []
        assert json.loads(json.dumps(ts.bindings_fingerprint(a))) == ts.bindings_fingerprint(a)   # cache meta is JSON

    def test_gate_uses_track_ids_and_leaves_what_it_does_not_name(self):
        ts.set_active(self._state({"track_id": 7, "active_ranges": [[10, 20]]}, {"track_id": 8}))
        faces = [{"_track_id": 7, "bbox": [0, 0, 10, 10]}, {"_track_id": 8, "bbox": [50, 0, 60, 10]},
                 {"_track_id": 99, "bbox": [90, 0, 99, 10]}, {"bbox": [0, 40, 9, 49]}]
        pending = [(0, f) for f in faces]
        kept, dropped = ts.gate_pending(pending, 25)
        assert dropped == [7] and [f["_track_id"] for _, f in kept if "_track_id" in f] == [8, 99] and len(kept) == 3
        kept, dropped = ts.gate_pending(pending, 15)
        assert dropped == [] and len(kept) == 4

    def test_a_preview_face_finds_its_track_by_overlap(self):
        ts.set_active(self._state({"track_id": 7, "active_ranges": [[0, 4]]}))
        ts.set_preview_tracks([([100, 100, 200, 200], 7), ([300, 100, 400, 200], 8)])
        on_seven = {"bbox": np.array([104, 102, 198, 205], np.float32)}
        elsewhere = {"bbox": np.array([500, 500, 560, 560], np.float32)}
        kept, dropped = ts.gate_pending([(0, on_seven), (0, elsewhere)], 9)
        assert dropped == [7] and kept == [(0, elsewhere)]

    def test_nothing_active_is_a_pass_through(self):
        pending = [(0, {"_track_id": 1})]
        assert ts.gate_pending(pending, 5) == (pending, [])


class TestPersistenceInProjectCheckpoints:
    def test_a_timeline_in_the_payload_survives_checkpoint_and_project_round_trip(self, tmp_path):
        import project_checkpoint as pc
        from roop import project_io
        timeline = {"version": 1,
                    "keyframes": [_kf(5, "ease_in_out", swap_opacity=0.3, yaw_erosion_bias=-0.5)],
                    "bindings": [{"track_id": 4, "label": "Bo", "faceset_path": "bo.fsz", "thumbnail_b64": "",
                                  "active_ranges": [[0, 9]], "locked": True}]}
        record = pc.new_project(
            job_id=None, name="t", payload={"timeline": timeline, "frame": 3}, sources=[], target={"path": "x.mp4"},
            frame_start=0, frame_end=0, output={}, cfg=types.SimpleNamespace())
        stored = record["settings"]["payload"]["timeline"]
        assert ts.TimelineState.from_dict(stored) == ts.TimelineState.from_dict(timeline)
        document = project_io.from_checkpoint(record, str(tmp_path / "p.roopproj"))
        assert document["settings"]["timeline"] == stored
        back = project_io.to_checkpoint_payload(document, str(tmp_path / "p.roopproj"))
        assert ts.TimelineState.from_dict(back["settings"]["timeline"]) == ts.TimelineState.from_dict(timeline)
        # a different timeline is a different fingerprint, so a resume does not silently reuse the old render
        other = pc.new_project(job_id=None, name="t", payload={"timeline": dict(timeline, keyframes=[]), "frame": 3},
                               sources=[], target={"path": "x.mp4"}, frame_start=0, frame_end=0, output={},
                               cfg=types.SimpleNamespace())
        assert other["settings"]["fingerprint"] != record["settings"]["fingerprint"]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# the real paste
# ══════════════════════════════════════════════════════════════════════════════════════════════
try:
    import cv2
    import roop.globals as roop_globals
    from roop.procmgr_masking import MaskingMixin
    from roop.face_parser import ParseResult, FaceParser
    from roop.occlusion_masker import OcclusionMasker
    _HAVE_PASTE = True
except Exception as _exc:                                                    # pragma: no cover - light profile
    _HAVE_PASTE = False
    _PASTE_WHY = f"production paste needs the full stack: {type(_exc).__name__}"

needs_paste = pytest.mark.skipif(not _HAVE_PASTE, reason="production paste needs the full stack")

CROP = 256


class _Stub(MaskingMixin if _HAVE_PASTE else object):
    def __init__(self, absolute=True):
        self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
        self._tls = threading.local()
        self._tls.frame_absolute = absolute
        self._masker = OcclusionMasker()

    def _temporal_engine(self, name):
        return self._masker


def _scene(turn=False):
    rng = np.random.default_rng(3)
    h, w, cx, cy, face = 480, 640, 320, 240, 220
    frame = np.full((h, w, 3), (120, 140, 170), np.uint8) + rng.integers(0, 12, (h, w, 3), dtype=np.uint8)
    crop = np.full((CROP, CROP, 3), (60, 90, 200), np.uint8) + rng.integers(0, 12, (CROP, CROP, 3), dtype=np.uint8)
    s = CROP * 0.8 / face
    M = np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)
    a = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    lm = np.stack([cx + face * 0.46 * np.cos(a), cy + face * 0.55 * np.sin(a)], 1).astype(np.float32)
    nose_x = cx + 30 if turn else cx
    kps = np.array([[cx - 40, cy - 20], [cx + 40, cy - 20], [nose_x, cy + 5], [cx - 25, cy + 45], [cx + 25, cy + 45]], np.float32)
    labels = np.zeros((CROP, CROP), np.uint8)
    cv2.ellipse(labels, (CROP // 2, CROP // 2), (int(CROP * 0.36), int(CROP * 0.44)), 0, 0, 360, 1, -1)
    return frame, crop, M, lm, kps, ParseResult(labels, "bisenet")


@pytest.fixture
def paste_env(monkeypatch):
    monkeypatch.setenv("ROOP_GRAIN_MATCH", "0")
    monkeypatch.setattr(roop_globals, "CFG", types.SimpleNamespace(glasses_frame_protect=True), raising=False)
    monkeypatch.setattr(roop_globals, "skin_lab_match", 1.0, raising=False)
    monkeypatch.setattr(FaceParser, "available", lambda self: False)          # no BiSeNet load for the swap-side parse

    def run(stub, scene, frame_index=5, offsets=None, **kw):
        frame, crop, M, lm, kps, parse = scene
        return stub.paste_upscale(crop, crop, M, frame.copy(), 1, offsets or [0, 0, 0, 0, 25.0, 10.0],
                                  face_landmarks=lm, face_kps=kps, inplace=False, frame_index=frame_index,
                                  semantic_parse=parse, **kw)

    def keyed(stub, scene, frame_index=5, **values):
        ts.set_active(ts.TimelineState.from_dict({"keyframes": [_kf(0, **values)]}))
        try:
            return run(stub, scene, frame_index)
        finally:
            ts.clear_active()
    return types.SimpleNamespace(run=run, keyed=keyed)


@needs_paste
class TestRealPaste:
    def test_no_timeline_and_a_neutral_timeline_are_bit_identical(self, paste_env):
        scene = _scene(turn=True)
        base = paste_env.run(_Stub(), scene)
        assert not np.array_equal(base, scene[0])                              # the paste really painted something
        for neutral in ({"swap_opacity": 1.0}, {"yaw_erosion_bias": 0.0}, {"color_match_strength": 1.0},
                        {"occlusion_sensitivity": 1.0}, {"mask_feather": 25.0}):
            assert np.array_equal(paste_env.keyed(_Stub(), scene, **neutral), base), neutral

    def test_swap_opacity_scales_the_paste_over_the_plate(self, paste_env):
        scene = _scene()
        frame = scene[0]
        base = paste_env.run(_Stub(), scene)
        assert np.array_equal(paste_env.keyed(_Stub(), scene, swap_opacity=0.0), frame)       # 0 pastes nothing
        half = paste_env.keyed(_Stub(), scene, swap_opacity=0.5).astype(float)
        expected = frame.astype(float) + 0.5 * (base.astype(float) - frame.astype(float))
        assert np.abs(half - expected).max() <= 6.0                                          # uint8 + warp rounding
        assert np.abs(half - frame).sum() < np.abs(base.astype(float) - frame).sum()

    def test_mask_feather_is_the_face_mask_blend_slider(self, paste_env):
        scene = _scene()
        keyed = paste_env.keyed(_Stub(), scene, mask_feather=60.0)
        direct = paste_env.run(_Stub(), scene, offsets=[0, 0, 0, 0, 60.0, 10.0])
        assert np.array_equal(keyed, direct)                         # keyed 60 IS offsets[4] = 60, bit for bit
        assert not np.array_equal(keyed, paste_env.keyed(_Stub(), scene, mask_feather=5.0))

    def test_color_match_strength_replaces_skin_lab_match(self, paste_env, monkeypatch):
        scene = _scene()
        off = paste_env.keyed(_Stub(), scene, color_match_strength=0.0)
        on = paste_env.keyed(_Stub(), scene, color_match_strength=1.0)
        assert np.abs(off.astype(int) - on.astype(int)).max() > 10                  # the Lab match visibly moves the face
        monkeypatch.setattr(roop_globals, "skin_lab_match", 0.0)
        assert np.array_equal(off, paste_env.run(_Stub(), scene))                   # keyed 0 == the setting at 0
        monkeypatch.setattr(roop_globals, "skin_lab_match", 1.0)
        assert np.array_equal(paste_env.keyed(_Stub(), scene, color_match_strength=1.0), paste_env.run(_Stub(), scene))

    def test_yaw_erosion_bias_moves_only_the_turned_face_far_side(self, paste_env):
        turned, front = _scene(turn=True), _scene(turn=False)
        lo = paste_env.keyed(_Stub(), turned, yaw_erosion_bias=-1.0)
        hi = paste_env.keyed(_Stub(), turned, yaw_erosion_bias=1.0)
        assert np.abs(lo.astype(int) - hi.astype(int)).max() > 5
        # a frontal head has no yaw-driven inset to scale: the bias is inert there (and says so in the docs)
        assert np.array_equal(paste_env.keyed(_Stub(), front, yaw_erosion_bias=-1.0),
                              paste_env.keyed(_Stub(), front, yaw_erosion_bias=1.0))

    def test_a_render_and_a_preview_agree_on_which_frame_a_keyframe_is(self, paste_env, monkeypatch):
        scene = _scene()
        frame = scene[0]
        base = paste_env.run(_Stub(), scene)
        ts.set_active(ts.TimelineState.from_dict({"keyframes": [_kf(12, swap_opacity=0.0), _kf(13, swap_opacity=1.0)]}))
        # a render: clip-relative index 2 of a clip starting at absolute frame 10 (roto_frame_base = 11) IS frame 12
        monkeypatch.setattr(roop_globals, "roto_frame_base", 11, raising=False)
        assert np.array_equal(paste_env.run(_Stub(absolute=False), scene, frame_index=2), frame)
        assert np.array_equal(paste_env.run(_Stub(absolute=False), scene, frame_index=3), base)
        # a preview: live_swap hands in the ABSOLUTE index and says so
        assert np.array_equal(paste_env.run(_Stub(absolute=True), scene, frame_index=12), frame)
        assert np.array_equal(paste_env.run(_Stub(absolute=True), scene, frame_index=13), base)

    def test_a_pasted_face_reports_what_the_paste_saw(self, paste_env):
        scene = _scene(turn=True)
        target_face = {"bbox": np.array([210, 130, 430, 350], np.float32), "_pose_routed_bucket": "half_right",
                       "_pose_routed_source_indices": (2, 0)}
        ts.begin_report(frame_channels=("bisenet", "occlusion"))
        paste_env.run(_Stub(), scene, target_face=target_face, track_id=4, target_crop=scene[1])
        report = ts.end_report()
        (face,) = report["faces"]
        assert face["track_id"] == 4 and face["pose_bucket_used"] == "half_right"
        assert face["pose_source_indices"] == [2, 0] and face["parse_source"] == "bisenet"
        assert 0.0 <= face["occlusion_coverage_pct"] <= 100.0
        assert report["channels"]["bisenet"].shape == (480, 640, 4) and report["channels"]["occlusion"].shape == (480, 640)
        assert not ts.reporting()                                  # a render that is not being previewed pays nothing

    def test_a_run_that_is_not_collecting_builds_no_report_and_no_planes(self, paste_env):
        scene = _scene()
        paste_env.run(_Stub(), scene, target_face={"bbox": np.array([1, 1, 2, 2], np.float32)}, target_crop=scene[1])
        assert ts.end_report() == {"faces": [], "channels": {}, "want": ()}


def _occlusion_scene(strength=0.6):
    rng = np.random.default_rng(1)
    n = 128
    target = np.clip(np.full((n, n, 3), (110, 135, 190), int) + rng.integers(-4, 5, (n, n, 3)), 0, 255).astype(np.uint8)
    sil = np.zeros((n, n), np.float32)
    cv2.ellipse(sil, (n // 2, n // 2), (50, 58), 0, 0, 360, 1.0, -1)
    obj = target.copy()
    obj[40:90, 14:50] = np.clip(np.array([110, 135, 190]) + strength * np.array([-60, -40, 30]), 0, 255)
    cv2.rectangle(obj, (14, 40), (50, 90), (40, 40, 40), 2)
    return obj, cv2.GaussianBlur(target, (0, 0), 3), sil


@needs_paste
class TestOcclusionSensitivity:
    def test_default_is_unchanged_zero_disables_and_higher_finds_more(self):
        target, swap, sil = _occlusion_scene()
        default = OcclusionMasker().compute(target, swap, sil, labels=None)
        assert np.array_equal(default, OcclusionMasker().compute(target, swap, sil, labels=None, sensitivity=1.0))
        cover = [float(OcclusionMasker().compute(target, swap, sil, labels=None, sensitivity=s).sum())
                 for s in (0.0, 0.5, 1.0, 1.5, 2.0)]
        assert cover[0] == 0.0 and cover[2] > 0.0 and cover == sorted(cover)

    def test_keyed_sensitivity_reaches_the_real_paste_and_changes_its_pixels(self, paste_env):
        target, swap, _ = _occlusion_scene()                       # target holds an intruding object; the swap is clean
        frame, _crop, M, lm, kps, parse = _scene()
        # A hull parse: no labels reach the masker, so the boundary-connected residual rule decides (the same rule
        # the masker-level test above uses); a semantic parse would label the object as skin and absorb it.
        scene = (frame, cv2.resize(swap, (CROP, CROP)), M, lm, kps, ParseResult(parse.labels, "hull"))
        seen = []

        class Spy(_Stub):
            def _temporal_engine(self, name):
                outer = self

                class Engine:
                    def reset(self):
                        pass

                    def compute(self, *a, **kw):
                        out = outer._masker.compute(*a, **kw)
                        seen.append((kw.get("sensitivity"), float(out.sum())))
                        return out
                return Engine()

        def paste(sens):
            if sens is None:
                ts.clear_active()
            else:
                ts.set_active(ts.TimelineState.from_dict({"keyframes": [_kf(0, occlusion_sensitivity=sens)]}))
            try:
                return paste_env.run(Spy(), scene, target_crop=target)
            finally:
                ts.clear_active()

        unkeyed, one, zero, two = paste(None), paste(1.0), paste(0.0), paste(2.0)
        assert [s for s, _ in seen] == [None, 1.0, 0.0, 2.0]               # the keyed value arrives at compute()
        cover = {s: c for s, c in seen}
        assert cover[0.0] == 0.0 and cover[1.0] > 0.0 and cover[2.0] >= cover[1.0]
        assert np.array_equal(unkeyed, one)                                # 1.0 is the shipped behaviour, bit for bit
        assert not np.array_equal(zero, one)                               # and the detector's verdict reaches the pixels


# ══════════════════════════════════════════════════════════════════════════════════════════════
# the gate inside the real swap_faces
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _swap_faces_painted(track_ranges, frame_idx, mode="all"):
    from insightface.app.common import Face
    from roop.ProcessMgr import ProcessMgr

    def face(x, track):
        f = Face()
        f.bbox = np.array([x, 10, x + 40, 60], np.float32)
        f.kps = np.array([[x + 10, 25], [x + 30, 25], [x + 20, 35], [x + 12, 48], [x + 28, 48]], np.float32)
        f.det_score = 0.9
        f["_track_id"] = track
        return f

    pm = ProcessMgr(progress=mock.MagicMock())
    pm.options = mock.MagicMock()
    pm.options.swap_mode = mode
    pm.options.selected_index = 0
    pm.input_face_datas = [types.SimpleNamespace(faces=[object()])]
    pm._temporal_mode = True
    pm._temporal_faces = {frame_idx: [face(10, 7), face(100, 8)]}
    pm._temporal_covered = 100
    painted = []

    def record(self_, src_index, f, temp_frame, plate=None, region=None):
        painted.append(f["_track_id"])
        return temp_frame

    ts.set_active(ts.TimelineState.from_dict({"bindings": track_ranges}))
    with mock.patch.object(type(pm), "process_face", record):
        found, _ = pm.swap_faces(np.full((120, 200, 3), 90, np.uint8), frame_idx=frame_idx)
    return found, sorted(painted), getattr(pm, "timeline_gated_faces", 0)


class TestGateAtSwapFaces:
    @pytest.fixture(autouse=True)
    def _need_insightface(self):
        pytest.importorskip("insightface.app.common")
        pytest.importorskip("roop.ProcessMgr")

    def test_a_bound_track_outside_its_ranges_is_not_painted_and_an_unbound_one_is(self):
        found, painted, gated = _swap_faces_painted([{"track_id": 7, "active_ranges": [[0, 2]]}], frame_idx=3)
        assert (found, painted, gated) == (1, [8], 1)

    def test_inside_its_ranges_nothing_changes(self):
        found, painted, gated = _swap_faces_painted([{"track_id": 7, "active_ranges": [[0, 5]]}], frame_idx=3)
        assert (found, painted, gated) == (2, [7, 8], 0)

    def test_without_a_timeline_both_are_painted(self):
        found, painted, gated = _swap_faces_painted([], frame_idx=3)
        assert (found, painted, gated) == (2, [7, 8], 0)

    def test_the_gate_reads_the_ABSOLUTE_frame(self, monkeypatch):
        import roop.globals as g
        monkeypatch.setattr(g, "roto_frame_base", 101, raising=False)      # the clip starts at absolute frame 100
        # clip-relative 3 is absolute 103: outside [0, 2] and inside [100, 103]
        assert _swap_faces_painted([{"track_id": 7, "active_ranges": [[0, 2]]}], 3)[1] == [8]
        assert _swap_faces_painted([{"track_id": 7, "active_ranges": [[100, 103]]}], 3)[1] == [7, 8]
        assert _swap_faces_painted([{"track_id": 7, "active_ranges": [[100, 102]]}], 3)[1] == [8]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# the HTTP / WebSocket contract
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _ffmpeg():
    from roop.ffmpeg_path import ffmpeg_binary
    path = ffmpeg_binary()
    return path if path and shutil.which(path) or (path and os.path.isfile(path)) else None


def _make_video(directory, frames=12, with_audio=True):
    """12 frames of 64x48 where frame k is a flat colour that encodes k: blue = 20k, red = 255 - 20k."""
    import cv2
    silent = os.path.join(directory, "silent.mp4")
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for k in range(frames):
        writer.write(np.full((48, 64, 3), (20 * k, 128, 255 - 20 * k), np.uint8))
    writer.release()
    if not with_audio:
        return silent
    out = os.path.join(directory, "clip.mp4")
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", silent, "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1.2", "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
    return out


def _decode_telemetry(response):
    assert response.headers["x-telemetry-encoding"] == "base64url-json"
    raw = response.headers["x-telemetry"]
    return json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))


@pytest.fixture(scope="module")
def api_module():
    try:
        import api
        from fastapi.testclient import TestClient  # noqa: F401
    except Exception as exc:                                                 # pragma: no cover - light profile
        pytest.skip(f"the app does not import here: {type(exc).__name__}")
    return api


_MISSING = object()


@pytest.fixture
def client(api_module, tmp_path, monkeypatch, request):
    """The real ASGI app with a real video loaded through the real ``/api/target/add_path``."""
    if not _ffmpeg():
        pytest.skip("ffmpeg not available")
    import settings
    from fastapi.testclient import TestClient
    import roop.globals as g
    cfg = settings.Settings(str(tmp_path / "config.yaml"))
    monkeypatch.setattr(g, "CFG", cfg, raising=False)
    # preview() writes dozens of process-wide roop.globals values BY DESIGN (it is what makes a preview apply the
    # request's settings). Put every one of them back, or the next test module reads this one's defaults.
    before = dict(vars(g))
    request.addfinalizer(lambda: ([delattr(g, k) for k in set(vars(g)) - set(before)],
                                  [setattr(g, k, v) for k, v in before.items() if vars(g).get(k, _MISSING) is not v]))
    monkeypatch.setattr(api_module, "list_files_process", [])
    api_module._preview_frames.clear() if hasattr(api_module._preview_frames, "clear") else None
    monkeypatch.setitem(api_module._progress, "processing", False)
    monkeypatch.setattr(api_module, "_faces_in_target_selection_order", lambda frame: [])
    video = _make_video(str(tmp_path))
    c = TestClient(api_module.app)
    added = c.post("/api/target/add_path", json={"paths": [video]}).json()
    assert added["added"], added
    c.video = video
    return c


def _mean_bgr(jpeg_bytes):
    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img, img.reshape(-1, 3).mean(axis=0)


class TestPreviewFrameApi:
    def test_compilation_reserves_the_scan_preview_slot_and_releases_it(self, client, api_module):
        release = api_module.reserve_engine_compilation()
        assert release is not None
        try:
            assert api_module._render_active()
            assert api_module.reserve_engine_compilation() is None
            response = client.post('/api/preview/frame', json={'frame_idx': 0, 'render_mode': 'original'})
            assert response.status_code == 409
        finally:
            release()
        assert not api_module._render_active()
        assert not api_module._preview_request_lock.locked()

    def test_studio_keys_reach_the_frame_preview_contract(self, client):
        response = client.post('/api/preview/frame', json={'frame_idx': 0, 'render_mode': 'original',
            'timeline': {'keyframes': [{'frame': 0, 'detector_model': 'gnkps', 'multi_angle_detect': True,
                'pixel_boost': 768, 'preserve_inner_mouth': True, 'mask_engine': 'bisenet+xseg',
                'grain_match_strength': .6, 'ultramax_ramp': 350}]}})
        assert response.status_code == 200
        parameters = _decode_telemetry(response)['timeline']['parameters']
        assert parameters['pixel_boost'] == 768 and parameters['preserve_inner_mouth'] is True
        assert parameters['detector_model'] == 'gnkps' and parameters['ultramax_ramp'] == 350

    def test_the_binary_frame_is_the_requested_frame(self, client):
        for k in (0, 3, 7, 11):
            r = client.post("/api/preview/frame", json={"frame_idx": k, "render_mode": "original"})
            assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
            assert r.headers["x-frame-index"] == str(k) and r.headers["x-render-mode"] == "original"
            assert float(r.headers["x-render-ms"]) > 0.0
            img, mean = _mean_bgr(r.content)
            assert img.shape[:2] == (48, 64)
            assert abs(mean[0] - 20 * k) < 14 and abs(mean[2] - (255 - 20 * k)) < 14, (k, mean)   # codec tolerance
            tel = _decode_telemetry(r)
            assert tel["frame_idx"] == k and tel["frame_base"] == 0 and tel["width"] == 64 and tel["height"] == 48

    def test_frame_idx_is_zero_based_and_the_legacy_frame_field_is_one_based(self, client):
        a = client.post("/api/preview/frame", json={"frame_idx": 4, "render_mode": "original"})
        b = client.post("/api/preview/frame", json={"frame": 5, "render_mode": "original"})
        assert a.headers["x-frame-index"] == b.headers["x-frame-index"] == "4"
        assert _mean_bgr(a.content)[1][0] == pytest.approx(_mean_bgr(b.content)[1][0], abs=1.0)

    def test_webp_and_quality(self, client):
        r = client.post("/api/preview/frame", json={"frame_idx": 2, "render_mode": "original",
                                                    "image_format": "webp", "image_quality": 70})
        assert r.status_code == 200 and r.headers["content-type"] == "image/webp" and r.content[:4] == b"RIFF"

    def test_a_body_without_the_new_fields_keeps_the_legacy_json_answer(self, client):
        legacy = client.post("/api/preview/frame", json={"frame": 3})
        assert legacy.status_code == 200 and legacy.headers["content-type"].startswith("application/json")
        body = legacy.json()
        assert body["image"].startswith("data:image/") and body["target_media_id"]
        assert set(client.post("/api/preview", json={"frame": 3}).json()) == set(body)       # same shape as /api/preview

    def test_bad_requests_are_422_and_change_nothing(self, client):
        for body, needle in (({"frame_idx": -1}, "frame_idx"), ({"frame_idx": 1.5}, "frame_idx"),
                             ({"frame_idx": 1, "render_mode": "x"}, "render_mode"),
                             ({"frame_idx": 1, "image_format": "png"}, "image_format"),
                             ({"frame_idx": 1, "landmarks": "all"}, "landmarks"),
                             ({"frame_idx": 1, "image_quality": 0}, "image_quality"),
                             ({"frame_idx": 1, "keyframes": [_kf(1, swap_opacity=2)]}, "swap_opacity"),
                             ({"frame_idx": 1, "track_bindings": [{"track_id": 1}, {"track_id": 1}]}, "twice")):
            r = client.post("/api/preview/frame", json=body)
            assert r.status_code == 422 and needle in r.json()["message"], (body, r.json())
        assert ts.active() is None

    def test_a_running_render_refuses_a_preview_exactly_like_the_legacy_route(self, client, api_module, monkeypatch):
        monkeypatch.setitem(api_module._progress, "processing", True)
        r = client.post("/api/preview/frame", json={"frame_idx": 1})
        assert r.status_code == 409 and r.json()["render_active"] is True

    def test_the_timeline_is_evaluated_in_the_telemetry_and_never_left_behind(self, client):
        body = {"frame_idx": 5, "render_mode": "original",
                "keyframes": [_kf(0, swap_opacity=0.0, mask_feather=10.0), _kf(10, swap_opacity=1.0, mask_feather=30.0)],
                "track_bindings": [{"track_id": 3, "label": "A", "active_ranges": [[0, 4]]},
                                   {"track_id": 4, "active_ranges": [[0, 9]]}]}
        tel = _decode_telemetry(client.post("/api/preview/frame", json=body))
        assert tel["timeline"]["parameters"] == {"swap_opacity": pytest.approx(0.5), "mask_feather": pytest.approx(20.0)}
        assert [(b["track_id"], b["active"]) for b in tel["timeline"]["bindings"]] == [(3, False), (4, True)]
        assert ts.active() is None and not ts.reporting()                    # nothing leaks into the next request
        again = _decode_telemetry(client.post("/api/preview/frame", json={"frame_idx": 5, "render_mode": "original"}))
        assert again["timeline"]["parameters"] == {} and again["timeline"]["bindings"] == []

    def test_composite_runs_the_swap_under_the_requested_timeline_and_frame(self, client, api_module, monkeypatch):
        seen = {}

        def fake_live_swap(frame, options, input_facesets=None, *, frame_idx=None, source_video=None):
            seen.update(frame_idx=frame_idx, params=ts.params_at(frame_idx), reporting=ts.reporting(),
                        source=source_video)
            ts.report_face(bbox=[8.0, 6.0, 40.0, 40.0], track_id=3, pose_bucket_used="half_left",
                           pose_source_indices=[1], parse_source="bisenet", parse_cover=0.9, occlusion_coverage_pct=12.5)
            out = frame.copy()
            out[:, :, 1] = 0
            return out

        monkeypatch.setattr("roop.core.live_swap", fake_live_swap)
        monkeypatch.setattr("roop.core.get_processing_plugins", lambda *a, **k: [])
        monkeypatch.setattr(api_module.roop_globals, "INPUT_FACESETS", [types.SimpleNamespace(faces=[object()])],
                            raising=False)
        face = {"bbox": np.array([8, 6, 40, 40]), "kps": np.array([[14, 15], [32, 15], [23, 24], [16, 33], [30, 33]])}
        monkeypatch.setattr(api_module, "_faces_in_target_selection_order", lambda frame: [face])
        body = {"frame_idx": 6, "render_mode": "composite", "swap_mode": "all",
                "keyframes": [_kf(0, swap_opacity=0.0), _kf(12, swap_opacity=1.0)]}
        r = client.post("/api/preview/frame", json=body)
        assert r.status_code == 200, r.text
        assert seen["frame_idx"] == 6 and seen["reporting"] and seen["params"]["swap_opacity"] == pytest.approx(0.5)
        assert seen["source"] == client.video
        img, mean = _mean_bgr(r.content)
        assert mean[1] < 8                                                    # the swapped (green-zeroed) frame is served
        (f,) = _decode_telemetry(r)["faces"]
        assert f["bbox"] == pytest.approx([8 / 64, 6 / 48, 40 / 64, 40 / 48], abs=1e-4) and f["bbox_px"] == [8, 6, 40, 40]
        assert f["pasted"] and f["track_id"] == 3 and f["pose_bucket_used"] == "half_left"
        assert f["occlusion_coverage_pct"] == 12.5 and f["parse_source"] == "bisenet"
        assert f["landmarks"]["kind"] == "kps5" and len(f["landmarks"]["points"]) == 5
        assert all(0.0 <= v <= 1.0 for p in f["landmarks"]["points"] for v in p)       # normalised media coordinates
        assert _decode_telemetry(r)["swapped"] is True

    def test_diff_mode_is_zero_where_nothing_changed_and_bright_where_it_did(self, client, api_module, monkeypatch):
        def fake_live_swap(frame, options, input_facesets=None, *, frame_idx=None, source_video=None):
            out = frame.copy()
            out[10:30, 10:30] = 255 - out[10:30, 10:30]
            return out

        monkeypatch.setattr("roop.core.live_swap", fake_live_swap)
        monkeypatch.setattr("roop.core.get_processing_plugins", lambda *a, **k: [])
        monkeypatch.setattr(api_module.roop_globals, "INPUT_FACESETS", [types.SimpleNamespace(faces=[object()])],
                            raising=False)
        r = client.post("/api/preview/frame", json={"frame_idx": 1, "render_mode": "diff", "swap_mode": "all",
                                                    "diff_gain": 8})
        img, _ = _mean_bgr(r.content)
        tel = _decode_telemetry(r)
        assert tel["diff"]["max"] > 100 and 0 < tel["diff"]["changed_pct"] < 50
        assert img[20, 20].sum() > img[40, 55].sum() + 100                    # changed block is hotter than the rest

    def test_a_mask_view_with_nothing_pasted_says_why_instead_of_drawing_nothing(self, client):
        for mode in ("bisenet_mask", "occlusion_mask"):
            r = client.post("/api/preview/frame", json={"frame_idx": 2, "render_mode": mode})
            assert r.status_code == 200
            tel = _decode_telemetry(r)
            assert tel["mask_available"] is False and "no face was pasted" in tel["reason"] or "original frame" in tel["reason"]

    def test_header_telemetry_is_capped_and_says_so(self, api_module):
        faces = [{"index": i, "track_id": i, "landmarks": {"kind": "contour106", "points": [[0.123456, 0.654321]] * 106}}
                 for i in range(40)]
        tel = {"frame_idx": 1, "faces": faces}
        raw = api_module._telemetry_header(tel)
        decoded = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        assert len(raw) <= api_module._HEADER_TELEMETRY_BUDGET and decoded["truncated"] in ("contour_landmarks", "landmarks")
        assert [f["track_id"] for f in decoded["faces"]] == list(range(40))     # truncation never drops a face

    def test_cors_exposes_the_new_headers_to_a_cross_origin_client(self, client):
        r = client.post("/api/preview/frame", json={"frame_idx": 1, "render_mode": "original"},
                        headers={"Origin": "http://localhost:5173"})
        exposed = {h.strip().lower() for h in r.headers.get("access-control-expose-headers", "").split(",")}
        assert {"x-frame-index", "x-render-ms", "x-telemetry"} <= exposed

    def test_websocket_pushes_telemetry_then_the_image(self, client):
        with client.websocket_connect("/ws/preview") as ws:
            ws.send_json({"frame_idx": 9, "render_mode": "original", "seq": "a1"})
            head = ws.receive_json()
            body = ws.receive_bytes()
            assert head["event"] == "frame" and head["mime"] == "image/jpeg" and head["bytes"] == len(body)
            assert head["telemetry"]["frame_idx"] == 9 and head["telemetry"]["seq"] == "a1"
            assert abs(_mean_bgr(body)[1][0] - 180) < 14
            ws.send_json({"frame_idx": -4})                                   # a bad request answers, the socket lives on
            err = ws.receive_json()
            assert err["event"] == "error" and err["status"] == 422
            ws.send_text("{not json")
            assert ws.receive_json()["status"] == 422
            ws.send_json({"frame_idx": 1, "render_mode": "original"})
            assert ws.receive_json()["event"] == "frame" and ws.receive_bytes()


class TestMediaWaveform:
    def test_peaks_come_from_a_real_ffmpeg_pcm_decode(self, client):
        r = client.get("/api/media/waveform", params={"buckets": 40, "wait": 20})
        body = r.json()
        assert r.status_code == 200 and body["state"] == "ready" and body["ready"] is True
        assert body["duration"] == pytest.approx(1.2, abs=0.25) and 1 <= body["buckets"] <= 40
        assert len(body["peaks"]) == body["buckets"] and all(isinstance(v, int) and 0 <= v <= 255 for v in body["peaks"])
        assert max(body["peaks"]) > 60                                         # a 440 Hz sine is not silence
        wide = client.get("/api/media/waveform", params={"buckets": 2000}).json()
        assert 1 <= wide["buckets"] <= 2000

    def test_a_window_is_max_pooled(self, client):
        client.get("/api/media/waveform", params={"wait": 20})
        r = client.get("/api/media/waveform", params={"buckets": 4, "t0": 0.0, "t1": 0.5}).json()
        assert r["state"] == "ready" and r["t1"] == pytest.approx(0.5) and r["buckets"] == 4

    def test_a_file_with_no_audio_is_none_not_building(self, api_module, client, tmp_path):
        silent = _make_video(str(tmp_path), with_audio=False)
        added = client.post("/api/target/add_path", json={"paths": [silent]}).json()
        index = len(api_module.list_files_process) - 1
        assert added["added"]
        r = client.get("/api/media/waveform", params={"index": index, "wait": 20}).json()
        assert r["state"] == "none" and r["ready"] is True and r["peaks"] == []

    def test_bad_requests(self, client):
        assert client.get("/api/media/waveform", params={"buckets": 0}).status_code == 422
        assert client.get("/api/media/waveform", params={"buckets": 20001}).status_code == 422
        assert client.get("/api/media/waveform", params={"path": "C:/definitely/not/uploaded.wav"}).status_code == 404
        assert client.get("/api/media/waveform", params={"index": 99}).status_code == 404


# ══════════════════════════════════════════════════════════════════════════════════════════════
# segments: kept until the join is verified, and visible on /api/queue
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _make_part(path, frames, color):
    import cv2
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for _ in range(frames):
        w.write(np.full((48, 64, 3), color, np.uint8))
    w.release()


def _writer_with_real_parts(directory, frame_counts):
    from roop import segment_writer
    w = segment_writer.SegmentedVideoWriter.__new__(segment_writer.SegmentedVideoWriter)
    w.target_video = os.path.join(directory, "clip__temp.mp4")
    w._dir, w._seg_prefix, w._seg_ext = directory, ".clip__temp.seg", ".mp4"
    w._writer, w._cur_seg_file, w._cur_frames = None, None, 0
    w._checkpoint_callback = None
    w._identity = {"frame_start": 0, "frame_end": 0}
    w._write_lock = threading.RLock()
    w.segments = []
    for i, n in enumerate(frame_counts):
        name = f".clip__temp.seg{i:04d}.mp4"
        _make_part(os.path.join(directory, name), n, (40 * i, 100, 200))
        w.segments.append({"file": name, "frames": n, "bytes": os.path.getsize(os.path.join(directory, name))})
    with open(segment_writer.manifest_path(w.target_video), "w", encoding="utf-8") as fh:
        json.dump({"segments": w.segments}, fh)
    return w


@pytest.fixture
def real_segments(tmp_path, monkeypatch):
    import roop.globals as g
    from roop import segment_writer
    if not _ffmpeg():
        pytest.skip("ffmpeg not available")
    monkeypatch.setattr(g, "processing", True, raising=False)
    segment_writer._set_retention(state="idle")
    return segment_writer


class TestSegmentRetention:
    def _parts_exist(self, w):
        return [os.path.exists(os.path.join(w._dir, s["file"])) for s in w.segments]

    def test_a_join_that_matches_the_parts_is_verified_then_the_parts_go(self, real_segments, tmp_path):
        w = _writer_with_real_parts(str(tmp_path), [12, 8, 10])
        files = [os.path.join(w._dir, s["file"]) for s in w.segments]
        w.close()
        assert os.path.getsize(w.target_video) > 0
        assert not any(os.path.exists(f) for f in files) and not os.path.exists(real_segments.manifest_path(w.target_video))
        snap = real_segments.retention_snapshot()
        assert snap["state"] == "verified" and snap["expected_frames"] == 30 and snap["joined_frames"] == 30

    def test_a_join_that_silently_dropped_frames_is_refused_and_the_parts_survive(self, real_segments, tmp_path):
        # ffmpeg exits 0 and writes a file with only the first part: the failure mode measured on mixed-encoder joins.
        w = _writer_with_real_parts(str(tmp_path), [12, 8, 10])

        def lossy_concat():
            shutil.copy(os.path.join(w._dir, w.segments[0]["file"]), w.target_video)
            return True

        w._concat = lossy_concat
        with pytest.raises(IOError) as caught:
            w.close()
        assert "12 frames" in str(caught.value) and "hold 30" in str(caught.value)
        assert all(self._parts_exist(w))                                       # nothing the render needs was deleted
        assert os.path.exists(real_segments.manifest_path(w.target_video))
        assert not os.path.exists(w.target_video)                              # the bad join is not left for downstream
        snap = real_segments.retention_snapshot()
        assert snap["state"] == "retained_failed" and snap["joined_frames"] == 12 and snap["expected_frames"] == 30

    def test_a_join_nobody_could_measure_keeps_the_parts_but_keeps_the_output(self, real_segments, tmp_path):
        w = _writer_with_real_parts(str(tmp_path), [12, 8])
        with mock.patch("roop.finalize_output.probe_container", return_value=None):
            w.close()
        assert os.path.getsize(w.target_video) > 0 and all(self._parts_exist(w))
        assert os.path.exists(real_segments.manifest_path(w.target_video))
        snap = real_segments.retention_snapshot()
        assert snap["state"] == "retained_unverified" and "ffprobe" in snap["reason"]

    def test_a_probe_that_crashes_is_unverified_not_verified(self, real_segments, tmp_path):
        w = _writer_with_real_parts(str(tmp_path), [12, 8])
        with mock.patch("roop.finalize_output.probe_container", side_effect=RuntimeError("boom")):
            w.close()
        assert all(self._parts_exist(w)) and real_segments.retention_snapshot()["state"] == "retained_unverified"

    def test_duration_is_the_fallback_measure_when_the_header_has_no_frame_count(self, real_segments, tmp_path):
        w = _writer_with_real_parts(str(tmp_path), [12, 8])
        ok = {"video": {"frames": None, "duration_s": 2.0, "fps": 10.0}}
        bad = {"video": {"frames": None, "duration_s": 1.0, "fps": 10.0}}
        with mock.patch("roop.finalize_output.probe_container", return_value=ok):
            assert w._verify_merge()["state"] == "verified"
        with mock.patch("roop.finalize_output.probe_container", return_value=bad):
            assert w._verify_merge()["state"] == "retained_failed"
        with mock.patch("roop.finalize_output.probe_container", return_value={"video": {"frames": None}}):
            assert w._verify_merge()["state"] == "retained_unverified"

    def test_a_failed_ffmpeg_and_a_full_disk_are_reported_as_retained(self, real_segments, tmp_path):
        w = _writer_with_real_parts(str(tmp_path), [12, 8])
        w._concat = lambda: False
        with pytest.raises(IOError):
            w.close()
        assert real_segments.retention_snapshot()["state"] == "retained_failed" and all(self._parts_exist(w))

    def test_a_single_part_is_promoted_in_place_and_reported(self, real_segments, tmp_path):
        w = _writer_with_real_parts(str(tmp_path), [12])
        w.close()
        assert os.path.getsize(w.target_video) > 0
        assert real_segments.retention_snapshot()["state"] == "verified"


class TestQueueExposesSegments:
    def test_queue_lists_split_groups_and_the_render_retention_state(self, api_module, monkeypatch):
        from fastapi.testclient import TestClient
        import routes_queue
        from roop import segment_writer
        groups = [{"group": "g1", "count": 2, "segments": [{"index": 1, "state": "COMPLETED", "exists": True}]}]
        monkeypatch.setattr(routes_queue, "_segment_groups_snapshot", lambda: groups)
        segment_writer._set_retention(state="retained_failed", reason="joined file has 12 frames, the parts hold 30")
        body = TestClient(api_module.app).get("/api/queue").json()
        assert body["segment_groups"] == groups
        assert body["render_segments"]["retention"]["state"] == "retained_failed"
        assert isinstance(body["render_segments"]["parts"], list)
        assert {"jobs", "running", "paused", "current", "schema_version", "job_states"} <= set(body)   # unchanged keys

    def test_a_failure_building_the_extras_never_breaks_the_queue(self, api_module, monkeypatch):
        from fastapi.testclient import TestClient
        import routes_queue
        monkeypatch.setattr(routes_queue, "_segment_groups_snapshot", mock.Mock(side_effect=RuntimeError("x")))
        body = TestClient(api_module.app).get("/api/queue").json()
        assert body["segment_groups"] == [] and "jobs" in body
