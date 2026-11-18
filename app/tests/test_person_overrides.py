"""Per-target-person render overrides (roop/person_overrides.py + studio_parameters.person_scope).

What each group proves, and what it does NOT:

* ``Validate`` / ``Resolve``   the wire contract: unknown names and out-of-range values are refused (never clamped), a
                               person id becomes the RANK the render addresses, a mode that cannot route by person says
                               so instead of accepting values nothing reads.
* ``Request``                  the same through the real ``normalize_processing_request`` the API calls for preview AND
                               render, including a JSON round trip (a queued job has string keys).
* ``Scope``                    precedence (settings < person < keyframe), the alias ``restore_identity_weight`` ->
                               ``codeformer_fidelity``, isolation between threads, and that two persons never share a
                               restorer batch key.
* ``ScopedFace``               the decorator on ``process_face`` really runs the body inside the right person's scope and
                               leaves a face with no override completely alone.
* ``RealPaste``                the numbers reach PIXELS: the production ``MaskingMixin.paste_upscale`` under a person
                               scope is BIT-identical to the same run with the setting changed directly, and two persons
                               painted in one frame differ.

NOT covered: a full render. No frame was swapped by a real model, so there is no fps, no pixel-noise-floor and no 3060
number here; ``restore_identity_weight`` reaching CodeFormer/UltraMax is shown through `studio_parameters.value`, the
call both of them make, not through a restorer run.
"""
import json
import os
import sys
import threading
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import person_overrides as po                      # noqa: E402
from roop import studio_parameters as sp                     # noqa: E402
from roop.processing_request import normalize_processing_request  # noqa: E402


# ── validation ────────────────────────────────────────────────────────────────────────────────
class TestValidate:
    def test_accepts_every_known_name_at_both_ends_of_its_range(self):
        for name, (lo, hi, integer) in po.PARAMETERS.items():
            assert po.validate_values({name: lo}) == {name: lo}
            assert po.validate_values({name: hi}) == {name: hi}

    def test_an_unknown_name_is_refused_and_named(self):
        with pytest.raises(po.PersonOverrideError) as e:
            po.validate_values({"face_mask_blend": 12})
        assert "face_mask_blend" in str(e.value) and "mask_feather" in str(e.value)     # says what IS known

    @pytest.mark.parametrize("name,value", [
        ("mask_feather", -0.01), ("mask_feather", 200.5), ("color_match_strength", 1.01), ("blend_ratio", -1),
        ("restore_identity_weight", 2), ("mask_erode_dilate_radius", 65), ("mask_erode_dilate_radius", -65),
    ])
    def test_out_of_range_is_refused_not_clamped(self, name, value):
        with pytest.raises(po.PersonOverrideError):
            po.validate_values({name: value})

    @pytest.mark.parametrize("bad", [None, True, "x", float("nan"), float("inf"), [], {}])
    def test_not_a_finite_number_is_refused(self, bad):
        with pytest.raises(po.PersonOverrideError):
            po.validate_values({"mask_feather": bad})

    def test_the_radius_is_a_whole_number(self):
        assert po.validate_values({"mask_erode_dilate_radius": 4.0}) == {"mask_erode_dilate_radius": 4}
        with pytest.raises(po.PersonOverrideError):
            po.validate_values({"mask_erode_dilate_radius": 4.5})

    def test_every_error_is_reported_at_once(self):
        with pytest.raises(po.PersonOverrideError) as e:
            po.validate_values({"nope": 1, "mask_feather": 999, "blend_ratio": "x"})
        assert len(e.value.errors) == 3

    def test_strings_that_are_numbers_are_accepted_like_the_settings_route_does(self):
        assert po.validate_values({"blend_ratio": "0.25"}) == {"blend_ratio": 0.25}


# ── resolution to ranks ───────────────────────────────────────────────────────────────────────
class TestResolve:
    IDS = ["p-a", "p-b", "p-c"]

    def test_person_ids_become_ranks_in_bank_order(self):
        r = po.resolve({"p-c": {"mask_feather": 5}, "p-a": {"blend_ratio": 0.5}}, self.IDS, "selected_multi")
        assert r["by_rank"] == {2: {"mask_feather": 5.0}, 0: {"blend_ratio": 0.5}}
        assert r["applied"] is True and r["reason"] is None

    def test_a_person_the_bank_does_not_have_is_reported_and_never_lands_on_somebody_else(self):
        r = po.resolve({"gone": {"mask_feather": 5}}, self.IDS, "selected")
        assert r["by_rank"] == {} and r["unknown_people"] == ["gone"] and r["applied"] is False

    @pytest.mark.parametrize("mode", ["all", "first", "all_input", "all_female", "all_male"])
    def test_a_mode_that_cannot_route_by_person_applies_nothing_and_says_why(self, mode):
        r = po.resolve({"p-a": {"mask_feather": 5}}, self.IDS, mode)
        assert r["by_rank"] == {} and r["applied"] is False
        assert mode in r["reason"] and "Selected" in r["reason"]

    def test_a_bad_value_is_refused_even_in_a_mode_that_would_not_have_used_it(self):
        with pytest.raises(po.PersonOverrideError):
            po.resolve({"p-a": {"mask_feather": 9999}}, self.IDS, "all")

    @pytest.mark.parametrize("empty", [None, {}, []])
    def test_nothing_sent_is_nothing_applied_and_no_reason(self, empty):
        r = po.resolve(empty, self.IDS, "all")
        assert r == {"by_rank": {}, "applied": False, "reason": None, "unknown_people": []}

    def test_an_empty_person_record_is_ignored(self):
        assert po.resolve({"p-a": {}}, self.IDS, "selected")["by_rank"] == {}


# ── through the real request normaliser ───────────────────────────────────────────────────────
class TestRequest:
    COMMON = dict(target_groups=[0, 1], source_count=2, target_media_index=0, target_person_ids=["p-a", "p-b"],
                  request_id="r-1")

    def _req(self, detection, overrides, **extra):
        payload = {"detection": detection, "face_mapping": {"p-a": 0, "p-b": 1},
                   "target_person_source_mapping": {"p-a": 0, "p-b": 1}, "person_overrides": overrides, **extra}
        if detection == "Selected people":
            payload["selection_state"] = {"selection_mode": "multi_person", "person_ids": ["p-a", "p-b"]}
        return normalize_processing_request(payload, **self.COMMON)

    def test_selected_people_carries_the_overrides_by_rank(self):
        r = self._req("Selected people", {"p-b": {"mask_feather": 40, "color_match_strength": 0.2}})
        assert r["person_overrides"] == {1: {"mask_feather": 40.0, "color_match_strength": 0.2}}
        assert r["person_overrides_status"] == {"applied": True, "reason": None, "unknown_people": []}

    def test_all_faces_carries_none_and_the_status_says_why(self):
        r = self._req("All faces", {"p-b": {"mask_feather": 40}})
        assert r["person_overrides"] == {}
        assert r["person_overrides_status"]["applied"] is False
        assert "does not route by person" in r["person_overrides_status"]["reason"]

    def test_no_overrides_leaves_the_request_otherwise_untouched(self):
        with_key = self._req("Selected people", {})
        without = normalize_processing_request(
            {"detection": "Selected people", "face_mapping": {"p-a": 0, "p-b": 1},
             "target_person_source_mapping": {"p-a": 0, "p-b": 1},
             "selection_state": {"selection_mode": "multi_person", "person_ids": ["p-a", "p-b"]}}, **self.COMMON)
        assert with_key == without

    def test_a_json_round_trip_still_finds_the_person_by_rank(self):
        """A queued job / saved project stores the request as JSON: int keys come back as strings."""
        r = self._req("Selected people", {"p-b": {"blend_ratio": 0.3}})
        revived = json.loads(json.dumps(r))
        assert list(revived["person_overrides"]) == ["1"]
        opts = types.SimpleNamespace(processing_request=revived)
        assert po.for_face(opts, 1) == {"blend_ratio": 0.3}
        assert po.for_face(opts, 0) == {}

    def test_a_bad_override_refuses_the_whole_request(self):
        with pytest.raises(po.PersonOverrideError):
            self._req("Selected people", {"p-a": {"blend_ratio": 5}})


# ── the scope ─────────────────────────────────────────────────────────────────────────────────
class TestScope:
    def test_outside_a_scope_nothing_is_overridden(self, monkeypatch):
        import roop.globals as g
        monkeypatch.setattr(g, "codeformer_fidelity", 0.5, raising=False)
        assert sp.person_values() == {} and sp.value("codeformer_fidelity") == 0.5

    def test_person_beats_settings_and_the_alias_is_translated(self, monkeypatch):
        import roop.globals as g
        monkeypatch.setattr(g, "codeformer_fidelity", 0.5, raising=False)
        with sp.person_scope({"restore_identity_weight": 0.9}):
            assert sp.value("codeformer_fidelity", 0.5) == 0.9       # what CodeFormer / UltraMax call
        assert sp.value("codeformer_fidelity", 0.5) == 0.5            # restored on exit

    def test_a_keyframe_beats_the_person(self):
        from roop.timeline_state import TimelineState
        import roop.timeline_state as ts
        ts.set_active(TimelineState.from_dict({"keyframes": [{"frame": 0, "mask_feather": 7.0}]}))
        try:
            with sp.frame_scope(0), sp.person_scope({"mask_feather": 60.0, "blend_ratio": 0.4}):
                assert sp.value("mask_feather") == 7.0                # keyed
                assert sp.value("blend_ratio") == 0.4                 # not keyed: the person's
        finally:
            ts.clear_active()

    def test_option_reads_the_person_blend_ratio(self):
        options = types.SimpleNamespace(blend_ratio=1.0)
        assert sp.option(options, "blend_ratio", 1.0) == 1.0
        with sp.person_scope({"blend_ratio": 0.25}):
            assert sp.option(options, "blend_ratio", 1.0) == 0.25

    def test_scopes_do_not_leak_between_threads(self):
        seen = {}

        def work(name, v):
            with sp.person_scope({"blend_ratio": v}):
                barrier.wait()
                seen[name] = sp.person_values()["blend_ratio"]
                barrier.wait()
        barrier = threading.Barrier(2)
        ts = [threading.Thread(target=work, args=("a", 0.1)), threading.Thread(target=work, args=("b", 0.9))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert seen == {"a": 0.1, "b": 0.9} and sp.person_values() == {}

    def test_two_persons_never_share_a_restorer_batch_key(self):
        keys = []
        for fid in (0.2, 0.8):
            with sp.person_scope({"restore_identity_weight": fid}):
                keys.append(sp.batch_context_key())
        with sp.person_scope({"restore_identity_weight": 0.2}):
            again = sp.batch_context_key()
        assert keys[0] != keys[1] and keys[0] == again

    def test_an_empty_scope_is_a_no_op(self):
        with sp.person_scope({}):
            assert sp.person_values() == {}


# ── the decorator ─────────────────────────────────────────────────────────────────────────────
class _FakeMgr:
    def __init__(self, table):
        self.options = types.SimpleNamespace(processing_request={"person_overrides": table})
        self.seen = []

    @po.scoped_face
    def process_face(self, face_index, target_face, frame, plate=None, region=None):
        self.seen.append((face_index, dict(sp.person_values()), sp.value("codeformer_fidelity", 0.5)))
        return frame


class TestScopedFace:
    def test_each_face_runs_inside_its_own_persons_scope(self):
        mgr = _FakeMgr({0: {"restore_identity_weight": 0.1}, 2: {"restore_identity_weight": 0.9}})
        for rank in (0, 1, 2):
            mgr.process_face(rank, object(), "frame")
        assert [(i, fid) for i, _v, fid in mgr.seen] == [(0, 0.1), (1, 0.5), (2, 0.9)]
        assert mgr.seen[1][1] == {}                                   # rank 1 has no override: no scope at all
        assert mgr.person_override_faces == 2                         # the counter a report/test reads
        assert sp.person_values() == {}                               # nothing left behind

    def test_the_scope_is_closed_even_when_the_swap_raises(self):
        class Boom(_FakeMgr):
            @po.scoped_face
            def process_face(self, face_index, target_face, frame, plate=None, region=None):
                raise RuntimeError("model fell over")
        with pytest.raises(RuntimeError):
            Boom({0: {"blend_ratio": 0.2}}).process_face(0, object(), "f")
        assert sp.person_values() == {}

    def test_no_request_no_table_no_cost(self):
        mgr = _FakeMgr({})
        mgr.process_face(0, object(), "frame")
        assert not hasattr(mgr, "person_override_faces")
        bare = _FakeMgr({})
        bare.options = types.SimpleNamespace()                         # a legacy caller with no processing_request
        bare.process_face(0, object(), "frame")

    def test_the_production_process_face_is_decorated(self):
        """Not just defined: ProcessMgr.process_face is the wrapped function."""
        pm = pytest.importorskip("roop.ProcessMgr")
        assert getattr(pm.ProcessMgr.process_face, "__wrapped__", None) is not None
        assert pm.ProcessMgr.process_face.__name__ == "process_face"


# ── the real paste ────────────────────────────────────────────────────────────────────────────
try:
    import cv2
    import roop.globals as roop_globals
    from roop.procmgr_masking import MaskingMixin
    from roop.face_parser import ParseResult, FaceParser
    from roop.occlusion_masker import OcclusionMasker
    _HAVE_PASTE = True
except Exception as _exc:                                                    # pragma: no cover - light profile
    _HAVE_PASTE = False
needs_paste = pytest.mark.skipif(not _HAVE_PASTE, reason="production paste needs the full stack")

CROP = 256


class _Stub(MaskingMixin if _HAVE_PASTE else object):
    def __init__(self, blend_ratio=1.0):
        self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=blend_ratio)
        self._tls = threading.local()
        self._tls.frame_absolute = True
        self._masker = OcclusionMasker()

    def _temporal_engine(self, name):
        return self._masker


def _scene():
    rng = np.random.default_rng(3)
    h, w, cx, cy, face = 480, 640, 320, 240, 220
    frame = np.full((h, w, 3), (120, 140, 170), np.uint8) + rng.integers(0, 12, (h, w, 3), dtype=np.uint8)
    crop = np.full((CROP, CROP, 3), (60, 90, 200), np.uint8) + rng.integers(0, 12, (CROP, CROP, 3), dtype=np.uint8)
    restored = np.clip(crop.astype(np.int16) + 40, 0, 255).astype(np.uint8)          # the "restorer's" face differs
    s = CROP * 0.8 / face
    M = np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)
    a = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    lm = np.stack([cx + face * 0.46 * np.cos(a), cy + face * 0.55 * np.sin(a)], 1).astype(np.float32)
    kps = np.array([[cx - 40, cy - 20], [cx + 40, cy - 20], [cx, cy + 5], [cx - 25, cy + 45], [cx + 25, cy + 45]],
                   np.float32)
    labels = np.zeros((CROP, CROP), np.uint8)
    cv2.ellipse(labels, (CROP // 2, CROP // 2), (int(CROP * 0.36), int(CROP * 0.44)), 0, 0, 360, 1, -1)
    return frame, crop, restored, M, lm, kps, ParseResult(labels, "bisenet")


@pytest.fixture
def paste(monkeypatch):
    monkeypatch.setenv("ROOP_GRAIN_MATCH", "0")
    monkeypatch.setattr(roop_globals, "CFG", types.SimpleNamespace(glasses_frame_protect=True), raising=False)
    monkeypatch.setattr(roop_globals, "skin_lab_match", 1.0, raising=False)
    monkeypatch.setattr(roop_globals, "mask_erode_dilate_radius", 0, raising=False)
    monkeypatch.setattr(FaceParser, "available", lambda self: False)

    def run(stub=None, offsets=None, enhanced=False, person=None):
        frame, crop, restored, M, lm, kps, parse = _scene()
        stub = stub or _Stub()
        face = restored if enhanced else crop
        with sp.person_scope(person or {}):
            return stub.paste_upscale(crop, face, M, frame.copy(), 1, offsets or [0, 0, 0, 0, 25.0, 10.0],
                                      face_landmarks=lm, face_kps=kps, inplace=False, frame_index=5,
                                      semantic_parse=parse)
    return run


@needs_paste
class TestRealPaste:
    def test_the_paste_paints_something_so_the_comparisons_below_mean_something(self, paste):
        assert not np.array_equal(paste(), _scene()[0])

    def test_no_override_is_bit_identical_to_the_pre_change_paste(self, paste):
        assert np.array_equal(paste(person={}), paste())

    def test_mask_feather_override_is_the_face_mask_blend_slider(self, paste):
        by_person = paste(person={"mask_feather": 60.0})
        direct = paste(offsets=[0, 0, 0, 0, 60.0, 10.0])
        assert np.array_equal(by_person, direct)                      # bit for bit
        assert not np.array_equal(by_person, paste())

    def test_color_match_strength_override_replaces_skin_lab_match(self, paste, monkeypatch):
        off = paste(person={"color_match_strength": 0.0})
        assert np.abs(off.astype(int) - paste().astype(int)).max() > 10     # the Lab match visibly moves the face
        monkeypatch.setattr(roop_globals, "skin_lab_match", 0.0)
        assert np.array_equal(off, paste())                                 # override 0 IS the setting at 0

    def test_blend_ratio_override_is_options_blend_ratio(self, paste):
        by_person = paste(enhanced=True, person={"blend_ratio": 0.3})
        direct = paste(stub=_Stub(blend_ratio=0.3), enhanced=True)
        assert np.array_equal(by_person, direct)
        assert not np.array_equal(by_person, paste(enhanced=True))

    def test_erode_dilate_override_is_the_global_setting(self, paste, monkeypatch):
        by_person = paste(person={"mask_erode_dilate_radius": -12})
        monkeypatch.setattr(roop_globals, "mask_erode_dilate_radius", -12)
        direct = paste()
        assert np.array_equal(by_person, direct)
        monkeypatch.setattr(roop_globals, "mask_erode_dilate_radius", 0)
        assert not np.array_equal(by_person, paste())

    def test_a_keyframe_still_wins_over_the_person_in_the_paste(self, paste):
        import roop.timeline_state as ts
        ts.set_active(ts.TimelineState.from_dict({"keyframes": [{"frame": 0, "mask_feather": 5.0}]}))
        try:
            keyed_only = paste()
            both = paste(person={"mask_feather": 120.0})
        finally:
            ts.clear_active()
        assert np.array_equal(both, keyed_only)

    def test_two_persons_in_one_frame_are_painted_differently(self, paste):
        """The end-to-end shape: the SAME decorated method, two ranks, one frame."""
        class Mgr(_Stub):
            def __init__(self):
                super().__init__()
                self.options.processing_request = {"person_overrides": {1: {"mask_feather": 90.0}}}

            @po.scoped_face
            def process_face(self, face_index, target_face, frame, plate=None, region=None):
                _f, crop, _r, M, lm, kps, parse = _scene()
                return self.paste_upscale(crop, crop, M, frame.copy(), 1, [0, 0, 0, 0, 25.0, 10.0],
                                          face_landmarks=lm, face_kps=kps, inplace=False, frame_index=5,
                                          semantic_parse=parse)
        mgr, frame = Mgr(), _scene()[0]
        out0, out1 = mgr.process_face(0, None, frame), mgr.process_face(1, None, frame)
        assert np.array_equal(out0, paste())                          # rank 0: untouched
        assert not np.array_equal(out0, out1)                         # rank 1: its own feather
        assert np.array_equal(out1, paste(offsets=[0, 0, 0, 0, 90.0, 10.0]))


# ── the API boundary ──────────────────────────────────────────────────────────────────────────
class TestApiBoundary:
    @pytest.fixture
    def api(self):
        try:
            import api as api_module
        except Exception as exc:                                    # the API pulls in the GPU stack
            pytest.skip("api module not importable here: %s" % type(exc).__name__)
        return api_module

    def test_a_refused_override_is_a_400_with_every_reason(self, api):
        import asyncio
        handler = api.app.exception_handlers[po.PersonOverrideError]
        error = po.PersonOverrideError(["person p-a: mask_feather must be within [0, 200], got 999",
                                        "person p-a: unknown override 'nope'"])
        response = asyncio.run(handler(None, error))
        body = json.loads(response.body)
        assert response.status_code == 400
        assert body["error"] == "invalid_person_overrides" and len(body["errors"]) == 2

    def test_the_preview_and_render_echo_says_whether_the_overrides_applied(self, api):
        request = normalize_processing_request(
            {"detection": "All faces", "person_overrides": {"p-a": {"mask_feather": 10}}},
            target_groups=[0], source_count=1, target_person_ids=["p-a"], request_id="r-2")
        echo = api._selection_response_fields(request, frame=1)
        assert echo["person_overrides_status"]["applied"] is False
        assert "Selected" in echo["person_overrides_status"]["reason"]
        multi = normalize_processing_request(
            {"detection": "Selected people", "person_overrides": {"p-a": {"mask_feather": 10}},
             "face_mapping": {"p-a": 0}, "target_person_source_mapping": {"p-a": 0},
             "selection_state": {"selection_mode": "multi_person", "person_ids": ["p-a"]}},
            target_groups=[0], source_count=1, target_person_ids=["p-a"], request_id="r-3")
        assert api._selection_response_fields(multi, frame=1)["person_overrides_status"]["applied"] is True
