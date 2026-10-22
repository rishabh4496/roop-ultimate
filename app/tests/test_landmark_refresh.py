"""roop/landmark_refresh.py and its use by the pre-pass gap bridge.

The helper is tested with a fake landmark model (anything with ``get(img, face)``); the bridge by calling
``TrackingMixin._flow_bridge_geometry`` on hand-built flow results. The scan-loop hook itself is a nested closure, so it is
proven separately by running the real pre-pass and reading its counters (docs/perf/flow_vs_neural/).
"""
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

pytest.importorskip("insightface")
from roop import landmark_refresh as lr          # noqa: E402
from roop.flow_coast import FlowResult, spread_residual  # noqa: E402

N = 106


def _lm(seed, spread=40.0):
    rng = np.random.default_rng(seed)
    return (rng.random((N, 2)) * spread + 100.0).astype(np.float32)


KPS = np.array([[110, 110], [150, 110], [130, 130], [115, 150], [145, 150]], np.float32)     # eye distance 40


def _result(lm, **kw):
    return FlowResult(consistent=True, landmarks=lm, kps=KPS.copy(), bbox=np.array([90, 90, 170, 170], np.float32),
                      transform=None, **kw)


class FakeModel:
    def __init__(self, answer=None, raises=False):
        self.answer, self.raises, self.seen = answer, raises, []

    def get(self, img, face):
        self.seen.append(np.asarray(face.bbox).copy())
        if self.raises:
            raise RuntimeError("boom")
        if self.answer is not None:
            face["landmark_2d_106"] = self.answer


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ROOP_PREPASS_LANDMARK_REFRESH", raising=False)
    lr.reset()


FRAME = np.zeros((300, 300, 3), np.uint8)


class TestRefreshLandmarks:
    def test_off_by_default_and_switchable(self, monkeypatch):
        assert not lr.enabled()
        monkeypatch.setenv("ROOP_PREPASS_LANDMARK_REFRESH", "1")
        assert lr.enabled()

    def test_the_net_is_run_on_the_flow_box_and_its_answer_returned(self):
        flow = _lm(1)
        answer = flow + 1.5
        model = FakeModel(answer)
        got = lr.refresh_landmarks(FRAME, _result(flow), model)
        assert np.allclose(got, answer) and got.dtype == np.float32 and got.shape == (N, 2)
        assert np.array_equal(model.seen[0], [90, 90, 170, 170])

    def test_an_answer_far_from_the_flow_geometry_is_refused(self):
        flow = _lm(2)
        got = lr.refresh_landmarks(FRAME, _result(flow), FakeModel(flow + 20.0))     # 0.5 x eye distance
        assert got is None and "1 answers refused" in lr.summary_lines()[0]

    @pytest.mark.parametrize("frame,model,key", [(None, FakeModel(), "no_frame"), (FRAME, None, "no_model")])
    def test_missing_inputs_are_counted_not_raised(self, frame, model, key):
        assert lr.refresh_landmarks(frame, _result(_lm(3)), model) is None
        assert lr._stats[key] == 1

    def test_a_model_that_raises_keeps_the_flow_landmarks_and_is_counted(self):
        assert lr.refresh_landmarks(FRAME, _result(_lm(4)), FakeModel(raises=True)) is None
        assert lr._stats["failed"] == 1

    def test_a_wrong_shape_answer_is_a_failure(self):
        assert lr.refresh_landmarks(FRAME, _result(_lm(5)), FakeModel(np.zeros((68, 2), np.float32))) is None
        assert lr._stats["failed"] == 1

    def test_nan_answer_is_a_failure(self):
        bad = _lm(6)
        bad[3, 0] = np.nan
        assert lr.refresh_landmarks(FRAME, _result(_lm(6)), FakeModel(bad)) is None

    def test_counters_and_summary(self):
        assert lr.summary_lines() == []
        flow = _lm(7)
        lr.refresh_landmarks(FRAME, _result(flow), FakeModel(flow + 0.5))
        lr.refresh_landmarks(None, _result(flow), FakeModel())
        text = lr.summary_lines()[0]
        assert "1/2 flow-placed face(s)" in text and "1 without a frame" in text
        lr.reset()
        assert lr.summary_lines() == []


def _bridge():
    try:
        from roop.procmgr_tracking import TrackingMixin
    except Exception as exc:                                     # pragma: no cover - light profile
        pytest.skip(f"pre-pass needs the full stack: {type(exc).__name__}")
    return TrackingMixin._flow_bridge_geometry


class Face(dict):
    __getattr__ = dict.get


def _scenario(drift=0.0):
    a = Face(landmark_2d_106=_lm(10), kps=KPS.copy(), bbox=np.array([90, 90, 170, 170], np.float32))
    b_lm = _lm(10) + 3.0
    b = Face(landmark_2d_106=b_lm, kps=KPS + 3.0, bbox=np.array([93, 93, 173, 173], np.float32))
    steps = {g: _result(_lm(10) + 1.0 * (g - 10) + drift) for g in (11, 12)}
    end = _result(b_lm - 0.4)                 # flow arrives 0.4 px short of the closing detection
    return a, b, {**steps, 13: end}


class TestBridgeUsesRefreshedSteps:
    def test_unrefreshed_steps_get_the_drift_correction_as_before(self):
        a, b, flow = _scenario()
        out = _bridge()(types.SimpleNamespace(), {"flow": flow}, a, b, 10, 13)
        expect = spread_residual([flow[11].landmarks, flow[12].landmarks], flow[13].landmarks, b["landmark_2d_106"])
        assert np.allclose(out[11][0], expect[0]) and np.allclose(out[12][0], expect[1])

    def test_a_refreshed_step_keeps_exactly_what_the_net_measured(self):
        a, b, flow = _scenario()
        measured = _lm(99)
        flow[12].landmarks, flow[12].refreshed = measured, True
        out = _bridge()(types.SimpleNamespace(), {"flow": flow}, a, b, 10, 13)
        assert np.array_equal(out[12][0].astype(np.float32), measured)
        expect = spread_residual([flow[11].landmarks, measured], flow[13].landmarks, b["landmark_2d_106"])
        assert np.allclose(out[11][0], expect[0])                   # its un-refreshed neighbour is unchanged

    def test_the_closing_detection_still_vouches_for_the_chain(self):
        a, b, flow = _scenario()
        flow[13] = _result(b["landmark_2d_106"] + 30.0)             # flow ended far from the closing detection
        for r in (flow[11], flow[12]):
            r.landmarks, r.refreshed = _lm(5), True
        me = types.SimpleNamespace()
        assert _bridge()(me, {"flow": flow}, a, b, 10, 13) is None
        assert me._flow_refused == 2

    def test_flow_result_defaults_to_not_refreshed(self):
        assert _result(_lm(1)).refreshed is False


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
