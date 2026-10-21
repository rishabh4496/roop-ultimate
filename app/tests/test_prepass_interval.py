"""`prepass_detect_interval` (K): one detector call per K frames, flow-carried faces between.

What is pinned here, and why each one is a test of its own:

* the POLICY (`roop/prepass_policy.py`): K is clamped to 1-8, an unset or junk value is the default, and the loss
  threshold grows with K (the default 3 would call every track of K >= 5 lost between two key frames);
* the SCHEDULE under the lag the real pre-pass has: a detector result is consumed up to pool_workers + 2 frames
  after its frame was planned. Consuming frame t - 4 used to move "the last detection" back to t - 4, so the next
  plan saw a gap of K and detected again: K = 4 measured 900 detector calls over 900 frames and 0 coast frames
  (Weeds.mp4, 2026-10-21). `OldScheduler` below is that behaviour; the test fails on it, which is what proves the
  test can tell;
* K = 1 is untouched: it detects on every frame and never coasts;
* a shot cut restarts the count (a forced full detection, then K frames of cadence from it);
* the settings plumbing: the setting, its env variable, its live-env membership, and that something READS it.
"""
import collections
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import prepass_policy as pp                            # noqa: E402
from roop.temporal_tracker import TemporalFaceTracker            # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


SHAPE = (480, 640)


# ── policy ───────────────────────────────────────────────────────────────────────────────────────
class TestPolicy:
    def test_interval_is_clamped_to_one_through_eight(self):
        assert [pp.clamp_interval(v) for v in (0, -3, 1, 4, 8, 9, 100)] == [1, 1, 1, 4, 8, 8, 8]

    def test_unset_or_junk_is_the_default(self):
        assert pp.detect_interval({}) == pp.DETECT_INTERVAL_DEFAULT
        for junk in ('', 'auto', 'AUTO', 'x', 'nan', None):
            env = {} if junk is None else {'ROOP_PREPASS_DETECT_INTERVAL': junk}
            assert pp.detect_interval(env) == pp.DETECT_INTERVAL_DEFAULT, junk
        assert pp.detect_interval({'ROOP_PREPASS_DETECT_INTERVAL': '4'}) == 4
        assert pp.detect_interval({'ROOP_PREPASS_DETECT_INTERVAL': '4.0'}) == 4
        assert pp.detect_interval({'ROOP_PREPASS_DETECT_INTERVAL': '99'}) == 8

    def test_a_track_survives_the_gap_between_two_key_frames(self):
        # undetected for K - 1 frames in a row is normal; one missed key frame doubles it
        for k in range(1, 9):
            assert pp.max_misses_for(k, 3) >= (2 * k - 1 if k > 1 else 3)
        assert pp.max_misses_for(1, 3) == 3                      # K = 1 keeps today's number exactly
        assert pp.max_misses_for(8, 3) == 15
        assert pp.max_misses_for(2, 9) == 9                       # a larger configured value is never lowered

    def test_the_filter_kind_is_chosen_or_the_default(self):
        assert pp.filter_kind({}) == pp.FILTER_DEFAULT
        for kind in pp.FILTER_KINDS:
            assert pp.filter_kind({'ROOP_PREPASS_FILTER': kind}) == kind
        assert pp.filter_kind({'ROOP_PREPASS_FILTER': 'One-Euro'}) == 'one_euro'
        assert pp.filter_kind({'ROOP_PREPASS_FILTER': 'bogus'}) == pp.FILTER_DEFAULT


# ── the schedule, under the real pre-pass's plan/consume lag ───────────────────────────────────────
def _face(frame, x=300.0):
    emb = np.random.default_rng(7).normal(size=512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    cx = x + 0.5 * frame
    return {'bbox': np.array([cx - 60, 200.0, cx + 60, 340.0], np.float32),
            'kps': np.array([[cx - 25, 240], [cx + 25, 240], [cx, 270], [cx - 20, 305], [cx + 20, 305]], np.float32),
            'det_score': 0.95, 'embedding': emb}


def drive(tracker, n_frames, lag, cuts=()):
    """plan() a frame when it is dispatched, update() with its result `lag` frames later - exactly the order the
    pre-pass's in-flight queue produces. Returns the plan modes per frame."""
    pending = collections.deque()
    modes = []
    for f in range(n_frames):
        plan = tracker.plan(f, SHAPE, force_full=(f in cuts))
        modes.append(plan.mode)
        pending.append((f, [] if plan.mode == 'coast' else [_face(f)], plan.mode))
        if len(pending) > lag:
            g, dets, mode = pending.popleft()
            tracker.update(dets, g, SHAPE, detection_mode=mode)
    while pending:
        g, dets, mode = pending.popleft()
        tracker.update(dets, g, SHAPE, detection_mode=mode)
    return modes


def tracker(k):
    t = TemporalFaceTracker(full_interval=8, max_misses=pp.max_misses_for(k, 3), roi_interval=k)
    t.detect_interval = k
    return t


class OldScheduler(TemporalFaceTracker):
    """The scheduler as it was: a consumed result overwrites the dispatch-time marks with its own, older, frame."""

    def _note_requested(self, frame_index):
        self._last_requested_frame = int(frame_index)

    def _note_full(self, frame_index):
        self._last_full_frame = int(frame_index)
        self._full_pending_frame = None
        self._last_requested_frame = int(frame_index)


def calls(modes):
    return sum(1 for m in modes if m != 'coast')


class TestSchedule:
    @pytest.mark.parametrize('lag', [0, 3, 5, 8])
    def test_interval_four_detects_about_a_quarter_of_the_frames(self, lag):
        modes = drive(tracker(4), 400, lag)
        steady = modes[40:]
        share = calls(steady) / len(steady)
        # a key frame every 4, a full pass every 8 (a key frame itself): 1/4 of the frames, and nothing like 1/1
        assert 0.22 <= share <= 0.30, (lag, share)

    def test_the_old_scheduler_ignored_the_interval_under_lag(self):
        """The defect, reproduced: with a lag the old marks moved backwards and every frame was a detection."""
        t = OldScheduler(full_interval=8, max_misses=7, roi_interval=4)
        t.detect_interval = 4
        modes = drive(t, 400, 5)
        assert calls(modes[40:]) / len(modes[40:]) > 0.9

    def test_interval_one_never_coasts_and_detects_every_frame(self):
        for lag in (0, 5):
            modes = drive(tracker(1), 200, lag)
            assert 'coast' not in modes
            assert calls(modes) == 200

    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7, 8])
    def test_a_steady_face_is_never_called_lost_at_any_interval(self, k):
        t = tracker(k)
        drive(t, 240, 5)
        assert t.stats['lost_tracks'] == 0, (k, t.stats)
        assert t.stats['new_tracks'] == 1, (k, t.stats)           # one face, one track, however sparse the detections

    def test_with_the_default_loss_threshold_interval_eight_would_lose_the_track(self):
        """Why `max_misses_for` exists: 7 undetected frames in a row exceed the default of 3."""
        t = TemporalFaceTracker(full_interval=8, max_misses=3, roi_interval=8)
        t.detect_interval = 8
        drive(t, 240, 5)
        assert t.stats['lost_tracks'] > 0

    def test_a_shot_cut_forces_a_full_detection_and_restarts_the_count(self):
        modes = drive(tracker(4), 200, 5, cuts={100})
        assert modes[100] == 'full'
        # after the cut: a key frame, then coast frames - the count restarted from the cut, not from frame 0
        after = modes[101:112]
        assert 'coast' in after and after.count('coast') >= 6, after

    def test_coast_frames_are_counted_not_hidden(self):
        t = tracker(4)
        modes = drive(t, 200, 5)
        assert t.stats['coast_frames'] == modes.count('coast') > 100

    def test_from_env_reads_the_interval_at_use_time(self, monkeypatch):
        monkeypatch.setenv('ROOP_PREPASS_DETECT_INTERVAL', '5')
        monkeypatch.delenv('ROOP_TEMPORAL_ROI_INTERVAL', raising=False)
        monkeypatch.delenv('ROOP_TEMPORAL_MAX_MISSES', raising=False)
        t = TemporalFaceTracker.from_env()
        assert (t.detect_interval, t.roi_interval, t.max_misses) == (5, 5, 9)
        monkeypatch.setenv('ROOP_PREPASS_DETECT_INTERVAL', '1')
        t = TemporalFaceTracker.from_env()
        assert (t.detect_interval, t.roi_interval, t.max_misses) == (1, 1, 3)

    def test_the_older_roi_interval_still_wins_when_it_is_larger(self, monkeypatch):
        monkeypatch.setenv('ROOP_PREPASS_DETECT_INTERVAL', '2')
        monkeypatch.setenv('ROOP_TEMPORAL_ROI_INTERVAL', '6')
        assert TemporalFaceTracker.from_env().roi_interval == 6


# ── the setting ──────────────────────────────────────────────────────────────────────────────────
class TestSetting:
    def test_the_setting_exists_with_the_policy_default(self):
        import settings
        inst = settings.Settings(os.path.join(APP, '__no_such_config__.yaml'))
        assert inst.prepass_detect_interval == pp.DETECT_INTERVAL_DEFAULT
        assert inst.prepass_landmark_filter == pp.FILTER_DEFAULT

    def test_it_is_exported_live_to_the_variable_the_tracker_reads(self):
        import settings
        env = {}
        settings.apply_env({'prepass_detect_interval': 4, 'prepass_landmark_filter': 'one_euro'}, env)
        assert env['ROOP_PREPASS_DETECT_INTERVAL'] == '4'
        assert env['ROOP_PREPASS_FILTER'] == 'one_euro'
        assert 'prepass_detect_interval' in settings.LIVE_ENV_SETTINGS
        assert 'prepass_landmark_filter' in settings.LIVE_ENV_SETTINGS

    def test_an_environment_value_wins_over_the_saved_config(self):
        import settings
        env = {'ROOP_PREPASS_DETECT_INTERVAL': '2'}
        settings.apply_env({'prepass_detect_interval': 6}, env)
        assert env['ROOP_PREPASS_DETECT_INTERVAL'] == '2'          # an A/B arm owns its environment

    def test_something_reads_what_the_setting_writes(self):
        """A control bound to a variable nothing reads looks completely wired (AGENTS.md, settings step 5)."""
        import re
        readers = {'ROOP_PREPASS_DETECT_INTERVAL': 'roop/prepass_policy.py',
                   'ROOP_PREPASS_FILTER': 'roop/prepass_policy.py'}
        for var, rel in readers.items():
            text = open(os.path.join(APP, rel), encoding='utf-8').read()
            assert re.search(r"environ\.get\(\s*['\"]%s['\"]|env\.get\(\s*['\"]%s['\"]" % (var, var), text), (var, rel)
        # and the consumers of those readers
        tracker_src = open(os.path.join(APP, 'roop', 'temporal_tracker.py'), encoding='utf-8').read()
        assert 'prepass_policy.detect_interval()' in tracker_src
        tracking_src = open(os.path.join(APP, 'roop', 'procmgr_tracking.py'), encoding='utf-8').read()
        assert 'prepass_policy.filter_kind()' in tracking_src
