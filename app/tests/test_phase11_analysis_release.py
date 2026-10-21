"""Phase 11 item 3: the Stage 1/2 analysis models are ALREADY unloaded when Stage 2 seals its replay.

The brief asked for the detector and the recognition session to be unloaded from VRAM after Stage 2. The site
is `ProcessMgr._release_replayed_analysis` (called at `procmgr_batch.py` right after the pre-pass), which runs
`face_util.release_face_analyser_aux` and `face_util.shrink_analysis_pools`. The older test of it
(`test_7_boost_fixes.test_fix2_...`) patches `release_face_analyser_aux` away, i.e. stubs the very thing the claim
is about. These tests run the REAL functions against stand-in analyser objects, so a regression that stops the
recognition / landmark sessions being dropped (or drops them on a PARTIAL replay, which would break the swap pass)
fails here. The measured effect on a real render is in `tests/probe_phase11_interconnect.py`:
7384 -> 5422 MiB (-1962) at the Stage 2 -> 3 boundary.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import roop.face_util as fu


class _Model:
    def __init__(self):
        self.session = object()


def _fake_analyser():
    return SimpleNamespace(
        models={'detection': _Model(), 'recognition': _Model(), 'landmark_2d_106': _Model(),
                'landmark_3d_68': _Model(), 'genderage': _Model()},
        lm68_model=_Model())


def _with_pool(pool):
    return patch.multiple(fu, FACE_ANALYSER_POOL=pool, FACE_ANALYSER=pool[0])


def test_aux_release_drops_recognition_and_landmarks_but_keeps_the_detector():
    pool = [_fake_analyser(), _fake_analyser()]
    recognisers = [fa.models['recognition'] for fa in pool]
    with _with_pool(pool), patch('torch.cuda.is_available', return_value=False):
        fu.release_face_analyser_aux()
    for fa, rec in zip(pool, recognisers):
        assert sorted(fa.models) == ['detection']            # w600k (recognition) + 106/68 landmarks + genderage gone
        assert rec.session is None                           # the ORT session itself is dropped, not just the dict entry
        assert fa.lm68_model is None


def _manager(covered, frame_count, temporal=True):
    from roop.ProcessMgr import ProcessMgr
    pm = ProcessMgr(progress=MagicMock())
    pm._temporal_mode = temporal
    pm._temporal_faces = {0: []} if temporal else None
    pm._temporal_covered = covered
    return pm


def test_complete_replay_releases_real_sessions_through_the_production_hook():
    pool = [_fake_analyser(), _fake_analyser()]
    pm = _manager(covered=100, frame_count=100)
    with _with_pool(pool), patch('torch.cuda.is_available', return_value=False), \
            patch.object(fu, 'shrink_analysis_pools', return_value={'analyser': 0, 'detectors': 0}) as shrink:
        assert pm._release_replayed_analysis(frame_count=100) is True
    assert all(sorted(fa.models) == ['detection'] for fa in pool)
    shrink.assert_called_once_with(1)
    assert pm._replay_analysis_released is True


def test_partial_replay_releases_nothing():
    """Stage 3 will still need live recognition for the frames the replay does not cover."""
    pool = [_fake_analyser()]
    pm = _manager(covered=99, frame_count=100)
    with _with_pool(pool), patch.object(fu, 'shrink_analysis_pools') as shrink:
        assert pm._release_replayed_analysis(frame_count=100) is False
    assert 'recognition' in pool[0].models and 'landmark_2d_106' in pool[0].models
    shrink.assert_not_called()


def test_no_replay_releases_nothing():
    pool = [_fake_analyser()]
    pm = _manager(covered=100, frame_count=100, temporal=False)
    with _with_pool(pool):
        assert pm._release_replayed_analysis(frame_count=100) is False
    assert 'recognition' in pool[0].models


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
