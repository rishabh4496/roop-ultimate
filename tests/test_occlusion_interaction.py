"""Stage 19: a second face or an object crossing the target, through the real pipeline.

Three deterministic 30-frame synthetic scenes (app/tests/synthetic_scenes.py) are rendered
losslessly through `two_face_video.py` (realswap + RealityUX, tracking on) and graded
against geometry that is known exactly (app/tests/occlusion_interaction.py):

  1. a bystander slides across the target, sprite IoU peaking at 0.75, drawn in front of
     the target and then behind it
  2. a skin-coloured bar sweeps over the target's eyes, nose and mouth
  3. the same bar scene: temporal flicker and colour of the swapped skin

Two kinds of test per metric, both driven by app/tests/occlusion_gates.py:

  RATCHET  what the pipeline does today. These must pass; a regression fails the suite.
  BRIEF    the thresholds the Stage 19 brief asked for. The ones the pipeline does not meet
           are `xfail(strict=True)`: the suite stays green, and goes red the day one of them
           starts passing (so the entry gets promoted) or its measured value is no longer
           the documented gap.

Needs a CUDA/TensorRT device and models (marker `gpu`); ~8 minutes, rendered once per
session. Run from the repo root:  pytest tests/test_occlusion_interaction.py
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'app'), os.path.join(ROOT, 'app', 'tests')):
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = pytest.mark.gpu

import occlusion_gates as gates      # noqa: E402


@pytest.fixture(scope='session')
def metrics(tmp_path_factory):
    import occlusion_interaction as oi
    return oi.run_all(str(tmp_path_factory.mktemp('s19')))


def _result(metrics, entry):
    return gates.check(metrics, [entry])[0]


@pytest.mark.parametrize('entry', gates.RATCHET, ids=[e[0] for e in gates.RATCHET])
def test_ratchet(metrics, entry):
    _name, ok, line = _result(metrics, entry)
    assert ok, line


def _brief_params():
    out = []
    for entry in gates.BRIEF:
        marks = []
        if entry[0] in gates.KNOWN_MISSES:
            marks.append(pytest.mark.xfail(strict=True, reason=gates.KNOWN_MISSES[entry[0]]))
        out.append(pytest.param(entry, id=entry[0], marks=marks))
    return out


@pytest.mark.parametrize('entry', _brief_params())
def test_brief_threshold(metrics, entry):
    _name, ok, line = _result(metrics, entry)
    assert ok, line


def test_every_known_miss_names_a_brief_entry():
    """A KNOWN_MISSES key that matches no BRIEF entry would silently never xfail anything."""
    ids = {e[0] for e in gates.BRIEF}
    assert set(gates.KNOWN_MISSES) <= ids
