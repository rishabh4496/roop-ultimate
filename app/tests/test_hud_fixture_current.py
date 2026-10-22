"""The React render check feeds `react-ui/.render-check/pipeline-hud.fixture.json` to the HUD. That file is only
evidence about the backend while its SHAPE equals what the real telemetry writers produce today, so this fails
when a field is renamed, added or dropped without `python tools/gen_hud_fixture.py` being re-run."""
import importlib.util
import json
import os
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")


def _shape(value, path=""):
    """Every key path in a JSON value (list items collapse to one `[]`); values are ignored."""
    out = set()
    if isinstance(value, dict):
        for k, v in value.items():
            out.add(f"{path}/{k}")
            out |= _shape(v, f"{path}/{k}")
    elif isinstance(value, list):
        for v in value:
            out |= _shape(v, f"{path}[]")
    return out


class HudFixtureIsCurrent(unittest.TestCase):
    def test_the_checked_in_fixture_has_the_shape_the_writers_produce(self):
        gen_path = os.path.join(ROOT, "tools", "gen_hud_fixture.py")
        spec = importlib.util.spec_from_file_location("gen_hud_fixture", gen_path)
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        with open(gen.FIXTURE, encoding="utf-8") as fh:
            checked_in = json.load(fh)
        fresh = gen.build()
        old, new = _shape(checked_in), _shape(fresh)
        self.assertEqual(sorted(new - old), [], "writers now emit fields the fixture lacks - run tools/gen_hud_fixture.py")
        self.assertEqual(sorted(old - new), [], "fixture has fields the writers no longer emit - run tools/gen_hud_fixture.py")


if __name__ == "__main__":
    unittest.main()
