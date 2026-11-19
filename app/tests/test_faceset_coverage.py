"""roop.faceset_coverage: the library gallery's left / front / right badge data.

Real ZIP archives are written to a temp dir, so the reader, the legacy path, the unsolved-pose path and the cache are
all exercised against the on-disk format, not a stub. ``test_library_entries_carry_pose_coverage`` drives the real
route builder to prove the field the React badge reads is actually emitted.
"""
from __future__ import annotations

import json
import os
import zipfile

import pytest

from roop import faceset_coverage as fc


def _meta(yaws):
    return {"format": "roop.fsz", "version": 2,
            "sources": [{"source_index": i, "geometry": {"yaw": y, "pitch": 0.0}} for i, y in enumerate(yaws)]}


def _write(path, metadata=None, members=("reference_000.png",)):
    with zipfile.ZipFile(path, "w") as zf:
        for m in members:
            zf.writestr(m, b"png")
        if metadata is not None:
            zf.writestr("metadata.json", json.dumps(metadata))


def test_bins_follow_the_archives_own_yaw_edges():
    c = fc.yaw_coverage(_meta([-60, -30, -24.9, 0, 24.9, 25.1, 70]))
    assert (c["left"], c["front"], c["right"]) == (2, 3, 2)
    assert (c["deep_left"], c["deep_right"]) == (1, 1)         # only |yaw| >= 50 is a true profile
    assert c["missing"] == [] and c["complete"] is True and c["total"] == 7


def test_frontal_only_faceset_reports_both_sides_missing():
    c = fc.yaw_coverage(_meta([0, 3, -5, 10]))
    assert c["missing"] == ["left", "right"] and c["complete"] is False


def test_one_sided_faceset_names_the_missing_side():
    c = fc.yaw_coverage(_meta([0, -45, -70]))
    assert c["missing"] == ["right"] and c["left"] == 2 and c["deep_left"] == 1


def test_unsolved_poses_are_not_counted_as_frontal():
    c = fc.yaw_coverage(_meta([None, float("nan"), "x", True, 0.0]))
    assert c["front"] == 1 and c["unsolved"] == 4 and c["total"] == 5


def test_nothing_solved_is_not_complete():
    c = fc.yaw_coverage(_meta([None, None]))
    assert c["complete"] is False and c["unsolved"] == 2


@pytest.mark.parametrize("bad", [None, [], "x", {}, {"sources": "nope"}])
def test_unusable_metadata_is_unknown_not_empty(bad):
    assert fc.yaw_coverage(bad) is None


def test_archive_roundtrip_and_legacy_archive(tmp_path):
    v2, legacy = str(tmp_path / "v2.fsz"), str(tmp_path / "legacy.fsz")
    _write(v2, _meta([0, -50, 55]))
    _write(legacy, None, members=("a.png", "b.png"))
    c = fc.coverage_for_fsz(v2)
    assert (c["left"], c["front"], c["right"]) == (1, 1, 1)
    assert fc.coverage_for_fsz(legacy) is None                 # no metadata.json: angles unknown, none invented


def test_corrupt_or_missing_archive_is_unknown(tmp_path):
    bad = tmp_path / "bad.fsz"
    bad.write_bytes(b"not a zip")
    assert fc.coverage_for_fsz(str(bad)) is None
    assert fc.coverage_for_fsz(str(tmp_path / "nope.fsz")) is None


def test_malformed_metadata_json_is_unknown(tmp_path):
    p = str(tmp_path / "m.fsz")
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("metadata.json", "{not json")
    assert fc.coverage_for_fsz(p) is None


def test_cache_is_keyed_by_size_and_mtime(tmp_path):
    p = str(tmp_path / "c.fsz")
    _write(p, _meta([0]))
    first = fc.coverage_for_fsz(p)
    assert fc.coverage_for_fsz(p) is first                     # same object: served from the cache
    _write(p, _meta([0, -60, 60]))
    os.utime(p, ns=(1, 3_000_000_000))                         # changed file: stamp differs, re-read
    again = fc.coverage_for_fsz(p)
    assert again is not first and again["total"] == 3 and again["complete"] is True


def test_library_entries_carry_pose_coverage(tmp_path, monkeypatch):
    routes = pytest.importorskip("routes_faceset")
    _write(str(tmp_path / "Front only.fsz"), _meta([0, 4]))
    _write(str(tmp_path / "Old.fsz"), None)
    monkeypatch.setattr(routes, "_faceset_library_dir", lambda: str(tmp_path))
    monkeypatch.setattr(routes, "_library_thumb_dataurl", lambda p: "")
    by_name = {e["name"]: e for e in routes._library_entries()}
    assert by_name["Front only"]["pose_coverage"]["missing"] == ["left", "right"]
    assert by_name["Old"]["pose_coverage"] is None             # present, null: the badge shows "angles unknown"
