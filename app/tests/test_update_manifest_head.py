"""HEAD must accept itself as an update candidate.

app/update_manager.py admits a fetched commit only when that commit's
update_manifest.json passes evaluate_manifest(). Between the manifest's
introduction and 2026-09-22 no commit on main carried one, so every existing
install would have been classified UNVERIFIED and left where it was. These
tests fail the suite (and CI) before that can happen again:

  * the manifest at HEAD is present and is exactly what
    tools/gen_update_manifest.py renders from HEAD's dependency contract;
  * the updater's own validation, given HEAD's manifest and HEAD's tree the
    way it reads a fetched candidate, admits HEAD for an installation that
    satisfies the declared contract -- and, on the GPU machine, for THIS one.

Everything is read from `HEAD:` through git, never from the working tree, so
the test sees what a user's `git fetch` would.
"""

import copy
import os
import subprocess
import sys
import unittest

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for p in (ROOT, APP, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import update_manager  # noqa: E402
import gen_update_manifest  # noqa: E402


def _head_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def _conforming_installation(manifest, candidate_hashes):
    """An installation that satisfies exactly what the manifest declares.

    The updater compares the candidate against local evidence (Python,
    provider, GPU, runtime versions, installed sensitive-file hashes). This is
    that evidence for a machine on the declared contract, so the only thing
    under test is the manifest and HEAD's tree."""
    compat = manifest["compatibility"]
    runtime = {name: rule[2:] if str(rule).startswith("==") else rule
               for name, rule in compat["runtime"].items()}
    return {
        "python": compat["python"]["min"],
        "platform": compat["platforms"][0],
        "provider": compat["providers"][0],
        "available_providers": [f"{name.upper()}ExecutionProvider" for name in compat["providers"]],
        "hardware": {
            "profile": sorted(update_manager.MANDATORY_HARDWARE_PROFILES)[0],
            "compute_capability": sorted(update_manager.MANDATORY_GPU_ARCHITECTURES)[0],
        },
        "runtime": runtime,
        "tracked_file_hashes": copy.deepcopy(candidate_hashes),
        "dirty": False,
        "active_work": [],
    }


class HeadManifestTests(unittest.TestCase):
    def test_head_manifest_is_committed_and_generated(self):
        problems = gen_update_manifest.check("HEAD")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_head_is_admissible_for_an_installation_on_the_declared_contract(self):
        manifest = update_manager._load_candidate_manifest("HEAD")
        self.assertIsNotNone(manifest, "HEAD has no update_manifest.json")
        candidate_hashes = update_manager._candidate_file_hashes("HEAD")
        current = _conforming_installation(manifest, candidate_hashes)
        result = update_manager.evaluate_manifest(manifest, _head_sha(), current, candidate_hashes)
        self.assertEqual(result["classification"], "SAFE", "\n".join(result["reasons"]))

    def test_every_sensitive_file_hash_matches_head_tree(self):
        # The identity binding: a manifest copied from another commit whose
        # dependency contract differs must not validate against this tree.
        manifest = update_manager._load_candidate_manifest("HEAD")
        self.assertIsNotNone(manifest)
        candidate_hashes = update_manager._candidate_file_hashes("HEAD")
        for relative in update_manager.SENSITIVE_FILES:
            self.assertEqual(manifest["tracked_file_hashes"].get(relative), candidate_hashes[relative],
                             f"{relative}: manifest hash differs from HEAD's blob")

    def test_a_foreign_manifest_is_rejected_against_head_tree(self):
        manifest = update_manager._load_candidate_manifest("HEAD")
        self.assertIsNotNone(manifest)
        candidate_hashes = update_manager._candidate_file_hashes("HEAD")
        foreign = copy.deepcopy(manifest)
        foreign["tracked_file_hashes"]["app/requirements.txt"] = "0" * 64
        current = _conforming_installation(manifest, candidate_hashes)
        result = update_manager.evaluate_manifest(foreign, _head_sha(), current, candidate_hashes)
        self.assertEqual(result["classification"], "UNVERIFIED")

    @pytest.mark.gpu
    def test_head_is_admissible_on_this_machine(self):
        # The real evidence collector: nvidia-smi, the installed torch/ORT/TRT,
        # config.yaml's provider, the working tree's own sensitive files. A
        # dirty checkout or a local sensitive-file edit is REQUIRES REVIEW by
        # design; UNVERIFIED or INCOMPATIBLE means this machine could not take
        # HEAD as an update.
        manifest = update_manager._load_candidate_manifest("HEAD")
        self.assertIsNotNone(manifest)
        current = update_manager._current_identity()
        result = update_manager.evaluate_manifest(manifest, _head_sha(), current,
                                                  update_manager._candidate_file_hashes("HEAD"))
        self.assertNotIn(result["classification"], ("UNVERIFIED", "INCOMPATIBLE"),
                         "\n".join(result["reasons"]))


if __name__ == "__main__":
    unittest.main()
