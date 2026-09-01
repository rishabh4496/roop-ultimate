"""Stage 8B persistence and recoverability tests."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import project_checkpoint as pc  # noqa: E402


class _Config:
    provider = "cuda"
    trt_precision = "mixed"
    hardware_signature = "v2|test-machine"
    hardware = {
        "hardware_profile_key": "test-gpu",
        "gpu": "test-gpu",
        "vram_tier": "desktop",
        "vram_gb": 12.0,
        "ram_gb": 32.0,
    }
    output_video_format = "mp4"
    output_video_codec = "libx264"
    video_quality = 14
    output_method = "File"
    output_template = "{file}_{time}"


class ProjectCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="project_checkpoint_")
        self.old_dir = pc.PROJECTS_DIR
        pc.PROJECTS_DIR = self.tmp.name
        self.source = os.path.join(self.tmp.name, "source.png")
        self.target = os.path.join(self.tmp.name, "target.mp4")
        self.partial = os.path.join(self.tmp.name, ".target.seg0000.mp4")
        for path, data in ((self.source, b"source-v1"), (self.target, b"target-v1"),
                           (self.partial, b"valid-segment")):
            with open(path, "wb") as fh:
                fh.write(data)

    def tearDown(self):
        pc.PROJECTS_DIR = self.old_dir
        self.tmp.cleanup()

    def _new(self):
        return pc.new_project(
            job_id="job-1", name="target.mp4",
            payload={"swap_model": "inswapper", "output_method": "File"},
            sources=[pc.file_identity(self.source)],
            target=pc.file_identity(self.target), frame_start=0, frame_end=10,
            output={"directory": self.tmp.name, "format": "mp4", "codec": "libx264",
                    "quality": 14, "method": "File", "template": "{file}_{time}"},
            cfg=_Config(), target_faces=[], app_version="main@test")

    def test_pause_close_reload_validate_resume_and_integrity(self):
        record = self._new()
        pc.update_checkpoint(
            record["id"], safe_frame=4, next_frame=4,
            segments=[{"file": os.path.basename(self.partial), "frames": 4,
                        "bytes": os.path.getsize(self.partial),
                        "sha256": pc.file_sha256(self.partial)}],
            manifest=self.partial + ".resume.json",
            partial_files=[pc.file_identity(self.partial)], state="PAUSED")

        # Simulate application close/reopen: only the JSON record is reloaded.
        reloaded = pc.load(record["id"])
        self.assertEqual(reloaded["state"], "PAUSED")
        self.assertEqual(reloaded["checkpoint"]["next_frame"], 4)
        self.assertEqual(pc.validate(reloaded, _Config()), [])

        output = os.path.join(self.tmp.name, "final.mp4")
        with open(output, "wb") as fh:
            fh.write(b"completed-output")
        pc.update_checkpoint(record["id"], safe_frame=10, next_frame=10,
                             segments=[], partial_files=[pc.file_identity(output)],
                             state="COMPLETED")
        completed = pc.load(record["id"])
        self.assertEqual(completed["state"], "COMPLETED")
        self.assertEqual(pc.validate(completed, _Config()), [])
        self.assertEqual(pc.file_sha256(output), completed["partial_output"]["files"][0]["sha256"])

    def test_changed_input_is_a_recoverability_error(self):
        record = self._new()
        with open(self.target, "ab") as fh:
            fh.write(b"changed")
        reasons = pc.validate(record, _Config())
        self.assertTrue(any("target changed" in reason for reason in reasons))

    def test_checkpoint_write_is_atomic_and_has_no_temp_leftover(self):
        record = self._new()
        path = pc.project_path(record["id"])
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(self.tmp.name)))
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["schema_version"], pc.PROJECT_SCHEMA_VERSION)

    def test_execution_provider_alias_is_stable_across_runtime_observation(self):
        self.assertEqual(pc.runtime_identity({}, _Config())["provider"], "cuda")
        self.assertEqual(
            pc.runtime_identity({}, _Config(), ["CUDAExecutionProvider"])["provider"],
            "cuda")


if __name__ == "__main__":
    unittest.main()
