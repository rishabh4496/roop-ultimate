"""Contract checks for the Stage 13 React UI 2.0 integration boundary.

These tests intentionally verify source-to-route ownership rather than pretend
to be browser acceptance.  Backend route behavior remains covered by the
existing API/queue/project/storage tests; a real browser is required for
interactive visual acceptance.
"""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "react-ui-v2" / "src"


def source(relative):
    return (V2 / relative).read_text(encoding="utf-8")


def _git(*args):
    """Run a read-only git query in the repository, or None if git cannot."""
    try:
        done = subprocess.run(("git",) + args, cwd=str(ROOT), text=True,
                              capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


class UI2IntegrationContractTests(unittest.TestCase):
    def test_creation_workflow_uses_verified_processing_boundary(self):
        text = source("workflow/useCreationWorkflow.js")
        for route in (
            "/api/meta", "/api/settings", "/api/state", "/api/progress",
            "/api/source/select", "/api/target/select",
            "/api/preview", "/api/swap", "/api/stop",
            "/api/pause", "/api/resume", "/api/output",
        ):
            self.assertIn(route, text)
        self.assertIn("postFiles(`/api/${kind}/add`", text)

    def test_v2_workflows_use_server_owned_queue_and_projects(self):
        queue = source("workflow/useQueue.js")
        projects = source("components/ProjectsPanel.jsx")
        for route in (
            "/api/queue", "/api/queue/add", "/api/queue/start",
            "/api/queue/pause", "/api/queue/resume", "/api/queue/stop",
            "/api/queue/cancel", "/api/queue/retry", "/api/queue/reorder",
        ):
            self.assertIn(route, queue)
        for operation in ("validateProject", "loadProject", "resumeProject"):
            self.assertIn(operation, projects)
        self.assertIn("recoverability_error", source("api.js"))

    def test_runtime_preview_and_telemetry_are_backend_owned(self):
        adapter = source("api.js")
        create = source("screens/CreateScreen.jsx")
        status = source("workflow/useOperationsStatus.js")
        for route in (
            "/api/runtime/state", "/api/system/hardware", "/api/system/profile",
        ):
            self.assertIn(route, adapter)
        for operation in ("getRuntimeState", "getHardwareProfile", "getSystemProfile"):
            self.assertIn(operation, status)
        self.assertIn("liveFrameUrl", create)
        self.assertIn("setRuntime(p.runtime || null)", source("workflow/useCreationWorkflow.js"))
        self.assertIn("pauseRequested", create)
        self.assertIn("isPaused", create)

    def test_settings_uses_only_verified_storage_and_diagnostics_routes(self):
        settings = source("screens/SettingsScreen.jsx")
        adapter = source("api.js")
        self.assertIn("/api/storage", adapter)
        self.assertIn("/api/storage/delete", adapter)
        self.assertIn("getStorageReview", settings)
        self.assertIn("deleteStorageItem", settings)
        self.assertIn("confirm: true", adapter)
        self.assertIn("SAFE_TO_DELETE", settings)
        self.assertIn("referenced", settings)
        self.assertIn("No browser update endpoint is verified", settings)

    def test_update_health_and_pinokio_boundaries_are_not_fabricated(self):
        settings = source("screens/SettingsScreen.jsx")
        self.assertIn("Pinokio-managed", settings)
        self.assertIn("full child-process updater health worker", settings)
        self.assertNotIn("/api/update", settings)
        self.assertNotIn("/api/health", settings)

    def test_v1_remains_available_and_is_not_imported_by_v2(self):
        # V1 preservation is asserted against TRACKED V1, not against the
        # `react-ui-v1-backup/` snapshot this test used to require.  That
        # directory is listed in .gitignore, so it exists only on the one
        # machine that happened to create it: this guard passed there and
        # FAILED on every other checkout, including a fresh user clone and the
        # RTX 3060 host.  A V1-preservation guard that cannot run on a second
        # machine cannot protect V1 anywhere.
        self.assertTrue((ROOT / "react-ui" / "src" / "App.jsx").is_file())
        self.assertTrue((ROOT / "react-ui" / "package.json").is_file())
        self.assertTrue((ROOT / "react-ui" / "index.html").is_file())

        # Present on disk is not the same as preserved in the repository: an
        # untracked V1 would vanish on clone.  Ask git what it actually tracks.
        tracked = _git("ls-files", "--", "react-ui")
        if tracked is not None:
            self.assertIn("react-ui/src/App.jsx", tracked.splitlines())

        # V1 must stay launchable through its own Pinokio action.
        launcher = (ROOT / "start_react.js").read_text(encoding="utf-8")
        self.assertIn("react-ui", launcher)
        self.assertNotIn("react-ui-v2", launcher)

        for path in (V2 / "App.jsx", V2 / "components" / "AppShell.jsx"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("react-ui/src", text)
            self.assertNotIn("react-ui-v1-backup", text)

    def test_v1_rollback_provenance_tag_exists(self):
        # The .gitignore entry for `react-ui-v1-backup/` states the canonical
        # backup is the `react-ui-v1` git tag.  Stage 17A recorded that tag as
        # ABSENT, which left rollback provenance resting on an ignored local
        # directory.  This asserts the immutable artifact actually exists.
        tags = _git("tag", "--list", "react-ui-v1")
        if tags is None:
            self.skipTest("git is unavailable in this environment")
        self.assertEqual(tags.strip(), "react-ui-v1")
        listing = _git("ls-tree", "-r", "--name-only", "react-ui-v1", "--",
                       "react-ui/src/App.jsx")
        self.assertEqual((listing or "").strip(), "react-ui/src/App.jsx")


if __name__ == "__main__":
    unittest.main()
