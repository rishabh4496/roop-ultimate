"""Stage 10 storage inventory and deletion safety tests."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage_manager import (PROTECTED, REVIEW_BEFORE_DELETE, SAFE_TO_DELETE,
                             StorageError, StorageManager, _canonical)


class StorageManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="storage_manager_")
        self.manager = StorageManager(self.tmp.name)
        for relative in ("app/env", "app/models", "app/output", "app/facesets",
                         "app/projects", "app/temp/api_uploads", ".pytest_cache"):
            os.makedirs(os.path.join(self.tmp.name, relative), exist_ok=True)
        self.free_upload = os.path.join(self.tmp.name, "app", "temp", "api_uploads", "free.mp4")
        self.used_upload = os.path.join(self.tmp.name, "app", "temp", "api_uploads", "used.mp4")
        for path, data in ((self.free_upload, b"free"), (self.used_upload, b"used")):
            with open(path, "wb") as handle:
                handle.write(data)

    def tearDown(self):
        self.tmp.cleanup()

    def _runtime(self, active=False, references=()):
        self.manager._runtime = lambda: {
            "references": {_canonical(path) for path in references},
            "active_work": active,
            "active_reasons": ["test active work"] if active else [],
        }

    def test_inventory_uses_references_and_known_roots(self):
        self._runtime(references=(self.used_upload,))
        cache = os.path.join(self.tmp.name, ".pytest_cache", "node")
        os.makedirs(cache)
        with open(os.path.join(cache, "state"), "wb") as handle:
            handle.write(b"cache")
        part = os.path.join(self.tmp.name, "app", "models", "model.onnx.part")
        with open(part, "wb") as handle:
            handle.write(b"partial")

        result = self.manager.scan()
        by_path = {item["relative_path"].replace("\\", "/"): item
                    for item in result["items"]}
        self.assertEqual(by_path["app/temp/api_uploads/free.mp4"]["classification"], SAFE_TO_DELETE)
        self.assertEqual(by_path["app/temp/api_uploads/used.mp4"]["classification"], PROTECTED)
        self.assertTrue(by_path["app/temp/api_uploads/used.mp4"]["referenced"])
        self.assertEqual(by_path[".pytest_cache"]["classification"], SAFE_TO_DELETE)
        self.assertEqual(by_path["app/models"]["classification"], PROTECTED)
        self.assertEqual(by_path["app/models/model.onnx.part"]["classification"], REVIEW_BEFORE_DELETE)
        names = {category["name"] for category in result["categories"]}
        self.assertIn("preview cache", names)
        self.assertIn("installers", names)
        self.assertIn("unsupported files", names)

    def test_active_work_protects_regenerable_candidates(self):
        self._runtime(active=True)
        result = self.manager.scan()
        upload = next(item for item in result["items"]
                       if item["path"] == self.free_upload)
        self.assertEqual(upload["classification"], PROTECTED)
        self.assertIn("active", upload["reason"].lower())

    def test_delete_requires_confirmation_and_fresh_safe_item(self):
        self._runtime()
        item = next(item for item in self.manager.scan()["items"]
                    if item["path"] == self.free_upload)
        with self.assertRaises(StorageError):
            self.manager.delete_item(item["id"], confirm=False)
        deleted = self.manager.delete_item(item["id"], confirm=True)
        self.assertEqual(deleted["deleted"]["id"], item["id"])
        self.assertFalse(os.path.exists(self.free_upload))

    def test_protected_and_unknown_ids_are_not_deletable(self):
        self._runtime()
        protected = next(item for item in self.manager.scan()["items"]
                         if item["path"] == os.path.join(self.tmp.name, "app", "models"))
        with self.assertRaises(StorageError):
            self.manager.delete_item(protected["id"], confirm=True)
        with self.assertRaises(StorageError):
            self.manager.delete_item("does-not-exist", confirm=True)


class StorageRouteRegistrationTest(unittest.TestCase):
    def test_storage_review_and_delete_routes_are_registered(self):
        import api
        paths = []
        for route in api.app.routes:
            owner = getattr(route, "original_router", None)
            children = owner.routes if owner is not None else [route]
            paths.extend(getattr(child, "path", None) for child in children
                         if getattr(child, "path", None))
        self.assertIn("/api/storage", paths)
        self.assertIn("/api/storage/delete", paths)


if __name__ == "__main__":
    unittest.main()
