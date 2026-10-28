"""Stage 15: Fast Schema Parity and API Contract Tests.

Verifies that FastAPI route response models in `app/api.py` and its mounted routers
match the expected JSON keys consumed by the React UI and declared in
`react-ui/src/types/api-contract.ts`.
Runs fast (< 50ms) under the CI light profile without ML stack dependencies.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
if APP not in sys.path:
    sys.path.insert(0, APP)

import settings


class TestApiSchemaParity(unittest.TestCase):
    def setUp(self):
        self.api_path = os.path.join(APP, "api.py")
        with open(self.api_path, "r", encoding="utf-8") as f:
            self.api_source = f.read()
        self.contract_path = os.path.join(ROOT, "react-ui", "src", "types", "api-contract.ts")
        with open(self.contract_path, "r", encoding="utf-8") as f:
            self.contract_source = f.read()

    def _extract_return_keys_for_function(self, source: str, func_name: str) -> set[str]:
        """Parse AST and extract dictionary string keys from Return statements of a function."""
        tree = ast.parse(source)
        keys: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                for ret in ast.walk(node):
                    if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict):
                        for k in ret.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                keys.add(k.value)
        return keys

    def test_state_route_ast_schema_parity(self):
        """Verify get_state in app/api.py returns the keys required by AppStateResponse."""
        keys = self._extract_return_keys_for_function(self.api_source, "get_state")
        required_keys = {
            "source_faces",
            "source_faces_info",
            "targets",
            "selected_target_index",
            "selected_source_index",
            "selected_source_id",
            "faceset_count",
        }
        missing = required_keys - keys
        self.assertEqual(missing, set(), f"Missing required keys in get_state: {missing}")

        # Also verify _target_context_payload provides target face fields
        target_keys = self._extract_return_keys_for_function(self.api_source, "_target_context_payload")
        required_target_keys = {"target_faces", "target_groups", "target_names", "target_faces_info"}
        missing_target = required_target_keys - target_keys
        self.assertEqual(missing_target, set(), f"Missing target context keys: {missing_target}")

    def test_progress_route_ast_schema_parity(self):
        """Verify get_progress unpacks _progress and returns operational progress keys."""
        self.assertIn('"output": _last_output', self.api_source)
        self.assertIn('"live_seq": runtime["frame_progress"]["live_seq"]', self.api_source)
        self.assertIn('"eta_s": runtime["eta_s"]', self.api_source)
        self.assertIn('"status_line": runtime["status"]["message"]', self.api_source)
        # Default keys in _progress dictionary
        self.assertIn('_progress = {"processing": False', self.api_source)
        self.assertIn('"progress": 0', self.api_source)

    def test_settings_parity_with_schema_json(self):
        """Verify settings defined in Settings match app/settings.schema.json."""
        schema_path = os.path.join(APP, "settings.schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        schema_props = set(schema.get("settings", {}).keys())

        cfg_dummy = os.path.join(APP, "config.yaml")
        s = settings.Settings(cfg_dummy)
        active_settings = set(s.__dict__.keys())

        core_required = {
            "swap_model",
            "selected_enhancer",
            "mask_engine",
            "provider",
            "trt_precision",
            "max_threads",
            "video_quality",
            "output_video_format",
            "output_video_codec",
            "face_detector_threshold",
            "face_detector_nms",
        }
        missing_props = core_required - schema_props
        self.assertEqual(missing_props, set(), f"Missing settings in settings.schema.json: {missing_props}")
        missing_active = core_required - active_settings
        self.assertEqual(missing_active, set(), f"Missing settings in Settings class: {missing_active}")

    def test_dashboard_route_ast_schema_parity(self):
        """Verify terminal dashboard route publishes dashboard snapshot."""
        dash_router_path = os.path.join(APP, "routes_pipeline_status.py")
        with open(dash_router_path, "r", encoding="utf-8") as f:
            dash_source = f.read()
        self.assertIn('get("/api/v1/pipeline/dashboard")', dash_source)
        self.assertIn("terminal_dashboard.snapshot()", dash_source)

    def test_trt_cache_route_schema_parity(self):
        """Verify routes_trt_cache.py exports expected TrtCacheSnapshot keys."""
        trt_router_path = os.path.join(APP, "routes_trt_cache.py")
        with open(trt_router_path, "r", encoding="utf-8") as f:
            trt_source = f.read()
        for k in ["root", "status", "active_engines", "total_mb", "stale_mb", "namespaces"]:
            self.assertIn(f'"{k}"', trt_source)

    def test_autotune_route_schema_parity(self):
        """Verify routes_autotune.py exports expected AutotuneSnapshot keys."""
        auto_router_path = os.path.join(APP, "routes_autotune.py")
        with open(auto_router_path, "r", encoding="utf-8") as f:
            auto_source = f.read()
        for k in ["ready", "target", "available_frames", "defaults"]:
            self.assertIn(f'"{k}"', auto_source)

    def test_recognition_route_schema_parity(self):
        """Verify roop/ui_recognition.py current() exports RecognitionCurrent keys."""
        rec_impl_path = os.path.join(APP, "roop", "ui_recognition.py")
        with open(rec_impl_path, "r", encoding="utf-8") as f:
            rec_source = f.read()
        for k in ["selection", "advice", "providers", "active", "scope"]:
            self.assertIn(f'"{k}"', rec_source)

    def test_identity_blend_route_schema_parity(self):
        """Verify routes_identity.py exports expected IdentityBlendResponse keys."""
        id_router_path = os.path.join(APP, "routes_identity.py")
        with open(id_router_path, "r", encoding="utf-8") as f:
            id_source = f.read()
        for k in ["recipe", "active", "limits", "directions", "diagnostics"]:
            self.assertIn(f'"{k}"', id_source)

    def test_canonical_typescript_contract_file_sync(self):
        """Verify react-ui/src/types/api-contract.ts exists and exports canonical contracts."""
        self.assertTrue(os.path.isfile(self.contract_path), "api-contract.ts missing")
        expected_contracts = [
            "export interface SwapRecipe",
            "export interface DetectionSlice",
            "export interface IdentitySlice",
            "export interface MaskingSlice",
            "export interface RestorationSlice",
            "export interface TemporalSlice",
            "export interface OutputSlice",
            "export interface ViewportSlice",
            "export interface AppStateResponse",
            "export interface StageTelemetry",
            "export interface TrtCacheSnapshot",
            "export interface AutotuneSnapshot",
            "export interface RecognitionCurrent",
            "export interface BatchQueueResponse",
        ]
        for contract in expected_contracts:
            self.assertIn(contract, self.contract_source, f"api-contract.ts missing contract: {contract}")


if __name__ == "__main__":
    unittest.main()
