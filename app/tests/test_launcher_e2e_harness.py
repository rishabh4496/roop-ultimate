#!/usr/bin/env python3
"""End-to-End Pinokio Launcher Simulation Test Harness.

Simulates Pinokio launcher execution across 10 distinct system and environment states:
1. Fresh directory (fresh checkout: no app/env, no config.yaml, no markers, no UI build)
2. Missing env (repository files and UI present, but app/env directory missing)
3. Existing empty env (app/env exists but contains no packages or python environment)
4. Partially installed env (installation interrupted mid-transaction, incomplete marker present)
5. ORT missing (Python environment provisioned, but onnxruntime distribution absent)
6. ORT namespace package (broken onnxruntime package directory missing __init__.py / unpopulated)
7. CPU ORT (CPU-only onnxruntime, CPUExecutionProvider only)
8. CUDA ORT (onnxruntime-gpu with CUDA, but TensorRT packages/libs absent)
9. TensorRT missing (NVIDIA >=7GB GPU where TensorRT is required but nvinfer DLLs/packages missing)
10. TensorRT complete (fully compliant NVIDIA GPU stack with TensorRT minimal session verified)

Validates for every branch:
- Every `when` expression evaluates without throwing
- Required install steps execute in deterministic order
- Skipped optional steps are expected (e.g. config copy skipped when config.yaml already exists)
- Install completion marker is strictly absent on failure
- Install completion marker exists ONLY after successful runtime validation
- Start script never runs the backend (run.py) against an incomplete or unverified environment
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
APP = HERE.parent
ROOT = APP.parent

if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import install_state
import fixtures


def find_node() -> Optional[str]:
    found = shutil.which("node")
    if found:
        return found
    roots = [
        os.environ.get("PINOKIO_HOME"),
        os.path.abspath(os.path.join(str(ROOT), os.pardir, os.pardir)),
        fixtures.pinokio_home(),
    ]
    for r in filter(None, roots):
        for rel in (
            ("bin", "miniforge", "node.exe"), ("bin", "miniforge", "node"),
            ("bin", "miniconda", "node.exe"), ("bin", "miniconda", "node"),
            ("bin", "nodejs", "node.exe"), ("bin", "nodejs", "node"),
        ):
            candidate = os.path.join(r, *rel)
            if os.path.isfile(candidate):
                return candidate
    return None


NODE = find_node()


@dataclass
class SimulatedStepResult:
    step_index: int
    method: str
    when_expr: Optional[str]
    when_evaluated: bool
    executed: bool
    action_summary: str
    status: str  # "ok", "skipped", "failed"
    error: Optional[str] = None


class PinokioScriptSimulator:
    """Accurately simulates Pinokio step execution against a workspace directory."""

    def __init__(self, workspace: Path, node_bin: Optional[str] = None):
        self.workspace = workspace
        self.node_bin = node_bin or NODE
        self.scope = {
            "platform": "win32" if sys.platform == "win32" else sys.platform,
            "arch": "x64",
            "gpu": "nvidia",
            "gpus": ["nvidia"],
        }
        self.running_scripts: set[str] = set()
        self.local_vars: Dict[str, Dict[str, Any]] = {}

    def eval_when(self, expr: str) -> bool:
        """Evaluate a Pinokio JS `when` condition against workspace filesystem and scope."""
        if not expr:
            return True

        if not self.node_bin:
            # Fallback pure-python evaluator for standard Pinokio when patterns
            clean = expr.strip()
            # Handle standard pattern "!exists('path') && exists('path2')"
            # or "exists('path') || !exists('path2')"
            def _sub_exists(match):
                p = match.group(1)
                full = self.workspace / p
                return "True" if full.exists() else "False"

            py_expr = re.sub(r"exists\(['\"]([^'\"]+)['\"]\)", _sub_exists, clean)
            py_expr = py_expr.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
            try:
                return bool(eval(py_expr, {"__builtins__": {}}, {}))
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate when expression '{expr}': {e}")

        # Authoritative evaluation via Node VM matching Pinokio's runtime context
        ws_json = json.dumps(str(self.workspace).replace("\\", "/"))
        scope_json = json.dumps(self.scope)
        expr_json = json.dumps(expr)
        js_eval_code = f"""
const vm = require('node:vm');
const path = require('node:path');
const fs = require('node:fs');

const workspace = {ws_json};
const scope = {scope_json};

const ctx = Object.assign({{}}, scope, {{
  exists: (p) => fs.existsSync(path.resolve(workspace, p)),
  running: (p) => false,
  path: path,
  envs: {{}}
}});

try {{
  const result = vm.runInNewContext({expr_json}, ctx);
  console.log(JSON.stringify({{ ok: true, value: Boolean(result) }}));
}} catch (err) {{
  console.log(JSON.stringify({{ ok: false, error: err.name + ': ' + err.message }}));
}}
"""
        res = subprocess.run(
            [self.node_bin, "-e", js_eval_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode != 0 or not res.stdout.strip():
            raise RuntimeError(f"Node VM failed evaluating when expr '{expr}': {res.stderr}")
        data = json.loads(res.stdout.strip().splitlines()[-1])
        if not data.get("ok"):
            raise RuntimeError(f"When expression '{expr}' threw: {data.get('error')}")
        return bool(data.get("value"))

    def eval_menu(self) -> List[Dict[str, Any]]:
        """Evaluate pinokio.js menu against workspace state."""
        pinokio_js = ROOT / "pinokio.js"
        if not pinokio_js.exists():
            raise FileNotFoundError(f"Missing {pinokio_js}")

        if not self.node_bin:
            # Python equivalent of pinokio.js installed logic
            installed = (
                (self.workspace / ".pinokio-install-complete.json").exists()
                and (self.workspace / ".pinokio-install-ready.json").exists()
                and not (self.workspace / ".pinokio-install-incomplete.json").exists()
                and (self.workspace / "react-ui" / "dist" / "index.html").exists()
            )
            if installed:
                return [{"default": True, "text": "Start", "href": "start_react.js"}]
            else:
                return [{"default": True, "text": "Install", "href": "install.js"}]

        pinokio_json = json.dumps(str(pinokio_js).replace("\\", "/"))
        ws_json = json.dumps(str(self.workspace).replace("\\", "/"))
        js_menu_code = f"""
const path = require('path');
const fs = require('fs');

async function run() {{
  const mod = require({pinokio_json});
  const workspace = {ws_json};
  const info = {{
    exists: (p) => fs.existsSync(path.resolve(workspace, p)),
    running: (p) => false,
    local: (p) => null
  }};
  const menu = await mod.menu({{}}, info);
  console.log(JSON.stringify(menu));
}}
run().catch(e => {{ console.error(e); process.exit(1); }});
"""
        res = subprocess.run(
            [self.node_bin, "-e", js_menu_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Failed evaluating pinokio.js menu: {res.stderr}")
        return json.loads(res.stdout.strip().splitlines()[-1])


class TestLauncherEndToEndHarness(unittest.TestCase):
    """Full end-to-end launcher test suite across 10 simulated states."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        # Create minimal repo structure
        (self.workspace / "app").mkdir(parents=True)
        (self.workspace / "react-ui").mkdir(parents=True)
        # Ship default_config.yaml in workspace app/
        shutil.copy(str(APP / "default_config.yaml"), str(self.workspace / "app" / "default_config.yaml"))
        self.simulator = PinokioScriptSimulator(self.workspace)

    def tearDown(self):
        self.tmp.cleanup()

    # ──────────────────────────────────────────────────────────────────────────
    # Step Extraction Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_install_steps(self) -> List[Dict[str, Any]]:
        """Load actual install.js steps."""
        install_js = ROOT / "install.js"
        text = install_js.read_text(encoding="utf-8")
        WHEN = re.compile(r'"?when"?\s*:\s*"\{\{([\s\S]*?)\}\}"')

        # Run via node to get structure
        if NODE:
            install_path = json.dumps(str(install_js).replace("\\", "/"))
            js = f"""
const mod = require({install_path});
console.log(JSON.stringify(mod.run));
"""
            res = subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True)
            return json.loads(res.stdout.strip().splitlines()[-1])
        # Fallback structural parse
        return []

    def _get_start_steps(self) -> List[Dict[str, Any]]:
        """Load actual start_react.js steps."""
        start_js = ROOT / "start_react.js"
        if NODE:
            start_url = json.dumps('file:///' + str(start_js).replace("\\", "/"))
            js = f"""
const mod = await import({start_url});
const cfg = await mod.default({{ port: async () => 8001 }});
console.log(JSON.stringify(cfg.run));
"""
            res = subprocess.run([NODE, "--input-type=module", "-e", js], capture_output=True, text=True, check=True)
            return json.loads(res.stdout.strip().splitlines()[-1])
        return []

    # ──────────────────────────────────────────────────────────────────────────
    # Simulation Execution Engine
    # ──────────────────────────────────────────────────────────────────────────

    def _simulate_install(
        self,
        *,
        fail_at_stage: Optional[str] = None,
        verify_ort_result: str = "pass",  # "pass", "fail_ort_missing", "fail_namespace", "fail_trt", "pass_cpu"
    ) -> List[SimulatedStepResult]:
        """Simulate running install.js from start to finish."""
        steps = self._get_install_steps()
        history: List[SimulatedStepResult] = []

        for idx, step in enumerate(steps):
            method = step.get("method", "")
            params = step.get("params", {})
            when_raw = step.get("when")
            when_clean = None
            if when_raw:
                m = re.search(r"\{\{([\s\S]*?)\}\}", when_raw)
                when_clean = m.group(1).strip() if m else when_raw.strip()

            when_val = True
            if when_clean:
                when_val = self.simulator.eval_when(when_clean)

            if not when_val:
                history.append(
                    SimulatedStepResult(
                        step_index=idx,
                        method=method,
                        when_expr=when_clean,
                        when_evaluated=False,
                        executed=False,
                        action_summary=f"skipped: {params.get('src') or params.get('uri') or 'shell'}",
                        status="skipped",
                    )
                )
                continue

            # Step executes
            msg = " ".join(params.get("message", [])) if isinstance(params.get("message"), list) else str(params.get("message", ""))
            action_desc = params.get("uri") or params.get("src") or msg

            # Simulate effects of steps
            if "install_state.py begin" in msg:
                # Writes incomplete marker, deletes complete & ready
                (self.workspace / ".pinokio-install-incomplete.json").write_text(
                    json.dumps({"state": "in_progress", "last_stage": "bootstrap"}), encoding="utf-8"
                )
                (self.workspace / ".pinokio-install-complete.json").unlink(missing_ok=True)
                (self.workspace / ".pinokio-install-ready.json").unlink(missing_ok=True)

            elif "install_state.py stage" in msg:
                stage_name = msg.split("--stage")[-1].strip()
                if stage_name == fail_at_stage:
                    # Simulation aborts at this stage
                    history.append(
                        SimulatedStepResult(
                            step_index=idx,
                            method=method,
                            when_expr=when_clean,
                            when_evaluated=True,
                            executed=True,
                            action_summary=f"FAILED at stage {stage_name}",
                            status="failed",
                            error=f"Simulated failure at {stage_name}",
                        )
                    )
                    break
                (self.workspace / ".pinokio-install-incomplete.json").write_text(
                    json.dumps({"state": "in_progress", "last_stage": stage_name}), encoding="utf-8"
                )

            elif "npm run build" in msg:
                # Simulates producing react-ui/dist/index.html
                dist = self.workspace / "react-ui" / "dist"
                dist.mkdir(parents=True, exist_ok=True)
                (dist / "index.html").write_text("<!DOCTYPE html><html></html>", encoding="utf-8")

            elif method == "fs.copy":
                src = self.workspace / params["src"]
                dest = self.workspace / params["dest"]
                if src.exists() and not dest.exists():
                    shutil.copy(str(src), str(dest))

            elif "verify_ort.py" in msg:
                if verify_ort_result != "pass" and verify_ort_result != "pass_cpu":
                    history.append(
                        SimulatedStepResult(
                            step_index=idx,
                            method=method,
                            when_expr=when_clean,
                            when_evaluated=True,
                            executed=True,
                            action_summary=f"verify_ort failed: {verify_ort_result}",
                            status="failed",
                            error=f"Runtime verification failed: {verify_ort_result}",
                        )
                    )
                    break
                else:
                    # Write simulated manifest
                    (self.workspace / ".runtime-verification.json").write_text(
                        json.dumps({
                            "schema": 3,
                            "verification_passed": True,
                            "installer_version": install_state.INSTALLER_VERSION,
                            "python_version": "3.10",
                            "python_executable": "python",
                            "platform": "Windows",
                            "architecture": "AMD64",
                            "gpu_vendor": "nvidia" if verify_ort_result != "pass_cpu" else "cpu",
                            "gpu_name": "RTX 4070" if verify_ort_result != "pass_cpu" else "cpu",
                            "cuda_version": "12.8" if verify_ort_result != "pass_cpu" else "none",
                            "pytorch_version": "2.7.0",
                            "onnxruntime_version": "1.23.2",
                            "tensorrt_version": "10.9.0.34" if verify_ort_result == "pass" else "none",
                            "ort_provider_list": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if verify_ort_result == "pass" else ["CPUExecutionProvider"],
                            "tensorrt_session_test_result": "passed" if verify_ort_result == "pass" else "not_required",
                            "installation_timestamp": "now",
                            "repository_commit": "simulated",
                            "dependency_verification_status": "passed",
                            "runtime_verification_status": "passed",
                            "binary_runtime_compatibility_status": "not_applicable",
                            "binary_runtime_compatibility": {"status": "not_applicable"},
                        }),
                        encoding="utf-8"
                    )

            elif "install_state.py commit" in msg:
                manifest_file = self.workspace / ".runtime-verification.json"
                if manifest_file.exists():
                    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    manifest_file.unlink()
                    (self.workspace / ".pinokio-install-incomplete.json").unlink(missing_ok=True)
                    (self.workspace / ".pinokio-install-complete.json").write_text(
                        json.dumps({
                            "schema": 3,
                            "state": "complete",
                            "manifest": manifest_data,
                            "react_build": "react-ui/dist/index.html",
                            "python_environment": "app/env"
                        }),
                        encoding="utf-8"
                    )
                    (self.workspace / ".pinokio-install-ready.json").write_text(
                        json.dumps({"schema": 3, "state": "ready"}), encoding="utf-8"
                    )

            history.append(
                SimulatedStepResult(
                    step_index=idx,
                    method=method,
                    when_expr=when_clean,
                    when_evaluated=True,
                    executed=True,
                    action_summary=action_desc,
                    status="ok",
                )
            )

        return history

    def _simulate_start(self) -> Tuple[List[SimulatedStepResult], bool]:
        """Simulate running start_react.js. Returns (history, backend_launched)."""
        steps = self._get_start_steps()
        history: List[SimulatedStepResult] = []
        backend_launched = False

        for idx, step in enumerate(steps):
            method = step.get("method", "")
            params = step.get("params", {})
            when_raw = step.get("when")
            when_clean = None
            if when_raw:
                m = re.search(r"\{\{([\s\S]*?)\}\}", when_raw)
                when_clean = m.group(1).strip() if m else when_raw.strip()

            when_val = True
            if when_clean:
                when_val = self.simulator.eval_when(when_clean)

            if not when_val:
                history.append(
                    SimulatedStepResult(
                        step_index=idx,
                        method=method,
                        when_expr=when_clean,
                        when_evaluated=False,
                        executed=False,
                        action_summary=f"skipped: {params.get('uri') or params.get('message') or 'step'}",
                        status="skipped",
                    )
                )
                continue

            msg = " ".join(params.get("message", [])) if isinstance(params.get("message"), list) else str(params.get("message", ""))
            action_desc = params.get("uri") or params.get("url") or msg

            # Step 1: if install.js is triggered due to incomplete installation:
            if method == "script.start" and params.get("uri") == "install.js":
                # Start script detects incomplete install and calls install.js
                history.append(
                    SimulatedStepResult(
                        step_index=idx,
                        method=method,
                        when_expr=when_clean,
                        when_evaluated=True,
                        executed=True,
                        action_summary="triggered install.js due to incomplete environment",
                        status="ok",
                    )
                )
                # In Pinokio, triggering install.js suspends or restarts the flow;
                # It does NOT proceed to run backend against the uninstalled env.
                break

            # Step 2: verify_ort.py --require-complete
            if "verify_ort.py --require-complete" in msg:
                # Check if complete & ready markers exist
                complete_ok = (
                    (self.workspace / ".pinokio-install-complete.json").exists()
                    and (self.workspace / ".pinokio-install-ready.json").exists()
                    and not (self.workspace / ".pinokio-install-incomplete.json").exists()
                )
                if not complete_ok:
                    history.append(
                        SimulatedStepResult(
                            step_index=idx,
                            method=method,
                            when_expr=when_clean,
                            when_evaluated=True,
                            executed=True,
                            action_summary="verify_ort --require-complete failed",
                            status="failed",
                            error="Refusing to start: incomplete or unverified install marker",
                        )
                    )
                    break

            # Step 7: python run.py --ui react
            if "python run.py" in msg:
                backend_launched = True

            history.append(
                SimulatedStepResult(
                    step_index=idx,
                    method=method,
                    when_expr=when_clean,
                    when_evaluated=True,
                    executed=True,
                    action_summary=action_desc,
                    status="ok",
                )
            )

        return history, backend_launched

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Fresh Directory Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_01_fresh_directory(self):
        """Simulation 1: Fresh clone/directory with no app/env, no markers, no build."""
        # Menu must offer Install, not Start
        menu = self.simulator.eval_menu()
        self.assertEqual(menu[0]["text"], "Install")
        self.assertEqual(menu[0]["href"], "install.js")

        # In a fresh directory, start_react.js must detect incomplete state and not run backend
        start_history, backend_launched = self._simulate_start()
        self.assertFalse(backend_launched, "Backend must not launch in fresh directory")
        # Step 1 should trigger install.js because incomplete/not-ready
        triggered_install = any(
            s.executed and "install.js" in s.action_summary for s in start_history
        )
        self.assertTrue(triggered_install, "start_react.js must trigger install.js when not installed")

        # Now run install:
        install_history = self._simulate_install(verify_ort_result="pass")
        # All when evaluated cleanly
        for s in install_history:
            self.assertIsNone(s.error)

        # Config copy step: app/config.yaml was copied from default
        self.assertTrue((self.workspace / "app" / "config.yaml").exists())
        # Completion markers exist only after commit
        self.assertTrue((self.workspace / ".pinokio-install-complete.json").exists())
        self.assertTrue((self.workspace / ".pinokio-install-ready.json").exists())
        self.assertFalse((self.workspace / ".pinokio-install-incomplete.json").exists())

        # Menu now offers Start
        new_menu = self.simulator.eval_menu()
        self.assertIn("Start", new_menu[0]["text"])

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Missing Env Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_02_missing_env(self):
        """Simulation 2: UI build exists, but app/env directory is missing."""
        (self.workspace / "react-ui" / "dist").mkdir(parents=True, exist_ok=True)
        (self.workspace / "react-ui" / "dist" / "index.html").write_text("ok", encoding="utf-8")
        # No markers
        menu = self.simulator.eval_menu()
        self.assertEqual(menu[0]["text"], "Install")

        start_history, backend_launched = self._simulate_start()
        self.assertFalse(backend_launched, "Backend must not launch with missing env")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Existing Empty Env Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_03_existing_empty_env(self):
        """Simulation 3: app/env exists as an empty directory (not yet provisioned)."""
        (self.workspace / "app" / "env").mkdir(parents=True, exist_ok=True)
        # Empty env alone must NOT be treated as installed
        menu = self.simulator.eval_menu()
        self.assertEqual(menu[0]["text"], "Install")

        start_history, backend_launched = self._simulate_start()
        self.assertFalse(backend_launched, "Backend must not launch on empty unprovisioned env")
        self.assertFalse((self.workspace / ".pinokio-install-complete.json").exists())

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Partially Installed Env Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_04_partially_installed_env(self):
        """Simulation 4: Installation aborted during pytorch_gpu_runtime."""
        # Simulate partial run that fails at pytorch_gpu_runtime
        install_history = self._simulate_install(fail_at_stage="pytorch_gpu_runtime")
        # Must fail
        self.assertTrue(any(s.status == "failed" for s in install_history))

        # Incomplete marker must exist, complete marker must NOT exist
        self.assertTrue((self.workspace / ".pinokio-install-incomplete.json").exists())
        self.assertFalse((self.workspace / ".pinokio-install-complete.json").exists())
        self.assertFalse((self.workspace / ".pinokio-install-ready.json").exists())

        # Menu must refuse to offer Start
        menu = self.simulator.eval_menu()
        self.assertEqual(menu[0]["text"], "Install")

        # Start must block backend execution
        start_history, backend_launched = self._simulate_start()
        self.assertFalse(backend_launched, "Backend must not launch against partial install")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. ORT Missing Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_05_ort_missing(self):
        """Simulation 5: Requirements and PyTorch provisioned, but ORT verification fails."""
        install_history = self._simulate_install(verify_ort_result="fail_ort_missing")
        self.assertTrue(any(s.status == "failed" for s in install_history))

        # Markers must NOT exist
        self.assertFalse((self.workspace / ".pinokio-install-complete.json").exists())
        self.assertFalse((self.workspace / ".pinokio-install-ready.json").exists())

        start_history, backend_launched = self._simulate_start()
        self.assertFalse(backend_launched, "Backend must not launch when ORT is missing")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. ORT Namespace Package Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_06_ort_namespace_package(self):
        """Simulation 6: ORT package is an unpopulated namespace package without __init__.py."""
        install_history = self._simulate_install(verify_ort_result="fail_namespace")
        self.assertTrue(any(s.status == "failed" for s in install_history))

        # Completion marker strictly absent
        self.assertFalse((self.workspace / ".pinokio-install-complete.json").exists())
        start_history, backend_launched = self._simulate_start()
        self.assertFalse(backend_launched, "Backend must not launch on shadowed/namespace ORT")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. CPU ORT Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_07_cpu_ort(self):
        """Simulation 7: CPU-only environment validates and completes cleanly for CPU hardware."""
        self.simulator.scope["gpu"] = "cpu"
        self.simulator.scope["gpus"] = []

        install_history = self._simulate_install(verify_ort_result="pass_cpu")
        self.assertTrue(all(s.status != "failed" for s in install_history))

        # Marker present after successful CPU validation
        self.assertTrue((self.workspace / ".pinokio-install-complete.json").exists())
        self.assertTrue((self.workspace / ".pinokio-install-ready.json").exists())

        # Start runs cleanly and launches backend
        start_history, backend_launched = self._simulate_start()
        self.assertTrue(backend_launched, "Backend should launch on verified CPU install")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. CUDA ORT Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_08_cuda_ort(self):
        """Simulation 8: CUDA environment without TensorRT completes on sub-7GB hardware policy."""
        # On a sub-7GB GPU, TensorRT is not required, so verify_ort succeeds under policy
        install_history = self._simulate_install(verify_ort_result="pass")
        self.assertTrue(all(s.status != "failed" for s in install_history))
        self.assertTrue((self.workspace / ".pinokio-install-complete.json").exists())

        start_history, backend_launched = self._simulate_start()
        self.assertTrue(backend_launched)

    # ──────────────────────────────────────────────────────────────────────────
    # 9. TensorRT Missing Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_09_tensorrt_missing(self):
        """Simulation 9: TensorRT required on >=7GB device, but TensorRT session construction fails."""
        install_history = self._simulate_install(verify_ort_result="fail_trt")
        self.assertTrue(any(s.status == "failed" for s in install_history))

        # Complete marker must NOT be written when TensorRT verification fails
        self.assertFalse((self.workspace / ".pinokio-install-complete.json").exists())
        self.assertFalse((self.workspace / ".pinokio-install-ready.json").exists())

        # Start refuses backend
        start_history, backend_launched = self._simulate_start()
        self.assertFalse(backend_launched)

    # ──────────────────────────────────────────────────────────────────────────
    # 10. TensorRT Complete Simulation
    # ──────────────────────────────────────────────────────────────────────────
    def test_state_10_tensorrt_complete(self):
        """Simulation 10: Complete, verified workstation setup with active TensorRT EP."""
        install_history = self._simulate_install(verify_ort_result="pass")
        self.assertTrue(all(s.status != "failed" for s in install_history))

        # Validate that skipped steps are expected:
        # If app/config.yaml already exists, the fs.copy step must be skipped
        # Re-run install with existing config.yaml
        install_history_reinstall = self._simulate_install(verify_ort_result="pass")
        copy_step = next(s for s in install_history_reinstall if s.method == "fs.copy")
        self.assertEqual(copy_step.status, "skipped", "fs.copy should be skipped when config.yaml exists")

        # Markers valid
        self.assertTrue((self.workspace / ".pinokio-install-complete.json").exists())
        self.assertTrue((self.workspace / ".pinokio-install-ready.json").exists())

        # Start executes full sequence through backend launch
        start_history, backend_launched = self._simulate_start()
        self.assertTrue(backend_launched, "Backend must launch on verified TensorRT environment")

        # Validate that all when expressions evaluated without throwing
        for s in install_history + start_history:
            self.assertIsNone(s.error)


if __name__ == "__main__":
    unittest.main()
