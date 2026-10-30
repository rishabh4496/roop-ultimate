from pathlib import Path
import subprocess

import pytest
from roop_3d.tools.git_guard import BoundaryError, allowed_path, check_staged
from roop_3d.tools.install_git_guard import install


def test_boundary_accepts_source_and_rejects_legacy_generated_and_secrets() -> None:
    for path in ("roop_3d/server/app.py", "roop_3d_ui/src/main.tsx", "start_roop3d.js"):
        assert allowed_path(path)
    for path in ("app/main.py", "face_engine/models.py", "react-ui/src/App.jsx", "pinokio.js", ".gitignore",
                 "roop_3d/.venv/x", "roop_3d/.env", "roop_3d/weights.onnx", "roop_3d/x.engine",
                 "roop_3d/../app/x", "roop_3d_ui/node_modules/x", "roop_3d/.runtime/x"):
        assert not allowed_path(path), path


def test_guard_reads_real_git_index_and_rejects_legacy_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "roop_3d").mkdir()
    (tmp_path / "roop_3d/service.py").write_bytes(b"source")
    subprocess.run(["git", "add", "roop_3d/service.py"], cwd=tmp_path, check=True)
    assert check_staged(tmp_path) == ["roop_3d/service.py"]
    (tmp_path / "app").mkdir()
    (tmp_path / "app/legacy.py").write_bytes(b"protected")
    subprocess.run(["git", "add", "app/legacy.py"], cwd=tmp_path, check=True)
    with pytest.raises(BoundaryError, match="legacy.py"):
        check_staged(tmp_path)


def test_hooks_are_conditioned_on_roop3d_branch_only(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "switch", "-c", "roop3d/test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "roop_3d").mkdir()
    (tmp_path / "roop_3d/gitconfig").write_text("[core]\n hooksPath = roop_3d/.githooks\n", encoding="utf-8")
    install(tmp_path)
    result = subprocess.check_output(["git", "config", "--get", "core.hooksPath"], cwd=tmp_path, text=True)
    assert result.strip() == "roop_3d/.githooks"
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=tmp_path, check=True)
    result = subprocess.run(["git", "config", "--get", "core.hooksPath"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    with pytest.raises(RuntimeError, match="isolated"):
        install(tmp_path)
