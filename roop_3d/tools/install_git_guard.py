"""Install a Git hook include scoped only to roop3d/* branches in this clone."""
from __future__ import annotations

from pathlib import Path
import subprocess


def install(root: Path) -> None:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    if not branch.startswith("roop3d/"):
        raise RuntimeError("Switch to an isolated roop3d/* branch before Roop 3D development")
    subprocess.run(["git", "config", "--local", "--replace-all", "includeIf.onbranch:roop3d/**.path",
                    "../roop_3d/gitconfig"], cwd=root, check=True)


def main() -> int:
    try:
        install(Path(__file__).absolute().parents[2])
        print("Roop 3D branch-specific Git guards enabled; legacy branch hooks unchanged", flush=True)
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ROOP3D_GUARD_ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
