# Isolated Roop 3D development

Work on `roop3d/*` branches. The current branch was created from
`8ffc2f95bcfd918f37035ca3ab95d79b75504973`; changes to legacy source are forbidden.
Do not merge this development branch into Roop Ultimate's `main` as part of a stage.

Only these versioned paths belong to Roop 3D:

- `roop_3d/**`
- `roop_3d_ui/**`
- `start_roop3d.js`

`pinokio.js`, `app/`, `face_engine/`, `react-ui/`, the root ignore file, root lockfiles,
and existing launchers remain protected. Read existing weights directly through the
verified local registry. Do not import the legacy backend or reuse its environment.

From the repository root on Windows:

```powershell
# Installs a conditional Git include in THIS clone, for roop3d/* branches only:
& ./roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.install_git_guard

# Full Roop 3D suite; all temporary and cache files stay inside roop_3d:
& ./roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q

# Strict immutable legacy hashes and additions, with runtime activity reported separately:
& ./roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report

# Stage explicit paths, then inspect the boundary BEFORE committing:
git add -- roop_3d roop_3d_ui start_roop3d.js
& ./roop_3d/.venv/Scripts/python.exe -B roop_3d/tools/git_guard.py --staged --branch
git diff --cached --stat
git commit -m "Describe the Roop 3D stage"
git push
```

On Unix, use `.venv/bin/python`. A new clone needs the conditional Git include
installed once; the launcher installs it automatically. The scoped pre-commit hook
rejects legacy files, secrets, weights, caches, environments and symlinks. The
pre-push hook rejects a branch containing legacy source changes and refuses targets
outside `refs/heads/roop3d/*`. It does not run the legacy hook that rewrites
`update_manifest.json`. The legacy branches retain their existing hook configuration.

Hooks protect commits and pushes; they are not an OS filesystem sandbox. All new
filesystem-writing code must use the boundary checks in `audit/isolation.py`.
Never bypass hooks or use broad staging commands during Roop 3D development.

The original audit manifests remain unchanged. `audit/runtime_policy.py` explicitly
distinguishes legacy output, project state, temporary uploads, logs, three named
history/calibration files, and runtime-profile **JSON** from immutable source,
configuration, dependencies and weights. Runtime activity is reported, never claimed
byte-identical. This distinction lets Roop Ultimate keep rendering independently;
it never authorizes Roop 3D to write those paths or commit them.

The machine-local comprehensive snapshot and detailed runtime report are ignored.
Do not commit user media paths, FaceSets, embeddings, credentials or generated data.
The other GPU remains unverified until the same stage gates run there.
