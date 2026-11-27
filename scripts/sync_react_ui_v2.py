"""Keep React UI 2.0 equal to React UI 1.0 plus a visual layer, and prove it.

React UI 2.0 (react-ui-v2/) is NOT a second client. It is 1.0's client (react-ui/), copied, with a
visual layer on top: src/v2fx/ (the 3D horizon, terminal dock, transitions, boot sequence, FX
panel, v2.css) and edits to a short list of 1.0 files that mount it. Every other file is 1.0's,
byte for byte, so every screen, control and endpoint call 1.0 has, 2.0 has.

That is enforced, because the first React UI 2.0 was a rewrite whose "complete parity" turned out
to be adapter wrappers no screen used, and it shipped unable to add a source
(app/tests/test_default_client_capability.py has the history).

    python scripts/sync_react_ui_v2.py            # copy 1.0 changes into 2.0, report owned files to merge
    python scripts/sync_react_ui_v2.py --check    # exit 1 if 2.0 has drifted from 1.0 (what the test runs)
    python scripts/sync_react_ui_v2.py --write-base   # after hand-merging an OWNED file: record 1.0's version

OWNED files are the ones 2.0 edits. For each, react-ui-v2/V1_BASE.json records the sha256 of the
1.0 file it was last merged from; when 1.0's file changes, the owned file is reported for a manual
merge (the copy step never overwrites 2.0's edits).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "react-ui")
V2 = os.path.join(ROOT, "react-ui-v2")
BASE_FILE = os.path.join(V2, "V1_BASE.json")

# 2.0 edits these (all paths relative to the client root, forward slashes).
OWNED = (
    "README.md",
    "index.html",
    "package.json",
    "package-lock.json",
    "vite.config.js",
    "src/App.jsx",
    "src/main.jsx",
    "src/themes.js",
    "src/components/Gallery.jsx",
    "src/components/ThemeGallery.jsx",
    "src/components/shell/WorkstationHeader.jsx",
    "src/components/shell/WorkstationRail.jsx",
)

# 2.0 only: never in 1.0.
V2_ONLY_PREFIXES = ("src/v2fx/", ".render-check/render-v2fx.mjs", "V1_BASE.json")

_SKIP_DIRS = {"node_modules", "dist", "test-results", "playwright-report", ".vite", "dist-ssr"}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _files(client_root, prefix):
    """Tracked files of a client (git), falling back to a walk without build output."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", prefix], cwd=ROOT, capture_output=True,
                             check=True, timeout=60).stdout.decode("utf-8")
        tracked = [p[len(prefix) + 1:] for p in out.split("\0") if p]
        # Untracked files the working tree has (a new 2.0 file not committed yet) count too.
        extra = subprocess.run(["git", "ls-files", "-z", "--others", "--exclude-standard", "--", prefix],
                               cwd=ROOT, capture_output=True, check=True, timeout=60).stdout.decode("utf-8")
        tracked += [p[len(prefix) + 1:] for p in extra.split("\0") if p]
        return sorted({p for p in tracked if os.path.isfile(os.path.join(client_root, p))})
    except (OSError, subprocess.SubprocessError):
        found = []
        for base, dirs, files in os.walk(client_root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                found.append(os.path.relpath(os.path.join(base, name), client_root).replace(os.sep, "/"))
        return sorted(found)


def _v2_only(rel):
    return any(rel == p or rel.startswith(p) for p in V2_ONLY_PREFIXES)


def load_base():
    try:
        with open(BASE_FILE, encoding="utf-8") as fh:
            return json.load(fh).get("owned", {})
    except (OSError, ValueError):
        return {}


def audit():
    """What separates 2.0 from "1.0 + its visual layer" right now."""
    v1 = _files(V1, "react-ui")
    v2 = set(_files(V2, "react-ui-v2"))
    base = load_base()
    report = {"stale_copy": [], "missing": [], "extra": [], "owned_needs_merge": [], "owned_unrecorded": []}
    for rel in v1:
        mine = os.path.join(V2, rel)
        theirs = os.path.join(V1, rel)
        if rel in OWNED:
            if not os.path.isfile(mine):
                report["missing"].append(rel)
            elif rel not in base:
                report["owned_unrecorded"].append(rel)
            elif base[rel] != _sha(theirs):
                report["owned_needs_merge"].append(rel)
            continue
        if not os.path.isfile(mine):
            report["missing"].append(rel)
        elif _sha(mine) != _sha(theirs):
            report["stale_copy"].append(rel)
    v1_set = set(v1)
    for rel in sorted(v2):
        if rel not in v1_set and not _v2_only(rel):
            report["extra"].append(rel)
    return report


def write_base():
    owned = {rel: _sha(os.path.join(V1, rel)) for rel in OWNED if os.path.isfile(os.path.join(V1, rel))}
    with open(BASE_FILE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"about": "sha256 of each 1.0 file a 2.0-owned file was last merged from "
                            "(scripts/sync_react_ui_v2.py)", "owned": owned}, fh, indent=2, sort_keys=True)
        fh.write("\n")


def sync():
    report = audit()
    for rel in report["stale_copy"] + [r for r in report["missing"] if r not in OWNED]:
        dst = os.path.join(V2, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(V1, rel), dst)
        print(f"copied   {rel}")
    for rel in report["extra"]:
        os.remove(os.path.join(V2, rel))
        print(f"removed  {rel}  (no longer in 1.0)")
    for rel in report["owned_needs_merge"]:
        print(f"MERGE    {rel}  (1.0 changed it; 2.0 owns it - merge by hand, then --write-base)")
    for rel in report["owned_unrecorded"]:
        print(f"RECORD   {rel}  (owned but no base hash; run --write-base after checking it)")
    return report


def main(argv):
    if "--write-base" in argv:
        write_base()
        print(f"recorded {len(OWNED)} owned files in {os.path.relpath(BASE_FILE, ROOT)}")
        return 0
    if "--check" in argv:
        report = audit()
        bad = {k: v for k, v in report.items() if v}
        if not bad:
            print("react-ui-v2 == react-ui + visual layer")
            return 0
        for k, v in bad.items():
            print(f"{k}: {', '.join(v)}")
        return 1
    report = sync()
    return 1 if (report["owned_needs_merge"] or report["owned_unrecorded"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
