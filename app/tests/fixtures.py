"""Locate the shared benchmark clips on whichever validation target is running.

WHY THIS EXISTS. Every harness here baked in a `G:/pinokio/roop-keep/...`
fixture path. That is one machine's drive layout -- the RTX 4070 workstation's.
On the physical RTX 3060 laptop `PINOKIO_HOME` is `C:\\pinokio` and there is no
`G:` drive at all, so a hardcoded fixture is not merely wrong, it is
unreachable.

That matters because the dual-GPU commands in `docs/HARDWARE_VALIDATION_MATRIX.md`
and `SESSION_HANDOFF.md` deliberately omit `--video`, on the stated promise that
"only the explicit report label changes" and "no configuration file rewrite is
required" when moving between the two targets. With a hardcoded root that
promise was false on the second target: every documented 3060 command would
have died on a missing file before measuring anything. The mandate is that the
application and its benchmarks work on both GPUs without rewriting
configuration, so the fixture root is resolved at runtime.

RESOLUTION ORDER

  1. ``$ROOP_CLIP_ROOT``     explicit operator override, wins outright
  2. ``<PINOKIO_HOME>/roop-keep`` then ``<PINOKIO_HOME>/roop keep``
  3. the same two names beside the launcher directory

`PINOKIO_HOME` itself is resolved the way the project guide requires: the
environment variable, then `~/.pinokio/config.json`'s `home`, then derived from
this file's own location. It is NOT assumed to be any particular drive.

BOTH SPELLINGS ARE REAL. The 4070 keeps `roop-keep` (hyphen); the 3060 keeps
`roop keep` (space). This is why `CLAUDE.md`'s instruction to read
`roop-keep/RECODE_STATUS.md` silently finds nothing on the laptop.

FLAT LAYOUTS ARE REAL TOO. The workstation stores clips in category folders
(`double/`, `single/`, `inverted/`); the laptop holds a flat copy of only the
clips it needs. So ``clip("double/d4.mp4")`` is tried as given and then by
basename anywhere in the root.

NOT FOUND IS NOT AN EXCEPTION HERE. When nothing resolves, the requested path
is returned unchanged so the caller fails exactly as it did before, with its own
message about its own missing file. A benchmark that cannot find its fixture
must report PENDING -- never substitute a different clip, because a fixture
swap silently invalidates a cross-target comparison.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.dirname(_HERE)

# <PINOKIO_HOME>/api/<launcher>/app -- three levels up from the app directory.
_DERIVED_HOME = os.path.dirname(os.path.dirname(os.path.dirname(_APP)))

# The 4070 uses a hyphen, the 3060 a space. Both are in active use.
_KEEP_NAMES = ("roop-keep", "roop keep")


def pinokio_home():
    """Resolve PINOKIO_HOME without assuming a drive letter."""
    env = os.environ.get("PINOKIO_HOME")
    if env and os.path.isdir(env):
        return env
    try:
        cfg = os.path.join(os.path.expanduser("~"), ".pinokio", "config.json")
        with open(cfg, "r", encoding="utf-8") as fh:
            home = json.load(fh).get("home")
        if home and os.path.isdir(home):
            return home
    except Exception:
        pass
    return _DERIVED_HOME


def clip_roots():
    """Candidate fixture roots, most explicit first, de-duplicated."""
    roots = []
    override = os.environ.get("ROOP_CLIP_ROOT")
    if override:
        roots.append(override)
    for base in (pinokio_home(), os.path.dirname(_APP), _DERIVED_HOME):
        for name in _KEEP_NAMES:
            roots.append(os.path.join(base, name))
    seen, out = set(), []
    for r in roots:
        key = os.path.normcase(os.path.abspath(r))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def clip(rel, required=False):
    """Resolve a fixture like ``"double/d4.mp4"`` on this machine.

    Tries the relative path as given, then the bare filename, in each candidate
    root. Returns `rel` unchanged when nothing matches so the caller's own
    "missing fixture" error still fires -- unless `required`, which raises with
    the full search list so the operator can see exactly where to put the clip.
    """
    if os.path.isabs(rel) and os.path.isfile(rel):
        return rel
    rel_norm = rel.replace("\\", "/")
    base = os.path.basename(rel_norm)
    roots = clip_roots()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for cand in (os.path.join(root, *rel_norm.split("/")),
                     os.path.join(root, base)):
            if os.path.isfile(cand):
                return cand
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if base in files:
                return os.path.join(dirpath, base)
    if required:
        raise SystemExit(
            "fixture %r not found on this machine.\nSearched:\n  %s\n"
            "Set ROOP_CLIP_ROOT to the folder holding the benchmark clips, or "
            "pass an explicit path. Do NOT substitute a different clip: a "
            "fixture change invalidates cross-target comparison."
            % (rel, "\n  ".join(roots)))
    return rel


def available(rel):
    """True when `rel` resolves to a real file here. Use to mark rows PENDING."""
    return os.path.isfile(clip(rel))


def fingerprint(path):
    """Identify a fixture by content shape, not by filename.

    THE BASENAME FALLBACK ABOVE IS NOT SAFE ON ITS OWN, and this was caught the
    first time it ran. The 4070 keeps `double/d4.mp4` at 1280x720; the 3060 held
    a *different* clip also called `d4.mp4` at 854x480 (the one the session logs
    call `duo/d4.mp4`). The resolver matched on name and produced a render that
    looked like a valid Phase 2 baseline while being a different workload
    entirely -- a smaller frame at a different face scale.

    A fixture swap does not fail loudly; it just makes two numbers
    incomparable, which is the most expensive kind of wrong in this project.
    So every harness that claims a cross-target comparison must fingerprint what
    it actually opened and record it beside the result.

    Returns {} when ffprobe is unavailable rather than blocking a run.
    """
    import json as _json
    import shutil
    import subprocess
    exe = shutil.which("ffprobe")
    if not exe:
        for root in (os.path.join(pinokio_home(), "bin", "miniforge", "Library", "bin"),
                     os.path.join(pinokio_home(), "bin", "miniconda", "Library", "bin")):
            cand = os.path.join(root, "ffprobe.exe")
            if os.path.isfile(cand):
                exe = cand
                break
    if not exe or not os.path.isfile(path):
        return {}
    try:
        out = subprocess.check_output(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_frames,r_frame_rate,codec_name",
             "-of", "json", path], text=True, timeout=30)
        st = (_json.loads(out).get("streams") or [{}])[0]
        return {
            "path": path,
            "bytes": os.path.getsize(path),
            "width": int(st.get("width") or 0),
            "height": int(st.get("height") or 0),
            "frames": int(st.get("nb_frames") or 0),
            "fps": st.get("r_frame_rate"),
            "codec": st.get("codec_name"),
        }
    except Exception:
        return {}


def matches(fp, expect):
    """Which expected fixture fields disagree with the resolved file."""
    return sorted(k for k, v in (expect or {}).items()
                  if fp.get(k) is not None and fp.get(k) != v)
