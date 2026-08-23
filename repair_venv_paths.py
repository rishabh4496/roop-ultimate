"""Repair a virtual environment that has been MOVED to a new folder.

A venv records its own absolute location in several places. Python itself does
not care — `sys.prefix` is derived from the interpreter's location and
`pyvenv.cfg` — so `env/Scripts/python.exe script.py` keeps working after a move
and everything looks healthy. What breaks is everything that goes through the
RECORDED path:

    Scripts/activate, activate.bat     hardcode VIRTUAL_ENV and prepend it to
                                       PATH. After a move they point at a folder
                                       that may not exist, so `python` silently
                                       resolves to whatever else is on PATH —
                                       typically a base conda install with none
                                       of the packages. The symptom is an
                                       ImportError for a package that is plainly
                                       installed.
    Scripts/*.exe                      pip, uvicorn, torchrun and friends are a
                                       launcher stub + a `#!<python>` shebang +
                                       a zip. With a stale shebang they exit 1
                                       and print nothing at all.
    Scripts/<script> (no extension)    the same, as a plain text shebang.

This happened here on 2026-08-23: `app/env` was moved out of another working
copy and that copy was then deleted, so `activate` pointed into thin air and the
app died at `import torch` with the venv sitting right there, intact.

Usage — from the repository root, with any Python:

    python repair_venv_paths.py                 # repair app/env
    python repair_venv_paths.py --dry-run       # show what would change
    python repair_venv_paths.py --venv other/env
    python repair_venv_paths.py --old "D:/old/env"    # if detection misses one

Stale paths are discovered from the venv itself — from `activate` AND,
separately, from the shebangs inside `Scripts/*.exe`. Those two can disagree:
repairing `activate` by hand and stopping there leaves every console script
broken while the venv reports itself healthy. That is exactly what happened
here, which is why detection reads both.
"""

import argparse
import os
import sys

DEFAULT_VENV = os.path.join('app', 'env')
ZIP_MAGIC = b'PK' + bytes([3, 4])
SEP = chr(92)


def _from_activate(venv):
    for parts in (('Scripts', 'activate.bat'), ('bin', 'activate')):
        bat = os.path.join(venv, *parts)
        if not os.path.exists(bat):
            continue
        with open(bat, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'VIRTUAL_ENV=' in line:
                    return line.split('VIRTUAL_ENV=', 1)[1].strip().strip('"\'')
    return None


def _venv_root_from_shebang(line):
    """`...<root>\\Scripts\\python.exe` -> `<root>`."""
    low = line.lower()
    for marker in (SEP + 'scripts' + SEP, '/scripts/', SEP + 'bin' + SEP, '/bin/'):
        i = low.find(marker)
        if i > 0:
            return line[:i]
    return None


def _from_launchers(venv):
    found = set()
    for sub in ('Scripts', 'bin'):
        d = os.path.join(venv, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith('.exe'):
                continue
            try:
                with open(os.path.join(d, fn), 'rb') as f:
                    data = f.read()
            except OSError:
                continue
            z = data.find(ZIP_MAGIC)
            if z <= 0:
                continue
            k = data.rfind(b'#!', 0, z)
            if k < 0:
                continue
            line = data[k + 2:z].split(b'\n')[0].decode('utf-8', 'ignore').strip()
            root = _venv_root_from_shebang(line)
            if root:
                found.add(root)
    return found


def recorded_paths(venv):
    """Every location this venv still believes it lives at, minus the real one."""
    out = set()
    a = _from_activate(venv)
    if a:
        out.add(a)
    out |= _from_launchers(venv)
    real = os.path.normpath(venv).lower()
    return sorted({os.path.normpath(p) for p in out
                   if os.path.normpath(p).lower() != real})


def _is_text(path):
    try:
        with open(path, 'rb') as f:
            return b'\0' not in f.read(4096)
    except OSError:
        return False


def _variants(p):
    return (p, p.replace(SEP, '/'))


def repair_text(path, old, new, dry):
    with open(path, 'rb') as f:
        data = f.read()
    out = data
    for a, b in zip(_variants(old), _variants(new)):
        out = out.replace(a.encode(), b.encode())
    if out == data:
        return False
    if not dry:
        with open(path, 'wb') as f:
            f.write(out)
    return True


def repair_launcher(path, old, new, dry):
    """Rebuild a Scripts/*.exe as stub + new shebang + unchanged zip payload.

    That is how these are assembled in the first place, so the shebang may
    change length freely.

    The shebang is located by anchoring on the STALE PATH itself, not by
    searching for the zip magic. Searching for `PK\\x03\\x04` finds the FIRST
    occurrence, and those four bytes can appear inside the launcher stub by
    coincidence — which would put the split point before the real shebang and
    corrupt the executable. Anchoring on the path cannot land anywhere else.
    """
    with open(path, 'rb') as f:
        data = f.read()

    # uv writes its OWN launcher ("trampoline"), which stores the interpreter
    # path inside the PE image rather than as a `#!` line before the zip. It is
    # not patchable the same way and MUST NOT be guessed at: replacing the path
    # bytes shortens the file, shifts every following offset, and the result is
    # "%1 is not a valid Win32 application" — verified on a copy of tqdm.exe.
    # Reported separately instead, with the supported fix.
    if b'UV_TRAMPOLINE_KIND' in data:
        return False

    idx = -1
    for v in _variants(old):
        idx = data.find(v.encode())
        if idx >= 0:
            break
    if idx < 0:
        return False

    k = data.rfind(b'#!', 0, idx)
    if k < 0:
        return False
    e = data.find(b'\n', idx)
    if e < 0:
        return False
    e += 1
    if data[e:e + 2] == b'\r\n':          # observed layout: "#!<path>\n\r\n" + zip
        e += 2
    if data[e:e + 4] != ZIP_MAGIC:
        return False                      # not the structure we understand — leave it

    stub, shebang, payload = data[:k], data[k:e], data[e:]
    new_shebang = shebang
    for a, b in zip(_variants(old), _variants(new)):
        new_shebang = new_shebang.replace(a.encode(), b.encode())
    if new_shebang == shebang:
        return False
    if not dry:
        with open(path, 'wb') as f:
            f.write(stub + new_shebang + payload)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--venv', default=DEFAULT_VENV)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--old', default=None,
                    help='stale path to replace, if auto-detection misses it')
    args = ap.parse_args()

    venv = os.path.abspath(args.venv)
    if not os.path.isdir(venv):
        sys.exit(f"no venv at {venv}")

    new = os.path.normpath(venv)
    olds = [os.path.normpath(args.old)] if args.old else recorded_paths(venv)
    if not olds:
        print(f"venv already agrees with its location:\n  {new}")
        return

    print(f"actual : {new}")
    for o in olds:
        print(f"stale  : {o}")
    print(f"\n{'would repair' if args.dry_run else 'repairing'} ...\n")

    text_fixed, exe_fixed, skipped = [], [], []
    for root, dirs, files in os.walk(venv):
        # __pycache__: a .pyc embeds the source path in its code objects, so
        # thousands of them "contain" the stale path. It only ever shows up in a
        # traceback, and Python rewrites them from source anyway — patching
        # compiled bytecode to tidy a cosmetic string is all risk, no benefit.
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fn in files:
            if fn.lower().endswith(('.pyc', '.pyo')):
                continue
            p = os.path.join(root, fn)
            try:
                if os.path.getsize(p) == 0:
                    continue
                with open(p, 'rb') as f:
                    head = f.read(4 * 1024 * 1024)
                hits = [o for o in olds
                        if any(v.encode() in head for v in _variants(o))]
                if not hits:
                    continue
            except OSError:
                continue
            if fn.lower().endswith('.exe'):
                ok = any(repair_launcher(p, o, new, args.dry_run) for o in hits)
                (exe_fixed if ok else skipped).append(p)
            elif _is_text(p):
                ok = any(repair_text(p, o, new, args.dry_run) for o in hits)
                (text_fixed if ok else skipped).append(p)
            else:
                skipped.append(p)

    def rel(p):
        return os.path.relpath(p, venv)

    for label, group in (('text', text_fixed), ('launcher exe', exe_fixed)):
        print(f"  {len(group)} {label} file(s)")
        for p in group[:8]:
            print(f"     {rel(p)}")
        if len(group) > 8:
            print(f"     ... and {len(group) - 8} more")
    if skipped:
        uv = [p for p in skipped if p.lower().endswith('.exe')]
        other = [p for p in skipped if p not in uv]
        if uv:
            print(f"\n  {len(uv)} uv-trampoline launcher(s) NOT repaired — these "
                  f"store the path inside the PE image and cannot be patched "
                  f"safely (verified: doing so produces 'not a valid Win32 "
                  f"application'). None is needed to run the app; regenerate "
                  f"one with:")
            print(f"     uv pip install --reinstall --no-deps <package>")
            for p in uv[:6]:
                print(f"     {rel(p)}")
            if len(uv) > 6:
                print(f"     ... and {len(uv) - 6} more")
        if other:
            # dist-info/direct_url.json and the like: a record of where a wheel
            # was installed from. Harmless, and rewriting it would falsify
            # install provenance.
            print(f"\n  {len(other)} metadata file(s) left alone deliberately")
            for p in other[:6]:
                print(f"     {rel(p)}")
            if len(other) > 6:
                print(f"     ... and {len(other) - 6} more")

    if args.dry_run:
        print("\ndry run - nothing written")
    else:
        print("\ndone. Verify with:")
        print(f'  "{os.path.join(venv, "Scripts", "activate.bat")}" && '
              f'where python && python -c "import torch"')


if __name__ == '__main__':
    main()
