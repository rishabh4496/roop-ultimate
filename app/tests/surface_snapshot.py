"""Public-surface snapshot — proves a refactor was a PURE MOVE.

Decomposing a large module is only safe if nothing observable changes. This
records everything a caller could depend on:

  * every module-level name, its kind, and (for callables) its exact signature
  * every class, its methods, and their signatures
  * for api.py, the full FastAPI route table — path, methods, endpoint name and
    declaration order, since route order decides which handler wins

Run it before a change and after, then diff:

    env/Scripts/python tests/surface_snapshot.py before.json
    ...refactor...
    env/Scripts/python tests/surface_snapshot.py after.json
    env/Scripts/python tests/surface_snapshot.py --diff before.json after.json

A clean diff plus a green test run is what makes "I only moved code" a checked
claim instead of an assurance.
"""

import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ("roop.ProcessMgr", "roop.face_util", "roop.globals", "settings", "api")


# Default values that repr as "<... object at 0x7f...>" embed a memory address
# that changes every interpreter run, so raw signature strings compare unequal
# for reasons that have nothing to do with the code. Third-party sentinels
# (FastAPI's DefaultPlaceholder, for one) are full of them.
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _normalise(sig):
    return _ADDR.sub(" at 0xADDR", sig)


def _sig(obj):
    try:
        return _normalise(str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "<no signature>"


def _describe(module):
    out = {"functions": {}, "classes": {}, "data": []}
    for name, value in sorted(vars(module).items()):
        if inspect.ismodule(value):
            continue
        # Only record things this module defines or deliberately re-exports;
        # a re-exported name is part of the surface too, so it is kept.
        if inspect.isclass(value):
            # Walk the MRO, not just vars(): moving methods onto a mixin keeps
            # them callable on the class, and that effective surface is what
            # callers depend on. Using vars() would report a mixin split as a
            # pile of deleted methods.
            methods = {}
            for mname in sorted(dir(value)):
                if mname.startswith("__") and mname.endswith("__"):
                    continue
                try:
                    mval = inspect.getattr_static(value, mname)
                except AttributeError:
                    continue
                if isinstance(mval, (staticmethod, classmethod)):
                    mval = mval.__func__
                elif isinstance(mval, property):
                    methods[mname] = "<property>"
                    continue
                if callable(mval):
                    methods[mname] = _sig(mval)
            out["classes"][name] = methods
        elif callable(value):
            out["functions"][name] = _sig(value)
        else:
            out["data"].append(name)
    return out


def _routes(api_module):
    """Every path the app will actually serve.

    Routes added via include_router do NOT appear directly in app.routes on
    modern FastAPI: it stores a lazy `_IncludedRouter` wrapper whose own `path`
    is None, so a naive walk silently reports every included endpoint as
    missing. Recurse into `original_router` (and the older `.router`/`.routes`
    shapes) so a decomposition into routers compares against the same table as
    the monolith it replaced.
    """
    app = getattr(api_module, "app", None)
    if app is None:
        return []

    rows, seen = [], set()

    def walk(routes, depth=0):
        if depth > 5:
            return
        for route in routes:
            inner = (getattr(route, "original_router", None)
                     or getattr(route, "router", None))
            if inner is not None and hasattr(inner, "routes"):
                walk(inner.routes, depth + 1)
                continue
            path = getattr(route, "path", None)
            if path is None:
                continue
            row = {
                "path": path,
                "methods": sorted(getattr(route, "methods", []) or []),
                "name": getattr(route, "name", None),
            }
            key = (row["path"], tuple(row["methods"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    walk(app.routes)
    return rows


def snapshot():
    data = {}
    for name in MODULES:
        try:
            module = __import__(name, fromlist=["*"])
        except Exception as exc:                       # noqa: BLE001
            data[name] = {"IMPORT_ERROR": f"{type(exc).__name__}: {exc}"}
            continue
        data[name] = _describe(module)
        if name == "api":
            data[name]["routes"] = _routes(module)
    return data


def _normalise_snapshot(snap):
    """Apply address normalisation to a snapshot taken before it existed, so old
    files stay comparable."""
    for mod in snap.values():
        for name, sig in mod.get("functions", {}).items():
            mod["functions"][name] = _normalise(sig)
        for cls in mod.get("classes", {}).values():
            for m, sig in cls.items():
                cls[m] = _normalise(sig)
    return snap


def diff(before, after):
    before, after = _normalise_snapshot(before), _normalise_snapshot(after)
    problems = []
    for mod in sorted(set(before) | set(after)):
        b, a = before.get(mod), after.get(mod)
        if b is None or a is None:
            problems.append(f"{mod}: module {'added' if b is None else 'removed'}")
            continue
        if "IMPORT_ERROR" in a and "IMPORT_ERROR" not in b:
            problems.append(f"{mod}: now fails to import — {a['IMPORT_ERROR']}")
            continue
        for section in ("functions", "classes"):
            bs, as_ = b.get(section, {}), a.get(section, {})
            for gone in sorted(set(bs) - set(as_)):
                problems.append(f"{mod}.{gone}: {section[:-2]} disappeared from the module surface")
            for name in sorted(set(bs) & set(as_)):
                if bs[name] != as_[name]:
                    if section == "classes":
                        for m in sorted(set(bs[name]) - set(as_[name])):
                            problems.append(f"{mod}.{name}.{m}: method removed")
                        for m in sorted(set(bs[name]) & set(as_[name])):
                            if bs[name][m] != as_[name][m]:
                                problems.append(
                                    f"{mod}.{name}.{m}: signature {bs[name][m]} -> {as_[name][m]}")
                    else:
                        problems.append(f"{mod}.{name}: signature {bs[name]} -> {as_[name]}")
        for gone in sorted(set(b.get("data", [])) - set(a.get("data", []))):
            problems.append(f"{mod}.{gone}: module-level value disappeared")
        if b.get("routes") is not None:
            if b["routes"] != a.get("routes"):
                bl = {(r["path"], tuple(r["methods"])) for r in b["routes"]}
                al = {(r["path"], tuple(r["methods"])) for r in a.get("routes", [])}
                for r in sorted(bl - al):
                    problems.append(f"{mod}: ROUTE LOST {r[1]} {r[0]}")
                for r in sorted(al - bl):
                    problems.append(f"{mod}: ROUTE ADDED {r[1]} {r[0]}")
                if bl == al:
                    problems.append(f"{mod}: route ORDER or names changed "
                                    f"(same paths) — later routes can shadow earlier ones")
    return problems


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--diff":
        with open(sys.argv[2], encoding="utf-8") as fh:
            before = json.load(fh)
        with open(sys.argv[3], encoding="utf-8") as fh:
            after = json.load(fh)
        issues = diff(before, after)
        if issues:
            print(f"SURFACE CHANGED — {len(issues)} problem(s):")
            for line in issues:
                print("  -", line)
            sys.exit(1)
        n = sum(len(v.get("functions", {})) for v in after.values())
        c = sum(len(v.get("classes", {})) for v in after.values())
        r = len(after.get("api", {}).get("routes", []))
        print(f"SURFACE IDENTICAL — {n} functions, {c} classes, {r} routes unchanged")
    else:
        target = sys.argv[1] if len(sys.argv) > 1 else "surface.json"
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(snapshot(), fh, indent=1, sort_keys=True)
        print("wrote", target)
