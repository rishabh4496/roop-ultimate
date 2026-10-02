"""Pre-build the TensorRT engines the APP will actually load.

Why this exists. ``tools/build_trt_engines.py`` builds a different set of models
(inswapper_128, GPEN-512, scrfd_2.5g) into a different, flat cache directory with
different provider options, so nothing it writes is ever read by the app: the app
looks in ``app/models/trt_cache/<namespace>/`` where the namespace carries the GPU,
compute capability, driver, CUDA/TensorRT/ORT versions, precision and tuning knobs.
Engines for a model are keyed on that namespace AND on the session's options, so a
build only counts if it goes through the app's own provider construction.

This does exactly that. It brings the app up the way a render does (config.yaml ->
``init_pipeline`` -> the same processor loop ``ProcessMgr.initialize`` runs), loads
every model of the stack you configured, and runs one dummy inference through each
session -- TensorRT allocates and BUILDS on the first inference, not at session
construction, so without that pass nothing is built.

Per stage it reports the wall time, the providers the session ended up on, and how
many bytes the engine cache grew by. Growth means the engine was built now (COLD);
none means it was already cached (warm). A session that was asked for TensorRT and
is not on it is reported, and the exit code is non-zero.

    python tools/prebuild_engines.py                  # the stack in config.yaml
    python tools/prebuild_engines.py --enhancer "GPEN 256 Pro" --swap-model realswap
    python tools/prebuild_engines.py --only analyser,swapper

Run it with the app STOPPED (a running app holds the GPU, and a render that starts
mid-build would wait on the same engines). It never touches the model files.

Not done here, on purpose: compiling in the background while the app runs on CUDA.
An engine is keyed on the exact session options of the loader, so a throwaway session
built beside the real one only helps if every loader's options match; run this once
before starting the app instead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")

# A stage that grew the cache by less than this was a cache hit (TensorRT rewrites
# small bookkeeping files even on a warm start).
COLD_BYTES = 256 * 1024
STAGES = ("analyser", "swapper", "mask", "enhancer")


def dir_bytes(path: str) -> int:
    """Total size of every file under *path*; 0 when it does not exist."""
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def classify(delta_bytes: int) -> str:
    return "COLD (built now)" if delta_bytes >= COLD_BYTES else "warm (cached)"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--provider", default="tensorrt",
                   help="execution provider to build for (default: tensorrt)")
    p.add_argument("--swap-model", default=None, help="default: config.yaml swap_model")
    p.add_argument("--enhancer", default=None, help="default: config.yaml selected_enhancer")
    p.add_argument("--mask-engine", default=None, help="default: config.yaml mask_engine")
    p.add_argument("--only", default=None,
                   help="comma list of stages to build: %s (default: all)" % ",".join(STAGES))
    args = p.parse_args(argv)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in STAGES]
        if unknown:
            p.error("unknown stage(s): %s (choose from %s)" % (", ".join(unknown), ", ".join(STAGES)))
        args.stages = wanted
    else:
        args.stages = list(STAGES)
    return args


def _is_session(value) -> bool:
    return callable(getattr(value, "get_inputs", None)) and callable(getattr(value, "run", None))


def _ort_sessions(processor, _depth: int = 0) -> Dict[str, object]:
    """Every ORT session a processor holds: direct attributes, pool items, and one
    level of nested processor (RealSwap keeps its second network on ``secondary``)."""
    found: Dict[str, object] = {}
    attrs = dict(vars(processor))
    for name, value in attrs.items():
        if _is_session(value):
            found[name] = value
    pool = attrs.get("pool")
    for i, item in enumerate(list(getattr(pool, "_items", None) or ())):
        if _is_session(item):
            found["pool[%d]" % i] = item
    if _depth == 0:
        for name, value in attrs.items():
            if not name.startswith("_") and hasattr(value, "__dict__") and not _is_session(value)                     and name in ("secondary", "primary"):
                for sub, sess in _ort_sessions(value, 1).items():
                    found["%s.%s" % (name, sub)] = sess
    return found


def classify_sessions(active_by_label: Dict[str, List[str]], requested_trt: bool):
    """(verified_on_trt, off_trt, status) for one stage.

    A stage that found no session at all is UNVERIFIED -- not "fine": saying
    nothing was wrong because nothing was looked at is how a check reports success
    while not running.
    """
    if not active_by_label:
        return [], [], "UNVERIFIED (no session found to check)"
    on = [k for k, a in active_by_label.items() if a and "tensorrt" in a[0].lower()]
    off = [k for k in active_by_label if k not in on]
    if requested_trt and off:
        return on, off, "NOT ON TENSORRT"
    return on, off, "verified %d session(s)%s" % (len(on), "" if requested_trt else " (TensorRT not requested)")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sys.path[:0] = [APP, os.path.join(APP, "tests")]
    os.chdir(APP)

    import angle_bench as ab

    from settings import Settings
    cfg = Settings("config.yaml")
    swap_model = args.swap_model or cfg.swap_model
    enhancer = args.enhancer or cfg.selected_enhancer
    mask_engine = args.mask_engine or getattr(cfg, "mask_engine", "None")
    mask_engine_2 = getattr(cfg, "mask_engine_2", "None") if not args.mask_engine else "None"

    print("[prebuild] stack: swap=%s enhancer=%s mask=%s provider=%s"
          % (swap_model, enhancer, mask_engine, args.provider), flush=True)
    g = ab.init_pipeline(args.provider, swap_model, enhancer, mask_engine, sync_config=True)

    import numpy as np
    import roop.globals
    from roop import predictor
    from roop.utilities import get_device

    cache_root = os.path.join(APP, "models", "trt_cache")
    namespace = getattr(roop.globals, "trt_active_cache_dir", None)
    print("[prebuild] engine cache: %s" % (namespace or cache_root), flush=True)

    rows = []

    requested_trt = "tensorrt" in str(args.provider).lower()

    def warm(label, sess, **kw):
        predictor.warmup_session(sess, "prebuild:%s" % label, once=False, **kw)

    def stage(name, fn):
        before_bytes = dir_bytes(cache_root)
        started = time.time()
        error, sessions = None, {}
        try:
            sessions = fn() or {}
        except Exception as exc:  # report, keep going: one bad model must not hide the rest
            error = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:160])
        seconds = time.time() - started
        delta = dir_bytes(cache_root) - before_bytes
        # Read the providers AFTER the warm-up: ORT can drop an EP during the first run.
        active = {}
        for label, sess in sessions.items():
            try:
                active[label] = list(sess.get_providers())
            except Exception:
                active[label] = []
        on, off, status = classify_sessions(active, requested_trt)
        rows.append({"stage": name, "seconds": seconds, "delta": delta, "status": status,
                     "off_trt": off, "error": error, "unverified": status.startswith("UNVERIFIED")})
        print("[prebuild] %-26s %7.1fs  %+8.1f MB  %-15s %s%s%s" % (
            name, seconds, delta / 1048576.0, classify(delta), status,
            "  [%s]" % ",".join(off) if off and requested_trt else "",
            "  ERROR: %s" % error if error else ""), flush=True)

    # -- analyser: detector + recognition + landmarks ------------------------------
    if "analyser" in args.stages:
        def analyser():
            from roop.face_util import get_face_analyser
            fa = get_face_analyser()
            fa = fa[0] if isinstance(fa, (list, tuple)) else fa
            det_px = int(str(getattr(roop.globals, "face_detector_size", "640")) or 640)
            held = {}
            for key, model in (getattr(fa, "models", None) or {}).items():
                sess = getattr(model, "session", None)
                if sess is not None:
                    warm("analyser/%s" % key, sess, default_hw=(det_px,))
                    held["analyser/%s" % key] = sess
            return held
        stage("analyser (buffalo_l)", analyser)

    # -- processors, through the SAME loop ProcessMgr.initialize runs ---------------
    from roop.ProcessMgr import ProcessMgr
    from roop.core import get_processing_plugins
    # config.yaml stores the UI LABEL ("DFL XSeg"); the processor key is what
    # api.map_mask_engines turns it into. A value that is already a key passes through.
    try:
        import api
        mapped = api.map_mask_engines(mask_engine, mask_engine_2, "")
    except Exception as exc:
        print("[prebuild] could not map mask engine labels (%s); using %r as given"
              % (exc, mask_engine), flush=True)
        mapped = mask_engine
    mask_list = ([] if mapped in (None, "", "None", []) else
                 list(mapped) if isinstance(mapped, (list, tuple)) else [mapped])
    options = get_processing_plugins(mask_list, swap_model=swap_model)
    manager = ProcessMgr(None)
    device = get_device()
    kind_of = {"faceswap": "swapper"}
    for key, extoption in options.items():
        kind = kind_of.get(key, "mask" if key.startswith("mask_") else "enhancer")
        if kind not in args.stages:
            continue

        def build(key=key, extoption=extoption):
            from roop.utilities import str_to_class
            classname = manager.plugins[key]
            proc = str_to_class("roop.processors." + classname, classname)
            if proc is None:
                raise RuntimeError("no processor class for %r" % key)
            opts = dict(extoption)
            opts["devicename"] = device
            proc.Initialize(opts)
            held = _ort_sessions(proc)
            for attr, sess in held.items():
                warm("%s.%s" % (key, attr), sess)
            return held
        stage("%s (%s)" % (key, kind), build)

    # -- summary -------------------------------------------------------------------
    cold = [r for r in rows if r["delta"] >= COLD_BYTES]
    bad = [r for r in rows if (requested_trt and r["off_trt"]) or r["error"]]
    unverified = [r for r in rows if r["unverified"] and not r["error"]]
    total = sum(r["seconds"] for r in rows)
    print("\n[prebuild] %d stage(s), %.1fs total, %d built now, %d already cached"
          % (len(rows), total, len(cold), len(rows) - len(cold)), flush=True)
    if unverified:
        print("[prebuild] UNVERIFIED (no session could be inspected, so nothing is claimed "
              "about them): " + ", ".join(r["stage"] for r in unverified), flush=True)
    if bad:
        print("[prebuild] PROBLEMS: " + "; ".join(
            "%s%s" % (r["stage"], " (not on TensorRT: %s)" % ",".join(r["off_trt"])
                      if r["off_trt"] else " (%s)" % r["error"]) for r in bad), flush=True)
        return 1
    verified = [r for r in rows if not r["unverified"]]
    print("[prebuild] OK for %d of %d stage(s): every session that could be inspected is on "
          "TensorRT with its engine in the active namespace." % (len(verified), len(rows)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
