"""Dry-run verification that CPU affinity and ORT thread settings ACTUALLY apply.

Run this BEFORE a render, not after, to answer three questions that this
project has repeatedly got wrong by reading intent instead of outcome:

  1. Where are the P-cores really?  The naive answer on an i9-14900K is
     "logical 0-15", and on this box that happens to be right -- but it is read
     from the OS efficiency class, never assumed. A hardcoded range(0, 16) is
     wrong on any chip whose enumeration differs, and wrong the moment
     hyperthreading is turned off.
  2. Did the affinity call actually take?  `apply_cpu_affinity` returns a dict
     that says so, but a returned dict is a claim. This re-reads the process
     affinity mask from the OS and, separately, samples the logical CPU that
     BUSY WORKER THREADS are really executing on via GetCurrentProcessorNumber.
     A process mask that is set while a worker still runs on an E-core is the
     failure this check exists to catch.
  3. Do the ORT thread settings reach the models the user actually runs?
     `--load-models` builds the REAL swapper and enhancer sessions through the
     app's own loaders and reads back what ORT holds. Reading the SessionOptions
     object that `get_onnx_session_options()` returns proves only that the
     helper works; it does not prove any model was built with it.

Exit status is 0 only if every requested check passes, so this is usable as a
gate in front of a long render.

Examples::

    env/Scripts/python.exe tests/verify_thread_binding.py
    env/Scripts/python.exe tests/verify_thread_binding.py --distribution p_only --intra 6 --inter 2
    env/Scripts/python.exe tests/verify_thread_binding.py --distribution p_only --load-models
    env/Scripts/python.exe tests/verify_thread_binding.py --control

`--control` is the arm that MUST fail: it asks for a P-core-only policy, does
not apply it, and asserts the checker notices. A verifier that cannot produce a
negative result is not evidence of anything.
"""
import argparse
import ctypes
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


# --------------------------------------------------------------------------
# observation primitives
# --------------------------------------------------------------------------

def _current_processor():
    """The logical CPU this thread is executing on right now, or None."""
    if os.name != "nt":
        try:
            return os.sched_getcpu()          # Linux
        except AttributeError:
            return None
    try:
        return int(ctypes.windll.kernel32.GetCurrentProcessorNumber())
    except Exception:
        return None


def observe_worker_cpus(n_threads, seconds=0.8):
    """Run n busy threads and collect the logical CPUs they actually land on.

    THE WORK MUST RELEASE THE GIL or this measures the GIL, not the affinity.
    A pure-Python arithmetic loop keeps the interpreter lock, so only one thread
    runs at a time and the sample stays tiny however many threads are started.
    Measured on this box, 36 threads for 0.8 s on a completely unpinned process:

        pure-Python loop   9 of 32 logical CPUs observed
        numpy matmul      32 of 32 logical CPUs observed

    The first would have reported "no worker on an E-core" for a process that
    was free to use every E-core -- a false PASS. numpy drops the GIL for the
    duration of the BLAS call, which is also the closer analogue of the real
    workload, where the time is spent inside ORT/CUDA native code.
    """
    import numpy as np

    seen = set()
    lock = threading.Lock()
    stop = time.time() + seconds
    left = np.random.rand(256, 256).astype(np.float32)
    right = np.random.rand(256, 256).astype(np.float32)

    def _work():
        local = set()
        while time.time() < stop:
            np.dot(left, right)          # releases the GIL
            cpu = _current_processor()
            if cpu is not None:
                local.add(cpu)
        with lock:
            seen.update(local)

    threads = [threading.Thread(target=_work, daemon=True)
               for _ in range(max(1, n_threads))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return seen


def process_affinity():
    try:
        import psutil
        return set(int(i) for i in psutil.Process(os.getpid()).cpu_affinity())
    except Exception as exc:
        return exc


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.rows = []
        self.failed = 0

    def add(self, ok, name, detail):
        self.rows.append((ok, name, detail))
        if ok is False:
            self.failed += 1

    def render(self):
        print("")
        for ok, name, detail in self.rows:
            tag = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            print("  [%s] %-32s %s" % (tag, name, detail))
        print("")
        if self.failed:
            print("RESULT: %d check(s) FAILED -- do not start the render."
                  % self.failed)
        else:
            print("RESULT: all checks passed.")
        return 1 if self.failed else 0


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_topology(rep):
    import psutil
    from roop.runtime_optimizer import detect_cpu_topology
    topo = detect_cpu_topology(psutil.cpu_count(False), psutil.cpu_count(True))
    p = tuple(int(i) for i in topo.get("p_indices") or ())
    e = tuple(int(i) for i in topo.get("e_indices") or ())
    src = topo.get("source")
    if not p:
        rep.add(None, "cpu topology",
                "no P/E split measured (source=%s) -- every policy other than "
                "'auto' will refuse to apply" % src)
        return p, e
    rep.add(True, "cpu topology",
            "%d P-cores (%d logical) + %d E-cores (%d logical), source=%s"
            % (topo["p_cores"], len(p), topo["e_cores"], len(e), src))
    naive = tuple(range(0, len(p)))
    if p != naive:
        rep.add(True, "topology vs naive range()",
                "MEASURED %d..%d is NOT range(0,%d) -- a hardcoded range would "
                "have pinned the wrong cores" % (p[0], p[-1], len(p)))
    else:
        rep.add(True, "topology vs naive range()",
                "measured P set equals range(0,%d) here; still read from the "
                "OS, not assumed" % len(p))
    return p, e


def check_affinity(rep, p_indices, e_indices, distribution, apply_it):
    from roop.runtime_optimizer import HardwareProfiler, apply_cpu_affinity

    before = process_affinity()
    if isinstance(before, Exception):
        rep.add(False, "psutil affinity readable",
                "%s: %s -- affinity cannot be verified on this host"
                % (type(before).__name__, before))
        return
    rep.add(True, "psutil affinity readable", "%d logical CPUs" % len(before))

    hardware = HardwareProfiler().profile()
    if not hardware.os_affinity_supported:
        rep.add(None, "os affinity supported",
                "OS reports affinity unavailable; any policy would no-op")
        return

    if apply_it:
        os.environ["ROOP_CPU_DISTRIBUTION"] = distribution
        result = apply_cpu_affinity(hardware, distribution)
        if distribution == "auto":
            rep.add(result.get("applied") is False, "affinity policy=auto",
                    "left to the OS scheduler as designed (%s)"
                    % result.get("reason"))
        elif not result.get("applied"):
            rep.add(False, "affinity applied",
                    "policy=%s did NOT apply: %s"
                    % (distribution, result.get("reason")))
            return
        else:
            rep.add(True, "affinity applied",
                    "policy=%s -> %d logical CPUs"
                    % (distribution, len(result.get("indices") or ())))
    else:
        rep.add(None, "affinity applied",
                "control arm: policy=%s requested but deliberately NOT applied"
                % distribution)

    # -- outcome, not intent -----------------------------------------------
    expected = set(p_indices) if (distribution == "p_only" and p_indices) else None

    after = process_affinity()
    if expected is not None:
        ok = set(after) == expected
        if ok:
            rep.add(True, "process mask == P-cores",
                    "OS mask is exactly the %d P-core logicals" % len(expected))
        else:
            strays = sorted(set(after) & set(e_indices))
            rep.add(False, "process mask == P-cores",
                    "OS mask is %d CPUs, expected the %d P-core logicals; "
                    "E-cores still in mask: %s"
                    % (len(after), len(expected), strays[:8] or "none"))

    # Oversubscribe the mask deliberately. With fewer busy threads than logical
    # CPUs an idle box can schedule them all onto P-cores by chance, so the
    # "no worker on an E-core" check would PASS on an unpinned process -- it
    # would be measuring the machine's idleness, not the pin. More threads than
    # the mask holds forces the scheduler to use everything it is allowed to,
    # which is the only condition under which a clean sample is evidence.
    observed = observe_worker_cpus(n_threads=len(after) + 4)
    if not observed:
        rep.add(None, "worker threads observed",
                "GetCurrentProcessorNumber unavailable on this host")
        return
    stray = sorted(observed - set(after))
    if stray:
        rep.add(False, "workers inside process mask",
                "workers ran on CPUs OUTSIDE the mask: %s" % stray)
    else:
        rep.add(True, "workers inside process mask",
                "workers ran on %d distinct logical CPUs, all inside the mask"
                % len(observed))

    if expected is not None:
        on_e = sorted(observed & set(e_indices))
        if on_e:
            rep.add(False, "no worker on an E-core",
                    "workers OBSERVED on E-cores %s -- the pin did not take"
                    % on_e)
        else:
            rep.add(True, "no worker on an E-core",
                    "sampled %d logical CPUs, none an E-core" % len(observed))


def check_ort_options(rep, intra, inter):
    from roop.utilities import get_onnx_session_options
    import onnxruntime

    opts = get_onnx_session_options()
    if opts is None:
        rep.add(False, "SessionOptions built",
                "get_onnx_session_options() returned None")
        return

    ok = opts.intra_op_num_threads == intra
    rep.add(ok, "ort intra_op_num_threads",
            "requested %d, session holds %d%s"
            % (intra, opts.intra_op_num_threads,
               "" if ok else "   <-- SILENTLY CHANGED"))
    ok = opts.inter_op_num_threads == inter
    rep.add(ok, "ort inter_op_num_threads",
            "requested %d, session holds %d%s"
            % (inter, opts.inter_op_num_threads,
               "" if ok else "   <-- SILENTLY CHANGED"))
    rep.add(opts.execution_mode == onnxruntime.ExecutionMode.ORT_SEQUENTIAL,
            "ort execution_mode", str(opts.execution_mode))
    rep.add(True, "ort graph_optimization_level",
            "%s  (ALL is ORT's default here; CodeFormer/UltraMax deliberately "
            "step down to EXTENDED -- ALL fails to build those exports on the "
            "CPU provider)" % opts.graph_optimization_level)


def _sessions_of(proc):
    """Collect InferenceSession objects hanging off a processor, pools included."""
    import onnxruntime
    found = []

    def _walk(obj, depth=0):
        if depth > 3 or obj is None:
            return
        if isinstance(obj, onnxruntime.InferenceSession):
            found.append(obj)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item, depth + 1)
            return
        for attr in ("sessions", "_sessions", "session", "_session",
                     "instances", "_instances", "pool", "_pool", "_items"):
            if hasattr(obj, attr):
                try:
                    _walk(getattr(obj, attr), depth + 1)
                except Exception:
                    pass

    _walk(proc)
    for name in dir(proc):
        if name.startswith("__"):
            continue
        try:
            value = getattr(proc, name)
        except Exception:
            continue
        if isinstance(value, onnxruntime.InferenceSession):
            found.append(value)
        else:
            _walk(value, 2)
    return list({id(s): s for s in found}.values())


def check_models(rep, intra, inter):
    """Build the REAL swapper and enhancer and read back what ORT holds.

    The helper returning the right numbers does not prove any model was built
    with them -- a loader that constructs its own SessionOptions, or one that
    reuses a session built before the environment was set, would pass every
    check above and still run with ORT's defaults.
    """
    import roop.globals
    from settings import Settings

    cfg = Settings("config.yaml")
    roop.globals.CFG = cfg
    provider = str(getattr(cfg, "provider", "cuda") or "cuda").lower()
    mapping = {"tensorrt": "TensorrtExecutionProvider",
               "cuda": "CUDAExecutionProvider",
               "cpu": "CPUExecutionProvider"}
    roop.globals.execution_providers = [mapping.get(provider,
                                                    "CUDAExecutionProvider")]
    roop.globals.swap_model = getattr(cfg, "swap_model", "realswap")

    targets = []
    try:
        from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
        targets.append(("swapper/%s" % roop.globals.swap_model,
                        FaceSwapInsightFace()))
    except Exception as exc:
        rep.add(False, "swapper importable",
                "%s: %s" % (type(exc).__name__, exc))
    try:
        from roop.processors.Enhance_GPEN256Pro import Enhance_GPEN256Pro
        targets.append(("enhancer/GPEN 256 Pro", Enhance_GPEN256Pro()))
    except Exception as exc:
        rep.add(None, "GPEN importable", "%s: %s" % (type(exc).__name__, exc))

    for label, proc in targets:
        try:
            proc.Initialize({"devicename": "cuda"})
        except Exception as exc:
            rep.add(False, "%s built" % label,
                    "%s: %s" % (type(exc).__name__, exc))
            continue
        sessions = _sessions_of(proc)
        if not sessions:
            rep.add(None, "%s sessions" % label,
                    "initialised but no InferenceSession located to inspect")
        else:
            bad = []
            for sess in sessions:
                try:
                    so = sess.get_session_options()
                except Exception:
                    continue
                if (so.intra_op_num_threads != intra
                        or so.inter_op_num_threads != inter):
                    bad.append((so.intra_op_num_threads,
                                so.inter_op_num_threads))
            if bad:
                rep.add(False, "%s ort threads" % label,
                        "%d of %d session(s) NOT at intra=%d inter=%d "
                        "(found %s)" % (len(bad), len(sessions), intra, inter,
                                        bad[:3]))
            else:
                rep.add(True, "%s ort threads" % label,
                        "%d session(s), all at intra=%d inter=%d"
                        % (len(sessions), intra, inter))
        try:
            proc.Release()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distribution", default="p_only",
                    choices=["auto", "p_only", "p_plus_e", "p_priority_e"])
    ap.add_argument("--intra", type=int, default=6)
    ap.add_argument("--inter", type=int, default=2)
    ap.add_argument("--load-models", action="store_true",
                    help="build the real swapper + enhancer and read back "
                         "their ORT thread settings (slow, loads weights)")
    ap.add_argument("--control", action="store_true",
                    help="request a P-core policy but do NOT apply it; the "
                         "run MUST fail, proving the checker detects the "
                         "negative case")
    args = ap.parse_args()

    os.environ["ROOP_ORT_INTRA_THREADS"] = str(args.intra)
    os.environ["ROOP_ORT_INTER_THREADS"] = str(args.inter)

    print("=" * 74)
    print("thread-binding dry run   distribution=%s intra=%d inter=%d%s"
          % (args.distribution, args.intra, args.inter,
             "   [CONTROL: must fail]" if args.control else ""))
    print("=" * 74)

    rep = Report()
    p_indices, e_indices = check_topology(rep)
    check_affinity(rep, p_indices, e_indices, args.distribution,
                   apply_it=not args.control)
    check_ort_options(rep, args.intra, args.inter)
    if args.load_models:
        check_models(rep, args.intra, args.inter)

    code = rep.render()
    if args.control:
        if code == 0:
            print("CONTROL DID NOT FAIL -- the checker cannot detect an "
                  "unapplied pin, so a PASS from it means nothing.")
            return 1
        print("Control failed as required; these checks can detect an "
              "unapplied pin.")
        return 0
    return code


if __name__ == "__main__":
    sys.exit(main())
