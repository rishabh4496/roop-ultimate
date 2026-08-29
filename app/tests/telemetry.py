"""Host and device telemetry sampling for the phase benchmarks.

Split out of `bench_phase9_nvdec.py` so Phase 2's controlled baseline, the
Phase 11 enhancer matrix and any later phase all report the same fields the
same way. The plan's Phase 2 exit criteria name CPU, P/E-core, GPU, VRAM, RAM,
and queue depth; a sampler that only covers some of them silently leaves those
rows blank forever.

Sampled from the PARENT process against the child and its descendants. The
render spawns ffmpeg for decode and encode, so a sampler that watches only the
direct child under-reports RSS by however much ffmpeg holds.
"""
import os
import subprocess
import threading


_SMI_FIELDS = ("utilization.gpu", "memory.used", "power.draw",
               "utilization.memory", "temperature.gpu", "clocks.current.sm")


def cpu_topology():
    """Best-effort P-core / E-core split.

    Windows does not report hybrid topology through anything psutil exposes, so
    this is INFERRED from the CPU brand string plus the logical core count and
    is labelled as such wherever it is printed. It is never presented as an OS
    topology report -- the plan says "P/E-core utilization where available",
    and an inferred split announced as measured would be worse than a blank.
    """
    info = {"source": "unavailable", "p_logical": None, "e_logical": None}
    try:
        import psutil
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
    except Exception:
        return info
    info["logical"] = logical
    info["physical"] = physical
    if not logical or not physical:
        return info

    brand = ""
    try:
        import platform
        brand = platform.processor() or ""
        if os.name == "nt" and not brand.strip():
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True,
                stderr=subprocess.DEVNULL, timeout=5)
            brand = " ".join(out.split("\n")[1:]).strip()
    except Exception:
        pass
    info["brand"] = brand.strip()

    # A hybrid Intel part has MORE logical than 2x physical is false; the tell is
    # physical > logical/2, i.e. some cores are single-threaded E-cores.
    if physical > logical / 2:
        e_cores = 2 * physical - logical      # single-threaded cores
        p_cores = physical - e_cores          # SMT cores
        if e_cores > 0 and p_cores > 0:
            info.update({
                "source": "inferred from core counts (NOT an OS topology report)",
                "p_cores": p_cores, "e_cores": e_cores,
                # Intel enumerates P-core threads first on Windows.
                "p_logical": list(range(0, p_cores * 2)),
                "e_logical": list(range(p_cores * 2, logical)),
            })
    return info


def _smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(_SMI_FIELDS),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=3)
        line = next((l.strip() for l in out.stdout.splitlines() if l.strip()), "")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_SMI_FIELDS):
            return {}
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_memory_mb": float(parts[1]),
            "power_w": float(parts[2]),
            "gpu_mem_util_pct": float(parts[3]),
            "gpu_temp_c": float(parts[4]),
            "gpu_sm_mhz": float(parts[5]),
        }
    except (OSError, ValueError, subprocess.SubprocessError, StopIteration):
        return {}


def sample_loop(pid, stop, samples, topo, interval=0.5):
    """Append one dict per tick until `stop` is set."""
    try:
        import psutil
        proc = psutil.Process(pid)
        psutil.cpu_percent(percpu=True)      # prime the delta
    except Exception:
        proc = None
        psutil = None

    while not stop.wait(interval):
        s = {}
        if proc is not None:
            try:
                tree = [proc] + proc.children(recursive=True)
                s["rss_gb"] = sum(p.memory_info().rss for p in tree) / (1024 ** 3)
                s["procs"] = len(tree)
            except Exception:
                pass
        if psutil is not None:
            try:
                per = psutil.cpu_percent(percpu=True)
                if per:
                    s["cpu_pct"] = sum(per) / len(per)
                    p_idx, e_idx = topo.get("p_logical"), topo.get("e_logical")
                    if p_idx and e_idx:
                        s["cpu_p_pct"] = sum(per[i] for i in p_idx if i < len(per)) / len(p_idx)
                        s["cpu_e_pct"] = sum(per[i] for i in e_idx if i < len(per)) / len(e_idx)
                s["ram_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)
            except Exception:
                pass
        s.update(_smi())
        if s:
            samples.append(s)


def summarise(samples):
    """Peak and mean for every key any sample carried."""
    if not samples:
        return {"telemetry_samples": 0}
    out = {"telemetry_samples": len(samples)}
    keys = set()
    for s in samples:
        keys.update(s.keys())
    for key in sorted(keys):
        vals = [s[key] for s in samples if key in s]
        if not vals:
            continue
        out["peak_" + key] = round(max(vals), 3)
        out["mean_" + key] = round(sum(vals) / len(vals), 3)
    return out


def run_sampled(cmd, env=None, cwd=None, on_line=None):
    """Run `cmd`, sampling host+device while it runs.

    Returns (returncode, stdout, telemetry). stderr is merged into stdout so a
    traceback cannot vanish; `on_line` is called per line for live progress.
    """
    topo = cpu_topology()
    samples = []
    stop = threading.Event()
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    sampler = threading.Thread(target=sample_loop,
                               args=(proc.pid, stop, samples, topo), daemon=True)
    sampler.start()
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            if on_line:
                on_line(line.rstrip("\n"))
    finally:
        proc.wait()
        stop.set()
        sampler.join(timeout=3)
    telemetry = summarise(samples)
    telemetry["cpu_topology"] = topo
    return proc.returncode, "".join(lines), telemetry


__all__ = ["cpu_topology", "sample_loop", "summarise", "run_sampled"]
