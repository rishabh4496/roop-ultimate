"""Terminal Progress Monitor for Roop Ultimate.

Displays rich real-time visual progress for active render jobs in the terminal:
- Vivid ANSI 256-color gradient progress bar
- Accurate completion percentage
- Stabilized time left (ETA) and estimated local finish clock time
- Frame counts (completed / total / remaining)
- GPU telemetry (temperature, utilization, VRAM usage)
- System resources (RAM, disk free space)
- Target file and active chunk segment info

Usage:
  python tools/terminal_monitor.py         # Print current snapshot
  python tools/terminal_monitor.py --live  # Continuously stream live progress
"""

import sys
import os
import re
import time
import json
import argparse
import subprocess
from datetime import datetime, timedelta

# ANSI 256-Color Palette
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CLR_TITLE = "\033[1;38;5;206m"   # Neon Pink / Coral
CLR_BORDER = "\033[38;5;240m"  # Slate Gray
CLR_LABEL = "\033[38;5;248m"   # Silver Muted
CLR_PCT = "\033[1;38;5;220m"    # Bold Amber Gold
CLR_BAR_FILL = "\033[38;5;39m"  # Electric Cyan
CLR_BAR_EMPTY = "\033[38;5;238m" # Dark Ash Gray
CLR_VAL = "\033[1;37m"          # Bright White
CLR_TIME = "\033[38;5;78m"      # Mint Green (Elapsed)
CLR_ETA = "\033[1;38;5;84m"     # Bright Lime (ETA)
CLR_FINISH = "\033[1;38;5;117m" # Sky Blue (Clock Finish)
CLR_SPEED = "\033[1;38;5;214m"  # Warm Gold (FPS)
CLR_GPU = "\033[38;5;141m"      # Violet
CLR_WARN = "\033[1;38;5;203m"    # Soft Red

def format_duration(seconds):
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--"
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"

def get_latest_log_data(workspace_root):
    """Parse the most recent progress line from logs/api/start_react.js/latest."""
    log_path = os.path.join(workspace_root, "logs", "api", "start_react.js", "latest")
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 65536))
            raw = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    # Look for tqdm progress pattern: e.g. 96600/147630 [1:17:13<37:53, 22.44frames/s, ...]
    matches = list(re.finditer(
        r"(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:?]+),\s*([0-9.]+)\s*(?:frames|it)/s(?:,\s*memory_usage=([0-9.]+[A-Za-z]+))?(?:,\s*execution_threads=(\d+))?",
        raw
    ))
    if not matches:
        return None
    last = matches[-1]
    n = int(last.group(1))
    total = int(last.group(2))
    elapsed_str = last.group(3)
    remaining_str = last.group(4)
    fps = float(last.group(5))
    mem = last.group(6) or "N/A"
    threads = last.group(7) or "N/A"

    return {
        "n": n,
        "total": total,
        "elapsed_str": elapsed_str,
        "remaining_str": remaining_str,
        "fps": fps,
        "mem": mem,
        "threads": threads
    }

def get_job_info(workspace_root):
    """Retrieve target filename and segments status from output resume JSON."""
    output_dir = os.path.join(workspace_root, "app", "output")
    if not os.path.isdir(output_dir):
        return {}
    try:
        resume_files = [f for f in os.listdir(output_dir) if f.endswith(".resume.json")]
        if not resume_files:
            return {}
        latest_resume = sorted(resume_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))[-1]
        with open(os.path.join(output_dir, latest_resume), "r", encoding="utf-8") as f:
            data = json.load(f)
        source = os.path.basename(data.get("source", "Video"))
        segments = data.get("segments", [])
        total_segs = max(1, round(data.get("frame_end", 1) / 7200))
        return {
            "source_name": source,
            "completed_segments": len(segments),
            "estimated_total_segments": total_segs,
            "fps": data.get("fps", 60.0),
            "resolution": f"{data.get('width', '?')}x{data.get('height', '?')}",
            "codec": data.get("effective_codec", "hevc_nvenc")
        }
    except Exception:
        return {}

def get_hardware_telemetry(workspace_root):
    """Query GPU usage and disk free space."""
    res = {"gpu_name": "NVIDIA GPU", "gpu_temp": None, "gpu_util": None, "vram_used": None, "vram_total": None, "disk_free_gb": None}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode("utf-8", errors="replace").strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 5:
            res["gpu_name"] = parts[0]
            res["gpu_temp"] = int(parts[1])
            res["gpu_util"] = int(parts[2])
            res["vram_used"] = round(float(parts[3]) / 1024, 1)
            res["vram_total"] = round(float(parts[4]) / 1024, 1)
    except Exception:
        pass

    try:
        import shutil
        total, used, free = shutil.disk_usage(workspace_root)
        res["disk_free_gb"] = round(free / (1024**3), 1)
    except Exception:
        pass

    return res

def render_dashboard(workspace_root, width=80):
    """Construct the rich visual terminal dashboard."""
    log_data = get_latest_log_data(workspace_root)
    job_info = get_job_info(workspace_root)
    hw_info = get_hardware_telemetry(workspace_root)

    if not log_data:
        return f"{CLR_WARN}[Roop Terminal]{RST} No active render stream detected in logs/api/start_react.js/latest."

    n = log_data["n"]
    total = log_data["total"]
    pct = (n / total * 100.0) if total > 0 else 0.0
    fps = log_data["fps"]
    
    # Calculate stabilized ETA and local finish time
    rem_frames = max(0, total - n)
    rem_secs = (rem_frames / fps) if fps > 0 else 0
    eta_formatted = format_duration(rem_secs)
    finish_dt = datetime.now() + timedelta(seconds=rem_secs)
    finish_str = finish_dt.strftime("%I:%M %p")

    # Bar rendering
    bar_width = 30
    filled_len = int(round(bar_width * (pct / 100.0)))
    filled_len = min(bar_width, max(0, filled_len))
    bar_str = f"{CLR_BAR_FILL}{'█' * filled_len}{CLR_BAR_EMPTY}{'░' * (bar_width - filled_len)}{RST}"

    # Build Header
    lines = []
    lines.append(f"{CLR_BORDER}╔{'═' * (width - 2)}╗{RST}")
    lines.append(f"{CLR_BORDER}║{RST} {BOLD}{CLR_TITLE}⚡ ROOP ULTIMATE — REAL-TIME RENDER TERMINAL{RST}{' ' * (width - 48)}{CLR_BORDER}║{RST}")
    lines.append(f"{CLR_BORDER}╠{'═' * (width - 2)}╣{RST}")

    # Job info section
    src = job_info.get("source_name", "Target Video")
    if len(src) > 52:
        src = src[:49] + "..."
    lines.append(f"{CLR_BORDER}║{RST} {CLR_LABEL}Target:{RST} {CLR_VAL}{src:<54}{RST} {CLR_BORDER}║{RST}")

    cur_seg = job_info.get("completed_segments", 0) + 1
    est_total_segs = max(cur_seg, job_info.get("estimated_total_segments", 1))
    codec = job_info.get("codec", "hevc_nvenc")
    res = job_info.get("resolution", "1280x720")
    job_sub = f"Chunk {cur_seg}/{est_total_segs} ({codec}, {res} @ {job_info.get('fps', 60):.0f}fps)"
    lines.append(f"{CLR_BORDER}║{RST} {CLR_LABEL}Status:{RST} {CLR_VAL}{job_sub:<54}{RST} {CLR_BORDER}║{RST}")
    lines.append(f"{CLR_BORDER}╠{'═' * (width - 2)}╣{RST}")

    # Progress Bar Line
    bar_line = f"[{bar_str}] {CLR_PCT}{pct:5.1f}%{RST}"
    # Raw visible length without ANSI
    raw_bar_line = f"[{'█' * filled_len}{'░' * (bar_width - filled_len)}] {pct:5.1f}%"
    pad = width - 4 - len(raw_bar_line) - 10
    lines.append(f"{CLR_BORDER}║{RST} {CLR_LABEL}Progress:{RST} {bar_line}{' ' * max(0, pad)}{CLR_BORDER}║{RST}")

    # Frame details
    frame_info = f"{CLR_VAL}{n:,}{RST} / {CLR_LABEL}{total:,}{RST} frames ({CLR_VAL}{rem_frames:,}{RST} remaining)"
    raw_frame_info = f"{n:,} / {total:,} frames ({rem_frames:,} remaining)"
    pad_frames = width - 4 - len(raw_frame_info) - 8
    lines.append(f"{CLR_BORDER}║{RST} {CLR_LABEL}Frames:{RST}   {frame_info}{' ' * max(0, pad_frames)}{CLR_BORDER}║{RST}")

    # Speed, Elapsed, ETA, Finish Time
    metrics = (
        f"{CLR_LABEL}Speed:{RST} {CLR_SPEED}{fps:.1f} fps{RST}  "
        f"{CLR_BORDER}│{RST}  {CLR_LABEL}Elapsed:{RST} {CLR_TIME}{log_data['elapsed_str']}{RST}  "
        f"{CLR_BORDER}│{RST}  {CLR_LABEL}ETA:{RST} {CLR_ETA}{eta_formatted}{RST}"
    )
    raw_metrics = f"Speed: {fps:.1f} fps  │  Elapsed: {log_data['elapsed_str']}  │  ETA: {eta_formatted}"
    pad_metrics = width - 4 - len(raw_metrics) - 2
    lines.append(f"{CLR_BORDER}║{RST}  {metrics}{' ' * max(0, pad_metrics)}{CLR_BORDER}║{RST}")

    finish_line = f"Est. Completion: {CLR_FINISH}{finish_str}{RST} (local time)"
    raw_finish = f"Est. Completion: {finish_str} (local time)"
    pad_finish = width - 4 - len(raw_finish) - 2
    lines.append(f"{CLR_BORDER}║{RST}  {finish_line}{' ' * max(0, pad_finish)}{CLR_BORDER}║{RST}")

    lines.append(f"{CLR_BORDER}╠{'═' * (width - 2)}╣{RST}")

    # Hardware Telemetry Section
    gpu_label = hw_info["gpu_name"]
    if len(gpu_label) > 28:
        gpu_label = gpu_label[:25] + "..."
    gpu_metrics = f"{CLR_GPU}{gpu_label}{RST}"
    if hw_info["gpu_temp"] is not None:
        gpu_metrics += f" @ {CLR_VAL}{hw_info['gpu_temp']}°C{RST} ({CLR_SPEED}{hw_info['gpu_util']}%{RST} load)"
    vram_str = f"VRAM: {CLR_GPU}{hw_info['vram_used'] or '?'} GB{RST} / {hw_info['vram_total'] or '?'} GB"
    sys_str = f"RAM: {CLR_VAL}{log_data['mem']}{RST}  {CLR_BORDER}│{RST}  Disk Free: {CLR_VAL}{hw_info['disk_free_gb'] or '?'} GB{RST}  {CLR_BORDER}│{RST}  Threads: {CLR_VAL}{log_data['threads']}{RST}"

    raw_gpu_full = f"{gpu_label} @ {hw_info['gpu_temp']}°C ({hw_info['gpu_util']}% load)  │  VRAM: {hw_info['vram_used']} GB / {hw_info['vram_total']} GB"
    pad_gpu = width - 4 - len(raw_gpu_full) - 2
    lines.append(f"{CLR_BORDER}║{RST}  {gpu_metrics}  {CLR_BORDER}│{RST}  {vram_str}{' ' * max(0, pad_gpu)}{CLR_BORDER}║{RST}")

    raw_sys = f"RAM: {log_data['mem']}  │  Disk Free: {hw_info['disk_free_gb']} GB  │  Threads: {log_data['threads']}"
    pad_sys = width - 4 - len(raw_sys) - 2
    lines.append(f"{CLR_BORDER}║{RST}  {sys_str}{' ' * max(0, pad_sys)}{CLR_BORDER}║{RST}")

    lines.append(f"{CLR_BORDER}╚{'═' * (width - 2)}╝{RST}")
    return "\n".join(lines)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Roop Ultimate Live Terminal Monitor")
    parser.add_argument("--live", "-w", action="store_true", help="Continuously monitor and update terminal display")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds (default: 1.0)")
    args = parser.parse_args()

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if not args.live:
        print(render_dashboard(workspace_root))
        return

    # Clear screen and hide cursor on live mode
    try:
        sys.stdout.write("\033[?25l\033[2J")
        sys.stdout.flush()
        while True:
            dash = render_dashboard(workspace_root)
            sys.stdout.write("\033[H" + dash + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
