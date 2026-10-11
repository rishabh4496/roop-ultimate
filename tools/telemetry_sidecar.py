"""Standalone Telemetry Sidecar for Roop Sustained Throughput Profiling.

Polls GPU, Host, and Windows Shared Memory counters every 1-2s into CSV.
Decoupled from roop pipeline code; observational only.
"""
import os
import time
import csv
import argparse
import subprocess
import psutil

def get_shared_gpu_memory_mb():
    try:
        cmd = ['typeperf', r'\GPU Adapter Memory(*)\Shared Usage', '-sc', '1']
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        if len(lines) >= 2:
            vals = lines[-1].replace('"', '').split(',')
            total_bytes = sum(float(v) for v in vals[1:] if v.replace('.', '', 1).isdigit())
            return round(total_bytes / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0

def get_gpu_telemetry():
    cmd = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active',
        '--format=csv,noheader,nounits'
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        parts = [p.strip() for p in out.split(',')]
        return {
            'gpu_util_pct': float(parts[0]),
            'gpu_mem_util_pct': float(parts[1]),
            'gpu_mem_used_mb': float(parts[2]),
            'gpu_temp_c': float(parts[3]),
            'gpu_power_w': float(parts[4]),
            'clocks_sm_mhz': float(parts[5]),
            'clocks_mem_mhz': float(parts[6]),
            'throttle_reasons': parts[7],
        }
    except Exception:
        return {}

def run_sidecar(target_pid, output_csv, interval=1.0):
    proc = psutil.Process(target_pid) if target_pid else psutil.Process()
    fieldnames = [
        'timestamp', 'elapsed_s', 'target_pid',
        'proc_cpu_pct', 'proc_rss_mb', 'proc_private_mb', 'proc_handles', 'proc_threads',
        'sys_free_ram_mb', 'sys_pagefile_used_mb',
        'gpu_util_pct', 'gpu_mem_util_pct', 'gpu_mem_used_mb', 'gpu_shared_mem_mb',
        'gpu_temp_c', 'gpu_power_w', 'clocks_sm_mhz', 'clocks_mem_mhz', 'throttle_reasons'
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        t0 = time.time()
        while True:
            if target_pid and not psutil.pid_exists(target_pid):
                break
            now = time.time()
            row = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': round(now - t0, 2), 'target_pid': target_pid or proc.pid}
            try:
                processes = [proc]
                processes.extend(proc.children(recursive=True))
                alive = []
                for child in processes:
                    try:
                        child.status()
                        alive.append(child)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                processes = alive
                row['proc_cpu_pct'] = round(sum(child.cpu_percent() for child in processes), 2)
                rss = 0
                private = 0
                handles = 0
                threads = 0
                for child in processes:
                    try:
                        mem = child.memory_info()
                        rss += mem.rss
                        try:
                            private += child.memory_full_info().uss
                        except (AttributeError, psutil.AccessDenied):
                            private += getattr(mem, 'vms', 0)
                        try:
                            handles += child.num_handles()
                        except (AttributeError, psutil.AccessDenied):
                            pass
                        threads += child.num_threads()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                row['proc_rss_mb'] = round(rss / (1024 * 1024), 2)
                row['proc_private_mb'] = round(private / (1024 * 1024), 2)
                row['proc_handles'] = handles
                row['proc_threads'] = threads
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            row['sys_free_ram_mb'] = round(vm.available / (1024 * 1024), 2)
            row['sys_pagefile_used_mb'] = round(sw.used / (1024 * 1024), 2)
            row.update(get_gpu_telemetry())
            row['gpu_shared_mem_mb'] = get_shared_gpu_memory_mb()
            writer.writerow(row)
            f.flush()
            time.sleep(interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Roop Telemetry Sidecar")
    parser.add_argument('--pid', type=int, default=None, help="Target process PID to monitor")
    parser.add_argument('--out', type=str, required=True, help="Path to output CSV")
    parser.add_argument('--interval', type=float, default=1.0, help="Sampling interval in seconds")
    args = parser.parse_args()
    run_sidecar(args.pid, args.out, args.interval)
