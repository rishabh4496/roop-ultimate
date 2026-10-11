"""Analyze Telemetry CSV and emit summary table.
"""
import sys
import csv

def analyze(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    if not reader:
        print("Empty telemetry")
        return

    n = len(reader)
    # Split into Early (first 25%), Mid (middle 50%), Late (last 25%)
    q1 = n // 4
    q3 = (3 * n) // 4

    def avg(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, '', 'N/A')]
        return sum(vals) / len(vals) if vals else 0.0

    def peak(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, '', 'N/A')]
        return max(vals) if vals else 0.0

    early = reader[:q1] if q1 > 0 else reader
    mid = reader[q1:q3] if q3 > q1 else reader
    late = reader[q3:] if q3 < n else reader

    print(f"Telemetry Summary for {csv_file} ({n} samples, {float(reader[-1]['elapsed_s']):.1f}s):")
    print(f"  GPU Util %: Early={avg(early, 'gpu_util_pct'):.1f}%, Mid={avg(mid, 'gpu_util_pct'):.1f}%, Late={avg(late, 'gpu_util_pct'):.1f}% (Peak={peak(reader, 'gpu_util_pct'):.1f}%)")
    print(f"  SM Clock:   Early={avg(early, 'clocks_sm_mhz'):.0f}MHz, Mid={avg(mid, 'clocks_sm_mhz'):.0f}MHz, Late={avg(late, 'clocks_sm_mhz'):.0f}MHz")
    print(f"  VRAM Used:  Early={avg(early, 'gpu_mem_used_mb'):.0f}MB, Mid={avg(mid, 'gpu_mem_used_mb'):.0f}MB, Late={avg(late, 'gpu_mem_used_mb'):.0f}MB (Peak={peak(reader, 'gpu_mem_used_mb'):.0f}MB)")
    print(f"  Host RSS:   Early={avg(early, 'proc_rss_mb'):.0f}MB, Mid={avg(mid, 'proc_rss_mb'):.0f}MB, Late={avg(late, 'proc_rss_mb'):.0f}MB (Peak={peak(reader, 'proc_rss_mb'):.0f}MB)")
    print(f"  Free RAM:   Early={avg(early, 'sys_free_ram_mb'):.0f}MB, Mid={avg(mid, 'sys_free_ram_mb'):.0f}MB, Late={avg(late, 'sys_free_ram_mb'):.0f}MB")
    print(f"  GPU Temp:   Early={avg(early, 'gpu_temp_c'):.1f}C, Mid={avg(mid, 'gpu_temp_c'):.1f}C, Late={avg(late, 'gpu_temp_c'):.1f}C (Max={peak(reader, 'gpu_temp_c'):.1f}C)")
    print(f"  GPU Power:  Early={avg(early, 'gpu_power_w'):.1f}W, Mid={avg(mid, 'gpu_power_w'):.1f}W, Late={avg(late, 'gpu_power_w'):.1f}W")

if __name__ == '__main__':
    analyze(sys.argv[1])
