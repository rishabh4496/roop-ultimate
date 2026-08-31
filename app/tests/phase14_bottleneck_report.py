"""Compare Phase 14 detailed profiles without changing runtime settings.

The report is intentionally a separate analysis tool: it never promotes a
faster candidate, changes a provider, or treats missing measurements as zero.
Use baseline_controlled.py with ROOP_PROFILE_DETAIL=1 to create the inputs.
"""

import argparse
import json
import sys

REQUIRED_STAGES = (
    "detection", "tracking", "alignment", "faceset_lookup", "swap",
    "expression_analysis", "occlusion_analysis", "detail_restoration",
    "enhancement", "lighting", "mask", "blending", "encoding",
)


def _profile(record):
    value = record.get("detailed_stage_profile", record)
    return value if isinstance(value, dict) else {}


def _stage_map(record):
    return _profile(record).get("canonical_stages", {})


def compare(before, after=None):
    """Return a JSON-serialisable bottleneck and A/B evidence report."""
    before_stages = _stage_map(before)
    rows = []
    gaps = []
    for name in REQUIRED_STAGES:
        item = before_stages.get(name, {})
        if item.get("status") != "measured":
            gaps.append(name)
        rows.append({
            "stage": name,
            "status": item.get("status", "missing"),
            "cpu_ms_total": item.get("cpu_ms_total"),
            "gpu_event_ms_total": item.get("gpu_event_ms_total"),
            "gpu_sync_window_ms_total": item.get("gpu_sync_window_ms_total"),
            "sync_ms_total": item.get("sync_ms_total"),
            "calls": item.get("calls"),
            "alloc_peak_mb": item.get("alloc_peak_mb"),
            "steady_state_allocated_mb": item.get("steady_state_allocated_mb"),
            "steady_state_reserved_mb": item.get("steady_state_reserved_mb"),
            "transfer_h2d_bytes": item.get("transfer_h2d_bytes"),
            "transfer_d2h_bytes": item.get("transfer_d2h_bytes"),
            "transfer_attribution": item.get("transfer_attribution"),
        })
    rows.sort(key=lambda row: float(row.get("cpu_ms_total") or 0.0), reverse=True)
    result = {
        "schema": "roop-phase14-bottleneck-report-v1",
        "workload": before.get("workload"),
        "measurement": {
            "profile_schema": _profile(before).get("schema"),
            "mode": _profile(before).get("mode"),
            "required_stage_gaps": gaps,
            "host_device_transfer_note": _profile(before).get(
                "host_device_transfer_note"),
            "full_card_memory_note": _profile(before).get(
                "full_card_memory_note"),
        },
        "bottlenecks_by_cpu_time": rows,
        "optimization_decision": (
            "Use only measured rows above; no setting is promoted by this report."
            if not gaps else
            "INCOMPLETE: collect the missing required stages before optimizing."
        ),
    }
    if after is not None:
        after_stages = _stage_map(after)
        ab = []
        for row in rows:
            name = row["stage"]
            old = before_stages.get(name, {})
            new = after_stages.get(name, {})
            old_cpu = old.get("cpu_ms_total")
            new_cpu = new.get("cpu_ms_total")
            ab.append({
                "stage": name,
                "before_cpu_ms_total": old_cpu,
                "after_cpu_ms_total": new_cpu,
                "delta_cpu_ms": (new_cpu - old_cpu)
                if isinstance(old_cpu, (int, float)) and isinstance(new_cpu, (int, float))
                else None,
            })
        result["ab"] = ab
        result["same_workload"] = before.get("workload") == after.get("workload")
        result["quality_guard"] = {
            "before_wrong_faceset": before.get("run", {}).get("wrong_faceset"),
            "after_wrong_faceset": after.get("run", {}).get("wrong_faceset"),
            "before_fps": before.get("run", {}).get("fps"),
            "after_fps": after.get("run", {}).get("fps"),
            "requires_visual_identity_review": True,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after")
    parser.add_argument("--out")
    args = parser.parse_args()
    with open(args.before, encoding="utf-8") as fh:
        before = json.load(fh)
    after = None
    if args.after:
        with open(args.after, encoding="utf-8") as fh:
            after = json.load(fh)
    report = compare(before, after)
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(output + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
