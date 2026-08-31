"""Phase 6 real-photo pose/source-bank evaluation.

This is an evaluation tool, not a production-path change.  Existing local V1
FaceSets are loaded with the same detector path as ``angle_bench`` and are
promoted only in memory to a V2 metadata view.  The original archives are
never rewritten.  The resulting source bank is then exercised through the
real ``roop.core.live_swap`` path with source-bank selection off and on.

The harness reports the axes actually represented by the supplied photographs
and does not manufacture pitch, inversion, or profile evidence by rotating
pixels.  A physical target GPU that is not present is recorded as ``pending``.

Example (run from ``app``):

    env/Scripts/python.exe tests/phase6_pose_quality.py \
        --target "RTX 4070" --source ashna --target-faceset harjot \
        --provider auto --rolls 0,90,180 --tag phase6_4070
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace


HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


def _pose_value(entry, key):
    try:
        value = (entry.get("geometry") or {}).get(key)
        return None if value is None else float(value)
    except (AttributeError, TypeError, ValueError):
        return None


def coverage_report(metadata):
    """Return measured pose coverage without inventing unsupported axes."""
    sources = list((metadata or {}).get("sources") or [])
    values = {key: sorted({round(value, 3) for value in
                           (_pose_value(entry, key) for entry in sources)
                           if value is not None})
              for key in ("yaw", "pitch", "roll")}
    return {
        "source_count": len(sources),
        "yaw_degrees": values["yaw"],
        "pitch_degrees": values["pitch"],
        "roll_degrees": values["roll"],
        "has_profile": any(abs(value) >= 75.0 for value in values["yaw"]),
        "has_pitch": any(abs(value) >= 20.0 for value in values["pitch"]),
        "has_inversion": any(abs(value) >= 135.0 for value in values["roll"]),
    }


def summarize_selection(rows):
    """Summarize source-choice error and 3D fallback requests."""
    rows = list(rows or [])
    valid = [row for row in rows
             if row.get("target_yaw") is not None and row.get("source_yaw") is not None]
    if not valid:
        return {"rows": len(rows), "valid": 0, "match_rate": None,
                "mean_yaw_error": None, "mean_pitch_error": None,
                "needs_3d_rate": None}
    matches = [row for row in valid
               if abs(float(row["yaw_error"])) <= 15.0
               and abs(float(row["pitch_error"])) <= 15.0]
    return {
        "rows": len(rows),
        "valid": len(valid),
        "match_rate": round(len(matches) / len(valid), 6),
        "mean_yaw_error": round(sum(abs(float(row["yaw_error"])) for row in valid) / len(valid), 6),
        "mean_pitch_error": round(sum(abs(float(row["pitch_error"])) for row in valid) / len(valid), 6),
        "needs_3d_rate": round(sum(bool(row.get("needs_3d")) for row in valid) / len(valid), 6),
    }


def _wrap_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _promote_v1_in_memory(faceset, source_name):
    """Attach V2 metadata to a loaded V1 FaceSet without touching its archive."""
    from roop.faceset_v2 import prepare_faceset_v2

    metadata, selected = prepare_faceset_v2(
        faceset.faces, faceset.ref_images, source_name=source_name,
        max_entries=max(1, len(faceset.faces)),
        max_per_bin=max(1, len(faceset.faces)),
    )
    if not selected:
        raise RuntimeError("no usable reference survived in-memory V2 preparation")
    faceset.faces = [faceset.faces[index] for index in selected]
    faceset.ref_images = [faceset.ref_images[index] for index in selected]
    faceset.attach_v2_metadata(metadata)
    return metadata


def _load_v1_and_promote(name, angle_bench):
    path = os.path.join(APP, "facesets", name + ".fsz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    faceset = angle_bench.load_faceset(path)
    metadata = _promote_v1_in_memory(faceset, name)
    return faceset, metadata


def _selection_rows(faceset, metadata, target_plates, rolls, angle_bench):
    from roop.pose_source_selector import estimate_target_pose, select_pose_aware_source
    from roop.faceset_v2 import measure_lighting

    rows = []
    for plate_index, plate in enumerate(target_plates):
        square, background = angle_bench.prepare_plate(plate)
        if square is None:
            continue
        upright = angle_bench.unroll(
            angle_bench.roll_frame(square, background, 0), 0, background)
        target_face = angle_bench.biggest_face(upright)
        if target_face is None:
            continue
        base_pose = estimate_target_pose(target_face, frame_shape=upright.shape)
        lighting = measure_lighting(upright, getattr(target_face, "bbox", None))
        for roll in rolls:
            target_pose = replace(
                base_pose, roll=float(roll),
                inverted=abs(_wrap_degrees(roll)) >= 135.0,
            )
            result = select_pose_aware_source(
                metadata, target_pose, appearance=lighting,
                expression=target_pose.expression,
            )
            source = metadata["sources"][result.index]
            target_yaw = float(target_pose.yaw)
            target_pitch = float(target_pose.pitch)
            source_yaw = _pose_value(source, "yaw")
            source_pitch = _pose_value(source, "pitch")
            rows.append({
                "plate": plate_index,
                "roll": int(roll),
                "target_yaw": round(target_yaw, 6),
                "target_pitch": round(target_pitch, 6),
                "source_index": int(result.index),
                "source_yaw": source_yaw,
                "source_pitch": source_pitch,
                "yaw_error": round(_wrap_degrees(source_yaw - target_yaw), 6)
                if source_yaw is not None else None,
                "pitch_error": round(source_pitch - target_pitch, 6)
                if source_pitch is not None else None,
                "reason": result.reason,
                "needs_3d": bool(result.needs_3d),
                "switched": bool(result.switched),
            })
    return rows


def _quality_arm(g, options, faceset, target_plates, tag, outdir, angle_bench):
    started = time.perf_counter()
    rows = []
    # angle_bench owns the exact plate preparation, detector, live_swap, and
    # quality grades used by the established pose evaluation tool.
    rows = angle_bench.sweep(
        g, options, faceset, target_plates, tag, outdir,
        rolls=options._phase6_rolls, sheet_rolls=set(), control=False,
        source_identity=getattr(faceset, "identity_embedding", None),
    )
    elapsed = time.perf_counter() - started
    detected = sum(int(row.get("detected", 0)) for row in rows)
    summary = {
        "rows": len(rows),
        "detected": detected,
        "detection_rate": round(detected / len(rows), 6) if rows else None,
        "elapsed_seconds": round(elapsed, 3),
        "quality": angle_bench.summarize(rows),
    }
    return summary, rows


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _target_present(target, device_id):
    from hardware_probe import query_gpus, target_on_device
    rows, _raw = query_gpus()
    return target_on_device(target, rows, device_id)[0]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=("RTX 3060", "RTX 4070"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--source", default="ashna")
    parser.add_argument("--target-faceset", default="harjot")
    parser.add_argument("--rolls", default="0,90,180")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", default=os.path.join(APP, "output", "phase6_pose_quality"))
    args = parser.parse_args(argv)
    args.out = os.path.abspath(args.out)
    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)
    rolls = [int(value.strip()) for value in args.rolls.split(",") if value.strip()]

    report = {
        "version": 1,
        "phase": 6,
        "target": args.target,
        "device_id": args.device_id,
        "provider_requested": args.provider,
        "source": args.source,
        "target_faceset": args.target_faceset,
        "rolls": rolls,
        "status": "pending",
    }
    if not _target_present(args.target, args.device_id):
        report["pending_reason"] = "requested GPU is not physically present"
        report["required_command"] = (
            "python tests/phase6_pose_quality.py --target %r --device-id %d --tag %s"
            % (args.target, args.device_id, args.tag))
        with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps(report, indent=2))
        return 0

    import angle_bench

    g = angle_bench.init_pipeline(
        args.provider, "realswap", "None", "None", cuda_device_id=args.device_id)
    source, source_metadata = _load_v1_and_promote(args.source, angle_bench)
    _target_unused, target_metadata = _load_v1_and_promote(args.target_faceset, angle_bench)
    target_plates = angle_bench.plates(
        os.path.join(APP, "facesets", args.target_faceset + ".fsz"))
    report["coverage"] = {
        "source": coverage_report(source_metadata),
        "target": coverage_report(target_metadata),
    }
    report["coverage_note"] = (
        "The local archives provide measured yaw/roll plates only; pitch and "
        "inversion remain pending until photographs with those poses are supplied."
    )
    selection_rows = _selection_rows(
        source, source_metadata, target_plates, rolls, angle_bench)
    report["selection"] = summarize_selection(selection_rows)
    _write_csv(os.path.join(outdir, "selection.csv"), selection_rows)

    # Run both orders.  The first arm pays any remaining process/session setup;
    # an ordered A/B would otherwise turn initialization cost into a false
    # source-bank performance result.
    order_results = {}
    for order_name, enabled_order in (
            ("off_on", (False, True)), ("on_off", (True, False))):
        order_results[order_name] = {}
        for enabled in enabled_order:
            arm_name = "source_bank_on" if enabled else "source_bank_off"
            options = angle_bench.build_options(
                g, "realswap", "None", source_bank=enabled)
            # Keep the established sweep API unchanged while passing this
            # harness's explicit roll set through a private, local attribute.
            options._phase6_rolls = rolls
            arm_dir = os.path.join(outdir, order_name, arm_name)
            os.makedirs(arm_dir, exist_ok=True)
            summary, quality_rows = _quality_arm(
                g, options, source, target_plates,
                order_name + "_" + arm_name, arm_dir, angle_bench)
            order_results[order_name][arm_name] = summary
            _write_csv(os.path.join(arm_dir, "quality.csv"), quality_rows)

    report["orders"] = order_results
    report["arms"] = {}
    for arm_name in ("source_bank_off", "source_bank_on"):
        summaries = [order_results[order][arm_name]
                     for order in ("off_on", "on_off")]
        total_rows = sum(item["rows"] for item in summaries)
        total_detected = sum(item["detected"] for item in summaries)
        report["arms"][arm_name] = {
            "rows": total_rows,
            "detected": total_detected,
            "detection_rate": round(total_detected / total_rows, 6)
            if total_rows else None,
            "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in summaries), 3),
            "order_balanced": True,
        }

    report["status"] = "complete"
    report["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
