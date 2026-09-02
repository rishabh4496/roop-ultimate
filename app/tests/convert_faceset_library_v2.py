"""Convert every archive in the faceset library to the current FaceSet V2.

Why this exists rather than `faceset_v2.migrate_legacy_fsz`: that function is
lossless for the reference PNGs, but called without `faceset`/`images` it
writes `identity_details: {}` and stamps `migrated_without_detection: True` --
a V2 archive that satisfies `format_version == 2` while every feature reading
it stays inert. That is the exact shape of defect this project keeps paying
for, so this tool runs the SAME real-detection ingest the application does
(`two_face_video.load_library_faceset` -> `face_util.extract_face_images`) and
hands the detected faces to `write_faceset_v2`.

Safety contract, in order:

  1. nothing is written until the replacement archive has been built AND
     verified (V2 schema, validates, identity detail readable for every
     selected source);
  2. the replacement lands on a temp path that does NOT end in `.fsz`, so a
     half-written file can never appear in the library listing;
  3. `os.replace` swaps it in atomically, on the same filesystem;
  4. an archive that fails at any step is LEFT EXACTLY AS IT WAS and reported.

`--dry-run` performs every step including the build and verify, and skips only
the replace, so a rehearsal exercises the same code the real run does.

    app/env/Scripts/python.exe app/tests/convert_faceset_library_v2.py --dry-run
    app/env/Scripts/python.exe app/tests/convert_faceset_library_v2.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LIB = os.path.join(APP, "facesets")


def archive_state(path):
    """Report `(version, png_count, missing_keys)` without importing roop."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            pngs = [n for n in names if n.lower().endswith(".png")]
            if "metadata.json" not in names:
                return 1, len(pngs), ["metadata.json"]
            metadata = json.loads(zf.read("metadata.json"))
            missing = [k for k in ("default_embedding", "pose_bins", "dermal_patch")
                       if k not in metadata]
            if (metadata.get("compatibility") or {}).get("migrated_without_detection"):
                missing.append("detection")
            return int(metadata.get("version", -1)), len(pngs), missing
    except Exception:
        return None, 0, ["unreadable"]


def verify(path):
    """Prove the built archive is V2 AND that its metadata is actually usable."""
    from roop.FaceSet import FaceSet
    from roop.faceset_v2 import read_faceset_archive

    metadata = read_faceset_archive(path)
    if metadata is None:
        raise ValueError("no V2 metadata member")
    if int(metadata.get("version", -1)) != 2:
        raise ValueError(f"version is {metadata.get('version')}, not 2")
    if (metadata.get("compatibility") or {}).get("migrated_without_detection"):
        raise ValueError("archive was migrated without detection")

    sources = metadata.get("sources") or []
    if not sources:
        raise ValueError("archive has no selected sources")

    faceset = FaceSet()
    faceset.attach_v2_metadata(metadata)
    # Read back through the same accessors the runtime uses. A key that is
    # present but unreadable is the failure mode a shape check alone misses.
    for i in range(len(sources)):
        if not faceset.identity_detail_for(i):
            raise ValueError(f"source {i} has no readable identity detail")
    if faceset.default_embedding is None:
        raise ValueError("default_embedding is unreadable")
    if not faceset.pose_bins:
        raise ValueError("pose_bins is empty")
    return metadata


def convert(name, dry_run):
    """Build, verify and (unless `dry_run`) swap in one archive."""
    import two_face_video as tfv
    from roop.faceset_v2 import write_faceset_v2

    src = os.path.join(LIB, name + ".fsz")
    # Deliberately not a `.fsz` suffix: the library listing globs `*.fsz`, so a
    # temp with that extension would surface as a phantom entry if this died
    # between write and replace.
    tmp = os.path.join(LIB, name + ".v2-pending")

    before_version, before_pngs, _ = archive_state(src)
    faceset = tfv.load_library_faceset(name)
    detected = len(faceset.faces)

    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        metadata = write_faceset_v2(tmp, faceset, faceset.ref_images, source_name=name)
        verify(tmp)
        selected = len(metadata.get("sources") or [])
        rejected = metadata.get("rejected") or []
        if dry_run:
            os.remove(tmp)
        else:
            os.replace(tmp, src)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

    return {
        "name": name,
        "before_version": before_version,
        "png_members": before_pngs,
        "detected": detected,
        "selected": selected,
        "rejected": [{"reason": r.get("reason"), "index": r.get("index")} for r in rejected],
        "pose_cells": sorted((metadata.get("pose_bins") or {}).keys()),
        "dermal_basis": (metadata.get("dermal_patch") or {}).get("frontal_basis"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="",
                    help="comma-separated stems; default is every .fsz in the library")
    ap.add_argument("--apply", action="store_true",
                    help="actually replace the archives (default is a rehearsal)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--provider", default=None, help="default: config.yaml's provider")
    ap.add_argument("--cuda-device-id", type=int, default=0)
    args = ap.parse_args()

    dry_run = not args.apply
    os.chdir(APP)

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = args.provider or str(cfg.provider)

    if args.names.strip():
        names = [s.strip() for s in args.names.split(",") if s.strip()]
    else:
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(LIB)
                       if f.lower().endswith(".fsz"))
    if not names:
        print("no archives found in", LIB)
        return 1

    print(f"library : {LIB}")
    print(f"archives: {len(names)}")
    print(f"mode    : {'DRY RUN (nothing is replaced)' if dry_run else 'APPLY (archives replaced in place)'}")
    print(f"provider: {provider}\n")

    import angle_bench as ab
    # Through init_pipeline, never a bare process: without the app's init the
    # TensorRT DLLs are off PATH and ORT falls back to CPU silently.
    ab.init_pipeline(provider, str(cfg.swap_model), "None", "None",
                     cuda_device_id=args.cuda_device_id)

    done, failed = [], []
    for i, name in enumerate(names, 1):
        try:
            row = convert(name, dry_run)
            done.append(row)
            flag = "" if row["selected"] == row["detected"] else f"  <-- dropped {row['detected'] - row['selected']}"
            print(f"[{i:2d}/{len(names)}] {name:24s} v{row['before_version']} -> v2  "
                  f"faces {row['detected']} -> {row['selected']}  "
                  f"cells={len(row['pose_cells'])}{flag}")
        except Exception as exc:
            failed.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:2d}/{len(names)}] {name:24s} FAILED, left untouched: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print(f"\nconverted {len(done)}, failed {len(failed)}")
    dropped = [r for r in done if r["selected"] != r["detected"]]
    if dropped:
        print("\narchives where the pre-screens dropped a reference:")
        for r in dropped:
            reasons = {}
            for item in r["rejected"]:
                reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
            print(f"  {r['name']:24s} {r['detected']} -> {r['selected']}   {reasons}")
    if failed:
        print("\nFAILED (originals untouched):")
        for r in failed:
            print(f"  {r['name']:24s} {r['error']}")
    if dry_run:
        print("\nrehearsal only - nothing was replaced. Re-run with --apply.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
