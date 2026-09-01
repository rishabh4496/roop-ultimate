"""Build a FaceSet V2 archive from a legacy V1 `.fsz`, WITH real detection.

The locked `harjot`/`gargee` archives are bare PNG bundles -- five root-level
`N.png` members and no metadata -- so `FaceSet.format_version` is 1 and
`identity_detail_for()` returns nothing. Every identity-detail arm run against
them measures a no-op, which is indistinguishable from "the feature does not
help".

`faceset_v2.migrate_legacy_fsz` is NOT the tool for this. It is lossless for
the reference PNGs, but without `faceset`/`images` it writes
`identity_details: {}` and `"migrated_without_detection": True` -- a V2 archive
that satisfies `format_version == 2` and still leaves the feature inert. That
is exactly the shape of defect this project keeps paying for: a gate that reads
PASS while the thing it gates never ran.

So this tool runs the SAME ingest the benchmark loader runs
(`two_face_video.load_library_faceset` -> `face_util.extract_face_images`),
then hands the detected faces and their images to `write_faceset_v2`, which
derives embeddings, pose bins, quality, appearance and the signed identity
detail residuals. The source PNGs are copied through byte-for-byte by that
writer's own re-encode of the selected images, so the identities are the
locked ones.

It writes to a NEW path by default and never overwrites the input.

    app/env/Scripts/python.exe app/tests/build_faceset_v2.py \
        --sources harjot,gargee --suffix _v2
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

LIB = os.path.join(APP, "facesets")


def verify(path):
    """Prove the archive is V2 AND that the feature can actually read it."""
    from roop.FaceSet import FaceSet
    from roop.faceset_v2 import read_faceset_archive

    metadata = read_faceset_archive(path)
    if metadata is None:
        return {"path": path, "ok": False, "why": "no V2 metadata member"}

    fs = FaceSet()
    fs.attach_v2_metadata(metadata)

    sources = metadata.get("sources") or []
    detail_ok, detail_missing = 0, []
    for i in range(len(sources)):
        try:
            detail = fs.identity_detail_for(i)
        except Exception as exc:                      # pragma: no cover
            detail, _ = None, exc
        if detail:
            detail_ok += 1
        else:
            detail_missing.append(i)

    compat = metadata.get("compatibility") or {}
    return {
        "path": path,
        "format_version": int(metadata.get("version", -1)),
        "schema": metadata.get("schema"),
        "sources": len(sources),
        "identity_detail_ok": detail_ok,
        "identity_detail_missing": detail_missing,
        # The tell for a detectionless migration. A V2 archive carrying this is
        # still inert for identity detail.
        "migrated_without_detection": bool(compat.get("migrated_without_detection")),
        "ok": (int(metadata.get("version", -1)) == 2
               and len(sources) > 0
               and detail_ok == len(sources)
               and not compat.get("migrated_without_detection")),
    }


def build(name, suffix, provider, swap_model, enhancer, mask_engine,
          cuda_device_id, overwrite=False):
    import two_face_video as tfv
    from roop.faceset_v2 import write_faceset_v2

    src = os.path.join(LIB, name + ".fsz")
    dst = os.path.join(LIB, name + suffix + ".fsz")
    if not os.path.exists(src):
        raise SystemExit(f"no such faceset: {src}")
    if os.path.exists(dst) and not overwrite:
        raise SystemExit(f"refusing to overwrite existing {dst} (use --overwrite)")
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("refusing to overwrite the locked source archive in place")

    fs = tfv.load_library_faceset(name)
    metadata = write_faceset_v2(dst, fs, fs.ref_images, source_name=name)
    return dst, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--provider", default=None,
                    help="default: config.yaml's provider")
    ap.add_argument("--enhancer", default="None",
                    help="irrelevant to the archive; kept so the ingest "
                         "pipeline initialises exactly as the bench does")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--cuda-device-id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    os.chdir(APP)
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = args.provider or str(cfg.provider)
    swap_model = str(cfg.swap_model)

    names = [s.strip() for s in args.sources.split(",") if s.strip()]

    if args.verify_only:
        for name in names:
            path = os.path.join(LIB, name + args.suffix + ".fsz")
            print(verify(path))
        return 0

    import angle_bench as ab
    ab.init_pipeline(provider, swap_model, args.enhancer, args.mask_engine,
                     cuda_device_id=args.cuda_device_id)

    failures = []
    for name in names:
        dst, metadata = build(name, args.suffix, provider, swap_model,
                              args.enhancer, args.mask_engine,
                              args.cuda_device_id, args.overwrite)
        report = verify(dst)
        print(f"[v2] {name} -> {os.path.basename(dst)}")
        print(f"     version={report['format_version']} schema={report['schema']} "
              f"sources={report['sources']} "
              f"identity_detail_ok={report['identity_detail_ok']} "
              f"missing={report['identity_detail_missing']} "
              f"migrated_without_detection={report['migrated_without_detection']}")
        if not report["ok"]:
            failures.append(report)

    if failures:
        print("\nFAILED: these archives would leave identity detail inert:")
        for r in failures:
            print("  ", r)
        return 1
    print("\nAll archives are V2 with a readable identity-detail residual for "
          "every selected source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
