#!/usr/bin/env python3
"""Does a SOURCE identity vector depend on in-plane (roll) rotation of the source photo?

Stage 0 keys its table on (yaw, pitch, roll) but every roll bin addresses the same vector, because nothing
that chooses a source reads roll (`pose_embedding_for_target` takes yaw and pitch; the swapper's
alignment removes the in-plane rotation of the TARGET before the net sees it). That is a claim about the
recogniser too: if buffalo_l / w600k_r50 embeddings of the same face moved with roll, a roll-aware gallery
would be a real feature and the shared vector would be an approximation.

Measured here: for each reference photo of a library faceset, rotate the photo in-plane by a ladder of
angles about the face centre (replicated border, canvas enlarged so no corner is cut), run the app's
detector + recogniser on the rotated image, and report cos(embedding at roll r, embedding at roll 0) and the
roll the five-point solver reads back (so a "rotation that was not detected" cannot pass as invariance).

    app/env/Scripts/python.exe app/tests/probe_source_roll_invariance.py --sources ashna,akansha,anushree
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                      # noqa: E402
import numpy as np              # noqa: E402

LADDER = (-60, -45, -30, -15, 15, 30, 45, 60)


def rotated(image, face, degrees):
    h, w = image.shape[:2]
    cx, cy = [(float(face.bbox[0]) + float(face.bbox[2])) / 2.0, (float(face.bbox[1]) + float(face.bbox[3])) / 2.0]
    side = int(np.ceil(np.hypot(h, w))) + 8
    canvas = cv2.copyMakeBorder(image, (side - h) // 2, side - h - (side - h) // 2, (side - w) // 2,
                                side - w - (side - w) // 2, cv2.BORDER_REPLICATE)
    c = (cx + (side - w) // 2, cy + (side - h) // 2)
    M = cv2.getRotationMatrix2D(c, degrees, 1.0)           # + = counter-clockwise in OpenCV
    return cv2.warpAffine(canvas, M, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def unit(v):
    v = np.asarray(v, np.float64).reshape(-1)
    return v / np.linalg.norm(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default='ashna,akansha,anushree')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import angle_bench as ab
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    ab.init_pipeline('tensorrt', cfg.get('swap_model', 'realswap'), 'None', mask)
    from roop.face_util import get_all_faces, solve_pose_5pt

    rows, per_angle = [], {r: [] for r in LADDER}
    for name in [s.strip() for s in args.sources.split(',') if s.strip()]:
        path = os.path.join(APP, 'facesets', name + '.fsz')
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if n.lower().endswith('.png')]
            images = [cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR) for n in members]
        for member, image in zip(members, images):
            faces = get_all_faces(image) or []
            if len(faces) != 1:
                continue
            face = faces[0]
            base = unit(face.embedding)
            base_pose = solve_pose_5pt(face.kps)
            row = {'source': name, 'member': member, 'base_roll_deg': None if base_pose is None else round(float(base_pose[2]), 1)}
            for r in LADDER:
                got = get_all_faces(rotated(image, face, r)) or []
                if len(got) != 1:
                    row[str(r)] = None
                    continue
                pose = solve_pose_5pt(got[0].kps)
                cos = float(unit(got[0].embedding) @ base)
                row[str(r)] = {'cos': round(cos, 4), 'read_roll_deg': None if pose is None else round(float(pose[2]), 1)}
                per_angle[r].append(cos)
            rows.append(row)
    summary = {str(r): ({'n': len(v), 'min_cos': round(min(v), 4), 'mean_cos': round(float(np.mean(v)), 4)} if v else None)
               for r, v in per_angle.items()}
    out = {'photos': len(rows), 'ladder_deg': list(LADDER), 'cos_vs_unrotated': summary, 'rows': rows}
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    if args.out:
        json.dump(out, open(args.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
