"""Sweep `detail_transfer_strength` against the skin-texture gap it exists to close.

    env/Scripts/python.exe tests/sweep_detail_transfer.py \
        --clip "G:/pinokio/roop-keep/inverted/s1.mp4" --source harjot \
        --values 0,0.4,0.7,1.0

WHY A FULL RENDER PER ARM. Detail transfer runs in face-template space and the
face is pasted back at ~300 px, so a crop-level sweep cannot see what survives —
that is exactly the trap that made UltraMax's texture restore measure as a total
no-op at sigma 1.1 (see tests/calibrate_ultramax_texture.py). Every number here
is read off rendered frames.

WHAT IS GRADED, and why each one is here rather than just the texture number.
The whole point of this stage is to raise skin texture toward the plate, so
grading it on texture ALONE would endorse any value that goes up — including
values that destroy the thing the swap is for. The four columns pull against
each other on purpose:

  skin texture   high-frequency std on the flat part of the swapped face,
                 over the plate's own. 100% = the swapped skin carries as much
                 micro-texture as the camera recorded. This is the objective.

  edge energy    Laplacian magnitude where the PLATE has real structure, over
                 the plate's. Detail transfer has a Sobel edge-stop gate meant
                 to keep the target's eyelid creases and lip margins off the
                 swapped face; if this climbs, that gate is leaking and the
                 output is being drawn on rather than textured.

  flicker        mean |frame_t - frame_t-1| inside the face box. The plate's
                 high frequency includes its SENSOR NOISE, which is temporally
                 independent — so injecting more of it can buy texture and pay
                 for it in shimmer. Nothing in the texture number would show
                 that.

  identity       cosine distance from the output face to the SOURCE faceset
                 mean (want it low) and to the TARGET person's own mean (want
                 it high). The UI's own help text warns that too much detail
                 transfer "reintroduces the target's skin identity"; this is
                 that warning, measured. `d_target - d_source` is the margin,
                 and it collapsing is the failure this sweep must catch.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_enhancers_video as C     # applies the perf env at import
import cv2
import numpy as np

GRADE_FRAMES = tuple(range(150, 1750, 100))     # 16 frames, spread over the clip


# ── grading ──────────────────────────────────────────────────────────────────

def _face_mask(rendered, plate):
    """Where the pipeline actually changed the frame — the swapped face."""
    d = cv2.GaussianBlur(cv2.cvtColor(cv2.absdiff(rendered, plate),
                                      cv2.COLOR_BGR2GRAY), (0, 0), 3)
    return d > 12


def skin_patches(kps, shape):
    """Cheeks and forehead, anchored on the PLATE's 5 landmarks.

    THE SKIN MASK MUST NOT BE DEFINED BY EITHER IMAGE'S HIGH-FREQUENCY CONTENT.
    Three edge-percentile definitions were tried and gave three contradictory
    answers on the same footage — 34%, 283% and 500% of the plate — because each
    one selects pixels according to the very quantity being measured:

        edge < 45th pct of the RENDERED arm   biases toward pixels the
                                              treatment touched least, so it
                                              cancels the effect and reads low
        edge < 45th pct of the PLATE          selects the plate's near-featureless
                                              pixels, collapsing the denominator
                                              and reading absurdly high
        edge < 75th pct of the PLATE          same bias, milder

    None of them measures "skin". These patches do: two cheeks and a forehead,
    placed from the 5-point landmarks of the face detected in the PLATE, so the
    window is identical for every arm and independent of what any arm did to the
    pixels inside it. Cheeks and forehead are also the regions where dermal
    micro-texture actually lives, which is what this whole question is about.

    kps order is insightface's: left eye, right eye, nose, left mouth, right
    mouth.
    """
    m = np.zeros(shape[:2], np.uint8)
    le, re, nose, lm, rm = [np.asarray(k, np.float32) for k in kps[:5]]
    inter = float(np.linalg.norm(re - le))
    if inter < 12:
        return m.astype(bool)
    r = int(max(4, inter * 0.20))
    eye_mid = (le + re) * 0.5
    up = eye_mid - nose                       # points out of the face, roughly
    centres = [
        (le + lm) * 0.5,                      # left cheek
        (re + rm) * 0.5,                      # right cheek
        eye_mid + up * 0.55,                  # forehead
    ]
    for c in centres:
        cx, cy = int(round(float(c[0]))), int(round(float(c[1])))
        if 0 <= cx < shape[1] and 0 <= cy < shape[0]:
            cv2.circle(m, (cx, cy), r, 255, -1)
    return m.astype(bool)


def grade_arm(path, clip, source_mean, target_mean, frames=GRADE_FRAMES,
              ref_path=None):
    """Grade one arm. `ref_path` fixes the masks across arms — see below.

    THE MASK MUST NOT MOVE WITH THE TREATMENT. The first version of this
    function chose the skin mask as `edge < 45th percentile` of the RENDERED
    arm, recomputed per arm. Injecting texture raises that arm's edge map, so
    the mask slid onto whatever pixels the treatment had touched LEAST, and the
    measurement partly cancelled the thing it was measuring. It reported the
    frame-space probe at gain 2.0 as +1.2 points when a fixed mask put it at
    +9.2. Both masks are now defined on material that every arm shares: the
    skin and structure masks come from the PLATE, and the face mask comes from
    one reference arm (`ref_path`, default this arm) rather than from each.
    """
    from roop.utilities import compute_cosine_distance as cos
    import roop.face_util as fu

    cap_r, cap_p = cv2.VideoCapture(path), cv2.VideoCapture(clip)
    cap_ref = cv2.VideoCapture(ref_path) if ref_path else None
    rows = []
    for fi in frames:
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, fi)
        cap_p.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok_r, fr = cap_r.read()
        ok_p, fp = cap_p.read()
        if not (ok_r and ok_p):
            continue
        if cap_ref is not None:
            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok_ref, f_ref = cap_ref.read()
            if not ok_ref:
                continue
        else:
            f_ref = fr
        face = _face_mask(f_ref, fp)
        if face.sum() < 3000:
            continue
        pf = fu.get_all_faces(fp) or []
        if not pf:
            continue
        plate_skin = skin_patches(
            max(pf, key=lambda x: x.bbox[2] - x.bbox[0]).kps, fp.shape)
        if plate_skin.sum() < 400:
            continue

        g_r = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g_p = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hf_r = g_r - cv2.GaussianBlur(g_r, (0, 0), 1.1)
        hf_p = g_p - cv2.GaussianBlur(g_p, (0, 0), 1.1)
        ed_r = np.abs(cv2.Laplacian(g_r, cv2.CV_32F, ksize=3))
        ed_p = np.abs(cv2.Laplacian(g_p, cv2.CV_32F, ksize=3))

        # Skin is geometric (cheeks + forehead off the plate's landmarks);
        # structure is where the PLATE has real edges. Neither depends on what
        # any arm did to the pixels. See skin_patches.
        skin = face & plate_skin
        struct = face & (ed_p > np.percentile(ed_p[face], 75))
        if skin.sum() < 400 or struct.sum() < 500:
            continue

        # Flicker: the next frame, inside the face box only.
        ok_r2, fr2 = cap_r.read()
        flick = float('nan')
        if ok_r2:
            ys, xs = np.where(face)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            flick = float(np.abs(fr2[y0:y1, x0:x1].astype(np.int16)
                                 - fr[y0:y1, x0:x1].astype(np.int16)).mean())

        d_src = d_tgt = float('nan')
        try:
            faces = fu.get_all_faces(fr) or []
            if faces:
                f = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])
                d_src = float(cos(f.embedding, source_mean))
                d_tgt = float(cos(f.embedding, target_mean))
        except Exception:
            pass

        rows.append((float(hf_p[skin].std()), float(hf_r[skin].std()),
                     float(ed_p[struct].mean()), float(ed_r[struct].mean()),
                     flick, d_src, d_tgt))
    cap_r.release()
    cap_p.release()
    if cap_ref is not None:
        cap_ref.release()
    if not rows:
        return None
    a = np.array(rows)
    return {
        'n': len(rows),
        'plate_hf': float(np.nanmean(a[:, 0])),
        'out_hf': float(np.nanmean(a[:, 1])),
        'skin': float(np.nanmean(a[:, 1]) / np.nanmean(a[:, 0])),
        'edge': float(np.nanmean(a[:, 3]) / np.nanmean(a[:, 2])),
        'flicker': float(np.nanmean(a[:, 4])),
        'd_src': float(np.nanmean(a[:, 5])),
        'd_tgt': float(np.nanmean(a[:, 6])),
    }


def means(clip, source_name, frames=GRADE_FRAMES):
    """The source faceset's mean embedding, and the TARGET person's own.

    The target mean comes from the untouched footage, so "did the output drift
    back toward the person who was actually filmed" is answerable.
    """
    import roop.face_util as fu
    fs = C.load_library_faceset(source_name)
    src = np.mean([f.embedding for f in fs.faces], axis=0)

    cap = cv2.VideoCapture(clip)
    embs = []
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        faces = fu.get_all_faces(fr) or []
        if faces:
            embs.append(max(faces, key=lambda x: x.bbox[2] - x.bbox[0]).embedding)
    cap.release()
    if not embs:
        raise SystemExit("[sweep] no face found in the plate — cannot build a "
                         "target mean, and identity would silently read as nan")
    return src, np.mean(embs, axis=0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=r"G:/pinokio/roop-keep/inverted/s1.mp4")
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--enhancer", default="UltraMax")
    ap.add_argument("--values", default="0,0.4,0.7,1.0")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml
    with open(os.path.join(APP, 'config.yaml')) as f:
        cfg = yaml.safe_load(f) or {}
    swapper = cfg.get('swap_model', 'realswap')
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg',
            'None': 'None'}.get(cfg.get('mask_engine', 'RealityUX'),
                                cfg.get('mask_engine'))
    threads = args.threads or int(cfg.get('max_threads', 16))
    values = [float(v) for v in args.values.split(',') if v.strip() != '']

    out_base = args.out or os.path.join(APP, "output", "detail_sweep")
    os.makedirs(out_base, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("=" * 78)
    print(f"[sweep] detail_transfer_strength {values} on "
          f"{os.path.basename(args.clip)} ({nframes} frames), "
          f"{args.enhancer} / {swapper} / {mask}, live value "
          f"{cfg.get('detail_transfer_strength')}")
    print("=" * 78, flush=True)

    arms = []
    for v in values:
        d = os.path.join(out_base, f"dt_{v:g}".replace('.', 'p'))
        t0 = time.time()
        path, elapsed = C.render(args.clip, args.source, args.enhancer, d,
                                 swapper, mask, threads,
                                 overrides={'detail_transfer_strength': v})
        fps = nframes / elapsed if elapsed > 0 else 0.0
        print(f"[sweep] dt={v:g}: {elapsed:.1f}s -> {fps:.2f} fps  ({path})",
              flush=True)
        arms.append((v, path, elapsed, fps))

    print("\n[sweep] grading against the original footage...", flush=True)
    src_mean, tgt_mean = means(args.clip, args.source)

    results = []
    ref = arms[0][1]        # one arm defines the face mask for all of them
    for v, path, elapsed, fps in arms:
        r = grade_arm(path, args.clip, src_mean, tgt_mean, ref_path=ref)
        if r is None:
            print(f"[sweep] dt={v:g}: no gradeable frames")
            continue
        r.update(dt=v, fps=fps)
        results.append(r)

    print()
    print("=" * 96)
    print(f"  {'dt':>5} {'fps':>6} | {'plate hf':>8} {'out hf':>7} {'skin tex':>9} "
          f"{'edge':>7} {'flicker':>8} | {'d_source':>9} {'d_target':>9} {'margin':>7}")
    print("  " + "-" * 92)
    for r in results:
        print(f"  {r['dt']:5.2f} {r['fps']:6.2f} | {r['plate_hf']:8.3f} "
              f"{r['out_hf']:7.3f} {r['skin']:8.1%} "
              f"{r['edge']:6.1%} {r['flicker']:8.3f} | {r['d_src']:9.4f} "
              f"{r['d_tgt']:9.4f} {r['d_tgt'] - r['d_src']:7.4f}")
    print("=" * 96)
    print("  skin tex / edge: 100% = matches the camera. Want skin UP toward 100")
    print("  without edge climbing (that is the edge-stop gate leaking) and")
    print("  without flicker climbing (that is the plate's sensor noise) and")
    print("  without the identity margin collapsing (that is the target's skin")
    print("  identity coming back, which is what the UI help text warns about).")


if __name__ == '__main__':
    main()
