"""Does a foreign object in front of the face survive the swap?

WHY A SYNTHETIC OCCLUDER. "Foreign-object occlusion works" is the campaign's
highest-priority behaviour and had no run behind it on this target. The clips
here that LOOK like occlusion (a face lost on 8-29% of sampled frames) carry no
ground truth: a face can disappear because a hand crossed it, or because it
turned away, or left frame, and nothing in the file says which. Grading against
a guess is how this project has twice published a conclusion it had to withdraw.

So the occluder is composited BY THIS HARNESS, which means its mask is known
exactly, per frame, to the pixel. The question then has a factual answer:

    after the swap, do the occluder's pixels still show the occluder?

If the pipeline paints the swapped face over the object, those pixels change. If
the mask engine excludes the object, they do not. There is no interpretation
step.

WHAT THIS DOES AND DOES NOT TEST. It tests the pipeline's ownership and masking
under a hard, opaque, moving occluder covering a known part of the face, through
the full approach -> touch -> cover -> cross -> leave cycle. It does NOT prove
behaviour on a real hand: a flat synthetic patch is out of distribution for
XSeg and BiSeNet, which were trained on real occluders, so a failure here is
strictly weaker evidence than a failure on real footage while a PASS here is
strictly weaker evidence than a pass on real footage. Real-occluder validation
stays open, and this file does not claim to close it.

The occluder is textured rather than flat on purpose -- a uniform block is the
easiest possible case for any segmentation and the least representative.

Frames are swapped in process through `roop.core.live_swap`, not rendered to a
file: an encoder would put a lossy layer between the measurement and the thing
being measured, and this comparison is exact.

    env/Scripts/python.exe tests/occlusion_ground_truth.py
    env/Scripts/python.exe tests/occlusion_ground_truth.py --frames 24 --out d/
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                       # noqa: E402
import angle_bench as ab              # noqa: E402
from two_face_video import load_library_faceset, map_mask_engine  # noqa: E402


# Three appearances, because "does the mask engine exclude a foreign object"
# and "does it exclude a foreign object THAT LOOKS LIKE THIS" are different
# questions, and only the second one is answerable without real footage.
#
#   texture  arbitrary low-frequency colour -- maximally unlike skin
#   skin     warm mid-tone, the hand case, and the hardest for a segmenter
#            keyed on skin colour: it looks like the thing it is covering
#   dark     a phone/microphone: low luma, low saturation, high contrast edge
OCCLUDER_STYLES = ("texture", "skin", "dark")


def make_occluder(shape, rng, style="texture"):
    """A hand-sized opaque patch. Textured, never flat.

    A uniform block is the easiest case any segmentation can face and the least
    like a real occluder, so every style carries low-frequency variation.
    """
    h, w = shape
    if style == "skin":
        # BGR. Warm mid-tone with the shading variation a hand has.
        base = rng.integers(0, 26, size=(6, 6, 3)).astype(np.int16)
        base += np.array([120, 150, 195], dtype=np.int16)
    elif style == "dark":
        base = rng.integers(18, 52, size=(6, 6, 3)).astype(np.int16)
    else:
        base = rng.integers(60, 190, size=(6, 6, 3)).astype(np.int16)
    base = np.clip(base, 0, 255).astype(np.uint8)
    patch = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(patch, (0, 0), max(w, h) / 24.0)


def occluder_mask(shape, cx, cy, rx, ry):
    """A filled ellipse: hard-edged, exactly known, no partial alpha.

    Partial alpha would make "did the face paint over it" a question of degree
    rather than of fact, and the point of this harness is the factual version.
    """
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
    return m


def sweep_positions(face_box, frames, w):
    """Approach -> touch -> cover -> cross -> leave, across the face.

    Starts and ends clear of the face so the run contains its own before/after:
    a pipeline that simply never paints anywhere would pass a cover-only test.
    """
    x0, y0, x1, y1 = face_box
    fw, fh = x1 - x0, y1 - y0
    cy = y0 + fh * 0.55
    rx, ry = fw * 0.30, fh * 0.34
    start = x0 - fw * 0.9
    end = x1 + fw * 0.9
    for i in range(frames):
        t = i / max(frames - 1, 1)
        yield float(np.clip(start + t * (end - start), -rx, w + rx)), cy, rx, ry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=fixtures.clip("single/s5.mp4"),
                    help="any clip with one clear, reasonably large face")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--swap-model", default=None)
    ap.add_argument("--enhancer", default=None)
    ap.add_argument("--mask-engine", default=None)
    ap.add_argument("--max-overpaint-pct", type=float, default=1.0,
                    help="share of the occluder's pixels allowed to move more "
                         "than --hard-delta. THIS is the discriminator: the "
                         "swap painting through shows as a large change on a "
                         "SUBSET of pixels, while colour transfer and the "
                         "merger move every pixel of the frame a little")
    ap.add_argument("--hard-delta", type=float, default=40.0,
                    help="per-pixel max-channel change counted as overpaint")
    ap.add_argument("--min-protection-pct", type=float, default=60.0,
                    help="how far the output inside the occluder sits toward "
                         "the OCCLUDER rather than toward the same frame "
                         "swapped with NO mask engine. 100%% = the object came "
                         "through untouched, 0%% = the mask engine did nothing "
                         "the unprotected pipeline would not have done")
    ap.add_argument("--occluder", default="texture", choices=OCCLUDER_STYLES,
                    help="what the foreign object looks like; see "
                         "OCCLUDER_STYLES")
    ap.add_argument("--control", action="store_true",
                    help="composite the occluder but grade the UNSWAPPED "
                         "frame. Must report 0.00 everywhere; if it does not, "
                         "the metric is measuring something else")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = args.provider or str(cfg.provider)
    swap_model = args.swap_model or str(cfg.swap_model)
    enhancer = args.enhancer or str(cfg.selected_enhancer)
    mask_engine = args.mask_engine or str(cfg.mask_engine)

    print("[occl] %s / %s / %s / %s | source %s | occluder: %s opaque ellipse, "
          "exact mask" % (swap_model, mask_engine, enhancer, provider,
                          args.source, args.occluder), flush=True)

    engine_key = map_mask_engine(mask_engine)
    if mask_engine not in ("", "None", None) and engine_key is None:
        raise SystemExit("unknown mask engine %r" % mask_engine)
    g = ab.init_pipeline(provider, swap_model, enhancer, mask_engine,
                         sync_config=True)
    options = ab.build_options(g, swap_model, engine_key)
    # THE PAIRED REFERENCE, and the reason this harness was rebuilt twice.
    #
    # A first metric read the absolute change inside the occluder. That is
    # confounded by the object's COLOUR: the swapped face is skin-toned, so
    # painting it over a skin-toned patch produces a small delta and over a
    # random-coloured patch a large one, with no difference in how much was
    # actually painted. Measured on this card, same engine, same geometry, only
    # the object's appearance changed: peak change 12.63 (arbitrary colour)
    # against 3.76 (skin) and 3.07 (dark). A 4x spread that says nothing about
    # masking.
    #
    # So each frame is swapped TWICE: once with the engine under test, once
    # with NO mask engine, which is the "nothing protects this object"
    # reference. The output inside the occluder then sits somewhere between the
    # untouched plate and the fully-painted reference, and where it sits is
    # colour-independent.
    unmasked_options = ab.build_options(g, swap_model, None)
    # ProcessMgr.initialize appends a foreground occluder to every swapping
    # chain that has no occlusion-aware engine in it -- which is exactly this
    # reference. Protecting the reference would protect the very object it
    # exists to leave unprotected, and `protection` would collapse toward 0 for
    # every engine while nothing had got worse. Opting out here keeps
    # "nothing protects this object" true.
    unmasked_options.disable_occlusion_injection = True
    src_fs = load_library_faceset(args.source)
    from roop.core import live_swap

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("could not read frame %d" % args.start)
    ref = ab.biggest_face(first)
    if ref is None:
        raise SystemExit("no face at frame %d -- pick another --start"
                         % args.start)
    face_box = [float(v) for v in ref.bbox]
    h, w = first.shape[:2]
    rng = np.random.default_rng(20260901)

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    rows, failures = [], []
    positions = list(sweep_positions(face_box, args.frames, w))
    for i, (cx, cy, rx, ry) in enumerate(positions):
        idx = args.start + i * args.stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        mask = occluder_mask(frame.shape, cx, cy, rx, ry)
        area = int((mask > 0).sum())
        if area == 0:
            continue
        patch = make_occluder(frame.shape[:2], rng, args.occluder)
        plate = np.where(mask[:, :, None] > 0, patch, frame)

        swapped = (plate.copy() if args.control
                   else live_swap(plate.copy(), options, input_facesets=[src_fs]))
        if swapped is None:
            swapped = plate.copy()
        unmasked = (plate.copy() if args.control
                    else live_swap(plate.copy(), unmasked_options,
                                   input_facesets=[src_fs]))
        if unmasked is None:
            unmasked = plate.copy()

        sel = mask > 0
        diff = np.abs(plate.astype(np.int16) - swapped.astype(np.int16))
        inside = float(diff[sel].mean())
        # How much of the occluder was overpainted HARD, not merely tinted.
        per_px = diff.max(axis=2)
        overpainted = float(100.0 * (per_px[sel] > args.hard_delta).sum() / area)

        # A far-corner background patch was also measured while diagnosing
        # this, and reads 0.00 on every frame: live_swap writes only inside the
        # pasted face region, so there is no global operation to subtract.
        d_plate = float(np.abs(swapped.astype(np.int16)
                               - plate.astype(np.int16))[sel].mean())
        d_painted = float(np.abs(swapped.astype(np.int16)
                                 - unmasked.astype(np.int16))[sel].mean())
        span = d_plate + d_painted
        # A tiny span means the two references agree -- the unprotected
        # pipeline did not paint here either, so there was nothing to protect
        # against and this frame carries no information about masking.
        protection = (100.0 * d_painted / span) if span > 0.5 else None
        unprotected = float(np.abs(unmasked.astype(np.int16)
                                   - plate.astype(np.int16))[sel].mean())

        # Overlap between the occluder and the face box: a frame where the
        # object is nowhere near the face proves nothing, so it is recorded and
        # excluded from the verdict rather than silently averaged in.
        ox0, oy0 = max(face_box[0], cx - rx), max(face_box[1], cy - ry)
        ox1, oy1 = min(face_box[2], cx + rx), min(face_box[3], cy + ry)
        overlap = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
        cover = 100.0 * overlap / max(
            (face_box[2] - face_box[0]) * (face_box[3] - face_box[1]), 1.0)

        row = {"frame": idx, "cover_pct": cover, "inside": inside,
               "unprotected": unprotected, "protection": protection,
               "overpainted_pct": overpainted}
        rows.append(row)
        verdict = "ok"
        if cover >= 10.0 and protection is None:
            verdict = "n/i: the unprotected arm did not paint here either"
        elif cover >= 10.0 and protection < args.min_protection_pct:
            verdict = ("FAIL: only %.0f%% protected -- the output inside the "
                       "object sits toward the fully-painted reference"
                       % protection)
        if verdict.startswith("FAIL"):
            failures.append("frame %d: %s" % (idx, verdict))
        print("[occl] frame %-6d cover %5.1f%%  unprotected-paint %6.2f  "
              "protection %6s  overpaint %5.2f%%  %s"
              % (idx, cover, unprotected,
                 ("%.0f%%" % protection) if protection is not None else "n/i",
                 overpainted, verdict), flush=True)

        if args.out:
            cv2.imwrite(os.path.join(args.out, "f%06d_plate.png" % idx), plate)
            cv2.imwrite(os.path.join(args.out, "f%06d_swap.png" % idx), swapped)
    cap.release()

    covering = [r for r in rows if r["cover_pct"] >= 10.0]
    if not covering:
        print("\n[occl] INCONCLUSIVE -- the occluder never covered >=10%% of "
              "the face box; nothing was actually tested")
        return 1
    scored = [r for r in covering if r["protection"] is not None]
    worst = max(covering, key=lambda r: r["overpainted_pct"])
    print("\n[occl] %d frames, %d with the occluder over >=10%% of the face"
          % (len(rows), len(covering)))
    print("[occl] worst overpaint: %.2f%% of the occluder, at %.1f%% face cover"
          % (worst["overpainted_pct"], worst["cover_pct"]))
    # The control arm is graded BEFORE the informative-frame check, because
    # "no frame was informative" is the CORRECT outcome when nothing was
    # swapped -- both arms are the untouched plate, so there is nothing to
    # protect against and nothing to protect. Grading it as inconclusive made
    # a correctly-behaving control return failure.
    if args.control:
        leaked = max(r["inside"] for r in covering)
        if failures or leaked > 0.001:
            print("[occl] CONTROL FAILED -- an unswapped frame must be "
                  "identical inside the occluder, and this one moved %.4f/255. "
                  "The metric is measuring something other than the swap."
                  % leaked)
            return 1
        print("[occl] control ok: 0.00 inside the occluder with no swap, and "
              "no frame informative, so the metric isolates the swap")
        return 0
    if not scored:
        print("[occl] INCONCLUSIVE -- no covering frame was informative: the "
              "unprotected pipeline did not paint the object either, so there "
              "was nothing to protect against")
        return 1
    least = min(scored, key=lambda r: r["protection"])
    print("[occl] protection: worst %.0f%%, median %.0f%% over %d informative "
          "frame(s); at worst the unprotected arm moved the object %.2f/255"
          % (least["protection"],
             float(np.median([r["protection"] for r in scored])),
             len(scored), least["unprotected"]))
    print("[occl] peak face cover reached: %.1f%%"
          % max(r["cover_pct"] for r in rows))

    if failures:
        print("[occl] %d FAILURE(S):" % len(failures))
        for f in failures:
            print("[occl]   %s" % f)
        return 1
    print("[occl] PASS -- the occluder survived the swap on every frame where "
          "it covered the face. NOT a claim about real hands: a synthetic "
          "patch is out of distribution for the mask engines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
