"""Who owns a pixel when two swapped faces overlap.

Every face in a frame is pasted back under its own matte — a warped ellipse
intersected with the landmark hull, then dilated and feathered (see
`procmgr_masking.paste_upscale`). That matte is built from ONE face's geometry
and knows nothing about anybody else, which is fine while the faces are apart
and wrong the moment they are not:

  * the hull is dilated and the feather blurs outward, so face A's matte spills
    a band of A's swapped skin over whatever is next to it — including B's
    cheek. Two people cheek to cheek get a smear of each other's swap along the
    join;
  * the forehead extension projects ~60 % of the face height past the brow, so
    a face leaning in front of another paints its invented forehead across the
    face behind it;
  * the faces composite in sequence, so whoever is pasted LAST wins the
    contested band. The order comes from detection / identity-match order,
    which is not stable frame to frame — so the winner alternates and the join
    visibly flips back and forth.

The fix is to decide ownership once per frame, from geometry alone, before
anything is pasted. Each face gets a signed distance field to its own hull,
normalised by its own size so a near face and a far one are compared on equal
terms, and a pixel belongs to whichever face is deepest inside relative to its
own scale. The boundary that falls out is the locus where the two are equally
deep — a real demarcation line between the two faces, in the same place every
frame because it depends only on the (already stabilised) landmarks.

The demarcation is enforced by PAINT ORDER, not by splitting the contested
pixels between the two faces. Splitting them looks right on paper and is wrong
in practice: the faces composite one after another, so two mattes at 0.5 give
`0.5·B + 0.25·A + 0.25·plate` — a hairline of untouched footage down the join,
which is a seam rather than a smear but still an artefact. Instead each face
defends its territory only against the faces painted BEFORE it, and is allowed
one hand-over band past the boundary into the territory of faces painted AFTER
it. The face underneath then backs the ramp of the face on top, the two blend
into each other, and the plate never shows through.

`build_regions` returns one `FaceRegion` per face, or None when the faces do
not interact at all — which is the common case and costs two bounding-box
tests. Nothing is computed for frames where nobody overlaps.
"""

import os

import cv2
import numpy as np

from roop.procmgr_masking import landmark_hull


def _env_float(name, default):
    try:
        v = float(os.environ.get(name, ''))
    except ValueError:
        return default
    return v if v > 0 else default


# Master switch. ROOP_FACE_DEMARCATE=0 restores the old behaviour (every face's
# matte pasted whole, in match order) — use it to check whether an artefact on
# interacting faces comes from this or from the swap itself.
ENABLED = os.environ.get('ROOP_FACE_DEMARCATE', '1').strip().lower() not in ('0', 'off', 'false')

# Width of the hand-over band, as a fraction of face radius. The face on top
# ramps in across it and the one underneath backs it, so this is a blend from
# one swapped face into the other rather than either of them into the plate —
# but it is still a place where two different faces are mixed, so it wants to be
# narrow enough to read as an edge and wide enough not to alias.
FEATHER = _env_float('ROOP_FACE_DEMARCATE_FEATHER', 0.02)

# ...and wide enough not to SHIMMER, which is the constraint the fraction alone
# could not express.
#
# The boundary is the locus where two normalised distance fields cross, and each
# field is rasterised from that frame's landmark hull. The 106-point landmarks
# are not temporally smoothed, so about a pixel of detector jitter moves the
# boundary about a pixel — every frame, on two people standing still. Whether
# that is visible depends entirely on how it compares to the width of the ramp
# it is moving inside, and a band given only as a fraction of face radius makes
# that comparison an accident of how big the face happens to be.
#
# Measured mean worst per-frame ownership change at the join, two faces cheek to
# cheek at a 117 px radius under 1 px of landmark noise (1.0 would be a pixel
# handed fully from one face to the other in a single frame):
#
#   band      4.7px   9.4px   14.0px   18.7px   28.1px
#   swing     0.383   0.260    0.182    0.135    0.070
#
# 4.7 px is what a 0.02 fraction gives at that size, and a third of a pixel's
# worth of ownership changing hands every frame is exactly the flicker along the
# join that gets reported. The noise is a fixed number of PIXELS, so the floor
# has to be too — the fraction then takes over for faces big enough to need a
# proportionally wider ramp.
MIN_BAND_PX = _env_float('ROOP_FACE_DEMARCATE_MIN_BAND_PX', 16.0)

# ...but not so wide that a small face is nothing but ramp. On a 50 px-radius
# face a 16 px floor would be a third of the head, which stops being a
# demarcation and starts being a dissolve, so the floor is capped as a fraction
# of radius and small faces get a proportionally narrower band instead.
MAX_BAND_FRAC = _env_float('ROOP_FACE_DEMARCATE_MAX_BAND', 0.16)

# How much a face's on-screen size counts as "in front". Two people at the same
# distance have the same size and this cancels exactly; when one head is twice
# the other's it is almost certainly nearer the camera, and this pushes the
# boundary that far into the smaller one. Expressed per doubling of face radius.
DEPTH_BIAS = _env_float('ROOP_FACE_DEMARCATE_DEPTH', 0.10)

# Padding on each claim's bounding box before the "do these two interact?" test,
# as a fraction of face radius. Covers the matte's own dilation and feather, so
# faces that are merely adjacent still get a boundary drawn between them rather
# than being declared disjoint and left to smear.
_RECT_PAD = 0.15

# How far a face keeps painting past the boundary into the territory of a face
# that has not been painted yet, in hand-over bands. 0.5 is the boundary itself;
# 1.5 is one full band beyond it, which is exactly enough to sit under the other
# face's ramp.
_BACKING = 1.5

# How much of one claim has to sit inside the other before the two are treated
# as duplicate detections of one face rather than as neighbours. Two people cheek
# to cheek reach maybe 0.2-0.3 containment through the padded boxes; a second box
# on one head is 0.8+. 0.6 sits in the empty middle.
#
# Measured EITHER WAY ROUND, which is the whole content of this being a pair of
# numbers rather than one. Which of two duplicate boxes ends up holding the
# source is arbitrary — it is whichever the tracker happened to associate — so
# the unpainted one is as often the LARGER box as the smaller. Tested in one
# direction only, "is the leftover inside the swapped face", half of all
# duplicates sail through: containment of a big box in a small one is ~0.12 and
# nothing fires.
_DUP_CONTAIN = _env_float('ROOP_FACE_DEMARCATE_DUP', 0.6)

# ...and containment on its own cannot be made symmetric safely, because it is
# also 1.0 for a genuinely separate SMALL face sitting anywhere inside a large
# one's padded box — a background head over a foreground shoulder. Dropping that
# claim is the smear this module exists to remove.
#
# What separates the two is where the claims sit, not how much they overlap:
# duplicate boxes of one head are concentric, two faces have centres a real
# distance apart however nested their boxes are. Centre separation over the
# SMALLER claim's radius, measured across the cases:
#
#   duplicate, jittered box        0.07     separate, cheek to cheek     0.76
#   duplicate, partial box         0.18     separate, small over big     1.08
#   duplicate, partial offset      0.24     separate, overlapping        0.52
#
# 0.35 sits in the empty band between them, and is the same number the per-track
# diagnostic in procmgr_tracking prints "THE SAME FACE, detected twice" at.
_DUP_CONCENTRIC = _env_float('ROOP_FACE_DEMARCATE_DUP_SEP', 0.35)


class FaceRegion:
    """One face's share of the frame, as a 0..1 field over a rectangle.

    Outside the rectangle the face owns everything (by construction the
    rectangle covers every place a competing claim could reach), so the callers
    only ever have to touch the ROI.
    """

    __slots__ = ('x0', 'y0', 'x1', 'y1', 'own')

    def __init__(self, x0, y0, x1, y1, own):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.own = own

    def trim_frame(self, matte):
        """Multiply a FULL-FRAME float matte by this face's ownership, in place.

        Returns the matte for convenience. A matte of a different size (a
        caller working at another resolution) is left untouched rather than
        silently mis-registered.
        """
        h, w = matte.shape[:2]
        if self.x1 > w or self.y1 > h:
            return matte
        view = matte[self.y0:self.y1, self.x0:self.x1]
        if view.ndim == 3:
            view *= self.own[:, :, None]
        else:
            view *= self.own
        return matte

    def crop(self, x0, y0, x1, y1):
        """Ownership over the frame-space box (x0, y0, x1, y1), or None when the
        box lies entirely outside the ROI (i.e. this face owns all of it).

        Used by the post-composite pastes — mouth restore, eye restore, lip-sync
        — which write into a plain rectangle rather than through the matte.
        Without this, restoring one person's mouth can paste the untouched plate
        over the face swapped next to them.
        """
        ix0, iy0 = max(x0, self.x0), max(y0, self.y0)
        ix1, iy1 = min(x1, self.x1), min(y1, self.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        out = np.ones((y1 - y0, x1 - x0), dtype=np.float32)
        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
            self.own[iy0 - self.y0:iy1 - self.y0, ix0 - self.x0:ix1 - self.x0]
        return out


def _claim_polygon(face):
    """The polygon this face claims, in frame coordinates.

    Prefers the same landmark hull the paste matte uses, so the demarcation is
    drawn where the matte actually reaches. Falls back to an ellipse inscribed
    in the detection box when the 106-point landmarks are absent (they are
    optional — several detector engines here do not produce them).
    """
    lm = getattr(face, 'landmark_2d_106', None)
    if lm is not None:
        try:
            hull, _, _ = landmark_hull(lm, getattr(face, 'kps', None))
            return hull.reshape(-1, 2).astype(np.int32)
        except Exception:
            pass
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    ax, ay = max(1.0, (x1 - x0) * 0.5), max(1.0, (y1 - y0) * 0.5)
    return cv2.ellipse2Poly((int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 12)


def _rects_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _containment(a, b):
    """Fraction of rect `a` that lies inside rect `b`."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return (iw * ih) / float(area)


def _same_face(claim_a, claim_b):
    """Whether two claims are one head detected twice rather than two faces.

    Nested AND concentric. Either test alone gets it wrong in a way that matters:
    containment is 1.0 for a small separate face anywhere inside a big one's
    padded box, and concentricity alone would collapse two people standing dead
    in line. See _DUP_CONTAIN and _DUP_CONCENTRIC for the measurements.
    """
    (_, ra, r_a), (_, rb, r_b) = claim_a, claim_b
    if max(_containment(ra, rb), _containment(rb, ra)) <= _DUP_CONTAIN:
        return False
    sep = float(np.hypot((ra[0] + ra[2]) - (rb[0] + rb[2]),
                         (ra[1] + ra[3]) - (rb[1] + rb[3])) * 0.5)
    return sep < _DUP_CONCENTRIC * min(r_a, r_b)


def build_regions(faces, frame_shape, order=None, feather=None, depth_bias=None):
    """Ownership fields for `faces`, as ``{index: FaceRegion}``.

    `order` is the sequence in which the caller will paint them, back to front,
    as indices into `faces`; it defaults to the order they are given in. It has
    to be known here because a face only needs to defend itself against the
    faces already on the canvas — see the module docstring.

    A face in `faces` but NOT in `order` is a claimant that will never be
    painted: an unswapped bystander. It gets no region of its own (there is
    nothing to trim), and it is treated as already on the canvas by everyone
    else — which is exactly true, since what is showing there is the original
    footage and that is what should stay. Without this a bystander is not in the
    competition at all and takes a neighbour's dilated matte, feather and
    invented forehead across their face; and because whether a face is being
    swapped can change frame to frame (a borderline identity match, a source that
    ran out), it takes it intermittently, which is a flicker rather than a
    steady artefact.

    Returns None when fewer than two faces actually compete for pixels — the
    normal case, decided by bounding-box tests before anything is rasterised.
    Indices are positions in `faces`; a face with no usable geometry, or one
    whose claim lands off-frame, is simply absent from the mapping and pastes
    unrestricted as before.
    """
    if not ENABLED or faces is None or len(faces) < 2:
        return None

    feather = FEATHER if feather is None else feather
    depth_bias = DEPTH_BIAS if depth_bias is None else depth_bias

    claims = []
    for f in faces:
        pts = _claim_polygon(f)
        if pts is None or len(pts) < 3:
            claims.append(None)
            continue
        x, y, w, h = cv2.boundingRect(pts)
        r = float(np.sqrt(max(1.0, float(w) * float(h))))
        pad = int(max(2.0, r * _RECT_PAD))
        claims.append((pts, (x - pad, y - pad, x + w + pad, y + h + pad), r))

    # A face cannot be its own bystander. Detectors do produce two boxes on one
    # face — a partial box beside a full one, or two engines' worth of the same
    # head — and such a duplicate is not being swapped (only one of them can hold
    # the source), so it arrives here as an unpainted claimant sitting exactly on
    # top of a painted one. Left in, it wins half of that face's own pixels and
    # the swap is carved apart along a line through the middle of it, on only the
    # frames where the duplicate happened to be detected. That is worse than the
    # smearing this whole module exists to fix, and it appears and disappears.
    #
    # Concentric claims are therefore dropped rather than competed with. The test
    # is containment, not IoU: a duplicate box is often much smaller than the real
    # one (a partial detection), which puts IoU low while the two are plainly the
    # same face. It is measured in BOTH directions, because which of the two
    # boxes ends up holding the source is arbitrary — see _same_face.
    if order is not None:
        painted_idx = set(order)
        for i in range(len(claims)):
            if claims[i] is None or i in painted_idx:
                continue
            for j in painted_idx:
                if claims[j] is None:
                    continue
                if _same_face(claims[i], claims[j]):
                    claims[i] = None
                    break

    hot = set()
    for i in range(len(claims)):
        if claims[i] is None:
            continue
        for j in range(i + 1, len(claims)):
            if claims[j] is not None and _rects_overlap(claims[i][1], claims[j][1]):
                hot.add(i)
                hot.add(j)
    if len(hot) < 2:
        return None

    h_frame, w_frame = frame_shape[:2]
    x0 = max(0, min(claims[i][1][0] for i in hot))
    y0 = max(0, min(claims[i][1][1] for i in hot))
    x1 = min(w_frame, max(claims[i][1][2] for i in hot))
    y1 = min(h_frame, max(claims[i][1][3] for i in hot))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    # Reference scale for the depth bias: the geometric mean of the competing
    # face radii, so the bias is a pure comparison between them and adds nothing
    # when they are the same size.
    radii = [claims[i][2] for i in hot]
    r_ref = float(np.exp(np.mean(np.log(radii))))

    fields = {}
    for i in sorted(hot):
        pts, _, r = claims[i]
        m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(m, pts - np.array([x0, y0], dtype=np.int32), 255)
        if not m.any():
            continue        # claim is entirely off the ROI — no say in it
        inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(255 - m, cv2.DIST_L2, 3)
        # Depth relative to this face's OWN size: without the normalisation a
        # big face is deeper everywhere simply for being big, and would take the
        # whole of a small face standing in front of it.
        field = (inside - outside) / r
        if depth_bias:
            field += depth_bias * float(np.log2(r / r_ref))
        fields[i] = field

    if len(fields) < 2:
        return None

    if order is None:
        order = range(len(faces))
    # -1 for anything never painted, so it sorts before every painted face and
    # every one of them therefore treats it as already on the canvas.
    paint_pos = {idx: k for k, idx in enumerate(order)}
    painted = set(paint_pos)

    def _best(fs):
        return fs[0] if len(fs) == 1 else np.maximum.reduce(fs)

    # `field` is a distance divided by the face radius, so a band of B pixels is
    # B / r_ref in field units. r_ref is the geometric mean of the competing
    # radii — the same common scale the depth bias is measured against, and the
    # right one here because the two fields are normalised by their OWN radii and
    # then compared with each other.
    band = max(1e-4, 2.0 * feather, min(MIN_BAND_PX / max(r_ref, 1.0), MAX_BAND_FRAC))
    regions = {}
    for i, field in fields.items():
        if i not in painted:
            continue            # a claimant, not a paste — nothing to trim
        own = np.ones_like(field)
        # Against faces ALREADY painted: stop at the boundary. Going past it
        # would paint this face's swap over one that is already down, which is
        # the bleed being removed. Unswapped bystanders count as already painted
        # (paint_pos -1): what is on the canvas there is the original footage,
        # and it should stay.
        earlier = [f for j, f in fields.items()
                   if j != i and paint_pos.get(j, -1) < paint_pos[i]]
        if earlier:
            own *= np.clip(0.5 + (field - _best(earlier)) / band, 0.0, 1.0)
        # Against faces STILL TO COME: carry on one band past the boundary. They
        # will paint over it, and their own hand-over ramp needs something under
        # it — without this backing the ramp blends into the untouched plate and
        # leaves a hairline of original footage along the join.
        #
        # One band is enough and more does not help (measured): what remains
        # after it is the part of the boundary that runs OUTSIDE the other
        # face's matte, most often where one face's invented forehead sat over
        # the other's hair. Withdrawing it uncovers the real plate there, which
        # is the right answer, not a gap to be filled.
        later = [f for j, f in fields.items()
                 if j != i and paint_pos.get(j, -1) > paint_pos[i]]
        if later:
            own *= np.clip(_BACKING + (field - _best(later)) / band, 0.0, 1.0)
        regions[i] = FaceRegion(x0, y0, x1, y1, own.astype(np.float32))
    return regions
