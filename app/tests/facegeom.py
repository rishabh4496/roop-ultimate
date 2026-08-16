"""Synthetic face geometry for the alignment / pose tests.

Projects a known 3-D head at a chosen yaw / pitch / roll so tests can assert on
pose behaviour without needing a video, a detector, or a GPU. The 3-D points are
taken from the project's own reference face (roop.face_3d_recon._REF3D_68) so
the tests and the shipping pose code agree on what a head is shaped like.

Orthographic (weak-perspective) projection is deliberate: it is the clean case.
If a heuristic already misbehaves under orthographic projection it cannot be
rescued by perspective, and any failure is unambiguously the heuristic's.
"""

import numpy as np

# The 5 arcface keypoints, in the same order the detectors emit them:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
# Eye centres are the means of _REF3D_68[36:42] / [42:48]; nose tip is index 30;
# mouth corners are indices 48 and 54.  OpenCV convention: +x right, +y up,
# +z toward the camera.
LEFT_EYE = np.array([-0.334, 0.379, 0.170])
RIGHT_EYE = np.array([0.334, 0.379, 0.170])
NOSE_TIP = np.array([0.000, -0.098, 0.547])
MOUTH_LEFT = np.array([-0.368, -0.547, 0.299])
MOUTH_RIGHT = np.array([0.368, -0.547, 0.299])

HEAD_5PT = np.vstack([LEFT_EYE, RIGHT_EYE, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT])


def rotation(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def project_kps(yaw_deg, pitch_deg=0.0, roll_deg=0.0, scale=200.0, cx=256.0, cy=256.0):
    """The 5 keypoints of a head at this pose, in image coordinates (y down)."""
    pts = HEAD_5PT @ rotation(yaw_deg, pitch_deg, roll_deg).T
    return np.column_stack([cx + scale * pts[:, 0],
                            cy - scale * pts[:, 1]]).astype(np.float32)


def head_106():
    """A 106-point head with real depth relief, in the same 3-D frame as
    HEAD_5PT, using the index layout create_landmark_mask assumes (0-32 contour,
    33-52 eyebrows).

    Depth is the whole point. A flat 2-D face rotated in-plane — which is all the
    roll tests need — collapses to a line under yaw, so it cannot say anything
    about how the mask behaves on a turned head. Contour points wrap around an
    ellipsoid so the cheeks recede, and the brows and nose stand proud.

    The 5 arcface keypoints are appended as members, because in the real pipeline
    the 106 landmarks and the 5 keypoints come from the same detector on the same
    face. Two synthetic heads that disagree on where the mouth is will report a
    keypoint outside the mask and look exactly like a bug in the mask.
    """
    pts = []
    for i in range(33):                                   # 0-32 jaw / contour
        a = np.pi * (-0.5 + i / 32.0)
        pts.append([0.50 * np.sin(a),
                    -0.62 + 0.50 * (1.0 - abs(np.sin(a))),
                    0.30 * np.cos(a)])
    for i in range(20):                                   # 33-52 eyebrows
        t = -0.36 + 0.72 * i / 19.0
        pts.append([t, 0.46, 0.28 - 0.35 * t * t])
    for i in range(20):                                   # eye region
        a = 2 * np.pi * i / 20.0
        pts.append([0.33 * np.cos(a), 0.36 + 0.06 * np.sin(a), 0.20])
    for i in range(13):                                   # nose ridge -> tip
        pts.append([0.0, 0.36 - 0.46 * i / 12.0, 0.20 + 0.35 * i / 12.0])
    for i in range(15):                                   # mouth
        a = 2 * np.pi * i / 15.0
        pts.append([0.30 * np.cos(a), -0.55 + 0.09 * np.sin(a), 0.30])
    return np.vstack([np.asarray(pts, dtype=np.float64), HEAD_5PT])


def project_points(pts3d, yaw_deg, pitch_deg=0.0, roll_deg=0.0,
                   scale=200.0, cx=256.0, cy=256.0):
    """Project arbitrary 3-D head points to image coordinates at this pose."""
    q = np.asarray(pts3d, dtype=np.float64) @ rotation(yaw_deg, pitch_deg, roll_deg).T
    return np.column_stack([cx + scale * q[:, 0], cy - scale * q[:, 1]])


def decompose(matrix):
    """(uniform scale, in-plane rotation in degrees) of a 2x3 similarity."""
    a, c = matrix[0, 0], matrix[1, 0]
    return float(np.hypot(a, c)), float(np.degrees(np.arctan2(c, a)))


def fit_residual(matrix, src, dst):
    """Mean distance in pixels between the transformed src points and dst."""
    homogeneous = np.column_stack([src, np.ones(len(src))])
    return float(np.linalg.norm(homogeneous @ matrix.T - dst, axis=1).mean())
