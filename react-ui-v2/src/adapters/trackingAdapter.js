/**
 * Spatial coordinate calculations for face tracking, bounding boxes, landmarks, and head pose.
 */

export const trackingAdapter = {
  /**
   * Convert native pixel bounding box [sx, sy, ex, ey] to layout percentage styles.
   */
  boxToPercentStyle: (bbox, imgDim) => {
    if (!bbox || !imgDim || !imgDim.w || !imgDim.h) return null;
    const [sx, sy, ex, ey] = bbox;
    return {
      left: `${(sx / imgDim.w) * 100}%`,
      top: `${(sy / imgDim.h) * 100}%`,
      width: `${((ex - sx) / imgDim.w) * 100}%`,
      height: `${((ey - sy) / imgDim.h) * 100}%`,
    };
  },

  /**
   * Format solved 3D head pose (yaw, pitch, roll in degrees).
   */
  formatPose: (poseArray) => {
    if (!poseArray || poseArray.length < 3) return null;
    const [yaw, pitch, roll] = poseArray.map((v) => Math.round(v));
    return `y${yaw}° p${pitch}° r${roll}°`;
  },

  /**
   * ArcFace 5-point landmark canonical structure:
   * 0: Left Eye, 1: Right Eye, 2: Nose, 3: Left Mouth, 4: Right Mouth
   */
  parseLandmarks: (kps) => {
    if (!kps || kps.length < 5) return null;
    const [eyeL, eyeR, nose, mouthL, mouthR] = kps;
    return {
      eyeL,
      eyeR,
      nose,
      mouthL,
      mouthR,
      eyeMid: [(eyeL[0] + eyeR[0]) / 2, (eyeL[1] + eyeR[1]) / 2],
      mouthMid: [(mouthL[0] + mouthR[0]) / 2, (mouthL[1] + mouthR[1]) / 2],
    };
  },
};
