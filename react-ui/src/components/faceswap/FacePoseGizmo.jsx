import React from 'react';

function poseAngles(pose) {
  if (Array.isArray(pose)) return { yaw: Number(pose[0]) || 0, pitch: Number(pose[1]) || 0, roll: Number(pose[2]) || 0 };
  if (!pose || typeof pose !== 'object') return { yaw: 0, pitch: 0, roll: 0 };
  return {
    yaw: Number(pose.yaw ?? pose.yawDeg ?? pose.yaw_deg) || 0,
    pitch: Number(pose.pitch ?? pose.pitchDeg ?? pose.pitch_deg) || 0,
    roll: Number(pose.roll ?? pose.rollDeg ?? pose.roll_deg) || 0,
  };
}

export default function FacePoseGizmo({ pose, faceIndex = 0, similarity }) {
  const { yaw, pitch, roll } = poseAngles(pose);
  const confidence = Number(similarity);
  const showConfidence = Number.isFinite(confidence) && confidence >= 0 && confidence <= 1;

  return (
    <aside
      className="vfx-pose-gizmo"
      aria-label={`Face ${faceIndex + 1} pose: yaw ${yaw.toFixed(1)} degrees, pitch ${pitch.toFixed(1)} degrees, roll ${roll.toFixed(1)} degrees`}
      data-testid="face-pose-gizmo"
    >
      <div className="vfx-pose-gizmo__heading">
        <span>POSE / #{faceIndex + 1}</span>
        {showConfidence && <span className="vfx-pose-gizmo__score">COS {confidence.toFixed(3)}</span>}
      </div>
      <svg viewBox="0 0 88 76" role="img" aria-label="Rotating head pose wireframe">
        <g className="vfx-pose-gizmo__head" style={{ transform: `perspective(180px) rotateX(${pitch}deg) rotateY(${yaw}deg) rotateZ(${roll}deg)` }}>
          <path d="M27 12 Q44 3 59 14 Q68 23 63 39 L58 49 Q55 61 43 64 Q30 62 26 52 Q19 46 20 35 Q15 20 27 12Z" />
          <path d="M42 8 Q37 22 42 34 Q45 48 42 63" />
          <path d="M22 25 Q42 31 65 24 M21 36 Q42 42 64 35 M25 48 Q43 54 59 46" />
          <path d="M27 27 Q34 22 40 27 M46 27 Q54 22 61 27 M37 43 Q43 46 49 42 M35 53 Q43 57 51 52" />
          <path d="M42 37 L44 44 L48 45" />
        </g>
        <g className="vfx-pose-gizmo__cube" style={{ transform: `perspective(130px) rotateX(${pitch}deg) rotateY(${yaw}deg) rotateZ(${roll}deg)` }}>
          <path d="M69 53 L76 49 L83 53 L76 57 Z M69 53 V62 L76 66 L83 62 V53 M76 57 V66 M69 62 L76 58 L83 62" />
        </g>
        <path className="vfx-pose-gizmo__axis vfx-pose-gizmo__axis--x" d="M11 68 H30" />
        <path className="vfx-pose-gizmo__axis vfx-pose-gizmo__axis--y" d="M11 68 V51" />
        <path className="vfx-pose-gizmo__axis vfx-pose-gizmo__axis--z" d="M11 68 L21 58" />
      </svg>
      <div className="vfx-pose-gizmo__angles">
        <span>Y {yaw >= 0 ? '+' : ''}{yaw.toFixed(1)}°</span>
        <span>P {pitch >= 0 ? '+' : ''}{pitch.toFixed(1)}°</span>
        <span>R {roll >= 0 ? '+' : ''}{roll.toFixed(1)}°</span>
      </div>
    </aside>
  );
}
