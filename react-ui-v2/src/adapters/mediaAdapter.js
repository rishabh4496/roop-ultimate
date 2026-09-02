import { postJSON, postFiles, postFile } from './apiClient';

export const mediaAdapter = {
  // Source Media
  addSourceFiles: (files, opts) => postFiles('/api/source/add', files, undefined, opts),
  removeSource: (index) => postJSON('/api/source/remove', { index }),
  moveSource: (fromIndex, toIndex) => postJSON('/api/source/move', { from: fromIndex, to: toIndex }),
  clearSources: () => postJSON('/api/source/clear', {}),
  selectSource: (index) => postJSON('/api/source/select', { index }),
  refreshSourceThumbs: () => postJSON('/api/source/refresh_thumbs', {}),
  addLipsyncAudio: (file, opts) => postFile('/api/lipsync/audio/add', file, undefined, opts),

  // Target Media
  addTargetFiles: (files, opts) => postFiles('/api/target/add', files, undefined, opts),
  addTargetPaths: (paths) => postJSON('/api/target/add_path', { paths }),
  selectTarget: (index) => postJSON('/api/target/select', { index }),
  removeTarget: (index) => postJSON('/api/target/remove', { index }),
  clearTargets: () => postJSON('/api/target/clear', {}),
  setTargetFrame: (which, frame) => postJSON('/api/target/set_frame', { which, frame }),

  // Target Face Banking & Angles
  useTargetFace: (index, frame, faceIndex = 0) =>
    postJSON('/api/target/use_face', { index, frame, face_index: faceIndex }),
  addTargetAngle: (index, frame, faceIndex = 0) =>
    postJSON('/api/target/add_angle', { index, frame, face_index: faceIndex }),
  autoCaptureAngles: (index, frame) =>
    postJSON('/api/target/auto_angles', { index, frame }),
  autoCaptureVideoFaces: (index) =>
    postJSON('/api/target/auto_capture', { index }),
  removeTargetFace: (index, faceIndex) =>
    postJSON('/api/target/remove_face', { index, face_index: faceIndex }),
  clearTargetFaces: (index) =>
    postJSON('/api/target/clear_faces', { index }),
  groupTargetFaces: (index, groups) =>
    postJSON('/api/target/group', { index, groups }),
  nameTargetPerson: (index, personId, name) =>
    postJSON('/api/target/name', { index, person_id: personId, name }),
  autoclusterTargetFaces: (index) =>
    postJSON('/api/target/autocluster', { index }),
};
