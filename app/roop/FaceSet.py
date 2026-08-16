import numpy as np

class FaceSet:
    faces = []
    ref_images = []
    embedding_average = 'None'
    embeddings_backup = None

    def __init__(self):
        self.faces = []
        self.ref_images = []
        self.embeddings_backup = None
        self.face_3d = None   # populated by face_3d_recon when use_3d_recon is enabled (first valid face's crop)
        # 3D recon per-face crop bank: list parallel to self.faces, each entry a
        # {'src_crop','src_M','src_lm68'} dict or None. Lets 3D recon warp the
        # source-bank-SELECTED face (not just face[0]) so the two features compose.
        self.face_3d_bank = None  # type: list[dict | None] | None
        # Multi-angle source bank: list of (yaw_deg, pitch_deg) or None per face in self.faces
        # Populated by ProcessMgr.initialize() when use_source_bank is enabled.
        self.face_poses = None  # type: list[tuple[float, float] | None] | None

    def AverageEmbeddings(self):
        if len(self.faces) > 1 and self.embeddings_backup is None:
            first_face = self.faces[0]
            if hasattr(first_face, 'embedding'):
                self.embeddings_backup = first_face.embedding
                embeddings = [face.embedding for face in self.faces]
                first_face.embedding = np.mean(embeddings, axis=0)
            else:
                self.embeddings_backup = first_face['embedding']
                embeddings = [face['embedding'] for face in self.faces]
                first_face['embedding'] = np.mean(embeddings, axis=0)
