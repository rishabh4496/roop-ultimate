class ProcessOptions:

    def __init__(self, processordefines:dict, face_distance,  blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, subsample_size, show_face_area, restore_original_mouth, show_mask=False, use_3d_recon=False,
                 use_source_bank=False, use_frontalization=False, frontalization_threshold=25.0, swap_model='inswapper',
                 stabilize_face=False, stabilize_method='one_euro', stabilize_min_cutoff=0.05, stabilize_beta=0.02,
                 stabilize_enhancer=False, stabilize_enhancer_strength=0.5,
                 stabilize_mask=False, stabilize_mask_strength=0.5):
        self.processors = processordefines
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        self.masking_text = masking_text
        self.imagemask = imagemask
        self.num_swap_steps = num_steps
        self.show_face_area_overlay = show_face_area
        self.show_face_masking = show_mask
        self.subsample_size = subsample_size
        self.restore_original_mouth = restore_original_mouth
        self.max_num_reuse_frame = 15
        # 3D source pose matching
        self.use_3d_recon = use_3d_recon
        # Multi-angle source bank (Option 1)
        self.use_source_bank = use_source_bank
        # Target frontalization (Option 2)
        self.use_frontalization = use_frontalization
        self.frontalization_threshold = frontalization_threshold
        self.swap_model = swap_model
        # One Euro temporal stabilization of face keypoints (video only)
        self.stabilize_face = stabilize_face
        self.stabilize_method = stabilize_method
        self.stabilize_min_cutoff = stabilize_min_cutoff
        self.stabilize_beta = stabilize_beta
        # One Euro temporal smoothing of the enhancer output (anti-flicker)
        self.stabilize_enhancer = stabilize_enhancer
        self.stabilize_enhancer_strength = stabilize_enhancer_strength
        # One Euro temporal smoothing of the mask edge (anti-flicker)
        self.stabilize_mask = stabilize_mask
        self.stabilize_mask_strength = stabilize_mask_strength