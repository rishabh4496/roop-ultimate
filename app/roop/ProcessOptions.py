from roop.target_selection import normalize_target_selection


class ProcessOptions:

    def __init__(self, processordefines:dict, face_distance,  blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, subsample_size, show_face_area, restore_original_mouth, show_mask=False, use_3d_recon=False,
                 use_source_bank=False, use_frontalization=False, frontalization_threshold=25.0, swap_model='inswapper',
                 stabilize_face=False, stabilize_method='one_euro', stabilize_min_cutoff=0.05, stabilize_beta=0.02,
                 stabilize_enhancer=False, stabilize_enhancer_strength=0.5,
                 stabilize_mask=False, stabilize_mask_strength=0.5,
                 stabilize_landmarks=True, stabilize_hf_texture=False,
                 stabilize_hf_texture_weight=0.15, selection_state=None,
                 processing_request=None):
        self.processors = processordefines
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        # The API creates one serializable request for preview and render. Keep
        # that snapshot available to ProcessMgr; selection_state remains as a
        # compatibility projection for direct/legacy callers.
        self.processing_request = (
            dict(processing_request)
            if isinstance(processing_request, dict) else None
        )
        if self.processing_request and selection_state is None:
            selection_state = self.processing_request.get("selection_state")
        # Canonical target-person selection. This is deliberately separate from
        # selected_index, which is a source-gallery index in the legacy modes.
        # The request's stable person ids are the universe this selection is
        # judged against; without them a "tp_..." id parsed as a missing rank
        # and the whole run swapped nothing (Stage 15 acceptance finding).
        stable_ids = (self.processing_request.get("target_person_ids")
                      if self.processing_request else None)
        self.selection_state = normalize_target_selection(
            selection_state, target_person_ids=stable_ids or None)
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
        # Dense-landmark smoothing, in step with the kps filter above; and the
        # flow-warped high-frequency carry over the restorer's output.
        # See roop/temporal_smoother.py.
        self.stabilize_landmarks = stabilize_landmarks
        self.stabilize_hf_texture = stabilize_hf_texture
        self.stabilize_hf_texture_weight = stabilize_hf_texture_weight
        # Opt OUT of the foreground occluder ProcessMgr.initialize otherwise
        # appends to every swapping chain (see roop/occlusion_mask.py).
        #
        # This is not a user setting -- `enable_occlusion_mask` is. It exists
        # for measurement: `tests/occlusion_ground_truth.py` grades a protected
        # arm against a deliberately UNPROTECTED reference built with no mask
        # engine at all, and injecting an occluder into that reference would
        # protect the thing the reference exists to leave unprotected, quietly
        # collapsing the metric toward "no difference".
        self.disable_occlusion_injection = False
