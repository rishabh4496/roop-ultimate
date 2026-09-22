"""Mutable module state shared between api.py and its route modules.

These three are REBOUND at runtime (a new value is assigned, rather than an
existing object being mutated), so they cannot be shared by handing a router
module a reference at import time — each side would end up rebinding its own
copy and drift apart. Everything reaches them through this module instead, so
there is exactly one binding.

Read and write them as attributes (`state.selected_target_index = 3`), never
via `from api_state import selected_target_index`, which would copy the value
and reintroduce the very problem this module exists to prevent.
"""

selected_input_face_index = 0          # which source faceset is "selected"
selected_target_index = 0              # which target file is shown in preview
active_target_media_id = None          # stable id for the selected target file
selected_target_face_index = 0         # compatibility mirror of active context
active_target_source_mapping = {}      # per-target source mapping compatibility mirror
active_target_person_source_mapping = {}  # target_person_id -> source_identity_id
active_target_selected_source_id = None  # target-scoped implicit source selection
active_target_person_names = {}           # target_person_id -> display name
selected_target_person_id = None
selected_reference_face_id = None
current_video_fps = 30
