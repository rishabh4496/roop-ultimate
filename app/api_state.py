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
current_video_fps = 30
