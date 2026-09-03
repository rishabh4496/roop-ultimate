"""Settings wiring — that a control reaching the UI actually reaches the render.

The recurring failure here is a HALF-WIRED setting: a toggle that appears in the
UI, saves fine, and silently changes nothing, because one of the places it has to
be listed was missed. It has bitten at least twice.

The frontend half is now structural rather than tested: FaceSwap.jsx builds the
request body and the cache signature from one `buildPreviewPayload`, and the
signature IS the request body with the frame coordinates zeroed, so a setting
cannot be sent without being keyed or keyed without being sent.

The backend half cannot be collapsed the same way — `preview()` sets globals
while the run path also passes kwargs — so it is guarded here instead: these
tests read the actual source and assert the two sides agree, with every
exception named and justified rather than merely tolerated.

Source parsing (rather than importing) is deliberate: it works across the
Python/JavaScript boundary and needs no browser or running server.
"""

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parents[2]
FACESWAP_JSX = REPO / "react-ui" / "src" / "components" / "FaceSwap.jsx"
API_PY = REPO / "app" / "api.py"


def _match_braces(text, start):
    depth, i = 0, start
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1


def frontend_payload_keys():
    """Every key FaceSwap.jsx sends to /api/preview."""
    src = FACESWAP_JSX.read_text(encoding="utf-8")
    m = re.search(r"const buildPreviewPayload = .*?=> \(", src, re.S)
    assert m, "buildPreviewPayload not found — did the preview builder get renamed?"
    body = _match_braces(src, src.index("{", m.end() - 1))
    keys = set(re.findall(r"(?:^|[\{,]|\n)\s*([a-z_][a-z_0-9]*)\s*:", body))
    keys.add("index")           # shorthand property
    keys.discard("overrides")   # the spread, not a field
    return keys


def _api_source():
    return API_PY.read_text(encoding="utf-8")


def _function_body(src, marker, from_index=0):
    # `i + 1`, not `i`: rindex searches src[0:i], which EXCLUDES the marker's
    # own `def ` when the marker is itself a def. It therefore walked back into
    # whatever function was declared just above and returned that one's text as
    # well. Harmless while the neighbour above preview() held no string
    # literals the scans below look for — and then _apply_eye_restore_settings
    # moved in next door, its one literal payload.get() was read as preview's,
    # and preview appeared to consume a key the run path did not.
    i = src.index(marker, from_index)
    start = src.rindex("def ", 0, i + 1)
    try:
        end = src.index("\n@app.", i)
    except ValueError:
        end = len(src)
    return src[start:end]


# Helpers that both endpoints delegate settings to. Each reads
# `payload.get(key, ...)` with a LOOP VARIABLE, so a plain scan of their bodies
# finds nothing — the key names live in a tuple of string literals instead.
# Parsing those out keeps this honest automatically: adding a knob to a helper
# is enough, with no allowlist here to remember to update (and none to go stale
# if a knob is ever removed). Both literal forms are accepted so a helper that
# reads one key directly is not silently missed.
SETTINGS_HELPERS = ("_apply_merger_settings", "_apply_eye_restore_settings",
                    "_apply_parser_region_settings", "_apply_enhancer_settings")


def _helper_keys(name):
    src = _api_source()
    start = src.index(f"def {name}(")
    body = src[start:src.index("\ndef ", start + 1)]
    return (set(re.findall(r'\(\s*"([a-z_0-9]+)"\s*,\s*[-0-9.]+\s*\)', body))
            | set(re.findall(r'payload\.get\(\s*"([a-z_0-9]+)"', body)))


def merger_helper_keys():
    return _helper_keys("_apply_merger_settings")


def _consumed(body):
    keys = set(re.findall(r'payload\.get\(\s*["\']([a-z_0-9]+)["\']', body))
    for helper in SETTINGS_HELPERS:
        if f"{helper}(payload)" in body:
            keys |= _helper_keys(helper)
    return keys


def preview_consumed_keys():
    return _consumed(_function_body(_api_source(), 'def preview('))


def run_consumed_keys():
    return _consumed(_function_body(_api_source(), "batch_process_regular("))


# Sent by the frontend but not read via payload.get in preview(): these are
# consumed by _update_mask_offsets_from_payload(payload), called at the top of
# preview() and of the run endpoint.
VIA_MASK_OFFSET_HELPER = {
    "mask_top", "mask_bottom", "mask_left", "mask_right",
    "face_mask_blend", "mouth_mask_blend",
    "mouth_top_scale", "mouth_bottom_scale", "mouth_left_scale", "mouth_right_scale",
}

# Preview-only concepts, meaningless to a full render.
PREVIEW_ONLY = {"index", "frame", "fake_preview", "show_mask_offsets"}

# Run-only: temporal settings that need more than one frame, and output/pipeline
# settings that do not affect pixels. A single-frame preview cannot show these,
# so their absence from preview() is correct, not an oversight.
RUN_ONLY_TEMPORAL = {
    "stabilize_enhancer", "stabilize_enhancer_strength", "stabilize_beta",
    "stabilize_min_cutoff", "track_identities", "temporal_detection",
    "lipsync_enabled", "stabilize_mask", "stabilize_mask_strength",
    # Both filters in roop/temporal_smoother.py refuse to act unless their
    # stored state is frame_index - 1 for the same track, so on a single
    # preview frame they are seeds and nothing else. Wiring them into preview()
    # would show the user a frame in which they demonstrably did nothing,
    # which is a worse answer than their absence.
    "stabilize_landmarks", "stabilize_hf_texture", "stabilize_hf_texture_weight",
}
RUN_ONLY_OUTPUT = {
    "keep_frames", "skip_audio", "output_method", "video_method",
    "wait_after_extraction", "interp_after_swap",
    "upscale_after_swap", "upscale_model_after", "target_index",
}


class TestFrontendReachesBackend(unittest.TestCase):
    def test_every_sent_setting_is_consumed(self):
        """A key in the request that the backend never reads is a control that
        silently does nothing — the exact bug this suite exists for."""
        unconsumed = frontend_payload_keys() - preview_consumed_keys() - VIA_MASK_OFFSET_HELPER
        self.assertEqual(unconsumed, set(), f"FaceSwap.jsx sends {sorted(unconsumed)} "
                                            f"but /api/preview never reads them")

    def test_merger_helper_is_actually_called_by_both_paths(self):
        """The helper only excuses those keys while it really runs — and it has
        to run on BOTH paths. A preview that skipped it would be previewing
        something other than what the render produces, which is worse than the
        setting not existing."""
        src = _api_source()
        self.assertTrue(merger_helper_keys(), "no keys parsed out of "
                                              "_apply_merger_settings — did its shape change?")
        for marker in ('def preview(', 'batch_process_regular('):
            self.assertIn("_apply_merger_settings(payload)",
                          _function_body(src, marker),
                          f"{marker} never applies the merger settings")

    def test_merger_knobs_default_to_neutral(self):
        """Every merger op is a no-op at 0, so a fresh install must start there
        — a non-zero default would silently change everyone's output."""
        import roop.globals as g
        for key in sorted(merger_helper_keys()):
            self.assertEqual(float(getattr(g, key)), 0.0,
                             f"roop.globals.{key} does not default to neutral")

    def test_mask_offset_helper_is_actually_called(self):
        """VIA_MASK_OFFSET_HELPER above is only a valid excuse while preview()
        really does call the helper."""
        body = _function_body(_api_source(), 'def preview(')
        self.assertIn("_update_mask_offsets_from_payload(payload)", body)

    def test_the_builder_is_the_only_preview_request_site(self):
        """Any hand-written /api/preview body would sit outside the cache
        signature and reintroduce the stale-preview bug.

        Scanned across FaceSwap.jsx AND the faceswap/ modules: the comparison
        grids' loader moved into a hook, so counting only the component would
        now miss three of the request sites — and, worse, would stop noticing
        if one of them were rewritten by hand.
        """
        sources = [FACESWAP_JSX.read_text(encoding="utf-8")]
        for path in sorted((FACESWAP_JSX.parent / "faceswap").glob("*.js*")):
            sources.append(path.read_text(encoding="utf-8"))
        src = "\n".join(sources)

        literals = re.findall(r"postJSON\('/api/preview',\s*\{", src)
        self.assertEqual(literals, [], "found a hand-built /api/preview body; "
                                       "route it through buildPreviewPayload")

        # The real assertion is the one above. This second one only proves the
        # regex still matches something, so an edit that renames the builder
        # cannot make the check vacuously pass.
        #
        # The floor was 5 when the enhancer, mask and swapper grids each had
        # their own copy of the same loader. Those are one hook now, so three
        # call sites legitimately became one: refreshPreview, the upscale grid's
        # single base swap, and useGridPreviewLoader. Fewer sites is the point.
        calls = len(re.findall(r"postJSON\('/api/preview',\s*buildPreviewPayload\(", src))
        self.assertGreaterEqual(calls, 3)

    def test_cache_signature_is_derived_from_the_payload(self):
        """If these ever diverge, a setting can be sent without invalidating the
        cache, which is what made toggles appear to do nothing."""
        src = FACESWAP_JSX.read_text(encoding="utf-8")
        self.assertRegex(src, r"const previewSignature = \([^)]*\) =>\s*"
                              r"JSON\.stringify\(buildPreviewPayload\(")
        self.assertIn("const previewKey = previewSignature(", src)
        self.assertNotIn("const previewKey = JSON.stringify({", src)


class TestPreviewAndRunAgree(unittest.TestCase):
    def test_preview_extras_are_preview_only(self):
        extra = preview_consumed_keys() - run_consumed_keys()
        self.assertEqual(extra - PREVIEW_ONLY, set(),
                         f"preview() reads {sorted(extra - PREVIEW_ONLY)} that the "
                         f"run path ignores — a preview would then not match the render")

    def test_run_extras_are_temporal_or_output_only(self):
        extra = run_consumed_keys() - preview_consumed_keys()
        unexplained = extra - RUN_ONLY_TEMPORAL - RUN_ONLY_OUTPUT - VIA_MASK_OFFSET_HELPER
        self.assertEqual(unexplained, set(),
                         f"the run path reads {sorted(unexplained)} that preview() "
                         f"does not. If it affects a single frame, wire it into "
                         f"preview(); if not, add it to RUN_ONLY_* with a reason")

    def test_run_only_settings_are_not_sent_to_preview(self):
        """Sending a temporal/output setting to the preview endpoint would imply
        it changes the frame, and would bloat the cache signature so unrelated
        changes discard cached previews."""
        leaked = frontend_payload_keys() & (RUN_ONLY_TEMPORAL | RUN_ONLY_OUTPUT)
        self.assertEqual(leaked, set(), f"{sorted(leaked)} are run-only but are in "
                                        f"the preview payload")


class TestSettingsPersistence(unittest.TestCase):
    pass




    def test_settings_defaults_are_exposed_for_the_ui(self):
        """GET /api/settings returns CFG.__dict__, so a field missing from
        Settings.__init__ never reaches the UI."""
        from settings import Settings
        cfg = Settings(str(REPO / "app" / "config.yaml"))
        for field in ("refine_landmarks", "swap_model", "mask_engine"):
            self.assertIn(field, cfg.__dict__, f"{field} missing from Settings")

    def test_saved_settings_round_trip(self):
        """Every field the UI can POST back must also be written by save(), or
        it silently resets on restart."""
        saved = re.findall(r"'([a-z_0-9]+)':\s*self\.",
                           _function_body(
                               (REPO / "app" / "settings.py").read_text(encoding="utf-8"),
                               "def save("))
        for field in ("refine_landmarks", "jaw_reshape"):
            self.assertIn(field, saved, f"{field} is loaded but never saved")


class TestPerfKnobWiring(unittest.TestCase):
    """The env-backed 'Advanced performance' knobs, which have a LONGER chain
    than the render settings above and no other test covering them.

    A perf knob reaches the code through five places, and missing any one fails
    silently in a different way: Settings.jsx (invisible), Settings.__init__
    (never reaches the UI, since GET returns CFG.__dict__), save() (resets on
    restart), run.py (saved but never applied — the worst, because the UI shows
    the value the run is not using), and finally the env var the consumer reads.
    """

    def _jsx_perf_keys(self):
        src = (REPO / "react-ui" / "src" / "components" / "Settings.jsx").read_text(encoding="utf-8")
        # Settings.jsx binds a control to its key through `bind('key')` /
        # `bindToggle('key')`, which also carries the default-drift marker and
        # the per-control reset. A bare `set('key', v)` is still used for the
        # handful of writes that are not a single bound control (theme edits),
        # so both shapes count as "the UI can change this".
        return set(re.findall(r"\b(?:bind|bindToggle|set)\(\s*'(perf_[a-z_0-9]+)'", src))

    def _settings_source(self):
        return (REPO / "app" / "settings.py").read_text(encoding="utf-8")

    def test_every_ui_perf_knob_is_loaded_and_saved(self):
        src = self._settings_source()
        saved = set(re.findall(r"'(perf_[a-z_0-9]+)':\s*self\.", src))
        keys = self._jsx_perf_keys()
        self.assertTrue(keys, "no perf knobs found in Settings.jsx — regex stale?")
        for key in sorted(keys):
            # Two readers, both of which load the key: default_get, and _hw_get
            # for values DERIVED FROM THE GPU, which revert to their auto default
            # when the config is opened on a different card. The question this
            # guard asks is unchanged — is the key read at all.
            self.assertRegex(src, rf"self\.{key}\s*=\s*self\.(?:default_get|_hw_get)\(",
                             f"{key} is in the UI but not loaded by Settings.__init__")
            self.assertIn(key, saved, f"{key} is loaded but never written by save()")

    def test_every_ui_perf_knob_reaches_an_env_var(self):
        """run.py is what turns a config value into the env var the consumer
        reads; a knob missing here saves cleanly and does nothing."""
        run_src = (REPO / "app" / "run.py").read_text(encoding="utf-8")
        for key in sorted(self._jsx_perf_keys()):
            # Two shapes: _set(VAR, cfg.get(key)) for free-form values, and the
            # (var, key) tuple loop for the auto/on/off tri-states.
            self.assertTrue(
                re.search(rf"_set\('([A-Z_]+)',\s*cfg\.get\('{key}'\)\)", run_src)
                or re.search(rf"'{key}'\)", run_src),
                f"{key} is in the UI but run.py never maps it to an env var")

    def test_the_expression_pool_knob_is_wired_end_to_end(self):
        """The most recently added one, and the one whose absence prompted this
        test: it must land on the exact env var session_pool reads."""
        self.assertIn("perf_expr_pool", self._jsx_perf_keys())
        self.assertIn("_set('ROOP_EXPR_POOL', cfg.get('perf_expr_pool'))",
                      (REPO / "app" / "run.py").read_text(encoding="utf-8"))
        self.assertIn("ROOP_EXPR_POOL",
                      (REPO / "app" / "roop" / "session_pool.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
