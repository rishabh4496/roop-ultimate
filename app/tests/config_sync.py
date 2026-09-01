"""The one implementation of "make this bench render what the user runs".

WHY THIS IS SHARED. The rule in CLAUDE.md -- read `app/config.yaml` live and
bench the models the user actually runs -- has been violated three times, each
time by a different harness, each time invalidating whole sessions:

  * every saved `yaw_*` arm ran the swap-model mask OFF while production ran 25;
  * every arm rendered through `angle_bench` before 2026-08-23 ran the entire
    merger stage OFF (hist / sharpen / grain / degrade all 0.0) while production
    ran 0.4 / 0.35 / 0.45 / 0;
  * `two_face_video.py` -- the end-to-end harness the Phase and Gate campaign
    runs through, via `baseline_controlled.py` -- ran `target_conditioned_
    appearance` False against a config.yaml carrying True, and with it
    `detail_transfer_strength` 0.0 against 0.4 and `color_match_after_enhance`
    False against True.

The third one is why this module exists rather than a fourth copy of the loop:
the correct implementation was already written, for exactly this reason, in
`compare_enhancers_video.py`, and simply never reached the harness that mattered
most. A per-key list in each harness reproduces the defect the moment somebody
adds a setting; a shared sync plus `test_bench_config_parity.py` does not.

Comparisons taken with a key unstated are not necessarily wrong -- both arms of
an A/B are equally off -- but their ABSOLUTE values are not production, and any
quality grading against real footage is measuring a stack nobody ships.
"""

# Keys whose config.yaml spelling is NOT their roop.globals spelling. A blanket
# copy corrupts these, so each one names the translator production uses.
#
#   no_face_action  config holds the dropdown LABEL, globals holds the
#                   eNoFaceAction int. Copying the string makes every `==`
#                   against the enum false, so none of the no-face actions fire
#                   at all -- and nothing in the output says so.
#   verify_swap     config holds 'auto'|'on'|'off', globals holds a bool that
#                   run.py derives via ROOP_VERIFY_SWAP.
TRANSLATED = {'no_face_action', 'verify_swap'}


def sync_globals_from_config(g, verbose=True, prefix="[config]"):
    """Push every config.yaml key that roop.globals also defines onto globals.

    The point is not the copying -- it is that an unstated setting becomes
    impossible, so a future reader does not have to guess whether this bench
    included the stage they care about.

    A key is copied only when config's value has the same TYPE as globals'. A
    type mismatch means the two layers use different representations and a
    translator exists somewhere -- see `TRANSLATED`. Copying across one of those
    is silent corruption, not a config sync.

    Returns the list of (key, was, now) it changed.
    """
    changed, skipped = [], []
    for k, v in vars(g.CFG).items():
        if k.startswith('_') or k in TRANSLATED or not hasattr(g, k):
            continue
        cur = getattr(g, k)
        if cur is not None and not isinstance(v, type(cur)) and not (
                isinstance(cur, (int, float)) and isinstance(v, (int, float))):
            skipped.append((k, cur, v))
            continue
        if cur != v:
            changed.append((k, cur, v))
            setattr(g, k, v)

    # The two translated keys, the way api.py and run.py do them.
    from api import index_of_no_face_action
    g.no_face_action = index_of_no_face_action(g.CFG.no_face_action)
    g.verify_swap = str(getattr(g.CFG, 'verify_swap', 'auto')).lower() != 'off'

    if verbose:
        print(f"{prefix} config.yaml -> roop.globals:")
        for k, was, now in sorted(changed):
            print(f"{prefix}   {k}: {was!r} -> {now!r}")
        print(f"{prefix}   no_face_action: {g.CFG.no_face_action!r} -> "
              f"{g.no_face_action} (translated)")
        print(f"{prefix}   verify_swap: {g.CFG.verify_swap!r} -> "
              f"{g.verify_swap} (translated)")
        for k, cur, v in sorted(skipped):
            print(f"{prefix}   SKIPPED {k}: type mismatch "
                  f"{type(cur).__name__} vs {type(v).__name__}")
    return changed
