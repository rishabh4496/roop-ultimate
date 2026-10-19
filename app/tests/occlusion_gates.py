"""Thresholds for the Stage 19 interaction scenes: the brief's, and the ratchets.

BRIEF   what the Stage 19 brief asked for, verbatim. Some are not met; each unmet one is a
        strict xfail in tests/test_occlusion_interaction.py, so it cannot start passing (or
        regress further) without somebody noticing.
RATCHET what the pipeline does today, with a little headroom. A change that makes any of
        these worse fails the test; a change that makes them better should lower them.

Measured 2026-10-19 on the RTX 4070 (see docs/CHANGELOG.md, Stage 19).
"""
import operator

_OPS = {'==': operator.eq, '<=': operator.le, '<': operator.lt, '>=': operator.ge}


def _get(m, path):
    for key in path.split('.'):
        m = m[key]
    return m


# (id, metric path, op, limit, text)
BRIEF = [
    ('t1_peak_iou', 'cross_bystander.peak_iou', '>=', 0.74, 'scene peaks at sprite IoU 0.75'),
    ('t1_target_swapped_front', 'cross_target.target_unswapped', '==', 0,
     'every detected target frame swapped (target in front)'),
    ('t1_target_swapped_behind', 'cross_bystander.target_unswapped', '==', 0,
     'every detected target frame swapped (target behind the bystander)'),
    ('t1_target_found_behind', 'cross_bystander.target_detected', '==', 30,
     'target detected on all 30 frames while the bystander covers it'),
    ('t1_zero_bystander_swaps_front', 'cross_bystander.bystander_swapped', '==', 0,
     'zero frames swapped onto the bystander (bystander in front)'),
    ('t1_zero_bystander_swaps_behind', 'cross_target.bystander_swapped', '==', 0,
     'zero frames swapped onto the bystander (bystander behind)'),
    ('t1_bystander_mse_front', 'cross_bystander.bystander_mse_max', '==', 0.0,
     'bystander unoccluded-pixel MSE strictly 0.0 (bystander in front)'),
    ('t1_bystander_mse_behind', 'cross_target.bystander_mse_max', '==', 0.0,
     'bystander unoccluded-pixel MSE strictly 0.0 (bystander behind)'),
    ('t2_translation', 'alignment.translation_max', '<', 0.75, 'alignment translation deviation < 0.75 px'),
    ('t2_rotation', 'alignment.rotation_max', '<', 0.4, 'alignment rotation deviation < 0.4 deg'),
    ('t3_delta_e', 'flicker.de_max', '<', 0.8, 'skin LAB dE vs the no-bar render < 0.8 per frame'),
]

# Measured 2026-10-19, RTX 4070, three runs of the grading on one set of renders. Headroom is
# small on purpose: a lossless render of a seeded scene is repeatable.
RATCHET = [
    ('r_peak_iou', 'cross_bystander.peak_iou', '>=', 0.74, 'scene peaks at sprite IoU 0.75'),
    ('r_front_bystander_swaps', 'cross_bystander.bystander_swapped', '==', 0,
     'no frame swapped onto the bystander (bystander in front)'),
    ('r_behind_bystander_swaps', 'cross_target.bystander_swapped', '==', 0,
     'no frame swapped onto the bystander (bystander behind)'),
    ('r_front_target_unswapped', 'cross_bystander.target_unswapped', '==', 0,
     'every detected target frame swapped (bystander in front)'),
    ('r_behind_target_unswapped', 'cross_target.target_unswapped', '==', 0,
     'every detected target frame swapped (target in front)'),
    ('r_front_target_found', 'cross_bystander.target_detected', '>=', 22,
     'target still detected on >= 22/30 frames while covered (measured 24)'),
    ('r_front_bystander_mse', 'cross_bystander.bystander_mse_max', '<=', 0.05,
     'bystander pixels unchanged vs the target-free control (measured exactly 0)'),
    ('r_behind_bystander_mse', 'cross_target.bystander_mse_max', '<=', 0.5,
     'bystander halo next to the swapped target stays small (measured 0.30)'),
    ('r_behind_bystander_px', 'cross_target.bystander_changed_max', '<=', 80,
     'bystander pixels changed by > 2 levels per frame (measured 42)'),
    ('r_behind_bystander_dmax', 'cross_target.bystander_dmax', '<=', 8,
     'largest bystander pixel change, levels (measured 5)'),
    ('r_align_matched', 'alignment.matched', '>=', 26, 'bar scene: detector finds one face on >= 26/30 frames (measured 28)'),
    ('r_align_translation', 'alignment.translation_max', '<=', 40.0,
     'GAP, not a target: crop drift under the bar must not grow past 40 px (measured 30.4)'),
    ('r_align_rotation', 'alignment.rotation_max', '<=', 40.0,
     'GAP, not a target: crop rotation under the bar must not grow past 40 deg (measured 33.0)'),
    ('r_flicker_d2', 'flicker.d2_bar', '<=', 1.6,
     'second-order luma derivative on swapped skin, levels (measured 1.21; no-bar 0.54)'),
    ('r_flicker_de', 'flicker.de_max', '<=', 1.1, 'skin LAB dE vs the no-bar render, max (measured 0.91, mean 0.39)'),
]

# BRIEF entries the pipeline does not meet, each with the measured reason.
KNOWN_MISSES = {
    't1_target_found_behind': 'the detector cannot see a face more than ~55% covered: 6/30 frames have no target to swap',
    't1_bystander_mse_behind': 'a faint halo (<= 5 levels, ~42 px/frame) where the target-in-front swap meets the bystander',
    't2_translation': 'the 5-keypoint detector itself moves with the bar: crop drift up to 30 px (Stage 17 fixes all lost end to end)',
    't2_rotation': 'same: up to 33 deg while the bar crosses the eyes',
    't3_delta_e': 'skin colour differs from the no-bar render by up to 0.91 dE on the worst frame (mean 0.39)',
}


def check(metrics, spec):
    out = []
    for name, path, op, limit, text in spec:
        value = _get(metrics, path)
        ok = bool(_OPS[op](value, limit))
        out.append((name, ok, f'{text}: {path} = {value:.4g} (want {op} {limit})'))
    return out
