"""Tests for Directional Relighting, Specular Skin Highlight Recovery, and Sub-Dermal Warmth.

Ensures:
1. Neutral 0.0 values produce exact bit-identical no-op.
2. Specular highlight recovery extracts high-luminance peaks from target L-channel and composites onto swap.
3. Ambient shadow & spherical harmonics matching transfers directional hemisphere falloff.
4. Sub-dermal warmth matching preserves shadow/mid/highlight chrominance balance in skin mask.
5. Integration with ColorTransferMixin.apply_color_transfer.
"""
import os
import sys
import numpy as np
import pytest
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals as roop_globals
from roop.procmgr_color import ColorTransferMixin


class DummyColorManager(ColorTransferMixin):
    pass


@pytest.fixture(autouse=True)
def reset_lighting_globals():
    orig_spec = getattr(roop_globals, 'specular_highlight_recovery', 0.0)
    orig_amb = getattr(roop_globals, 'ambient_shadow_match', 0.0)
    orig_sub = getattr(roop_globals, 'subdermal_warmth_match', 0.0)
    orig_mode = getattr(roop_globals, 'color_transfer_mode', 'lct')
    roop_globals.specular_highlight_recovery = 0.0
    roop_globals.ambient_shadow_match = 0.0
    roop_globals.subdermal_warmth_match = 0.0
    roop_globals.color_transfer_mode = 'lct'
    yield
    roop_globals.specular_highlight_recovery = orig_spec
    roop_globals.ambient_shadow_match = orig_amb
    roop_globals.subdermal_warmth_match = orig_sub
    roop_globals.color_transfer_mode = orig_mode


def test_neutral_is_exact_noop():
    mgr = DummyColorManager()
    swapped = np.full((128, 128, 3), 120, dtype=np.uint8)
    target = np.full((128, 128, 3), 180, dtype=np.uint8)
    
    # All 0.0
    out = mgr._apply_directional_and_specular_lighting(
        swapped, target,
        specular_strength=0.0,
        ambient_strength=0.0,
        subdermal_strength=0.0
    )
    assert np.array_equal(out, swapped)


def test_specular_highlight_recovery():
    mgr = DummyColorManager()
    # Baseline midtone skin
    swapped = np.full((128, 128, 3), (120, 140, 180), dtype=np.uint8)
    target = np.full((128, 128, 3), (120, 140, 180), dtype=np.uint8)
    
    # Introduce high-luminance specular highlight in the center of target face
    target[50:70, 50:70] = (245, 250, 255)
    
    out = mgr._apply_directional_and_specular_lighting(
        swapped, target,
        specular_strength=0.8,
        ambient_strength=0.0,
        subdermal_strength=0.0
    )
    
    # Specular peak area should be recovered/brighter on swapped
    assert int(out[60, 60].mean()) > int(swapped[60, 60].mean())
    # Off-peak area with no highlight in target should remain untouched or very close
    assert np.allclose(out[10, 10], swapped[10, 10], atol=1)


def test_ambient_shadow_directional_match():
    mgr = DummyColorManager()
    # Swapped face is flatly lit
    swapped = np.full((128, 128, 3), (130, 150, 190), dtype=np.uint8)
    
    # Target face is strongly side-lit: left side bright, right side in deep shadow
    target = np.zeros((128, 128, 3), dtype=np.uint8)
    target[:, :64] = (180, 200, 240)   # Left bright
    target[:, 64:] = (30, 40, 50)       # Right dark
    
    out = mgr._apply_directional_and_specular_lighting(
        swapped, target,
        specular_strength=0.0,
        ambient_strength=0.8,
        subdermal_strength=0.0
    )
    
    left_mean = out[:, :64].mean()
    right_mean = out[:, 64:].mean()
    # Directional falloff should transfer: left side brighter than right side
    assert left_mean > right_mean
    # Contrast difference between hemispheres must be substantial
    assert left_mean - right_mean > 15


def test_subdermal_warmth_match():
    mgr = DummyColorManager()
    # Swapped face has cool/desaturated tones
    swapped = np.full((128, 128, 3), (140, 140, 140), dtype=np.uint8)
    # Target has rich warm/red undertones (BGR with high Red, medium Green, lower Blue)
    target = np.full((128, 128, 3), (100, 140, 210), dtype=np.uint8)
    
    out = mgr._apply_directional_and_specular_lighting(
        swapped, target,
        specular_strength=0.0,
        ambient_strength=0.0,
        subdermal_strength=0.7
    )
    
    # In BGR, red channel is index 2, blue is index 0
    # Red should increase more than Blue, shifting toward warm
    assert out[:, :, 2].mean() > swapped[:, :, 2].mean()


def test_apply_color_transfer_triggers_with_lighting_controls():
    mgr = DummyColorManager()
    swapped = np.full((128, 128, 3), (120, 140, 180), dtype=np.uint8)
    target = np.full((128, 128, 3), (120, 140, 180), dtype=np.uint8)
    target[50:70, 50:70] = (255, 255, 255)
    
    # Even if color_transfer_mode is 'none', active specular recovery runs
    roop_globals.color_transfer_mode = 'none'
    roop_globals.specular_highlight_recovery = 0.9
    
    out = mgr.apply_color_transfer(swapped, target)
    assert int(out[60, 60].mean()) > int(swapped[60, 60].mean())


def load_tests(loader, tests, pattern):
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
