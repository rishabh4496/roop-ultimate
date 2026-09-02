# React UI 2.0 Preview Subsystem Validation & Verification Guide

This document defines the manual and automated verification procedures for the **React UI 2.0 Preview & Tracking Subsystem**.

---

## 1. Automated Test Suite

Run the automated coordinate transformation and overlay rendering unit tests:

```bash
cd app
env\Scripts\python.exe -m unittest tests/test_ui2_preview_coordinates.py
```

### Verified Assertions:
- **16:9 Aspect Ratio Coordinate Mapping:** `1920x1080` frame coordinate translation.
- **4K Ultra-HD Coordinate Mapping:** `3840x2160` frame coordinate translation.
- **9:16 Portrait Video Mapping:** `1080x1920` vertical video coordinate translation.
- **1:1 Square & 21:9 Ultrawide:** `1024x1024` and `3440x1440` letterbox/pillarbox alignment.
- **5-Point ArcFace Geometry:** Eye midpoint and mouth midpoint rotation vectors.
- **3D Head Pose String Formatting:** Solved yaw/pitch/roll degree formatting.
- **Zoom & Pan Clamping:** Sub-pixel bounds enforcement at 1× to 8× zoom factors.

---

## 2. Multi-Resolution & Aspect Ratio Manual Checklist

Perform this checklist across multiple test media to verify visual rendering accuracy:

| Aspect Ratio | Test Media Spec | Target Verification Criteria | Pass / Fail |
| :--- | :--- | :--- | :--- |
| **16:9 Widescreen** | `1920x1080` MP4 | Bounding boxes align over face contours without vertical or horizontal drift. | [ ] |
| **9:16 Portrait** | `1080x1920` WebP / MP4 | Pillarboxing is symmetrically centered; SVG overlay aligns with facial landmarks. | [ ] |
| **1:1 Square** | `1024x1024` PNG / JPG | Face box percentage matches `left` and `top` offsets without squishing. | [ ] |
| **21:9 Cinematic** | `2560x1080` MKV | Letterboxing is top/bottom padded; scrubber deck remains docked. | [ ] |
| **4K Ultra-HD** | `3840x2160` MOV | Sub-pixel rendering stays crisp; 3.5× loupe magnifier inspects native pixels. | [ ] |

---

## 3. Interactive Tooling & Gesture Matrix

1. **Flicker-Free Crossfade:**
   - Scrub playhead across video frames.
   - Verify: No white flashes or blank frames during scrubbing.
2. **Sub-Pixel Pan & Wheel Zoom:**
   - Scroll mouse wheel over face; verify smooth zoom in/out (1× to 8×).
   - Drag pointer to pan; verify pan clamps strictly to image boundaries.
   - Double-click face; verify centering zoom to 2.5×; double-click again to reset fit.
3. **Face Landmarking & Pose Vectors (Hotkey `D`):**
   - Press `D` to toggle landmarks.
   - Verify: Cyan eye axis, pink mouth axis, green crop axis, and degree text (`y...° p...° r...°`) align on facial features.
4. **Comparison Wipe Suite (Hotkey `X`, `O`, `A`):**
   - Click "Preview Swap".
   - Press `X` to switch vertical/horizontal wipe.
   - Press `O` to toggle 50% alpha blend.
   - Press `A` to toggle automatic sinusoidal sweeping.
5. **3.5× Magnifying Loupe (Hotkey `G`):**
   - Press `G` and hover over eyes and lips.
   - Verify: Native resolution pixels appear centered inside the circular reticle.
6. **Dual-Canvas Mask Brush Tool (Hotkey `B`):**
   - Press `B` and paint over face area.
   - Verify: Translucent pink brush marks appear on screen; off-screen canvas exports solid white mask PNG.

---

## 4. Hardware Profile Constraints

- **Desktop (RTX 4070, 12GB VRAM):** Full 60 FPS preview rendering, simultaneous comparison wipes, and zero frame-drop scrubbing.
- **Laptop (RTX 3060, 6GB VRAM):** Render-Lite active during swap jobs; overlays smoothly throttle to preserve GPU compute for inference.
