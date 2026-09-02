# React UI 2.0 UX/UI Forensic Review & Enhancement Plan

---

## 1. Evaluation of Current 20 Core Dimensions

| Dimension | Evaluation | Issue Identified | Concrete Correction |
| :--- | :--- | :--- | :--- |
| **1. Information Hierarchy** | Needs Polish | Fluffy marketing copy ("Make the moment yours") and oversized hero banners waste vertical space. | Replace with compact technical session bar showing Target, Source, Engine, and Status. |
| **2. Preview Prominence** | Compromised | `width: min(1180px, 100%)` container and `72px` padding constrain the preview canvas to a narrow column. | Expand workstation to 100% viewport width with full-bleed desktop layout. |
| **3. Face-Selection Discoverability** | Good | Source and target panels function well but card padding is slightly chunky. | Tighten thumbnail cards, sharpen selection rings, add explicit `#1 Active` badges. |
| **4. Tracking Visibility** | Excellent | 5-point ArcFace landmark lines and 3D pose degrees are accurate and mathematically aligned. | Retain and enhance contrast with dark outer glow. |
| **5. Control Density** | Medium | Cards have 22px padding and oversized borders, causing unnecessary vertical scroll. | Switch to 12px pro workstation panel padding with 1px hairline dividers. |
| **6. Navigation Clarity** | Good | Nav sidebar works well; topbar needs tighter integration with active session. | Unify topbar with target media metadata and telemetry status. |
| **7. Alignment & Spacing** | Irregular | Varied button paddings and input heights across different components. | Standardize to 32px height for controls, 26px for compact tools, 40px for primary actions. |
| **8. Typography** | Inconsistent | Headings were too large (`54px`), meta labels were too small (`9px`). | Adopt strict desktop scale: 18px page title, 13px section headers, 12px body, 11px mono readouts. |
| **9. Button Hierarchy** | Good | Action buttons distinct; secondary buttons need crisper borders. | Standardize primary (Electric Blue/Emerald), secondary (Dark raised), danger (Muted coral red). |
| **10. Panel Widths** | Responsive | Sidebar was static 310px. | Use dynamic CSS Grid `minmax(0, 1fr) 340px` with collapsible drawer toggles. |
| **11. Scroll Behavior** | Suboptimal | Whole page scrolls vertically, scrolling the preview off-screen. | Lock preview stage in viewport; place scrollbars strictly inside asset/control sidebars. |
| **12. Responsive Behavior** | Strong | Adapts down to 1366×768; mobile layout wraps cleanly. | Polish compact desktop drawer toggles for 1280×720 and 1366×768. |
| **13. Empty States** | Clean | Empty states have good prompts. | Add drag-and-drop hover highlights and one-click demo presets. |
| **14. Loading States** | Clean | Skeleton and spinners present. | Use subtle progress pulses without blocking entire UI. |
| **15. Error States** | Strong | Error notices preserve backend details. | Ensure inline banners don't displace canvas layout. |
| **16. Processing State** | Excellent | Live frame peek, SVG progress ring, and real-time telemetry are clear. | Add unified pipeline summary pill ("Swapping 1 face with InSwapper + GPEN"). |
| **17. Before/After Workflow**| Excellent | Wipe slider, blend mode, diff mode, and auto-swipe operate smoothly. | Maintain current hotkeys (`X`, `O`, `A`, `G`, `B`, `D`). |
| **18. Advanced Settings** | Improved | Disclosure card present. | Organize into clean segmented tabs (Engine/TRT, Masking/SAM2, Output/Codec). |
| **19. Visual Consistency** | Polished | Token system unified across dark surfaces. | Eliminate legacy rounded pill badges (`border-radius: 99px`) in favor of 6px radius. |
| **20. Desktop Aesthetic** | Modern Pro | Moves decisively away from generic SaaS towards a professional media workstation. | Complete removal of decorative radial gradient orbs. |

---

## 2. Elimination of Amateur Elements

1. **Eliminate Oversized Cards:** Remove bulky card containers; replace with crisp workstation panels framed by `1px solid rgba(255,255,255,0.08)`.
2. **Eliminate Excessive Rounded Corners:** Standardize on `6px` (`--radius-sm`) and `10px` (`--radius-md`) instead of `99px` pills and `24px` bubbles.
3. **Eliminate Decorative Gradient Orbs:** Remove background floating glowing orbs and landing page radial gradients.
4. **Eliminate Unnecessary Vertical Page Scrolling:** Lock the center stage so the video canvas never scrolls away while adjusting parameters.
5. **Eliminate Marketing Headings:** Replace "Make the moment yours" with real-time technical status:
   `target_1080p.mp4 • 1920×1080 • Frame 142/600 • 2 Faces Tracked • CUDA FP16`.

---

## 3. The 6 Core Workstation Questions (Visual Answers)

1. **WHAT MEDIA AM I WORKING ON?**  
   $\to$ Displayed in top technical session bar: filename, resolution, duration, FPS, current frame.
2. **WHAT FACES ARE DETECTED?**  
   $\to$ Preview top-left badge: `2 FACES DETECTED` + glowing bounding boxes with `#1`, `#2` badges.
3. **WHICH FACE(S) ARE SELECTED?**  
   $\to$ Active face glowing in Cyan `#38bdf8` with `#1 ACTIVE` badge in Source Gallery and Preview.
4. **IS TRACKING WORKING?**  
   $\to$ 5-point ArcFace landmark lines + real-time 3D pose readout (`y-14° p4° r2°`) rendered on face.
5. **WHAT WILL HAPPEN IF I PRESS PROCESS?**  
   $\to$ Execution badge in Action Bar: `Ready: Swap Person 1 -> Source #1 (InSwapper + GPEN 256)`.
6. **HOW FAR ALONG IS PROCESSING?**  
   $\to$ Bottom status dock with SVG progress ring, frame counter `1420/1800`, `14.2 FPS`, `ETA: 00:26`, and Live Peek popover.
