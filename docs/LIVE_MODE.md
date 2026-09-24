# Real-Time / Live Webcam Mode

Roop Ultimate's live mode is a separate path from the batch video renderer. It captures frames with the native webcam backend, keeps only the newest frame, swaps detector keyframes, and uses Lucas-Kanade optical flow for the intervening frames.

## Capture and virtual camera

The capture backend is selected automatically:

- Windows: DirectShow
- Linux: V4L2
- macOS: AVFoundation

The requested driver queue is `buffer_size=1`, and the application uses a one-slot latest-frame mailbox as a hard latency bound even when a camera driver ignores that property. No intermediate frame is written to disk.

Enable **Stream to virtual camera (OBS)** to send raw RGB24 frames through `pyvirtualcam`. Install the OBS Virtual Camera, Unity Capture, or another supported pyvirtualcam backend first. OBS, Zoom, Discord, and browser camera pickers can then select the published device.

## Inference policy

The live graph disables CodeFormer, GPEN, UltraMax, and other multi-pass enhancers. Detection defaults to every sixth frame. Between keyframes, the processor tracks each detected face with Lucas-Kanade point flow, estimates a local affine motion, and warps only the prior swapped face mask onto the newest camera frame. Tracking failure forces an early keyframe rather than emitting a stale warp.

A content-aware histogram detector watches for shot changes. On a cut it invalidates the optical-flow state and calls the shared temporal flush hook so landmarks, face-swap temporal state, and cached appearance data cannot smear across the boundary.

The live status endpoint exposes processing FPS, measured latency, detector cadence, dropped capture frames, flow-frame count, and scene-cut count. The 45 ms target is a runtime budget and should be verified on the target camera resolution, selected face model, and hardware. It is not a guarantee for an overloaded GPU or a slow camera driver.

## Audio passthrough

Install `sounddevice` and select a microphone input plus a virtual audio cable output, such as VB-CABLE on Windows or BlackHole on macOS. The audio bridge uses an in-memory circular delay line and continuously follows the measured visual latency. No audio file is created. If no virtual cable is installed, webcam video remains available and audio is simply disabled.

The API endpoints are:

- `GET /api/livecam/status`
- `GET /api/livecam/audio/devices`
- `POST /api/livecam/start`
- `POST /api/livecam/stop`
- `GET /api/livecam/frame`

Example start payload:

```json
{
  "cam_number": 0,
  "resolution": "1280x720",
  "fps": 30,
  "detector_interval": 6,
  "stream_obs": true,
  "audio_input_device": "Microphone",
  "audio_output_device": "CABLE Input"
}
```
