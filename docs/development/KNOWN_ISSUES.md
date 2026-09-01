# Known Issues and Open Questions

## CURRENT IMPLEMENTATION / VERIFIED

1. **Phase 16 is open.** The final report has 17 missing clips, 425 planned rows, zero complete runs, and no winners. This is the primary release-blocking validation issue.
2. **3060 RSS acceptance is not met.** Existing hardware records report the strict `<2.5 GB` gate as failing, while decomposing the floor into stabilization/temporal and enhancer residency rather than a monotonic leak.
3. **3060 TensorRT precision E2E is intentionally unadmitted.** The sub-7GB provider policy chooses CUDA/CPU unless the explicit override is used; therefore TRT precision rows cannot be treated as validated on that target.
4. **Existing validation records contain target-specific open rows.** Visual review, long-run/soak evidence, DMDNet behavior, telemetry aggregation/classification, and some quality effects remain limited or open as described in the hardware and phase documents.
5. **Historical test totals disagree.** Existing documents cite older totals such as 1666/1691, while this audit run produced 1730 passed, 1 skipped, and 4 warnings. The older claims must not be silently rewritten as current results.
6. **Some repository documents describe different campaign hosts.** `docs/FINAL_VALIDATION_MATRIX.md` has separate 3060 and 4070 sections with “not present on this host” wording that is valid only for the campaign context. `docs/HARDWARE_VALIDATION_MATRIX.md` is the more recent combined record. Do not merge campaign numbers without preserving host identity.
7. **The current React telemetry HUD includes a hard-coded `CUDA / TensorRT (FP16)` label.** Runtime diagnostics are available separately, but the label is not proven to be provider-aware for every launcher branch.
8. **The current React README says the backend is on port 8001, while `start_react.js` allocates the next Pinokio port.** This may be documentation shorthand, but the distinction is not consistently expressed.
9. **The API contract is distributed.** Payloads are assembled across React components and Python handlers; no generated schema or API-version compatibility document was found.
10. **The test environment is not fully declared.** `pytest` is used successfully from the local environment, but it is not listed in `app/requirements.txt`; the documented app test command uses `unittest`.
11. **Cross-frame batching cannot apply the swapper-provided mask.** The batcher deliberately clears mask attribution when tiles from multiple worker requests are combined, so the visible swap-mask strength control is partial on that path (`app/roop/ProcessMgr.py:5072-5077`).
12. **Encoder resume identity omits some writer options.** `SegmentedVideoWriter` records preset, bitrate, threads, extra FFmpeg parameters, and colorspace in its writer options, but resume identity does not include all of them; changing those options between segments may mix encoding behavior (`app/roop/segment_writer.py:145-166`).
13. **Odd-dimension output colorspace handling is not validated.** The FFmpeg command uses the scale branch instead of the normal colorspace-filter branch for odd dimensions (`app/roop/ffmpeg_writer.py:270-281`).
14. **V2 live preview is sampled at approximately 1 Hz.** The existing watched publisher measured approximately 1.979 Hz, but V2 intentionally reuses the existing one-second progress poll to avoid extra status traffic. Browser timing and full-render impact remain unverified (`react-ui-v2/src/screens/CreateScreen.jsx:45-58`, `app/roop/live_preview.py:116-158`).
15. **Runtime telemetry migration is partial.** Stage 6B adds the backend-owned
    `runtime` object to `/api/progress` and exposes `/api/runtime/state`, but
    the legacy `/api/system/telemetry` projection, V1 dashboard consumers, and
    historical terminal log tail remain in place for compatibility. Full
    migration and a retained end-to-end overhead comparison are unverified.

16. **Stage 7A queue runtime validation is incomplete.** The canonical queue
    lifecycle and V2 controls are covered by automated tests and builds, but no
    fresh physical RTX 4070/RTX 3060 queue render, browser interaction pass, or
    live application-restart recovery test was run in this session.

## DESIRED FUTURE STATE

Resolve issues through their authorized gates: complete evidence, keep host-specific records separate, make diagnostics authoritative, define API schemas, validate the V2 live path in a browser and during a real render, and declare the test toolchain reproducibly.

## UNVERIFIED / UNKNOWN

- Whether the HUD label is user-visible on all provider paths in normal operation.
- Whether the README port wording causes an actual user failure.
- Whether any untracked runtime cache is stale or unsafe without a targeted cache audit.

## Scope note

No unrelated issue listed here was fixed during Stage 7A. The queue's old
five-state semantics were repaired at the queue boundary; physical hardware,
browser, and live restart validation remain open.

17. **Stage 8A pause acknowledgement is cooperative.** An in-flight inference
or long FFmpeg minterpolate operation is not interruptible and may delay the
`PAUSED` acknowledgement. Pause state is process-local; restart recovery still
re-queues the active job rather than resuming from a frame checkpoint. Physical
RTX 4070/RTX 3060 pause/resume output validation and browser interaction remain
open.

18. **Stage 8B restart validation is not physical shutdown evidence.** The
durable project record, atomic writes, segment identities, reload path, and
recoverability errors are covered by focused automated tests, but no actual
application-close/PC-shutdown/reopen render has been run in this session.
Physical RTX 4070 and RTX 3060 resume behavior, browser interaction, and final
playback integrity remain unverified.
19. **Stage 8B resumes only committed segmented output.** A pause request can
wait for an in-flight inference or encoder operation; frames not committed at a
safe writer boundary are recomputed. Legacy segment manifests without the
newer writer-options identity are conservatively not trusted for continuation.
20. **Stage 8B model identity is configuration identity.** The project records
the selected model/provider/precision and hardware assumptions, but does not
yet hash every downloaded model artifact. A replaced artifact with unchanged
configuration may therefore require a future model-manifest gate.
