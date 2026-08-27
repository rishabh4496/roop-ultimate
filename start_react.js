module.exports = async (kernel) => {
  const API_PORT = await kernel.port()
  const VITE_PORT = API_PORT + 1
  const GRADIO_PORT = API_PORT + 2

  return {
    daemon: true,
    run: [
      {
        method: "shell.run",
        params: {
          venv: "env",
          env: {
            ROOP_PROFILE: "1",
            ROOP_BATCH_SWAP_XFRAME: "1",
            ROOP_BATCH_SWAP: "1",
            ROOP_STAB_PARALLEL: "1",
            // Work-stealing granularity for the parallel stabilizer: 2 blocks per
            // thread recovers most of the idle-thread imbalance (25-59% of chunk
            // time on static splits) for ~+19% warm-up recompute. See ProcessMgr.
            ROOP_STAB_BLOCKS_PER_THREAD: "2",
            // Scan stride for the "Analyzing faces" pre-pass. Back to "1", which
            // is the code default, after "2" was reported as flicker on a moving
            // face with nothing in front of it — the exact failure the old
            // comment here warned about, so this is the warning being taken
            // rather than a new discovery.
            //
            // At "2" only frames 0, 2, 4... are ever detected. The others get a
            // bbox, keypoints and 106 landmarks LINEARLY INTERPOLATED between
            // their neighbours, and those decide the swap crop, the paste mask
            // and the mouth region. Measured on the reported clip: 649 of 1281
            // faces (51%) were interpolated. Linear is exact for a head moving
            // steadily and wrong by the second-order term for one that is
            // turning, so the registration alternates good/approximate every
            // other frame — which is a 2-frame flicker on exactly the footage
            // where the face is moving.
            //
            // The cost of "1" is real and worth stating: the pre-pass is
            // detection-bound (profiled at track_wait 10.81ms/frame against
            // track_detect 43.52ms across a 4-instance pool — the main thread is
            // blocked on the detector ~97% of the time) and it is ~30% of a long
            // run, so this roughly doubles it. On the reported clip that is
            // track_detect 52.6s -> ~105s. Trading a minute of pre-pass for a
            // swap that sits still is the right way round for a quality tool;
            // set it back to "2" for footage with little head movement.
            //
            // Capped at ROOP_TEMPORAL_GAP (10) by the script either way, since a
            // stride past the gap limit would leave skipped frames with no faces
            // at all. The SWAP AUDIT now prints what fraction of the faces it
            // actually swapped had interpolated landmarks — read that before
            // changing this.
            ROOP_TEMPORAL_STEP: "1",
            // ROOP_EXPR_POOL is deliberately NOT set here. It used to be pinned
            // to "2" — measured +28% on the expression stage for +654MB — but
            // this file ships to every install, and forcing two extra restorer
            // contexts on a card too small for even the swapper pool is how you
            // turn a benchmark into an OOM. session_pool._auto_expression_pool()
            // now picks it from the machine's own VRAM (12GB+ still gets 2).
            // Set ROOP_EXPR_POOL here only to override that on THIS machine.
            // Diagnostic for wrong-face swaps: prints [TRACKASSIGN] once per
            // track (which person each track was bound to, and at what cosine
            // distance) and [TRACKMATCH] per face per frame (chosen track,
            // resolved source, and the reason for any veto). Read it to tell a
            // too-permissive distance threshold apart from a tracker identity
            // switch. VERBOSE — [TRACKMATCH] fires for every face of every
            // frame, so set it back to "0" once the question is answered.
            // Off: it was left on after a wrong-face investigation and was
            // producing 461 of every 587 log lines on a real render. Set to "1"
            // when a wrong-face question comes up again. (It no longer breaks
            // the progress bar either way — these go through bar_write now.)
            ROOP_DEBUG_MATCH: "0",
            // OFF (the code default). This was set to "1.0" to close a hole that
            // no longer exists: at the time, a single-person tracked swap applied
            // its source "by SPATIAL association with no identity check at all",
            // so a tracker ID switch kept swapping the wrong face and only an
            // absolute per-frame distance test could catch it.
            //
            // ROOP_TRACK_EMB_MAX (0.7) has since become exactly that missing
            // swap-time check, and it runs FIRST — a face must be within 0.7 of a
            // track's mean embedding before that track's entry can even be
            // considered (ProcessMgr, the appearance gate in the entry loop),
            // and the track's mean must itself have passed ROOP_TRACK_ASSIGN_MAX
            // (0.6) against the captured person. So by the time the single-person
            // veto is reached, identity has already been established twice over,
            // from the track MEAN — a far cleaner measurement than any one frame.
            //
            // What was left was the failure mode its own comment warns about:
            // "anything near the match threshold will make hard frames blink
            // instead." The distance it tests is computed from the CURRENT
            // frame's embedding, which is exactly what occlusion corrupts — a
            // hand, a mic or another face across the subject pushes it past 1.0,
            // the source is vetoed, the face drops to per-frame matching at the
            // TIGHTER threshold, that fails too, and the frame is left unswapped.
            // Object passes, swap returns. That on/off is the flicker.
            //
            // Note it also refused only REAL detections: a gap-filled face
            // carries the track mean as its embedding by construction, so its
            // distance here is the track's own and it was never vetoed. The gate
            // was therefore hardest on the frames where detection had actually
            // succeeded. Set back to "1.0" only if strangers get swapped AND the
            // SWAP AUDIT at the end of a run shows no `veto: single-person`.
            ROOP_TRACK_VETO_SINGLE: "0",
            OMP_NUM_THREADS: "1",
            OPENBLAS_NUM_THREADS: "1",
            MKL_NUM_THREADS: "1",
            NUMEXPR_NUM_THREADS: "1",
            ROOP_API_PORT: String(API_PORT),
            ROOP_GRADIO_PORT: String(GRADIO_PORT)
          },
          path: "app",
          message: [
            "python run.py",
          ],
          on: [{
            "event": "/(http:\\/\\/[0-9.:]+)/", 
            "done": true
          }]
        }
      },
      {
        method: "shell.run",
        params: {
          env: {
            ROOP_API_PORT: String(API_PORT),
            PORT: String(VITE_PORT)
          },
          path: "react-ui",
          message: [
            "npm run dev"
          ],
          on: [{
            "event": "/(http:\\/\\/[a-zA-Z0-9.:]+)/",
            "done": true
          }]
        }
      },
      {
        method: "local.set",
        params: {
          url: "{{input.event[1]}}",
          // Direct address of the FastAPI backend (api.py binds 127.0.0.1:ROOP_API_PORT).
          // Surfaced so pinokio.js can offer a graceful "Stop Swap" that POSTs
          // /api/stop — which finalizes the output video (moov atom) instead of
          // the hard process-kill the Terminal square does.
          api_url: `http://127.0.0.1:${API_PORT}`
        }
      }
    ]
  }
}
