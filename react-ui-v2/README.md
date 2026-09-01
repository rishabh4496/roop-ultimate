# Roop Ultimate React UI 2.0

This directory is a parallel, reversible React UI 2.0 integration. It is
deliberately independent of `react-ui/` and `react-ui-v1-backup/`.

## Current scope

The V2 client provides:

- a responsive application shell and hash navigation (`#/home`, `#/create`, `#/settings`)
- shared design tokens and one theme engine for light, dark, professional,
  modern, minimal, gaming, and anime themes
- reusable controls, cards, fields, notices, progress, and loading states
- reducer-based application state
- an error boundary and notification center
- backend-backed processing, visual options, provider/model state, runtime
  telemetry, live preview, queue controls, pause/resume, and project recovery
- read-only environment evidence and reference-aware storage review through
  the verified application routes
- an explicit Pinokio-only boundary for updates and the full child-process
  health worker; no browser endpoint is invented for either

The Create route uses only verified existing FastAPI operations for source and
target uploads, selection, frame preview, generation, progress, live output,
queue, pause/resume, and project recovery. The Settings route uses the
verified runtime diagnostics and storage review/deletion routes. Update
execution remains in Pinokio because the repository exposes no browser update
route. The existing V1/current client remains under
`react-ui/` and the V1 snapshot remains under `react-ui-v1-backup/`.

## Run independently

```bash
npm install
npm run dev
```

The default V2 development server is `http://127.0.0.1:5174`. Set `PORT` to
choose another port. The Vite proxy is prepared for the existing FastAPI
backend.

```bash
npm run build
npm run lint
```

## Ownership boundary

```text
V2 screen -> V2 state/router/theme -> future API adapter -> existing FastAPI routes
```

Provider policy, model/session lifetime, queue execution, GPU policy, and
output finalization remain backend-owned. V2 sends commands and renders the
returned state; it does not reproduce those owners in React.
