# Roop Ultimate React UI 2.0 foundation

This directory is a parallel, reversible React UI 2.0 foundation. It is
deliberately independent of `react-ui/` and `react-ui-v1-backup/`.

## Current scope

The foundation provides:

- a responsive application shell and hash navigation (`#/home`, `#/workspace`, `#/settings`)
- shared design tokens and one theme engine for light, dark, professional,
  modern, minimal, gaming, and anime themes
- reusable controls, cards, fields, notices, progress, and loading states
- reducer-based application state
- an error boundary and notification center

It does not call the backend and does not implement processing features yet.
The existing V1/current client remains under `react-ui/` and the V1 snapshot
remains under `react-ui-v1-backup/`.

## Run independently

```bash
npm install
npm run dev
```

The default V2 development server is `http://127.0.0.1:5174`. Set `PORT` to
choose another port. The Vite proxy is prepared for the existing FastAPI
backend, but this foundation does not issue API requests.

```bash
npm run build
npm run lint
```

## Ownership boundary

```text
V2 screen -> V2 state/router/theme -> future API adapter -> existing FastAPI routes
```

Processing, providers, models, queue execution, GPU policy, and output
finalization remain outside this foundation.
