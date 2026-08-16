// Thin client for the FastAPI backend (app/api.py) proxied via Vite.
export const API = window.location.origin;

async function handle(res) {
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).message || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
}

// Opt-in request deadline. Fetch has no default timeout, so a backend that
// accepts the socket and then stalls (mid-GPU-stall, or killed between accept
// and response) leaves the promise pending forever — a polled request that
// never settles silently stops the poll loop. Callers that must notice a dead
// server pass `timeout`; long-running endpoints (start, auto-capture, upscale)
// deliberately pass nothing and wait as long as it takes.
//
// An explicit `signal` still wins: aborting it aborts the request, and we clear
// our timer either way so no stray abort fires after the response lands.
const withDeadline = (opts = {}) => {
  if (!opts.timeout) return { signal: opts.signal, done: () => {} };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error('Request timed out')), opts.timeout);
  const abortFromCaller = () => ctrl.abort(opts.signal.reason);
  if (opts.signal?.aborted) abortFromCaller();
  else opts.signal?.addEventListener('abort', abortFromCaller, { once: true });
  return {
    signal: ctrl.signal,
    done: () => {
      clearTimeout(timer);
      opts.signal?.removeEventListener('abort', abortFromCaller);
    },
  };
};

export const getJSON = (path, opts = {}) => {
  const d = withDeadline(opts);
  return fetch(`${API}${path}`, { signal: d.signal }).then(handle).finally(d.done);
};

export const postJSON = (path, body, opts = {}) => {
  const d = withDeadline(opts);
  return fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
    signal: d.signal,
    // keepalive lets the request outlive a page teardown (e.g. Pinokio's
    // Run<->Dev webview reload) so a last-moment flush still reaches the server.
    keepalive: opts.keepalive,
  }).then(handle).finally(d.done);
};

// Uploads go through XHR, not fetch.
//
// fetch() cannot report request-body progress — there is no event for it, and
// the streaming-request API that would allow one is not available here. So a
// 4 GB video was posted with `fetch` and the UI had nothing to show but an
// indeterminate spinner: no bytes, no percentage, no rate, no way to tell a
// slow upload from a wedged one, and no way to cancel a file dropped by
// mistake short of restarting the app. XHR still exposes `upload.onprogress`,
// which is the only reason it is preferred over fetch anywhere in 2026.
//
// onProgress receives { loaded, total, phase }:
//   phase 'upload'  — bytes still going out; `total` is 0 if not computable
//   phase 'analyse' — every byte is sent and the server is now decoding and
//                     running detection, which for a long video is the LONGER
//                     half. Without this the bar would sit at 100% for minutes
//                     and read as a hang.
const xhrUpload = (path, fd, { onProgress, signal } = {}) => new Promise((resolve, reject) => {
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `${API}${path}`);

  if (onProgress) {
    xhr.upload.onprogress = (e) => onProgress({
      loaded: e.loaded,
      total: e.lengthComputable ? e.total : 0,
      phase: 'upload',
    });
    xhr.upload.onload = () => onProgress({ loaded: 0, total: 0, phase: 'analyse' });
  }

  const onAbort = () => xhr.abort();
  if (signal) {
    // Reject explicitly rather than calling xhr.abort() here: abort() on a
    // request that has been open()ed but not send()t fires no abort event, so
    // the handler below never runs and the promise would never settle — a
    // caller awaiting it would hang forever with no error to show.
    if (signal.aborted) {
      reject(new DOMException('Upload cancelled', 'AbortError'));
      return;
    }
    signal.addEventListener('abort', onAbort, { once: true });
  }
  const cleanup = () => signal?.removeEventListener('abort', onAbort);

  xhr.onload = () => {
    cleanup();
    if (xhr.status >= 200 && xhr.status < 300) {
      const ct = xhr.getResponseHeader('content-type') || '';
      if (!ct.includes('application/json')) { resolve(xhr.responseText); return; }
      // Mirrors handle(): a 2xx that is not parseable is a server bug, and
      // saying so beats a bare "undefined" surfacing three components away.
      try { resolve(JSON.parse(xhr.responseText)); }
      catch { reject(new Error('Malformed JSON in the server response')); }
      return;
    }
    let msg = xhr.statusText || `HTTP ${xhr.status}`;
    try { msg = JSON.parse(xhr.responseText).message || msg; } catch { /* ignore */ }
    reject(new Error(msg));
  };
  xhr.onerror = () => { cleanup(); reject(new Error('Network error during upload')); };
  xhr.onabort = () => { cleanup(); reject(new DOMException('Upload cancelled', 'AbortError')); };

  xhr.send(fd);
});

export const postFiles = (path, files, fields, opts) => {
  const fd = new FormData();
  const list = files instanceof FileList ? Array.from(files) : [].concat(files);
  list.forEach((f) => fd.append('files', f));
  if (fields) Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  return xhrUpload(path, fd, opts);
};

export const postFile = (path, file, fields, opts) => {
  const fd = new FormData();
  fd.append('file', file);
  if (fields) Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  return xhrUpload(path, fd, opts);
};

export const fileUrl = (p) => `${API}/api/file?path=${encodeURIComponent(p)}`;
