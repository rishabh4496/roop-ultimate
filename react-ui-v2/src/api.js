// V2's small API adapter. Queue actions use the same JSON boundary as V1 and
// consume server-owned state from /api/queue; no terminal text is parsed.
export const API = window.location.origin;

async function handle(response) {
  if (!response.ok) {
    let message = response.statusText || `HTTP ${response.status}`;
    try { message = (await response.json()).message || message; } catch { /* keep status text */ }
    throw new Error(message);
  }
  const type = response.headers.get('content-type') || '';
  return type.includes('application/json') ? response.json() : response.text();
}

export function getJSON(path) {
  return fetch(`${API}${path}`).then(handle);
}

export function postJSON(path, body) {
  return fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }).then(handle);
}

export function postFiles(path, files, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    Array.from(files || []).forEach((file) => form.append('files', file));
    xhr.open('POST', `${API}${path}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(Math.round((event.loaded / event.total) * 100), 'upload');
    };
    xhr.upload.onload = () => onProgress?.(100, 'analyzing');
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); } catch { reject(new Error('Malformed JSON in the server response')); }
      } else {
        let message = xhr.statusText || `HTTP ${xhr.status}`;
        try { message = JSON.parse(xhr.responseText).message || message; } catch { /* keep status text */ }
        reject(new Error(message));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.onabort = () => reject(new Error('Upload cancelled'));
    xhr.send(form);
  });
}

export const targetPreviewUrl = (index, frame = 1) => `${API}/api/target/preview?index=${index}&frame=${frame}`;
// The backend publishes one already-encoded JPEG and exposes its monotonic
// sequence through /api/progress. The sequence is the cache key; do not inline
// frame bytes into progress JSON or request a new swap for every poll.
export const liveFrameUrl = (seq) => `${API}/api/live_frame?seq=${encodeURIComponent(seq)}`;
export const fileUrl = (path) => `${API}/api/file?path=${encodeURIComponent(path)}`;
