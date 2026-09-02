// Universal Network & API Transport Layer for React UI 2.0
export const API = window.location.origin;

async function handleResponse(response) {
  if (!response.ok) {
    let message = response.statusText || `HTTP ${response.status}`;
    try {
      const body = await response.json();
      message = body.message || message;
      if (body.recoverability_error && Array.isArray(body.reasons) && body.reasons.length) {
        message += `\n${body.reasons.join('\n')}`;
      }
    } catch {
      /* use status text if JSON parse fails */
    }
    throw new Error(message);
  }
  const contentType = response.headers.get('content-type') || '';
  return contentType.includes('application/json') ? response.json() : response.text();
}

/**
 * Opt-in request deadline wrapper with AbortController.
 */
export const withDeadline = (opts = {}) => {
  if (!opts.timeout) return { signal: opts.signal, done: () => {} };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error('Request timed out')), opts.timeout);
  const abortFromCaller = () => ctrl.abort(opts.signal?.reason);
  if (opts.signal?.aborted) {
    abortFromCaller();
  } else if (opts.signal) {
    opts.signal.addEventListener('abort', abortFromCaller, { once: true });
  }
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
  return fetch(`${API}${path}`, { signal: d.signal })
    .then(handleResponse)
    .finally(d.done);
};

export const postJSON = (path, body, opts = {}) => {
  const d = withDeadline(opts);
  return fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
    signal: d.signal,
    keepalive: opts.keepalive,
  })
    .then(handleResponse)
    .finally(d.done);
};

/**
 * XHR Upload supporting dual-phase progress:
 *  - 'upload': bytes transmitted to server
 *  - 'analyse': server-side decoding & face detection
 */
export const xhrUpload = (path, formData, { onProgress, signal } = {}) =>
  new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}${path}`);

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        const percent = e.lengthComputable ? Math.round((e.loaded / e.total) * 100) : 0;
        onProgress({
          loaded: e.loaded,
          total: e.lengthComputable ? e.total : 0,
          percent,
          phase: 'upload',
        });
      };
      xhr.upload.onload = () => {
        onProgress({ loaded: 0, total: 0, percent: 100, phase: 'analyse' });
      };
    }

    const onAbort = () => xhr.abort();
    if (signal) {
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
        if (!ct.includes('application/json')) {
          resolve(xhr.responseText);
          return;
        }
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error('Malformed JSON in the server response'));
        }
        return;
      }
      let msg = xhr.statusText || `HTTP ${xhr.status}`;
      try {
        msg = JSON.parse(xhr.responseText).message || msg;
      } catch {
        /* ignore */
      }
      reject(new Error(msg));
    };

    xhr.onerror = () => {
      cleanup();
      reject(new Error('Network error during upload'));
    };
    xhr.onabort = () => {
      cleanup();
      reject(new DOMException('Upload cancelled', 'AbortError'));
    };

    xhr.send(formData);
  });

export const postFiles = (path, files, fields, opts) => {
  const fd = new FormData();
  const list = files instanceof FileList ? Array.from(files) : [].concat(files || []);
  list.forEach((f) => fd.append('files', f));
  if (fields) {
    Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  }
  return xhrUpload(path, fd, opts);
};

export const postFile = (path, file, fields, opts) => {
  const fd = new FormData();
  fd.append('file', file);
  if (fields) {
    Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  }
  return xhrUpload(path, fd, opts);
};

export const fileUrl = (path) => `${API}/api/file?path=${encodeURIComponent(path)}`;
