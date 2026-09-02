// ── Object-URL ownership, in one place ────────────────────────────────────
//
// `URL.createObjectURL` pins its Blob for the LIFETIME OF THE DOCUMENT. Nothing
// collects it: the string looks like garbage the moment the last reference to
// it drops, but the bytes behind it stay resident until `revokeObjectURL` is
// called by name. In a panel that decodes a frame per scrub tick and caches two
// hundred rendered previews, a single missed revoke is not a slow leak — it is
// hundreds of megabytes over one render session, and it survives every
// navigation inside the app because the document never goes away (Pinokio's
// webview reload is the only thing that clears it, which is why this went
// unnoticed: switching tabs "fixed" it).
//
// The rule this module enforces is that a URL is never created loose. It is
// created against an OWNER — a cache, a component instance, a request — and the
// owner is released as one unit. Callers cannot forget to pair a create with a
// revoke because they never write either call.
//
// `__stats()` is what makes the invariant testable rather than aspirational:
// after a scrub, a grid load and an unmount, `live` must return to what it was.

const owners = new Map();          // owner token -> Set<url>
const urlOwner = new Map();        // url -> owner token (one owner per url)

let created = 0;
let revoked = 0;

const isBlobUrl = (u) => typeof u === 'string' && u.startsWith('blob:');

/**
 * Create an object URL owned by `owner`. Releasing the owner revokes it.
 * @param {Blob} blob
 * @param {object|string|symbol} owner  any stable token (a ref object works)
 */
export function createOwnedUrl(blob, owner) {
  const url = URL.createObjectURL(blob);
  created += 1;
  let set = owners.get(owner);
  if (!set) { set = new Set(); owners.set(owner, set); }
  set.add(url);
  urlOwner.set(url, owner);
  return url;
}

/** Revoke ONE url, whoever owns it. Safe to call twice, or on a non-blob url. */
export function revokeUrl(url) {
  if (!isBlobUrl(url)) return;
  const owner = urlOwner.get(url);
  if (owner !== undefined) {
    owners.get(owner)?.delete(url);
    urlOwner.delete(url);
  }
  // Revoke even for a url this module never issued: callers migrating older
  // code hand us urls from `URL.createObjectURL` directly, and refusing to free
  // those would make this module a liability rather than a guarantee.
  URL.revokeObjectURL(url);
  revoked += 1;
}

/** Revoke every url held by `owner`. Idempotent. */
export function releaseOwner(owner) {
  const set = owners.get(owner);
  if (!set) return 0;
  const n = set.size;
  // Copy first: revokeUrl mutates the set it is iterating.
  for (const url of [...set]) revokeUrl(url);
  owners.delete(owner);
  return n;
}

/** How many urls `owner` currently holds (0 when it holds none / is unknown). */
export function ownedCount(owner) {
  return owners.get(owner)?.size || 0;
}

/**
 * Turn a `data:` URL into an owned `blob:` URL.
 *
 * The backend returns rendered previews as base64 data URLs. Held in React
 * state that is a multi-megabyte STRING: it is copied on every state read,
 * compared by identity on every render of every consumer, cannot be written to
 * localStorage without hitting quota, and posting one to a pop-out window
 * serialises the whole thing through postMessage. As a blob URL it is a
 * ~50-character token, the bytes live once in the browser's blob store, and
 * `<img>`/`drawImage`/`createImageBitmap` all read it without a base64 decode
 * on the main thread.
 *
 * Returns the input unchanged when it is not a data URL (already a blob/http
 * url, or empty), so call sites do not have to branch.
 */
export function dataUrlToOwnedBlobUrl(dataUrl, owner) {
  if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:')) return dataUrl;
  const comma = dataUrl.indexOf(',');
  if (comma < 0) return dataUrl;
  const header = dataUrl.slice(5, comma);            // e.g. "image/jpeg;base64"
  if (!header.includes('base64')) return dataUrl;    // percent-encoded: leave it
  const mime = header.split(';')[0] || 'application/octet-stream';
  try {
    const bin = atob(dataUrl.slice(comma + 1));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return createOwnedUrl(new Blob([bytes], { type: mime }), owner);
  } catch {
    // Malformed base64 — hand back the original rather than blanking the stage.
    return dataUrl;
  }
}

/**
 * Diagnostics. `live` is the number of urls this module believes are still
 * resident; a test that scrubs, loads a grid and unmounts should see it return
 * to its starting value. Exposed on `window` in dev so the invariant can be
 * checked from the console during a long session.
 */
export function __stats() {
  return { created, revoked, live: urlOwner.size, owners: owners.size };
}

if (typeof window !== 'undefined' && import.meta.env?.DEV) {
  window.__roopObjectUrls = { __stats, owners, releaseOwner };
}

/**
 * Read an owned blob URL back out as a `data:` URL.
 *
 * The two places that still need BYTES rather than a reference — POSTing the
 * displayed frame to /api/preview_upscale, and parking it in localStorage so
 * the view survives Pinokio's webview reload — are both one-off, user-initiated
 * and off the render path, so paying a re-encode there is the right trade for
 * keeping the hot path free of multi-megabyte strings.
 *
 * Resolves to '' rather than rejecting: every caller's fallback is "skip the
 * image", and none of them should fail loudly because a frame could not be
 * re-serialised.
 */
export async function blobUrlToDataUrl(url) {
  if (typeof url !== 'string' || !url) return '';
  if (url.startsWith('data:')) return url;
  try {
    const res = await fetch(url);
    if (!res.ok) return '';
    const blob = await res.blob();
    return await new Promise((resolve) => {
      const fr = new FileReader();
      fr.onload = () => resolve(typeof fr.result === 'string' ? fr.result : '');
      fr.onerror = () => resolve('');
      fr.readAsDataURL(blob);
    });
  } catch {
    // A revoked url, or a cross-origin one. Nothing to recover.
    return '';
  }
}
