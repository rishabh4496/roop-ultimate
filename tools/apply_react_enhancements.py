"""Apply React UI enhancements: OutputVideoPlayer, base64 data prefix normalization, and ObjectURL lifecycle."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REACT_SRC = os.path.join(ROOT, "react-ui", "src")

def update_object_urls():
    path = os.path.join(REACT_SRC, "components", "faceswap", "objectUrls.js")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_fn = """export function dataUrlToOwnedBlobUrl(dataUrl, owner) {
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
}"""

    new_fn = """export function normalizeDataUrl(dataUrl) {
  if (typeof dataUrl !== 'string' || !dataUrl) return dataUrl;
  const trimmed = dataUrl.trim();
  if (trimmed.startsWith('data:') || trimmed.startsWith('blob:') || trimmed.startsWith('http:') || trimmed.startsWith('https:') || trimmed.startsWith('/')) {
    return trimmed;
  }
  // If receiving raw base64 image data without the data: scheme, attach data:image/jpeg;base64,
  if (trimmed.length > 50 && /^[A-Za-z0-9+/=]+$/.test(trimmed.slice(0, 100))) {
    return `data:image/jpeg;base64,${trimmed}`;
  }
  return trimmed;
}

export function dataUrlToOwnedBlobUrl(rawInput, owner) {
  const dataUrl = normalizeDataUrl(rawInput);
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
}"""
    if old_fn in content:
        content = content.replace(old_fn, new_fn, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Updated react-ui/src/components/faceswap/objectUrls.js")
    else:
        print("[SKIP] objectUrls.js already updated or pattern not found")


def update_frame_decoder():
    path = os.path.join(REACT_SRC, "components", "faceswap", "frameDecoder.js")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_decode = """export function decodeFrame(url, { signal } = {}) {
  if (signal?.aborted) return Promise.reject(new DOMException('aborted', 'AbortError'));

  const w = getWorker();
  if (!w) return decodeOnMainThread(url, signal);"""

    new_decode = """export function normalizeFrameUrl(url) {
  if (typeof url !== 'string' || !url) return url;
  const trimmed = url.trim();
  if (trimmed.startsWith('data:') || trimmed.startsWith('blob:') || trimmed.startsWith('http:') || trimmed.startsWith('https:') || trimmed.startsWith('/')) {
    return trimmed;
  }
  if (trimmed.length > 50 && /^[A-Za-z0-9+/=]+$/.test(trimmed.slice(0, 100))) {
    return `data:image/jpeg;base64,${trimmed}`;
  }
  return trimmed;
}

export function decodeFrame(rawUrl, { signal } = {}) {
  if (signal?.aborted) return Promise.reject(new DOMException('aborted', 'AbortError'));
  const url = normalizeFrameUrl(rawUrl);

  const w = getWorker();
  if (!w) return decodeOnMainThread(url, signal);"""

    if old_decode in content:
        content = content.replace(old_decode, new_decode, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Updated react-ui/src/components/faceswap/frameDecoder.js")
    else:
        print("[SKIP] frameDecoder.js already updated or pattern not found")


def update_faceswap_player():
    path = os.path.join(REACT_SRC, "components", "FaceSwap.jsx")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "import OutputVideoPlayer" not in content:
        content = "import OutputVideoPlayer from './OutputVideoPlayer';\n" + content

    old_vid = "? <video src={outUrl} controls className=\"w-full rounded-xl border border-white/5\" />"
    new_vid = "? <OutputVideoPlayer src={outUrl} renderKey={out?.path || out?.url} className=\"w-full rounded-xl border border-white/5\" />"

    if old_vid in content:
        content = content.replace(old_vid, new_vid, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Updated react-ui/src/components/FaceSwap.jsx with OutputVideoPlayer")
    else:
        print("[SKIP] FaceSwap.jsx already updated or pattern not found")


def update_processing_player():
    path = os.path.join(REACT_SRC, "components", "Processing.jsx")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "import OutputVideoPlayer" not in content:
        content = "import OutputVideoPlayer from './OutputVideoPlayer';\n" + content

    old_vid = "? <video src={outUrl} controls className=\"w-full max-h-[52vh] rounded-xl border border-white/5\" />"
    new_vid = "? <OutputVideoPlayer src={outUrl} renderKey={out?.path || out?.url} className=\"w-full max-h-[52vh] rounded-xl border border-white/5\" />"

    if old_vid in content:
        content = content.replace(old_vid, new_vid, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Updated react-ui/src/components/Processing.jsx with OutputVideoPlayer")
    else:
        print("[SKIP] Processing.jsx already updated or pattern not found")


def update_extras_player():
    path = os.path.join(REACT_SRC, "components", "Extras.jsx")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "import OutputVideoPlayer" not in content:
        content = "import OutputVideoPlayer from './OutputVideoPlayer';\n" + content

    old_vid1 = "? <video src={fileUrl(enhResult.path)} controls className=\"w-full rounded-lg border border-white/10\" />"
    new_vid1 = "? <OutputVideoPlayer src={fileUrl(enhResult.path)} renderKey={enhResult.path} className=\"w-full rounded-lg border border-white/10\" />"

    old_vid2 = "? <video src={fileUrl(result.path)} controls className=\"w-full rounded-lg border border-white/10\" />"
    new_vid2 = "? <OutputVideoPlayer src={fileUrl(result.path)} renderKey={result.path} className=\"w-full rounded-lg border border-white/10\" />"

    if old_vid1 in content or old_vid2 in content:
        if old_vid1 in content:
            content = content.replace(old_vid1, new_vid1, 1)
        if old_vid2 in content:
            content = content.replace(old_vid2, new_vid2, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[OK] Updated react-ui/src/components/Extras.jsx with OutputVideoPlayer")
    else:
        print("[SKIP] Extras.jsx already updated or pattern not found")


if __name__ == "__main__":
    update_object_urls()
    update_frame_decoder()
    update_faceswap_player()
    update_processing_player()
    update_extras_player()
    print("\nAll React UI components updated successfully!")
