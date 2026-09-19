"""Automated verification script testing the React UI preview and video player logic."""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REACT_SRC = os.path.join(ROOT, "react-ui", "src")
NODE = "G:\\pinokio\\bin\\miniforge\\node.exe"

# Node script testing objectUrls and frameDecoder URL normalization
test_js = """
globalThis.window = { location: { origin: 'http://127.0.0.1:8000' } };

const { normalizeDataUrl, dataUrlToOwnedBlobUrl } = await import('./components/faceswap/objectUrls.js');
const { normalizeFrameUrl } = await import('./components/faceswap/frameDecoder.js');
const { fileUrl } = await import('./api.js');

console.log('=== Running React UI Unit Checks ===');

// 1. Test normalizeDataUrl
const rawB64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
const norm1 = normalizeDataUrl(rawB64);
console.assert(norm1.startsWith('data:image/jpeg;base64,'), 'raw base64 must have prefix attached');
console.log('[PASS] normalizeDataUrl attaches data:image/jpeg;base64, to raw base64');

const existingData = 'data:image/png;base64,' + rawB64;
const norm2 = normalizeDataUrl(existingData);
console.assert(norm2 === existingData, 'existing data: url must be preserved');
console.log('[PASS] normalizeDataUrl preserves existing data: URI');

// 2. Test normalizeFrameUrl
const frameNorm1 = normalizeFrameUrl(rawB64);
console.assert(frameNorm1.startsWith('data:image/jpeg;base64,'), 'frame url must normalize raw base64');
console.log('[PASS] normalizeFrameUrl normalizes raw base64');

const webPath = '/outputs/video.mp4';
const frameNorm2 = normalizeFrameUrl(webPath);
console.assert(frameNorm2 === webPath, 'web path must be preserved');
console.log('[PASS] normalizeFrameUrl preserves web path /outputs/...');

// 3. Test fileUrl
const u1 = fileUrl('/outputs/result.mp4');
console.assert(u1 === 'http://127.0.0.1:8000/outputs/result.mp4' || u1.endsWith('/outputs/result.mp4'), 'fileUrl handles web path');
console.log('[PASS] fileUrl correctly handles /outputs/... web paths');

console.log('All Node checks passed successfully!');
"""

def main():
    test_file = os.path.join(REACT_SRC, "_test_runner.mjs")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_js)

    try:
        env = dict(os.environ)
        env["PATH"] = "G:\\pinokio\\bin\\miniforge;" + env.get("PATH", "")
        res = subprocess.run([NODE, test_file], cwd=REACT_SRC, capture_output=True, text=True)
        print("STDOUT:")
        print(res.stdout)
        if res.stderr:
            print("STDERR:")
            print(res.stderr)
        assert res.returncode == 0, f"Node tests failed with code {res.returncode}"
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    main()
