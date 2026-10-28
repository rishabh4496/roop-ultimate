/**
 * WebGL2 3D .cube LUT Parser, Domain Validator, and Premultiplied Alpha Engine.
 *
 * Supports standard Adobe / DaVinci Resolve .cube specifications:
 *   - LUT_3D_SIZE <size>
 *   - DOMAIN_MIN <r> <g> <b>
 *   - DOMAIN_MAX <r> <g> <b>
 *   - Float triplets for Red, Green, Blue
 *
 * Invariants:
 *   1. DOMAIN_MIN and DOMAIN_MAX are strictly validated: DOMAIN_MIN must be strictly less than DOMAIN_MAX.
 *   2. Invalid domains are refused with an explicit error rather than silently grading incorrect code values.
 *   3. Strict premultiplied alpha scaling (rgb * a, a) over Stage 7 XSeg 3 and Stage 11 vector roto masks.
 */

/**
 * Validates DOMAIN_MIN and DOMAIN_MAX domains.
 * @param {number[]} domainMin
 * @param {number[]} domainMax
 */
export function validateCubeDomain(domainMin, domainMax) {
  if (!Array.isArray(domainMin) || domainMin.length !== 3 || !domainMin.every(Number.isFinite)) {
    throw new Error(`Invalid .cube domain: DOMAIN_MIN (${String(domainMin)}) must be 3 finite numbers`);
  }
  if (!Array.isArray(domainMax) || domainMax.length !== 3 || !domainMax.every(Number.isFinite)) {
    throw new Error(`Invalid .cube domain: DOMAIN_MAX (${String(domainMax)}) must be 3 finite numbers`);
  }
  for (let i = 0; i < 3; i++) {
    if (domainMin[i] >= domainMax[i]) {
      throw new Error(
        `Invalid .cube domain: DOMAIN_MIN (${domainMin.join(' ')}) must be strictly less than DOMAIN_MAX (${domainMax.join(' ')}) at channel ${i}`
      );
    }
  }
  return true;
}

/**
 * Parses a standard .cube 3D LUT string.
 * @param {string} text
 * @returns {{ title: string, size: number, domainMin: number[], domainMax: number[], data: Float32Array, rgbaData: Float32Array }}
 */
export function parseCubeLut(text) {
  if (!text || typeof text !== 'string') {
    throw new Error('Invalid .cube data: empty or non-string input');
  }

  const lines = text.split(/\r?\n/);
  let title = '';
  let size = 0;
  let domainMin = [0.0, 0.0, 0.0];
  let domainMax = [1.0, 1.0, 1.0];
  let domainMinExplicit = false;
  let domainMaxExplicit = false;

  const rawFloats = [];

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i].trim();
    if (!rawLine || rawLine.startsWith('#')) continue;

    // Check for TITLE
    if (/^TITLE\b/i.test(rawLine)) {
      const match = rawLine.match(/^TITLE\s+["']?([^"']+)["']?/i);
      if (match) title = match[1];
      continue;
    }

    // Check for LUT_3D_SIZE
    if (/^LUT_3D_SIZE\b/i.test(rawLine)) {
      const parts = rawLine.split(/\s+/);
      const parsedSize = parseInt(parts[1], 10);
      if (!Number.isInteger(parsedSize) || parsedSize < 2 || parsedSize > 256) {
        throw new Error(`Invalid .cube LUT_3D_SIZE: ${parts[1]} (must be between 2 and 256)`);
      }
      size = parsedSize;
      continue;
    }

    // Check for 1D LUT rejection (we require 3D LUT)
    if (/^LUT_1D_SIZE\b/i.test(rawLine)) {
      throw new Error('Unsupported 1D LUT format: only 3D .cube LUTs with LUT_3D_SIZE are supported');
    }

    // Check for DOMAIN_MIN
    if (/^DOMAIN_MIN\b/i.test(rawLine)) {
      const parts = rawLine.split(/\s+/).slice(1).map(Number);
      if (parts.length < 3 || parts.some((v) => !Number.isFinite(v))) {
        throw new Error(`Invalid .cube DOMAIN_MIN line: "${rawLine}"`);
      }
      domainMin = [parts[0], parts[1], parts[2]];
      domainMinExplicit = true;
      continue;
    }

    // Check for DOMAIN_MAX
    if (/^DOMAIN_MAX\b/i.test(rawLine)) {
      const parts = rawLine.split(/\s+/).slice(1).map(Number);
      if (parts.length < 3 || parts.some((v) => !Number.isFinite(v))) {
        throw new Error(`Invalid .cube DOMAIN_MAX line: "${rawLine}"`);
      }
      domainMax = [parts[0], parts[1], parts[2]];
      domainMaxExplicit = true;
      continue;
    }

    // Numerical data line
    const tokens = rawLine.split(/\s+/);
    if (tokens.length >= 3) {
      const r = Number(tokens[0]);
      const g = Number(tokens[1]);
      const b = Number(tokens[2]);
      if (Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b)) {
        rawFloats.push(r, g, b);
      }
    }
  }

  if (size <= 0) {
    throw new Error('Invalid .cube LUT: missing or invalid LUT_3D_SIZE tag');
  }

  // Explicit validation of DOMAIN_MIN and DOMAIN_MAX
  validateCubeDomain(domainMin, domainMax);

  const expectedValues = size * size * size * 3;
  if (rawFloats.length !== expectedValues) {
    throw new Error(
      `Corrupt .cube data: expected ${expectedValues / 3} table entries for LUT_3D_SIZE ${size}, but found ${rawFloats.length / 3}`
    );
  }

  const data = new Float32Array(rawFloats);

  // Expand to RGBA Float32Array for WebGL2 sampler3D upload
  const totalVoxels = size * size * size;
  const rgbaData = new Float32Array(totalVoxels * 4);
  for (let v = 0; v < totalVoxels; v++) {
    const srcIdx = v * 3;
    const dstIdx = v * 4;
    rgbaData[dstIdx] = data[srcIdx];
    rgbaData[dstIdx + 1] = data[srcIdx + 1];
    rgbaData[dstIdx + 2] = data[srcIdx + 2];
    rgbaData[dstIdx + 3] = 1.0;
  }

  return {
    title: title || `Cube_${size}x${size}x${size}`,
    size,
    domainMin,
    domainMax,
    domainMinExplicit,
    domainMaxExplicit,
    data,
    rgbaData,
  };
}

/**
 * Uploads a parsed 3D LUT to a WebGL2 texture unit as a float sampler3D texture.
 * @param {WebGL2RenderingContext} gl
 * @param {ReturnType<typeof parseCubeLut>} lut
 * @param {number} [unitIndex=2] Texture unit index (e.g. 2 for gl.TEXTURE2)
 * @returns {{ texture: WebGLTexture, unitIndex: number, size: number, domainMin: number[], domainMax: number[] } | null}
 */
export function uploadLut3DTexture(gl, lut, unitIndex = 2) {
  if (!gl || typeof gl.texImage3D !== 'function') {
    return null; // Not a WebGL2 context
  }

  const texture = gl.createTexture();
  if (!texture) return null;

  gl.activeTexture(gl.TEXTURE0 + unitIndex);
  gl.bindTexture(gl.TEXTURE_3D, texture);

  // Trilinear interpolation in 3D: linear min/mag filters
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  // Edge clamping to prevent domain wrapping
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  // Upload RGBA float table
  const size = lut.size;
  try {
    gl.texImage3D(
      gl.TEXTURE_3D,
      0,
      gl.RGBA16F || gl.RGBA,
      size,
      size,
      size,
      0,
      gl.RGBA,
      gl.FLOAT,
      lut.rgbaData
    );
  } catch (err) {
    // Fallback if RGBA16F float texture upload fails
    try {
      gl.texImage3D(
        gl.TEXTURE_3D,
        0,
        gl.RGBA,
        size,
        size,
        size,
        0,
        gl.RGBA,
        gl.FLOAT,
        lut.rgbaData
      );
    } catch (fallbackErr) {
      gl.deleteTexture(texture);
      throw new Error(`Failed to upload 3D LUT texture: ${fallbackErr.message || err.message}`);
    }
  }

  return {
    texture,
    unitIndex,
    size: lut.size,
    domainMin: lut.domainMin,
    domainMax: lut.domainMax,
  };
}

/**
 * Strict premultiplied alpha scaling (rgb * a, a).
 * Used when compositing Stage 7 XSeg 3 and Stage 11 vector roto masks.
 * @param {[number, number, number]} rgb Linear RGB [0..1]
 * @param {number} alpha Mask opacity [0..1]
 * @returns {[number, number, number, number]} Premultiplied RGBA [rgb * a, a]
 */
export function premultiplyAlpha(rgb, alpha) {
  const a = Math.max(0, Math.min(1, Number(alpha) || 0));
  return [
    (Number(rgb[0]) || 0) * a,
    (Number(rgb[1]) || 0) * a,
    (Number(rgb[2]) || 0) * a,
    a,
  ];
}

/**
 * Composites a premultiplied source layer over a destination layer.
 * Formula: C_out = C_src + C_dst * (1 - A_src)
 * @param {[number, number, number, number]} srcPremul [r*a, g*a, b*a, a]
 * @param {[number, number, number, number]} dstPremul [r*a, g*a, b*a, a]
 * @returns {[number, number, number, number]} Composed premultiplied RGBA
 */
export function blendPremultiplied(srcPremul, dstPremul) {
  const [sr, sg, sb, sa] = srcPremul;
  const [dr, dg, db, da] = dstPremul;
  const oneMinusSa = 1.0 - sa;
  return [
    sr + dr * oneMinusSa,
    sg + dg * oneMinusSa,
    sb + db * oneMinusSa,
    sa + da * oneMinusSa,
  ];
}
