import React, { useEffect, useRef } from 'react';
import { useFxStore, AMBIENT_QUALITY, prefersReducedMotion } from './fxStore';
import { useTelemetryStore } from '../store/telemetryStore';

// ── The ambient 3D horizon ──────────────────────────────────────────────────
//
// A planet limb rising from the lower right, ray-traced in one fragment shader: a real sphere (ray /
// sphere intersection, so the wireframe on its surface foreshortens and rotates in 3D), a three-band
// atmosphere (a tight bright rim, a violet halo, a wide plum haze - the same 1 : 8 : 40 falloff
// ratios as a scattering glow), a parallax star field in three depths, and film grain that dithers
// the gradients so they do not band on an 8-bit panel.
//
// THE ENGINE IS GPU-BOUND, so this is built to cost as close to nothing as a moving picture can:
//   * it renders at a fraction of the CSS size (0.35-0.75, see AMBIENT_QUALITY) and is upscaled by
//     the compositor, with a frame cap (24-60 fps) below the display rate;
//   * while a render runs it draws ONE still frame and stops (unless the user opts in with
//     `ambientDuringRender`), so a render never shares the GPU with it;
//   * it stops while the page is hidden, and draws a single frame under prefers-reduced-motion;
//   * it is one full-screen triangle and one draw call - no textures, no buffers beyond the quad.
// Nothing here has been measured against render fps; the still-frame-during-render rule is what
// makes that measurement unnecessary, and the toggle is in the FX panel.

const VERT = `
attribute vec2 p;
void main(){ gl_Position = vec4(p, 0.0, 1.0); }
`;

const FRAG = `
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform vec2 uMouse;
uniform vec3 uSky;
uniform vec3 uViolet;
uniform vec3 uPlum;
uniform vec3 uGrid;
uniform float uIntensity;

float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float hash3(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c,0.,-s, 0.,1.,0., s,0.,c); }
mat3 rotX(float a){ float c = cos(a), s = sin(a); return mat3(1.,0.,0., 0.,c,s, 0.,-s,c); }

// Stars on a direction: a cell grid over the view sphere, one candidate star per cell.
vec3 stars(vec3 rd, float scale, float t, float seed){
  vec3 q = rd * scale;
  vec3 cell = floor(q);
  vec3 f = fract(q) - 0.5;
  float h = hash3(cell + seed);
  if (h < 0.86) return vec3(0.0);
  vec3 off = vec3(hash3(cell + 1.7 + seed), hash3(cell + 3.1 + seed), hash3(cell + 5.3 + seed)) - 0.5;
  float d = length(f - off * 0.6);
  float tw = 0.65 + 0.35 * sin(t * (1.0 + h * 3.0) + h * 40.0);
  float s = smoothstep(0.075, 0.0, d) * tw * (h - 0.86) * 7.0;
  return mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.8, 0.95), hash3(cell + 9.0)) * s;
}

void main(){
  vec2 frag = gl_FragCoord.xy;
  vec2 uv = (frag - 0.5 * uRes) / uRes.y;

  // Camera: a gentle pointer parallax (yaw/pitch of a few degrees) and a slow drift.
  vec3 rd = normalize(vec3(uv, 1.55));
  rd = rotY(uMouse.x * 0.08 + sin(uTime * 0.03) * 0.02) * rotX(-uMouse.y * 0.05) * rd;

  // The planet: a large sphere whose limb crosses the lower right of the frame.
  vec3 C = vec3(0.78, -1.62, 3.1);
  float R = 1.72;
  float tc = dot(rd, C);
  vec3 closest = rd * tc - C;
  float d = length(closest);           // closest approach of the ray to the centre
  float alt = d - R;                    // > 0: the ray passes above the surface

  vec3 col = vec3(0.0);

  // Deep space: three parallax depths of stars, hidden behind the planet.
  if (alt > 0.0 || tc < 0.0) {
    vec3 sd = rotY(uTime * 0.004) * rd;
    col += stars(sd, 70.0, uTime, 0.0) * 0.9;
    col += stars(rotY(uMouse.x * 0.03) * sd, 140.0, uTime * 1.3, 11.0) * 0.6;
    col += stars(rotY(uMouse.x * 0.06) * sd, 260.0, uTime * 1.7, 23.0) * 0.35;
  }

  if (alt < 0.0 && tc > 0.0) {
    // On the sphere: the near intersection, its normal, and a rotating latitude/longitude cage.
    float th = sqrt(R * R - d * d);
    vec3 P = rd * (tc - th);
    vec3 N = normalize(P - C);
    vec3 n = rotY(uTime * 0.035) * rotX(0.35) * N;
    float lon = atan(n.z, n.x) / 6.28318 + 0.5;
    float lat = asin(clamp(n.y, -1.0, 1.0)) / 3.14159 + 0.5;
    float gl = abs(fract(lon * 36.0) - 0.5);
    float ga = abs(fract(lat * 18.0) - 0.5);
    float lineW = 0.035;
    float grid = max(smoothstep(lineW, 0.0, gl - 0.465), smoothstep(lineW, 0.0, ga - 0.465));
    float fres = pow(1.0 - max(dot(N, -rd), 0.0), 2.2);   // brighter toward the limb
    // Back-lit: the light sits behind the planet, so the body is night side with a lit rim.
    vec3 L = normalize(vec3(-0.35, 0.85, 0.9));
    float rim = pow(max(dot(N, L), 0.0), 3.0) * fres;
    vec3 body = vec3(0.004, 0.004, 0.008);
    body += uGrid * grid * (0.05 + 0.55 * fres);
    body += uSky * rim * 0.9;
    body += uViolet * fres * 0.10;
    // City-light speckle on the dark side: rare, dim, and faded out toward the limb so it reads as
    // distant light, not as noise (at 0.9965 / 0.55 it looked like scattered dashes).
    float sp = hash(floor(vec2(lon * 1400.0, lat * 700.0)));
    body += uPlum * step(0.9988, sp) * pow(1.0 - fres, 2.0) * 0.28;
    col = body;
  } else {
    // Above the limb: the atmosphere. Three exponential bands, tight -> wide.
    float a = max(alt, 0.0);
    float tight = exp(-a / 0.010);
    float halo = exp(-a / 0.085) * min(1.0, a / 0.006) * 0.85;
    float haze = exp(-a / 0.42) * min(1.0, a / 0.02) * 0.70;
    // A slow angular shimmer along the limb.
    float ang = atan(closest.y, closest.x);
    float shimmer = 1.0 + 0.16 * sin(ang * 46.0 - uTime * 0.6) * (tight + halo * 0.5);
    col += uSky * tight * 1.25 * shimmer;
    col += uViolet * halo * shimmer;
    col += uPlum * haze * 0.55;
  }

  // Edge falloff toward the corners (keeps the panels' contrast where the content sits).
  float vig = 1.0 - pow(length(uv * vec2(0.82, 1.0)) * 0.9, 3.0);
  col *= clamp(vig, 0.2, 1.0) * uIntensity;

  // Grain: +-12/255 of hashed noise, re-seeded each frame, dithers the gradients.
  float g = (hash(frag + fract(uTime * 7.31) * 113.0) - 0.5) * (12.0 / 255.0);
  col += g;

  gl_FragColor = vec4(max(col, 0.0), 1.0);
}
`;

const hexToVec = (value, fallback) => {
  const m = String(value || '').trim().match(/^#?([0-9a-f]{6})$/i);
  const h = m ? m[1] : fallback;
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
};

// The horizon's palette is the theme's: four CSS variables a theme may set (v2.css gives every
// theme a derivation from its accent). Re-read whenever the <html> class or style changes.
function readPalette() {
  const cs = getComputedStyle(document.documentElement);
  return {
    sky: hexToVec(cs.getPropertyValue('--fx-sky'), '59d8ff'),
    violet: hexToVec(cs.getPropertyValue('--fx-violet'), '8b5cf6'),
    plum: hexToVec(cs.getPropertyValue('--fx-plum'), 'd23aa0'),
    grid: hexToVec(cs.getPropertyValue('--fx-grid'), '59d8ff'),
    intensity: parseFloat(cs.getPropertyValue('--fx-intensity')) || 1,
  };
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error(`horizon shader: ${log}`);
  }
  return s;
}

export default function HorizonCanvas({ busy = false }) {
  const ref = useRef(null);
  const ambient = useFxStore((s) => s.ambient);
  const quality = useFxStore((s) => s.ambientQuality);
  const duringRender = useFxStore((s) => s.ambientDuringRender);
  const parallax = useFxStore((s) => s.parallax);
  // A TensorRT build / startup canary is GPU work too, even with no job running.
  const compiling = useTelemetryStore((s) => (s.run?.engine_state || 'IDLE') !== 'IDLE');
  const frozen = (busy || compiling) && !duringRender;

  useEffect(() => {
    if (!ambient) return undefined;
    const canvas = ref.current;
    if (!canvas) return undefined;
    const gl = canvas.getContext('webgl', { antialias: false, alpha: false, depth: false, stencil: false, powerPreference: 'low-power', preserveDrawingBuffer: false });
    if (!gl) return undefined;           // no WebGL: the CSS background underneath is the fallback

    const q = AMBIENT_QUALITY.find((x) => x.id === quality) || AMBIENT_QUALITY[1];
    let prog;
    try {
      prog = gl.createProgram();
      gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    } catch (e) {
      console.warn('[v2fx] horizon disabled:', e?.message || e);
      return undefined;
    }
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const u = (n) => gl.getUniformLocation(prog, n);
    const U = {
      res: u('uRes'), time: u('uTime'), mouse: u('uMouse'), sky: u('uSky'), violet: u('uViolet'),
      plum: u('uPlum'), grid: u('uGrid'), intensity: u('uIntensity'),
    };

    let palette = readPalette();
    const mo = new MutationObserver(() => { palette = readPalette(); if (stillOnly) draw(lastT); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'style', 'data-theme-mode'] });

    const resize = () => {
      const w = Math.max(1, Math.round(window.innerWidth * q.scale));
      const h = Math.max(1, Math.round(window.innerHeight * q.scale));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h);
    };
    resize();

    const target = { x: 0, y: 0 };
    const mouse = { x: 0, y: 0 };
    const onMove = (e) => {
      if (!parallax) return;
      target.x = (e.clientX / window.innerWidth) * 2 - 1;
      target.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener('pointermove', onMove, { passive: true });

    const t0 = performance.now() - 40000; // start mid-drift, not at the seam
    let lastT = 0;
    const draw = (now) => {
      lastT = now;
      mouse.x += (target.x - mouse.x) * 0.04;
      mouse.y += (target.y - mouse.y) * 0.04;
      gl.uniform2f(U.res, canvas.width, canvas.height);
      gl.uniform1f(U.time, (now - t0) / 1000);
      gl.uniform2f(U.mouse, mouse.x, mouse.y);
      gl.uniform3fv(U.sky, palette.sky);
      gl.uniform3fv(U.violet, palette.violet);
      gl.uniform3fv(U.plum, palette.plum);
      gl.uniform3fv(U.grid, palette.grid);
      gl.uniform1f(U.intensity, palette.intensity);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };

    const reduced = prefersReducedMotion();
    const stillOnly = frozen || reduced;
    let raf = 0;
    let last = 0;
    const frameMs = 1000 / q.fps;
    const loop = (now) => {
      raf = requestAnimationFrame(loop);
      if (document.hidden) return;
      if (now - last < frameMs) return;
      last = now;
      draw(now);
    };
    const onResize = () => { resize(); if (stillOnly) draw(lastT || performance.now()); };
    window.addEventListener('resize', onResize, { passive: true });

    if (stillOnly) draw(performance.now());
    else raf = requestAnimationFrame(loop);
    canvas.dataset.state = stillOnly ? 'still' : 'live';

    const onLost = (e) => { e.preventDefault(); cancelAnimationFrame(raf); };
    canvas.addEventListener('webglcontextlost', onLost);

    return () => {
      cancelAnimationFrame(raf);
      mo.disconnect();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('resize', onResize);
      canvas.removeEventListener('webglcontextlost', onLost);
      gl.deleteBuffer(buf);
      gl.deleteProgram(prog);
    };
  }, [ambient, quality, frozen, parallax]);

  if (!ambient) return null;
  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      data-testid="v2-horizon"
      className="v2-horizon"
    />
  );
}
