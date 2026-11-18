import React, { useMemo } from 'react';
import { PITCH_BANDS, YAW_BANDS, computeCoverage } from './identityModel';

// Head-angle coverage radar. Yaw runs along X (-90° left .. +90° right), pitch along Y (up at the top), so a
// reference photo sits where the head was pointing. The plane is cut into the 5x3 zones identityModel.js defines;
// concentric rings are angular distance from straight-on (a polar read of the same plane). A zone a swap needs but
// no photo covers is hatched amber and named underneath, so the user knows which angle to add.

const W = 300;
const H = 190;
const PAD = 22;
const X = (yaw) => PAD + ((Math.max(-90, Math.min(90, yaw)) + 90) / 180) * (W - PAD * 2);
const Y = (pitch) => PAD + ((45 - Math.max(-45, Math.min(45, pitch))) / 90) * (H - PAD * 2);
const sharpColor = (s) => `hsl(${Math.round(Math.max(0, Math.min(1, s)) * 120)} 70% 52%)`;

export default function CoverageRadar({ points, selected = [], onSelect, title = 'Head angle coverage' }) {
  const cov = useMemo(() => computeCoverage(points), [points]);
  const sel = useMemo(() => new Set(selected), [selected]);
  const empty = points.length === 0;

  return (
    <figure className="m-0 space-y-1.5" data-testid="coverage-radar">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img"
           aria-label={`${title}: ${cov.covered} of ${cov.required} needed angles covered. ${cov.missing.map((m) => m.label).join(', ') || 'Nothing missing'}`}>
        <defs>
          <pattern id="cov-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="#fbbf24" strokeOpacity="0.45" strokeWidth="2" />
          </pattern>
        </defs>

        {/* zones */}
        {YAW_BANDS.map((yb) => PITCH_BANDS.map((pb) => {
          const id = `${yb.id}/${pb.id}`;
          const z = cov.zones[id];
          const miss = cov.missing.some((m) => m.id === id);
          const x = X(yb.lo);
          const y = Y(pb.hi);
          const w = X(yb.hi) - x;
          const h = Y(pb.lo) - y;
          return (
            <g key={id} data-zone={id} data-covered={z.count > 0 ? 'true' : 'false'} data-missing={miss ? 'true' : 'false'}>
              <rect x={x} y={y} width={w} height={h}
                    fill={miss ? 'url(#cov-hatch)' : 'var(--accent)'}
                    fillOpacity={miss ? 1 : z.count ? Math.min(0.5, 0.14 + z.count * 0.08) : 0}
                    stroke={miss ? '#fbbf24' : 'currentColor'} strokeOpacity={miss ? 0.7 : 0.16}
                    strokeDasharray={miss ? '3 3' : undefined} />
              {z.count > 0 && (
                <text x={x + w - 3} y={y + 9} textAnchor="end" fontSize="8" fill="currentColor" fillOpacity="0.6">{z.count}</text>
              )}
            </g>
          );
        }))}

        {/* angular-distance rings around straight-on */}
        {[30, 60, 90].map((r) => (
          <ellipse key={r} cx={X(0)} cy={Y(0)} rx={X(r) - X(0)} ry={(Y(0) - Y(45)) * (r / 90)}
                   fill="none" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="2 4" />
        ))}
        <line x1={X(0)} x2={X(0)} y1={PAD} y2={H - PAD} stroke="currentColor" strokeOpacity="0.2" />
        <line x1={PAD} x2={W - PAD} y1={Y(0)} y2={Y(0)} stroke="currentColor" strokeOpacity="0.2" />

        {/* axis labels */}
        <text x={PAD} y={H - 6} fontSize="8.5" fill="currentColor" fillOpacity="0.55">← −90° yaw</text>
        <text x={W - PAD} y={H - 6} fontSize="8.5" textAnchor="end" fill="currentColor" fillOpacity="0.55">+90° yaw →</text>
        <text x={W / 2} y={11} fontSize="8.5" textAnchor="middle" fill="currentColor" fillOpacity="0.55">↑ +45° pitch</text>

        {/* faces */}
        {points.map((p, i) => {
          const on = sel.has(p.key);
          return (
            <circle key={`${p.key}-${i}`} data-testid="coverage-dot" cx={X(p.yaw)} cy={Y(p.pitch)} r={on ? 6 : 4.2}
                    fill={typeof p.sharpness === 'number' ? sharpColor(p.sharpness) : 'var(--accent)'}
                    stroke={on ? '#fff' : 'rgba(0,0,0,0.55)'} strokeWidth={on ? 2 : 1}
                    style={{ cursor: onSelect ? 'pointer' : 'default' }}
                    onClick={(e) => onSelect?.(p.key, e)}>
              <title>{`Face ${Number(p.key) + 1}: yaw ${Math.round(p.yaw)}°, pitch ${Math.round(p.pitch)}°${p.approx ? ' (estimated from pose label)' : ''}`}</title>
            </circle>
          );
        })}
      </svg>

      <figcaption className="text-nano text-muted flex flex-wrap items-center gap-x-3 gap-y-0.5">
        <span data-testid="coverage-score"><span className="font-bold text-white/70">{cov.covered}/{cov.required}</span> needed angles</span>
        {cov.approx && <span title="Target people report a pose label, not degrees">angles estimated from pose label</span>}
      </figcaption>

      {empty ? (
        <p className="text-mini text-muted m-0" data-testid="coverage-empty">No faces here yet.</p>
      ) : cov.missing.length > 0 ? (
        <ul className="m-0 p-0 list-none flex flex-wrap gap-1.5" data-testid="coverage-missing">
          {cov.missing.map((m) => (
            <li key={m.id} data-missing-zone={m.id}
                className="px-2 py-0.5 rounded-md border border-amber-500/40 bg-amber-950/30 text-nano font-semibold text-amber-300">
              {m.label}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-mini text-emerald-400 m-0" data-testid="coverage-complete">Every needed angle is covered.</p>
      )}
    </figure>
  );
}
