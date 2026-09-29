// Polar coverage chart for the 9 pose bins.
//
// Horizontal = yaw as the VIEWER sees it (a face turned to the viewer's right
// plots right), vertical = pitch with up at the top. Each bin is a cell at its
// centre angle; a filled cell carries a dot at the chosen frame's MEASURED pose,
// so a "nearest" fill visibly sits outside its own cell. Rings at 10/25/45/90
// degrees are the lattice's yaw edges.
//
// The radial scale is piecewise linear, not proportional: on a linear 90-degree
// scale the frontal, quarter and half-profile cells sit ~19 px apart and
// overlap. The knots below give each bin room; dots and rings use the same
// mapping, so everything on the chart stays on one scale.
import React from 'react';
import {
  BINS, type AngleBinName, type BinEntry, type PortfolioMap,
} from './angleHudModel';

const SIZE = 232;
const C = SIZE / 2;
const R_MAX = 100;                       // px at 90 degrees
const CELL_R = 12;
const RINGS = [10, 25, 45, 90];
// (degrees, px) knots of the radial scale.
const KNOTS: [number, number][] = [[0, 0], [10, 17], [17.5, 32], [25, 44], [35, 60], [45, 74], [62, 88], [90, R_MAX]];

const px = (deg: number): number => {
  const a = Math.min(90, Math.abs(deg));
  for (let i = 1; i < KNOTS.length; i += 1) {
    const [d1, p1] = KNOTS[i];
    if (a <= d1) {
      const [d0, p0] = KNOTS[i - 1];
      return Math.sign(deg) * (p0 + ((a - d0) / (d1 - d0)) * (p1 - p0));
    }
  }
  return Math.sign(deg) * R_MAX;
};
const at = (yaw: number, pitchUp: number): [number, number] => [C + px(yaw), C - px(pitchUp)];

const STATUS_STYLE: Record<string, { fill: string; stroke: string; dash?: string }> = {
  selected: { fill: 'rgba(34,211,238,0.22)', stroke: 'rgb(34,211,238)' },
  nearest: { fill: 'rgba(251,191,36,0.16)', stroke: 'rgb(251,191,36)' },
  override: { fill: 'rgba(167,139,250,0.2)', stroke: 'rgb(167,139,250)' },
  missing: { fill: 'rgba(15,23,42,0.6)', stroke: 'rgba(251,191,36,0.55)', dash: '3 3' },
};

export interface AngleCoverageRadarProps {
  portfolio: PortfolioMap;
  selected?: AngleBinName | null;
  scanning?: boolean;
  onSelect?: (bin: AngleBinName) => void;
}

export default function AngleCoverageRadar({ portfolio, selected, scanning, onSelect }: AngleCoverageRadarProps) {
  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="w-full max-w-[232px] mx-auto block select-none"
      role="img"
      aria-label={`Angle coverage: ${Object.keys(portfolio).length} of 9 pose bins filled`}
      data-testid="angle-radar"
    >
      <defs>
        <radialGradient id="angle-radar-bg" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(34,211,238,0.10)" />
          <stop offset="100%" stopColor="rgba(2,6,23,0)" />
        </radialGradient>
      </defs>
      <circle cx={C} cy={C} r={R_MAX + 8} fill="url(#angle-radar-bg)" />
      {RINGS.map((deg) => (
        <circle key={deg} cx={C} cy={C} r={Math.abs(px(deg))} fill="none"
                stroke="rgba(148,163,184,0.18)" strokeWidth={deg === 90 ? 1 : 0.75}
                strokeDasharray={deg === 90 ? undefined : '2 3'} />
      ))}
      <line x1={C - R_MAX} y1={C} x2={C + R_MAX} y2={C} stroke="rgba(148,163,184,0.18)" strokeWidth={0.75} />
      <line x1={C} y1={C - R_MAX} x2={C} y2={C + R_MAX} stroke="rgba(148,163,184,0.18)" strokeWidth={0.75} />
      {scanning && (
        <g className="origin-center animate-spin" style={{ transformOrigin: `${C}px ${C}px`, animationDuration: '2.4s' }}>
          <path d={`M ${C} ${C} L ${C} ${C - R_MAX} A ${R_MAX} ${R_MAX} 0 0 1 ${C + R_MAX * 0.5} ${C - R_MAX * 0.866} Z`}
                fill="rgba(34,211,238,0.10)" />
        </g>
      )}
      <text x={C - R_MAX} y={C - 5} fill="rgba(148,163,184,0.55)" fontSize="8" fontFamily="ui-monospace, monospace">L</text>
      <text x={C + R_MAX - 5} y={C - 5} fill="rgba(148,163,184,0.55)" fontSize="8" fontFamily="ui-monospace, monospace">R</text>
      <text x={C + 4} y={C - R_MAX + 8} fill="rgba(148,163,184,0.55)" fontSize="8" fontFamily="ui-monospace, monospace">UP</text>
      <text x={C + 4} y={C + R_MAX - 2} fill="rgba(148,163,184,0.55)" fontSize="8" fontFamily="ui-monospace, monospace">DN</text>

      {BINS.map((spec) => {
        const entry: BinEntry | undefined = portfolio[spec.name];
        const status = entry ? entry.status : 'missing';
        const style = STATUS_STYLE[status] || STATUS_STYLE.missing;
        const [x, y] = at(spec.radar[0], spec.radar[1]);
        const isSel = selected === spec.name;
        return (
          <g key={spec.name} data-bin={spec.name} data-status={status}
             onClick={onSelect ? () => onSelect(spec.name) : undefined}
             style={{ cursor: onSelect ? 'pointer' : 'default' }}>
            <title>{`${spec.label}: ${status}`}</title>
            <circle cx={x} cy={y} r={CELL_R} fill={style.fill} stroke={style.stroke}
                    strokeWidth={isSel ? 2 : 1} strokeDasharray={style.dash} />
            <text x={x} y={y + 3} textAnchor="middle" fontSize="7.5" fontWeight={600}
                  fontFamily="ui-monospace, monospace"
                  fill={status === 'missing' ? 'rgba(251,191,36,0.75)' : 'rgba(226,232,240,0.92)'}>
              {spec.short}
            </text>
          </g>
        );
      })}
      {BINS.map((spec) => {
        const entry = portfolio[spec.name];
        if (!entry || entry.yaw == null || entry.pitch_up == null) return null;
        const [x, y] = at(entry.yaw, entry.pitch_up);
        return <circle key={`${spec.name}-dot`} cx={x} cy={y} r={2.2} fill="rgb(165,243,252)" pointerEvents="none" />;
      })}
    </svg>
  );
}
