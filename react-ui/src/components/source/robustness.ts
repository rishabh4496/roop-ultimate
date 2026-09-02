export type PoseAngle = 'frontal' | 'left45' | 'right45' | 'up' | 'down';

export interface FaceCapture {
  id: string;
  file: File;
  previewUrl: string;
  angle: PoseAngle;
  weight: number; // 0.1 to 1.0
  luminance: number; // 0 to 255
  width: number;
  height: number;
  error?: string;
}

export interface CentroidResponse {
  status: 'ok' | 'error';
  centroid_id?: string;
  identity_name?: string;
  embedding_dim?: number;
  faces_aggregated?: number;
  robustness_score?: number;
  message?: string;
  invalid_indices?: number[];
}

export const POSE_DEFINITIONS: Record<
  PoseAngle,
  { label: string; shortLabel: string; description: string; color: string; bgClass: string; borderClass: string }
> = {
  frontal: {
    label: 'Frontal',
    shortLabel: 'Front',
    description: 'Direct forward-facing anchor pose',
    color: '#10b981', // Emerald
    bgClass: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    borderClass: 'border-emerald-500',
  },
  left45: {
    label: 'Left 45°',
    shortLabel: 'Left 45',
    description: 'Left yaw profile for left turns',
    color: '#06b6d4', // Cyan
    bgClass: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
    borderClass: 'border-cyan-500',
  },
  right45: {
    label: 'Right 45°',
    shortLabel: 'Right 45',
    description: 'Right yaw profile for right turns',
    color: '#8b5cf6', // Violet
    bgClass: 'bg-violet-500/15 text-violet-300 border-violet-500/30',
    borderClass: 'border-violet-500',
  },
  up: {
    label: 'Up',
    shortLabel: 'Up',
    description: 'Elevated pitch angle',
    color: '#f59e0b', // Amber
    bgClass: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    borderClass: 'border-amber-500',
  },
  down: {
    label: 'Down',
    shortLabel: 'Down',
    description: 'Lowered pitch angle',
    color: '#f43f5e', // Rose
    bgClass: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
    borderClass: 'border-rose-500',
  },
};

export const CANONICAL_ANGLES: PoseAngle[] = ['frontal', 'left45', 'right45', 'up', 'down'];

/**
 * Calculates Centroid Robustness Score & Diversity Analysis:
 *  - 1 frontal photo only = 40% (Warning: "Prone to dropouts on turns")
 *  - Frontal + Left + Right profiles = 85% ("High stability")
 *  - Full 5-angle coverage with varied lighting = 98% - 100% ("Optimal tracking")
 */
export function evaluateCentroidRobustness(captures: FaceCapture[]): {
  score: number;
  status: 'critical' | 'warning' | 'high' | 'optimal';
  label: string;
  description: string;
  coverage: Record<PoseAngle, { count: number; maxWeight: number; covered: boolean }>;
  lightingVariance: number;
  missingAngles: PoseAngle[];
} {
  const coverage: Record<PoseAngle, { count: number; maxWeight: number; covered: boolean }> = {
    frontal: { count: 0, maxWeight: 0, covered: false },
    left45: { count: 0, maxWeight: 0, covered: false },
    right45: { count: 0, maxWeight: 0, covered: false },
    up: { count: 0, maxWeight: 0, covered: false },
    down: { count: 0, maxWeight: 0, covered: false },
  };

  if (captures.length === 0) {
    return {
      score: 0,
      status: 'critical',
      label: '0% Robustness',
      description: 'Upload 3 to 8 multi-angle face captures to generate a centroid.',
      coverage,
      lightingVariance: 0,
      missingAngles: CANONICAL_ANGLES,
    };
  }

  // Aggregate angle coverage and weights
  captures.forEach((c) => {
    if (coverage[c.angle]) {
      coverage[c.angle].count += 1;
      coverage[c.angle].maxWeight = Math.max(coverage[c.angle].maxWeight, c.weight);
      coverage[c.angle].covered = true;
    }
  });

  // Calculate luminance standard deviation across images
  const luminances = captures.map((c) => c.luminance);
  const avgLum = luminances.reduce((acc, l) => acc + l, 0) / luminances.length;
  const variance =
    luminances.reduce((acc, l) => acc + Math.pow(l - avgLum, 2), 0) / luminances.length;
  const lightingStdDev = Math.sqrt(variance);

  // Exact benchmark math:
  // Frontal anchor: 40.0%
  // Left 45 yaw profile: 22.5%
  // Right 45 yaw profile: 22.5%
  // Up pitch profile: 6.5%
  // Down pitch profile: 6.5%
  // Total angle coverage = 98.0%
  let rawScore = 0;
  if (coverage.frontal.covered) {
    rawScore += 40.0 * coverage.frontal.maxWeight;
  }
  if (coverage.left45.covered) {
    rawScore += 22.5 * coverage.left45.maxWeight;
  }
  if (coverage.right45.covered) {
    rawScore += 22.5 * coverage.right45.maxWeight;
  }
  if (coverage.up.covered) {
    rawScore += 6.5 * coverage.up.maxWeight;
  }
  if (coverage.down.covered) {
    rawScore += 6.5 * coverage.down.maxWeight;
  }

  // Lighting diversity bonus (+2% if varied lighting captured)
  if (lightingStdDev >= 12 && captures.length >= 3) {
    rawScore += 2.0;
  }

  const score = Math.max(0, Math.min(100, Math.round(rawScore)));
  const missingAngles = CANONICAL_ANGLES.filter((a) => !coverage[a].covered);

  let status: 'critical' | 'warning' | 'high' | 'optimal' = 'critical';
  let label = `${score}% Robustness`;
  let description = '';

  if (score <= 45) {
    status = 'warning';
    description = 'Warning: Prone to dropouts on turns';
  } else if (score <= 75) {
    status = 'warning';
    description = 'Moderate stability — add lateral profile angles';
  } else if (score <= 90) {
    status = 'high';
    description = 'High stability';
  } else {
    status = 'optimal';
    description = 'Optimal tracking';
  }

  return {
    score,
    status,
    label,
    description,
    coverage,
    lightingVariance: Math.round(lightingStdDev),
    missingAngles,
  };
}

export async function analyzeCaptureLuminance(file: File): Promise<{ luminance: number; width: number; height: number }> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const w = Math.min(128, img.naturalWidth || 128);
      const h = Math.min(128, img.naturalHeight || 128);
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        URL.revokeObjectURL(url);
        resolve({ luminance: 128, width: img.naturalWidth, height: img.naturalHeight });
        return;
      }

      ctx.drawImage(img, 0, 0, w, h);
      const imgData = ctx.getImageData(0, 0, w, h).data;
      let sumL = 0;
      let count = 0;
      for (let i = 0; i < imgData.length; i += 4) {
        const lum = 0.299 * imgData[i] + 0.587 * imgData[i + 1] + 0.114 * imgData[i + 2];
        sumL += lum;
        count++;
      }

      URL.revokeObjectURL(url);
      resolve({
        luminance: count > 0 ? Math.round(sumL / count) : 128,
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({ luminance: 128, width: 0, height: 0 });
    };
    img.src = url;
  });
}
