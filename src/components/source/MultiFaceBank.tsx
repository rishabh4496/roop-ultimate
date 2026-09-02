import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Users,
  UploadCloud,
  Trash2,
  Check,
  AlertCircle,
  AlertTriangle,
  Loader2,
  Sparkles,
  ShieldCheck,
  Plus,
  Info,
  ChevronDown,
} from 'lucide-react';

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

export interface MultiFaceBankProps {
  /** Target identity name or subject label */
  initialIdentityName?: string;
  /** Backend endpoint for serializing the centroid (default: '/api/faces/centroid') */
  apiEndpoint?: string;
  /** Callback fired when centroid is successfully processed */
  onCentroidGenerated?: (result: CentroidResponse) => void;
  /** Callback fired whenever the capture list or weights update */
  onChange?: (captures: FaceCapture[]) => void;
  /** Custom CSS styling classes */
  className?: string;
}

export interface MultiFaceBankRef {
  submitCentroid: () => Promise<CentroidResponse>;
  getCaptures: () => FaceCapture[];
  clearCaptures: () => void;
  addFiles: (files: File[]) => Promise<void>;
  getRobustnessScore: () => number;
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

export const MultiFaceBank = forwardRef<MultiFaceBankRef, MultiFaceBankProps>(
  (
    {
      initialIdentityName = 'Subject_01',
      apiEndpoint = '/api/faces/centroid',
      onCentroidGenerated,
      onChange,
      className = '',
    },
    ref
  ) => {
    const [identityName, setIdentityName] = useState<string>(initialIdentityName);
    const [captures, setCaptures] = useState<FaceCapture[]>([]);
    const [isDragging, setIsDragging] = useState<boolean>(false);
    const [openAngleDropdownId, setOpenAngleDropdownId] = useState<string | null>(null);

    // Upload & Network States
    const [uploadPhase, setUploadPhase] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error'>('idle');
    const [uploadProgress, setUploadProgress] = useState<number>(0);
    const [uploadStatusText, setUploadStatusText] = useState<string>('');
    const [centroidResult, setCentroidResult] = useState<CentroidResponse | null>(null);

    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const currentXhrRef = useRef<XMLHttpRequest | null>(null);

    // Compute Centroid Robustness
    const robustness = useMemo(() => evaluateCentroidRobustness(captures), [captures]);

    // Cleanup object URLs on unmount
    useEffect(() => {
      return () => {
        captures.forEach((c) => URL.revokeObjectURL(c.previewUrl));
        if (currentXhrRef.current) {
          currentXhrRef.current.abort();
        }
      };
    }, [captures]);

    // Close dropdowns on outside click
    useEffect(() => {
      const handleOutsideClick = () => setOpenAngleDropdownId(null);
      window.addEventListener('click', handleOutsideClick);
      return () => window.removeEventListener('click', handleOutsideClick);
    }, []);

    // ── File Ingestion with Smart Angle Distribution ───────────────────────
    const processFiles = useCallback(
      async (files: File[]) => {
        const imageFiles = files.filter((f) => f.type.startsWith('image/'));
        if (imageFiles.length === 0) return;

        const currentCount = captures.length;
        const availableSlots = Math.max(0, 8 - currentCount);
        if (availableSlots <= 0) return;

        const filesToProcess = imageFiles.slice(0, availableSlots);

        // Intelligently cycle through missing angles or canonical sequence
        const newCaptures: FaceCapture[] = [];
        for (let i = 0; i < filesToProcess.length; i++) {
          const file = filesToProcess[i];
          const previewUrl = URL.createObjectURL(file);
          const { luminance, width, height } = await analyzeCaptureLuminance(file);

          // Select next recommended angle
          const overallIndex = currentCount + i;
          let assignedAngle: PoseAngle = 'frontal';
          if (overallIndex === 0) assignedAngle = 'frontal';
          else if (overallIndex === 1) assignedAngle = 'left45';
          else if (overallIndex === 2) assignedAngle = 'right45';
          else if (overallIndex === 3) assignedAngle = 'up';
          else if (overallIndex === 4) assignedAngle = 'down';
          else assignedAngle = CANONICAL_ANGLES[overallIndex % CANONICAL_ANGLES.length];

          newCaptures.push({
            id: `capture_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
            file,
            previewUrl,
            angle: assignedAngle,
            weight: 1.0,
            luminance,
            width,
            height,
          });
        }

        setCaptures((prev) => {
          const updated = [...prev, ...newCaptures];
          onChange?.(updated);
          return updated;
        });

        // Reset server error banner on new uploads
        if (uploadPhase === 'error') {
          setUploadPhase('idle');
          setUploadStatusText('');
        }
      },
      [captures.length, onChange, uploadPhase]
    );

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        processFiles(Array.from(e.target.files));
      }
      e.target.value = '';
    };

    const handleDragOver = (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      if (e.dataTransfer.files) {
        processFiles(Array.from(e.dataTransfer.files));
      }
    };

    // ── Item Actions: Weight, Angle, Delete ─────────────────────────────────
    const handleSetWeight = (id: string, weight: number) => {
      setCaptures((prev) => {
        const updated = prev.map((c) =>
          c.id === id ? { ...c, weight: Math.max(0.1, Math.min(1.0, weight)) } : c
        );
        onChange?.(updated);
        return updated;
      });
    };

    const handleSetAngle = (id: string, angle: PoseAngle) => {
      setCaptures((prev) => {
        const updated = prev.map((c) => (c.id === id ? { ...c, angle } : c));
        onChange?.(updated);
        return updated;
      });
      setOpenAngleDropdownId(null);
    };

    const handleDelete = (id: string) => {
      setCaptures((prev) => {
        const target = prev.find((c) => c.id === id);
        if (target) URL.revokeObjectURL(target.previewUrl);
        const updated = prev.filter((c) => c.id !== id);
        onChange?.(updated);
        return updated;
      });
    };

    const handleClearAll = useCallback(() => {
      captures.forEach((c) => URL.revokeObjectURL(c.previewUrl));
      setCaptures([]);
      onChange?.([]);
      setCentroidResult(null);
      setUploadPhase('idle');
    }, [captures, onChange]);

    // ── Payload Serialization & Backend Transmission ────────────────────────
    const submitCentroid = useCallback(async (): Promise<CentroidResponse> => {
      if (captures.length === 0) {
        throw new Error('At least one face capture is required to generate a centroid.');
      }

      setUploadPhase('uploading');
      setUploadProgress(0);
      setUploadStatusText('Serializing multi-angle captures...');

      return new Promise<CentroidResponse>((resolve, reject) => {
        const formData = new FormData();
        const cleanName = identityName.trim() || 'Target_Subject';
        formData.append('identity_name', cleanName);
        formData.append('robustness_score', String(robustness.score));

        // Serialize all image captures with their assigned angles and weights
        captures.forEach((c, idx) => {
          formData.append('images', c.file, c.file.name);
          formData.append(`weights`, String(c.weight));
          formData.append(`angles`, c.angle);
          formData.append(`file_${idx}_angle`, c.angle);
          formData.append(`file_${idx}_weight`, String(c.weight));
        });

        // Structured JSON metadata payload
        formData.append(
          'metadata',
          JSON.stringify({
            identity_name: cleanName,
            image_count: captures.length,
            robustness_score: robustness.score,
            lighting_variance: robustness.lightingVariance,
            angle_map: captures.map((c) => ({ angle: c.angle, weight: c.weight, name: c.file.name })),
            timestamp: Date.now(),
          })
        );

        const xhr = new XMLHttpRequest();
        currentXhrRef.current = xhr;
        xhr.open('POST', apiEndpoint);

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            setUploadProgress(percent);
            setUploadStatusText(
              `Uploading captures: ${(e.loaded / 1048576).toFixed(1)} MB / ${(e.total / 1048576).toFixed(1)} MB (${percent}%)`
            );
          } else {
            setUploadProgress(50);
            setUploadStatusText('Transmitting multi-file payload...');
          }
        };

        xhr.upload.onload = () => {
          setUploadPhase('processing');
          setUploadProgress(100);
          setUploadStatusText('Extracting 512-D face embeddings & computing centroid cluster...');
        };

        xhr.onload = () => {
          currentXhrRef.current = null;
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const res: CentroidResponse =
                typeof xhr.response === 'object' && xhr.response !== null
                  ? xhr.response
                  : JSON.parse(xhr.responseText);

              setUploadPhase('success');
              setUploadStatusText(
                res.message ||
                  `Identity centroid created with ${res.faces_aggregated ?? captures.length} multi-angle captures.`
              );
              setCentroidResult(res);
              onCentroidGenerated?.(res);
              resolve(res);
            } catch {
              const fallbackRes: CentroidResponse = {
                status: 'ok',
                identity_name: cleanName,
                faces_aggregated: captures.length,
                robustness_score: robustness.score,
                message: 'Centroid computed successfully.',
              };
              setUploadPhase('success');
              setUploadStatusText('Identity centroid generated successfully.');
              setCentroidResult(fallbackRes);
              onCentroidGenerated?.(fallbackRes);
              resolve(fallbackRes);
            }
          } else {
            let errorMsg = `Server error (HTTP ${xhr.status})`;
            let invalidIndices: number[] = [];

            try {
              const errObj = JSON.parse(xhr.responseText);
              if (errObj.message) errorMsg = errObj.message;
              if (Array.isArray(errObj.invalid_indices)) {
                invalidIndices = errObj.invalid_indices;
              }
            } catch {
              errorMsg = xhr.statusText || errorMsg;
            }

            // Flag individual bad files if returned by backend
            if (invalidIndices.length > 0) {
              setCaptures((prev) =>
                prev.map((c, idx) =>
                  invalidIndices.includes(idx)
                    ? { ...c, error: 'No face detected in this capture' }
                    : c
                )
              );
            }

            setUploadPhase('error');
            setUploadStatusText(errorMsg);
            const err = new Error(errorMsg);
            reject(err);
          }
        };

        xhr.onerror = () => {
          currentXhrRef.current = null;
          setUploadPhase('error');
          setUploadStatusText('Network error while connecting to /api/faces/centroid');
          reject(new Error('Network error during centroid upload'));
        };

        xhr.onabort = () => {
          currentXhrRef.current = null;
          setUploadPhase('idle');
          setUploadStatusText('Upload cancelled');
          reject(new DOMException('Upload aborted', 'AbortError'));
        };

        xhr.send(formData);
      });
    }, [apiEndpoint, captures, identityName, onCentroidGenerated, robustness.lightingVariance, robustness.score]);

    // ── Imperative Handle ──────────────────────────────────────────────────
    useImperativeHandle(
      ref,
      () => ({
        submitCentroid,
        getCaptures: () => captures,
        clearCaptures: handleClearAll,
        addFiles: processFiles,
        getRobustnessScore: () => robustness.score,
      }),
      [captures, handleClearAll, processFiles, robustness.score, submitCentroid]
    );

    // ── SVG Radar Coordinates (5 Canonical Angles Pentagon) ────────────────
    const radarData = useMemo(() => {
      const cx = 100;
      const cy = 100;
      const radius = 72;

      const axes: Array<{ angleKey: PoseAngle; label: string; x: number; y: number; val: number }> = [
        {
          angleKey: 'frontal',
          label: 'Front',
          val: robustness.coverage.frontal.covered ? robustness.coverage.frontal.maxWeight : 0.05,
          x: cx + radius * Math.cos((-90 * Math.PI) / 180),
          y: cy + radius * Math.sin((-90 * Math.PI) / 180),
        },
        {
          angleKey: 'right45',
          label: 'R 45°',
          val: robustness.coverage.right45.covered ? robustness.coverage.right45.maxWeight : 0.05,
          x: cx + radius * Math.cos((-18 * Math.PI) / 180),
          y: cy + radius * Math.sin((-18 * Math.PI) / 180),
        },
        {
          angleKey: 'down',
          label: 'Down',
          val: robustness.coverage.down.covered ? robustness.coverage.down.maxWeight : 0.05,
          x: cx + radius * Math.cos((54 * Math.PI) / 180),
          y: cy + radius * Math.sin((54 * Math.PI) / 180),
        },
        {
          angleKey: 'left45',
          label: 'L 45°',
          val: robustness.coverage.left45.covered ? robustness.coverage.left45.maxWeight : 0.05,
          x: cx + radius * Math.cos((126 * Math.PI) / 180),
          y: cy + radius * Math.sin((126 * Math.PI) / 180),
        },
        {
          angleKey: 'up',
          label: 'Up',
          val: robustness.coverage.up.covered ? robustness.coverage.up.maxWeight : 0.05,
          x: cx + radius * Math.cos((198 * Math.PI) / 180),
          y: cy + radius * Math.sin((198 * Math.PI) / 180),
        },
      ];

      const levels = [0.33, 0.66, 1.0];
      const levelPolygons = levels.map((lvl) => {
        return axes
          .map((a, i) => {
            const rad = radius * lvl;
            const deg = -90 + i * 72;
            const px = cx + rad * Math.cos((deg * Math.PI) / 180);
            const py = cy + rad * Math.sin((deg * Math.PI) / 180);
            return `${px.toFixed(1)},${py.toFixed(1)}`;
          })
          .join(' ');
      });

      const valuePoints = axes
        .map((a, i) => {
          const rad = radius * a.val;
          const deg = -90 + i * 72;
          const px = cx + rad * Math.cos((deg * Math.PI) / 180);
          const py = cy + rad * Math.sin((deg * Math.PI) / 180);
          return { x: px, y: py };
        });

      const valuePolygonString = valuePoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');

      return { cx, cy, radius, axes, levelPolygons, valuePoints, valuePolygonString };
    }, [robustness.coverage]);

    return (
      <div
        className={`flex flex-col rounded-2xl border border-white/10 bg-neutral-950/85 text-white shadow-2xl backdrop-blur-xl ${className}`}
      >
        {/* ── Header ──────────────────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 text-emerald-400 border border-emerald-500/30">
              <Users className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold tracking-wide text-white">Multi-Angle Face Bank</h2>
              <p className="text-xs text-white/50">
                Generate 3D identity centroid to eliminate tracking dropouts during head turns
              </p>
            </div>
          </div>

          {/* Identity Name Input & Count Indicator */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 focus-within:border-emerald-500/50">
              <span className="text-xs font-semibold text-white/50">Identity:</span>
              <input
                type="text"
                value={identityName}
                onChange={(e) => setIdentityName(e.target.value)}
                placeholder="e.g. Hero_Scarlett"
                className="w-32 sm:w-40 bg-transparent text-xs font-medium text-white focus:outline-none"
              />
            </div>

            <div
              className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold ${
                captures.length >= 3 && captures.length <= 8
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                  : 'border-amber-500/30 bg-amber-500/10 text-amber-300'
              }`}
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>{captures.length} / 8 Images</span>
            </div>
          </div>
        </div>

        {/* ── Main Content Body ────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 gap-6 p-6 lg:grid-cols-12">
          {/* Left Column: Upload Dropzone & Thumbnail Grid (8 cols) */}
          <div className="flex flex-col gap-4 lg:col-span-8">
            {/* Drag & Drop Upload Zone */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/png,image/jpeg,image/webp"
              multiple
              className="hidden"
            />

            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`group flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 transition-all ${
                isDragging
                  ? 'border-emerald-400 bg-emerald-500/10 shadow-lg shadow-emerald-950/30'
                  : 'border-white/15 bg-neutral-900/40 hover:border-white/30 hover:bg-neutral-900/70'
              }`}
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/5 text-white/70 transition-transform group-hover:scale-110 group-hover:text-white">
                <UploadCloud className="h-6 w-6" />
              </div>
              <p className="mt-2 text-xs font-semibold text-white">
                Drag & Drop 3 to 8 Face Captures or <span className="text-emerald-400 underline">Browse Files</span>
              </p>
              <p className="mt-1 text-[11px] text-white/40">
                Frontal, Left 45°, Right 45°, Up, and Down angles for maximum centroid stability
              </p>
            </div>

            {/* Thumbnail Cluster Grid */}
            {captures.length > 0 ? (
              <div className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 md:grid-cols-4">
                {captures.map((capture, idx) => {
                  const angleDef = POSE_DEFINITIONS[capture.angle];
                  const isDropdownOpen = openAngleDropdownId === capture.id;

                  return (
                    <div
                      key={capture.id}
                      className={`group relative flex flex-col rounded-xl border p-2.5 transition-all ${
                        capture.error
                          ? 'border-red-500/80 bg-red-950/20 shadow-md shadow-red-950/30'
                          : 'border-white/10 bg-neutral-900/60 hover:border-white/20'
                      }`}
                    >
                      {/* Image Thumbnail Container */}
                      <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-neutral-950">
                        <img
                          src={capture.previewUrl}
                          alt={capture.file.name}
                          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                        />

                        {/* Top Overlay Badges */}
                        <div className="absolute left-1.5 top-1.5 flex items-center gap-1">
                          <span className="rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-white/80 backdrop-blur-md">
                            #{idx + 1}
                          </span>
                        </div>

                        {/* Delete Button */}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(capture.id);
                          }}
                          className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-lg bg-black/60 text-white/70 opacity-0 backdrop-blur-md transition-all hover:bg-red-500 hover:text-white group-hover:opacity-100"
                          title="Delete capture"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>

                        {/* Error Ribbon if no face detected */}
                        {capture.error && (
                          <div className="absolute inset-x-0 bottom-0 flex items-center gap-1 bg-red-600/90 px-2 py-1 text-[10px] font-semibold text-white backdrop-blur-sm">
                            <AlertCircle className="h-3 w-3 shrink-0" />
                            <span className="truncate">{capture.error}</span>
                          </div>
                        )}
                      </div>

                      {/* Pose Angle Badge & Selector Dropdown */}
                      <div className="relative mt-2.5">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setOpenAngleDropdownId(isDropdownOpen ? null : capture.id);
                          }}
                          className={`flex w-full items-center justify-between rounded-lg border px-2 py-1 text-xs font-semibold transition-all ${angleDef.bgClass}`}
                          title="Click to switch detected pose angle"
                        >
                          <span className="truncate">{angleDef.label}</span>
                          <ChevronDown className="h-3 w-3 opacity-60" />
                        </button>

                        {/* Dropdown Menu for Angle Selection */}
                        {isDropdownOpen && (
                          <div
                            onClick={(e) => e.stopPropagation()}
                            className="absolute left-0 top-full z-30 mt-1 w-full rounded-xl border border-white/15 bg-neutral-900/95 p-1 shadow-2xl backdrop-blur-xl"
                          >
                            {CANONICAL_ANGLES.map((ang) => {
                              const itemDef = POSE_DEFINITIONS[ang];
                              return (
                                <button
                                  key={ang}
                                  type="button"
                                  onClick={() => handleSetAngle(capture.id, ang)}
                                  className={`flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left text-xs font-medium transition-all ${
                                    capture.angle === ang
                                      ? 'bg-white/15 text-white'
                                      : 'text-white/70 hover:bg-white/5 hover:text-white'
                                  }`}
                                >
                                  <span>{itemDef.label}</span>
                                  {capture.angle === ang && <Check className="h-3.5 w-3.5 text-emerald-400" />}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </div>

                      {/* Weight Slider (0.1 to 1.0) */}
                      <div className="mt-2.5 flex flex-col gap-1">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="text-white/50">Weight:</span>
                          <span className="font-mono font-semibold text-white/90">
                            {capture.weight.toFixed(2)}x
                          </span>
                        </div>
                        <input
                          type="range"
                          min="0.1"
                          max="1.0"
                          step="0.05"
                          value={capture.weight}
                          onChange={(e) => handleSetWeight(capture.id, Number(e.target.value))}
                          className="h-1.5 cursor-pointer appearance-none rounded-lg bg-white/15 accent-emerald-400 focus:outline-none"
                          title="Adjust weight contribution in centroid embedding"
                        />
                      </div>

                      {/* Capture Specs */}
                      <div className="mt-2 flex items-center justify-between text-[10px] text-white/40">
                        <span className="truncate">{capture.width > 0 ? `${capture.width}×${capture.height}` : 'Image'}</span>
                        <span className="font-mono">{(capture.file.size / 1024).toFixed(0)} KB</span>
                      </div>
                    </div>
                  );
                })}

                {/* Add More Button if slots available */}
                {captures.length < 8 && (
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="flex flex-col items-center justify-center rounded-xl border border-dashed border-white/15 bg-white/5 p-4 text-white/50 transition-all hover:border-emerald-500/40 hover:bg-emerald-500/5 hover:text-emerald-300"
                  >
                    <Plus className="h-6 w-6" />
                    <span className="mt-2 text-xs font-semibold">Add Angle</span>
                    <span className="text-[10px] text-white/40">({8 - captures.length} slots left)</span>
                  </button>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-white/2 py-10 text-center">
                <p className="text-xs text-white/40">No reference images uploaded yet.</p>
                <p className="text-[11px] text-white/30">Drag files above to preview angle clustering.</p>
              </div>
            )}
          </div>

          {/* Right Column: Centroid Robustness & Diversity Analytics (4 cols) */}
          <div className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-neutral-900/40 p-5 lg:col-span-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-emerald-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-white/80">Centroid Robustness</h3>
              </div>
              <span className="font-mono text-lg font-extrabold text-white">{robustness.score}%</span>
            </div>

            {/* Score Progress Bar */}
            <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-white/10">
              <div
                className={`h-full transition-all duration-500 ${
                  robustness.score >= 90
                    ? 'bg-gradient-to-r from-emerald-500 to-green-400'
                    : robustness.score >= 75
                    ? 'bg-gradient-to-r from-cyan-500 to-emerald-400'
                    : robustness.score >= 40
                    ? 'bg-gradient-to-r from-amber-500 to-yellow-400'
                    : 'bg-gradient-to-r from-rose-500 to-red-400'
                }`}
                style={{ width: `${robustness.score}%` }}
              />
            </div>

            {/* Status Headline Banner */}
            <div
              className={`rounded-xl border px-3 py-2 text-xs font-semibold ${
                robustness.status === 'optimal'
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                  : robustness.status === 'high'
                  ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300'
                  : 'border-amber-500/30 bg-amber-500/10 text-amber-300'
              }`}
            >
              <div className="flex items-center gap-1.5">
                {robustness.status === 'optimal' ? (
                  <Check className="h-3.5 w-3.5 shrink-0" />
                ) : (
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                )}
                <span>{robustness.description}</span>
              </div>
            </div>

            {/* Interactive SVG Radar Chart for 5 Angle Coverage */}
            <div className="flex flex-col items-center justify-center py-2">
              <div className="relative h-[200px] w-[200px]">
                <svg viewBox="0 0 200 200" className="h-full w-full overflow-visible">
                  <defs>
                    <radialGradient id="radarGradient" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#10b981" stopOpacity="0.5" />
                      <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.15" />
                    </radialGradient>
                  </defs>

                  {/* Concentric Guide Polygons */}
                  {radarData.levelPolygons.map((poly, idx) => (
                    <polygon
                      key={idx}
                      points={poly}
                      fill="none"
                      stroke="rgba(255, 255, 255, 0.08)"
                      strokeWidth="1"
                    />
                  ))}

                  {/* Spokes from Center */}
                  {radarData.axes.map((ax, idx) => (
                    <line
                      key={idx}
                      x1={radarData.cx}
                      y1={radarData.cy}
                      x2={ax.x}
                      y2={ax.y}
                      stroke="rgba(255, 255, 255, 0.12)"
                      strokeWidth="1"
                      strokeDasharray="2,2"
                    />
                  ))}

                  {/* Active Coverage Polygon */}
                  {captures.length > 0 && (
                    <polygon
                      points={radarData.valuePolygonString}
                      fill="url(#radarGradient)"
                      stroke="#10b981"
                      strokeWidth="2"
                      strokeLinejoin="round"
                    />
                  )}

                  {/* Axis Marker Dots and Labels */}
                  {radarData.axes.map((ax, idx) => {
                    const isCovered = robustness.coverage[ax.angleKey].covered;
                    const count = robustness.coverage[ax.angleKey].count;

                    return (
                      <g key={idx}>
                        {/* Outer Vertex Node */}
                        <circle
                          cx={radarData.valuePoints[idx]?.x ?? ax.x}
                          cy={radarData.valuePoints[idx]?.y ?? ax.y}
                          r={isCovered ? 4.5 : 3}
                          fill={isCovered ? POSE_DEFINITIONS[ax.angleKey].color : '#404040'}
                          stroke="#0a0a0a"
                          strokeWidth="1.5"
                        />

                        {/* Text Label */}
                        <text
                          x={ax.x + (ax.x - radarData.cx) * 0.22}
                          y={ax.y + (ax.y - radarData.cy) * 0.22 + 4}
                          textAnchor="middle"
                          fill={isCovered ? '#ffffff' : 'rgba(255,255,255,0.4)'}
                          fontSize="9"
                          fontWeight={isCovered ? '700' : '500'}
                          fontFamily="monospace"
                        >
                          {ax.label}
                          {count > 0 && ` (${count})`}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>

            {/* Multi-Segment Angle Progress Bar */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[11px] font-semibold text-white/50">Angle Coverage Breakdown</span>
              <div className="grid grid-cols-5 gap-1">
                {CANONICAL_ANGLES.map((ang) => {
                  const cov = robustness.coverage[ang];
                  const def = POSE_DEFINITIONS[ang];
                  return (
                    <div
                      key={ang}
                      className={`flex flex-col items-center rounded-lg p-1 text-center transition-all ${
                        cov.covered
                          ? 'border border-white/20 bg-white/10 text-white'
                          : 'border border-dashed border-white/10 bg-white/2 text-white/30'
                      }`}
                      title={`${def.label}: ${cov.covered ? `${cov.count} capture(s)` : 'Missing'}`}
                    >
                      <span className="text-[10px] font-mono font-bold truncate">{def.shortLabel}</span>
                      <span className="text-[9px]">{cov.covered ? '✓' : '—'}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Recommendations List */}
            {robustness.missingAngles.length > 0 && (
              <div className="mt-1 flex flex-col gap-1.5 rounded-xl border border-white/5 bg-white/5 p-3 text-[11px] text-white/70">
                <span className="font-semibold text-amber-300">Coverage Recommendations:</span>
                <ul className="list-disc pl-4 space-y-1 text-white/60">
                  {robustness.missingAngles.includes('left45') && (
                    <li>Add a <strong>Left 45°</strong> capture to avoid yaw dropouts when subject turns left.</li>
                  )}
                  {robustness.missingAngles.includes('right45') && (
                    <li>Add a <strong>Right 45°</strong> capture to avoid yaw dropouts when subject turns right.</li>
                  )}
                  {(robustness.missingAngles.includes('up') || robustness.missingAngles.includes('down')) && (
                    <li>Add <strong>Up/Down</strong> pitch captures for tilt & high-angle cameras.</li>
                  )}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* ── Status Banner for Upload & Diagnostics ──────────────────────── */}
        {uploadPhase !== 'idle' && (
          <div
            className={`flex items-center justify-between border-t px-6 py-3 text-xs font-medium transition-all ${
              uploadPhase === 'uploading' || uploadPhase === 'processing'
                ? 'border-amber-500/20 bg-amber-500/10 text-amber-300'
                : uploadPhase === 'success'
                ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
                : 'border-rose-500/20 bg-rose-500/10 text-rose-300'
            }`}
          >
            <div className="flex items-center gap-2">
              {(uploadPhase === 'uploading' || uploadPhase === 'processing') && (
                <Loader2 className="h-4 w-4 animate-spin shrink-0" />
              )}
              {uploadPhase === 'success' && <Check className="h-4 w-4 shrink-0 text-emerald-400" />}
              {uploadPhase === 'error' && <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />}
              <span>{uploadStatusText}</span>
              {uploadPhase === 'success' && centroidResult && (
                <span className="ml-2 rounded-md bg-emerald-500/20 px-2 py-0.5 font-mono text-[10px] text-emerald-300 border border-emerald-500/30">
                  ID: {centroidResult.centroid_id || identityName} &bull; {centroidResult.embedding_dim || 512}-D Centroid
                </span>
              )}
            </div>

            {/* Numerical Progress if Uploading */}
            {uploadPhase === 'uploading' && (
              <span className="font-mono font-bold text-amber-200">{uploadProgress}%</span>
            )}
          </div>
        )}

        {/* ── Bottom Action Toolbar ────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/10 bg-neutral-900/60 px-6 py-4">
          <div className="flex items-center gap-2 text-xs text-white/50">
            <Info className="h-3.5 w-3.5" />
            <span>3 to 8 captures required for normalized weighted centroid vectorization.</span>
          </div>

          <div className="flex items-center gap-3">
            {captures.length > 0 && (
              <button
                type="button"
                onClick={handleClearAll}
                disabled={uploadPhase === 'uploading' || uploadPhase === 'processing'}
                className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-white/80 transition-all hover:bg-red-500/15 hover:border-red-500/40 hover:text-red-300 disabled:opacity-40"
              >
                Clear All
              </button>
            )}

            <button
              type="button"
              onClick={submitCentroid}
              disabled={captures.length === 0 || uploadPhase === 'uploading' || uploadPhase === 'processing'}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 px-5 py-2 text-xs font-bold text-white shadow-lg shadow-emerald-950/40 transition-all hover:from-emerald-400 hover:to-teal-500 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
            >
              {uploadPhase === 'uploading' || uploadPhase === 'processing' ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Computing Centroid...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Confirm & Generate Centroid</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }
);

MultiFaceBank.displayName = 'MultiFaceBank';
export default MultiFaceBank;
