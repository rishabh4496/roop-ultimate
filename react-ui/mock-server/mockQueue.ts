// ┌──────────────────────────────────────────────────────────────────────────┐
// │  MOCK. This simulates app/routes_queue.py for UI development only.       │
// │  Nothing here renders, validates a selection against a real person bank,  │
// │  or proves anything about the real pipeline. Where the real queue makes   │
// │  a guarantee (dispatch-time invalidation, per-job outputs, join by concat  │
// │  demuxer) this file only IMITATES its observable HTTP shape so the React   │
// │  queue panel and Batch Matrix can be exercised without a GPU. When the    │
// │  mock and the real backend disagree, the real backend is right; see       │
// │  docs/development/BATCH_MATRIX_DATA_FLOW.md for the actual contract.       │
// └──────────────────────────────────────────────────────────────────────────┘
//
// Shape mirrored from app/routes_queue.py (2026-09-22):
//   POST /api/queue/add        one job          -> snapshot
//   POST /api/queue/add_batch  { jobs: [...] }  -> snapshot | 400 (whole batch)
//   GET  /api/queue                             -> snapshot
//   POST /api/queue/{start,pause,resume,stop,cancel,remove,clear,reorder,
//                    update,duplicate,retry,join}
//   snapshot = { schema_version: 2, job_states, jobs, running, paused, current }
//
// A job is ONE target x ONE person->faceset mapping. The Batch Matrix never
// sends a target list: every strategy ("grouped", "recipe matrix", ...) is N
// single-target jobs in one add_batch call, so this mock accepts exactly that
// and drives each job's progress independently.
import type { Express, Request, Response } from 'express';
import { randomBytes } from 'crypto';

export const JOB_STATES = [
  'QUEUED', 'PREPARING', 'PROCESSING', 'PAUSE_REQUESTED', 'PAUSED', 'COMPLETED',
  'FAILED', 'CANCELLED', 'INTERRUPTED', 'RECOVERABLE',
] as const;
export type JobState = typeof JOB_STATES[number];

const LEGACY_STATUS: Record<JobState, string> = {
  QUEUED: 'pending', PREPARING: 'running', PROCESSING: 'running',
  PAUSE_REQUESTED: 'running', PAUSED: 'running', COMPLETED: 'finished',
  FAILED: 'failed', CANCELLED: 'stopped', INTERRUPTED: 'stopped', RECOVERABLE: 'pending',
};
const RUNNING_STATES: JobState[] = ['PREPARING', 'PROCESSING', 'PAUSE_REQUESTED', 'PAUSED'];
const RETRYABLE_STATES: JobState[] = ['FAILED', 'CANCELLED', 'INTERRUPTED'];

export interface JobProgress {
  fraction: number;
  frames_done: number | null;
  frames_total: number | null;
  fps: number | null;
  eta_s: number | null;
  phase: string;
  updated_at: number;
}

export interface MockJob {
  schema_version: 2;
  id: string;
  target_name: string;
  target_media_id: string;
  source_index: number;
  source_name: string;
  source_id: string;
  payload: Record<string, any>;
  processing_selection: Record<string, any>;
  selection_version: number | null;
  request_id: string;
  frame_start: number | null;
  frame_end: number | null;
  label: string;
  status: string;
  error: string;
  added: number;
  started: number;
  finished: number;
  state: JobState;
  progress: JobProgress;
  outputs: string[];
  cancel_requested: boolean;
  recoverable: boolean;
  project_id: string;
  // Mock-only bookkeeping, stripped from snapshots.
  _sim?: { frame: number; total: number; fps: number };
}

export interface MockTarget { name: string; media_id?: string; frames: number; start_frame?: number; end_frame?: number; fps?: number }
export interface MockSource { id: string; name: string }

export interface MockQueueDeps {
  /** The loaded targets, as /api/state.targets reports them. */
  targets: () => MockTarget[];
  /** The source gallery, as /api/state.source_faces_info reports it. */
  sources: () => MockSource[];
  /** How many people the mock's (single, shared) person bank holds. The real
   *  backend keeps one bank PER target media; the mock has one for all. */
  personCount: () => number;
  /** Simulated render speed for the active hardware profile. */
  fps: () => number;
  /** True while the single-run /api/swap simulation is processing. */
  singleRunActive: () => boolean;
  /** Mirror the running job into the shared /api/progress state, as the real
   *  queue does through api.py's `_progress`. */
  onProgress: (job: MockJob | null, phase: 'start' | 'tick' | 'end') => void;
  /** Record an output file / history row for a finished job; returns the
   *  output name the job lists under `outputs`. */
  onOutput: (job: MockJob) => string;
}

const now = () => Date.now() / 1000;
const newId = () => randomBytes(6).toString('hex');

export function emptyProgress(): JobProgress {
  return { fraction: 0, frames_done: null, frames_total: null, fps: null, eta_s: null,
    phase: 'QUEUED', updated_at: now() };
}

// routes_queue._validate_job_payload, message for message: the whole batch is
// rejected on the first bad entry and nothing is queued.
export function validateJob(payload: any): string | null {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return 'job must be an object';
  if (payload.payload != null && (typeof payload.payload !== 'object' || Array.isArray(payload.payload))) {
    return 'job payload must be an object';
  }
  const sourceIndex = Number(payload.source_index ?? 0);
  if (!Number.isInteger(sourceIndex) || sourceIndex < 0) return 'source_index must be a non-negative integer';
  const starts = payload.frame_start == null ? null : Number(payload.frame_start);
  const ends = payload.frame_end == null ? null : Number(payload.frame_end);
  if ((starts !== null && !Number.isInteger(starts)) || (ends !== null && !Number.isInteger(ends))) {
    return 'frame_start and frame_end must be integers';
  }
  if (starts !== null && starts < 0) return 'frame_start must be non-negative';
  if (ends !== null && ends < 0) return 'frame_end must be non-negative';
  if (starts !== null && ends !== null && ends < starts) return 'frame_end must not be before frame_start';
  return null;
}

// routes_queue._job_selection, reduced to what the mock can honour. Ranks stay
// ranks: the real backend converts them to stable person ids at dispatch
// against the job's OWN target; the mock has no per-target bank to do that.
function freezeSelection(job: any): Record<string, any> {
  const body = job.payload && typeof job.payload === 'object' ? job.payload : {};
  const top = job.processing_selection && typeof job.processing_selection === 'object'
    ? job.processing_selection : null;
  if (top) return { ...top, target_media_id: job.target_media_id || top.target_media_id || null };
  const state = body.selection_state && typeof body.selection_state === 'object' ? body.selection_state : {};
  const mode = state.selection_mode;
  const personId = mode === 'selected' && state.person_id != null ? String(state.person_id) : null;
  let personIds: string[] = Array.isArray(state.person_ids) ? state.person_ids.map(String) : [];
  if (personId && !personIds.includes(personId)) personIds = [personId, ...personIds];
  if (mode === 'selected' && personId) personIds = [personId];
  return {
    schema: 1,
    request_id: body.request_id || newId(),
    selection_version: body.selection_version ?? null,
    mapping_version: null,
    target_media_id: job.target_media_id || null,
    target_person_id: personId,
    target_person_ids: personIds,
    target_reference_face_id: null,
    source_identity_id: body.selected_source_id || job.source_id || null,
    detection_mode: body.detection || null,
    // BatchSwap addresses people by rank, so this is {} for its jobs (F2 in
    // the data-flow doc) -- the mock reproduces that, not fixes it.
    target_person_source_mapping: body.target_person_source_mapping || {},
    selection_state: { ...state },
  };
}

export function normalizeJob(payload: any): MockJob {
  const selection = freezeSelection(payload);
  return {
    schema_version: 2,
    id: newId(),
    target_name: String(payload.target_name || ''),
    target_media_id: String(payload.target_media_id || selection.target_media_id || ''),
    source_index: Number(payload.source_index || 0),
    source_name: String(payload.source_name || ''),
    source_id: String(payload.source_id || selection.source_identity_id || ''),
    payload: payload.payload || {},
    processing_selection: selection,
    selection_version: selection.selection_version ?? null,
    request_id: selection.request_id,
    frame_start: payload.frame_start ?? null,
    frame_end: payload.frame_end ?? null,
    label: String(payload.label || ''),
    status: 'pending',
    error: '',
    added: now(),
    started: 0,
    finished: 0,
    state: 'QUEUED',
    progress: emptyProgress(),
    outputs: [],
    cancel_requested: false,
    recoverable: false,
    project_id: String(payload.project_id || ''),
  };
}

export class MockQueue {
  jobs: MockJob[] = [];
  running = false;
  paused = false;
  current: string | null = null;
  private timer: NodeJS.Timeout | null = null;
  private pauseAcknowledged = false;

  constructor(private deps: MockQueueDeps, private tickMs = 100) {}

  // ── snapshot ────────────────────────────────────────────────────────────
  snapshot() {
    const jobs = this.jobs.map((original, i) => {
      const { _sim, ...job } = original;
      void _sim;
      return { ...job, status: LEGACY_STATUS[job.state], position: i + 1,
        progress: { ...job.progress } };
    });
    return {
      schema_version: 2,
      job_states: [...JOB_STATES],
      jobs,
      running: this.running,
      paused: this.paused,
      current: this.current,
      // Not a field of the real snapshot. Present so a screenshot or a saved
      // response can never be mistaken for the real queue's behaviour.
      mock: 'SIMULATED by react-ui/mock-server/mockQueue.ts -- no render happened',
    };
  }

  find(id: string) { return this.jobs.find((j) => j.id === id) || null; }

  private setState(job: MockJob, state: JobState) {
    job.state = state;
    job.status = LEGACY_STATUS[state];
    job.recoverable = state === 'RECOVERABLE';
    job.progress.phase = state;
    job.progress.updated_at = now();
  }

  // ── mutations (routes_queue.py order) ───────────────────────────────────
  addBatch(raw: any): { status: number; body: any } {
    const list = raw && typeof raw === 'object' ? raw.jobs : null;
    if (!Array.isArray(list)) return { status: 400, body: { message: 'jobs must be a list' } };
    for (const item of list) {
      const error = validateJob(item);
      if (error) return { status: 400, body: { message: error } };
    }
    this.jobs.push(...list.map(normalizeJob));
    return { status: 200, body: this.snapshot() };
  }

  add(raw: any) {
    const error = validateJob(raw);
    if (error) return { status: 400, body: { message: error } };
    this.jobs.push(normalizeJob(raw));
    return { status: 200, body: this.snapshot() };
  }

  remove(id: string) {
    if (this.current === id) return { status: 409, body: { message: 'that job is running — stop it first' } };
    this.jobs = this.jobs.filter((j) => j.id !== id);
    return { status: 200, body: this.snapshot() };
  }

  clear(keepRunning = true) {
    if (this.current && keepRunning) {
      this.jobs = this.jobs.filter((j) => j.id === this.current);
    } else {
      this.jobs = [];
      this.running = false;
    }
    return { status: 200, body: this.snapshot() };
  }

  reorder(ids: string[]) {
    const byId = new Map(this.jobs.map((j) => [j.id, j]));
    const ordered: MockJob[] = [];
    for (const id of ids.map(String)) {
      const job = byId.get(id);
      if (job) { ordered.push(job); byId.delete(id); }
    }
    ordered.push(...this.jobs.filter((j) => byId.has(j.id)));
    this.jobs = ordered;
    return { status: 200, body: this.snapshot() };
  }

  update(patch: any) {
    const job = this.find(String(patch.id || ''));
    if (!job) return { status: 404, body: { message: 'no such job' } };
    if (RUNNING_STATES.includes(job.state)) return { status: 409, body: { message: 'job is running' } };
    for (const key of ['payload', 'target_name', 'target_media_id', 'source_index', 'source_name',
      'source_id', 'label', 'frame_start', 'frame_end'] as const) {
      if (key in patch) (job as any)[key] = patch[key];
    }
    if ('payload' in patch || 'processing_selection' in patch) {
      job.processing_selection = freezeSelection({ ...job, ...patch });
      job.request_id = job.processing_selection.request_id;
    }
    if (patch.requeue || ['COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(job.state)) {
      this.setState(job, 'QUEUED');
      job.error = '';
      job.started = job.finished = 0;
      job.progress = emptyProgress();
      job.cancel_requested = false;
    }
    return { status: 200, body: this.snapshot() };
  }

  duplicate(id: string) {
    const job = this.find(id);
    if (!job) return { status: 404, body: { message: 'no such job' } };
    const clone: MockJob = { ...job, id: newId(), state: 'QUEUED', status: 'pending', error: '',
      added: now(), started: 0, finished: 0, payload: { ...job.payload },
      progress: emptyProgress(), outputs: [], cancel_requested: false, recoverable: false };
    delete clone._sim;
    this.jobs.splice(this.jobs.indexOf(job) + 1, 0, clone);
    return { status: 200, body: this.snapshot() };
  }

  retry(id?: string) {
    const targets = id ? [this.find(String(id))] : this.jobs.filter((j) => RETRYABLE_STATES.includes(j.state));
    for (const job of targets) {
      if (job && !RUNNING_STATES.includes(job.state)) {
        this.setState(job, 'QUEUED');
        Object.assign(job, { error: '', started: 0, finished: 0, progress: emptyProgress(), cancel_requested: false });
      }
    }
    return { status: 200, body: this.snapshot() };
  }

  // ── runner ──────────────────────────────────────────────────────────────
  start() {
    if (this.running) return { status: 200, body: this.snapshot() };
    if (!this.jobs.some((j) => j.state === 'QUEUED' || j.state === 'RECOVERABLE')) {
      return { status: 400, body: { message: 'nothing left to run — retry or add a job' } };
    }
    if (this.deps.singleRunActive()) {
      return { status: 409, body: { message: 'a single run is already processing' } };
    }
    this.running = true;
    this.paused = false;
    this.scheduleTick();
    return { status: 200, body: this.snapshot() };
  }

  pause() {
    this.paused = true;
    const job = this.current ? this.find(this.current) : null;
    if (job && job.state === 'PROCESSING') {
      this.setState(job, 'PAUSE_REQUESTED');
      this.pauseAcknowledged = false;
    }
    return { status: 200, body: this.snapshot() };
  }

  resume() {
    this.paused = false;
    this.pauseAcknowledged = false;
    const job = this.current ? this.find(this.current) : null;
    if (job && (job.state === 'PAUSED' || job.state === 'PAUSE_REQUESTED')) this.setState(job, 'PROCESSING');
    if (this.running) this.scheduleTick();
    return { status: 200, body: this.snapshot() };
  }

  cancel(id: string) {
    if (!id) return { status: 400, body: { message: 'job id is required' } };
    const job = this.find(id);
    if (!job) return { status: 404, body: { message: 'no such job' } };
    if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(job.state)) {
      return { status: 409, body: { message: `job is already ${job.state.toLowerCase()}` } };
    }
    job.cancel_requested = true;
    if (this.current !== id) {
      this.setState(job, 'CANCELLED');
      job.error = 'cancelled before processing';
      job.finished = now();
    }
    return { status: 200, body: this.snapshot() };
  }

  stop() {
    this.running = false;
    this.paused = false;
    const job = this.current ? this.find(this.current) : null;
    if (job) job.cancel_requested = true;
    return { status: 200, body: this.snapshot() };
  }

  // The real join stream-copies the segments' files through ffmpeg's concat
  // demuxer and refuses mixed formats; the mock only checks the counts and
  // invents a joined output name.
  join(ids: string[], name?: string) {
    const jobs = ids.map((id) => this.find(String(id))).filter((j): j is MockJob => !!j);
    const outputs = jobs.flatMap((j) => j.outputs);
    if (outputs.length < 2) return { status: 400, body: { message: 'need at least two finished segments to join' } };
    const exts = new Set(outputs.map((o) => o.split('.').pop()?.toLowerCase()));
    if (exts.size > 1) return { status: 400, body: { message: 'segments have different formats; render them alike first' } };
    const joined = name || `joined_${newId()}.${[...exts][0]}`;
    return { status: 200, body: { ...this.snapshot(), joined, output: joined } };
  }

  // ── the simulated dispatch: routes_queue._run_one, without a render ───────
  private scheduleTick() {
    if (this.timer) return;
    this.timer = setInterval(() => this.tick(), this.tickMs);
  }

  private stopTicking() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private tick() {
    if (!this.running) { this.stopTicking(); this.finishCurrent('CANCELLED', 'cancelled by user'); return; }
    let job = this.current ? this.find(this.current) : null;
    if (job && this.paused) {
      // Real: PAUSED only after the writer reports output is safe; here one tick later.
      if (job.state === 'PAUSE_REQUESTED' && !this.pauseAcknowledged) {
        this.pauseAcknowledged = true;
        this.setState(job, 'PAUSED');
        this.deps.onProgress(job, 'tick');
      }
      return;
    }
    if (!job) {
      if (this.paused) return;
      job = this.jobs.find((j) => j.state === 'QUEUED' || j.state === 'RECOVERABLE') || null;
      if (!job) {
        this.running = false;
        this.current = null;
        this.stopTicking();
        this.deps.onProgress(null, 'end');
        return;
      }
      this.dispatch(job);
      return;
    }
    if (job.cancel_requested) { this.finishCurrent('CANCELLED', 'cancelled by user'); return; }
    // Advance this job's OWN counter; other jobs keep their own progress.
    const sim = job._sim!;
    sim.frame = Math.min(sim.total, sim.frame + Math.max(1, Math.round(sim.fps / 10)));
    job.progress = {
      fraction: Number((sim.frame / sim.total).toFixed(3)),
      frames_done: sim.frame, frames_total: sim.total, fps: sim.fps,
      eta_s: Math.max(0, Math.round((sim.total - sim.frame) / sim.fps)),
      phase: 'PROCESSING', updated_at: now(),
    };
    this.deps.onProgress(job, 'tick');
    if (sim.frame >= sim.total) this.finishCurrent('COMPLETED', '');
  }

  private dispatch(job: MockJob) {
    this.setState(job, 'PREPARING');
    job.started = now();
    job.error = '';
    job.cancel_requested = false;
    this.current = job.id;

    // 1. Target by media id, else by basename (the Batch Matrix sends no
    //    media id, so it is always the basename path -- F1 in the doc).
    const targets = this.deps.targets();
    let target: MockTarget | undefined;
    if (job.target_media_id) {
      target = targets.find((t) => t.media_id === job.target_media_id);
      if (!target) return this.finishCurrent('FAILED', `target media "${job.target_media_id}" is no longer loaded`, true);
    } else {
      const base = job.target_name.split(/[\\/]/).pop();
      const matches = targets.filter((t) => t.name.split(/[\\/]/).pop() === base);
      if (matches.length === 0) return this.finishCurrent('FAILED', `target "${job.target_name}" is no longer loaded`, true);
      if (matches.length > 1) return this.finishCurrent('FAILED', `legacy target "${job.target_name}" is ambiguous; requeue it`, true);
      target = matches[0];
    }

    // 2. Selection invalidation. Real: ranks -> that target's stable person
    //    ids, every id compared with what is loaded NOW. Mock: one shared
    //    person bank, and only the primary faceset (source_identity_id) is
    //    checked -- exactly the coverage the real dispatch has (F2).
    const reasons: string[] = [];
    const sel = job.processing_selection || {};
    const people = this.deps.personCount();
    for (const rank of sel.target_person_ids || []) {
      const n = Number(rank);
      if (Number.isInteger(n) && n >= people) reasons.push(`target person ${rank} was removed`);
    }
    const sources = this.deps.sources();
    const wanted = String(sel.source_identity_id || job.source_id || '');
    if (wanted && !sources.some((s) => s.id.toLowerCase() === wanted.toLowerCase())) {
      reasons.push(`source ${wanted} was removed`);
    }
    if (reasons.length) return this.finishCurrent('FAILED', 'selection invalidated: ' + reasons.join('; '), true);

    // 3. Re-resolve the primary faceset by id (then name) -- the stored
    //    numeric index is only the fallback.
    let sourceIndex = job.source_index;
    const byId = sources.findIndex((s) => s.id.toLowerCase() === wanted.toLowerCase());
    if (byId >= 0) sourceIndex = byId;
    else if (job.source_name) {
      const byName = sources.findIndex((s) => s.name.toLowerCase() === job.source_name.toLowerCase());
      if (byName >= 0) sourceIndex = byName;
    }
    job.payload = { ...job.payload, target_index: targets.indexOf(target), source_index: sourceIndex,
      target_media_id: target.media_id || '' };

    // 4. The segment, clamped to the clip, sizes this job's simulation.
    const total = Math.max(1, target.frames || 1);
    const fs = job.frame_start == null ? 0 : Math.max(0, Number(job.frame_start));
    const fe = job.frame_end == null ? total : Math.min(total, Number(job.frame_end));
    job._sim = { frame: 0, total: Math.max(1, fe - fs), fps: this.deps.fps() };
    job.progress = { ...emptyProgress(), frames_done: 0, frames_total: job._sim.total, fps: job._sim.fps, phase: 'PROCESSING' };
    this.setState(job, 'PROCESSING');
    this.deps.onProgress(job, 'start');
  }

  private finishCurrent(state: JobState, error: string, beforeRender = false) {
    const job = this.current ? this.find(this.current) : null;
    this.current = null;
    if (!job) return;
    this.setState(job, state);
    job.error = error;
    job.finished = now();
    if (state === 'COMPLETED') {
      job.progress.fraction = 1;
      job.outputs = [this.deps.onOutput(job)];
    } else if (beforeRender) {
      // A pre-dispatch failure keeps its own empty snapshot, never the
      // previous job's numbers.
      job.progress = { ...emptyProgress(), phase: state };
    }
    job.cancel_requested = false;
    delete job._sim;
    this.deps.onProgress(job, 'end');
  }
}

// Registers every /api/queue/* route on the express app.
export function installMockQueue(app: Express, deps: MockQueueDeps): MockQueue {
  const queue = new MockQueue(deps);
  const send = (res: Response, result: { status: number; body: any }) => res.status(result.status).json(result.body);
  const body = (req: Request) => (req.body && typeof req.body === 'object' ? req.body : {});

  app.get('/api/queue', (_req, res) => res.json(queue.snapshot()));
  app.post('/api/queue/add', (req, res) => send(res, queue.add(body(req))));
  app.post('/api/queue/add_batch', (req, res) => send(res, queue.addBatch(body(req))));
  app.post('/api/queue/remove', (req, res) => send(res, queue.remove(String(body(req).id || ''))));
  app.post('/api/queue/clear', (req, res) => send(res, queue.clear(body(req).keep_running ?? true)));
  app.post('/api/queue/reorder', (req, res) => send(res, queue.reorder(body(req).ids || [])));
  app.post('/api/queue/update', (req, res) => send(res, queue.update(body(req))));
  app.post('/api/queue/duplicate', (req, res) => send(res, queue.duplicate(String(body(req).id || ''))));
  app.post('/api/queue/retry', (req, res) => send(res, queue.retry(body(req).id)));
  app.post('/api/queue/start', (_req, res) => send(res, queue.start()));
  app.post('/api/queue/pause', (_req, res) => send(res, queue.pause()));
  app.post('/api/queue/resume', (_req, res) => send(res, queue.resume()));
  app.post('/api/queue/cancel', (req, res) => send(res, queue.cancel(String(body(req).id || ''))));
  app.post('/api/queue/stop', (_req, res) => send(res, queue.stop()));
  app.post('/api/queue/join', (req, res) => send(res, queue.join(body(req).ids || [], body(req).name)));
  return queue;
}
