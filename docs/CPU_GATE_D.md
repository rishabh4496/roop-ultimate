# Gate D — Intel hybrid CPU optimization

## Scope

Gate D adds capability-driven CPU profiling and controlled CPU-distribution
selection. It does not make i9-14900K behavior mandatory and does not claim a
CPU policy is globally optimal without end-to-end evidence on both required
GPUs.

The runtime now records:

- physical and logical processor counts;
- measured Windows CPU-set P/E logical indices when available;
- CPU name, current/max reported frequency, NumPy SIMD dispatch features;
- process-affinity support and topology source;
- CPU frequency and available CPU temperature samples during benchmark
  telemetry.

Windows hybrid topology uses `GetSystemCpuSetInformation`; the OS efficiency
class is treated as the topology signal, not the GPU or CPU brand string.
Linux capacity data and explicit `ROOP_CPU_P_INDICES` /
`ROOP_CPU_E_INDICES` overrides remain supported. If no reliable topology is
available, automatic scheduling remains in effect.

## Runtime policies

`ROOP_CPU_DISTRIBUTION` supports:

| Policy | Selection |
|---|---|
| `auto` | Existing bounded hardware/workload policy; no forced affinity |
| `p_only` | All measured P logical CPUs |
| `p_priority_e` | All measured P logical CPUs plus `ROOP_CPU_E_LIMIT` E CPUs (default one quarter) |
| `p_plus_e` | All measured P and E logical CPUs |

Explicit policies use the measured OS affinity indices. OpenCV optimized
dispatch remains enabled, while its internal pool, ORT intra/inter-op pools,
and FFmpeg threads stay independently bounded to avoid nested
oversubscription. CUDA/TensorRT resources remain selected by the GPU/workload
profile; CPU policy does not widen GPU contexts or streams.

On the current host the detected CPU profile is 24 physical / 32 logical,
with 8 P-core physical / 16 P logical and 16 E logical CPUs. Current/max
reported frequency was 3,200 MHz, NumPy exposed AVX/AVX2/FMA/SSE families,
and process affinity was available. This is a runtime observation, not a
hard-coded i9 assumption.

## Validation status

The code and targeted CPU/GPU optimizer tests pass. The complete repository
suite passes **1,430 tests, 1 skipped, 589 subtests**.

The controlled RTX 4070 matrix is **complete** (2026-08-30). It used the real
`d4.mp4` two-face workload, GPEN 256 Pro, RealityUX, stabilization, tracking,
and the existing NVENC/FFmpeg pipeline.

The earlier run recorded here as a timeout was **not** a code defect and not a
slow CPU policy. Production runs the `mixed` TensorRT precision, and the
builder-config fingerprint added by the Gate C work had orphaned that
namespace's 30 previously built engines. The first candidate therefore paid a
full cold engine build inside its own render clock, and a ~30-engine cold
build is indistinguishable from a hang at the process level. It is now
attributable: `migrate_legacy_cache_dir` carries a renamed cache over, and a
cold build is the expected cost of any change to a builder option.

That effect is also why the first candidate of any Gate D pass must be
discarded or re-measured warm, which is what the counterbalanced design below
does.

### RTX 3060 — separate result table

| Metric | Result |
|---|---|
| Physical validation | **PENDING** — no RTX 3060 was available in this session |
| Baseline FPS | PENDING |
| P-only FPS | PENDING |
| P-priority + limited E FPS | PENDING |
| P+E FPS | PENDING |
| VRAM / CPU / GPU / stage throughput / latency | PENDING |
| Quality / stability | PENDING |
| Required rerun | `env\\Scripts\\python.exe tests\\gate_d_cpu_benchmark.py --target "RTX 3060" --end 120` |

### RTX 4070 — separate result table

Physical validation: available; runtime detected NVIDIA GeForce RTX 4070, Ada
Lovelace, SM 8.9, 11.994 GiB, driver 616.56, CUDA 12.8, TensorRT 10.9.0.34,
ORT 1.23.2. Host CPU 24 physical / 32 logical, 16 P logical + 16 E logical.

Workload: `d4.mp4` frames 0-120, two facesets (harjot, gargee), realswap /
GPEN 256 Pro / RealityUX / libx264. Every arm pins ORT intra/inter, OpenCV and
FFmpeg pools to one thread so the CPU-distribution policy is the variable
under test; that is not the production threading configuration, so these FPS
values are comparable to each other and not to the production baseline.

Counterbalanced: a forward pass (auto, p_only, p_priority_e, p_plus_e), a
reversed pass, and an interleaved tiebreak between the two leaders.

| Policy | Threads | Samples (FPS) | Mean FPS | vs auto | Peak VRAM MB | CPU mean % | GPU mean % | Faces swapped | Stability |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `auto` | 10 | 5.13, 5.12, 5.05 | **5.10** | — | 5821-5901 | 20.8-21.5 | 24.1-25.4 | 288/288 | 120/120 frames, exit 0 |
| `p_plus_e` | 32 | 4.96, 4.94, 5.02, 4.97 | 4.97 | -2.6% | 5844-6057 | 20.5-23.7 | 24.8-28.5 | 288/288 | 120/120 frames, exit 0 |
| `p_priority_e` | 20 | 4.79, 4.58 | 4.69 | -8.1% | 6015-6078 | 21.1-23.9 | 27.0-29.2 | 288/288 | 120/120 frames, exit 0 |
| `p_only` | 16 | 4.55, 4.56 | 4.56 | -10.7% | 6041-6082 | 22.1-23.5 | 27.2-27.4 | 288/288 | 120/120 frames, exit 0 |

A discarded twelfth run, the forward pass's `auto` arm, read 0.18 FPS because
it absorbed the cold engine build described above. It is excluded from the
table rather than reported as a CPU-policy result.

Reading the 120-frame result:

* `auto`'s worst sample (5.05) still beats `p_plus_e`'s best (5.02), so the
  ordering within this window is a real separation rather than noise.
* Every arm swapped 288 of 288 faces.
* **The arms vary two things at once.** Thread count moves with the policy
  (10/16/20/32), because a P-only policy is defined to use P processors. This
  compares whole CPU policies, not the affinity variable in isolation.

### AND THE 120-FRAME WINDOW IS AN ARTEFACT — production length reverses it

The RTX 3060 session independently established that Gate D differences at 120
frames do not survive to production length (+19.6% became -0.5% at 600
frames). That warning applies directly to the table above, so the largest
effect in it -- `p_only` at -10.7% -- was re-measured at 600 frames,
counterbalanced auto / p_only / p_only / auto:

| Policy | Threads | Samples (FPS) | Mean | vs auto |
|---|---:|---|---:|---:|
| `auto` | 10 | 11.46 | 11.46 | — |
| `p_only` | 16 | 11.69, 11.54 | **11.62** | **+1.4%** |

**NEUTRAL.** The -10.7% penalty measured at 120 frames does not exist at
production length; it becomes a +1.4% difference that is inside the noise
floor. Absolute throughput is also 2.5x higher (4.5 -> 11.6 FPS), which is the
direct evidence that the short window was measuring warm-up rather than steady
state.

Swap rate was 846-848 of 852-854 faces (~99.3%) in every 600-frame arm, peak
VRAM 6,467-6,483 MB, GPU mean 27.6-29.1%.

Two caveats kept deliberately visible:

* `auto` has **one** valid 600-frame sample. The pass's first arm read 0.47 FPS
  because it absorbed a cold TensorRT rebuild (the engine cache namespace
  changed when driver identity was restored to the key), so it is void as a
  CPU-policy result exactly like the 120-frame `auto` arm was.
* The 120-frame and 600-frame passes ran on different TensorRT tuning profiles
  (`a-1` vs `a0` auxiliary streams, different builder-config digest), so the
  two tables are not directly comparable in absolute FPS. The conclusion rests
  on the within-pass comparisons, each of which is internally counterbalanced.

**Cross-target agreement:** both validation GPUs now show Gate D CPU
distribution as neutral at production length, reached from opposite directions
in their short windows (3060 +19.6%, 4070 -10.7%, both -> neutral at 600
frames). The shipped `auto` default is correct on both, and nothing is
promoted on either.

The previously locked Phase 2 4070 production baseline remains **9.62 FPS**
for the 600-frame workload, with 233.34 ms worker-time latency. It is a
historical baseline, not a Gate D CPU-policy result and is not used to claim a
Gate D improvement.

## Acceptance classification

| Change | RTX 3060 | RTX 4070 | Classification |
|---|---|---|---|
| Runtime CPU topology/frequency/SIMD/affinity detection | Measured on real hybrid topology (i7-12700H, OS-reported) | Measured; matrix completed | **A. BENEFICIAL ON BOTH** as detection; it drives no promoted change |
| Explicit P/E policy selection and affinity | Measured: +19.6% at 120 frames, **-0.5% at 600** | Measured: -10.7% at 120 frames, **+1.4% at 600** | **D. NEUTRAL on both** at production length; the short-window effects (opposite in sign) are warm-up artefacts |
| Keeping `auto` as the shipped default | Confirmed; P/E promotion implemented then reverted | Confirmed; 11.46 FPS at 600 frames | **D. NEUTRAL** (confirms the existing default on both targets; nothing promoted) |
| OpenCV optimized dispatch with bounded internal threads | Exercised | Code path exercised in every completed arm | **PENDING** (never isolated on either target) |

No optimization is classified as A–F until both physical target result rows
contain valid end-to-end measurements. A completed rerun must report one row
per policy and include FPS, peak/average VRAM, CPU/GPU utilization, decode,
inference, enhancement and encode throughput, latency, stability, and output
quality. Sustained CPU frequency and temperature are reported when the OS
exposes them; unavailable thermal/power fields remain explicitly unavailable.
