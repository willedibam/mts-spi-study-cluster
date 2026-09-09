# Boundary and observation audit, 2026-09-10

Latest approved follow-up: see [candidate selection and first experiment](candidate-selection-260910.md).
Vicsek is now the first new benchmark pilot; inspect its local outputs and live
processes before starting further work. Preserve the original resource/deadline
ceiling below. Short physics/observation runs are not a p90 or coexistence pass.
The first 24 Vicsek runs and analyses are complete; see [results](vicsek-pilot-results-260910.md).
The four-case long physics gate is now submitted as178580950.gadi-pbs. Do not
repeat the completed smoke/size/time scouts or submit p90 before resolving truth.

## Current execution (heartbeat follow-up)

- Committed/pushed only the new Vicsek source/tests/configs/report and scheduler
  runner in `00096b4fb787d8c1d0d2002c3ad86161fba965f6`; unrelated dirty work preserved.
  Gadi fast-forwarded its clean branch; pyspi is `65317c9` (not used by this scout).
- Submitted **178580558.gadi-pbs**: two-case Linux/scheduler smoke, 2 CPUs, 8GB,
  15min maximum = at most1 SU. Immutable source snapshot from the pinned commit;
  no running code dependency on later branch edits. Nine current local tests pass.
  Output `/scratch/ql44/we2614/mts-spi-data/order_parameter/vicsek_observation_260910/gadi-smoke-00096b4/`.
  Inspect qstat/PBS log, nine tests, two archives, and accounting before long gate.
- Four long cases remain NOT submitted. Planned maximum4 CPUs/16GB/2h =16 SU.
  Combined authorized-stage maximum17 SU, still inside the existing1 KSU ceiling.
- Started the remaining streaming Stuart–Landau temporal cases locally with
  `scripts/run_stuart_landau_temporal_scout.py`; it verifies/reuses the existing
  matching N32/gamma.8 archive. Inspect processes and archives before rerunning.
  This does not extract SPIs, and two initial seeds are not a confirmation bank.
- Smoke178580558 ended exit1 after all nine tests passed: direct-file launch
  could not import `scripts` while loading the pytest-populated Numba cache in
  jobfs. No case archives produced. Cost .04 SU. Fixed launcher to module mode;
  a repeat smoke is required before scaling. Retain failed logs as provenance.
- SL baseline batch now complete:28 archives (27 new +1 verified reused).
  Additional five half-dt/long-burn cases completed in convergence/. Mean/SD
  sensitivity analysis follows; these checks do not diagnose chaos from spectra.
- Repeat smoke **178580749.gadi-pbs** submitted from fixed runner commit
  `90543f1fc8f8b9e61298d122d10ea0462d9fdfe6`, output gadi-smoke-90543f1/ under
  the same Vicsek remote base. Expected source/config hashes remain unchanged;
  only the launcher changed. Long gate remains blocked on this smoke outcome.
  Spent .04 SU; repeat maximum1 SU plus long maximum16 SU leaves ample budget.
- Repeat178580749 **passed**: nine tests, two finite2000-sample archives, exit0,
  wall19s, memory580MB, charge .02 SU. Both successful archives and failed/success
  PBS/case logs were collected locally through gadi-dm. Total actual spent .06 SU.
- **Long gate178580950.gadi-pbs submitted**, four CPUs,16GB,2h (maximum16 SU),
  fixed commit90543f1; output remote base + `gadi-long-90543f1/`.
  Four N32768 cases: eta=.44/.476, ordered/random, seed910203, burn20000 and
  record100000 microscopic steps, saved every25 steps (4000 samples).
  Next heartbeat: inspect job status/logs, collect complete archives through
  gadi-dm, verify count/hash/finite data, then analyze separately from stride1
  short pilots. Do not resubmit or fit SPI coordinates while this is running.
- [Stuart–Landau audit](stuart-landau-temporal-results-260910.md) complete:
  28 baseline+5 sensitivities. Half-dt mean/SD changes <=1.1e-7 in the two tested
  cells; gamma1.2/N800 remains burn-sensitive. Source/report committed87f6bb9.
  Eleven combined current local tests pass. No new p90 has been computed.

## Objective and authorization

The user authorized overnight literature investigation and focused simulation runs.
Seek 2–3 recognizable systems beyond the Kuramoto control with established physical
order parameters and reproducible changes over a published local boundary. Success
means sensitivity/recovery in a common SPI–SPI representation across M and T;
outperformance of purpose-built statistics is not required. Observed M stays <=32;
physical N may grow to 10^6 when size/time checks warrant it. Do not assume success.

## Sources and priority

- Stuart–Landau: Matthews–Strogatz, PRL 65, 1701 (1990),
  https://doi.org/10.1103/PhysRevLett.65.1701. K=.8, uniform frequencies;
  locking to large collective oscillations, then more complex collective motion.
  Retain Z(t), |Z(t)|, spectra and block variability. Mean |Z| alone is insufficient.
- Miller–Huse: https://doi.org/10.1103/PhysRevE.48.2528 and refined study
  https://doi.org/10.1103/PhysRevE.55.2606. Sign magnetization and symmetry breaking.
- Desai–Zwanzig: https://doi.org/10.1103/PhysRevResearch.5.013078 (Zagli et al.,
  not Evangelou et al.) and https://doi.org/10.1103/PhysRevE.110.014121.
  First moment; investigate convergence concerns before any million-particle run.
- Exact Kaneko ring: https://doi.org/10.1103/PhysRevE.87.052905. Verify the
  persistence definition, mapped parameter conventions, and observation horizon
  before simulating a published persistence boundary. No generic scalar across
  the whole Kaneko regime diagram has been established here.

Independent literature checks determine targets; existing repository evidence only
prevents duplicate work and supplies implementation/provenance. Previous assistant
claims about a universally optimal sensor layout or guaranteed M=32 performance
are hypotheses, not established results. Preserve the prior Ising implementation
exclusion unless the user explicitly revises it.

## First concrete run

Use scripts/scout_stuart_landau_streaming.py, which retains O(N + 32T) state,
unlike the existing generator's internal O(NT) allocation. Verify short traces
against the existing generator before production. Pilot N=32 and N=800, K=.8,
gamma=.70,.74,.75,.80,.90,1.0,1.2; two seeds; 200 burn units and 8000 samples
at sample_dt=.1, dt=.02. This is 28 exploratory rows. Compare consecutive blocks
and rerun selected boundary rows with half dt and longer burn if warranted.
Use the full complex order trace in the rotating frame for collective spectra;
the explicit laboratory carrier affects real-channel SPIs and must be retained
as an observation choice. Do not declare chaos from spectral entropy alone.

## Iteration and stopping

1. Inspect existing jobs/artifacts first, including other active workstreams.
2. Verify exact equations, physical targets and literature boundary provenance.
3. Run bounded physics pilots, checking timestep, burn, block stability and seeds.
4. For lattices, compare equal-M contiguous, dispersed and dispersed-block views;
   none is presumed optimal. Mean-field views use fixed random particle indices.
5. Use M=8,16,32 nested views and temporal windows long enough for the dynamics.
   Compute full p90 only after a physical contrast and valid observations exist.
6. Freeze reduction without controls/Q; keep independent seeds for evaluation.
   Report target recovery, transition location, and validity; distinguish a
   time-window regime coordinate from reconstruction of an instantaneous Q(t).
7. Increment N only to resolve a stated question; do not multiply an entire bank
   by the million-particle size. First bound total work and memory with a pilot.

Overnight resource ceiling chosen conservatively: at most 1 KSU of new Gadi work
in this pass, with small smoke jobs before any farm. Stop new submissions after
10 hours from 2026-09-09 18:51 UTC; finish collecting already running bounded jobs.
Produce a concise cited findings report with run IDs, costs, plots, failures and
the next justified step. Pause the follow-up once complete or requiring new input.

## Initial operational evidence

Gadi reachable at 2026-09-09 18:51 UTC; qstat showed no jobs for this user.
nci_account: 87.34 KSU available; Scratch 185.41k/202k inodes, gdata
44.38k/70k. Prefer compact archives. Local tree contains extensive unrelated
changes; preserve them. No production simulation has been submitted yet.

Streaming integrator passed a matched-seed 12-sample comparison against the
existing generator (sensor trace and complex global order, tolerance 1e-12).
The first full local pilot completed: N=32, gamma=.8, 8000 samples, seed=910001;
archive data/order_parameter/stuart_landau_dynamics_260910/local-pilot-N32-gamma0p8.npz.
Mean R=.2211, SD R=.1580. Eight consecutive block means drift from .2262 to
.2170; investigate time convergence rather than treating this pilot as truth.
