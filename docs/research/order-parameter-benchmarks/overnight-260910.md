# Boundary and observation audit, 2026-09-10

## Final status (supersedes historical execution notes below)

All submitted jobs completed. No new SPIs/p90 were computed. The user-requested overnight reminder `order-parameter-boundary-experiments-overnight` was deleted through the app after reviewing the completed results; it will not repeat. No further submissions or recurring checks are authorized by this ledger.

| Job | Outcome | Walltime | Actual SU |
|---|---|---|---:|
| 178580558 | Launcher failure after nine tests passed; no simulation archives | — | .04 |
| 178580749 | Corrected smoke passed; two archives | 19s | .02 |
| 178580950 | Four long cases, exit0 | 53m39s | 7.15 |
| 178628828 | Sixteen narrow-control cases, exit0 | 56m11s | 29.96 |
| Total | Completed collection/review stage; no p90 | | **37.17** |

The narrow job requested16 CPUs/64GB; measured peak memory1.43GB and aggregate CPU time13h51m08s. Aggregate CPU time is not elapsed walltime. Its apparent duration reflects simulation of N32768 particles for120k microscopic steps per case, followed by a slow/retried file transfer—not SPI extraction at M<=32. See [final narrow results](vicsek-narrow-results-260910.md). Control-average polarization falls .3442→.2269→.1170→.0773 over eta=.460,.468,.476,.484; all seed/start curves decrease. Temporal episodes near the middle motivate a future window-level order-tracking pilot, not a claim of stationary coexistence or thermodynamic-discontinuity confirmation. No additional simulations are needed merely to chase a precise critical point for this proof-of-concept objective.

## Historical execution record (not instructions to resume)

Latest approved follow-up: see [candidate selection and first experiment](candidate-selection-260910.md). Vicsek is now the first new benchmark pilot; inspect its local outputs and live processes before starting further work. Preserve the original resource/deadline ceiling below. Short physics/observation runs are not a p90 or coexistence pass. The first 24 Vicsek runs and analyses are complete; see [results](vicsek-pilot-results-260910.md). The four-case long physics gate178580950.gadi-pbs is now complete (exit0). All four archives were collected locally and verified for count, finite arrays, 4000-sample shape, and source/config hashes. Long-only analysis is in data/order_parameter/vicsek_observation_260910/long-analysis/. At eta=.44 mean polarization agrees between starts (.430053/.429941); at .476 it is .105521/.092820. These are two-control/one-seed results, not an independently confirmed jump. No new p90. Total actual charge7.21 SU.

## Submitted narrow refinement (now completed; see final status)

Submitted **178628828.gadi-pbs** before the04:51 UTC cutoff:16 cases at N32768, eta=.460,.468,.476,.484; seeds910204/910205; ordered/random starts. Same 20k burn+100k microscopic steps, stride25 as the completed long gate.16 CPUs, 64GB,2h gives a64-SU maximum; total actual+maximum exposure71.21 SU. Nine local tests and uniqueness of all16 cases passed before submission. Frozen [protocol](vicsek-narrow-protocol-260910.md); source commit `eade8c2e0bb54b5cd00510289cc84de6a72063f0`, branch `codex/vicsek-narrow-260910`. This isolated commit starts from the published base and changes only the config, protocol and scheduler case-count cap. Two unrelated unpublished main-branch commits were deliberately not pushed; the live Gadi checkout was not switched. Remote output base: `/scratch/ql44/we2614/mts-spi-data/order_parameter/vicsek_observation_260910/gadi-narrow-eade8c2/`.

The planned collection was through gadi-dm, with16-archive source/config, finite-array and shape checks. The04:51 UTC cutoff prohibited new submissions, not completion of already submitted jobs. The reminder is now deleted.

## Earlier execution chronology

Latest outcome: long178580950 completed in53m39s, cost7.15 SU, four CPUs, peak reported memory314.88MB; all nine tests passed. Earlier smoke costs .04+.02 SU. The submission notes below are provenance, not instructions to rerun those jobs. This was followed by the narrow-control/independent-seed check above.

- Committed/pushed only the new Vicsek source/tests/configs/report and scheduler runner in `00096b4fb787d8c1d0d2002c3ad86161fba965f6`; unrelated dirty work preserved. Gadi fast-forwarded its clean branch; pyspi is `65317c9` (not used by this scout).
- Submitted **178580558.gadi-pbs**: two-case Linux/scheduler smoke, 2 CPUs, 8GB, 15min maximum = at most1 SU. Immutable source snapshot from the pinned commit; no running code dependency on later branch edits. Nine current local tests pass. Output `/scratch/ql44/we2614/mts-spi-data/order_parameter/vicsek_observation_260910/gadi-smoke-00096b4/`. Inspect qstat/PBS log, nine tests, two archives, and accounting before long gate.
- Four long cases remain NOT submitted. Planned maximum4 CPUs/16GB/2h =16 SU. Combined authorized-stage maximum17 SU, still inside the existing1 KSU ceiling.
- Started the remaining streaming Stuart–Landau temporal cases locally with `scripts/run_stuart_landau_temporal_scout.py`; it verifies/reuses the existing matching N32/gamma.8 archive. Inspect processes and archives before rerunning. This does not extract SPIs, and two initial seeds are not a confirmation bank.
- Smoke178580558 ended exit1 after all nine tests passed: direct-file launch could not import `scripts` while loading the pytest-populated Numba cache in jobfs. No case archives produced. Cost .04 SU. Fixed launcher to module mode; a repeat smoke is required before scaling. Retain failed logs as provenance.
- SL baseline batch now complete:28 archives (27 new +1 verified reused). Additional five half-dt/long-burn cases completed in convergence/. Mean/SD sensitivity analysis follows; these checks do not diagnose chaos from spectra.
- Repeat smoke **178580749.gadi-pbs** submitted from fixed runner commit `90543f1fc8f8b9e61298d122d10ea0462d9fdfe6`, output gadi-smoke-90543f1/ under the same Vicsek remote base. Expected source/config hashes remain unchanged; only the launcher changed. Long gate remains blocked on this smoke outcome. Spent .04 SU; repeat maximum1 SU plus long maximum16 SU leaves ample budget.
- Repeat178580749 **passed**: nine tests, two finite2000-sample archives, exit0, wall19s, memory580MB, charge .02 SU. Both successful archives and failed/success PBS/case logs were collected locally through gadi-dm. Total actual spent .06 SU.
- **Long gate178580950.gadi-pbs submitted**, four CPUs,16GB,2h (maximum16 SU), fixed commit90543f1; output remote base + `gadi-long-90543f1/`. Four N32768 cases: eta=.44/.476, ordered/random, seed910203, burn20000 and record100000 microscopic steps, saved every25 steps (4000 samples). Next heartbeat: inspect job status/logs, collect complete archives through gadi-dm, verify count/hash/finite data, then analyze separately from stride1 short pilots. Do not resubmit or fit SPI coordinates while this is running.
- [Stuart–Landau audit](stuart-landau-temporal-results-260910.md) complete: 28 baseline+5 sensitivities. Half-dt mean/SD changes <=1.1e-7 in the two tested cells; gamma1.2/N800 remains burn-sensitive. Source/report committed87f6bb9. Eleven combined current local tests pass. No new p90 has been computed.

## Objective and authorization

The user authorized overnight literature investigation and focused simulation runs. Seek 2–3 recognizable systems beyond the Kuramoto control with established physical order parameters and reproducible changes over a published local boundary. Success means sensitivity/recovery in a common SPI–SPI representation across M and T; outperformance of purpose-built statistics is not required. Observed M stays <=32; physical N may grow to 10^6 when size/time checks warrant it. Do not assume success.

## Sources and priority

- Stuart–Landau: Matthews–Strogatz, PRL 65, 1701 (1990), https://doi.org/10.1103/PhysRevLett.65.1701. K=.8, uniform frequencies; locking to large collective oscillations, then more complex collective motion. Retain Z(t), |Z(t)|, spectra and block variability. Mean |Z| alone is insufficient.
- Miller–Huse: https://doi.org/10.1103/PhysRevE.48.2528 and refined study https://doi.org/10.1103/PhysRevE.55.2606. Sign magnetization and symmetry breaking.
- Desai–Zwanzig: https://doi.org/10.1103/PhysRevResearch.5.013078 (Zagli et al., not Evangelou et al.) and https://doi.org/10.1103/PhysRevE.110.014121. First moment; investigate convergence concerns before any million-particle run.
- Exact Kaneko ring: https://doi.org/10.1103/PhysRevE.87.052905. Verify the persistence definition, mapped parameter conventions, and observation horizon before simulating a published persistence boundary. No generic scalar across the whole Kaneko regime diagram has been established here.

Independent literature checks determine targets; existing repository evidence only prevents duplicate work and supplies implementation/provenance. Previous assistant claims about a universally optimal sensor layout or guaranteed M=32 performance are hypotheses, not established results. Preserve the prior Ising implementation exclusion unless the user explicitly revises it.

## First concrete run

Use scripts/scout_stuart_landau_streaming.py, which retains O(N + 32T) state, unlike the existing generator's internal O(NT) allocation. Verify short traces against the existing generator before production. Pilot N=32 and N=800, K=.8, gamma=.70,.74,.75,.80,.90,1.0,1.2; two seeds; 200 burn units and 8000 samples at sample_dt=.1, dt=.02. This is 28 exploratory rows. Compare consecutive blocks and rerun selected boundary rows with half dt and longer burn if warranted. Use the full complex order trace in the rotating frame for collective spectra; the explicit laboratory carrier affects real-channel SPIs and must be retained as an observation choice. Do not declare chaos from spectral entropy alone.

## Iteration and stopping

1. Inspect existing jobs/artifacts first, including other active workstreams.
2. Verify exact equations, physical targets and literature boundary provenance.
3. Run bounded physics pilots, checking timestep, burn, block stability and seeds.
4. For lattices, compare equal-M contiguous, dispersed and dispersed-block views; none is presumed optimal. Mean-field views use fixed random particle indices.
5. Use M=8,16,32 nested views and temporal windows long enough for the dynamics. Compute full p90 only after a physical contrast and valid observations exist.
6. Freeze reduction without controls/Q; keep independent seeds for evaluation. Report target recovery, transition location, and validity; distinguish a time-window regime coordinate from reconstruction of an instantaneous Q(t).
7. Increment N only to resolve a stated question; do not multiply an entire bank by the million-particle size. First bound total work and memory with a pilot.

Overnight resource ceiling chosen conservatively: at most 1 KSU of new Gadi work in this pass, with small smoke jobs before any farm. Stop new submissions after 10 hours from 2026-09-09 18:51 UTC; finish collecting already running bounded jobs. Produce a concise cited findings report with run IDs, costs, plots, failures and the next justified step. Pause the follow-up once complete or requiring new input.

## Initial operational evidence

Gadi reachable at 2026-09-09 18:51 UTC; qstat showed no jobs for this user. nci_account: 87.34 KSU available; Scratch 185.41k/202k inodes, gdata 44.38k/70k. Prefer compact archives. Local tree contains extensive unrelated changes; preserve them. No production simulation has been submitted yet.

Streaming integrator passed a matched-seed 12-sample comparison against the existing generator (sensor trace and complex global order, tolerance 1e-12). The first full local pilot completed: N=32, gamma=.8, 8000 samples, seed=910001; archive data/order_parameter/stuart_landau_dynamics_260910/local-pilot-N32-gamma0p8.npz. Mean R=.2211, SD R=.1580. Eight consecutive block means drift from .2262 to .2170; investigate time convergence rather than treating this pilot as truth.
