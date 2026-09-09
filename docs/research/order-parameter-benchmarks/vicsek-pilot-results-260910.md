# Vicsek first-pass results, 2026-09-10

## Outcome

Implemented and tested the published angular-noise, forward-streaming model.
Completed 24 local physics runs: 18 size/control/start cases plus 6 longer-time
cases. No new p90/SPIs were computed; no Gadi jobs submitted or SUs charged.
The new benchmark is computationally feasible, has clear order contrast and
nonconstant small observations, but stationary near-boundary statistics are
not yet established. Do not promote these plots to a confirmed discontinuity
or SPI–SPI recovery result.

The proposed main suite remains Kuramoto, Miller–Huse, Stuart–Landau, and
Vicsek if the next physics gate supports it. Existing results are not replaced.
See [selection/protocol](candidate-selection-260910.md) for reserve models,
literature, observation definitions and inference limits.

## Actual experiments

All cases: density2, speed.5, noise in [-pi*eta,pi*eta], unit radius and timestep.

| Pass | Physical size | Controls/starts | Time | Seeds |
|---|---|---|---|---|
| Size scout, 18 cases | N=2048,8192,32768 (L=32,64,128) | eta=.44,.476,.52; aligned/random | burn1000, record2000 steps | 910201, matched across starts/controls |
| Time check, 4 cases | N=8192 | eta=.44,.476; aligned/random | burn5000, record8000 steps | 910201 |
| Additional seed, 2 cases | N=8192 | eta=.476; aligned/random | burn5000, record8000 steps | 910202 |

Ten relevant tests pass (eight new tests and two existing ZGB regression tests).
The cell-list update agrees with a separate dense all-pairs oracle at tolerance
1e-12, including periodic neighbors and noninteger box sizes. Tests also cover
aligned/no-noise motion, reproducibility, observation nesting, constant-channel
handling, and E(phi^2)=1/N at maximal angular noise.
Total recorded simulation walltimes summed across cases: 489.88 seconds;
L=128 short cases cost 34–39 seconds each locally. This is timing evidence,
not an HPC scaling guarantee. Observation archives retain code/config hashes.

## Physics findings

1. Ordering contrast exists at every size, but is size- and time-dependent.
   At N=32768, eta=.52 yields mean phi=.0354/.0369 across aligned/random
   starts. At eta=.44 the short means are .2506/.3491, with considerable
   evolution within each recording. The middle eta=.476 means are .0961/.1665.
   A three-point sweep cannot establish a jump or precisely locate the boundary.
2. Extending N=8192 eta=.44 reduces the observed start-mean difference from
   .1817 (short) to .0175 (long); long means are .4167/.4342. This is evidence
   that the short protocol was inadequate, not a formal convergence proof.
3. At eta=.476 the four long-run means span .1300–.2119. Consecutive block
   ranges reach .163. Persistent fluctuations are physically allowed; block
   variation alone is NOT proof of nonstationarity. These few runs do not yet
   determine the stationary distribution, mixing time, or start sensitivity.
4. No observed Binder estimate is negative in these runs. That does not refute
   the published first-order transition: the reference's angular-noise finite-
   size figure uses much longer histories (2e7 steps). Our own discontinuity
   evidence remains insufficient. We need not rederive universality to use the
   model, but cannot replace finite-time behavior with assumed equilibrium truth.

Source: Chate et al., PRE77,046113 (2008), Eqs1,2,7,8 and Fig2,
https://arxiv.org/abs/0712.2062.

## Observation findings and limits

Each master provides nested M=8,16,32 and T=100,500,1000,2000 views.
Across the tested prefixes, all channels had temporal SD>1e-8. This basic
check is NOT a full-p90 validity pass. Tested observations: fixed dispersed and
initially-local particles (x/y components separately), and dispersed/contiguous
fixed spatial bins (density and x/y current separately).

Particles selected locally disperse over time; they are not persistent local
patches. Fixed spatial bins are true local views, but each observes an aggregate
of particles (mean32 per width4 bin here), not one microscopic process.

An illustrative N=32768, eta=.52, aligned-start T=1000 window has mean global
phi=.0395 but mean 32-random-particle polarization about .1680 (bias .1285,
instantaneous RMSE .1558). A small-sample magnitude has a positive noise floor.
This neither proves nor disproves z recovery; it does rule out assuming a tiny
subsample is an unbiased instantaneous global-order measurement. In that same
window mean absolute x-channel correlation is .037 for dispersed particles,
.083 for dispersed-bin current and .126 for contiguous-bin current. These
are observation-specific descriptive checks, not a chosen winning layout.

## Evidence and reproduction

- configs/scout/vicsek-observation-{smoke,size-pilot,time-pilot}.yaml
- scripts/scout_vicsek_streaming.py
- scripts/analyze_vicsek_observation_scout.py
- data/order_parameter/vicsek_observation_260910/{smoke,size-pilot,time-pilot}/
- [Size/control plot](../../../data/order_parameter/vicsek_observation_260910/size-analysis/physics.png)
- [Longer-time plot](../../../data/order_parameter/vicsek_observation_260910/time-analysis/physics.png)
- Each analysis directory contains physics.csv, observation_checks.csv,
  observation_summary.csv and summary.json. Do not pool unequal horizons.

Recreate the size analysis with `.venv/bin/python
scripts/analyze_vicsek_observation_scout.py --inputs
data/order_parameter/vicsek_observation_260910/smoke
data/order_parameter/vicsek_observation_260910/size-pilot --output-dir
data/order_parameter/vicsek_observation_260910/size-analysis`.
For the long analysis, use time-pilot as the sole input and time-analysis output.

## Next bounded action

Prepared, NOT launched: configs/scout/vicsek-long-physics-gate.yaml, four L=128
cases with a new seed, 20000 burn steps and 100000 subsequent microscopic steps.
Observe every25 steps for this physics gate; do not assume this is suitable p90
sampling. Scheduler/provenance smoke and the existing overnight resource/time
limits apply. If start dependence persists, use an explicitly branch/window-
conditioned estimand or defer this benchmark, rather than force a stationary
mean interpretation. No million-particle run or full-p90 sweep is justified yet.
