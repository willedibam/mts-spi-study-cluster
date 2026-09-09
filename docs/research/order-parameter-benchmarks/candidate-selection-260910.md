# Recognizable order-parameter benchmarks: selection and first experiment

Status: user approved proceeding after the literature comparison, 2026-09-10.
This updates the portfolio priorities in report-source.md; it does not alter
past analyses, failed gates, or confirmation labels. No new SPI result yet.

## Scientific purpose

Test whether one target-blind representation retains interpretable collective
state information across unequal recording dimensions and observation layouts.
Separate sensitivity across controls, held-master tracking, and robustness to
observation changes. Numerical Q calibration is supervised even if the coordinate
was unsupervised. Do not claim new physics, a newly discovered known order
parameter, recovery of all unobserved dynamics, or superiority to a dedicated Q.
Sensor-count compatibility does not establish invariance to modality/mixing.

Retain Kuramoto (positive control), Miller–Huse (chaotic lattice ordering), and
Stuart–Landau (collective temporal dynamics). Pilot Vicsek as a fourth,
mechanistically distinct example. Desai–Zwanzig remains an existing supported
reserve, not something to rerun merely to fill a slot. Final selection depends
on physics/observation validity and complementary evidence, not largest q–Q rho.

## Literature decisions

| Candidate | Exact target/boundary | Decision |
|---|---|---|
| Vicsek | Global polarization across angular-noise ordering boundary | First new physics/observation pilot |
| Blume–Capel | Absolute magnetization across first-order segment of crystal-field/temperature diagram | Strong scalar-spin reserve; no implementation in this pass |
| Potts q=8/10 | Majority-state magnetization at square-lattice first-order transition | Reserve; integer category encoding is not innocuous |
| Open TASEP | Bulk density across alpha=beta<1/2 | Reserve; occupation complement/reflection can obscure dependence-only contrast; local shock sampling problematic |
| Montbrio–Pazo–Roxin QIF | Population rate/voltage across saddle-node low/high-activity boundary | Neural bridge for a later pilot; rate is a collective bifurcation variable, not automatically a symmetry-breaking order parameter |

Primary sources:
- Vicsek: Chate et al., PRE 77, 046113 (2008),
  https://arxiv.org/abs/0712.2062; Eqs (1),(2),(7),(8), Figs 1,2,10.
- Blume–Capel: Zierenberg et al., https://arxiv.org/abs/1612.02138;
  spin {-1,0,1}, phase plane and finite-size analysis of both boundary segments.
- Potts: Brown et al., https://www.nature.com/articles/s41598-022-17359-w;
  physical transition and prior information-theoretic measurements.
- TASEP: Derrida et al., https://doi.org/10.1088/0305-4470/26/7/011;
  exact stationary phase diagram, density rather than current jumps at LD/HD.
- QIF: Montbrio et al., https://arxiv.org/abs/1506.06581, Fig 1 and Eq (12).
  Unforced instantaneous-synapse macrodynamics has fixed-point attractors;
  periodic forcing is used for sustained complex population dynamics. Avoid
  divergent QIF voltage as a naive channel; bounded phase or defined synaptic
  observations need an explicit protocol, including quiescent-channel handling.

QIF observation warning (our calculation from that mean-field model, not a new
simulation): at the paper's J=15, mean drive=-5, Delta=1, the three stationary
rates solve pi^2 r^4 -15 r^3 +5 r^2 -1/(4 pi^2)=0. Positive roots are
.0811344, .4729803, 1.0305968 (low stable / saddle / high stable).
The Lorentzian fraction of tonically active neurons in a stationary mean field
is 1/2+atan(mean_drive+J*r)/pi: .0823 on the low branch versus .9697 on the high.
Thus 32 randomly selected individual neurons contain only about 2.6 active
neurons on average on the low branch in this limit. This does not prove finite-N
voltage channels are exactly constant, but makes input validity a substantive
gate rather than assuming a neural model automatically fits all SPIs.

Equilibrium spin statistics do not disqualify dynamical benchmarks: local
Glauber/heat-bath dynamics is legitimate. Nonlocal accelerated sampling is not
the same temporal process. Prior Ising-family implementation exclusion is not
silently overridden by adding a reserve to a literature table.

## Vicsek specification and staged experiment

Implement Chate et al.'s angular-noise forward-streaming version, not a mixture
of update conventions: headings update synchronously from radius-1 neighbors
including self; add uniform noise on [-pi*eta,pi*eta]; positions advance with
NEW velocity. Periodic square, speed .5, density 2, timestep 1.
N=2 L^2. The published angular-noise crossover is around L=128 (N=32768),
not a universal threshold. Fig 2 uses 2e7 steps: our short scouts cannot claim
the same stationary phase coexistence or first-order finite-size scaling.

1. Verify linked-cell step against independent O(N^2) oracle, periodic edges,
   aligned noiseless limit, speed, RNG repeatability, and eta=1 polarization floor.
2. Smoke: L=32; eta=.44,.476,.52; ordered/random starts; matched seed 910201;
   burn1000 and record2000 steps. Record paired views, not separate simulations.
3. Repeat at L=64,128 to measure cost and expose finite-size/start dependence.
4. Extend time and add independent seeds on difficult anchors BEFORE refining
   the boundary or submitting p90. Keep persistent branch dependence explicit;
   hysteresis in a short run is not a measured coexistence line.
5. Only after physical windows and nondegenerate observations pass, freeze a
   one-control pilot and full-p90 development/held-master split. A target-blind
   mask/transform is shared across M,T; do not separately fit each view then
   call their coordinates comparable. No label-selected projection/sign/layout.

Observation: M=8,16,32 nested prefixes; each Cartesian component is a separate
M-channel observation arm, not a concatenation secretly doubling M. Store both
components to audit rotational/projection dependence. Compare fixed random
particle identities with initially-local identities (NOT a persistent local
patch after motion). True fixed spatial observations use width-4 bins: dispersed
or nested contiguous rectangles (2x4,4x4,4x8). Store counts and current density
sum(unit headings)/area, whose zero in an empty bin is physically defined.
Never substitute zero for an undefined conditional mean heading. Bins average
many particles, so do not describe them as M individually observed agents.

Track full phi(t), blocks, Binder, start dependence, histogram, runtime, and
snapshot morphology. Audit short-view validity and sample polarization error
without interpreting these raw-statistic checks as evidence about SPI–SPI.
Global order and observed sensors overlap: leave-observed-out truth is a later
sensitivity for particles, not a claim of entirely hidden information today.

## Execution/provenance

Code: scripts/scout_vicsek_streaming.py; tests/test_vicsek_streaming.py.
Configs: configs/scout/vicsek-observation-{smoke,size-pilot}.yaml.
Run with `.venv/bin/python scripts/scout_vicsek_streaming.py --config CONFIG
--output-dir UNIQUE_DIR`. Refuses overwrite; each compact NPZ stores parameters,
script/config hashes, full scalar order, small observation banks and final state.
Memory O(N+MT) at fixed density/bin width; cell-list runtime can still grow with
local clustering. No dense N-by-N graph; no full N-by-T trajectory.

Outputs: data/order_parameter/vicsek_observation_260910/{smoke,size-pilot}/.
All first-pass work is local; no Gadi submission. Queue empty at inspection;
87.34 KSU available, Scratch inode185.41k/202k. Existing overnight 1-KSU/time
ceiling is not expanded by this pilot. Preserve unrelated working-tree changes.

The short L=64 eta=.44 means differ by .182 between starts (.387 versus .206).
That triggers a time check, not a p90 submission. Added
configs/scout/vicsek-observation-time-pilot.yaml: L=64, burn5000/record8000,
eta=.44,.476 with the original matched seed and both starts; an additional
seed 910202 at eta=.476. Output time-pilot/. Analysis script
scripts/analyze_vicsek_observation_scout.py writes reproducible descriptive
tables/figures; do not combine unequal horizons into one convergence curve.

Next staged (not launched with the local first pass; subsequently submitted as
Gadi178580950 after a successful scheduler smoke, see overnight-260910.md):
configs/scout/vicsek-long-physics-gate.yaml uses L=128, burn20000, 100000 further
microscopic steps, storing every 25th step; eta=.44,.476 and both starts, new
seed910203. This is physics-only thinning, not a chosen p90 sampling interval.
Measured local timing suggests roughly 25 minutes/case, with environment and
clustering uncertainty; use a scheduler-backed smoke before expansion. Inspect
new outputs/processes and respect the existing overnight deadline/resource cap.
If prolonged start dependence persists, report a branch/window-conditioned
estimand or defer Vicsek; do not silently average metastable branches or keep
adding runs to obtain a desired cliff.
