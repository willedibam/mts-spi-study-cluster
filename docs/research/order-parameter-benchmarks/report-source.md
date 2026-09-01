# SPI--SPI dynamical order-coordinate benchmark audit

Audience: paper authors. Date: 2026-09-01. Status: active source report.

## Scope and direct answer

Select a small suite for testing whether a target-blind SPI--SPI coordinate
recovers an independently defined changing physical order parameter or localizes
a known dynamical boundary. Verify exact model variants before scaling.

Current recommendation:

1. Miller--Huse `mu=3` is the cleanest next canonical benchmark; its focused
   two-million-step truth gate passed with uncertainty retained explicitly.
2. The original uniform-frequency Stuart--Landau population is accepted as the
   best additional system. Repository pilots reproduce its published phase path
   and coarse control plane.
3. The quadratic CML is an informative secondary/negative regime-coordinate
   stress test, not a canonical scalar-order-parameter headline.
4. Defer Vicsek, CGLE and neural application bridges until these pass full p90.

This is a gated portfolio, not an open-ended model search. Retain two or three
systems only when they contribute distinct evidence: Miller--Huse as the
canonical nonequilibrium transition, Stuart--Landau as the published collective
oscillator path and `M x T` robustness assay, and quadratic CML only if its
branch-structured regime coordinate is recoverable without target-guided
selection. Do not add another system merely because a run fails. The strongest
canonical fallback is periodically driven kinetic Ising, but the prior request
to delete Ising-family work takes precedence unless explicitly revised.

The current one-control grids are evidence-led. Stuart--Landau fixes the
published `K=.8` intercept and uses interleaved confirmation values spanning the
four Fig. 1 regimes. Miller--Huse uses coarse ordered/disordered anchors and a
dense bracket around the refined `g_c=.20534(2)`. Quadratic CML preserves the
established/current `eps=.3, alpha=1.60:.01:2.00` development path; its long-run
physics gate localizes branch coexistence to roughly `alpha=.74-.76`. If that
system warrants an independent confirmation, use coarse regime anchors plus a
dense `alpha=1.70:.01:1.82` window rather than repeating all 41 development
controls.

## Audited systems

### Miller--Huse CML

- Exact synchronous periodic square-lattice equation:
  `x'=(1-4g)f(x)+g sum_4nn f(x_nn)`, with the odd piecewise-linear `mu=3` map.
  Repository implementation and `q_spin_abs=<|L^-2 sum sign(x)|>` match.
- Refined critical coupling is `g_c=.20534(2)`. The original paper reported
  exponents consistent with 2-D Ising; later precision work found
  `nu=.887(18)`, suggesting a synchronous-update class while not excluding a
  very slow Ising crossover. Do not claim established Ising universality.
- Existing `L={32,64,128}` scouts reproduce the order rise and susceptibility
  peak. A focused `L=128`, two-million-step future-truth audit gives p05
  effective samples `63.2` and adjacent 250k-block-difference p95 `.099`.
  At the hardest `g=.205` cell, median block-mean SE is about `.021` on a total
  Q range near `.8`. This passes development with uncertainty shown. Store the
  long-run scalars and eight block summaries, not the redundant future series.
- Physical `N=L^2`. Adequate `L=128` means `N=16,384`; full-p90 `M=N` is
  infeasible because dense MPI memory is quadratic in M. Primary observation is
  one frozen dispersed ordering with `M={8,16,32}`; patches are sensitivities.
- Full-p90 development completed 288 rows and all 289 SPIs. Target-blind
  hygiene retained 32,836/41,616 pairs; PC1 explained `.2443` and had loading
  cosine `.976-.994` across M-specific fits. Held instances 4--7 gave row-level
  `rho(q,Q)=.9125`, pooled control-mean `.9650`, and per-M cell-mean
  `.9650-.9860`; within-control rho was only `.2381`. Long-truth block-mean SE
  was median `.00072`, p95 `.0193`. Temporal spectral entropy was stronger
  overall (`rho=.9607`) and control-only MAE `.00325` was far below q-readout
  `.0892`. This is strong canonical order-coordinate development evidence, not
  predictive superiority; independent controls/seeds are still required for a
  confirmation claim.

### Quadratic/Kaneko CML

- Repository exactly matches the mapped-output 1-D periodic lattice
  `x_i'=(1-eps)f_alpha(x_i)+eps/2[f_alpha(x_i-1)+f_alpha(x_i+1)]`,
  `f_alpha(x)=1-alpha*x^2`.
- The relevant Kaneko `(alpha,eps)` diagram is qualitative. At `eps=.3`, the
  `alpha=1.60:2.00` path crosses documented pattern-selection,
  intermittency/pattern-competition and fully developed spatiotemporal-chaos
  regions. A reported exact-model `alpha=1.8,eps=.3` transient lasted more than
  1.5 million updates, whereas `alpha=1.88` was fully developed turbulence.
  The old 2,000-step burn is therefore inadequate for an asymptotic claim.
- Diagrams disagree when authors diffuse `x` before applying `f`, use global or
  one-way coupling, change dimension/topology, follow different weak-coupling
  paths, or change initial basin/size/time. Those are different models or
  protocols, not contradictory measurements of one phase plane.
- There is no canonical scalar across the plane. Use the predeclared vector
  given below; promote a scalar only if stable to N, seed, block and diagnostic
  choice.
- Primary p90 observation uses fixed `N=512`, nested dispersed
  `M={8,16,32}` views and the original `alpha=1.60:0.01:2.00` path.
  `M=N={8,16,32}` use ten control anchors as separate finite-size arms that
  must not be pooled. A focused dense physics gate over `alpha=1.71:0.01:1.80`
  precedes p90.
- The `10 alpha x 3 N x 8 seed` scout and dense `N=512`,
  `alpha=1.71:0.01:1.80` extension are stable over four consecutive 5k blocks
  away from the reorganization. Merged pooled physical-diagnostic prefix-error
  p95, normalized by full-record IQR, is `.194,.119,.096` at
  `T=100,500,1000`. All eight dense-path seeds are low-temporal-entropy through
  `alpha=.73`; coexistence occurs over `.74-.76`; all eight are high-entropy by
  `.77`. At `.75`, five are low- and three high-entropy. Preserve this branch
  structure. The predeclared physical vector is temporal spectral entropy,
  dynamical spatial-pattern entropy, selected-band power and period-two
  residual. The p90 physics gate passed.
- Full-p90 development completed 1,224 rows with all 289 SPIs; target-blind
  hygiene retained 15,673/41,616 pairs and q-PC1 explained `.4107`. The
  predeclared four-observable physical vector is `.9519` one-dimensional on
  this intercept. Held q1 versus physical-PC1 has `rho=-.7332`; 2-D distance
  geometry is `.6669` overall and `-.2325` within alpha. Mean absolute
  correlation and temporal spectral entropy are much stronger
  (`|rho|=.9306/.9501`). Supervised q-vector MAE `.1518` only slightly improves
  on the two-input baseline `.1603` and is far worse than control-only `.0630`.
  Keep this as a secondary/negative stress test and do not spend a confirmation
  run trying to rescue it.

### Stuart--Landau population

- Exact original model:
  `zdot_j=(1-|z_j|^2+i*omega_j)z_j+K(Z-z_j)`, `Z=N^-1 sum z_j`, using
  `N=800` and evenly spaced frequencies on `[-gamma,gamma]`.
- Published `(K,gamma)` regions are locking, incoherence, amplitude death and an
  intervening unsteady region containing large/Hopf oscillations,
  quasiperiodicity and chaos. `R=|Z|` is the published order-parameter amplitude.
- Fixed `K=.8` is not an arbitrary intercept. It is the paper's Fig. 1 path:
  `gamma=.6,.8,1.0,1.2` respectively illustrate locking, large oscillations,
  irregular/chaotic order-parameter motion and incoherence. A `K>1` path can
  reach amplitude death, but is a worse first p90 assay because it creates
  exactly constant channels.
- The implemented `K=.8` path reproduces the paper at `N=800`:
  `(gamma, mean R, sd R)=(.6,.784,~0),(.8,.290,.182),
  (1.0,.175,.077),(1.2,.034,.018)`. A 168-run coarse plane reproduces the four
  broad published regions. `N=M={8,16,32}` retains the sequence with a finite-N
  incoherence floor; fixed `N=800`, `M<N` is the phase-diagram reference.
- First p90 path excludes amplitude death because exactly constant channels are
  genuine input degeneracy. Use physical vector `(mean R, sd R, mean |z|^2)`;
  the primary scalar-order coordinate remains `mean R`.
- Full-p90 development used all 289 SPIs. Target-blind hygiene retained 24,466
  pairs spanning 234 SPIs; PC1 explained `.661` and had loading cosine
  `.977-.987` across all `M x T>=500` source fits. Held per-cell mean
  `rho(q,Q)=.85-.99`; the pooled gamma-mean curve has `.976`, while raw pooled
  rows have `.756`. `T=100` remains a cell-level stress result: paired q-rank
  agreement with `T=1000` is `.628/.740/.944` for full `M=8/16/32`; at `T=500`
  it is `.777/.909/.957`. Within-gamma recovery is weak. Analytic phase
  coherence is a stronger baseline (`rho=.932`, within-gamma `.779`), so do not
  make a superiority or microscopic-fluctuation claim. The sealed 1,296-row
  joint confirmation stopped before outcome access: 14 secondary
  partial-observation rows exceeded the predeclared 5% selected-feature
  missingness gate (max `.0785`, p99 `.0526`). All 648 primary full-observation
  rows pass the same gate (max `.0266`). Their separately sealed arm-specific
  result gives row-level `rho=.7485` (95% CI `.7356-.7607`), within-gamma
  `.3306`, pooled gamma-mean `rho=1.0`, and cell-mean `rho=.833-1.0`. Analytic
  phase coherence remains stronger (`rho=.9299`, within-gamma `.7521`), while
  frozen-q isotonic MAE `.0407` is worse than control-only `.0334` (bootstrap
  difference CI `.0056-.0091`). Preserve the joint failure artifact and claim
  feasibility of order-coordinate recovery, not superiority or information
  beyond the known control. No `T=2000` extension is needed.

### Deferred candidates

- Vicsek has canonical polarization but the angular-noise transition is
  discontinuous only at populations around tens of thousands in the published
  finite-size study; `M=N<=32` and `T=100` are not physical.
- 1-D CGLE has established phase/defect-turbulence diagrams but no unique scalar
  order parameter, and the apparent defect boundary moves with size/time. It is
  a later regime-coordinate example.
- Sompolinsky--Crisanti--Sommers is the strongest neural-chaos bridge, but the
  canonical subcritical state decays to zero and makes p90 degenerate. A driven
  or noisy variant changes the transition and requires its own validation.
- Brunel E/I spiking has a recognized state plane but requires large N and a
  vector of rate/irregularity/synchrony observables; sparse binning is a poor
  first p90 input.
- Periodically driven kinetic Ising is the strongest additional canonical
  candidate: it has explicit stochastic dynamics, a one-control frequency or
  half-period sweep, finite-size scaling, and cycle-averaged magnetization as
  its established dynamic order parameter. It is scientifically different from
  the deleted equilibrium/static Ising assay, but implementation is deferred
  pending resolution of the user's broader instruction to remove Ising work.
- The contact process has a canonical active-site-density order parameter, but
  finite subcritical runs terminate in an absorbing all-zero state. A
  quasi-stationary simulation avoids extinction by changing the sampling
  measure, so it is not a clean first p90 benchmark.

## Observation and sample-length contract

- Use `M=N` where the published finite-population dynamics remains meaningful;
  also show small-N effects rather than concealing them. For spatial/large-N
  systems, keep physical N fixed, compute Q globally and take nested dispersed
  sensor prefixes. Treat these as distinct full-state and information-limited
  assays.
- Primary robustness cells are `M={8,16,32}`, `T={100,500,1000}`. Fit the
  claim-bearing shared transform on `T>=500`; apply it to `T=100` as a stress
  cell. Promote T=100 only if missingness and coordinate stability pass. This
  complete grid is assigned to Stuart--Landau; initial Miller--Huse and
  quadratic-CML p90 development uses `T=1000` across the three M values.
- Existing 2,520-row p90 evidence retains `12,720`, `13,499`, and `13,855`
  99%-valid meta-features at `T=500,1000,2000`. Thus 1000-to-2000 does not rescue
  many features. A same-trajectory prefix pilot must decide precision.
- Independent-sample Fisher-z SE is approximately `1/sqrt(T-3)`:
  `.102,.0449,.0317,.0224` at `T=100,500,1000,2000`; autocorrelation worsens it.
  P90 also contains lag-20 correlations, history-10 entropy, kNN/kernel MI/TE,
  order-20 spectral Granger and lag-10 cointegration. T=100 is computable for
  many, but high variance/low power is expected and some DCE estimators can fail.
- Claim-bearing p90 grids are one-dimensional by default. A cheap replicated
  two-control physics map may validate a chosen intercept; a full
  `control-plane x M x T x instance` grid is justified only if no one-control
  path separates the regimes.

## Required visual evidence

1. Physical `c -> Q` curve with seed uncertainty, or `Q(c1,c2)` heatmap with
   replicated/published boundaries.
2. Same control axis with physical Q and frozen target-blind q.
3. Held-out q versus Q; show `q -> Qhat` separately as supervised.
4. Selected robustness system: facet by T, colour by M, faint replicates and
   bold cell means under one frozen transform, plus an M-by-T performance map.
5. Optional target-free `Var(q)`/susceptibility localization.
6. Control plane: physical and learned maps plus a discrepancy/boundary panel.
7. One small orientation panel per system: equation, channel meaning,
   representative regimes and `plot_mts_heatmap(..., method="robust")` using the
   existing per-process `icefire` implementation. It is context, not evidence.

## Claim-to-source ledger

- Miller & Huse (1993), *Macroscopic equilibrium from microscopic
  irreversibility in a chaotic coupled-map lattice*, APS:
  https://doi.org/10.1103/PhysRevE.48.2528
- Marcq et al. (1997), *Critical behavior of a two-dimensional coupled-map
  lattice*, APS: https://doi.org/10.1103/PhysRevE.55.2606
- Lemaître & Chaté (1999), *Nontrivial critical behavior in a coupled map
  lattice*, APS: https://doi.org/10.1103/PhysRevLett.82.1140
- Kaneko (1989), *Pattern dynamics in spatiotemporal chaos*, Elsevier:
  https://doi.org/10.1016/0167-2789(89)90227-3
- Willeboordse (1993), different pre-map coupling convention, APS:
  https://doi.org/10.1103/PhysRevE.47.1419
- Loskutov, Prokhorov & Rybalko (2002), exact quadratic-CML long transients:
  https://chaos.phys.msu.ru/loskutov/PDF/TMPh_quadratic_cml.PDF
- Matthews & Strogatz (1990), *Phase Diagram for the Collective Behavior of
  Limit-Cycle Oscillators*, APS: https://doi.org/10.1103/PhysRevLett.65.1701
- Chaté et al. (2008), large-system Vicsek finite-size study:
  https://arxiv.org/abs/0712.2062
- Egolf & Greenside (1995), CGLE phase/defect transition characterization:
  https://doi.org/10.1103/PhysRevLett.74.1751
- Sompolinsky, Crisanti & Sommers (1988), random neural-network chaos:
  https://doi.org/10.1103/PhysRevLett.61.259
- Brunel (2000), sparse E/I spiking-network state diagrams:
  https://doi.org/10.1023/A:1008925309027
- Sides, Rikvold & Novotny (1998), kinetic-Ising dynamic transition and
  cycle-averaged-magnetization order parameter:
  https://doi.org/10.1103/PhysRevLett.81.834
