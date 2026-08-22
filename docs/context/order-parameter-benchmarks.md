# Canonical order-parameter benchmarks

## Scientific status

- Only one canonical order parameter has yet received a full SPI--SPI evaluation: Kuramoto phase coherence `R_N`. The quadratic Kaneko-CML `Q_sel`, entropy, period-2 score and Lyapunov diagnostics are not canonical order parameters.
- Completed Kuramoto run: 880/880 datasets, `M=20,T=1000,N=256`; Gaussian-fitted PC1 had held-out overall/within-control Spearman `-.945/-.689` on Gaussian and `-.915/-.596` on logistic data. The physics and leakage controls were sound.
- That result is not claim-bearing under its frozen rule. Four evaluation rows had structured high-synchrony missingness and violated the maximum-row gate; cross-path MAE `.0789` was worse than mean absolute input correlation `.0583` and a selected individual SPI `.0676`. Existing data are now development-only.
- Interpret it as: a frozen unsupervised coordinate strongly associated with future global `R`, including fixed-control variation, but failed feature eligibility. PC1 was a legitimate predeclared linear baseline, not an empirically optimal coordinate (explained variance `.381`).
- Sine-circle `k=3.1` remains rejected: its printed modulo recurrence does not preserve the claimed absorbing set and direct replication failed. Quadratic Kaneko remains exploratory regime discovery.

Primary evidence: `notebooks/embeddings/kuramoto-order-parameter-benchmark.ipynb` and `data/order_parameter/kuramoto_order_benchmark/`. Sources: Kuramoto <https://doi.org/10.1103/RevModPhys.77.137>; Miller--Huse <https://doi.org/10.1103/PhysRevE.48.2528>; finite MH analysis <https://doi.org/10.1103/PhysRevE.55.2606>; Yang magnetization <https://doi.org/10.1103/PhysRev.85.808>.

## Prospective benchmarks

### Miller--Huse (secondary benchmark)

- Dynamics: `x'=(1-4g)f_mu(x)+g sum_nn f_mu(x_nn)` on a periodic 2-D lattice. The original `mu=3` path passes start-state convergence; `mu=1.9` does not and is excluded from confirmatory evidence.
- Define `s_r=+1` for `x_r>=0`, otherwise `-1`; `m_s=L^-2 sum_r s_r`. The canonical finite-size target is `Q_MH=<|m_s|>_future`; `sqrt(<m_s^2>)` is a second-moment sensitivity. Never average signed `m_s` across symmetry flips.
- SPI sees only a contiguous `4x5` (`M=20`) field patch. The target uses a disjoint future full lattice; patch-excluded future magnetization is the anti-self-inclusion sensitivity.
- Generator now supports rectangular patches, general `mu`, future truth, hidden complement, hot/cold/random starts and compact storage. Full field movies are off by default.

### Kinetic Ising (primary new benchmark)

- Zero-field anisotropic square Ising, `H=-Jx sum_x sisj-Jy sum_y sisj`, equilibrated with Wolff clusters then observed under checkerboard heat-bath updates. Fully simultaneous updates are not substituted.
- Primary target is `Q_Ising=<|m|>_future`; RMS magnetization is a sensitivity. Use isotropic `(Jx,Jy)=(1,1)` and anisotropic `(1,.5)` paths matched by `u=sinh(2 beta Jx)sinh(2 beta Jy)`.
- The exact critical line is `u=1`; thermodynamic `m=0` for `u<=1` and `(1-u^-2)^(1/8)` otherwise. Equal `u` therefore gives the same exact macroscopic magnetization but different spatial microstructure.
- SPI sees a `4x5` binary patch. Continuous estimators legitimately fail on ties; the target-blind stress scout must remove them rather than jittering the data.

## Physics-first contract

- Smoke configs and the full `benchmarked_p90.yaml` pipeline pass locally. Smoke contrasts are not physics validation.
- Dataset-scoped seed derivation is clone-invariant: absolute output paths were removed from the seed payload after the first Gadi smoke exposed local/cluster trajectory mismatch. Every dataset still records its resolved seed.
- Coarse and convergence scouts completed on Gadi. Ising used `L={32,64,128}`, three initial states, two anisotropy paths, 80k future steps and 864 tasks. Initial-state mean spread was median `.0061`, max `.0282`; matched-path gap mean `.0059`, max `.0214`; hidden/full target p95 gap `.00027`. Finite-size rounding and critical slowing are visible as expected.
- The per-realization Ising future target remains noisy near `u=1` (20k-block difference p95 up to about `.14` at `L=64`). The primary physical target is therefore an independently estimated cell/ensemble `Q_L=<|m|>`; per-realization future truth is a sensitivity. Do not call within-cell failure evidence against representation if the target itself is unreliable.
- Miller--Huse `mu=3` showed small initial-state spreads (max `.044` in the audited cells) and the expected size-dependent transition. `mu=1.9` retained invariant ordered basins: initial-state mean spread reached `.943`. Exclude `mu=1.9`; retain `mu=3` only as a single-path secondary benchmark.
- `M=20` remains fixed unless local observability fails. Choose production `T`, `L` and burn only after the scouts; do not substitute computation volume for construct validity.

## Frozen representation contract to write before confirmation

- Keep PC1 as the primary linear coordinate. Fit on representation-development data only; deterministic sign; never choose a component using `Q` or control.
- One diffusion-map sensitivity is allowed: development-centred, unwhitened PCA; target-blind grid selection by connectivity, eigengap and nested-view/bootstrap stability; first nontrivial eigenfunction only; frozen Nyström extension. Mark unavailable if unstable.
- Build a stress-tested SPI core across every path, transition/end point and `M/T` corner. For Kuramoto exclude explicit phase/synchronization SPIs in the primary; all p90 SPIs are sensitivity. For binary Ising, exclude mathematically invalid continuous/tie-free estimators.
- Pearson correlation-of-SPI edge patterns; development-median imputation; no missingness indicators. Primary meta-features are development-centred but not whitened.
- Confirmation eligibility: p99 row missingness `<=1%`, no row `>10%`, missingness not associated with target/control residuals, and full-set coordinate rank agreement `>=.95` with a frozen zero-failure core. Freeze exact gates before new targets are inspected.
- Derive target noise `e_truth` from disjoint future blocks and local-oracle error `e_crop` from physics-only data. Freeze an absolute accuracy margin before SPI confirmation; repeatability is an error floor, not by itself a scientific success threshold.
- Keep physics scout, representation scout, calibration, and untouched confirmation masters disjoint. Split/bootstrap by master; include paired paths and independent-cell samples.
- Negative control: independently circular-shift channels, preserving marginal spectra but destroying collective alignment; apply the frozen representation without refitting.
- Ising representation scout is development-only: 12 paired masters, two paths, three stress controls, `M=20` at `T={500,1000,2000}`, and nested `M={10,32}` brackets at `T=1000`. `M=32` is an upper convergence diagnostic; production remains `M=20`.
- Select a zero-failure SPI core on every `M>=10` stress row. Fit unwhitened PC1 at `M=20,T=1000` with meta-feature SD threshold `.05`. Require exact trajectory nesting, worst-path bootstrap stability, and master-bootstrap/leave-one-out loading and score stability before production.

## Claim ladder

1. **Tracks Q:** eligible frozen coordinate has strong held-out association and calibrated error within a physics-derived margin.
2. **Inference beyond control:** additionally recovers within-control variation, hidden-complement truth and conditional information beyond the control value. This is the minimum for “performs unsupervised data-driven inference of a changing order parameter.”
3. **Path-general coordinate:** calibration transfers and matched-`Q` path differences satisfy a predeclared equivalence margin.
4. **Replicated capability:** analogous success on Miller--Huse and kinetic Ising.

Control-only, local physical oracle, mean `|r|`, raw-correlation PC1 and development-selected individual SPI are mandatory comparators. They need not be worse; superiority is a separate claim.

## Operational state

- Local authoritative branch: `refactor-lagged-warping`; user notebook changes remain untouched. Gadi physics jobs `177032284/177032287` finished successfully. Available allocation was `25.68 KSU`; Scratch inode headroom about `20.25k` at last check.
- Next: run a tiny Ising p90 feature-scout subset, then the 360-dataset target-blind scout. Submit confirmation only if its frozen stability gates pass. Miller--Huse `mu=3` representation work follows as secondary evidence; `mu=1.9` is closed.
