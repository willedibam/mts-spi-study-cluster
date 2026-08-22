# Canonical order-parameter benchmarks

## Scientific status

- Only one canonical order parameter has yet received a full SPI--SPI evaluation: Kuramoto phase coherence `R_N`. The quadratic Kaneko-CML `Q_sel`, entropy, period-2 score and Lyapunov diagnostics are not canonical order parameters.
- Completed Kuramoto run: 880/880 datasets, `M=20,T=1000,N=256`; Gaussian-fitted PC1 had held-out overall/within-control Spearman `-.945/-.689` on Gaussian and `-.915/-.596` on logistic data. The physics and leakage controls were sound.
- That result is not claim-bearing under its frozen rule. Four evaluation rows had structured high-synchrony missingness and violated the maximum-row gate; cross-path MAE `.0789` was worse than mean absolute input correlation `.0583` and a selected individual SPI `.0676`. Existing data are now development-only.
- Interpret it as: a frozen unsupervised coordinate strongly associated with future global `R`, including fixed-control variation, but failed feature eligibility. PC1 was a legitimate predeclared linear baseline, not an empirically optimal coordinate (explained variance `.381`).
- Sine-circle `k=3.1` remains rejected: its printed modulo recurrence does not preserve the claimed absorbing set and direct replication failed. Quadratic Kaneko remains exploratory regime discovery.

Primary evidence: `notebooks/embeddings/kuramoto-order-parameter-benchmark.ipynb` and `data/order_parameter/kuramoto_order_benchmark/`. Sources: Kuramoto <https://doi.org/10.1103/RevModPhys.77.137>; Miller--Huse <https://doi.org/10.1103/PhysRevE.48.2528>; finite MH analysis <https://doi.org/10.1103/PhysRevE.55.2606>; Yang magnetization <https://doi.org/10.1103/PhysRev.85.808>.

## Prospective benchmarks

### Miller--Huse (primary new benchmark)

- Dynamics: `x'=(1-4g)f_mu(x)+g sum_nn f_mu(x_nn)` on a periodic 2-D lattice. Primary path is the original `mu=3`; `mu=1.9` is exploratory until long-run coarsening/pinning checks pass.
- Define `s_r=+1` for `x_r>=0`, otherwise `-1`; `m_s=L^-2 sum_r s_r`. The canonical finite-size target is `Q_MH=<|m_s|>_future`; `sqrt(<m_s^2>)` is a second-moment sensitivity. Never average signed `m_s` across symmetry flips.
- SPI sees only a contiguous `4x5` (`M=20`) field patch. The target uses a disjoint future full lattice; patch-excluded future magnetization is the anti-self-inclusion sensitivity.
- Generator now supports rectangular patches, general `mu`, future truth, hidden complement, hot/cold/random starts and compact storage. Full field movies are off by default.

### Kinetic Ising (independent replication)

- Zero-field anisotropic square Ising, `H=-Jx sum_x sisj-Jy sum_y sisj`, equilibrated with Wolff clusters then observed under checkerboard heat-bath updates. Fully simultaneous updates are not substituted.
- Primary target is `Q_Ising=<|m|>_future`; RMS magnetization is a sensitivity. Use isotropic `(Jx,Jy)=(1,1)` and anisotropic `(1,.5)` paths matched by `u=sinh(2 beta Jx)sinh(2 beta Jy)`.
- The exact critical line is `u=1`; thermodynamic `m=0` for `u<=1` and `(1-u^-2)^(1/8)` otherwise. Equal `u` therefore gives the same exact macroscopic magnetization but different spatial microstructure.
- SPI sees a `4x5` binary patch. Continuous estimators legitimately fail on ties; the target-blind stress scout must remove them rather than jittering the data.

## Physics-first contract

- Smoke configs and the full `benchmarked_p90.yaml` pipeline pass locally. Smoke contrasts are not physics validation.
- Dataset-scoped seed derivation is clone-invariant: absolute output paths were removed from the seed payload after the first Gadi smoke exposed local/cluster trajectory mismatch. Every dataset still records its resolved seed.
- Primary scouts: `configs/scout/miller-huse-physics-primary.yaml` (480 tasks) and `kinetic-ising-physics-primary.yaml` (624 tasks), run by `scripts/spin_order_parameter_scout.py`. They store small JSON parts; aggregation produces compressed numeric arrays and a concise diagnostic figure.
- Before pyspi require: future-block repeatability small relative to `Q` range; hot/cold/random agreement; expected Binder/susceptibility/finite-size behavior; and useful held-out local-to-global observability. Add `L={32,64,128}` and burn/equilibration audits after the coarse scout identifies the transition region.
- Miller--Huse starts at `L=64`, burn `200k`, exposed `T=4000`, four future blocks of 4000, 24 seeds/cell. Kinetic Ising starts at `L=64`, 200 Wolff-equivalent equilibration sweeps, exposed `T=4000`, four future blocks of 5000, 24 seeds/cell. These are hypotheses to validate, not fixed production values.
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

## Claim ladder

1. **Tracks Q:** eligible frozen coordinate has strong held-out association and calibrated error within a physics-derived margin.
2. **Inference beyond control:** additionally recovers within-control variation, hidden-complement truth and conditional information beyond the control value. This is the minimum for “performs unsupervised data-driven inference of a changing order parameter.”
3. **Path-general coordinate:** calibration transfers and matched-`Q` path differences satisfy a predeclared equivalence margin.
4. **Replicated capability:** analogous success on Miller--Huse and kinetic Ising.

Control-only, local physical oracle, mean `|r|`, raw-correlation PC1 and development-selected individual SPI are mandatory comparators. They need not be worse; superiority is a separate claim.

## Operational state

- Local authoritative branch: `refactor-lagged-warping`; user notebook changes remain untouched. Gadi repositories were clean at main `0c2ff27`, pyspi-v3 `65317c9`; allocation available `25.71 KSU`, Scratch inode headroom about `20.25k` at last check.
- Next: commit only generator/scout/test/context changes; fast-forward Gadi; run two-dataset p90 smoke; submit physics scouts; audit results before any representation or production farm.
