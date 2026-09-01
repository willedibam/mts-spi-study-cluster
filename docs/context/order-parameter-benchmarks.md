# Dynamical order-coordinate benchmarks

## Question and representation policy

Test whether an unlabeled SPI--SPI coordinate changes with an independently defined physical order parameter or dynamical-regime diagnostic as a control is swept.

- Run the complete `configs/pyspi/benchmarked_p90.yaml`; do not intentionally exclude an SPI family. Target-blind development-only validity handling, imputation, variance filtering, centring and reduction are allowed.
- For MPI edge vectors `v_a`, `z_ab=corr(v_a,v_b)` is a correlation-Gram geometry of estimator response profiles over channel pairs. It loses each MPI's affine level/scale and dyad labels, but an informative MPI can affect every pair containing it; the representation is not blind to individual-SPI information.
- Call `q` unsupervised only if every representation and hyperparameter choice is target/control blind. A later `q -> Q` calibration is supervised.

## Kuramoto: full-catalogue retrospective result

- Reconstruction used all 289 p90 SPIs (`41,616` pairs), one Gaussian random-frequency population and the existing `M=20,T=1000,N=256` banks. Development-only 99% validity and SD `>=.05` gates retained 26,367 pairs spanning 244 SPIs; no family was deliberately removed.
- Frozen target-free PC1 explained `.359` of development variance. On 512 paired terminal rows, Spearman with future full-system `R_N` was `-.956` (`95% CI [-.966,-.945]`) overall and `-.628` (`[-.749,-.452]`) within coupling cells. Across-master PC1 variance peaked at reduced coupling `.9925`, near the known boundary. A separate supervised isotonic readout had MAE `.0589` (`[.0510,.0672]`). Mean absolute input correlation was stronger (`.975/.680` overall/within-control).
- Two terminal rows had 25.5% selected-feature missingness; excluding rows above 5% left `|rho|=.955/.629`. This is retrospective because outcomes were disclosed previously, although neither outcomes nor controls fitted the representation.
- The notebook presents this first and keeps the prospective 164-SPI non-phase Gaussian/logistic assay below a divider as historical provenance (`|rho|=.924/.933` overall).
- Natural frequencies `omega_i` are quenched equation parameters; no sampling law is uniquely canonical. Use Gaussian random frequencies as primary. A uniform law changes the transition, and extra logistic/quantile paths distract from the main question.
- Partial observation tested recovery of a hidden global future quantity from a sensor view and avoided self-inclusion. It is meaningful but secondary. A future primary generator should use feasible full observation (`M=N`, likely 20 or 32).
- Evidence: `notebooks/inference/kuramoto-order-parameter-confirmation.ipynb` and `data/order_parameter/kuramoto_full_catalogue_reanalysis/`; historical compact data remain under `kuramoto_final_confirmation_contract/`.

## Miller--Huse: best immediate next system

- Deterministic 2-D chaotic CML: `x'=(1-4g)f_mu(x)+g sum_nn f_mu(x_nn)`, using the original odd piecewise map at `mu=3`.
- It has a coupling-driven macroscopic symmetry-breaking transition. The original exponents were consistent with 2-D Ising behavior (https://doi.org/10.1103/PhysRevE.48.2528), but refined synchronous-update estimates, especially `nu=.887(18)`, suggest a distinct class while not excluding a very slow Ising crossover (https://doi.org/10.1103/PhysRevE.55.2606). Do not call the universality established.
- Use `Q_MH=<|L^-2 sum_r sign(x_r)|>_t`; RMS magnetization, susceptibility and Binder cumulant are sensitivities. The attempted `mu=1.9` path failed start-state convergence and remains excluded.
- The repository exactly matches the original synchronous `mu=3` map. `L={32,64,128}` scouts reproduce the sharp rise near refined `g_c=.20534(2)`, susceptibility peak and start-state agreement. A focused `L=128`, two-million-step future-truth audit passed with explicit uncertainty: p05 effective samples is `63.2`; adjacent 250k-block difference p95 is `.099`; at the hardest `g=.205` cell, median block-mean SE is about `.021` on a total Q range near `.8`. Use the two-million-step scalar and eight block summaries in p90; do not persist redundant future trajectories.
- Here physical `N=L^2`. Full p90 observation at adequate `L=64/128` is infeasible because every MPI scales as `N^2`; dataset parallelism does not remove per-row memory. Primary observation is a frozen nested dispersed sensor set `M={8,16,32}` from one full `L=128` trajectory; local patches and small full lattices are sensitivities.

## Portfolio, rather than a fixed three-system list

1. **Headline candidates:** Miller--Huse and Stuart--Landau have passed their physics gates. The latter's `K=.8`, `gamma=.6,.8,1.0,1.2` path is exactly Fig. 1 of Matthews--Strogatz, not an arbitrary slice: it traverses locking, large oscillations, irregular order-parameter motion and incoherence. It keeps the published `R=|mean z_j|` while avoiding amplitude-death constant channels.
2. **Secondary stress test:** quadratic CML at `eps=.3`, varying `alpha`. Its regimes are literature-grounded, but no accepted scalar order parameter spans the path. It must earn inclusion as recovery of a predeclared regime-coordinate vector, not be presented as canonical scalar-Q recovery.
3. **Best next canonical backup:** periodically driven kinetic Ising, with stochastic spin dynamics and cycle-averaged magnetization as the established dynamic order parameter (https://doi.org/10.1103/PhysRevLett.81.834). This is scientifically distinct from the deleted equilibrium/static Ising assay, but do not implement it without resolving the user's prior instruction to remove Ising-family work.
4. **Deferred:** Vicsek has canonical polarization but requires large physical populations for defensible transition behavior; the contact process has canonical active-site density but an all-zero absorbing phase that degenerates p90 inputs; CGLE and neural-chaos models are useful application bridges but lack an equally clean single observed `Q` or require a modified driven/noisy process.

Application bridges are a separate tier. Sompolinsky--Crisanti--Sommers gives a clean neural fixed-to-chaos transition; Brunel gives noisy irregular E/I spiking; Epileptor/Jansen--Rit gives seizure-like bifurcations; power-grid swing equations give noisy synchronization loss. None is a faithful generic proxy for real data. There is no accepted scalar consciousness order parameter, and no universal clinical seizure order parameter across onset types.

## Analysis contract

1. Reproduce physical behavior without pyspi; establish burn-in, stationarity, finite-size/time convergence and independent seeds.
2. Freeze controls, physical `Q`, observation, development and untouched confirmation masters; then run full p90 on Gadi.
3. Fit feature hygiene and reduction on development rows only. Use PC1 only if target-blind one-dimensionality and bootstrap/leave-group stability pass. Curved geometry can use a frozen out-of-sample PCA(d)-to-manifold map chosen by eigengap/connectivity/stability, never `Q` correlation. Branching/hysteretic planes need at least 2-D or branch-specific coordinates.
4. Evaluate held-control/seed `q--Q` association, within-control association, and a separate supervised decoder only if numerical inference is claimed. Compare raw/simple, control-only and individual-SPI baselines.

For `M,T` transfer, keep physical `N` fixed, take nested channel views and prefixes from many independent maximal trajectories, and apply one common development-fitted mask, scale and embedding. Start with `M={8,16,32}`. Separately fitted/standardized coordinates are not comparable; nested views are paired sensitivities, not independent replicates.

Current sample-length contract is `T={100,500,1000}` for the selected robustness system: fit the primary transform on `T>=500` and apply it to `T=100` as a stress cell. Use paired `T=2000` only to test whether it materially improves feature/coordinate stability. Existing p90 validity gains from 1000 to 2000 are small; precision remains to be measured by paired prefixes.

Stuart--Landau is the selected full `M x T` robustness system. Initial Miller--Huse and quadratic-CML p90 development uses `T=1000` across `M={8,16,32}`; add shorter prefixes only after a useful coordinate is demonstrated.

Claim-bearing p90 sweeps are one-dimensional by default. A cheap two-control physics map may validate an intercept (already done for Stuart--Landau), but do not expand it to the full `control x M x T x instance` design unless no one-dimensional path separates the relevant regimes.

The quadratic-CML two-million-burn gate covers `10 alpha x 3 N x 8 seeds`, plus a dense `N=512,alpha=1.71:0.01:1.80` check. Four consecutive 5k truth blocks are stable away from the reorganization; over the merged gate, pooled physical-diagnostic prefix-error p95 is `.194/.119/.096` at `T=100/500/1000`. All eight `N=512` seeds are low-temporal-entropy patterned through `alpha=.73`; coexistence occurs over `.74-.76`; all eight are high-entropy by `.77`. At `.75`, five seeds are low- and three high-entropy. Do not average the branches into a scalar truth. The p90 gate passed: use fixed `N=512,M={8,16,32}` on `alpha=1.60:0.01:2.00`, with ten anchors for separate `M=N={8,16,32}` finite-size arms. The physical vector is temporal spectral entropy, dynamical spatial-pattern entropy, selected-band power and period-two residual.

## Minimum figures

- One-control sweep: control horizontally; physical `Q` and sign-oriented standardized `q` vertically with independent-seed uncertainty.
- Held-out recovery: `q` versus `Q`, coloured by control; add a monotone calibration only when labelled supervised.
- Two controls: `Q` heatmap/boundaries beside frozen `q` map plus discrepancy; do not replace a plane with one arbitrary path.
- Each system notebook gets one compact orientation panel: exact equation/control/`Q`, representative traces or fields, and an `icefire` MTS heatmap using `src.corpus_visualization.plot_mts_heatmap(..., method="robust")`. This is context, not evidence.
- Preserve explicit one-control examples: Miller--Huse varies `g`; Stuart--Landau fixes `K=.8` and varies `gamma`; quadratic CML fixes `eps=.3` and varies `alpha`. The last crosses documented finite-time pattern-selection/intermittency/turbulence reorganizations, not two numerically precise asymptotic phase boundaries.
