# Canonical order-parameter benchmarks

## Scientific status

- Only one canonical order parameter has received a full SPI--SPI evaluation: finite-population Kuramoto phase coherence `R_N`. The quadratic Kaneko-CML `Q_sel`, entropy, period-2 score and Lyapunov diagnostics are not canonical order parameters.
- The terminal prospective Kuramoto bank passed every frozen gate: `M=20,T=1000,N=256`, 1,536 rows, 164 non-phase SPIs and 12,737 retained SPI--SPI features. Target-free PC1 recovered future global `R_N` on untouched random-frequency controls and seeds under Gaussian/logistic laws (overall Spearman `-.924/-.933`; within-control `-.523/-.618`).
- A separately supervised isotonic readout reached MAE `.080/.069`. Mean absolute input correlation and analytic phase coherence associated more strongly with `R_N`; coupling-only calibrated MAE was better on Gaussian and tied on logistic. The result establishes capability, not superiority, uniqueness or efficiency.
- The earlier 880-row result is development-only. Its strong association is not claim-bearing because four evaluation rows violated the frozen missingness gate.
- Sine-circle `k=3.1` remains rejected: its printed modulo recurrence does not preserve the claimed absorbing set and direct replication failed. Quadratic Kaneko remains exploratory regime discovery.

Primary evidence: `notebooks/embeddings/kuramoto-order-parameter-confirmation.ipynb` and the compact numeric contract in `data/order_parameter/kuramoto_final_confirmation_contract/`; the large feature bank remains on Gadi. Sources: Kuramoto <https://doi.org/10.1103/RevModPhys.77.137>; Miller--Huse <https://doi.org/10.1103/PhysRevE.48.2528>; finite MH analysis <https://doi.org/10.1103/PhysRevE.55.2606>; Yang magnetization <https://doi.org/10.1103/PhysRev.85.808>.

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

The frozen `M=20,T=1000` gate failed target-blindly. All 360 rows completed and 217/289 SPIs formed a zero-failure core; PC1 explained `.619` and its master-bootstrap loading/score lower bounds were `.988/.997`. `M=32` passed, but temporal geometry did not: lower-bound worst-path geometry was `.712` at `T=500` and `.711` at `T=2000`; the `T=2000` coordinate lower bound was `.828`. This is a clean null for the `T=1000` candidate and may not be relabelled a pass.

### Frozen target-blind `T=2000` refinement

- Existing `T=2000` rows are development data for a new candidate; no order-parameter values are read. Fit PC1 on their frozen 217-SPI core with the same `.05` meta-feature SD threshold.
- Validation uses 16 new masters at the same two paths and three stress controls. Each master supplies two `M=20,T=2000` blocks from the same equilibrated trajectory: sweeps 1--2000 and 4001--6000, leaving a 2000-sweep gap. PySPI is computed separately per block; a single `T=4000` feature vector is not a reproducibility test.
- Keep the original worst-path lower-bootstrap gates unchanged: coordinate `.90`, geometry `.85`, median row correlation `.90`. Also require `.90` for development bootstrap/LOO loading and coordinate stability and for validation-block loading/coordinate agreement.
- Keep the already frozen missingness gates: p99 `<=.01`, maximum `<=.10`, and rank agreement `>=.95` with the validation zero-failure subset.
- Failure is the `T=2000` representation null. Success permits untouched order-parameter confirmation; it does not itself evidence magnetization recovery.

The untouched `T=2000` validation also failed. All 192/192 datasets completed, the frozen 217-SPI core had zero missingness, and validation PC loadings were stable (minimum cosine `.989`; between-block cosine `.996`). The median row-correlation lower bound passed (`.938 >= .90`), but the worst-path lower bounds for direct coordinate agreement (`.821 < .90`) and pairwise geometry (`.655 < .85`) failed. No magnetization/order-parameter values were read. This closes Ising as a target-blind representation null under the frozen contract; confirmation was not launched.

## Claim ladder

1. **Tracks Q:** eligible frozen coordinate has strong held-out cell-level association with independently estimated finite-size ensemble `Q_L`.
2. **Infers a changing order coordinate:** target-blind PC1 plus calibration held out by both control value and physical path predicts `Q_L` within a physics-derived margin. This supports the requested phrase at the ensemble-coordinate level, not realization-specific inference.
3. **Path-general coordinate:** calibration transfers and matched-`Q` path differences satisfy a frozen finite-size tolerance (`.04` for the Ising design).
4. **Replicated capability:** analogous success on another canonical system.

For ensemble order parameters, replicates in one control cell share one target. Analyze cell means, jointly bootstrap independent truth chains and MTS masters, and treat the number of cells as the effective sample size. Do not claim within-control recovery. Control-only, local physical oracle, mean `|r|`, raw-correlation PC1 and development-selected individual SPI remain mandatory comparators; they need not be worse.

The independent Ising truth bank uses 24 Wolff-equilibrated chains per path/control cell and 80k hidden heat-bath steps. All 624 chains completed; maximum cell SE across independent chain means was `.0056`, and matched-path gaps were mean `.0037`, max `.0128`. SE is never computed from raw correlated time points.

### Frozen Ising confirmation margins

- The physics-only local oracle is the cell mean of patch RMS magnetization, calibrated to independent cell `Q_L` by isotonic regression. It is model-specific and therefore a comparator, not part of SPI--SPI.
- Calibration uses isotropic controls `u={.75,.92,.98,1.005,1.025,1.1,1.4}`. Held-control evaluation uses the other six isotropic controls; held-path evaluation uses the anisotropic path. These sets are frozen before exposing an Ising SPI coordinate to `Q_L`.
- Across the 26 truth-bank cells, the local oracle has Spearman `.993`. A 5,000-draw within-cell chain bootstrap gave 95th-percentile MAE `.051` for held isotropic controls, `.045` for the complete held path, and `.059` where both path and control were held out. The bootstrap already propagates truth-chain uncertainty.
- Freeze `.06` as the stringent absolute numerical-recovery margin. Freeze absolute Spearman lower confidence bound `.80` as the separate tracking gate. A result may track the order parameter without passing numerical recovery.
- Freeze `.04` as the matched-`u` path-equivalence tolerance, conservatively above the observed finite-size path gap (`.0128`) and the earlier maximum convergence-audit start spread (`.0282`).
- These margins do not require SPI--SPI to beat the purpose-built local oracle or other simple baselines. Any advantage claim requires a paired uncertainty interval and is logically separate from recovery.

## Operational state

- Local authoritative branch: `refactor-lagged-warping`; user notebook changes remain untouched. The terminal Kuramoto confirmation is complete and passed. Its paired circular-shift sensitivity is the remaining non-gating computation. The compact result notebook reads only small numeric contract artifacts; large feature banks remain on Gadi.
- Ising confirmation is ineligible after two target-blind representation nulls. Miller--Huse `mu=3` remains a secondary physics-qualified benchmark; `mu=1.9` is closed.

### Prospective Kuramoto confirmation

- Treat all 880 existing Kuramoto rows as disclosed development data: their outcomes have been seen, although the new representation fit reads no target or control values. The confirmation bank is wholly new.
- Freeze the exact 197-SPI non-phase core whose 19,306 pair features were finite across every development row. Equal-weight, unwhitened PC1 uses only features with development SD `>=.05`; its center, loading, ordered features and hashes are immutable.
- Confirmation uses `M=20,T=1000,N=256`, clean `cos(theta)` observations and a disjoint `T=1000` full-system future target. Twelve reduced-coupling midpoints absent from development are crossed with Gaussian/logistic frequency laws: 32 paired masters plus 8 independent-cell draws per path. Eight regular-frequency masters per path are an unseen finite-population sensitivity.
- Before reading confirmation `R`, every frozen pair feature and PC1 score must be finite. Primary tracking gates on each random-frequency path are clustered 95% lower bounds `|rho_S|>=.70` overall and `>=.30` within coupling, repeated against the hidden-complement target.
- A Gaussian-development isotonic `q -> R` map is frozen only after PC1; it is a supervised readout of an unsupervised representation. Numerical inference additionally requires each path's bootstrap MAE upper bound `<=.10` and a paired MAE improvement over the frozen intercept. Kappa and simple input statistics are comparators, not advantage gates.
- Diffusion maps are a non-rescuing sensitivity: development-centred unwhitened PCA (first dimension reaching 90% variance, capped at 20), alpha-1 diffusion map with `floor(sqrt(n))` neighbours and median kNN bandwidth, first nontrivial eigenfunction, frozen Nyström extension. It is reported unavailable unless target-free bootstrap rank stability is `>=.90`.
- A preselected negative-control subset uses paired masters `0..7` on every new control/path. Each observed channel is circularly shifted by an independent, deterministic nonzero offset before p90 computation; no target is copied or read. Loss of association supports collective temporal alignment, but retained association is not an eligibility failure because channel marginals can also carry synchronization information.

The target-free freeze passed before confirmation generation. PC1 retained 18,513/19,306 meta-features and explained `.369` of development variance. Cluster-bootstrap loading/coordinate lower bounds were `.994/.999`; worst leave-path/design/control values were `.981/.997`. The diffusion sensitivity was also available (bootstrap coordinate lower bound `.997`; 20-component capped PCA retained `.855` variance). The frozen representation and readout hashes are recorded in `data/order_parameter/kuramoto_confirmation_contract/`.

The first prospective bank is an eligibility null, not an order-parameter result. All 1,152 p90 datasets completed, but the frozen analysis exited before reading `R_N`: three Gaussian-random rows produced non-finite features, with maximum selected-feature missingness `.292`. The failure implicated LCSS and directed multitaper families; all logistic and regular-frequency rows were complete. `confirmation_eligibility.json` records `outcomes_read=false`; this bank's targets remain permanently sealed.

Exactly one target-blind assay redesign is permitted. Select every non-explicit-phase SPI that is finite and nonconstant on every one of the 880 disclosed plus 1,152 eligibility-null `X` rows, then retain all SPI pairs before the unchanged SD `>=.05` PC1 filter. The diagnostic found 164 old-core SPIs with zero failures on the null bank, above the frozen minimum of 100. Require zero development missingness, bootstrap loading/coordinate lower bounds `>=.95`, and worst leave-source/path/control values `>=.90`. The failed bank supplies no outcomes or control-association criterion; only disclosed-old Gaussian outcomes may calibrate the later isotonic readout.

If those target-free gates pass, the terminal bank uses 16 entirely new, interleaved reduced couplings, both Gaussian and logistic laws, random paired and independent-cell designs, and regular-frequency sensitivity (`M=20,T=1000,N=256`; 1,536 datasets). Eligibility again requires every frozen meta-feature finite before `R_N` is opened; outcome gates are unchanged. Failure ends the Kuramoto confirmation sequence—no third filtering cycle. A pass supports the narrower statement that a stability-filtered, non-phase SPI--SPI representation recovers the canonical finite-size Kuramoto order parameter on untouched paths and controls.

### Terminal Kuramoto result

- The redesign retained 164 non-phase SPIs, all 13,366 SPI pairs, and 12,737 pairs after the unchanged SD filter. All were finite across 2,032 target-free development rows. PC1 explained `.361`; bootstrap loading/coordinate lower bounds were `.991/.998`, and worst leave-source/path/control values were `.969/.993`. Diffusion maps passed independently and agreed with final PC1 at Spearman `.981`.
- The first terminal-analysis launch stopped before target access because the two smoke rows recorded an earlier clean commit than the 1,534 production rows. No provenance was edited: those exact deterministic rows were deleted and regenerated at the production commit. This was an operational provenance correction, not another representation or scientific redesign.
- All 1,536 final rows passed strict zero-missingness eligibility before `R_N` was opened. On paired random-frequency data, PC1 versus future full-system `R_N` was Spearman `-.924` (Gaussian) and `-.933` (logistic); clustered 95% absolute-correlation lower bounds were `.904/.917`. Within-coupling correlations were `-.523/-.618`, with lower bounds `.362/.463`. The hidden-complement future target also passed every frozen gate.
- The disclosed-Gaussian isotonic readout transferred with MAE `.080` (95% upper `.092`) on Gaussian and `.069` (upper `.079`) on logistic data, and beat the frozen intercept on both. This numerical readout is supervised; PC1 fitting and selection are not.
- SPI--SPI is not the best scalar estimator. Mean absolute input correlation reached overall Spearman `.975/.977` and within-coupling `.680/.705`; analytic phase coherence also exceeded PC1. The coupling-only calibrated MAE was `.057/.069`, versus PC1 `.080/.069`. Therefore make no accuracy or superiority claim.
- Independent-cell overall correlations remained strong (`-.915/-.909`). Regular-frequency overall correlations were `-.893/-.891`, but their within-coupling intervals crossed zero. The result establishes random finite-population and cross-frequency-law recovery of the macroscopic trend; it does not establish recovery of fixed-coupling fluctuations under deterministic frequency quantiles.
- Both frequency laws were present during target-free representation development. Logistic targets were unseen by the readout, but this is not representation transfer to an unseen physical path or a matched-`R` path-collapse test. The independent-cell Gaussian within-coupling lower bound also fell below the primary threshold.
- Independent channel shifts attenuated overall absolute association from `.936` to `.589` (Gaussian) and `.948` to `.438` (logistic); the clustered difference intervals excluded zero. Within-coupling attenuation was conclusive only for logistic. Shifting also induced maximum selected-feature missingness `.136`, so this supports sensitivity to cross-channel temporal alignment but does not isolate it mechanistically from estimator failure and frozen imputation.
- The analysis wrote and passed eligibility before loading any target, as its control flow records. The completed run then updated the same eligibility JSON with `outcomes_read=true`; it did not retain an immutable pre-read copy. This is a reporting limitation, not evidence of target leakage. Future runs now create an exclusive immutable `confirmation_eligibility_pre_read.json` before target access.

Defensible wording: **“In a finite-`N` Kuramoto benchmark, a prospectively frozen, stability-filtered non-phase SPI--SPI PC1 learned without coupling or order-parameter labels recovered, up to a monotone transformation, changes in the canonical phase-coherence order parameter from partial observations on untouched controls and random-frequency realizations under Gaussian and logistic frequency laws.”** A separately supervised readout estimated numerical `R_N`. This is a favorable proof of capability in one canonical system, not universal order-parameter discovery, unseen-path generalization, instantaneous tracking or an advantage claim.
