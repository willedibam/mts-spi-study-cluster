# Direct phase-coupling transfer: feasibility protocol

Declared 2026-09-10 before raw feasibility outcomes. Configuration:
`configs/analysis/oscillatory-mechanism-scout-260910.yaml`. This is a model-validity
scout, not a SPI–SPI benchmark. No z extraction, fitting or model-based parameter
selection belongs in the scout. The [scientific framing](spi-representation-scientific-framing.md)
states the claim this would extend.

## One change and its scientific meaning

Replace the four common phase drivers with direct interactions among phase
oscillators. Keep the original amplitude-envelope process, class partitions,
carrier range, phase measurement jitter, gains, additive noise and sensor sampling
laws. Thus this tests a change in the mechanism of phase coordination, not
transfer between completely unrelated physical systems or biological validation.
Both models still belong to a deliberately designed oscillatory family.

The Kuramoto model is an established model of collective phase synchronization;
see [Acebrón et al. 2005](https://doi.org/10.1103/RevModPhys.77.137).
Empirical phase- and amplitude-coupling networks need not coincide; see
[Siems and Siegel 2020](https://doi.org/10.1016/j.neuroimage.2020.116538).
These motivate the model ingredients and comparison of coupling modes, not these
particular parameters, binary classes, or a clinical use of our representation.
This is correspondence of phase-coupling and amplitude-coupling networks,
not within-channel phase–amplitude coupling.

## Equations and parameter rationale

There are 32 oscillators, four phase groups of eight, and four amplitude groups
of eight. Amplitude groups either coincide with phase groups or cross them with
two channels per intersection, exactly as in the existing task. For phase group
g(i), evolve

\[
d\theta_i = \left[2\pi f + {\kappa\over 8}
\sum_{j:g(j)=g(i)}\sin(\theta_j-\theta_i)\right]dt
+\sigma\,dW_i, \qquad \sigma=\sqrt{8\times100}\,s.
\]

All W_i are independent. f is uniform on 6–12 Hz, s on .10–.18, and kappa on
40–80 inverse seconds. No inter-group coupling or frequency heterogeneity is
introduced in this first mechanism test. Antisymmetric coupling cancels in the
unwrapped group-average phase: its noise variance per .01 seconds is exactly
sigma²×.01/8=s², matching the original common-driver increment variance. This
does not imply that the full signals or their coherence distributions match.

For orientation, linearizing around synchronization gives within-group relative
phase variance approximately sigma²(1−1/8)/(2 kappa). The selected range aims at
coherent but noisy groups; this approximation is not a feasibility result. It
also gives kappa×dt≤.08 at dt=.001. Euler–Maruyama is checked against dt=.0005
using the same underlying Brownian increments. Initial phases are independent
uniform angles; discard 10 seconds. Rotation symmetry supplies uniform absolute
phase, but equilibration of relative phases still needs checking.

Retain the original stationary AR(1) amplitude processes and measurement law:

\[
a_i = a_{\rm sd}(\sqrt{.8}\,u_{h(i)}+\sqrt{.2}\,v_i),\qquad
x_i = G_i[\exp(a_i-a_{\rm sd}^2/2)
\cos(\theta_i+b_i+\epsilon_i)+\eta_i].
\]

u and v have stationary unit variance with rho in .94–.98; a_sd is .25–.55.
b_i is a fixed uniform phase offset, epsilon has SD .15–.35, log G_i is uniform
on [−.3,.3], and eta has SD .05–.15. Nuisances and sensor permutation are shared
within each paired scout draw; the only class change is amplitude assignment.
Phase and amplitude processes have independent randomness. Keeping shared
amplitude drives is intentional isolation of the phase-mechanism change.

## Independent functional reference and finite observations

Use 16 paired nuisance draws, with disjoint random streams for parameters, long
reference trajectories and short prediction recordings. Each pair contains both
classes; the 32 class records are not 32 independent nuisance draws. Save seeds,
parameter values, hashes and all gate outcomes. No selection among the 16 draws.

For each class, generate an independent 120-second reference at all 32 channels.
The target property is correspondence of functional phase and envelope networks,
not merely the names of the imposed groups. Compute phase-locking values and
log-envelope correlations using the existing 3–20 Hz Hilbert control, plus the
same matrices from latent phases/log amplitudes as simulator diagnostics only.
Evaluate the reference separately on its two 60-second halves.

For each half, require observed agreement AUROC≥.95, and at least 90% of records
in each class to have Pearson agreement >.5 (aligned) or <.25 (crossed). These
separated intervals are feasibility criteria, not trained class thresholds.
Also require latent within-minus-between-group means ≥.3 for phase and envelope
in every full reference. Report observed contrasts too: Hilbert estimation can
mix amplitude and phase effects even when latent processes are independent.

Generate separate 10-second, 32-channel records using the same nuisance draws.
Use the original nested views: first 16 shuffled sensors/all 1000 samples and
first eight/final 500 samples. Direct observed agreement AUROC must be ≥.90 in
both views. No SPI results are used in any acceptance decision.

For the first four draws/both classes, rerun the finite path at half the step,
sharing fine Brownian increments and all observation randomness. Require maximum
absolute observed agreement change ≤.03 across both views and latent full-matrix
phase edge RMSE≤.02. For the first four draws, repeat the long reference with
kappa=0 and otherwise identical stochastic inputs: require absolute latent phase
within-minus-between contrast ≤.1 in each. This tests that the interaction term,
not shared phase noise or a grouping implementation error, creates coordination.
Report half-wise latent phase contrasts as an equilibration diagnostic; the
half-wise observed gates already require stable functional classification.

All gates must pass. A failed gate is preserved; no automatic parameter search.
Numerical or implementation failures may be repaired with explicit provenance;
scientific parameter changes require a new protocol and fresh scout seeds.

## Conditional next step

A pass justifies freezing one independent 200-master/400-view target test and
running affordable p90 extraction on Gadi. Reuse the existing source fits and
10/20/40 budgets, including PCA/ridge, PLS, marginal/shape/fusion baselines, learned
SPI pooling, raw baselines/encoders and the direct specialist. No target training,
calibration or selection; distinguish AUROC from fixed-threshold accuracy and
Brier. Confirmation generation and source-model references will be frozen in a
separate configuration before those data are generated. A scout pass is not a
positive result for z, and a new-mechanism result is not automatically an
application or cross-task representation-learning claim.

## Scout outcome and confirmation decision

Protocol frozen at `eb2a1d2`; implementation frozen at `dfa2838` before the scout.
All eight gates pass on all 16 paired nuisance draws. Both reference-half AUROCs
and both finite-view AUROCs are 1.0; all class-interval checks pass. Minimum latent
phase/envelope within-minus-between contrasts are .6937/.7668. Maximum step-size
agreement change is .000780, phase-edge RMSE .001626. With coupling removed, all
four absolute phase contrasts are below .00220. Maximum absolute change between
the two half-wise latent phase contrasts is .0604. These establish feasibility,
not SPI–SPI performance. Five generator/invariance tests pass, including exact
uncoupled integration and cancellation of coupling in each group-average phase.

Authoritative scout: `results/oscillatory_mechanism_scout_260910/report.json`, SHA
`e146c14b5db0b42b10f1b9ce42329e678d44dbb48a022a4957bf608d2d62bfda`.
Raw finite records and reference matrices are saved with hashes; long references
are reproducible from stored nuisance parameters and independent random streams.

Proceed with `configs/analysis/oscillatory-mechanism-transfer-260910.yaml`: 100
independent masters/class, 200 total and 400 nested views, new master seed
260910211, no training records. Both nuisance draws and paths are independent
across confirmation classes/records, unlike paired scout draws. Preserve the
predeclared parameter ranges and equations. The configuration binds a manifest
of all 99 source fits, their prediction hashes and neural checkpoint hashes;
the evaluator verifies these before use. Existing source preprocessing/settings
remain fixed; original source predictions must replay before target statistical
prediction. M16/T1000 is primary, M8/T500 secondary. No target calibration,
retraining, selected test subsets or new head/grid. No changes to the older
completed outcomes. Stop after this test and its verification; any further
extension needs a result-based scientific rationale.

This remains a controlled mechanism change within oscillatory co-organization.
The strong-coupling limit approaches common phase motion, and amplitude dynamics
are deliberately retained. A positive result would not establish transfer
between arbitrary unrelated dynamical systems.

Execution: confirmation frozen at `0f2a227`, extraction configuration at
`1ab1595`. All 400 views and input artifact hashes pass local verification; four
boundary-index master realizations replay exactly. Four finite scout records and
their four independent long reference trajectories also replay exactly. Target
manifest SHA `34e901ecea7aacbaa918a493a4ff5d87b7d2c1829238ec3c05fe4594c3a023d7`;
views SHA `0b286040216dc6176ef57ffa2b6015a62e2d7aa9a50b12c81a0b99eb409f207e`.
Gadi validates the same archive/400 records. Smoke `178575173.gadi-pbs` finished
with exit 0 in 9:02 (4.12 GiB peak); gate `178575197.gadi-pbs` passed all 48 audits
in 6:25 (39.04 GiB peak), reusing smoke outputs. This supports the planned
192-core/768-GB homogeneous farms: M16 `178575765.gadi-pbs`, M8
`178575775.gadi-pbs`, each 200 selected views/24 cached, 1200-second task timeout
and 30-minute allocation ceiling. Dependent audit/bank job `178575814.gadi-pbs`
completed with exit 0 in 3:03 (3.70 GiB peak). Both production jobs finished
with exit 0: M16 in 7:13 (174.09 GiB peak), M8 in 1:53 (130.93 GiB peak); all
400 outputs pass audit. Banks contain 263–282 valid SPIs per recording, with
24.659 summed per-record computation hours. Source extraction code remains `1ab1595`,
pyspi `65317c9`; no completed records are recomputed. State and continuation instructions:
`results/oscillatory_mechanism_transfer_260910/cluster-state.json`. The existing
follow-up monitor is active every 15 minutes and pauses at verified completion
or the declared 2026-09-10 05:30 UTC deadline. No new z outcome has been inspected.

The 132-MB feature/normalized-edge bank download is in progress (session 87010);
partial NPZ files must not be used. Eight sampled MPIs are already downloaded
and hash verified. Frozen-model prediction and independent feature/edge replay
await the complete banks; no extraction or model fitting is to be duplicated.
