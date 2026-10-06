# 2:1 cross-frequency locking: protocol fixed before p90

## Question

Does an unsupervised SPI–SPI PC1 register a change in dependence type that mean absolute Pearson cannot, and does it do so more cleanly than the unsupervised per-SPI-mean PC1? The target is a transition where linear dependence is unchanged and nonlinear dependence switches on. Supervised mean readouts are reported as information ceilings; no claim that the mean vector lacks the information is intended.

## System

Microscopic model: [Komarov and Pikovsky, arXiv:1502.06193](https://arxiv.org/abs/1502.06193) (PRE 92, 012906), Eq. (10), with zero phase shifts and equal couplings. Slow community A (frequency 1) and fast community B (frequency 2 + δ) are each all-to-all coupled with strength ε and share the resonant cross-coupling γ: A feels sin(ψ − 2φ), B feels sin(2φ − ψ). Implementation and reproduction: `scripts/cross_frequency_locking.py`; tests `tests/test_cross_frequency_locking.py` check the mean-field velocity against the published pairwise sums.

Departures from the paper, chosen here: narrow Gaussian-quantile frequencies (SD .05) instead of unit-width Lorentzians; independent phase noise σ = .1; eight oscillators per community; and a third community C (frequency 1.37, internal coupling only). C is a construction, not part of the published model. It supplies channel pairs that stay unrelated. Without it every SPI separates the same two edge classes and z, which ignores each SPI's level and scale, cannot change: a two-community scratch version kept z(Pearson, MI) near .98 through locking.

Fixed parameters: δ = .3, ε = .5, M = N = 24 (channels A 0–7, B 8–15, C 16–23), sin(phase) observed every .5 time units, T = 1000, z-scored per channel. Keep ε at least about twice γ: the paper shows the second-harmonic drive can split the slow community into two antiphase branches, and a coupling phase shift near π produces chaos and unlocking.

## Control, truth and boundary

Control: γ ∈ [0, .20] in steps of .01 (21 values); only the cross-coupling changes. Truth Q: the 2:1 locking index |⟨exp i(Ψ_B − 2Θ_A)⟩| of the collective phases on a disjoint future window of 1000 time units; the phase-slip rate is the companion diagnostic. For coherent communities the slow phase obeys dΦ/dt = δ − γ(X₂ + 2Y) sin Φ, so locking sets in at γ_c = δ/(X₂ + 2Y) ≈ .102. This reduction is derived here, not taken from the paper, whose published diagrams (Figs. 2–4) chart other transitions. Noise-free, Q = (δ − √(δ² − a²))/a with a = γ(X₂ + 2Y) below the boundary and 1 above: a ramp ending in a square-root corner, not a flat–jump–flat step.

## Design and readouts

336 records: 21 controls × 16 seeds, an independent realization per control and seed (`SeedSequence([261085, n, control index, seed])`). Seeds 0–7 fit every target-blind PC1 and the supervised ceilings; seeds 8–15 evaluate. Full `configs/pyspi/benchmarked_p90.yaml`, estimator seed 261086.

Unsupervised readouts: mean absolute Pearson (first gate), mean signed Pearson, standardized per-SPI-mean PC1 (second gate), distribution PC1, centered z-PC1 (primary) and standardized z-PC1 (sensitivity). Ceilings: training-selected single mean and full-mean ridge/RBF. PC signs are oriented by the control on fit rows only. Records above 5% selected-feature missingness are excluded from every paired comparison; at least four eligible held records per control are required.

Scores on held records, fixed before outcomes: step contrast (change in the held mean across γ = .08–.12 divided by the pooled held SD at those two controls), step share of the total end-to-end change, post-locking drift (change from .12 to .20 over the total), steepest interval, and pooled |Spearman| with Q for reference only. Pooled Spearman is not the criterion: a .002 monotone drift in mean |r| earned .73 in the scratch run.

## Scratch evidence motivating the run

Seven cheap probes, M = 24, four fit and four held seeds, same parameters (seeds shared across controls in that run): cross-community |r| stayed at .003–.004; mean |r| moved .302 → .304 over the whole sweep (step contrast 1); mean-SPI PC1 was noisy and kept drifting (contrast 2); centered and standardized z-PC1 stepped at the boundary and plateaued (contrast 22 and 17); mean KSG MI alone stepped strongly (contrast 45). Community sizes 8, 16 and 64 located the boundary between .10 and .115. These are exploratory probes that motivated the protocol, not p90 evidence.

## Known risks

Within-community channels are nearly collinear (|r| ≈ .98, correlation-matrix condition number 1200–1800), so precision-type SPIs may be unstable; the missingness gate decides. With 289 SPIs many nonlinear means will step, so mean-SPI PC1 may do better than with seven probes. z-PC1 localizes the boundary but under-tracks the ramp in Q before it.

## First full-p90 result, 6 October

All 336 records completed with the 289-SPI catalogue; all 168 held records pass the missingness gate with zero selected-feature missingness. Per record, 234–282 SPIs are fully finite; 211 means and 58% of z features are finite in every record. Eight independent MPI replays reproduce means exactly and z within 3e-8. Reference boundary from measured coherences: γ_c = .1017; Q's steepest interval is .10–.11.

| Held readout | Step contrast | Step share | Post-locking drift | Steepest interval | Pooled \|ρ\| with Q |
| --- | ---: | ---: | ---: | --- | ---: |
| Mean absolute Pearson | 2.2 | .40 | .61 | .06–.07 | .73 |
| Mean signed Pearson | .7 | .58 | .53 | .19–.20 | .15 |
| Mean-SPI PC1 | 10.2 | .71 | .33 | .10–.11 | .86 |
| Distribution PC1 | 11.2 | .71 | .36 | .10–.11 | .84 |
| Centered z-PC1 (primary) | 8.4 | .92 | .20 | .11–.12 | .83 |
| Standardized z-PC1 | 12.1 | .82 | .15 | .10–.11 | .89 |
| Selected mean (`si_kernel_W-0.5_k-1`), ceiling | 11.2 | .61 | .25 | .10–.11 | .98 |
| Full-mean ridge / RBF, ceilings | 20.5 / 19.5 | .55 / .57 | .02 / .00 | .10–.11 | .93 / .93 |

**First gate passes.** Mean absolute Pearson moves .302 → .305 over the whole sweep with no feature at the boundary; cross-community |r| stays near .003. Every catalogue-based unsupervised coordinate steps at the locking boundary.

**Second gate is not passed.** With the full catalogue the mean-SPI PC1 steps at the correct interval with a contrast comparable to the z coordinates; the large gap seen with seven probes did not survive. The z coordinates concentrate more of their change at the boundary and drift less afterwards, and standardized z has the highest contrast, but the primary centered z-PC1 has the lowest contrast of the four and its steepest interval is one grid step late. All four unsupervised PC1s dip in the wrong direction over γ ≈ .02–.07 before stepping (centered z most strongly), so none tracks the ramp in Q below the boundary. Supervised mean readouts track Q closely; the information is in the mean vector.

Scope: one size (M = N = 24), one sweep, eight fit and eight held seeds, exploratory. The claim supported is unsupervised detection and localization of a dependence-type change that mean correlation cannot register, not an advantage over the mean-SPI vector.

Provenance: source commit `c144491` at `operations/sources/cross-frequency-c144491`, pyspi `65317c9`, estimator seed 261086, archive SHA `06e29ec5…96dcba`. Jobs 180570622 (smoke, 2), 180570654 (node, 48), 180572852 (remaining 286), 180572918 (analysis); all exit 0, 750–840 s per record. Remote data `/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/cross-frequency-locking-261006`. Local outputs `results/order-parameter-inference/cross-frequency-locking-261006/`; `gadi-original/` keeps the cluster metrics, which a local rerun reproduces within 8e-6. The cluster figure had a title bug (`rows.T` is a transpose), fixed afterwards without changing any computation. Reproduce with `python -m scripts.cross_frequency_locking prepare`, the external-corpus farm, then `extract` and `analyze`.

## Sensor-noise variant, fixed before its p90 outcomes

Why the mean-SPI PC1 succeeded above: 121 of the 211 usable SPI means step at the boundary (contrast above 3), so the mean vector's dominant variance is the transition. Part of that response is a strength effect in disguise: locking tightens the communities (within-community |r| .979 → .986), and statistics that diverge near perfect correlation amplify it. The seven-probe scratch set contained one responsive mean in seven, which is why its mean-PC1 failed.

Variant (`scripts/cross_frequency_locking_snr.py`): identical dynamics and truth, each channel observed through independent white noise of SD η in signal-SD units. Arm `fixed-noise`: η = .5 in every recording, which removes the near-collinearity. Arm `random-noise`: η ~ U[.3, .9] drawn once per recording, independently of γ, as recording quality varies across sessions and subjects. Hypothesis: the nuisance moves every dependence estimate together, so mean |r| and mean-SPI PC1 follow η; z discards each SPI's level and scale across pairs, so z-PC1 follows the locking index. 672 records (2 arms × 21 controls × 16 fresh seeds, 8 fit / 8 held), M = N = 24, T = 1000, estimator seed 261092; arms analysed separately.

Declared: centered z-PC1 is primary, standardized z-PC1 a sensitivity; headline scores are held |Spearman| with Q and, in the random arm, with η; step scores are secondary. Supervised mean readouts stay as information ceilings and are expected to recover Q in both arms. A scratch run with eight cheap probes (4 fit / 4 held seeds) gave, in the random arm, |ρ| with Q of .01 (mean |r|), .06 (mean-probe PC1) and .91 (centered z-PC1), with z-PC1's |ρ| with η at .13; in the fixed arm .41, .81 and .90. The earlier probe-to-p90 discrepancy means this is motivation, not evidence.

### Sensor-noise variant: full-p90 result, 6 October

All 672 records completed; all 168 held records per arm pass the gate with zero selected-feature missingness; eight MPI replays reproduce means exactly and z within 3e-8. Held |Spearman|, 168 records per arm:

| Readout | Fixed noise: with Q | Random noise: with Q | Random noise: with η |
| --- | ---: | ---: | ---: |
| Mean absolute Pearson | .38 | .03 | 1.00 |
| Mean-SPI PC1 | .82 | .02 | 1.00 |
| Distribution PC1 | .82 | .04 | 1.00 |
| Centered z-PC1 (primary) | .88 | .87 | .02 |
| Standardized z-PC1 | .82 | .68 | .69 |
| Selected mean / full-mean ridge, ceilings | .97 / .92 | .79 / .89 | .22 / .01 |

**Recording-specific noise: both baselines fail and the primary SPI–SPI coordinate does not.** Mean |r| and the unsupervised mean-SPI and distribution PC1s follow the noise level and carry no trace of the transition. Centered z-PC1 tracks Q (seed-bootstrap 95% interval .85–.90; z minus mean-SPI PC1 .60–.87), is unrelated to η, steps at the correct interval .10–.11 (step contrast 12.8) and follows part of the ramp below the boundary (|ρ| .51 for γ ≤ .10, against .10 for mean-SPI PC1). Standardized z-PC1 is contaminated by η; the scaling sensitivity is real and must be reported with the result. The supervised full-mean ridge recovers Q, so the information is present in the mean vector; what fails is unsupervised access to it.

**The advantage is an ordering of variance, not an invariance (added after the run; same features).** Both representations contain both factors. In the random arm the standardized mean vector puts η first (67% of variance; |ρ| with η 1.00) and the transition second (8%; |ρ| with Q .76). Centered z puts the transition first (38%; .87) and η second (23%; |ρ| with η .97). Standardized z has near-equal leading eigenvalues (25% and 23%), so its PC1 and PC2 each mix the two, which is the contamination reported above. Consequences: z attenuates the nuisance relative to the transition but does not remove it; a wide enough noise range would put η first in z as well; and an analyst who inspects the second mean-SPI component recovers the transition. The defensible claim is that the default unsupervised coordinate is the transition for z and the nuisance for the means. `analysis.json` now stores the three leading components per representation and arm.

**Homogeneous noise:** mean |r| fails (flat at .250); mean-SPI PC1 tracks (.82) with a weaker step (contrast 6.5) than centered or standardized z-PC1 (21.6, 26.1); z minus mean-SPI PC1 in |ρ| is .02–.09. Without a nuisance the unsupervised mean vector remains a competitive baseline.

Scope: the nuisance is additive white sensor noise with SD U[.3, .9] of signal SD, one construction; one size; exploratory first pass, though the primary readout and scores were recorded before the run. Provenance: source `ef8cc5e` at Scratch `operations/sources/cross-frequency-snr-ef8cc5e`, archive SHA `43ee5f68…c08866`, jobs 180583800/803/804 (farm) and 180583805 (analysis), all exit 0. Outputs `results/order-parameter-inference/cross-frequency-locking-snr-261006/`.

## M = 48 confirmation and a second nuisance, fixed before p90 outcomes

Purpose: confirm the varying-noise result at twice the size on fresh seeds, and test whether it is specific to additive independent noise. Code `scripts/cross_frequency_locking_confirm.py`; run `cross-frequency-locking-confirm-261006`. Sixteen oscillators per community (M = N = 48), T = 1000, otherwise the dynamics, control grid and truth above. Three arms, each 21 controls × 16 fresh seeds (8 fit, 8 held), 1,008 records, estimator seed 261095; arms are analysed separately. Every arm has white sensor noise; one nuisance per arm is drawn once per recording, independently of γ:

- `random-noise`: η ~ U[.3, .9]. Confirmation of the M = 24 result. Prediction: mean |r| and mean-SPI PC1 follow η; centered z-PC1 follows Q.
- `common-mode`: η = .5 plus a shared white signal of SD U[0, .8] added to every channel, as a common reference or shared pickup would. It inflates every dependence estimate, including those of unrelated pairs, where sensor noise attenuates them. Prediction as above.
- `internal-coupling`: η = .5 and within-community coupling ε ~ U[.15, .6], which moves the slow community's coherence between .40 and 1 (fast community .91 to 1). Declared boundary test, not a nuisance of the same kind: it changes the dependence structure itself (and shifts γ_c through X₂ and Y), not the level or scale of each SPI's edge profile. Prediction: centered z-PC1 is contaminated by ε. A success here would be a surprise; a failure marks the limit of the invariance.

Declared: centered z-PC1 primary, standardized z-PC1 sensitivity; headline scores are held |Spearman| with Q and with the arm's nuisance, with seed-bootstrap intervals for the primary and for z minus mean-SPI PC1; step scores secondary; ceilings as before. The confirmation counts as passed in an arm if centered z-PC1 has |ρ| with Q above .8 and with the nuisance below .2 while mean-SPI PC1 is below .3 with Q.

Motivation only (eight cheap probes, M = 24, four held seeds per control): held |ρ| with Q and with the nuisance were .88 and .06 for centered z-PC1 against .09 and .99 for mean-probe PC1 under common mode; .78 and .74 against .63 and .79 under internal coupling. Per-recording length T ~ U{250..1000} was also probed (.91 and .07 against .74 and .28) and not taken to p90, because length is under the analyst's control by cropping.

Figures from this point show Q on the left axis in physical units and each target-blind coordinate on a shared right axis in fit-record SD units; `python -m scripts.cross_frequency_locking{,_snr} figure` redraws the earlier ones from saved scores.

### Nuisance arms: full-p90 result at M = 24, 6 October

The first M = 48 submission (jobs 180609775/778/780) was killed for memory: 4 GB per task, and 8 GB in the two-task smoke job, were not enough; 44 of 1,008 records completed before cancellation and are kept. Before any outcome was read, the same three arms were run at M = N = 24 on separate fresh seeds (run `cross-frequency-locking-nuisance-261006`, `--size m24`, estimator seed 261097, archive SHA `abedd5c9…813c38`). All 1,008 records completed, zero selected-feature missingness, eight MPI replays exact for means and within 3e-8 for z. Held |Spearman|, 168 records per arm:

| Arm | Readout | With Q | With nuisance | Step contrast |
| --- | --- | ---: | ---: | ---: |
| Sensor noise | mean \|r\| / mean-SPI PC1 | .15 / .16 | 1.00 / 1.00 | .1 / .1 |
| | centered z-PC1 | .88 (.85–.90) | .09 | 7.8 |
| | standardized z-PC1 | .66 | .55 | 2.5 |
| Common mode | mean \|r\| / mean-SPI PC1 | .03 / .03 | .99 / 1.00 | .5 / .4 |
| | centered z-PC1 | .09 (.01–.21) | .98 | .5 |
| | standardized z-PC1 | .11 | .99 | .5 |
| Internal coupling | mean \|r\| / mean-SPI PC1 | .27 / .61 | .88 / .74 | .2 / .3 |
| | centered z-PC1 | .85 (.83–.86) | .36 | 3.7 |
| | standardized z-PC1 | .74 | .60 | 1.6 |

Full-mean ridge ceilings recover Q in every arm (.90, .88, .93).

**Sensor noise replicates** on independent seeds and passes the declared criterion; z minus mean-SPI PC1 is .58–.87. The second mean-SPI component again carries the transition (.82).

**Common mode fails, against the prediction.** Centered z-PC1 follows the shared signal exactly as the baselines do; the transition is the second component of both representations (|ρ| with Q .85 for z, .84 for means). Mechanism, from the features: a shared white signal adds instantaneous, undirected dependence to every pair and no lagged or directed dependence. That is a change in the mix of dependence types across the recording, which is what z measures, not a rescaling. Total z variance is five times that of the sensor-noise arm and 85% of it follows the nuisance; PC1 loads on pairs of a zero-lag statistic with a directed or lagged one (spectral Granger, transfer entropy); z(covariance, transfer entropy) falls from .47 to −.01 as the shared signal grows. The eight-statistic probe predicted success because it contained no directed statistic: the second probe-to-p90 misprediction.

**Internal coupling does better than predicted.** z-PC1 tracks Q (.85 against .61 for mean-SPI PC1; difference .16–.34) and steps at the boundary. It is not clean: within a control value it follows ε (median |ρ| .86), part of which is legitimate because Q itself rises with ε there (.69).

**Revised claim.** Centered SPI–SPI PC1 is robust to recording differences that rescale dependence (sensor noise, coupling heterogeneity) and not to ones that change the mix of dependence types (a shared instantaneous signal). The second kind is, for this representation, signal by construction. A common signal identical on all channels is removable by average re-referencing; that was not tested here.

Provenance: source `02faa24` at Scratch `operations/sources/cross-frequency-nuisance-02faa24`; jobs 180617499/500/501 (farm), 180617502 (analysis), all exit 0. Outputs `results/order-parameter-inference/cross-frequency-locking-nuisance-261006/` including `component-scatter-<arm>.png`.

## Real-data attempt: NeuroTycho anaesthetic induction (exploratory, 6 October)

Question: does the unsupervised picture carry over to a real regime change? Data already staged for the transfer pilot: 15 dates, four macaques, ketamine–medetomidine (11) and propofol (4), 16 bipolar ECoG channels, pilot preprocessing unchanged. New here: 2,239 eight-second windows tiling each injection session (20 s stride through induction and the labelled anaesthetized interval, 60 s through emergence and recovery) plus the awake-eyes-closed interval; full p90 at M = 16, T = 2000. Truth is event markers only: there is no measured control or per-window order parameter. Coordinates are fitted without the evaluated animal. Code `scripts/neurotycho_induction.py`; outputs `results/order-parameter-inference/neurotycho-induction-261006/`.

**The result is the reverse of the synthetic one.** Awake versus anaesthetized on the held animal, AUROC averaged over dates (ketamine–medetomidine / propofol): mean |r| .96 / .99, mean-SPI PC1 1.00 / 1.00, centered z-PC1 .72 / .70, standardized z-PC1 .86 / .89. Mean |r| and mean-SPI PC1 show a level, a transition within a few minutes of injection, and a plateau; under propofol they also return towards baseline through recovery. Centered z-PC1 separates animals, not states (fit rows: 47% of its variance between animals, 3% between states); the state is the second z component. Fitted within a single date, where no between-animal difference exists, mean-SPI PC1 still separates the states perfectly (1.00), centered z-PC1 less reliably (mean .90 / .88, minimum .70 / .59) and standardized z-PC1 .97 / .99.

**Why.** Anaesthesia here is first a change of strength: mean |r| roughly doubles (.09–.10 to .16–.19). Means register that directly and z discards it by construction. What differs between animals is which cortical sites the bipolar pairs sample, which changes the pattern of dependence types across pairs; that is what z measures, as in the common-mode arm. Both observations fit the revised claim above. The earlier supervised pilot, in which z transferred across animals better than means, is not contradicted: a supervised readout can pick the state direction that an unsupervised PC1 does not rank first.

**Reading.** For a strength-dominated real transition the simple baselines are the right unsupervised tool, and SPI–SPI PC1 adds nothing. The synthetic locking result stands as a controlled demonstration of one property (a type change at fixed strength, under rescaling nuisances), not as evidence that real regime changes are of that kind. One KTMD animal (Su) shows no change in any leave-animal-out readout; within its own dates mean-SPI PC1 does separate the states; not investigated further.

## Carry-forward: state, lessons and open decisions (6 October)

**Where it stands.** Two full-p90 results at M = N = 24: (i) clean recordings — mean |r| fails, every catalogue PC1 steps, no advantage over mean-SPI PC1; (ii) recording-specific sensor noise — mean |r| and mean-SPI PC1 follow the noise, centered z-PC1 tracks Q (.87). The three nuisance arms have full-p90 results at M = 24 (above); M = 48 awaits a memory-sized resubmission. Promotion of this system into the lean benchmark notebook was deferred until confirmation; two concise cells are in `notebooks/inference/dependence-character-transitions-261005.ipynb` (uncommitted, alongside other uncommitted edits).

**Design lessons, each backed by a run.**
- z needs at least three edge types. With two homogeneous classes every SPI profile is a two-level pattern and z, which discards level and scale, cannot move (two-community scratch run).
- Sweep only the cross-coupling. When internal coupling co-varied, z-PC1 was captured by the ordinary synchrony onset and the locking appeared in PC2.
- Pooled Spearman rewards any monotone drift; split it at the boundary or use the step scores. No unsupervised readout tracks Q's ramp below the boundary in clean recordings.
- Cheap-probe runs mispredicted p90 twice (mean-SPI PC1 in clean recordings; z-PC1 under common mode) and predicted it twice. Both misses came from statistic families absent from the probe set. Treat probes as motivation.
- Centered versus standardized z-PC1 is unresolved: centered PCA puts half of PC1's weight on the 5% highest-variance z features; standardized spreads it. Standardized looked better in clean recordings, centered was robust under varying noise and in the copula run where standardized failed. Report both.
- Working hypothesis, not a theorem: in homogeneous recordings with a rich catalogue, mean-SPI PC1 sees what z-PC1 sees; the baselines fail when a per-recording nuisance moves every dependence estimate together. The failure is one of ordering: the transition is then the second mean-SPI component, and the nuisance the second z component.

**Tried and set aside.** Conformist–contrarian Kuramoto (Hong and Strogatz, PRL 106, 054102; π-state to travelling wave): right mechanism, smeared crossover at N = 32–64, no clear win in a nine-probe scratch run. Off-diagonal means depend only on the symmetric part of an MPI, but onset of directionality usually changes that part too, so directionality is a weak lead. Ginzburg–Landau phase-to-defect turbulence stays excluded for the reason already in the workstream context. Scratch scripts for these and for the locking probes are kept locally (not in Git) under `results/order-parameter-inference/cross-frequency-locking-261006/scratch-probes/`.

**Real-data recommendation given to the user.** For unsupervised transition inference: seizure onset in TUSZ scalp EEG (19–22 channels, full p90 without pair sampling, 4 s windows at 250 Hz, annotated boundaries, naturally varying recording quality); caveat that the truth is a binary annotation and generalized seizures change strength strongly. Second: anaesthesia induction in the existing NeuroTycho ECoG banks. Expected framing is validity, not superiority; the method's strongest use is likely comparing recordings of different size and length.

**Operational.** Push the local HEAD to `origin studies/dependence-transition-261005`, fetch on Gadi, add a detached worktree under `operations/sources/` (gdata inodes are nearly exhausted, so the second run's worktree is on Scratch). `submit_external_corpus_farm.sh` computes memory as cores × GB and is rejected above 190 GB per node; call `qsub` on `run_external_corpus_farm.pbs` directly with `mem` = 190 GB × nodes. M = 24, T = 1000 costs 750–840 s and about 2 GB per record.
