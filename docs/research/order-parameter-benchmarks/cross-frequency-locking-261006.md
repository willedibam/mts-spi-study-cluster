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
