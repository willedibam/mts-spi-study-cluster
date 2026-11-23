# What form does the MI pair's non-Gaussian dependence take? Cross-cumulant decomposition on the macaque 10-s windows

Frozen 23 November 2026 at the commit that adds it, after the synthetic calibration and before any real-data output of this test exists. Code: [`../cumulants/stats.py`](../cumulants/stats.py), [`../cumulants/calib.py`](../cumulants/calib.py), [`../cumulants/run.py`](../cumulants/run.py); job [`../jobs/cumulants.pbs`](../jobs/cumulants.pbs).

## Why

- The anchor, z(Gaussian MI, kNN MI) beyond joint IAAFT, falls under four anaesthetics (4/4 macaques, t(3) −6.1). It reads as non-Gaussian dependence: after normal-scoring, I − G is the KL divergence from the covariance-matched Gaussian.
- The segment surrogates ([nonstat-prereg.md](nonstat-prereg.md)) place most of the departure in variation between about 1-s stretches, a Gaussian mixture. A sub-second part remains.
- What is still unnamed is the *form* of the dependence the kNN estimate sees beyond G. This is a descriptive question about the joint distribution, not about mechanism.
- This replaces the Canberra readout in the design the user approved on 23 November (amplitude co-modulation). Envelope correlation against |γ|² is kept as `env`. Canberra is dropped because its gain invariance only helps if amplitude stratification is identifiable, and it is not.
- Amplitude-stratified readouts of the same question failed on real data: the noise floor's spatial coherence sets their sign ([amplitude-prereg.md](amplitude-prereg.md)). This test does not stratify by amplitude.

## Readouts

**Decomposition.** For weak non-Gaussianity, the negentropy of a whitened pair is approximately (1/12) Σ κ₃² + (1/48) Σ κ₄² over all index tuples (Edgeworth / Gram–Charlier; Comon 1994). With normal-scored marginals, I − G equals the joint negentropy. So, per edge:
- **J3** (third-order cross-cumulants): sign-asymmetric joint structure, such as curved coupling or regimes that shift the mean. It is exactly zero for any mixture of zero-mean Gaussians, so gain or coupling mixtures cannot produce it.
- **J4** (fourth-order): symmetric co-modulation of magnitudes, such as gain or coupling mixtures, decoupling episodes and heavy joint tails.
- **env:** envelope power correlation of the analytic signals minus its exact Gaussian value |γ|², i.e. co-modulation at the scale of the oscillation.

**Per window, real minus the anchor's own IAAFT draw ("net"):**
- `zGK`, the anchor;
- `zJ`, `zJ3`, `zJ4` = z(G, G + J), the anchor's organisation rebuilt from the cumulant estimate of I;
- `zP` = partial z(G, K | J3, J4), the anchor with both cumulant terms partialled out of the edge profiles;
- `link3`, `link4`, the across-edge r of the kNN excess (K − K_surr) with the J3 and J4 excess;
- the levels `L3`, `L4` and `Lenv`, mean edge excesses;
- `zenv` = z(|γ|², p).

## Calibration

The segment calibration's generators plus `updown`, a shared two-state regime that shifts the channel means (DOWN 40% of the time, mean dwell .4 s), 12 seeds each. Every window is 32 channels, T = 1000, normal-scored, and gets one spectrum-final IAAFT draw. Source: [`calib1.txt`](../../../../../results/representation/atlas-mi-pair-261117/cumulants/calib1.txt) and `calib1.npz` (`python calib.py 12 8 calib1.npz`). Net = real − surrogate, given as the mean over seeds; t over seeds is in calib1.txt. S_z = zJ3 / zJ and S_L = L3 / (L3 + L4) are the J3 shares; the brackets give the 10th–90th percentile over seeds.

| generator | zGK | zJ | zJ / zGK | S_z | S_L | R_P = zP / zGK | link3 | link4 | Lenv |
|---|---|---|---|---|---|---|---|---|---|
| linear | −.0015 | −.0015 | 1.0 | .45 [.12, .69] | unstable | 1.22 | .18 | .13 | +.007 |
| burst | −.068 | −.084 | 1.2 | .13 [.07, .19] | .08 | .85 | .16 | .26 | +.36 |
| burst_slow | −.048 | −.064 | 1.3 | .10 [.05, .29] | .07 | .77 | .12 | .20 | +.27 |
| gain_cmfloor | −.029 | −.158 | 5.5 | .03 [.02, .06] | .08 | .82 | .13 | .29 | +.35 |
| gain_local | −.062 | −.055 | 0.9 | .05 [.01, .07] | .02 | 1.06 | .08 | .32 | −.04 |
| switch | −.012 | −.077 | 6.6 | .04 [.02, .07] | .10 | .26 | .21 | .26 | +.07 |
| decouple | −.017 | −.077 | 4.6 | .03 [.02, .05] | .05 | .64 | .25 | .38 | −.003 |
| quadratic | −.046 | −.105 | 2.3 | .38 [.29, .51] | .66 | .07 | .61 | .36 | +.04 |
| quad_weak | −.0026 | −.0074 | 2.9 | .36 [.29, .66] | .81 | .45 | .24 | .17 | +.001 |
| updown | −.0098 | −.017 | 1.7 | .30 [.22, .37] | .39 | 2.09 | .32 | .18 | +.07 |

What the calibration shows:
- **The J3 share separates the two families.**
  - Every zero-mean mixture (burst, gain, switch, decouple) gives S_z .03–.13 and S_L ≤ .10.
  - Curved coupling and mean-shifting regimes give S_z .30–.38 and S_L ≥ .39.
  - This holds even though zero-mean mixtures have zero J3 in expectation only in the weak limit; their finite-sample J3 net is small but not zero.
- **The cumulant estimate gets the sign of the anchor right in every generator but not its size.** zJ / zGK runs from 0.9 to 6.6, and is largest for heavy-tailed mixtures, where the Edgeworth expansion overshoots. So zJ is read for sign only.
- **Partial retention R_P is not interpretable.**
  - It runs from .07 (quadratic) to 2.09 (updown), and gain_local, a pure J4 generator, keeps 1.06.
  - The kNN excess of zero-mean mixtures is not linear in the edge profiles of J3 and J4.
  - R_P is reported, not read. The same holds for link3 and link4: their null baseline is .13–.18 under the linear generator.
- **Lenv is positive (.27–.36) only when the envelope co-modulation is shared across channels** (burst, shared gain). It is near zero or negative for local gain, decoupling and quadratic.

## Real data

- **Windows:** the anchor's windows, subsets, normal-scoring and IAAFT seed exactly, as in the segment test. This covers NeuroTycho 10-s stages A and B, 1,260 windows, 32-channel subsets. G and K per edge are read from the stored `zns` arrays.
- **Reproduction gate:** recomputed G must match the stored G within 1e-4 for real and surrogate, or the run stops.
- **Unit:** per session, Δ = drug (or sleep) − awake eyes-closed. Animal means pool the drug sessions; t(3) on the 4 animal means; eyes-open SD as the scale. Also reported: per agent, sleep, the median within-drug-session window r of each net with the anchor net, and the partial retention R_P = Δ`zP` / Δ`zGK`.

## Reading map (stated before real data)

All readings are on the drug sessions' Δ (drug − awake eyes-closed), pooled, with the 4 animal values. Sleep is reported alongside but not read against the thresholds.

1. **Precondition.** Δ`zJ` must have the anchor's sign (negative) in at least 3 of the 4 animals. Otherwise the cumulant decomposition does not see the anchor's change: report that, and read nothing below.
2. **Primary: the form of the added non-Gaussianity, from S_z** (pooled; animal values reported).
   - **S_z < .20, with at least 3/4 animals below .25:** the added dependence is a **symmetric co-modulation of magnitudes**, a zero-mean mixture (gain or coupling varying between stretches, or decoupling episodes). Sign-asymmetric joint structure is not needed.
   - **S_z > .25, with at least 3/4 animals above .20:** a **sign-asymmetric component** is substantial. It could come from curved coupling or from regimes that shift the channel means, such as up/down states; the two are not separated (see 4).
   - **Otherwise:** indeterminate between the two.
3. **Check: S_L.** S_L should fall on the same side (zero-mean ≤ .10 against ≥ .39 in calibration; boundary .20 / .30). If it disagrees with S_z, the primary reading is downgraded to indeterminate.
4. **Not separated by this test:** curved coupling against mean-shifting regimes. R_P differs between them in calibration (.07 against 2.09), but it is erratic across the zero-mean generators, so it is not used.
5. **Secondary: shared envelope co-modulation, from Lenv** (descriptive). ΔLenv > 0 in 4/4 animals is consistent with envelope co-modulation shared across channels at the oscillation scale (burst, shared gain). ΔLenv ≤ 0 is consistent with local or decoupling forms. The amplitude test showed that the noise floor's coherence can fake the shared-against-local contrast, so this is not read as gain against coupling.
6. **Size.** Neither the size of Δ`zJ` nor R_P or the links is read. They are reported.

The prior is stated, not tested. The segment test found a mixture over about 1-s stretches. Anaesthetic slow oscillations alternate up and down states at about this scale (Steriade et al. 1993; established), which favours a mean-shifting component and so S_z > .25. A gain or coupling mixture would give S_z < .20.

## Limits

- The Edgeworth terms approximate I − G only for weak non-Gaussianity. Fourth-order sample cumulants are noisy and slow spectra lower the effective sample size; the IAAFT net removes their bias but not their variance.
- Odd cumulants depend on channel polarity. They are squared here, so polarity does not cancel within an edge, but a mean-shift regime and a curved coupling both give J3.
- Shared amplitude co-modulation is the primary readout of a separate frozen test, on all channels with the amplitude envelope ([axis screen, Gaussian-referenced readouts](../../axis_screen/result.md)). `Lenv` here is a secondary check on the anchor's 32-channel normal-scored windows, not an independent replication. That test's calibration also confirms that Canberra cannot read amplitude-linked decoupling.
- One surrogate draw per window (the anchor's); 32-channel subsets; four animals. This is a description of form, not a mechanism test: gain and coupling remain unidentifiable from passive data.

## Compute

Gadi, 48 cpus. 1,260 windows × (one IAAFT draw + two cumulant passes), well under an hour wall; walltime 1 h. Expected about 0.02–0.04 kSU (10–25 min on 48 cpus at 2 SU per cpu-hour), worst case 0.1 kSU (the full 1-h walltime).

## Result (job 184940876, commit f59024b4, 23 November; 2 min 37 s on 48 cpus, about 0.004 kSU)

Reproduction gate passed: recomputed G matches stored G exactly (0.0e+00). The anchor reproduces: Δ`zGK` −.0133, animals −.019 to −.009, t(3) −6.1. Output: [`score.txt`](../../../../../results/representation/atlas-mi-pair-261117/cumulants/score.txt) and `cum-score.json`.

| readout | Δ (drug − awake EC) | animals | t(3) | eyes-open SD |
|---|---|---|---|---|
| zJ | −.042 | −.055, −.026, −.039, −.055 | −6.2 | .015 |
| zJ3 | −.0029 | −.0039, −.0022, −.0015, −.0035 | −5.0 | .0007 |
| zJ4 | −.039 | −.053, −.024, −.036, −.048 | −6.2 | .014 |
| L3 | +.0015 | +.0028, +.0012, −.0004, +.0008 | +1.7 | .0008 |
| L4 | +.0072 | +.011, +.0057, +.0026, +.0060 | +3.6 | .0013 |
| Lenv | +.009 | +.014, +.018, −.035, +.013 | +0.2 | .027 |

Reading by the map:
1. **Precondition met.** Δ`zJ` is negative in 4/4 animals.
2. **Primary: S_z = .07; animals .07, .09, .04, .06, all below .20.** The added non-Gaussian dependence is a **symmetric co-modulation of magnitudes**, the form of a zero-mean mixture (gain or coupling varying between stretches, or decoupling episodes). The J3 change is real (4/4, t −5.0) but small: 7% of the cumulant change, against .30–.38 for the calibrated curved-coupling and mean-shifting generators.
3. **Check: S_L = .17 (animals .20, .18, −.19, .12)** falls below the .20 boundary and agrees. It sits above the zero-mean generators' ≤ .10, so the agreement is not strong, and one animal's ratio is negative because its ΔL3 is.
4. **Lenv: no reading.** It is positive in 3/4 animals, but one is −.035 and the pooled change is under one eyes-open SD.
5. **Reported, not read:**
   - Δ`zJ` / Δ`zGK` = 3.1, inside the calibrated range of 0.9–6.6.
   - R_P = 1.46: partialling J3 and J4 out of the edge profiles enlarges the anchor's change, as for several calibrated zero-mean generators.
   - The links rise under the drug (+.11 each, t +3.9 and +4.2).
6. **By agent:**
   - S_z is KT .06, KTMD .07 and PF .07.
   - MD gives .21 on a small cumulant change (Δ`zJ` −.003).
   - Sleep gives .03 on a negligible anchor change.

**The prior was not supported.** Up/down states with the per-channel mean shift calibrated in `updown` would have given S_z near .30. Slow-wave up/down activity remains possible if it acts on the data as a gain or coupling change, not as a mean shift.

**What this adds to the segment test:** the mixture over about 1-s stretches is symmetric. Stretches differ in the magnitude of shared fluctuations or of coupling, not in the sign-asymmetric shape of the joint distribution. This agrees with the propofol-only control-edge diagnostics (curvature −0.7, no curved coupling function). It does not separate gain from coupling (Limits).
