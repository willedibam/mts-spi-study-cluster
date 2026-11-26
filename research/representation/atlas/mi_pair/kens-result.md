# Result: surrogate-ensemble fourth-order readouts (26 November 2026)

- **Rule:** [preregs/kens-prereg.md](preregs/kens-prereg.md), frozen at aacb335 and amended at 10e6b9e, both before any output. It follows [kdiff-result.md](kdiff-result.md).
- **Code:** [`cumulants/kens.py`](cumulants/kens.py) at 10e6b9e and [`jobs/kens.pbs`](jobs/kens.pbs). The scorer [`cumulants/kens_score.py`](cumulants/kens_score.py) was committed at c92a07cb before any output was read; it fixes the choices the prereg left open.
- **Gadi job:** 185197676, at commit 10e6b9e. Held on ql44's scratch inode quota until 00:35 on 26 November, then ran 54 min on 48 CPUs (about 0.09 KSU) and exited 0.
- **Outputs:** [`ke-score.txt`](../../../../results/representation/atlas-mi-pair-261117/ensemble/ke-score.txt), `ke-score.json` and `ke-windows.csv.gz` are tracked in `results/representation/atlas-mi-pair-261117/ensemble/`. The `ke_*.npz` files (154 MB) are on gdata at `/g/data/ql44/we2614/mts-spi-study/representation/atlas-mi-pair-261117/ensemble/`, checked against `SHA256SUMS-kens`.

## Verdict: no detectable drug effect on the fourth-order excess beyond the matched null

**Gates (all pass).** Draw 0's window-mean net k04 reproduces kdiff's `KD` within 1.3e-8 in all 10 files. `KD` and Δz run through the scorer reproduce `kd-score.json` exactly. The `im` arm's marginals equal the real window's exactly, and no pair has zero null SD.

Panel (c)'s units. "Drug" is the mean of 4 animals (Chibi, George, Kin2, Su); sleep and AUC are Chibi and George only.

| Readout | Drug | Animals | t(3) | Sleep | AUC |
|---|---|---|---|---|---|
| Δz (panel (c)) | −3.27 | −4.69 −2.16 −2.85 −3.38 | −6.11 | −0.11 −0.41 | .93 .82 |
| `KD` (kdiff, single draw) | +1.19 | +2.11 +1.42 −0.00 +1.24 | +2.71 | +0.81 +0.64 | .61 .62 |
| `KZ` (primary: k04 z against `im`) | −0.00 | +0.71 +0.17 −1.00 +0.10 | −0.01 | +0.54 +0.42 | .56 .55 |
| `MZ` (Mardia z against `im`) | +0.04 | +0.94 −0.08 −0.83 +0.12 | +0.10 | +0.45 +0.41 | .49 .40 |
| `KZ` tightest \|r\| tercile | +0.50 | +0.79 +0.91 −0.51 +0.80 | +1.48 | +0.54 +0.34 | .46 .58 |
| `KD_ens` (net against `im` mean) | +0.82 | +1.80 +0.91 −0.50 +1.06 | +1.71 | +0.76 +0.69 | .53 .54 |
| `KD_ens_ia` (net against `ia` mean) | +1.20 | +2.17 +1.36 −0.02 +1.29 | +2.65 | +0.80 +0.75 | .60 .60 |
| Tail fraction (k04) | −0.15 | +0.33 −0.07 −0.84 −0.01 | −0.60 | +0.54 +0.43 | .59 .64 |

- **Decision 5.** `KZ` does not replace `KD`: t(3) −0.01 against +2.71, and it is positive in 3 of 4 animals. It does not improve on Δz for anaesthesia against sleep: AUC .56 and .55 against .93 and .82. Under the prereg rule, the level readouts' drug effect is attributed to the null's spectral and marginal shifts, not to the data. This is absence of evidence: the 95% interval of `KZ`'s animal mean is about ±1.1 SD, nearly reaching `KD`'s +1.19, and the widening null (below) lowers sensitivity under drug, since a fixed raw excess gives a smaller z.
- `KD_ens_ia` ≈ `KD`, so kdiff's single draw was representative of its ensemble. Every standardised secondary has |t(3)| ≤ 1.48 (k40 and k22 z, z against `ia` and `ft`, Mardia tail). Unstandardised nets keep part of the effect, for the reason given under Mechanism: Mardia net against `im` +1.00 (t 2.25), L-kurtosis of the difference +0.90 (t 1.81). See `ke-score.txt`.

## Mechanism (raw window-mean k04, drug minus eyes-closed, pooled over the 4 animals)

- **Decomposition of `KD`'s drug change (+.084).**
  - The real k04 rises by +.030 (36% of the change); by animal, +.098, +.033, −.066 and +.056.
  - The matched null (`im`) falls by −.026, contributing 31%.
  - The marginal leak (M2) contributes 33%. The spectrum-final IAAFT's marginal excess kurtosis falls by −.051 with drug, so `ia` falls by −.053, about twice as far as `im`.
- **Spectral concentration holds for k04.** The random-phase prediction from each window's own Fourier coefficients reproduces the `ia` mean's drug change: ratio 1.10, window r .97. Against the phase-randomised arm the ratio is 0.995, with r .99. The null's k04 fall is the finite-window kurtosis of a random-phase signal with the drug spectrum, not something specific to IAAFT.
- **For k22 it does not hold against `ia`.** The pred/`ia` ratio is 1.30, outside ±25% (window r .94), while pred/`ft` is 0.99. The spectrum-final IAAFT's k22 departs from the random-phase prediction.
- **k22.**
  - Real +.016; `im` +.011; `ia` −.016.
  - kdiff's 2·k22 share of MARDIA's drug effect was mostly `ia`'s marginal term, which enters as (ka + kb) / (4(1 − r²)).
- **Why the net against `im` survives (+0.82, t 1.71) but `KZ` does not.** This is post hoc and descriptive: [`cumulants/kens_posthoc.py`](cumulants/kens_posthoc.py), written after the outputs were read, with output in [`ke-posthoc.txt`](../../../../results/representation/atlas-mi-pair-261117/ensemble/ke-posthoc.txt).
  - Under drug the `im` null's per-pair SD of k04 grows by a factor of about 1.4 to 1.6: the mean log SD rises by +0.34 to +0.49 across animals, against +0.08 to +0.10 in sleep.
  - The real excess grows in proportion. In eyes-closed wake, real k04 already exceeds the matched null by +0.73 null SDs per pair. The drug change in that z is +0.26, +0.06, −0.36 and +0.04.
  - So, in units of what spectrum- and marginal-matched linear Gaussian dynamics produce at that spectrum and window length, drug adds nothing.

## Reading

- **What is there, and what drug does to it.**
  - Every animal and state has a fourth-order excess over spectrum- and marginal-matched linear Gaussian dynamics. Window-mean `KZ` runs from +0.32 to +1.08 null SDs per pair, and 8.5% to 19% of pairs sit in the top 2 of 100 draws, against 3% by chance (`ke-posthoc.txt`).
  - Anaesthesia does not change that excess in null-SD units, and it does not separate anaesthesia from sleep.
  - As the prereg states, an excess does not show nonlinear coupling. It can come from a nonlinear coupling function, a gain or coupling mixture, or linear mixing of non-Gaussian sources.
- **For panel (c): keep Δz alone.** This withdraws kdiff-result's "companion" recommendation.
  - About two-thirds of `KD`'s and MARDIA's drug effect is the null moving (spectral concentration plus the marginal leak).
  - The remaining third scales with the null's own widening spread, so the per-pair z is unchanged.

## Panel (c) redrawn with mean difference kurtosis (post hoc, 26 November)

- **Figure:** [`kd-panel-c.png`](../../../../results/representation/atlas-mi-pair-261117/ensemble/kd-panel-c.png) (and `.pdf`), from [`cumulants/kd_panel.py`](cumulants/kd_panel.py) on `ke-windows.csv.gz` and `kd-windows.csv.gz`. Gate: its Δz equals `plot_character_draft.load()`'s window by window (max gap 1.8e-15), and the printed state means reproduce panel (c)'s (+0.1, −0.2, −3.9, −3.8, −1.5, −2.9). Real and null are drawn on KD's scale, so real − null = KD exactly. "All pairs" is the 496 pairs of each window's 32-channel subset, as in the anchor.
- **Window-pooled means mislead for the level readouts.** Panel (c)'s printed numbers are window means per agent. For KD they read +3.7 (ketamine) and +2.2 (propofol). Each is dominated by one animal: Chibi's ketamine (+6.05; George +1.33) and George's propofol (+3.74; Chibi +0.58). About half of George's propofol rise is a low eyes-closed baseline (raw k04 .078 against .154 eyes open). Medetomidine is +0.1, below sleep (+0.7). Δz's ketamine mean is also Chibi-dominated in size (−6.66; George −1.20), but Δz is negative for every animal under every agent; KD and KZ are not.
- **Chibi under ketamine is the one large, consistent real rise (verified; cause untested).**
  - Raw window-mean k04 is .58, against .15 to .16 in that session's wake, in all 20 drug windows over both sessions (KZ 1.6 to 4.2). Its wake baseline is ordinary.
  - It survives the robust readouts. The median over pairs (`KDmed`) is .32 against .15. The L-kurtosis net against `im` (`TVN`) is .028 against .008, the largest cell in the data. So it is neither a few pairs nor a few extreme samples.
  - It is not slow-wave share. Chibi's 4 sleep windows at delta above .6 have k04 .28. Its ketamine + medetomidine and propofol windows, at higher delta, have .03 and .11.
  - It is broad over pairs: the loose and tight |r| terciles rise alike.
- **Δz and kurtosis dissociate across agents (descriptive, one animal).** Chibi's ketamine and ketamine + medetomidine have about the same Δz (−6.7 and −6.1) at opposite kurtosis (k04 .58 against .03). So in that animal, Δz's fall does not need a fourth-order rise. Within a condition the two covary where there is variation: Spearman of Δz with KZ is −.39 to −.80 in all 5 of Chibi's cells, −.44 in Su's and −.49 in George's sleep. In George's drug cells and Kin2's it is −.22 to +.33 (n 20 to 30), and the pooled within-session value is −.2.
- **Why the dissociation (post hoc, descriptive; [`cumulants/dz_excess.py`](cumulants/dz_excess.py) on the zns edge arrays and `ke_*.npz`).** Δz is a correlation over pairs: it falls when the per-pair excess E = K − G of the recording scatters more, relative to the spread of G, than its surrogate's does. In Chibi the recording's extra scatter is of similar size under both agents (quadrature difference of the cell-mean SDs of E, recording against surrogate: .039 nats under ketamine, .034 under ketamine + medetomidine, .006 eyes closed), but it arises differently.
  - Ketamine: the spread of G across pairs doubles (SD .138 against .066 eyes closed). The surrogate's corr(G, K) therefore rises (r .988 against .955), while the recording's stays at its wake level (.963), because its E scatter doubles too. So this Δz fall is the surrogate rising, with the recording's r flat (as Chibi's z_surr +3.85 in [kdiff-result.md](kdiff-result.md)). The excess beyond the surrogate tracks the pair's fourth-order excess beyond the matched null (within-window Spearman of E − E_surr with |k04 z| .40, against .06 eyes closed; the raw E–|k04| value, .45, is inflated because both scale with |r|). That Δz and kurtosis see the same pair-specific structure here is inferred.
  - Ketamine + medetomidine (delta .93): the G spread is unchanged and the surrogate's own E scatter doubles (.041 against .020), so both correlations collapse (r .857 and .884). Signed mean k04 is .03 because 57% of pairs are platykurtic; mean |k04| rises (.36 against .22) but no further than the matched null's spread (RMS z of k04 against `im` 1.77, against 1.82 eyes closed). E − E_surr barely tracks |k04 z| (.09). So the extra scatter here is something the pairwise fourth-order cumulants do not carry: other orders (third-order skew is not measured), non-stationarity that IAAFT removes, or estimator behaviour. Untested.
  - Estimator: mean E is negative under ketamine + medetomidine (recording −.029, surrogate −.017), which true MI with Gaussian marginals cannot be (MI ≥ G at fixed r). Per [ksg-bias-result.md](ksg-bias-result.md) part is the Theiler-window bias shared with the surrogate and part (about −.01 at weak edges) is real-only, inferred to be amplitude non-stationarity; the bias-corrected kshift keeps ketamine + medetomidine's Δz (retention 1.14), so the real-only bias does not carry it. Whether non-stationarity carries the scatter by another route belongs to the nonstat and envelope tests.
- **Chibi's ketamine coupling rise is broad over electrodes, not a few channels (post hoc, descriptive; [`cumulants/dz_nodal.py`](cumulants/dz_nodal.py)).** An additive node model (G_ij = a_i + a_j) explains more of G under ketamine (R² .30 against .11 eyes closed), so some electrodes take a larger share, but dropping every pair that touches the 3 strongest nodes leaves the G spread at .113 (eyes closed .059), and the node profile matches wake (Pearson .74 and .80 in the two sessions). The k04 excess over the matched null is likewise more node-structured (R² .43 against .18) on the same electrodes as wake (.48, .62). Higher r here means more shared variance; whether that is stronger coupling, a larger common generator, or less local independent activity is not separable from pairwise statistics.
- **Chibi's ketamine heavy tails are episodic decoupling at large amplitude (post hoc, descriptive; [`cumulants/episodic.py`](cumulants/episodic.py), 1-s segments, draw-0 surrogate; [`../../../../results/representation/atlas-mi-pair-261117/cumulants/ep-windows.csv.gz`](../../../../results/representation/atlas-mi-pair-261117/cumulants/ep-windows.csv.gz)).**
  - **Coupling varies within the window.** After normal scoring, k04 > 0 means the pair's coupling changes over the window. Segment correlations vary more than the surrogate's (excess SD of segment z .048, against .015 eyes closed). The heavy-tailed pairs vary most, but only a little more than in wake (Spearman .33, against .19).
  - **The largest-amplitude segments are the least coupled.** Over segments, the Spearman of global coupling with segment RMS:

    | Global coupling | Recording | Surrogate | Eyes-closed wake (recording, surrogate) |
    |---|---|---|---|
    | Mean z over pairs | −.63 | +.44 | +.34, +.24 |
    | Mean \|z\| | −.26 to −.51 per session | +.70 | +.46, +.63 |

    The unsigned version rules out a common-average-reference artefact, where a large local wave would add anticorrelation.
  - **It opposes shared gain.** A gain on a shared generator would raise coupling at high amplitude, and this is the opposite. Of the two candidates in `findings.md`, it favours amplitude-linked decoupling in this cell.
  - **Amplitude here is mostly 1–4 Hz power** (Spearman with coupling −.57, against −.11 for 25–45 Hz). At 100 Hz sampling, that is weak evidence against muscle artefact.
  - **Large events carry part of it.** The transient cut (beyond 5 robust SDs; a third of the samples here) shrinks the session's Δz residual from −.027 to −.018, against −.031 for the shifted control.
  - **Within Chibi it tracks Δz only weakly.** Spearman over ketamine windows is .07 and .45 in the two sessions (.26 over both). The .79 over all ketamine windows is mostly the difference between Chibi and George.
  - **Physiology against artefact is not decided.** It is broad over electrodes and on wake's node profile. But the drug sessions have no EMG or video, so movement artefact is not excluded.
- **As an unsigned readout, amplitude-linked decoupling falls under every drug and not in sleep (post hoc, exploratory; the second variant tried, so needs a hold-out).**
  - Readout: real minus surrogate Spearman of segment mean |z| with segment RMS, session against own eyes closed, in eyes-open SDs.
  - Animals: Chibi −1.20, George −0.42, Kin2 −0.31, Su −0.38 (t(3) −2.76). Sessions below 0: KT 3/4, KTMD 11/11, MD 3/4, PF 3/4.
  - Sleep is 0.00 (ratio to drug 0.00), and same-session sleep against wake AUC is .48 and .51. Same grouping as Δz: sleep with wake.
  - The signed version (mean z) is Chibi-only (t(3) −0.58; KTMD 5/11), and sleep −0.23.
  - At matched delta in George it fails as Δz does: KT and MD against sleep, AUC .50 and .53.
  - The coupling-variability excess is not a sleep separator: it rises under drugs (KT +2.0, PF +2.2 SD) but also in sleep (+0.51, 6/7; ratio .44).
