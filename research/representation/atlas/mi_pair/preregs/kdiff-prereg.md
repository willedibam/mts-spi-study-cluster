# Mean difference kurtosis: a level readout of the MI pair's non-Gaussian dependence, and whether it separates anaesthesia from sleep

Post hoc, frozen 25 November 2026 at the commit that adds it, before any output of this readout exists. Code: [`../cumulants/kdiff.py`](../cumulants/kdiff.py); job [`../jobs/kdiff.pbs`](../jobs/kdiff.pbs).

## Why

- Panel (a)'s strip prints, for one selected pair, the excess (Fisher) kurtosis of the normal-scored electrode difference a − b (Gaussian 0). The proposal is to average it over all 496 pairs per window and plot that across states in place of, or beside, panel (c)'s Δz.
- The two answer different questions. Δz is an *organisation* readout: how well G's ranking of edges accounts for kNN MI's. Mean difference kurtosis is a *level* readout: how far, on average, each pair's difference departs from the Gaussian at the same r. It is the declared level measure the reading rules ask for before writing "more non-Gaussian".
- With normal-scored marginals (marginal κ4 = 0) the difference kurtosis is a pure fourth-order cross-cumulant of the pair. The post hoc split ([`split.py`](../cumulants/split.py)) put about 90% of the drug change in the fourth-order term along the difference direction.

## Already seen (so this is not blind)

- `split.py`'s `Kdiff` (squared difference kurtosis, net, all edges): drug Δ +.0064 against EC, t(3) +4.0, with MD +.0004 below SLP +.0015. `rk` (signed residual kurtosis) is drug +.111 (t(3) +5.3), SLP +.017 and MD +.031.
- Δz in panel (c)'s units, from the stored zns arrays in this script's smoke test: drug −3.3 eyes-open window SDs; sleep −0.1 (Chibi) and −0.4 (George). Drug − sleep is −4.6 (AUC .93, Chibi) and −1.8 (AUC .82, George). The drug-versus-sleep coefficient is −3.2, falling to −1.8 given the delta fraction.

## Readouts (per window; net = real − the anchor's own IAAFT draw)

- **Primary.** `KD` is the mean over all 496 pairs of net k04, with k04 = scipy.stats.kurtosis(a − b) on the normal-scored window.
- **Secondary.**
  - `KD_real` and `KD_surr`, the two halves of the net.
  - `KDmed`, the median.
  - `KD_loose`, `KD_mid` and `KD_tight`, by within-window |r| tercile.
  - `MARDIA`, Mardia's bivariate excess kurtosis k40 + 2k22 + k04, which is rotation-free and so has no privileged difference direction.
  - `KD_raw`, k04 before normal-scoring, real only. It mixes marginal burstiness in and is shown only to say why normal-scoring is required.
- **Units.** The same as panel (c): each window minus its session's eyes-closed mean, in SDs of the eyes-open windows.

## Tests

1. **Gates.**
   - Net `Kdiff` recomputed from k04 must match `split.py`'s stored values within 1e-5.
   - The glyph pair's k04 must equal panel (a)'s printed 2.37 (2.366 from `figwin`).
2. **Drug effect.** Animal means of drug − EC, t(3), as for the anchor.
3. **Anaesthesia against sleep.**
   - Sleep exists only in Chibi and George, so this is n = 2 animals and is descriptive. There is no inferential test.
   - Per animal, compute drug-unit mean − sleep-unit mean, the window AUC (drug against sleep, in the direction of the drug effect), and MD against sleep.
   - Over both animals' drug and sleep windows, compute the drug coefficient with and without the delta fraction (0.5–4 Hz share) as a covariate, with an animal intercept. Also compute the within-state Spearman of each readout with the delta fraction.
4. **Decision.**
   - `KD` separates anaesthesia from sleep better than Δz only if its AUC exceeds Δz's in both animals.
   - It is less confounded by slow-wave share only if its drug coefficient keeps a larger fraction after the delta adjustment than Δz's does (.56).
   - Otherwise it is a companion level readout, not a replacement for panel (c).

## Reading

- A rise in `KD` means the pairs' differences are heavier-tailed than a Gaussian with the same r. In the character reading, this is non-Gaussian dependence. In the literal reading, the local difference signals become burst-like.
- All three routes produce it: a nonlinear coupling function, a time-varying gain or coupling mixture (the segment surrogates' 1-s mixture does), and linear mixing of non-Gaussian sources (local bursts, K-complexes, burst suppression). It is not evidence of nonlinear coupling.
