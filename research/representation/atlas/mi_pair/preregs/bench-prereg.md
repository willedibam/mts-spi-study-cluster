# Prereg: conventional anaesthesia and coupling metrics against Δz (26 November 2026, frozen before any output)

**Why.** Slow-wave share separates drug from sleep better than Δz, but it also moves sleep a third of the way to drug (`delta_panel.py`: sleep +1.2, drugs +2.6 to +3.8 wake SDs; Δz sleep −0.2, drugs −1.5 to −3.9). The question is whether any common metric does both things Δz does: move the same way under every agent, and not move sleep toward drug. This is post hoc with respect to Δz (its values are known), so it is a benchmark, not a confirmation of Δz.

**Code.** [`../bench.py`](../bench.py). Same windows and 32-channel subsets as `edges.py recompute` and `zns` (rowmode `first` per agent group, per-row for stage B sleep), duplicates dropped by window id. Every metric is a channel mean (univariate) or a pair mean over the 496 pairs (coupling).

## Metrics (fixed list)

- **Spectral** (Welch, 4-s segments): `SW` slow-wave share (0.5–4 over 0.5–40 Hz, recomputed on the subset); `ALPHA` relative 8–13 Hz power; `BR` BIS beta-ratio proxy log(P30–40 / P11–20) (BIS uses 30–47 Hz; the bank is low-passed at 40 Hz); `SEF95` spectral edge frequency; `SE` spectral entropy over 0.8–32 Hz, normalised (the GE Entropy state-entropy band); `AP` aperiodic slope, least squares of log P on log f over 20–40 Hz (Lendner et al. 2020 fit 30–45 Hz; truncated here).
- **Burst suppression** `BSR`: share of 0.5-s segments whose channel-mean RMS is below 25% of the session's eyes-closed median segment RMS (relative threshold, because absolute µV calibration is not used here; BIS uses ±5 µV).
- **Complexity:** `LZC` Lempel–Ziv (LZ76) complexity of the median-binarised channel, normalised by n / log2 n (Schartner et al. 2015); `PE` permutation entropy, order 3, lag 1, normalised.
- **Coupling:** `ABSR` mean |r|; `EIG` top-eigenvalue share of the correlation matrix; `COHD`, `COHA` mean magnitude-squared coherence over 1–4 and 8–13 Hz; `WPLID`, `WPLIA` weighted phase-lag index over the same bands (volume-conduction-robust); `KURA` time-mean Kuramoto order parameter of 8–13 Hz Hilbert phases; `WSMI` weighted symbolic MI, order 3, τ = 4 samples (King et al. 2013).
- **MI levels without a surrogate** (from `zns`): `MG` mean Gaussian MI, `MK` mean KSG MI, `MKG` mean K − G.
- **Test metric:** `DZ`, panel (c)'s Δz.

Mean signed r is excluded: under the common average reference it is pinned near zero.

## Units and contrasts

- Every metric is taken as window minus its session's eyes-closed mean, in eyes-open SD units (panel (c)), then oriented so that the pooled drug session mean is positive.
- Cells: agent by animal (10 drug cells: Chibi and George × KT, KTMD, MD, PF; Kin2 and Su × KTMD), session means first.

## Win criterion (both required)

- **C1, consistent drug effect:** all 10 drug cells have the pooled sign.
- **C2, sleep not moved toward drug:** sleep/drug ratio (sleep session mean over drug session mean) with a 90% bootstrap CI (10,000 resamples of sessions), upper bound below .25 (the margin of `mens-prereg.md`).

A conventional metric that passes both is reported as matching Δz on this criterion.

## Reported, not part of the criterion

- Window AUCs per animal (Chibi, George), oriented: drug against sleep, and sleep against same-session eyes closed (> .5 means sleep moves toward drug).
- Matched slow-wave test: in each animal's overlap of drug and sleep `SW` ranges, AUC drug against sleep with n and a 90% bootstrap CI over windows. Every metric goes through it, so a null for Δz can be read against the others (power).
- For metrics that pass C1: the session bootstrap of the difference in sleep/drug ratio against Δz's.
- Two sleep animals: no animal-level sleep claim. Window AUCs are descriptive.

## Not decided here

The matched-delta question, which needs the staged archive hold-out. This benchmark cannot settle it.
