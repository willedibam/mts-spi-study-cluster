# Which frequency band carries the macaque MI-pair drug fall? z(G, K) on band-limited 100 Hz windows

Frozen 25 November 2026 at the commit that adds it, before any band arm is built or scored. Exploratory addendum to [`lowrate-prereg.md`](lowrate-prereg.md); it does not replace the 100 Hz anchor. Code: [`../lowrate/bands.py`](../lowrate/bands.py), [`../lowrate/jobs/bands.pbs`](../lowrate/jobs/bands.pbs), scoring by the unchanged `lowrate/score.py` and `jobs/score.pbs`, table by `lowrate/summarise.py`.

## Why

- Changing the sampling rate moves the band, N, the Theiler window in samples and KSG's operating point together. The low-rate arms showed the 10-s fall at 20 and 10 Hz is lost mainly through N (L20 at T = 1000 keeps most of it; D20 = S20 at T = 200 do not). Band is better asked at fixed rate, fixed windows and fixed N.
- Here only the band changes: the anchor's 0.5–40 Hz signal, the bank's own 10-s windows (T = 1000, 100 Hz), the same channels, subsets, seeds, estimator and IAAFT null, with one zero-phase order-8 Butterworth filter on the continuous signal before CAR.
- Mechanism is open. The earlier segment and cumulant tests point to a ~1-s mixture of magnitudes. If that is gain shared across frequencies, the fall appears in low and high bands alike (broadband); if it is the anaesthetic slow wave itself (Chauvette et al. 2011, J. Neurosci. 31:14998), it lives below a few Hz; if it is gain on a faster carrier, it lives above. No prior is adopted.

## Arms (all 100 Hz, 10-s windows, T = 1000, Theiler cap 100)

| Ladder | Arms | Band (Hz) |
|---|---|---|
| Low-pass (primary) | LP2, LP4 = L10, LP8 = L20, LP15, LP25, anchor | 0.5–f |
| High-pass (primary) | anchor, HP2, HP4, HP8, HP15 | f–40 |
| Band-pass (secondary) | LP2, BP2-4, BP4-8, BP8-15, HP15 | octave-like |

LP4 and LP8 are the low-rate L10 and L20, already scored (same filter, same windows); they are not rebuilt. Gains (white noise, checked before freezing): −6 dB at every edge, −45 to −60 dB at 1.5× (or ÷1.5) the edge. Gate A as before: the rebuilt unfiltered windows must equal the bank's, or nothing is written.

## Statistic and reading (stated before output)

Per arm as in `lowrate-prereg.md`: normal-scored residual z_real − z_surr per window (raw reported), per session Δ = state − awake eyes-closed, the arm's own eyes-open session SD, pooled drug Δ in own SD and raw, four animal means, one-sided t(3), eyes-open and sleep controls (flag at 2 own SD), K < 0 fractions and Theiler cap hits by state. Retention against the anchor is reported in raw Δ and in own SD.

- **Retains:** t(3) ≤ −2.35, pooled drug Δ ≤ −2 own SD, controls not flagged. **Lost:** pooled drug Δ within ±1 own SD or t(3) > −1. Otherwise partial.
- **Slow-only:** for some f in {2, 4, 8}, LP_f retains and HP_f is lost. **Fast-only:** HP_f retains and LP_f is lost. **Broadband:** at f = 4 or 8, LP_f and HP_f both retain. Otherwise the two ladders are described, with where retention starts and stops.
- MI is not additive across bands; retentions need not sum to 1, and both halves of a split can each retain.
- **Low-pass arms are oversampled** (K < 0 runs at .32–.51 in L20 and L10 against .17 at the anchor). LP2, LP4 and LP8 are read as above only if the low-rate long arm D20L recovers the fall (t(3) ≤ −2.35); otherwise their retention may be an oversampling effect and they are descriptive. High-pass arms are not oversampled and are read unconditionally.
- **Band-pass arms are read asymmetrically:** retention is evidence that the band carries the fall; loss is not evidence that it does not (few independent samples, below).
- All arms are reported, not the best one. Figure changes are not planned; a ladder panel is proposed only if a slow-only, fast-only or broadband reading results.

## Pre-stated limits

- Independent samples per window fall with bandwidth, about 2 × bandwidth × 10 s: ~30 for LP2, ~40 for BP2-4, ~80 for BP4-8, ~140 for BP8-15, ~500 for HP15. Narrow and low bands lose power regardless of where the structure lives.
- LP2 is the most fragile arm: 5–20 cycles per window, the longest Theiler windows and the worst IAAFT end mismatch.
- Narrowband filtering moves a linear process with non-Gaussian innovations toward Gaussian (Mallows 1967, J. Appl. Prob. 4:313; abstract only). Normal-scoring per window removes marginal shape, so what remains is joint structure the IAAFT null lacks; shared amplitude modulation is such structure.
- Drug spectra are slower, so a band's share of power differs by state. IAAFT matches each window's spectrum, so this enters the residual only through the estimator's spectral sensitivity (the 2-s stationary-null finding: spectrum alone gives 60–65% of the raw fall).
- One surrogate draw per window; 32-channel subsets; four animals (t(3)). Filter edges are −6 dB, not brick walls.

## Compute

Gadi, pushed snapshot `operations/sources/mi-lowrate-<commit>`. Build (30 archives, 30 cpus): expected 0.02 kSU, worst 0.1. Scoring: ten T = 1000 arms at about 60 SU each, in three jobs under the 5-h walltime: expected 0.6 kSU, worst 1.2 kSU.
