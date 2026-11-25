# Robustness of the MI-pair drug Δz: Theiler window, k, window length, sampling rate and bias correction

Frozen 25 November 2026 at the commit that adds it, before any new real-data value (ACF, Theiler sweep, k = 10, 2-s arm) was computed or looked at. Code: [`../robust.py`](../robust.py) (sweep, 2-s arms, calibration, scoring), [`../robust_acf.py`](../robust_acf.py) (ACF, descriptive), [`../jobs/robust10.pbs`](../jobs/robust10.pbs), [`../jobs/robust2.pbs`](../jobs/robust2.pbs). The result goes to [`../robustness-result.md`](../robustness-result.md).

## Why

- One robustness section for the drug Δz of z(G, K), consolidating finished work and computing only what is missing: the Theiler window and k on real data, and the 2-s arm under the shift-corrected KSG.
- The user expects these checks not to hurt the result, and specifically that a decorrelation-informed Theiler (DCE) window makes the effect "strictly better". Both are tested as predictions, not assumed.
- Configurations are **not** chosen by the size of the drug effect. Where a configuration is recommended, it is on drug-blind grounds: synthetic calibration at the real spectra (false-positive offset under stationary coupling, sensitivity to known non-Gaussian dependence) and the eyes-open null spread.

## Real-data looks already taken (disclosed)

- k4 (anchor), kpool, k2te and kshift at 10 s ([`../ksg-bias-result.md`](../ksg-bias-result.md)): R 1.07–1.11, t(3) −5.6 to −5.7 against −6.11. k2te uses the 2τ₁/ₑ window with pool ψ; the 2te arm here uses the same window with ψ(T), so it is not new as a window, only as a ψ form.
- The synthetic mechanism table (w = 0 to 8×AUTO, k = 10 at AUTO, at MI 0).
- The low-rate session's 10-s arms (D20/S20/D10/S10/L20/L10) as summarised in its notes: decimation and subsampling shrink the fall alike; low-pass at full N keeps 69–86%; K < 0 fractions by arm.
- The old 2-s arm: 16 channels, 126-channel joint IAAFT, raw and normal-scored (22/23 sessions down, −.018 to −.024). Nothing at 32 channels and T = 200 has been computed.
- No ACF, τ₁/ₑ or Theiler-window distribution of the real windows has been looked at.

## Arms

Pipeline as `edges.py recompute` throughout: 32-channel subset (`first` for drug-agent calls, `row` for stage-B SLP), normal-scoring, one spectrum-final joint IAAFT draw per window seeded by its row index within the per-agent call, z = corr over the 496 edges of G and K, residual = z real − z surrogate.

| Arm | Data | T | Configurations |
|---|---|---|---|
| 10 s (`rb10`) | stage A and B banks, 1,260 windows, calls as `jobs/edges_recompute.pbs` | 1000 | w ∈ {0, AUTO, 2τ₁/ₑ, 4×AUTO} × k ∈ {4, 10}, ψ(T); pool ψ at k = 4 |
| 2 s, own bank (`rbS2`) | `neurotycho-short`, 1,260 windows, 30 units; one call per agent (`first`; SLP `row`) | 200 | as 10 s, plus kshift (shifts of T/4, T/2, 3T/4 = 0.5, 1, 1.5 s) |
| 2 s, cut (`rbC2`) | the 10-s windows cut into five contiguous 2-s pieces, each with its own surrogate (seed 5 × row + piece) | 200 | as `rbS2` |

- **Windows.** w0 excludes only the point itself. AUTO = min(int(2⟨a_i, a_j⟩), T//10), the `edges.py` window. 2τ₁/ₑ = min(2 max(τ_e,i, τ_e,j), T//10), τ_e the first lag with ACF < 1/e. 4×AUTO = min(4 int(2⟨a_i, a_j⟩), T//4); the larger cap keeps the arm from collapsing onto the cap under drug, where AUTO is about 17–22. Cap-hit fraction and median w are reported by state (awake, drug).
- **ψ form.** Primary K = count term + ψ(T), as `edges.py`: ψ(T) is the same for every edge, so across w only the exclusion changes z. Pool ψ (`ksgbias.pool_psi`) varies per edge with w and is a second manipulation; it is secondary, at k = 4.
- **The cut arm** holds recordings, channels and preprocessing fixed and changes only T. It tests the user's point that more, shorter windows land on the same Δz.
- **Rows taken from finished work, not recomputed:** kpool, k2te and kshift at 10 s from [`../ksg-bias-result.md`](../ksg-bias-result.md); rate and long-window rows from the low-rate session (branch `claude/mi-pair-lowrate-3a8e2a`, scores on gdata `atlas-mi-pair-lowrate-261125/scores/`), scored with its own `summarise.py` and reported with its own prereg verdict. The long arms are labelled "50 s @ 20 Hz" and "100 s @ 10 Hz": duration is confounded with rate and band there. If job 185158077 has not landed, those rows are marked pending.

## Gates

- **10 s:** the AUTO, k = 4, ψ(T) z (real and surrogate) must equal the stored `zns_[AB]_*.npz` normal-scored z within 1e-4, or nothing from `rb10` is read. (On synthetic data `robust.edge_sweep` reproduces `ksgbias.edge_all` k4, k2te, kshift and w exactly.)
- **2 s:** no stored 32-channel T = 200 value exists, so there is no reproduction gate. The 2-s AUTO k4 baseline is itself a new value and is reported as such.

## Statistic

As `ksgbias.py score` and `montage/animal_level.py`: per session, Δ = drug (or sleep) − awake eyes-closed of the window-mean residual; four animal means over drug sessions; one-sided one-sample t(3). Per configuration: pooled drug Δz (raw, and in that configuration's own eyes-open session SD), R = pooled Δz ÷ the arm's AUTO k4 ψ(T) pooled Δz, t(3) and p, per-agent Δz with sessions down, sleep Δz, and a paired t(3) of the configuration's animal means against the arm's AUTO k4.

## Reading (stated before output)

**Per configuration** (as the ksgbias prereg): SURVIVES if 4/4 animal means < 0, one-sided p < .05, every drug agent's Δz < 0 and R ≥ .5; LOST if R < .25 or ≤ 2/4 animal means < 0; PARTIAL otherwise.

**The user's Theiler claim ("strictly better").**
- "Won't remove the effect": SURVIVES at every w (10 s, k = 4, ψ(T)).
- "Better", primary contrast AUTO against w0 at 10 s: better in magnitude if |pooled Δz| in own eyes-open SD is larger at AUTO; better in strength if |t(3)| is larger at AUTO. "Materially" better or worse only if the paired animal-level t(3) of (configuration − AUTO), as `robust.py score` prints it, reaches ±2.35 (one-sided .05 at 3 df): ≥ +2.35 means AUTO's animal means are materially more negative than the configuration's; ≤ −2.35 the reverse.
- Secondary: whether 2τ₁/ₑ and 4×AUTO are better than AUTO, by the same two measures and threshold. "Strictly better" in the user's sense holds only if AUTO beats w0 and no longer window is materially worse.

**The 2-s arms.** Beyond SURVIVES under kshift, the real 2-s pooled drug Δz must be more negative than the stationary drug − awake offset from the T = 200 calibration (below) by more than twice its SE, under the same configuration. Otherwise the 2-s arm is "not distinguished from the estimator's response to drug spectra" and is not cited as robustness.

**Predictions (mine, before output).**
- w0: survives, R .6–1.2; |t| lower than at AUTO. The temporal-neighbour bias at w0 is large (+.10 at MI 0 under drug spectra) but shared with the surrogate; its edge-to-edge variation should add noise to z. So I expect the user's claim to hold for AUTO against w0 in |t|, without reaching the paired ±2.35 threshold.
- 2τ₁/ₑ and 4×AUTO: survive, R .85–1.15, |t| within ±15% of AUTO. The mechanism run shows the bias barely deepening past AUTO, so I expect no gain from longer windows: the claim fails beyond AUTO in the sense of no improvement, not of damage.
- k = 10: survives, R .8–1.2.
- 2-s own bank, k4 and kshift: survive, kshift R .7–1.2 against 2-s k4; own-SD effect smaller than at 10 s. The calibration criterion is the real risk (the slow-short-window null was imperfect in the GC control).
- 2-s cut: survives; raw Δz .5–1.0 of the 10-s value (KSG noise at T = 200 attenuates z in real and surrogate alike); own-SD units closer to 10 s.

## Calibration (synthetic; run in the same job)

`robust.py calib`: `ksgbias_calib.gen` generators `block` (stationary linear coupling within four blocks of four; false-positive offset) and `gain_shared` (one shared gain switch: genuine non-Gaussian dependence on every edge; sensitivity), 16 channels with periodograms of real stage-A normal-scored channels (awake eyes-closed, KTMD drug, PF drug), T = 1000 and T = 200 (first 200 samples), n = 40 per cell, each with its own surrogate, every configuration including kshift. Reported per configuration: z real − surrogate (block) ± SE per spectrum; the stationary drug − awake offset (mean of KTMD and PF minus awake); detectability = (block − gain_shared) / SD(block). The drug-blind recommendation of a configuration uses only these numbers and the eyes-open SD.

## ACF (descriptive, no decision rule)

`robust_acf.py`, on the normal-scored 10-s windows and the same 32-channel subsets (what the estimator sees):
- ACF per window and channel (`edges.acf`), averaged over the windows of each unit × state × channel; lag axis in seconds.
- τ₁/ₑ: first crossing of 1/e, linearly interpolated, on the averaged ACF and per window (the latter is what AUTO and 2te use).
- Exponential check, only up to the first 1/e crossing or zero: the fitted e-folding time of log ACF over lags up to τ₁/ₑ, and the ACF at 2τ₁/ₑ against e⁻² ≈ .135. An oscillatory ACF (negative lobe below −.1 within 1 s, or a secondary peak) is flagged; there the 1/e crossing is a quarter-period, not a decay time, so the 1/e crossing of the ACF upper envelope (max |ACF| at that lag or later) and the integral to the first zero are reported beside it. No full-sum integral time: the 0.5 Hz high-pass makes the two-sided ACF integral ≈ 0.
- Distributions by state (each drug agent, sleep, awake EC/EO, recovery) of τ₁/ₑ and 2τ₁/ₑ, mapped to samples at 100, 20 and 10 Hz; N_eff ≈ T/(2τ₁/ₑ) per channel and T/w_AUTO per pair, for 2-, 10-, 50- and 100-s windows at each rate, with the w in use and its cap T//10.
- Figure per `docs/figure-style.md`.

## Limits

- One surrogate draw per window; 32-channel subsets; four animals.
- kshift at 2 s assumes no dependence survives a shift of 0.5 s; the segment test found structure on about 1-s scales, so a shared slow component would be partly subtracted (pushes R down).
- The 50-s and 100-s rows confound duration with rate (and, for D arms, with band). No 50-s window at 100 Hz is computed.
- The axes are varied one at a time from the anchor, not crossed; interactions (for example w × T) are tested only where an arm crosses two axes (2-s sweep).

## Compute

Gadi, pushed snapshot `operations/sources/mi-robust-<commit>`, two jobs of 48 cores. 10-s sweep: expected 0.17 kSU (about 1.8 h), worst 0.38 kSU (4 h walltime). 2-s arms and calibration: expected 0.13 kSU, worst 0.38 kSU (4 h). In all: expected 0.3 kSU, worst 0.8 kSU.
