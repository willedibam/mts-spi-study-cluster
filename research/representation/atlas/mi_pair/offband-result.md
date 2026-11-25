# Off-band count of fig:mi panel (b): chance, wake and sleep (25 November 2026)

Rules frozen in [`preregs/offband-prereg.md`](preregs/offband-prereg.md) (commit `6c88ce42`) before any state-wise count was computed. Code: `offband.py` (scorer; `python offband.py <zns dir> [<offband dir>]`), `offband_null.py` + `jobs/offband.pbs` (Gadi draws, snapshot `mi-offband-bbde1273`), `plot_offband_ref.py` (mock-ups). Inputs: the checksum-listed `zns` arrays (`E[:, 1]`, KSG k = 4) and the new draws on gdata `.../atlas-mi-pair-261117/offband/` (checksums, scorer output and mock-ups in [`results/.../offband/`](../../../../results/representation/atlas-mi-pair-261117/offband/)). The scorer reruns unchanged on revised `zns`; the null steps need new draws if K changes. Items marked *post hoc* were added after step 1 was seen and are not covered by the prereg.

## Answer

- **The frozen rule does not support "wake and sleep at chance".** Under surrogate against surrogate the count averages 25.9 per window (near-binomial). Eyes-open wake has +7.9 excess pairs and 35% of its windows have p < .05; sleep has +12.0 and 46%. Every state is "in excess" by the prereg's ≤ 10% rule. Wake's excess is real and uniform. It is a 5% wider recording scatter about the G–K line than the surrogate's (SD ratio 1.055), so it reads as a small departure from linear-stationary-Gaussian dependence. Ruled out (step 2): non-exchangeable surrogates (the null is near-binomial), dependence on G, and the recording–surrogate Theiler mismatch. Not tested: a known linear-Gaussian recording run through the whole pipeline, so a residual estimator mismatch between recordings and their surrogates is not excluded.
- **The relative claim holds.**
  - The animal-level test on the excess passes: t(3) = 5.03, p = .0076, 4/4 animals, 23/23 drug sessions.
  - The ratio rule passes only on a knife edge: pooled drug excess +36.1 against 3 × sleep +36.08. That pooling is window-weighted, and sleep's mean is driven by George.
  - The robust reading is session-matched: against the same session's eyes-closed wake, anaesthesia adds +27.9 ± 5.5 excess pairs (animal mean ± SE, the frozen statistic) and sleep +2.7 (7/7 sessions up, 2 animals), a ratio of 0.10.
  - So the supportable sentence is: **wake and sleep show a small excess over the surrogate; anaesthesia an excess 3–5 times larger** (pooled drug E 3.0× sleep's and 4.6× wake's; session-matched, about 10× sleep's).
- **Which procedure answers the supervisor's question: the count-level null (step 2) plus the animal-level contrast, not a per-pair correction.**
  - The question concerns how many pairs exceed chance per state. Bonferroni, Holm and Westfall–Young answer a different question: which individual pairs are off with family-wise confidence.
  - Here the excess is a diffuse widening of the residual scale, not a few outlying pairs. Those procedures therefore find under 1 pair per window in wake and sleep and 1–11 under anaesthesia (step 3). That is correct for their estimand, but it hides the effect the panel shows.
  - The off-band count is Δz in pair units (Spearman −.83 with Δz), not independent evidence. It also does not count nonlinear pairs.
- **A Gaussian parametric null is accurate to the 1% tail but not at Bonferroni depth.** At that depth it is anti-conservative (realised FWER .09–.23 against .05; step 4).
- **Bayesian: not fitted, by design.** There is no pair or channel structure to pool on (step 5). The state-level question is already answered by the animal-level model.

## Step 1: counts by state and the level test (no new compute)

Count c per 10-s window = recording pairs outside the surrogate's 95% band about the recording's own G–K line (`band()`), of 496 pairs; nominal chance 24.8.

| State | Windows | Mean c | Median | Over 24.8 | Over own-animal wake | Above wake 95th pct (54) |
|---|---|---|---|---|---|---|
| Eyes-open wake | 300 | 33.8 | 32 | +9.0 | 0 | (5%) |
| Eyes-closed wake | 300 | 33.7 | 31 | +8.9 | −0.0 | .047 |
| Recovery | 220 | 41.2 | 38 | +16.4 | +7.3 | .173 |
| Sleep | 210 | 38.0 | 36 | +13.2 | +4.1 | .090 |
| Ketamine | 40 | 96.5 | 85 | +71.7 | +62.2 | .775 |
| Ket. + med. | 110 | 56.3 | 54 | +31.5 | +23.2 | .500 |
| Medetomidine | 40 | 48.0 | 46 | +23.2 | +13.7 | .225 |
| Propofol | 40 | 58.5 | 58 | +33.7 | +24.2 | .575 |

- Per animal, the wake mean is 30.6–37.4; George is highest in every no-drug state.
- The panel (b) windows and their percentiles within state:

  | State | Panel window c | Percentile within state |
  |---|---|---|
  | Wake | 35 | 58th |
  | Sleep | 26 | **17th** |
  | Ketamine | 70 | 38th |
  | Ket. + med. | 49 | 30th |
  | Propofol | 50 | 28th |

  The sleep window understates sleep, which is the point raised under panel (b).
- **Level test (frozen, one test).**
  - Session Δc = state − eyes-closed wake, with drug sessions pooled per animal: Chibi +43.0, George +17.2, Kin2 +25.0, Su +27.5.
  - One-sided t(3) = 5.22, p = .0068. The sign test is 4/4, p = .0625, which is its floor.
  - 23/23 drug sessions are up. Animal mean +28.2 ± SE 5.4; a random-animal-intercept model gives +28.5 ± 8.1 (converged).
  - By agent: ketamine +62.2 (4/4), ket. + med. +23.8 (11/11), propofol +22.9 (4/4), medetomidine +13.7 (4/4).
  - Sleep +4.2 (7/7 up; Chibi +2.4, George +6.6), 0.15 of the pooled drug change. Recovery +7.3. The eyes-open control per animal is −2.6 to +2.1.
- **The count is Δz in other units, not new evidence.**
  - Window Spearman(c, z_rec − z_surr) = −.83 (−.75 within state groups); session Spearman(Δc, Δz) = −.85.
  - *Post hoc:* c tracks the ratio of the recording's residual SD to the surrogate's (Spearman +.91). Median ratios: wake 1.055, sleep 1.085, recovery 1.10, medetomidine 1.17, ket. + med. and propofol 1.26, ketamine 1.45. A normal residual with that SD alone predicts the count within 1–9 pairs per state.
  - So an off-band count is the residual scale of 1 − z², read through a fixed tail cut. It shows Δz's change in a pair-countable unit; it does not add an independent test.

## Step 2: the count under surrogate against surrogate (Gadi, 99 extra draws per window)

Each window gets 99 further joint IAAFT draws (draw 0 = the stored surrogate; the reproduction gate passes with 0 deviation over 1,260 windows). Each draw's residuals are taken about its own G–K line and counted against draw 0's limit, as `band()` does.

- **Null.** The pooled null count has mean 25.87 and within-window SD 5.22. The binomial SD is 4.85, so the inflation is 1.08: the 496 pairs are nearly exchangeable. The null correlation of residuals is +.001 to +.015 for pairs sharing a channel and −.004 for disjoint pairs (tier 2). The single-window 2.5–97.5% range is [13, 41]. The mean exceeds 24.8, probably (inference) because draw 0's limit is itself a noisy 95th percentile, applied out of sample.
- **Per state.** E = c − null mean; "share p < .05" is each window's count against its own 99 draws.

  | State | Null mean | c | E | Share p < .05 | Frozen verdict |
  |---|---|---|---|---|---|
  | Eyes-open wake | 25.9 | 33.8 | +7.9 | .350 | in excess |
  | Eyes-closed wake | 25.7 | 33.7 | +8.0 | .327 | in excess |
  | Recovery | 25.6 | 41.2 | +15.6 | .591 | in excess |
  | Sleep | 26.0 | 38.0 | +12.0 | .457 | in excess |
  | Ketamine | 25.7 | 96.5 | +70.8 | .975 | in excess |
  | Ket. + med. | 26.1 | 56.3 | +30.2 | .964 | in excess |
  | Medetomidine | 26.2 | 48.0 | +21.9 | .800 | in excess |
  | Propofol | 26.6 | 58.5 | +31.9 | .950 | in excess |

- **Decision (frozen).**
  - (i) Animal test on ΔE: Chibi +43.2, George +17.5, Kin2 +22.7, Su +28.1; t(3) = 5.03, p = .0076; animal mean +27.9 ± SE 5.5. The mixed model (+28.3 ± 8.0) did not converge, so it is not quoted. 23/23 sessions up. Eyes-open control −1.9 to +1.5.
  - (ii) Pooled drug E +36.1 ≥ 3 × wake +23.6 (4.6×) and ≥ 3 × sleep +36.08 (3.003×). **The claim holds**, by 0.04 pairs.
  - Only ketamine clears 3 × sleep alone: ketamine 5.9×, ket. + med. 2.5×, propofol 2.7×, medetomidine 1.8×.
  - Sleep's E is George-heavy (20.3 against Chibi 5.9), and George's wake E is 11.7.
  - The session-matched contrast removes this: sleep ΔE +2.7 (Chibi +1.1, George +5.0) against drugs +28.3, ratio 0.10. This is the number to quote.
- **Is the wake excess non-exchangeability? No.**
  - Off-band share, recording against null, by tercile within window:
    - *By G:* recording higher in every tercile (wake .066/.066/.072 against .052), with the null flat.
    - *By the recording–surrogate Theiler difference W_rec − W_surr0:* flat (wake .067/.066/.068). The Theiler windows differ for 45% of pairs, but this does not produce the excess.
  - So the wake excess is a real recording-versus-surrogate scale difference, spread over all pairs.
- **Theiler window, a caveat on which pairs are highlighted.**
  - The null exceedance under the pooled limit rises with the pair's own Theiler window. Wake goes .050 → .055; under anaesthesia it is much steeper: ket. + med. .033 → .074, propofol .031 → .077. The recording rises with it.
  - So in drug windows, the pooled limit preferentially marks high-autocorrelation pairs, whose K is noisier.
  - *Post hoc check of the total (`offband.py` step 2).* Each pair is standardised by its own null SD over draws 1–99 and counted against the pooled 95% limit of those standardised draws. Draw 0 is held out of both steps and scored like the recording, which gives the null of this count (25.4–26.8 by state).
    - Recording counts: wake 34.0, sleep 38.3, ketamine 96.5, ket. + med. 59.7, medetomidine 49.4, propofol 61.4.
    - Excess over the held-out null: wake +7.8, sleep +12.2, ketamine +70.2, ket. + med. +33.9, medetomidine +22.6, propofol +36.0. Against E: +7.9, +12.0, +70.8, +30.2, +21.9, +31.9.
    - The level test holds: t(3) = 5.99, p = .0047; animal mean +30.2 ± SE 5.1; sleep +4.1.
  - The count is therefore robust to this heteroscedasticity. The identity of the marked pairs in drug panels is not.

## Step 3: corrections compared

Per pair, p = the share of the window's null residuals (|e| over all 496 pairs and 100 draws, 49,600 values) at or above the recording's |e|. The minimum p is 2.0e-5, so Bonferroni's .05/496 = 1.0e-4 is reachable; the supervisor's .05/300 = 1.7e-4 is the same rule for 300 tests. Mean discoveries per window:

| Procedure | Estimand | Wake (EO) | Sleep | Ketamine | Ket. + med. | Medetomidine | Propofol |
|---|---|---|---|---|---|---|---|
| Off-band count c | pairs beyond the 95% surrogate band | 33.8 | 38.0 | 96.5 | 56.3 | 48.0 | 58.5 |
| Bonferroni / Holm (α .05) | pairs individually off, FWER ≤ .05, any dependence | 0.37 | 0.76 | 10.8 | 1.66 | 1.28 | 1.93 |
| Westfall–Young max-\|e\| | the same, using the null's dependence | 0.42 | 0.84 | 11.4 | 1.87 | 1.33 | 2.08 |
| Benjamini–Hochberg (q .05) | a pair list with ≤ 5% false share (positive dependence) | 0.55 | 1.56 | 31.4 | 4.30 | 3.25 | 3.98 |
| Benjamini–Yekutieli | the same, any dependence | 0.06 | 0.39 | 14.3 | 0.85 | 0.85 | 0.83 |
| Storey 1 − π̂0 (λ .5) | **share of pairs not null** (count-level) | .049 | .070 | .287 | .165 | .139 | .158 |
| Empirical null, wake-pooled (*post hoc*) | pairs beyond 1.96 × wake's central scale (σ0 = 1.052) | 26.6 | 30.5 | 85.5 | 48.5 | 40.4 | 50.3 |
| Empirical null, window's own bulk (*post hoc*) | pairs beyond the window's own central scale | 26.3 | 27.4 | 29.8 | 32.4 | 27.0 | 32.8 |

- **Effective number of tests.** Null correlations between pairs are about 0 (above) and the count's SD is only 1.08× binomial. An eigenvalue-based M_eff would therefore be about 496, and that correction equals Bonferroni here. No citation is used for this.
- **Tier 2: the five panel (b) windows, each pair with its own 20,000-draw null.** Minimum p is 5e-5; the gate passed.

  | Window | c | Bonferroni / Holm | Westfall–Young (per-pair standardised) | BH | BY | Storey 1 − π̂0 | Per-pair null SD CV |
  |---|---|---|---|---|---|---|---|
  | Wake | 35 | 1 | 1 | 1 | 0 | .052 | .013 |
  | Sleep | 26 | 0 | 0 | 0 | 0 | .069 | .013 |
  | Ketamine | 70 | 7 | 6 | 25 | 7 | .25 | .061 |
  | Ket. + med. | 49 | 4 | 3 | 10 | 0 | .141 | .209 |
  | Propofol | 50 | 0 | 0 | 0 | 0 | .036 | .174 |

  - Per-pair and pooled p agree in rank (Spearman .97–1.00).
  - In the drug windows the per-pair null SD varies across pairs (CV .17–.21 against .013 in wake and sleep; the Theiler effect above). So the per-pair null moves the propofol window's excess off any single pair: it has 50 off-band pairs but nothing individually significant. Its Storey estimate, .036, is a single-window value and noisy.
- **Verdicts.**
  - **Bonferroni / Holm / Westfall–Young.** These are valid and answer "which pairs, with family-wise confidence". They find about 1 pair per window except under ketamine. That is the right tool only for a figure that names pairs; panel (b) does not.
  - **BH / BY.** These answer "which pairs to list with a controlled false share". The same objection applies, and BY is near zero except under ketamine.
  - **Storey's 1 − π̂0.** This is the one per-pair procedure whose estimand matches the question (how much of the family is non-null). Its state ordering matches c, but it is noisy per window and adds nothing over E.
  - **Empirical null** ([Efron 2004](https://doi.org/10.1198/016214504000000089)). Estimating the null from the window's own bulk absorbs the drug effect: ketamine falls from 96.5 to 29.8. This is the decisive diagnostic. The anaesthesia excess is mostly a wider bulk scatter, plus a small tail remainder (26–33 against 25). A wake-pooled σ0 is the reference-(2) idea, developed below.
  - **Count-level null (step 2) with the animal-level contrast.** These answer the question as asked.

## Step 4: is a Gaussian parametric null usable?

Surrogate residuals (draw 0), standardised per window by their SD, two-sided tail mass against the normal:

| State | > 1.96 (.05) | > 2.58 (.01) | > 3.29 (.001) | > 3.89 (1.0e-4) | Excess kurtosis |
|---|---|---|---|---|---|
| Eyes-open wake | .0495 | .0098 | .0010 | .0001 | −0.00 |
| Sleep | .0493 | .0101 | .0011 | .0002 | 0.04 |
| Ket. + med. | .0515 | .0123 | .0019 | .0003 | 0.23 |
| Propofol | .0511 | .0131 | .0025 | .0004 | 0.31 |
| All 1,260 windows | .0499 | .0105 | .0013 | .0002 | 0.08 |

- **No heteroscedasticity in G.** The median Spearman(|e|, G) is −.01 to +.01 in every state, and the SD of the top G tercile over the bottom is .97–1.02. So pooling over pairs within a window is justified with respect to G. It is not justified with respect to the pair's autocorrelation (Theiler window, step 2), though the window total is unaffected.
- **Tails.** The normal is accurate to the 1% tail in every state and to the 0.1% tail in wake and sleep. Drug-window surrogates are heavier beyond that: 2–2.5× at .001 and about 3–5× at 1e-4, though the counts there are few.
- **Realised FWER of a Gaussian Bonferroni cut** (frozen check: |e| > Φ⁻¹(1 − .05/992) × draw-0 SD, counted over the null draws):
  - Over all windows (99 draws): wake .086, sleep .119, recovery .139, ketamine .131, medetomidine .140, ket. + med. .190, propofol .227.
  - Tier 2 (20,000 draws): wake .045, sleep .056, ketamine .070, ket. + med. .165, propofol .335.
- **Verdict.** A Gaussian null is adequate for the 95% band that `band()` draws. At Bonferroni depth it is anti-conservative in drug windows by up to 7× (propofol). Part of the all-window excess over tier 2 probably comes from the noise in a single draw's SD (inference). Use the surrogate draws, not Φ, for any per-pair claim.

## Step 5: Bayesian hierarchical reading

- **Not fitted.**
  - No sampler is in the pinned venv (pymc, bambi and numpyro are absent). Adding one is an environment change outside this task.
  - The pair-level model would also have nothing to pool on. A hierarchical model of pair-level residual magnitude would partially pool pairs through channel or window effects ([Gelman, Hill & Yajima 2012](https://doi.org/10.1080/19345747.2011.618213)). That structure is absent here:
  - A channel-additive model of log u² explains R² .063–.088 against .063 at chance, with the same for surrogates.
  - The between-window share of var(log u²) is .011.
  - Partial pooling would therefore shrink each pair to its window's scale, which is the step-1 SD ratio.
- **What it would answer.** The multiplicity-free reading Gelman et al. advocate applies to the state contrast.
- **Closed-form posterior instead.** For the frozen animal-level statistic, the reference prior (flat on the mean and on log σ) gives a Student-t posterior for the animal mean. Computed from the printed animal means:
  - drug ΔE: 27.9, 95% interval [10.3, 45.5];
  - sleep ΔE: 3.0, 95% interval [−21.8, 27.8] (two animals, so t(1)).
  - So, at the animal level, sleep's excess is not tightly bounded. The 0.10 ratio rests on 7/7 sessions within two animals, and the figure text should say "two animals".
- **Scoped proposal, if a reviewer asks for posterior uncertainty with four animals (two for sleep).**
  - Model: log(SD ratio) per window ~ state + (1 | animal) + (1 | session), with weakly informative priors, in PyMC or brms. It runs in minutes on a laptop and needs no new draws.
  - Output: a posterior for the drug-to-sleep ratio of scale excess.
  - Expectation (inference): it will reproduce the 0.10 ratio, with wide animal-level intervals. It is not needed for panel (b).

## Reference (2): beyond the awake state (*post hoc*, coordinator's request)

- **Construction.**
  - Each window's recording residuals are expressed in its own surrogate's SD (u = |e_rec| / SD(e_0)), which matches state-specific estimator noise.
  - u is counted against the 95th percentile of u pooled over the same session's eyes-open wake windows. Each eyes-open window is left out of its own reference.
  - Wake is ≈ 24.8 by construction. The leave-one-out eyes-open spread is the reference's null: mean 25.0, SD 7.5, 95% range [13, 42].
  - The wake limit is a median 2.08 surrogate SDs, against 1.95 for the surrogate's own 95%. That gap is the wake excess of step 2, now absorbed.

Share of pairs = mean count / 496. Null: surrogate 26 (5.2%); wake 25 (5.0%).

| State | Surrogate ref.: c (E; share p < .05) | Wake ref.: mean (over 24.8; share > wake 97.5%) | Share of pairs (surrogate / wake) |
|---|---|---|---|
| Eyes-open wake | 33.8 (+7.9; .35) | 25.0 (+0.2; .020) | 6.8% / 5.0% |
| Eyes-closed wake | 33.7 (+8.0; .33) | 25.2 (+0.4; .033) | 6.8% / 5.1% |
| Recovery | 41.2 (+15.6; .59) | 32.1 (+7.3; .145) | 8.3% / 6.5% |
| Sleep | 38.0 (+12.0; .46) | 26.5 (+1.7; .057) | 7.7% / 5.3% |
| Ketamine | 96.5 (+70.8; .98) | 79.9 (+55.1; .700) | 19.5% / 16.1% |
| Ket. + med. | 56.3 (+30.2; .96) | 48.8 (+24.0; .682) | 11.4% / 9.8% |
| Medetomidine | 48.0 (+21.9; .80) | 39.1 (+14.3; .325) | 9.7% / 7.9% |
| Propofol | 58.5 (+31.9; .95) | 46.5 (+21.7; .575) | 11.8% / 9.4% |

- **Level test, same form as step 1.**
  - Δ wake-reference count by animal: Chibi +41.3, George +14.7, Kin2 +23.7, Su +25.5; t(3) = 4.75, p = .0088. Animal mean +26.3 ± SE 5.5; mixed model +26.6 ± 7.0 (converged); 23/23 sessions up.
  - Sleep +2.6 (7/7; Chibi +2.2, George +3.1), ratio 0.10.
- **Per pair**, BH at q .05 against the session's eyes-open pool: wake 0.03, sleep 0.15, ketamine 24.0, ket. + med. 2.3 discoveries per window.
- **Panel (b) windows:** wake 22, sleep 26, ketamine 66, ket. + med. 44, propofol 37.

**Which claim each reference supports.**

- **(1) Surrogate reference: "departure from linear-stationary-Gaussian dependence" at the pair level.** It is absolute. Wake departs slightly, sleep slightly more, and anaesthesia 3–5 times as much excess (step 2). It is the reference `band()` already draws and that z is built on, and it is preregistered.
- **(2) Wake reference: "pair-level G–K scatter beyond the awake state's".** It is relative. It cannot speak to Gaussianity, since wake is 0 by construction, and it restates the animal-level Δ of reference (1) almost exactly (animal means +26.3 against +27.9). Its added value is visual: sleep sits on the wake band, while anaesthesia leaves it.

## Panel (b): recommendation

- **Main figure: keep reference (1).**
  - Draw the strip of counts by state (mock-up [`offband-strip.png`](../../../../results/representation/atlas-mi-pair-261117/offband/offband-strip.png), `plot_offband_ref.py --null`). Use the surrogate-against-surrogate mean (26) and single-window 95% range [13, 41] in place of the nominal 24.8.
  - Quote the session-matched contrast in the caption.
- **Replace the sleep example window.**
  - c = 26 sits at the 17th percentile of sleep windows (median 36), so it shows sleep at chance, which the data do not support.
  - Pick each state's example near its median, or state the percentile. This is a `PANEL_B` choice in `plot_character_draft.py`, which this session has not edited.
- **Supplement: reference (2)**, as the right half of the side-by-side mock-up ([`offband-both.png`](../../../../results/representation/atlas-mi-pair-261117/offband/offband-both.png), `plot_offband_ref.py --null <dir> --both`).
  - In the top row, the filled points are off the surrogate band (shaded) and the ringed points are off the wake band (dashed); titles give "surrogate / wake" counts.
  - The reverse placement (wake reference in the main figure) would suit a text whose claim is "anaesthesia, not sleep, leaves the awake state". It is post hoc, though, and the same contrast is already in the main figure's matched numbers.
- **Do not add per-pair significance marks.** Bonferroni, Holm and Westfall–Young mark about 1 pair per window outside ketamine. In drug windows, which pairs are marked depends on pair autocorrelation (step 2).
- `plot_character_draft.py` is untouched; the mock-ups are a separate script.

Caption, reference (1) (main):

> (b) G against K for the 496 channel pairs of one 10-s window per state. Shading: the 95% band of the window's joint IAAFT surrogate about its own G–K line; coloured points lie outside it. Strip: the count of such pairs in every window. Surrogate against surrogate, the count averages 26 (dashed; 95% of single windows 13–41). Wake and sleep exceed it slightly but consistently (mean excess 8 and 12 pairs; 35% and 46% of windows above their own surrogate null at p < .05, against 5% expected). Anaesthesia exceeds it 3–5 times as much (excess 22–71 pairs; 80–98% of windows). Against the same session's eyes-closed wake, anaesthesia adds 28 ± 6 pairs (animal mean ± SE; 23/23 sessions, 4 animals) and sleep 3 (7/7 sessions, 2 animals). Off-band pairs mark departures from linear-stationary-Gaussian dependence, not a count of nonlinear pairs. The excess is mostly a wider scatter about the G–K line, the same change that lowers z.

Caption, reference (2) (supplement):

> Off-band pairs against the awake state. Each window's G–K residuals, in units of its own surrogate's SD, are counted against the 95th percentile of the same session's eyes-open wake windows; each eyes-open window is left out of its own reference. Eyes-open wake is therefore about 25 by construction (95% of windows 13–42). Sleep stays on the wake band (mean 26.5), while anaesthesia leaves it (39–80; ketamine highest). This shows how far each state's pair-level scatter exceeds the awake state's; whether that scatter departs from Gaussian dependence is shown by panel (b) against its surrogate.

## Compute

Gadi, project ql44:

| Job | Configuration | Cost |
|---|---|---|
| Gate | — | small (not itemised) |
| Tier 1 | 10 × 48 cores, ~84 min | ~1.39 kSU |
| Tier 2 | 5 windows × 20,000 draws, ~2.3 h | ~1.13 kSU |

Total ~2.5 kSU, against 1–2 expected and 4.8 worst case. The tier-2 runtime was underestimated.

## Literature (each checked 25 November)

- Bonferroni-type FWER: [Holm 1979](https://doi.org/10.2307/4615733), valid under any dependence; max-statistic resampling: [Westfall & Young 1993](https://www.wiley.com/en-us/9780471557616) (Wiley book), [Nichols & Holmes 2002](https://doi.org/10.1002/hbm.1058).
- FDR: [Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) (valid under positive regression dependence, [Benjamini & Yekutieli 2001](https://doi.org/10.1214/aos/1013699998), whose modification is valid under any dependence); non-null share: [Storey 2002](https://doi.org/10.1111/1467-9868.00346).
- Empirical null: [Efron 2004](https://doi.org/10.1198/016214504000000089) (the null estimated from the bulk of the family, often wider than the theoretical one).
- Partial pooling instead of correction: [Gelman, Hill & Yajima 2012](https://doi.org/10.1080/19345747.2011.618213).
- Edge-wise inference on networks with component-level FWER: [Zalesky, Fornito & Bullmore 2010](https://doi.org/10.1016/j.neuroimage.2010.06.041) (NBS); relevant only to a "which connected set of pairs" claim, which the figure does not make.
- Surrogate-null reading ("dependence beyond the linear-Gaussian expectation"): [Theiler et al. 1992](https://doi.org/10.1016/0167-2789(92)90102-S), as cited in `docs/interpreting-spi-pairs.md`.
