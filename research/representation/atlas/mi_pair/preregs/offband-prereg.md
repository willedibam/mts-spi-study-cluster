# Off-band count of fig:mi panel (b): calibration against chance, wake and sleep (frozen 25 November 2026)

Frozen and pushed before any state-wise off-band count was computed. Code: `offband.py` (scoring) and `offband_null.py` (Gadi null draws). Result: [`../offband-result.md`](../offband-result.md).

## Seen before freezing

- The figure docstring: the median eyes-open wake window has 36 of 496 pairs off-band.
- The published panel (b) shows the off-band dots of its five windows; I have not tabulated them.
- Structure, not outcome: G_surr equals G_rec exactly in every window and pair (spectrum-final IAAFT keeps the cross-spectrum, so r), so recording and surrogate residuals sit at the same G pair for pair. On the Mac, IAAFT draw 0 does not reproduce the stored surrogate K (479 of 496 pairs differ, up to .11), so every draw is made on Gadi.

## Definitions (identical to `plot_character_draft.band`)

Per window w (32-channel subset, normal-scored, `zns` `E[:, 1]`): e_rec = K − line_rec(G), the residual about the recording's own least-squares line over its 496 pairs; e_b = K_b − line_b(G) for surrogate draw b about its own line. L_w = 95th percentile of |e_0| (draw 0 is the stored `zns` surrogate). Count c_w = #{pairs: |e_rec| > L_w}; nominal chance 24.8.

The count is not independent evidence for the MI-pair result: c_w and Δz are both functions of the spread of the recording's residuals relative to the surrogate's about a linear G–K fit. The step-1 test re-expresses the Δz result in the figure's units; it is reported with the window-level Spearman correlation of c_w with z_rec − z_surr and of session Δc with session Δz.

## Step 1 (no new compute): primary contrast, one test

- Session Δc = mean c(state) − mean c(awake_ec), state = drug or sleep; eyes-open minus eyes-closed as the control contrast. As `montage/animal_level.py`: drug sessions (KT, KTMD, MD, PF) pooled per animal; one-sided t(3) on the four animal means, H1 Δc > 0, α = .05. Also reported, not tested: sign count, random-animal-intercept model over the 23 drug sessions, each agent, sleep (two animals, so no test; reported as a ratio to the pooled drug Δc, as for Δz), recovery, eyes-open.
- Descriptive per state group (awake_eo, awake_ec, sleep, recovery_ec, KT, KTMD, MD, PF) and per animal and session: mean and median c, excess over 24.8 and over the awake_eo mean of the same animal.

## Step 2 (Gadi): the count's null from surrogate against surrogate

- Draws: for every one of the 1,260 windows, the `edges.py recompute` invocations of `jobs/edges_recompute.pbs` (same files, agents, `first`/`row` modes, seeds), draw 0 = `default_rng(k)` (k the row index in the per-agent call), draws b = 1..99 = `default_rng([k, b])`. Only K is stored per draw (G is the recording's), plus each pair's Theiler window. Tier 2: the five `PANEL_B` windows get draws b = 1..19,999 for per-pair nulls.
- Reproduction gate: draw 0's K matches the stored `zns` surrogate K within 1e-4 (max abs) in every window. If it fails, stop and report; any continuation is a declared deviation.
- Null count: c⁰_{w,b} = #{|e_b| > L_w}, b ≥ 1 (each draw about its own line, against draw 0's limit; draw 0 against itself is excluded, being 25 by construction). Reported: its mean, SD and 2.5–97.5% range per window and pooled, against Binomial(496, .05) SD 4.85.
- Per window: null mean m_w, excess E_w = c_w − m_w, empirical p_w = (1 + #{b: c⁰_{w,b} ≥ c_w}) / 100.
- Decision rule:
  - A state group is "at chance" if at most 10% of its windows have p_w < .05 (twice the nominal 5%), and "in excess" otherwise; its size is the mean E_w and the share of windows with p_w < .05.
  - The figure's claim "anaesthesia shows a large excess, wake and sleep little" holds if (i) the step-1 animal-level test on E_w (in place of c_w) passes, and (ii) the pooled drug mean E_w is positive and at least three times the awake_eo mean E_w and the sleep mean E_w.
  - If awake_eo is "in excess", the wake excess is decomposed (descriptive): by tercile of G and of the pair's Theiler window, recording against null shares.

## Step 3: per-pair procedures (descriptive; each says which estimand it answers)

On all windows with the 99-draw null pooled over pairs (min p 1/(496·99 + 1)), and on the five `PANEL_B` windows with each pair's own 19,999-draw null: Bonferroni and Holm (α = .05 per window), single-step max-|t| (Westfall–Young; |e| standardised by each pair's null SD where per-pair nulls exist, else raw), Benjamini–Hochberg and Benjamini–Yekutieli (q = .05), Storey's π0 (λ = .5) as the estimated non-null share, and an empirical null fitted on the awake_eo windows' standardised recording residuals (Efron-style: central scale by MAD), applied to every state. Reported per state group: mean discoveries per window.

## Step 4: Gaussian parametric null

Before using one: surrogate residuals standardised per window (by SD), pooled by state: QQ against normal, two-sided tail mass beyond the normal 95, 99 and 99.9% points against .05, .01, .001; heteroscedasticity as the Spearman correlation of |e_0| with G per window and the ratio of residual SD in the top to the bottom G tercile. Then the realised family-wise rate of a normal-fit Bonferroni threshold (|e| > Φ⁻¹(1 − .05/992) × SD of draw 0) in the null draws, against .05.

## Not decided here

The Bayesian hierarchical reading (step 5) and the panel (b) recommendation (step 6) are judgement, written in the result.
