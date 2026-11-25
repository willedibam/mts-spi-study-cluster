# Result: negative KSG MI in panel (b), and whether the drug Δz survives a bias-corrected KSG (25 November 2026)

- **Rule:** [preregs/ksgbias-prereg.md](preregs/ksgbias-prereg.md), frozen at 17556ab8 before any corrected real-data value was read.
- **Code:** [`ksgbias.py`](ksgbias.py) (estimators, scoring), [`ksgbias_calib.py`](ksgbias_calib.py) (calibration, mechanism), [`jobs/ksgbias.pbs`](jobs/ksgbias.pbs), [`jobs/ksgbias_calib.pbs`](jobs/ksgbias_calib.pbs).
- **Gadi jobs**, both run from the snapshot `ksgbias-17556ab8`:
  - real rescore 185158310: 134.5 SU, 1 h 24 min;
  - mechanism run 185158311: 2.8 SU.
- **Outputs:**
  - [`score.txt`](../../../../results/representation/atlas-mi-pair-261117/ksgbias/score.txt), [`mech.txt`](../../../../results/representation/atlas-mi-pair-261117/ksgbias/mech.txt) and [`calib.txt`](../../../../results/representation/atlas-mi-pair-261117/ksgbias/calib.txt) are in `results/representation/atlas-mi-pair-261117/ksgbias/`.
  - The per-window arrays `kb_*.npz` are on gdata at `.../atlas-mi-pair-261117/ksgbias/`, checked against `SHA256SUMS`.

## Verdict: SURVIVES

The reproduction gate passed: k4 z matches the stored `zns` z to a maximum difference of 7.5e-9.

| Estimator | Animal means | t(3), one-sided p | Pooled drug Δz | R vs k4 | KT / KTMD / MD / PF Δz |
|---|---|---|---|---|---|
| k4 (as `edges.py`) | −.019 −.009 −.012 −.014 | −6.11, .0044 | −.0133 | 1 | −.0159 / −.0155 / −.0062 / −.0117 |
| **kshift (primary)** | −.022 −.010 −.012 −.013 | **−5.73, .0053** | −.0148 | **1.11** | −.0154 / −.0176 / −.0064 / −.0148 |
| kpool | −.021 −.009 −.012 −.015 | −5.59, .0056 | −.0142 | 1.07 | −.0162 / −.0163 / −.0063 / −.0141 |
| k2te | −.021 −.009 −.011 −.016 | −5.63, .0056 | −.0143 | 1.08 | −.0162 / −.0165 / −.0066 / −.0142 |

- Under kshift, all four SURVIVES conditions hold (verified):
  - 4/4 animal means are below 0;
  - p = .0053;
  - every drug agent's Δz is below 0;
  - R = 1.11, against a threshold of .5.
- Per-agent retention is KT .97, KTMD 1.14, MD 1.04, PF 1.26. Sessions down under kshift: KT 4/4, KTMD 11/11, MD 4/4, PF 4/4.
- R was predicted at .85–1.1 and came in at 1.11, just above that range. R is at least 1 under every candidate (kpool and k2te 1.07–1.08 as well), so no correction finds the bias carrying the drug effect.
- **Descriptive:**
  - Sleep Δz under kshift is −.0007, with 5/7 sessions down, against −.0010 and 7/7 under k4. It stays at control level.
  - Eyes-open SD is .0019–.0022 for every estimator.

## Why K is negative: mechanism

**The GCMI bound (verified, first principles).** With normal-scored marginals, MI ≥ G = −½log(1 − r²). True MI is ≥ 0. So, at weak edges, a mean K < 0 beyond sampling error is estimator error. A single edge below 0 can still be sampling spread.

The negative mean has two parts. Both are verified on synthetic pairs with true MI 0 and the spectra of real channels, n = 40 per cell ([`mech.txt`](../../../../results/representation/atlas-mi-pair-261117/ksgbias/mech.txt), [`calib.txt`](../../../../results/representation/atlas-mi-pair-261117/ksgbias/calib.txt)).

### 1. Stationary bias under the Theiler window, shared with the surrogate

Mean K at MI 0, KSG k = 4 with pool ψ. Spectra: KTMD drug (median AUTO w 17), PF drug (w 22), awake eyes-closed (w 5).

| | w = 0 | AUTO | 2× | 4× | 8× AUTO | same pair, time-shuffled | k = 10 at AUTO |
|---|---|---|---|---|---|---|---|
| KTMD | +.107 | −.037 | −.039 | −.042 | −.046 | .000 | −.017 |
| PF | +.102 | −.040 | −.043 | −.047 | −.051 | −.001 | −.019 |
| Awake EC | +.015 | −.005 | −.005 | −.006 | −.006 | .000 | −.003 |

- **Serial dependence is necessary.** The time-shuffled pair has no bias.
- **The sign flips with the window.**
  - With no window, the bias is the familiar positive one from temporal neighbours.
  - AUTO and every longer window run give a negative bias. Windows between 0 and AUTO were not run.
  - It deepens only slowly with w: about ×1.25 for an 8× longer window. It roughly halves at k = 10.
  - It tracks the spectrum, not w/T. It is about 8× larger under drug slowing than in awake windows, which is why it shows most under KTMD and PF. At T = 200 its AUTO value (−.024) is smaller than at T = 1000.
- **ψ(N) masks most of it.** `edges.py` uses ψ(N) (JIDT's convention: unverified) rather than ψ of the pool left after the window, which adds roughly 2w/N (+.034 at w = 17). So k4's level at stationary independence is ≈ 0 (KTMD −.0006, PF +.006 in calibration), while kpool's is −.038 / −.040.
- **Why it does not reach z.** IAAFT matches the spectrum and w, so the surrogate carries the same offset. Real − surrogate cancels it, and kpool and k2te change pooled Δz by 7–8%. Per agent the change reaches 20–21% (PF); the section below shows why.
- **The micro-mechanism is unknown.** A run-structure account (neighbours arriving in serially correlated runs) was considered and dropped, because it does not fix the sign.
- **Independent corroboration** (mi-pair-lowrate session, branch `claude/mi-pair-lowrate-3a8e2a`, scores on gdata `atlas-mi-pair-lowrate-261125/scores/`; not rerun here). The measure is the fraction of all edges with K < 0, real / surrogate.
  - **At 100 Hz:** drug .33 / .28, awake .17 / .17.
  - **Decimating or subsampling to 20 Hz** removes the drug excess (drug .12, awake .13), with median w 2–4.
  - **Low-pass filtering at 8 or 4 Hz while keeping 100 Hz** raises the fraction in every state, to .32–.51.
  - This fits a bias that follows how slow the spectrum is (mechanism table), not the anaesthetic state.
  - Their wording, "serial dependence surviving the Theiler window", needs a qualification. With no window the bias is positive here; the window is what makes it negative.
- **Literature (unverified).** No source found documents a *negative* KSG bias induced by a Theiler window. The positive bias without a window is standard; the window is meant to remove it (Theiler 1986; Kraskov et al. 2004).

### 2. Real-only bias under drug: per-channel structure that a shift keeps and IAAFT does not (amplitude nonstationarity inferred)

Calibration at T = 1000, real − surrogate K at MI-0 edges:

| | indep / block | gain_ind (×3) | burst_ind (×5) | gain_shared (true non-Gaussian dependence) |
|---|---|---|---|---|
| k4 | ≈ 0 | −.017 / −.021 (KTMD / PF) | −.030 / −.035 | +.013 / +.011 |
| kshift | ≈ 0 | +.000 / −.002 | −.000 / −.001 | +.024 / +.028 |

- Independent gain switching per channel lowers real K below its stationary IAAFT surrogate at edges that are truly independent (verified, synthetic). kshift removes this (verified, synthetic). It also keeps genuine shared-gain dependence: z real − surrogate for gain_shared is −.029 / −.039 under kshift, against −.033 / −.034 under k4.
- **In the real data.** Weak edges are those with G < .01; values are K real − K surrogate.
  - k4: KTMD −.010, PF −.016, MD −.003, KT +.011, sleep .000.
  - kshift: KTMD +.002, PF +.001, MD +.003, KT +.018.
  - So the real-only part shows where the drug slows the spectrum, and kshift removes it (verified). That its cause is amplitude nonstationarity is inferred, by analogy with the calibration.
  - The positive KT residual is inferred to be genuine weak-edge dependence, as in the gain_shared case. That inference is not tested.
- **Its effect on z is small.** Calibration gain alone lowers z beyond IAAFT by about the same amount under k4 and kshift. The real-only bias at MI-0 edges therefore carries little of z, which the real R = 1.11 confirms.
  - Whether the z fall itself reflects gain/nonstationarity rather than coupling nonlinearity is a separate question. That belongs to the [nonstat](preregs/nonstat-prereg.md) and [envelope](preregs/envelope-prereg.md) tests, not to this one.
- **Earlier k4-only looks** (disclosed in the prereg) agree:
  - dropping edges with G < .01 retains 90% of Δz (t(3) −7.0);
  - clipping K below the surrogate band retains 78% (t(3) −5.45).

## Across edges and states (post hoc, descriptive)

- **Source:** [`ksgbias_edges.py`](ksgbias_edges.py), output [`edges.txt`](../../../../results/representation/atlas-mi-pair-261117/ksgbias/edges.txt).
- **Definitions.**
  - b = k4 − kshift is k4's per-edge bias at independence under the pair's own serial structure.
  - p = kpool − k4 is the ψ term, ≈ −2w/N.
  - Values are medians over windows of the mean over edges. ρ is the Spearman correlation over edges within a window.

**Bias level, by state** (b real − b surrogate):

| | KT | KTMD | MD | PF | Sleep |
|---|---|---|---|---|---|
| Drug or sleep | −.007 | −.013 | −.006 | −.016 | −.001 |
| Awake (EC and EO) | −.000 to −.001 | −.001 | −.000 to −.001 | −.001 | −.000 to −.001 |

- **State.** Real b is ≈ +.005 in every awake and sleep condition, with the surrogate at the same value. Under drug it falls (KTMD −.010, PF −.012) while the surrogate stays positive. The real-only bias is a drug-state phenomenon; it is absent awake and asleep (verified).
- **Variation across edges with G and w.**
  - b varies weakly with G: ρ(b, G) is +.07 to +.24.
  - It varies somewhat more with w: ρ(b, w) is +.10 to +.32.
  - In both cases the real and surrogate values match to within .07 in every cell. So the edge-to-edge pattern of the bias is shared with the surrogate; only its level is real-only. This is why removing it hardly moves z = corr(G, K) beyond the surrogate (inferred from these numbers).
- **Why kpool shifts per-agent Δz.**
  - w correlates with G: ρ(w, G) is +.34 to +.52, highest under KT, MD and PF drug.
  - So the ψ correction is more negative at strong edges (ρ(p, G) −.33 to −.53). This lowers corr(G, K) in real and surrogate by about the same amount, and the effect varies with how much drug changes w.
  - That explains R up to 1.20 under PF for kpool and k2te. It does not carry the effect.

## Panel (b)

Panel (b) is the per-edge K in `mi-figure.png`. The fraction of weak edges with K < 0 is:

| | KT | KTMD | MD | PF | Sleep |
|---|---|---|---|---|---|
| k4 | .45 | .65 | .52 | .68 | .41 |
| kshift | .39 | .53 | .48 | .55 | .48 |

- The prediction was .4–.6 under kshift. Observed values are .39–.55: KT at .39 is just outside the range, the rest are inside.
- Per-edge SD at MI 0 and T = 1000 is about .045–.05. So under an estimator whose bias at independence is removed, about half the weak edges are still negative.
- **What the negative values are.** Mostly sampling spread around a near-zero level. Under KTMD and PF, k4 also puts the weak-edge mean at −.018 and −.025. Of that, −.010 and −.016 is the real-only bias (real − surrogate); the rest is shared with the surrogate.
- **Not a Kraskov bias that a Theiler/DCE correction fixes.**
  - A longer or 2τ(1/e) window (k2te) makes the level *more* negative, not less.
  - The w already in use is 2⟨acf_i, acf_j⟩, a cross-integral-time version of the supervisor's 1/e criterion (inferred). Any longer window deepens the bias (mechanism table).

## Recommendation

1. **Keep the statistic (k4) as primary.** Δz is robust to every candidate correction (R 1.07–1.11, every t(3) ≤ −5.6). Switching the primary after seeing the data would trade a frozen estimator for a 4× costlier one with an untested assumption: independence after a shift of ≥ 2.5 s. One line of robustness is enough: "with the circular-shift bias offset removed per edge, Δz is retained (R = 1.11; t(3) = −5.73, p = .005)".
2. **Panel (b): change the caption, not the plotted estimator.** Say that negative per-edge K is expected sampling spread (SD ≈ .05 at T = 1000), plus a small downward bias at weak edges under drug spectra, which the surrogate comparison and a shift-corrected estimator both absorb. If the figure owner prefers, plotting kshift would centre the weak-edge cloud near 0. About half of it would still be below 0, so that change is cosmetic.
3. **Do not lengthen the 10-s windows for this reason.** The shared part cancels. The real-only part comes from nonstationarity, which longer windows contain more of.

## Open

- **The 2-s arm (T = 200) is not rescored.**
  - Its stationary mechanism looks different: w = 0 gives +.26, AUTO gives −.024 with SD .16, and the w cap binds in 17–22% of windows.
  - Under block coupling, calibration z real − surrogate is ≈ −.022 ± .007 for every estimator, kshift included.
  - Whether that offset cancels in drug − awake is untested. Treat the 2-s arm as robustness only until it is rescored with kshift.
- **kshift's assumption is not tested on real data.** A slow shared component would push R down. R > 1 is consistent with that component being absent, but does not show it.
