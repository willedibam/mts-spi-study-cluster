# Tail-dependence asymmetry on the axis-screen windows (pre-registered, 23 November 2026)

Criteria stated and committed before any computation on data. Nothing below has been run on real windows.

## Question

Does the drug make the channels' joint distribution asymmetric between its tails: more joint extremes at the top than at the bottom, or the reverse? The readout is pyspi's `tailasym` (TailDependenceAsymmetry, pyspi #116, in the repo pin ce4b20e): λ_U − λ_L, the difference between the empirical upper- and lower-tail dependence coefficients (Schmidt & Stadtmüller 2006).

## Why this test, and not the `edge_acf1` test offered first

The open question for every surviving axis is whether gain (amplitude modulation on unchanged coupling) can produce it. The axis screen could not separate scalar from multi-source gain at 10 s ([result.md](../axis_screen/result.md), scalar-gain test). Both the MI and MinCovDet axes are copula-level and gain-compatible in sign. Tail asymmetry sidesteps that question instead of trying to resolve it:

- **Gain cannot move it.** Take x = a(t)u, y = b(t)v with positive gains independent of the sign of a zero-mean, symmetric carrier (u, v). Then (x, y) and (−x, −y) have the same distribution, so λ_U = λ_L. This holds for a shared or a multi-source gain, and for any radially symmetric copula, the Gaussian and t included.
  - Synthetic check (local, AR(1) a .95, ρ .5, T 1,000, q .9, 400 reps): mean λ_U − λ_L is +.012 Gaussian, +.016 shared gain and +.013 independent gains, all within 1.5 SE of 0 and SD .15–.18 per edge. A Gaussian rerun with 4,000 reps gives +.0001 (SE .002).
  - Shared one-polarity transients give −.109 (SD .14).
  - So a drug change in tail asymmetry is a change in dependence character that no gain account reaches.
- **It is copula-level by construction.** It is rank-based, so normal-scored and z-scored windows give identical real values. The preprocessing sensitivity that left MinCovDet and `l1|linf` copula-level only cannot arise here.
- **There is a hint, never gated.** The signed against two-sided barycentre maximum (`bary-sign`, the single-sample cousin of this statistic) moved beyond IAAFT in the matched 2-s screen: −.029, 18/19 sessions.
- **`edge_acf1` would mostly repeat known results.**
  - The MinCovDet mechanism check already measured temporal clustering of the outlying samples (Fano factor and run length in 0.5-s bins). Those samples are clustered in both states, and their clustering beyond IAAFT falls under the drug.
  - The MI pair's segment surrogates already place most of its departure in about 1-s stretches.
  - `edge_acf1` is a lag-1 statistic (10 ms at 100 Hz). On channels with lag-1 autocorrelation near .9 it reads sampling-scale persistence, not coupling episodes.
  - `edge_excessvar` is the cokurtosis cum(x, x, y, y), which a shared gain raises by its own docstring.

Both outcomes are informative:
- A pass is the first gain-immune character change.
- A null rules asymmetric shared events out as the content of the MI and MinCovDet departures.

## What is computed

**Windows and versions:**
- Every window of the 10-s arm (primary; T 1,000) and the 2-s arm (secondary; T 200), all channels.
- Real and the screen's three spectrum-final joint IAAFT draws, regenerated with `axis_screen/run.py`'s code and seeds on normal-scored windows.
- Under IAAFT the expectation is 0 by the argument above, so the net mainly removes finite-sample bias.

**Rows, per edge** (k = ⌊(1 − q)T⌋):

| Row | Definition | k at 10 s / 2 s |
|---|---|---|
| `ta` | λ_U − λ_L at q .9 | 100 / 20 |
| `ta95` | λ_U − λ_L at q .95 | 50 / 10 |
| `ta_up`, `ta_lo` | λ_U and λ_L at q .9 | 100 / 20 |

The rows are computed as one rank pass per channel plus two M × T indicator products. Before scoring, they must equal pyspi's `tailasym_q-0-9`, `tailasym_upper_q-0-9` and `tailasym_lower_q-0-9` (pin ce4b20e) to 1e-12 on one window per archive. If they don't, nothing is scored.

**Reproduction check:** the regenerated r profile must match the screen's stored one within 1e-4 in every window and version, as in the ξ screen.

## Readouts

**Primary level:**
- Per window: the mean over edges of `ta`. Net = real − mean of the three IAAFT draws.
- Contrast: drug − awake eyes-closed, per agent and pooled, with animal-level t(3).
- No direction is predicted.

**Gates** (the screen's, [prereg.md](../axis_screen/prereg.md) "Axis gates", gates 1–5, applied to the level):
1. Animal level.
2. Size of at least 3 null SD, from the within-awake split-half.
3. Split-half of at most .25 |Δ|.
4. Eyes-open of at most .33 |Δ|.
5. Covariates. This test adds one covariate to the screen's F1 set: the median over channels of |skewness| of the z-scored (not normal-scored) window. A drug-induced rise in one-polarity events in single channels must not carry the effect alone.

The screen's gate 6 (disattenuated z) does not apply to a level. Instead, the awake eyes-closed split-half reliability of the `ta` edge profile is reported.

**Polarity:**
- The signed edge mean is interpretable only if all channels share one reference polarity. That is unverified for NeuroTycho, and is checked from the archive documentation before scoring.
- If it cannot be confirmed, the primary becomes the mean over edges of |ta| net (real − IAAFT), with the same gates. Its IAAFT net absorbs the positive bias of |·|.
- This choice is made before any drug contrast is looked at.

**Secondary,** reported with the same gates and not used for the reading:

| Readout | Role |
|---|---|
| `ta95` level | tail depth: does the asymmetry grow toward the extremes |
| `ta_lo\|ta_up` z, F1 | edge ordering of the two tails. Its z₀ is 1 minus noise under symmetry, so it is judged by null SD (guide proposal 7), not by z₀ |
| `r\|ta` z, F1 | does the asymmetry follow coupling strength |
| 2-s arm `ta` level | link to the `bary-sign` hint; expected to fail reliability at k = 20 |

**Links,** descriptive only:
- Window-level correlation within drug sessions of the `ta` net with the MI net and with the MinCovDet net, computed as in the ξ screen.

## Outcomes and readings

- **Primary passes every gate.**
  - The drug changes the tail asymmetry of the joint distribution. No gain on symmetric carriers can produce this.
  - The sign names the tail: negative means more joint extremes at the bottom (shared one-polarity, negative-going events), positive the reverse.
  - Then read the links:
    - **If the window r with the MI net is ≥ .5,** part of the MI anchor's departure is gain-immune asymmetric co-excursion. That gives the anchor a partial reading, not a new axis.
    - **If r < .5,** it is a candidate axis in its own right.
  - **Caveat:** a single skewed source spread by volume conduction also produces copula asymmetry. So the reading is "shared one-polarity events", not inter-areal coupling, unless the effect survives restriction to distant edges. If electrode positions are available, the restriction to pairs above the median inter-electrode distance is reported when the primary passes.
- **Primary fails, with an awake profile reliability of at least .3:**
  - The drug's copula change is radially symmetric at this power.
  - Asymmetric shared events are then not what the MI and MinCovDet departures carry. What remains open for them is symmetric non-Gaussianity: heavy joint tails, a mixture of covariance regimes, or gain.
- **Reliability below .3 in awake eyes-closed:** uninformative at this window length.

## Cost

Ranks and indicator products cost well under 0.1 s per window and version at M 126, T 1,000. As in the ξ screen, the cost is regenerating the IAAFT versions.
- Expected: about 25 SU (10 s) and 10 SU (2 s) on one 48-core node.
- Worst case: 0.1 kSU.

## Result (23 November)

Jobs 184939935 (10 s, 9.8 SU) and 184939936 (2 s, 3.2 SU), commit 890dde9, pyspi 3.0.0. Scores: [`results/representation/atlas-pair-catalogue-tailasym-261123/`](../../../../results/representation/atlas-pair-catalogue-tailasym-261123/).

**Checks passed:**
- The rows equal pyspi's `TailDependenceAsymmetry` exactly (maximum difference 0.0, two windows per archive, every archive).
- The regenerated r and z reproduce the screen to 3e-8 and 6e-8.

**Polarity:** every window is common-average referenced by `prepare.py` with no per-channel sign change, so the signed primary stands. The common average partly removes events shared by all channels and shows a focal event with the opposite sign elsewhere. Array-wide one-polarity events are therefore attenuated in this readout.

**Primary, by the pre-stated rule: uninformative.**
- The awake eyes-closed split-half reliability of the `ta` edge profile is .11 at 10 s and .02 at 2 s, both below .3.
- Descriptively, the level is flat: net −.0008 (0.2 null SD, t(3) −0.4, 14/23 sessions). The four animals split two and two, and the agents disagree (KT −.014, the others within ±.004).
- No null is claimed. The level is too noisy per edge to rule asymmetric shared events in or out.

**Secondary results:**

| Readout | 10 s | 2 s | Gates |
|---|---|---|---|
| `ta95` level | −.0010, 0.2 SD | −.0033, 0.5 SD | fail |
| \|ta\| level | +.0053, t(3) 2.3, 2.3 SD, 19/23 | +.0074, t(3) 4.3, 2.2 SD, 22/23 | fail: size and covariates. At 2 s the F1 covariates alone absorb it (adjusted +.0015); with skewness, −.0013 |
| `ta_up\|ta_lo` z, F1 | −.028, t(3) −3.9, 3.2 SD, 22/23, all four animals, rMI −.02 | −.024, 1.7 SD | 10 s: all six screen gates pass. 2 s: fails size, covariates and reliability |
| `r\|ta` z | −.014, 0.2 SD | −.033, 0.6 SD | fail (`ta` reliability .11 / .02) |

- **Median channel \|skewness\| rises with the drug** in all four animals: +.055 at 10 s and +.089 at 2 s, from .19 and .22.
- **Window-level links within drug sessions:** the `ta` level net with the MI net −.06 and with the MinCovDet net +.09; \|ta\| −.13 and −.10.

**`ta_up|ta_lo` at 10 s is not read as a gain-immune character.** It passes the screen's gates, but three things weaken it:

1. **Skewness absorbs part of it (post hoc,** [`posthoc-skew-long.json`](../../../../results/representation/atlas-pair-catalogue-tailasym-261123/posthoc-skew-long.json)). The level gates' covariate set, F1 plus channel skewness, was pre-stated for the level, not for this coordinate. Under it the effect keeps 94% of its size (−.026) but loses two animals: Kin2 −.004 and Su +.007, t(3) −1.6. Within drug sessions the window net correlates −.31 with skewness.
2. **The spectra carry most of the real change.** The real z falls by −.089, and IAAFT with the drug's spectra already falls by −.061. The fall is mostly the noise of the λ estimates rising under slower spectra; the net is the remaining third.
3. **Its gain immunity is weaker than the level's.** Radial symmetry fixes the expected λ_U and λ_L of each edge, not their sampling noise. A gain that clusters extremes in time lowers the effective tail count and so lowers this z without any asymmetry. The MinCovDet mechanism check makes this less likely: there, clustering of outlying samples beyond IAAFT falls with the drug. But that check used a different statistic, so the point is not tested.

**Reading:** edge-specific tail asymmetry of mixed sign rises under the drug, with no consistent net sign. It follows the rise in channel skewness. That is consistent with one-polarity waveforms within channels (for example skewed slow waves) shared over a few neighbouring channels. It is not a coupling character that gain cannot reach. Recovery keeps .69 of the drug change, and sleep does not reproduce it (ratio −.14).

**What this test settles:**
- Asymmetry of joint extremes is not a candidate gain-immune axis at this power.
- The drug raises channel skewness. Because ranks and normal scores remove the marginals, this shows up only through edges that share skewed events.
- The open question for the MI and MinCovDet axes stays where it was: symmetric non-Gaussianity, a mixture of covariance regimes, or gain.
