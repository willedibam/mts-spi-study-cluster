# Is the MI-pair drug change explained by time-varying linear dependence? Segment surrogates on the macaque 10-s windows

Frozen 19 November 2026 at the commit that adds it, after the synthetic calibration and before any real-data output of this test exists. The calibration showed the planned decisive test cannot decide, so the real-data run is descriptive (see "Primary L"). Code: [`../nonstat/nulls.py`](../nonstat/nulls.py), [`../nonstat/calib.py`](../nonstat/calib.py), [`../nonstat/run.py`](../nonstat/run.py).

## Why

- **The claim.** The MI pair is read as nonlinear (non-Gaussian) dependence in the operational sense: dependence beyond the linear-Gaussian expectation, whatever produces it ([reading](../../../../../docs/interpreting-spi-pairs.md#worked-example-gaussian-against-flexible-mi-clarified-17-november)). Its null, joint IAAFT, is stationary linear-Gaussian.
- **The known trap.** In the surrogate-data literature, the main false positive for nonlinearity is non-stationarity. A stationary surrogate reads a time-varying linear process as nonlinear (Schreiber and Schmitz 2000; Lancaster et al. 2018).
- **Why now.** Post hoc ([matched-result.md](../../movers/matched-result.md)), the MI and Chebyshev residuals within drug windows track local amplitude non-stationarity beyond the surrogate (r = −.51, −.43), whose excess doubles under drug. A linear adjustment removed only 18% of the MI change, but that adjustment under-corrects.

This test asks whether linear-Gaussian dependence that is allowed to change over time reproduces the drug change. Under the operational definition the claim stands either way. What changes is the description: "time-varying linear dependence" if the change is absorbed, or "dependence beyond time-varying linear dependence on scales slower than L" if it survives.

## Nulls

- **N0** (the anchor): one joint multivariate IAAFT draw over the 10-s window, spectrum-final. It keeps exact auto- and cross-spectra and approximate marginals.
- **N2 at L samples:** the same IAAFT run separately on consecutive segments of L samples and concatenated; a tail shorter than L/2 joins the previous segment.
  - H0: linear-Gaussian dependence that may change between segments, covering gain with any noise floor, slow coupling switching and slow spectral drift.
  - L is on a grid of 500, 200, 100 and 50 samples (5, 2, 1 and 0.5 s at 100 Hz).
- **N1, dropped before any real data:** a per-channel envelope-remodulation null (`nulls.n1`).
  - It was calibrated on a linear network.
  - But it gave +.039 to +.125 on shared gain over an independent floor, i.e. it injects non-Gaussian dependence whenever the gain does not scale the floor.
  - This is the same floor problem that broke the [amplitude gate](amplitude-prereg.md).

## Calibration (synthetic, known answers; fills the primary choice)

[`calib.py`](../nonstat/calib.py): 32 channels, T = 1000, normal-scored, the anchor's `edges.edge_mi`, one draw per null, 12 windows per generator. The generators are described in its docstring:
- linear;
- on/off shared burst gain with 0.8-s and 3-s dwell;
- slow gain over a common-mode floor;
- per-channel slow gain;
- slow coupling switching;
- local decoupling episodes;
- strong and weak static quadratic coupling.

Result (`results/representation/atlas-mi-pair-261117/nonstat/calib1.txt`, `.npz`), the share of each generator's N0 residual retained under N2:

| Generator | N0 residual (t) | L500 (5 s) | L200 (2 s) | L100 (1 s) | L50 (0.5 s) |
|---|---|---|---|---|---|
| linear | −.0015 (−1.4) | raw −.0020 | raw −.0024 | raw −.0018 | raw −.0014 |
| burst gain, 0.8-s dwell | −.068 (−8.9) | 1.00 | .92 | .80 | .34 |
| burst gain, 3-s dwell | −.048 (−4.4) | .91 | .87 | .49 | .17 |
| slow gain, common-mode floor | −.029 (−5.0) | .96 | .78 | .55 | .08 |
| per-channel slow gain | −.062 (−16.2) | .77 | .63 | .36 | .08 |
| slow coupling switching | −.012 (−3.0) | 1.18 | 1.29 | .46 | −.15 |
| local decoupling episodes | −.017 (−8.4) | .97 | 1.08 | .74 | .00 |
| static quadratic | −.046 (−12.1) | 1.03 | 1.02 | .92 | .14 |
| weak static quadratic | −.0026 (−2.6) | .78 | 1.69 | 1.51 | .01 |

**Reading of the calibration: the segment null cannot separate gain from a nonlinear coupling function at this resolution.**
- At 1-s segments a static nonlinearity is retained (.92), but so is most realistic gain: .80 for 0.8-s bursts, which are like slow-wave UP/DOWN states, and .36–.55 for slower gain.
- At 0.5-s segments every generator is absorbed, the static nonlinearity included (.14), because IAAFT on 50 samples nearly reproduces the segment.
- The weak quadratic is too close to the noise (t −2.6) to grade. Every gain generator lowers z beyond N0, so gain is a sufficient generator of the sign.
- This is the non-identifiability noted in the workstream context (gain and floor against coupling, passive recordings), now quantified for this statistic.

**Primary L: none.** The draft rule (three gates: linear bias |mean| ≤ .002; the weak quadratic retaining ≥ .5 with t ≤ −3; decoupling retaining ≥ .5) would have chosen L = 100. That draft lacked a gate the test needs: gain generators must be absorbed (retained ≤ .5) at the primary L. Otherwise SURVIVES is the expected outcome under gain too: 0.8-s burst gain retains .80 at L100. With that gate no L qualifies. This revision was made after the calibration and before any real-data output of this test existed. The verdict rule below is therefore not applied, and the real-data run is descriptive with the reading map stated here.

## Real data

- **Windows:** the anchor's windows and subsets exactly (`estimators.py` / `edges.py recompute`). NeuroTycho 10-s stages A and B, 1,260 windows, 32-channel subsets (rowmode `first`, except stage-B sleep `row`). Each window is normal-scored. N0 is seeded by the row index (the anchor's own draw); N2 is seeded by [row index, L].
- **Reproduction gate:** the real and N0 z must match the stored `zns` z within 1e-4, or stop.
- **Statistic:** the residual z(Gaussian MI, KSG k = 4) real − null per window. Per session, Δ = state − awake eyes-closed. Animal means pool the drug sessions (KT, KTMD, MD, PF), as `montage/animal_level.py`.
- **Descriptive output:** the retention R(L) = pooled drug Δ under N2(L) ÷ pooled drug Δ under N0, at every L; animal means; per agent; sleep; the eyes-open contrast.

## Reading map (stated before real data; comparisons with the calibration table, no test)

- **R(100) ≤ .5:** like slow gain, slow switching and per-channel gain (.36–.55). Unlike a static nonlinearity (.92), local decoupling (.74) and 0.8-s burst gain (.80). This would favour slow non-stationarity of linear dependence as the generator.
- **R(100) ≥ .75:** consistent with a static nonlinearity, local decoupling or 0.8-s burst gain alike. Not informative about gain.
- **R(50) ≥ .5:** beyond every generator here (all ≤ .34). The structure would be finer than 0.5 s and unmodelled; report it, do not interpret it.
- **Otherwise:** report the curve.

## Pre-stated limits

- One surrogate draw per window and null; 32-channel subsets; four animals.
- N2's segment boundaries add a few discontinuities. The linear calibration bounds their effect.
- A short segment constrains the surrogate more, so N2 absorbs more as L falls whatever the truth. Hence the sensitivity gate on a static nonlinearity.
- This covers the 10-s macaque arm only. 2-s windows (the matched screen, PRIOS) are too short for informative segments.

## Compute

Gadi, 48 cpus. About 1,260 windows × 6 MI evaluations × ~10 s ≈ 21 cpu-hours, about 0.5 h wall; walltime request 2 h. Expected ≈ 0.05 kSU, worst case 0.2 kSU.

## Result (19 November 2026; job 184637760, 55 min on 48 cpus, about 0.09 kSU)

Output `results/representation/atlas-mi-pair-261117/nonstat/score.txt`; per-window arrays on gdata `.../atlas-mi-pair-261117/nonstat/` (SHA256SUMS here). Reproduction gate passed exactly: real and N0 z equal the stored zns values.

| Null | Pooled drug Δ | Animal means < 0 | One-sided p | Retained R(L) | KT | KTMD | MD | PF | Sleep |
|---|---|---|---|---|---|---|---|---|---|
| N0 | −.0133 | 4/4 | .004 | 1 | −.0159 | −.0155 | −.0062 | −.0117 | −.0010 |
| N2, 5 s | −.0145 | 4/4 | .004 | 1.09 | −.0158 | −.0180 | −.0063 | −.0119 | −.0014 |
| N2, 2 s | −.0128 | 4/4 | .006 | .96 | −.0146 | −.0152 | −.0062 | −.0110 | −.0007 |
| N2, 1 s | −.0057 | 4/4 | .018 | **.43** | −.0095 | −.0049 | −.0040 | −.0056 | −.0005 |
| N2, 0.5 s | −.0039 | 4/4 | .0004 | .29 | −.0028 | −.0047 | −.0010 | −.0055 | +.0004 |

Eyes-open SD is .0017–.0023 under every null.

**Reading by the prestated map:**
- **R(1 s) = .43 ≤ .5.** That is like slow gain, slow switching and per-channel gain (.36–.55 in calibration), and unlike a static nonlinearity (.92), local decoupling (.74) and 0.8-s burst gain (.80).
- So the larger part of the drug change is reproduced by linear-Gaussian dependence that changes between 1-s segments: slow non-stationarity of linear dependence is the main generator.
- **R(0.5 s) = .29 < .5,** within the calibrated range. A smaller residual persists below 1 s in every animal (4/4, p = .0004; KTMD 11/11). It is the size 0.8-s burst gain leaves (.34), and it is not identifiable further.

**Descriptive, post hoc:**
- Retention at 1 s differs by agent: KT .60, MD .65, PF .48, KTMD .32. KT's larger remainder fits its reproducible pair-level reordering (`reorder.py`).
- Sleep is ≈ 0 under every null.

**Limits:**
- These are comparisons with synthetic generators whose timescales and SNR were chosen, not fitted. One draw per window and null; four animals.
- The reading is "favours", not a verdict, because the calibration showed no segment length separates gain from coupling outright.

**Terminology correction (19 November, after the result; the reading map is unchanged).** "Slow non-stationarity of linear dependence" above means variation within the 10-s window on about 1-s scales. Drift slower than the window, the classic false positive, is excluded by the 5-s and 2-s rows. A within-window Gaussian mixture is itself non-Gaussian dependence under the operational definition ([reading guide](../../../../../docs/interpreting-spi-pairs.md)), so this result gives the timescale and form of the departure; it does not reclassify it as linear.
