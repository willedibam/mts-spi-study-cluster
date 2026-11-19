# PRIOS full-pyspi screen: which dependence characters change under propofol in human ECoG, beyond a per-window joint IAAFT null

Frozen 19 November 2026, before any pyspi output on PRIOS exists. Code: [`prios.py`](prios.py), [`jobs/prios.pbs`](jobs/prios.pbs), pyspi through [`../jobs/farm.pbs`](../jobs/farm.pbs). Reading rules: [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md). Seen before freezing: the PRIOS MI-pair results (`../mi_pair/preregs/human-prereg.md`, S-AB: −.009, 7/7) and the macaque movers and named-family results ([result.md](result.md); the per-window IAAFT family check `families_iaaft.py`, scored 19 November: Chebyshev and kernel lagged MI against TE survive on the normal-scored scale, precision × phase coherence is reproduced by the surrogate).

## Why a screen, and how it stays honest

The MI pair is one minimal pair of about 700 among the atlas's 272 SPIs, each reading one assumption. A screen of all of them is the general method; selection is controlled by declaring the whole family, a false-discovery rule, a negative control, and reporting every row. Two kinds of test, kept apart: predictions from the macaque (directional, few, Holm or exact permutation) and discovery (two-sided, BH over the family).

## Data and computation

- Windows: the S-full PRIOS set (`../mi_pair/preregs/human-prereg.md`, sensitivity set): S-AB exclusion rules, every passing 2-s epoch, every good channel; per epoch the stimulated and annotated electrodes leave the CAR and the channel set, last channel dropped. 7 patients, awake ward against propofol theatre.
- Real bank `prios-ns`: each channel normal-scored (rank to normal), as the MI-pair statistic. Null bank `prios-iaaft`: one spectrum-final joint multivariate IAAFT draw of the normal-scored window (exact auto- and cross-spectra; the seed is the MI-pair S-full draw 0, so the surrogate is the same one).
- pyspi: the NeuroTycho 2-s arm's computation exactly (config `configs/pyspi/atlas/benchmarked_p90_pcorr_symbolic.yaml`, 272 SPIs; source `pyspi-upstream-0c798b4`, r13; `atlas/extract.py`, per-window seed by id, `normalise=True`), so coordinates match the macaque atlas one to one. Reduced with `discovery.py reduce` (z, double-centred z, node R²).
- Net per window and coordinate: z(real) − z(surrogate). Per patient: Δ = mean net (propofol) − mean net (awake). Inclusion: ≥ 20 windows per state.

## Usable coordinates (blind to the contrast)

- ≥ 95% finite windows in every patient × state × bank (means over finite windows);
- both SPIs' awake edge-profile reliability ≥ .3 (`assumptions.py reliability`, reference awake, mean over patients; as the macaque screen);
- |awake real z| < .98 (mean over patients).

## Tests

1. **Primary discovery: every usable minimal pair** (`assumptions.edges` on the 272 SPIs, 704 pairs). Two-sided one-sample t on patient Δ; Benjamini–Hochberg q ≤ .05 over the usable minimal pairs; and at most one patient against the sign of the mean. Negative control: the identical pipeline on awake first half minus second half (windows in recording order); estimated false-discovery proportion = control discoveries ÷ discoveries; nothing is claimed if it exceeds .2.
2. **Predictions from the macaque** (one-sided in the macaque direction):
   - the 11 macaque minimal-pair movers ([result.md](result.md)), each by one-sided t, Holm over those usable;
   - all 225 confirmed macaque movers together: mean of sign-aligned patient-mean Δ over those usable, exact sign-flip test over patients (2⁷ patterns);
   - whole-profile concordance: Pearson r across coordinates usable in both datasets between PRIOS mean Δ and the macaque drug net (KT, KTMD and PF sessions pooled; net of the stationary and bias banks, as `movers.py`), over minimal pairs (primary) and all coordinates (secondary), exact sign-flip test. The two nets use different nulls (Gaussian bank against per-window IAAFT); a positive r is the claim "the same coordinates move the same way", not equal sizes.
3. **Secondary, descriptive:** minimal pairs without the reliability filter (BH within); every usable coordinate (BH within; count, control count, SPIs ranked by participation, as the macaque screen).

## Reported per discovery

Class and reading (`assumptions.READINGS`: character, literal, confound) with the wording set by its validation status; mean Δ with 95% interval, patients negative, p, q; Δ of real and surrogate z separately; Δ of double-centred z (pair-level change with channel effects removed); awake node R² of both SPIs (≥ .5: the profile reads channel properties, so the reading is about channels, not pair dependence); the macaque net for the same coordinate. The MI pair `mi_gaussian|mi_kraskov_NN-4_DCE-AUTO` is reported as the anchor whether or not it passes.

## Pre-stated limits

- Day and setting confound (awake ward days, propofol in theatre on a later day); epilepsy patients; one IAAFT draw per window; 2-s windows only, so lag and phase-lag SPIs are likely to fail the reliability filter, as in the macaque.
- Seven patients: two-sided t with 6 df. With m usable pairs the first BH discovery needs p ≤ .05/m (m = 200: t ≈ 7.5, d_z ≈ 2.8), so an MI-pair-sized effect (PRIOS d_z ≈ 1.6, t ≈ 4) cannot pass the open screen; the macaque predictions are the powered tests. The 30 best non-passing pairs are listed, as candidates only.
- The PRIOS MI-pair result was seen before freezing; the anchor row is therefore not an independent test.

## Compute

One profile farm before the full farms; the per-window time it measures sets the request. Every submission states expected and worst-case KSU (recorded with the result).

## Deviation 1 (19 November, before any score was computed)

The first post stage (job 184616789) passed audit (1,956 of 1,956 shards) and reduce, then failed at `assumptions.py reliability`: it stacks each patient-state's edge profiles and assumes one channel set, but each S-full epoch drops its own stimulated electrodes, so M varies by 1–4 channels within a patient and state. Fix: `prios.py channels` writes each window's original channel indices (`<bank>/channels.json`); when that file exists, `discovery.reliability` places windows on the union of their channels and correlates two windows' profiles over the edges both have (pairwise-complete Pearson, identical to the old computation when every window has every channel, so the macaque reliability is unchanged). Nothing else changes.

## Deviation 2 (19 November, found after the first summary was printed)

The first scored run (job 184625303) reported 0 discoveries in every BH screen, but every q was NaN: a few usable coordinates have Δ identically 0 in every patient (two SPIs both fixed by the covariance, which the spectrum-final IAAFT keeps exactly), their t-test p is undefined, and one NaN propagated through the BH step-up. Fix: an undefined p counts as p = 1 (no evidence, kept in m, the conservative choice) in `bh` and `holm`; the count is reported as `undefined_p`. The macaque-prediction tests, sign-flip and concordance results were unaffected (finite p throughout) and are unchanged. This was found after the faulty summary was seen; the fix computes the rule as written and makes no other change.
