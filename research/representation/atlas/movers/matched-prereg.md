# Macaque full-pyspi screen against per-window joint IAAFT, and its test on PRIOS

Frozen 19 November 2026, before any output of the PRIOS screen ([prios-prereg.md](prios-prereg.md)) was read and before any pyspi output on these banks exists. Code: [`matched.py`](matched.py), [`jobs/matched.pbs`](jobs/matched.pbs), pyspi through [`../jobs/farm.pbs`](../jobs/farm.pbs). Reading rules: [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md). Seen before freezing: the macaque movers screen and the per-window IAAFT family check ([result.md](result.md)), and the PRIOS MI-pair results.

## Why

The macaque movers screen found 11 minimal-pair movers, but under a different rule from the PRIOS screen: a state-mean Gaussian null (looser than IAAFT, which withdrew one of its families), raw rather than normal-scored windows, and a magnitude gate (|net| ≥ .04 on even windows) that the MI anchor itself (−.024) fails. So it is not the macaque version of the PRIOS screen, and its 11 are a weak prediction set. This screen applies the PRIOS construction to the macaque, where 19 sessions in 4 animals give power the 7 PRIOS patients do not, and its discoveries become the prediction set for PRIOS.

## Data and computation

- Windows: every window of the NeuroTycho 2-s bank `neurotycho-short` (1,260 windows, 30 archives, 10 per archive-state; M 126–127, 100 Hz, 1–40 Hz). This is the bank the atlas and movers screen used; re-windowing all of each interval would cost about 20× more than the grant allows.
- Banks: `nt-ns` (each channel normal-scored, as PRIOS) and `nt-iaaft` (one spectrum-final joint multivariate IAAFT draw of the normal-scored window, seed `[seed(id), 0, 1]`, the same draw as `families_iaaft.py` iaaft_s0). pyspi exactly as PRIOS and the atlas (config `configs/pyspi/atlas/benchmarked_p90_pcorr_symbolic.yaml`, `pyspi-upstream-0c798b4`, per-window seed by id). Reduced with `discovery.py reduce`; reliability with `assumptions.py reliability`, reference awake eyes-closed.
- Net per window: z(ns) − z(iaaft). Per session (archive): Δ = mean net (drug) − mean net (awake eyes-closed). Primary sessions: KT, KTMD and PF, 19 sessions in 4 animals.

## Usable coordinates (blind to the contrast)

As PRIOS: ≥ 95% finite windows in every primary session × state (awake_ec, awake_eo, drug) × bank; both SPIs' awake_ec reliability ≥ .3; |awake_ec z(ns)| < .98, means over sessions.

## Macaque tests

1. **Primary discovery: every usable minimal pair** (`assumptions.edges`). Two-sided one-sample t over the 19 session Δ; BH q ≤ .05; and every animal's mean Δ with the sign of the overall mean. Negative control: the identical pipeline on awake eyes-open minus eyes-closed in the same sessions (a real but non-drug change, so the control is conservative). FDP estimate = control ÷ discoveries; no claims, and no prediction set, if it exceeds .2.
2. **Secondary, descriptive:** every usable coordinate (BH within, control count, participation by SPI); minimal pairs without the reliability filter. Per discovery: animal means and t(3); Δ of real and surrogate z separately; double-centred Δ; awake node R²; specificity contrasts (MD drug, natural sleep, recovery, each against awake_ec); the old movers-screen net; the reading per `assumptions.READINGS`.

## Test on PRIOS (frozen now, before the PRIOS screen output is read)

PRIOS Δ and usability exactly as [prios-prereg.md](prios-prereg.md). Prediction set: the macaque primary discoveries usable in PRIOS, each with its macaque sign.

- **A (primary): aggregate.** Mean over the prediction set of sign-aligned PRIOS patient-mean Δ; exact sign-flip over the 7 patients (2⁷ patterns), one-sided.
- **B: per pair.** One-sided t in the macaque direction, Holm over the prediction set.
- **C: concordance.** Pearson r between PRIOS mean Δ and macaque mean Δ over minimal pairs usable in both (all usable coordinates secondary), exact sign-flip over patients, one-sided. Unlike the PRIOS prereg's concordance, both nets here use the same null construction.
- If the macaque screen makes no claims, A and B are not run and C still is.

## Pre-stated limits

- Four animals: the session t treats 19 sessions as replicates; the every-animal rule and the reported t(3) guard against one animal carrying a result, but a session-level q is not an animal-level inference.
- Drug and calendar are confounded within archives in the same way as in the movers screen; eyes-open, MD and sleep are the specificity contrasts.
- One IAAFT draw per window; 2-s windows, so lag and phase-lag SPIs are likely unusable (as before).
- The 11 old movers and the PRIOS MI result were seen; this screen's rule copies the PRIOS rule, so no choice in it was tuned to either.
- PRIOS has 7 patients, so test A's smallest attainable p is 1/128.

## Compute

Bank: 48 cpus, under 1 h. Two farms, one window per shard, 432 cpus each: the atlas 2-s arm measured about 0.78 cpu-hours per window at M 126, so about 3 rounds of 0.8 h, about 2 kSU per farm. Post: 48 cpus, at most 4 h. Expected about 4.2 kSU in all; worst case (both farms at a 4-h walltime, post at 4 h) about 7.4 kSU.

## Note before any output (19 November)

`bh` and `holm` (imported from `prios.py`) now count an undefined p as 1 (PRIOS Deviation 2); the post stage is resubmitted from the fixed snapshot before any macaque output exists.
