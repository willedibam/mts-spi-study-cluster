# Does the MI-pair drug change depend on brief large transients? A transient cut on the macaque 10-s windows

Frozen 23 November 2026 at the commit that adds it, before any output of this test exists. Code: [`../transient/run.py`](../transient/run.py).

## Why

- Some drug windows carry brief large events visible in the raw traces: ketamine + medetomidine, Chibi, session 20110621, windows drug:00 (Δz −17.5 wake SDs, the most extreme window) and drug:02. In drug:02 the pair of largest K − G (electrodes 78–90) owes most of it to two such events: K − G .67 with them, .27 with them cut.
- Such events pass the bank's artefact rule, which only rejects a channel with more than 5% of its samples beyond 6 robust SDs of its state. Whether they are artefact or physiology (spikes, K-complex-like events) is not known.
- Normal-scoring bounds their amplitude but not their timing. A few samples where every channel jumps together lower the linear correlation and raise the kNN MI of many pairs at once, so a handful of events could carry a window's Δz.
- The question: is the drug change a property of the ongoing signal, or of brief events within it?

## Procedure

- Windows, subsets, normal-scoring and N0 seeds exactly as `nonstat/run.py` (1,260 windows; reproduction gate: real and N0 z equal the stored zns z within 1e-4, or stop).
- **CUT:** in each window (raw, before normal-scoring), mark every sample beyond 5 robust SDs (median, 1.4826 MAD, per channel and window) in any of the 32 channels, and the 20 samples (0.2 s) either side. Remove those samples from every channel, concatenate the rest, normal-score again, and score z on it and on its own joint IAAFT draw (seed [row, 1]).
- **SHIFT (control):** the same cut mask circularly shifted by a random offset (seed [row, 0]); scored the same way (seed [row, 2]). Same number of samples removed and the same splice points, but unrelated to the events. It separates the effect of removing the events from the effect of removing data and splicing.
- A window with fewer than 300 samples left is dropped from that variant.
- Statistic and pooling as `nonstat/run.py`: residual real − null per window; per session Δ = state − awake eyes-closed of the same variant; animal means over the drug sessions; one-sided t on the four animal means.

## Reading (stated before output)

Retention R = pooled drug Δ under a variant ÷ pooled drug Δ under N0.

- **Independent of transients:** R(CUT) ≥ .75, or R(CUT) within .15 of R(SHIFT). The drug change is carried by the ongoing signal.
- **Carried by transients:** R(CUT) ≤ .5 while R(SHIFT) ≥ .75. Removing the events, and not removing data, takes most of the change away.
- **Otherwise:** partial; report R, the per-agent and per-animal values, and the Chibi ketamine + medetomidine sessions.
- In every case report the cut fraction per state: if drug windows lose far more samples than wake, the transients themselves are a drug effect, which is worth stating whatever R is.

## Pre-stated limits

- One surrogate draw per window and variant; 32-channel subsets; four animals.
- The threshold (5 robust SDs, ±0.2 s) is one choice, fixed here, not tuned.
- Splicing adds discontinuities; SHIFT has the same number, so their effect is in both.
- Cutting does not say what the events are. It only says whether the change needs them.

## Compute

Gadi, 48 cpus, as `nonstat` (1,260 windows × 6 MI evaluations; nonstat took 55 min). Walltime 2 h. Expected ≈ 0.1 kSU, worst case 0.2 kSU.
