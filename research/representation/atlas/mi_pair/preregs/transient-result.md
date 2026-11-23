# Result: transient cut of the MI-pair drug change (23 November 2026)

- **Rule:** [transient-prereg.md](transient-prereg.md), frozen at e196c58b before any output.
- **Code:** [`../transient/run.py`](../transient/run.py), run from snapshot `operations/sources/mi-transient-e196c58b`.
- **Gadi job:** 184961046, 48 cpus, 45 min, exit 0.
- **Outputs:**
  - per-window arrays on gdata at `/g/data/ql44/we2614/mts-spi-study/representation/atlas-mi-pair-261117/transient/` (`tr_[AB]_*.npz`, checksums in `SHA256SUMS`);
  - `score.txt` and `transient-score.json` in `results/representation/atlas-mi-pair-261117/transient/`.

## Verdict: independent of transients

- **Reproduction gate: passed.** Real and N0 z equal the stored zns z exactly (max difference 0).
- **No window was dropped:** every window kept at least 300 samples in both variants.

| Variant | Pooled drug Δ | R | Animals < 0 | One-sided t p | Eyes-open SD |
|---|---|---|---|---|---|
| N0 | −.0133 | | 4/4 | .0044 | .0020 |
| CUT | −.0154 | **1.16** | 4/4 | .0009 | .0020 |
| SHIFT | −.0179 | 1.35 | 4/4 | .0029 | .0034 |

R(CUT) ≥ .75, so the prereg reads the drug change as carried by the ongoing signal. R(CUT) is .19 from R(SHIFT), outside the ".15 within" clause; the ≥ .75 clause decides.

## What else the numbers say

**Splicing inflates the change by itself.** SHIFT removes the same samples at a random offset and raises the change by 35%. It also raises the eyes-open spread, from .0020 to .0034. So the like-for-like comparison is CUT against SHIFT.

| | Chibi | George | Kin2 | Su | Pooled |
|---|---|---|---|---|---|
| N0 | −.0189 | −.0087 | −.0115 | −.0137 | −.0133 |
| CUT | −.0185 | −.0120 | −.0150 | −.0181 | −.0154 |
| SHIFT | −.0233 | −.0121 | −.0168 | −.0224 | −.0179 |
| CUT / SHIFT | .79 | .99 | .89 | .81 | .86 |

On that comparison, the events carry about a seventh of the change pooled, and at most about a fifth in any one animal. This is one surrogate draw per window and variant, so these are point estimates.

**Per agent, CUT against N0:**

| Agent | N0 | CUT | CUT / N0 |
|---|---|---|---|
| Ketamine | −.0159 | −.0118 | .74 |
| Ketamine + medetomidine | −.0155 | −.0206 | 1.33 |
| Medetomidine | −.0062 | −.0071 | 1.15 |
| Propofol | −.0117 | −.0133 | 1.14 |

Ketamine is the only agent that shrinks. Every session of every agent still falls under CUT: 4/4, 11/11, 4/4 and 4/4. Sleep is near zero throughout: N0 −.0010 (7/7 down), CUT −.0005 (6/7).

**The session that motivated the test grows under the cut.** This is Chibi's ketamine + medetomidine, with the most extreme window (drug:00, −17.5 wake SDs) and visible large events in drug:00 and drug:02.

| Session | N0 | CUT | SHIFT |
|---|---|---|---|
| 20110621 | −.0279 | −.0363 | −.0433 |
| 20110622 | −.0216 | −.0235 | −.0233 |

**The cut is heavy, and it is not a drug effect.** Mean fraction of samples removed per window:

| Eyes closed | Eyes open | Sleep | Ketamine | Ketamine + medetomidine | Medetomidine | Propofol |
|---|---|---|---|---|---|---|
| .144 | .201 | .165 | .204 | .192 | .129 | .090 |

- Five robust SDs in any of 32 channels, ±0.2 s, removes far more than a few events, because these recordings are heavy-tailed.
- Drug windows lose no more samples than eyes-open wake, so the transients are not themselves a drug effect.

## Limits

- One surrogate draw per window and variant, 32-channel subsets, four animals.
- One threshold (5 robust SDs, ±0.2 s), fixed in advance and not tuned.
- The test shows the change does not need brief large events. It does not identify what the events are, and it says nothing about slower nonstationarity; for that, see the segment surrogates in [nonstat-prereg.md](nonstat-prereg.md).
