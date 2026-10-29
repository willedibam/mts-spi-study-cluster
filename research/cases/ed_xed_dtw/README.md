# Timing over time: Euclidean distance, lag-tolerant Euclidean distance, DTW

## Current case (2026-10-29): `drift_blocks_261029.py`, figure `drift_overview_261029.py`

Lag and warping each in their own block, so each $z$ answers to one mechanism. M = 15 in three blocks of five read one smooth mother (the same OU process as below; generator `delay_blocks_mts`): block 1 aligned; block 2 aligned or lagged by 1–5 samples; block 3 aligned or warped by $A_k\sin(2\pi t/250)$, $A_k=1.2$–6 samples, one phase. Conditions: no lag, lag, warping, and all three (the recording in panel a). Noise 0.1–0.4, five levels per block in the order (1, 4, 2, 0, 3), rank-uncorrelated with lag and warp. Lags are whole samples because xED searches whole-sample shifts; fractional lags (1.25–5) left a residual that xED tracked (pilot, 2026-10-29). 20 recordings per condition; outputs in `results/cases/drift-blocks-261029/`.

| $z$ (ordered entries) | no lag | lag | warping | all three |
| --- | --- | --- | --- | --- |
| ED–xED | 1.00 | 0.26 | 1.00 | 0.43 |
| xED–DTW | 0.99 | 0.99 | 0.51 | 0.51 |

- In the lag condition ED tracks the pair's $|\Delta\tau|$ and xED does not; under warping xED tracks the warp-amplitude difference $|\Delta A|$ and DTW does not (it ranks pairs by noise).
- Stacking warp on lag in the same channels (the 2026-10-28 design below, or lags 0–5 on all channels plus warp) makes $z$(ED, xED) climb back (0.17 → 0.74 in the pilot), because warped pairs raise ED and xED together. Keeping the mechanisms in separate blocks avoids that; "all three" shows the mixture.

## Previous case (2026-10-28): `drift_261028.py`

One system, one knob: whether each channel's delay is absent, constant, or drifts slowly over time. Channel $i$ reads one smooth mother at $t-\tau_i-Dc_i\sin(2\pi t/250+\theta_i)$ (generator `drifting_delay_mts`). The mother is an Ornstein–Uhlenbeck process on an 8× finer grid with lag-one autocorrelation 0.9, so fractional and drifting delays do not smooth the signal. Levels: no delay; $\tau_i=0$–4 samples; the same plus drift $D=3$ and $D=6$, with $c_i\sim U(0,1)$ so pairs drift by different amounts. M = 12, T = 1,000, observation noise log-spaced in 0.1–0.4, 20 recordings per level, the same draws at every level. SPIs: `spis/dtw_euclidean_norm.yaml` (xED is the minimum over shifts $|h|\le10$). Outputs in `results/cases/drift-261028/`.

| $z$ (ordered entries) | no delay | delay | drift 3 | drift 6 |
| --- | --- | --- | --- | --- |
| ED–xED | 1.00 | 0.30 | 0.46 | 0.77 |
| xED–DTW | 0.99 | 0.96 | 0.31 ± 0.05 | 0.21 ± 0.06 |

- **Rung 1 duplicates the other timing case:** on z-scored data ED $=\sqrt{2-2r}$ exactly and xED ≈ $\sqrt{2-2\max_h r(h)}$.
- **Why the rebound-warp design was replaced.** Every channel is warped equally and independently, so warp does not vary across pairs. With a rough mother ($a=0.5$), xED saturates for all pairs by $p=0.1$, and $z$(xED, DTW) drops as a step that is not organised by pair deformation (z(DTW, deformation) ≈ 0.02). A half-warped variant returns to $z\approx0.99$, because DTW cost also rises with jump-type warps. The step itself is not the problem; the missing pair-level mechanism is.
- **Mechanism (scratch check, 2026-10-28):** xED tracks each pair's drift amplitude ($z\approx0.94$), while DTW does not (−0.11) and instead tracks noise. Narrower noise spreads give larger breaks: with noise 0.1–0.8, xED–DTW is only 0.75 / 0.51 at $D=3$ / 6.
- **$z$(ED, xED) rises with drift** because both statistics lose the drifting pairs.

## Earlier material

- `configs/`: generated corpora, named by date. `260924_lagged-warping.yaml` and its `-extra` extension are the current ones.
- `spis/`: the SPI sets these corpora are run with (`dtw_euclidean_norm.yaml` for the current corpus).
- `notebooks/`: `pdist-euclid_dtw_261020.ipynb` is the current figure viewer (Fig. 1-style edge planes, redundancy checks, interpretation and flags; outputs in `results/cases/dtw-euclidean-261020/`). `pdist-euclid_dtw_261016.ipynb` holds the generating maps, fitted alignments and realised warp.

```bash
python -m mtsspi.run_corpus --config research/cases/ed_xed_dtw/configs/260924_lagged-warping.yaml --count-only
python -m research.cases.refresh_case_figures --help
```
