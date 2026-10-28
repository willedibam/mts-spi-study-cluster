# Timing over time: Euclidean distance, lag-tolerant Euclidean distance, DTW

## Current case (2026-10-28): `drift_261028.py`

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
