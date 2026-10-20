# Cases

Small generated corpora in which one property is varied so that a few named statistics can be compared directly.

| Study | Contrast |
| --- | --- |
| [`r_rho_mi/`](r_rho_mi/README.md) | Pearson correlation, Spearman correlation and mutual information as the dependence goes from linear to monotone to non-monotone |
| [`dtw_euclidean/`](dtw_euclidean/README.md) | Euclidean distance, lag-tolerant distance and dynamic time warping under lags and warping |

- `refresh_case_figures.py`: analysis and figures for both studies from existing results (it never runs pyspi).
- `edge_planes.py`: the 2026-10-20 Fig. 1-style edge-plane figures and checks for both studies, from the same archive.
- `build_refresh_case_notebooks.py`: writes the dated viewer notebooks into each study's `notebooks/`.
- `legacy_case_figures.py`: the six plots still used by the older exploratory cases; the June r–rho–MI raincloud is defined in its notebook.

Context for the September 2026 reruns is in [`docs/context/cross-mt-transfer.md`](../../docs/context/cross-mt-transfer.md).
