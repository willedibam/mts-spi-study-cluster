# Cases

Small generated corpora in which one property is varied so that a few named statistics can be compared directly.

| Study | Contrast |
| --- | --- |
| [`r_rho_mi/`](r_rho_mi/README.md) | Pearson correlation, Spearman correlation and mutual information as the dependence goes from linear to monotone to non-monotone |
| [`dtw_euclidean/`](dtw_euclidean/README.md) | Euclidean distance, lag-tolerant distance and dynamic time warping under lags and warping |
| [`delay_alias/`](delay_alias/README.md) | Zero-lag and lag-tolerant similarities on synchronous and delayed MTS with identical zero-lag correlation matrices (introduction Fig. 1) |

- `refresh_case_figures.py`: analysis and figures for both studies from existing results (it never runs pyspi).
- `build_refresh_case_notebooks.py`: writes the dated viewer notebooks into each study's `notebooks/`.
- `legacy_case_figures.py`: the six plots still used by the older exploratory cases; the June r–rho–MI raincloud is defined in its notebook.

Context for the September 2026 reruns is in [`docs/context/cross-mt-transfer.md`](../../docs/context/cross-mt-transfer.md).
