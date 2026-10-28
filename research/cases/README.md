# Cases

Small generated corpora in which one property is varied so that a few named statistics can be compared directly.

| Study | Contrast |
| --- | --- |
| [`r_rho_mi/`](r_rho_mi/README.md) | Pearson correlation, Spearman correlation and mutual information as the coupling map goes linear → monotone → folded (`form_ladder_261028.py`) |
| [`ed_xed_dtw/`](ed_xed_dtw/README.md) | Euclidean distance, lag-tolerant distance and DTW as delays go absent → constant → drifting over time (`drift_261028.py`) |
| [`delay_alias/`](delay_alias/README.md) | Zero-lag and lag-tolerant similarities on synchronous and delayed MTS with identical zero-lag correlation matrices (introduction Fig. 1) |
| [`r_xcorr_p/`](r_xcorr_p/README.md) | Zero-lag correlation, maximum cross-correlation and phase-aligned correlation P as delays go absent → shared → frequency-dependent (`dispersion_261028.py`) |

- `case_run.py`: shared runner and working system figure for the one-knob cases (z over all ordered off-diagonal entries). Generators: `mtsspi.generators.case_ladders` and `band_delayed_mts`.
- `refresh_case_figures.py`: analysis and figures for both studies from existing results (it never runs pyspi).
- `edge_planes.py`: the 2026-10-20 Fig. 1-style edge-plane figures and checks for both studies, from the same archive.
- `build_refresh_case_notebooks.py`: writes the dated viewer notebooks into each study's `notebooks/`.
- `legacy_case_figures.py`: the six plots still used by the older exploratory cases; the June r–rho–MI raincloud is defined in its notebook.

Context for the September 2026 reruns is in [`docs/context/cross-mt-transfer.md`](../../docs/context/cross-mt-transfer.md).
