# Cases

Small generated corpora in which one property is varied so that a few named statistics can be compared directly.

| Study | Contrast |
| --- | --- |
| [`r_rho_mi/`](r_rho_mi/README.md) | Pearson correlation, Spearman correlation and mutual information as the coupling map goes linear → monotone → folded (`form_ladder_261028.py`) |
| [`ed_xed_dtw/`](ed_xed_dtw/README.md) | Euclidean distance, lag-tolerant distance and DTW as delays go absent → constant lag → time-varying warp, each in its own block (`drift_blocks_261029.py`) |
| [`delay_alias/`](delay_alias/README.md) | Zero-lag and lag-tolerant similarities on synchronous and delayed MTS with identical zero-lag correlation matrices (introduction Fig. 1) |
| [`r_xcorr_p/`](r_xcorr_p/README.md) | Zero-lag correlation, maximum cross-correlation and phase-aligned correlation P as delays go absent → shared → frequency-dependent (`dispersion_261028.py`) |

- `case_run.py`: shared runner and working system figure for the one-knob cases (z over all ordered off-diagonal entries). Generators: `mtsspi.generators.case_ladders` and `band_delayed_mts`.
- `composite_style.py`: shared look of the case figures.
- `cases_page_261026.py`: first mock-up of a one-page case figure (form over timing), kept for reference.

The dated viewers of the 24 September corpus and their builders were removed on 18 November 2026; see [research/README.md](../README.md).

Context for the September 2026 reruns is in [`docs/context/cross-mt-transfer.md`](../../docs/context/cross-mt-transfer.md).
