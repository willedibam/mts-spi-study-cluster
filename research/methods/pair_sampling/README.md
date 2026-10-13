# Channel-pair sampling

Does a uniform sample of channel dyads reproduce the SPI–SPI vector computed from all of them, and how many dyads are needed? The findings are in [`assessment.md`](assessment.md).

- `large_m_pair_baselines.py`: prepares and extracts the full-table reference panel (48 records, M = 64, 100 and 243).
- `large_m_var_replacement.py`: the VAR records that replaced the first panel's.
- `scout_spi_pair_sampling.py`, `scout_spi_coordinate_sampling.py`: draw dyad samples from the full tables and measure the error.
- `plot_large_m_pair_sampling.py`, `plot_pair_sampling_coordinate_gallery.py`: the convergence figures and coordinate gallery.
- `notebooks/channel-pair-sampling.ipynb`: the summary notebook.

These results are pyspi computation r7. pyspi's `compute_pairs` was validated against them (`pyspi-fork/bench/pair_sampling/validate.py retrospective`).
