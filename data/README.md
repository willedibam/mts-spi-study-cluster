# Data layout

Group data by study: `proof/`, `order-parameter-inference/`, `representation/`, `zenodo_7118947/`, `pyspi-optimisation/`, and `cases/`.

- Keep time series in `inputs/` and computed SPI banks in sibling `features/` folders within the relevant dataset or run.
- Add dataset, system or run folders only when needed to distinguish them; create no empty layers.
- Give distinct new studies descriptive folders. Use `other/` only for miscellaneous items.
- Keep one copy of each dataset. Put figures, fitted models and analysis tables under `results/`, grouped by the same study names.

This is the target layout; older data has not all been moved yet.
