# Data layout

Group data by study: `proof/`, `order-parameter-inference/`, `representation/`, `zenodo_7118947/`, `pyspi-optimisation/`, and `cases/`.

- Keep time series in `inputs/` and computed SPI banks in sibling `features/` folders within the relevant dataset or run.
- Add dataset, system or run folders only when needed to distinguish them; create no empty layers.
- Preserve development/confirmation splits and coherent historical run folders, including mixed inputs and outputs.
- Give distinct new studies descriptive folders. Use `other/` only for miscellaneous items.
- Keep one copy of each dataset. Put figures, fitted models and analysis tables under `results/`, grouped by the same study names.

For example, `proof/p90_260824/inputs/confirmation/` contains held-out time series; sibling `features/confirmation.npz` contains their computed features. Historical paths inside notebooks, configurations and artifacts may still refer to the previous layout; update them when reusing that work.
