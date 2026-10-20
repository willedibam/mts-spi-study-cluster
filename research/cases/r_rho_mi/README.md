# Pearson, Spearman and mutual information

- `configs/`: generated corpora, named by date and generator setting. `260924_beta.yaml` is the current one.
- `spis/`: the SPI sets these corpora are run with (`r_rho_mi_260924.yaml` for the current corpus).
- `notebooks/`: `r_rho_mi_261020.ipynb` is the current figure viewer (Fig. 1-style edge planes, interpretation and flags; outputs in `results/cases/r-rho-mi-261020/`). `r_rho_mi_261016.ipynb` holds the construction, matrices and pair-type decomposition; the others are earlier explorations.
- `gen_sweep_configs.py`: writes a parameter sweep of configs from a base config.

```bash
python -m mtsspi.run_corpus --config research/cases/r_rho_mi/configs/260924_beta.yaml --count-only
python -m research.cases.refresh_case_figures --help
```
