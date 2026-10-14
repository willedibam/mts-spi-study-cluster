# Pearson, Spearman and mutual information

- `configs/`: generated corpora, named by date and generator setting. `260924_beta.yaml` is the current one.
- `spis/`: the SPI sets these corpora are run with (`r_rho_mi_260924.yaml` for the current corpus).
- `notebooks/`: `r_rho_mi_261014.ipynb` is the current analysis viewer of the completed September corpus; the others are earlier explorations.
- `gen_sweep_configs.py`: writes a parameter sweep of configs from a base config.

```bash
python -m mtsspi.run_corpus --config research/cases/r_rho_mi/configs/260924_beta.yaml --count-only
python -m research.cases.refresh_case_figures --help
```
