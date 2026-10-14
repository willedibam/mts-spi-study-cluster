# Euclidean distance and dynamic time warping

- `configs/`: generated corpora, named by date. `260924_lagged-warping.yaml` and its `-extra` extension are the current ones.
- `spis/`: the SPI sets these corpora are run with (`dtw_euclidean_norm.yaml` for the current corpus).
- `notebooks/`: `pdist-euclid_dtw_261014.ipynb` is the current analysis viewer of the completed September corpus.

```bash
python -m mtsspi.run_corpus --config research/cases/dtw_euclidean/configs/260924_lagged-warping.yaml --count-only
python -m research.cases.refresh_case_figures --help
```
