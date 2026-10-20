# Euclidean distance and dynamic time warping

- `configs/`: generated corpora, named by date. `260924_lagged-warping.yaml` and its `-extra` extension are the current ones.
- `spis/`: the SPI sets these corpora are run with (`dtw_euclidean_norm.yaml` for the current corpus).
- `notebooks/`: `pdist-euclid_dtw_261020.ipynb` is the current figure viewer (Fig. 1-style edge planes, redundancy checks, interpretation and flags; outputs in `results/cases/dtw-euclidean-261020/`). `pdist-euclid_dtw_261016.ipynb` holds the generating maps, fitted alignments and realised warp.

```bash
python -m mtsspi.run_corpus --config research/cases/dtw_euclidean/configs/260924_lagged-warping.yaml --count-only
python -m research.cases.refresh_case_figures --help
```
