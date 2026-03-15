# Data Generation Reference — EEML GNN-on-SPI Study

*For the `eeml-2026-application` agent. Describes what the data is and how to regenerate or extend it.*

---

## Generators

All generators embed a small motif into M nodes; remaining nodes are independent AR(1) nuisance (ρ=0.5, noise_std=0.2). Motif node indices are randomly permuted per sample.

| Generator | Registry name | Classes | Motif | What it tests |
|-----------|--------------|---------|-------|---------------|
| A: directed VAR motifs | `var_chat_a` | chain (0→1→2), fork (0→1,0→2), collider (1→0,2→0) | 3 nodes | Can SPIs distinguish directed topologies? |
| B: common-driver confounder | `var_chat_b` | no-direct (0→1,0→2), with-direct (adds 1→2) | 3 nodes | Can SPIs detect a direct edge amid confounding? |
| C: nonlinear coupling | `var_chat_c` | linear g(u)=cu, tanh g(u)=tanh(cu) | 2 nodes | Can SPIs distinguish linear vs nonlinear coupling? |
| D: lag discrimination | `var_chat_d` | lag-1 (0→1 at τ=1), lag-3 (0→1 at τ=3) | 2 nodes | Can temporal SPIs discriminate lag structure? |

## Current Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| M | 20 | 2–3 motif nodes + 17–18 nuisance |
| T | 300 | Time series length |
| Instances/class | 500 | 9 classes × 500 = 4500 total |
| Coupling (A, D) | α ∈ [0.25, 0.8] | Uniform per sample |
| Coupling (B) | a, b, c ∈ [0.25, 0.8] | Uniform per sample |
| Coupling (C) | c ∈ [2.0, 5.0] | Strengthened from [0.5, 1.5] to make tanh saturation visible |
| ρ nuisance | 0.5 | AR(1) coefficient for non-motif nodes |
| noise_std | 0.2 | Gaussian innovation noise |
| zscore | true | Per-channel z-scoring |

## SPIs (30 dimensions, ~28 active after filtering)

Config: `configs/pyspi/cases/topology-chat.yaml`

| Family | SPIs |
|--------|------|
| Basic | cov_EmpiricalCovariance, spearmanr-sq, kendalltau-sq |
| Lagged correlation | corr_pearson_tau-{1..5}, corr_spearman_tau-{1..5} |
| Cross-correlation | xcorr_max |
| Distance | pdist_euclidean, pdist_cosine, dtw |
| Info-theoretic | je_gaussian, ce_gaussian, mi_gaussian, mi_kraskov, tlmi_gaussian, **tlmi_kraskov** |
| Transfer entropy | te_kraskov, gc_gaussian |
| Spectral | psi_multitaper, plv_multitaper, psi_wavelet |
| Misc | lmfit_Ridge, ~~lmfit_Lasso~~ (zero-variance, always dropped) |

## Data Location

Main dataset: `data/eeml_260315/chat/<class>/M20_T300_I*/`

Each instance directory contains:
- `timeseries.npy` — (300, 20) float32
- `spi_mpis.npz` — 30 keys, each (20, 20)
- `meta.json` — generator params, seed, motif edges, coupling values

## Running

```bash
# Generate + compute SPIs (PBS cluster)
qsub jobs/physics/run_eeml_chat.pbs

# Or locally (single instance)
python -m src.run_experiments \
  --job-index 1 \
  --experiment-config configs/generate/topology-chat.yaml \
  --threads 4 --parquet

# Export to eeml-2026-application
cd ../eeml-2026-application
./scripts/export_data.sh ../mts-spi-study-cluster/data/eeml_260315/chat
```

---

## Adding or Changing Generators

If you need to add a new generator or modify parameters:

**Parameter changes only** (coupling ranges, noise, M, T): edit `configs/generate/topology-chat.yaml`. Change `base_output_dir` to a new dated path to avoid overwriting existing data.

**New generator**: requires changes in `mts-spi-study-cluster` (not `eeml-2026-application`):

1. **Write the generator** in `src/generators/chat.py` — follow the pattern of existing `generate_var_chat_*` functions. Must accept `M, T, rng, return_internals` and return `(T, M) ndarray` or `(ndarray, ChatMotifInternals)`.

2. **Register it** in `src/generators/registry.py` — add import and registry entry (e.g., `"var_chat_e": generate_var_chat_e`).

3. **Export it** in `src/generators/__init__.py` — add to import and `__all__`.

4. **Wire `return_internals`** in `src/run_experiments.py` — add the registry name to the `elif spec.generator in (...)` check (~line 338). This ensures motif edges are saved to `meta.json`.

5. **Add classes to config** in `configs/generate/topology-chat.yaml` — one entry per class with `name`, `labels`, `generator`, and `base_params`.

6. **Update PBS array size** in `jobs/physics/run_eeml_chat.pbs` — set `#PBS -J 1-N` where N = total classes × instances/class.

7. **In `eeml-2026-application`**: add the new generator group to `_GENERATOR_GROUPS` in `eeml/run_pipeline.py` (e.g., `"chat-e": ["chat-e-class0", "chat-e-class1"]`). No other changes needed — the pipeline discovers SPI dimensions and class structure from the data.
