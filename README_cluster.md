## Overview

`mts-spi-study-cluster/` is a self-contained, PBS-friendly driver for running large multivariate time-series (MTS) experiment sweeps with PySPI. It reuses the generators that underpin the main project (VAR(1), CML, Kuramoto, Gaussian/Cauchy noise), but exposes them through declarative configs so that the same code path works on laptops, head nodes, and Taiji-style PBS Pro clusters.

## Layout

```
mts-spi-study-cluster/
├── configs/                 # experiment + PySPI configs
│   ├── experiments_dev.yaml
│   ├── experiments_full.yaml
│   ├── basic_config.yaml
│   ├── info_theory_config.yaml
│   ├── spectral_config.yaml
│   └── oliver_spi_config.yaml
├── data/
│   ├── dev/                 # dev datasets land here
│   └── full/                # full-scale datasets land here
├── jobs/
│   ├── run_spi_array.pbs    # PBS array script
│   └── run_spi_single.py    # convenience wrapper for local runs
├── logs/                    # PBS/stdout logs (one per array index)
├── src/
│   ├── compute.py           # thin PySPI wrapper + MPI reconstruction
│   ├── generate.py          # generator registry (VAR, CML, Kuramoto, noise)
│   ├── mapping.py           # YAML parsing + job-index mapping
│   ├── run_experiments.py   # main entry point (local + PBS)
│   └── utils.py             # metadata helpers + loaders
├── pyproject.toml           # uv/pip project definition
└── requirements.txt         # fallback dependency list
```

Per-dataset folders live under `data/{dev,full}/<Class>/M{M}_T{T}_I{I}[_variant]/`. Each folder contains:

- `calc.csv` (+ optional `calc.parquet`) – raw PySPI table
- `spi_mpis.npz` – consolidated archive of all SPI adjacency matrices
- `mts_heatmap.png` – optional quick diagnostic (only extras like `mts_heatmap_deltaK.png` when explicitly configured)
- `meta.json` – metadata (see below)

No `results/` tree is created; downstream analysis can build its own workspace by walking the structured data directories.

## Environment setup

> The repo root already contains a `.venv`; these instructions only cover the standalone cluster driver. If you want one consolidated environment, install `requirements.txt` at the repo root (it now includes `-r mts-spi-study-cluster/requirements.txt`).

### Cluster prerequisites

1. **Load site modules (Taiji examples)**  
   ```
   module load python/3.11.7
   module load java/21.0
   module load octave/8.4.0          # only if Octave-based SPIs are enabled
   ```

2. **Create/activate a venv or uv environment inside `mts-spi-study-cluster/`** (see below).  

3. **Java/JIDT are auto-started** – `src/java_bridge.py` runs before PySPI and tries, in order:
   - `PYSPI_JVM` (if you set it),
   - `JAVA_HOME`/`JAVA_HOME/jre`,
   - `jpype.getDefaultJVMPath()` (works on Taiji after `module load java/...`).

   In practice you can skip manual exports; only set these env vars if the auto-detect cannot find your cluster’s JDK.

4. **Octave-backed SPIs (via `oct2py`)**: ensure `octave` is on `PATH` or set `OCTAVE_EXECUTABLE=/usr/bin/octave`. No change is needed if you only use PySPI kernels that stay in Python.

5. **Test before queueing**:
   ```
   python -m src.run_experiments --mode dev --job-index 1 --dry-run
   python -m src.run_experiments --mode dev --job-index 1
   ```

### Preferred (`uv`)

```
cd mts-spi-study-cluster
uv sync
uv run python -m pip list
```

`uv` reads `pyproject.toml` and will install PySPI + numeric deps in a reproducible lockfile.

### Fallback (`venv` + `pip`)

```
cd mts-spi-study-cluster
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PBS script tries to `source .venv/bin/activate` if present; otherwise ensure `python` on the cluster resolves to an environment that already has PySPI installed.

## Experiment configs & mapping

- `configs/experiments_dev.yaml` and `configs/experiments_full.yaml` are the single source of truth for which MTS classes, sizes (`M`, `T`), instances, and variants to run. Each file has a `defaults:` block (shared `M_values`, `T_values`, `instances`); individual classes only override those when absolutely necessary.
- Each class selects a generator (`var`, `cml_logistic`, `kuramoto`, `gaussian_noise`, `cauchy_noise`), provides default parameters, and optionally defines variants (e.g. CML α/ε combos, Kuramoto directed vs undirected).
- `src/mapping.py` expands the Cartesian product of `(class × M × T × instances)` into a deterministic list, then distributes variants across those datasets in a round-robin fashion so each class still contributes exactly the same number of combinations (e.g. 45 datasets per class with 3 Ms × 5 Ts × 3 instances). PBS job indices (1-based) map directly onto this list. If `include_base_variant: true` (default), the “base” parameter set participates in that rotation alongside any variant blocks.
- Run `python -m src.run_experiments --mode full --count-only` after editing the YAML to obtain the new array size, then update `#PBS -J 1-N` inside `jobs/run_spi_array.pbs`.

## Running locally

Quick sanity check on a workstation (default dev config):

```
cd mts-spi-study-cluster
python -m src.run_experiments --mode dev --job-index 1 --heatmap
```

Alternative with the helper wrapper:

```
python jobs/run_spi_single.py --mode dev --job-index 3
```

Use `--list` to inspect dataset mappings and `--dry-run` to print a summary without doing any work.

## Running on PBS

1. Ensure you are inside `mts-spi-study-cluster/` before submitting.
2. Adjust `#PBS -J 1-XXX` in `jobs/run_spi_array.pbs` to match `--count-only`.
3. Submit the array:

   ```
   qsub jobs/run_spi_array.pbs
   ```

4. Monitor with `qstat -u $USER` (Taiji also provides `qload`).
5. Cancel with `qdel <jobid>`.

Logs land in `mts-spi-study-cluster/logs/<jobname>_<arrayindex>.{out,err}`. All datasets for the `full` profile are written under `mts-spi-study-cluster/data/full/...`.

The script exports `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` so NumPy/SciPy respect the PBS CPU request. Pass `--threads` to `run_experiments.py` (or edit the YAML) to keep code + environment in sync.

## PySPI configs & directed SPIs

- `configs/basic_config.yaml`, `configs/info_theory_config.yaml`, `configs/spectral_config.yaml`, and `configs/oliver_spi_config.yaml` follow the standard PySPI layout (`.statistics.*` groups).
- The default experiment wiring runs *dev* jobs with `configs/fast_config.yaml` (copied from `pyspi/fast_config.yaml`) and *full* jobs with the comprehensive `configs/config.yaml`.
- `src/java_bridge.py` is imported before `pyspi.calculator.Calculator` so the JVM is started exactly once per worker; configure it with `PYSPI_JVM` / `JAVA_HOME`.
- `src/compute.py` parses the labels to determine whether an SPI is directed or undirected.
- Dataset metadata retains, for every SPI, the lower-cased label list and a `directed` flag so downstream tooling knows whether a matrix was symmetrised.
- Select a different PySPI config per run with `--pyspi-config` or by editing the experiment YAML.

## Metadata schema

Each `meta.json` includes:

```json
{
  "name": "CauchyNoise_M5_T250_I0",
  "mode": "dev",
  "mts_class": "CauchyNoise",
  "M": 5,
  "T": 250,
  "instance_index": 0,
  "variant": {
    "name": "",
    "params": {},
    "slug": ""
  },
  "generator": {
    "name": "cauchy_noise",
    "params": {},
    "seed": 42024
  },
  "pyspi": {
    "config": "configs/basic_config.yaml",
    "subset": "default",
    "n_spis": 8,
    "spis": [
      {"name": "SpearmanR", "directed": false, "labels": ["undirected", "..."]},
      {"name": "TransferEntropy", "directed": true, "...": "..."}
    ]
  },
  "paths": {
    "timeseries": "",
    "calc_csv": "calc.csv",
    "calc_parquet": "",
    "spi_archive": "spi_mpis.npz",
    "per_spi": {},
    "heatmap": "",
    "heatmaps": []
  },
  "job": {
    "index": 3,
    "threads": 4,
    "heatmap": false,
    "compute_seconds": 112.8
  },
  "timestamp": "2025-11-14 06:12:30"
}
```

Absolute Windows paths never appear; everything is relative to the cluster project root.

## Post-processing helpers

`src/utils.py` exposes ready-made loaders so notebooks can aggregate results quickly:

```python
from src import utils

# Tabular view of every meta.json
meta_dev = utils.load_all_meta("dev")

# Load metadata + MPI tensors for one dataset
bundle = utils.load_spi_for_dataset("data/dev/CauchyNoise/M5_T250_I0")
bundle.mpis["SpearmanR"].shape  # -> (M, M)

# Iterate through all datasets in a mode
for dataset in utils.load_all_spi("full"):
    print(dataset.dataset_path, dataset.meta["pyspi"]["n_spis"])
```

Because `meta.json` carries generator parameters, variant information, and directed/undirected SPI flags, downstream analysis (e.g. pandas/xarray pipelines) can join everything without guessing.

## Next steps

1. Adjust `configs/experiments_*.yaml` as new dynamical models or larger grids are required.
2. Run `python -m src.run_experiments --mode <mode> --list` to verify the mapping order before submitting cluster jobs.
3. Hook post-processing notebooks directly into `data/{dev,full}/...` by using the helper loaders instead of re-reading CSVs manually.
