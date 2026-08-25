# Workflow: pyspi-fork + mts-spi-study-cluster

## Repository Structure

### Local
```
~/Desktop/2025USYD/USYD/
├── pyspi-fork/                    # github.com/willedibam/pyspi (branch v3, pyspi 3.0)
│   └── pyspi/statistics/          # SPI implementations (numpy, no JIDT)
└── mts-spi-study-cluster/         # github.com/willedibam/mts-spi-study-cluster (branch refactor-lagged-warping)
    ├── configs/pyspi/             # SPI configs (copies of pyspi 3.0's own)
    ├── src/                       # Pipeline code
    └── jobs/physics/              # PBS scripts
```

### Cluster (Physics PBS)
```
/suphys/wedi0306/
├── pyspi-fork/                                # branch v3, editable-installed
└── mts-spi-study-cluster -> /import/taiji1/wedi0306/mts-spi-study-cluster

/import/taiji1/wedi0306/mts-spi-study-cluster/ # main repo and bulk storage
├── .venv/
└── data/
```

## 1. Editing pyspi

Edit directly in `pyspi-fork/pyspi/`. Editable install means changes are **immediately live**.

- SPIs: `pyspi-fork/pyspi/statistics/<module>.py`
- Configs: `mts-spi-study-cluster/configs/pyspi/`

## 2. Syncing to cluster

### Push locally, pull on cluster — both repos:
```bash
# --- Local ---
cd pyspi-fork && git add -A && git commit -m "msg" && git push origin v3
cd ../mts-spi-study-cluster && git add -A && git commit -m "msg" && git push origin refactor-lagged-warping

# --- Cluster ---
cd /suphys/wedi0306/pyspi-fork && git fetch --prune && git merge --ff-only origin/v3
cd /suphys/wedi0306/mts-spi-study-cluster && git fetch --prune && git merge --ff-only origin/refactor-lagged-warping
```

After pulling, changes are live (editable install). No reinstall unless `pyproject.toml` changed.

### First-time cluster setup
```bash
# 1. Put the main repo on bulk storage and pyspi in home
git clone --branch refactor-lagged-warping \
  https://github.com/willedibam/mts-spi-study-cluster.git \
  /import/taiji1/wedi0306/mts-spi-study-cluster
ln -s /import/taiji1/wedi0306/mts-spi-study-cluster /suphys/wedi0306/mts-spi-study-cluster
cd /suphys/wedi0306
git clone https://github.com/willedibam/pyspi.git pyspi-fork
cd pyspi-fork
git checkout v3

# 2. Create the venv inside the bulk-backed main repo
cd /suphys/wedi0306/mts-spi-study-cluster
uv venv .venv --python 3.12

# 3. Install both repositories editable
source .venv/bin/activate
uv pip install -e .                       # install mts-spi-study-cluster
uv pip install -e /suphys/wedi0306/pyspi-fork  # editable install of fork
```

## 3. Running experiments

```bash
# PBS array job
qsub jobs/physics/run_eeml.pbs

# Single job (testing)
source .venv/bin/activate
python -m src.run_experiments \
  --job-index 1 \
  --experiment-config configs/generate/eeml.yaml \
  --threads 4 --parquet
```

## Notes

- No JVM, JIDT or `jpype` anywhere: pyspi 3.0 removed the Java dependency
  outright, and nothing starts a JVM on import.
- `MAX_CORR_AIS` auto-embedding is implemented in numpy and enabled; it now
  selects the source embedding as well as the destination.
- `configs/pyspi/*.yaml` are verbatim copies of `pyspi-fork/pyspi/configs/*.yaml`.
  Re-copy them after any change to the fork's configs; `cases/` and `test.yaml`
  are this repo's own and have no upstream counterpart.
