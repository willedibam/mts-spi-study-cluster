# Workflow: pyspi-fork + mts-spi-study-cluster

## Repository Structure

### Local
```
~/Desktop/2025USYD/USYD/
├── pyspi-fork/                    # github.com/willedibam/pyspi (branch v3, pyspi 3.0)
│   └── pyspi/statistics/          # SPI implementations (numpy, no JIDT)
└── mts-spi-study-cluster/         # github.com/willedibam/mts-spi-study-cluster (branch refactor)
    ├── configs/pyspi/             # SPI configs (copies of pyspi 3.0's own)
    ├── src/                       # Pipeline code
    └── jobs/physics/              # PBS scripts
```

### Cluster (Physics PBS)
```
/suphys/wedi0306/                              # NFS home (code lives here)
├── pyspi-fork/                                # fork, editable-installed
└── mts-spi-study-cluster/
    ├── .venv -> /taiji1/.../mts-spi-study-cluster/.venv   # symlink
    └── data  -> /taiji1/.../mts-spi-study-cluster/data    # symlink

/taiji1/wedi0306/mts-spi-study-cluster/        # bulk storage (107T)
├── .venv/                                     # actual venv (heavy)
└── data/                                      # experiment output
```

## 1. Editing pyspi

Edit directly in `pyspi-fork/pyspi/`. Editable install means changes are **immediately live**.

- SPIs: `pyspi-fork/pyspi/statistics/<module>.py`
- Configs: `mts-spi-study-cluster/configs/pyspi/`

## 2. Syncing to cluster

### Push locally, pull on cluster — both repos:
```bash
# --- Local ---
cd pyspi-fork && git add -A && git commit -m "msg" && git push origin v2
cd ../mts-spi-study-cluster && git add -A && git commit -m "msg" && git push origin refactor

# --- Cluster ---
cd /suphys/wedi0306/pyspi-fork && git pull origin v2
cd /suphys/wedi0306/mts-spi-study-cluster && git pull origin refactor
```

After pulling, changes are live (editable install). No reinstall unless `pyproject.toml` changed.

### First-time cluster setup
```bash
cd /suphys/wedi0306

# 1. Clone both repos
git clone https://github.com/willedibam/mts-spi-study-cluster.git
cd mts-spi-study-cluster
git checkout refactor

cd /suphys/wedi0306
git clone https://github.com/willedibam/pyspi.git pyspi-fork
cd pyspi-fork
git checkout v2

# 2. Create venv directly on taiji1, symlink back to NFS
mkdir -p /taiji1/wedi0306/mts-spi-study-cluster
cd /taiji1/wedi0306/mts-spi-study-cluster
uv venv .venv --python 3.12
ln -s /taiji1/wedi0306/mts-spi-study-cluster/.venv /suphys/wedi0306/mts-spi-study-cluster/.venv

# 3. Symlink data output to taiji1
mkdir -p /taiji1/wedi0306/mts-spi-study-cluster/data
ln -s /taiji1/wedi0306/mts-spi-study-cluster/data /suphys/wedi0306/mts-spi-study-cluster/data

# 4. Install
cd /suphys/wedi0306/mts-spi-study-cluster
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
