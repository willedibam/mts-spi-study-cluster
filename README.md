# MTS–SPI study

Research code, experiments and notebooks for SPI–SPI representations and order-parameter inference.

- [Workstream context](docs/context/INDEX.md): current questions, evidence and reproduction pointers.
- [Cluster operations](AGENTS_CLUSTER_CONTEXT.md) and [storage layout](docs/operations/gadi-storage-layout.md).
- `notebooks/embeddings/` and `notebooks/inference/`: retained study notebooks.
- `notebooks/cases/`, `notebooks/optimization/`, `notebooks/benchmark/` and `notebooks/presentation/`: examples, optimisation, benchmarks and presentation material.
- `src/`, `scripts/`, `configs/`, `jobs/` and `tests/`: implementation, runners, experiment definitions, cluster launchers and tests.

Local datasets and feature banks live together under `data/`; see the [data layout](data/README.md). Analysis outputs live under `results/` with matching study groups. These directories are ignored apart from explicitly tracked documentation and selected results; a clone does not include the full datasets. Temporary review/render output belongs in ignored `tmp/`.
