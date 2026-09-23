# MTS–SPI study

Research code, experiments and notebooks for SPI–SPI representations and order-parameter inference.

- [Workstream context](docs/context/INDEX.md): current questions, evidence and reproduction pointers.
- [Cluster operations](AGENTS_CLUSTER_CONTEXT.md) and [storage layout](docs/operations/gadi-storage-layout.md).
- `notebooks/embeddings/` and `notebooks/inference/`: retained study notebooks.
- `notebooks/cases/`, `notebooks/optimization/`, `notebooks/benchmark/` and `notebooks/presentation/`: examples, optimisation, benchmarks and presentation material.
- `src/`, `scripts/`, `configs/`, `jobs/` and `tests/`: implementation, runners, experiment definitions, cluster launchers and tests.

Local `data/`, `features/` and `results/` are excluded from Git. A clone does not include those artifacts. Preserve each notebook's actual inputs before deleting a bank; filenames containing “p90” can refer to the older **297-SPI** catalogue, whereas the current `benchmarked_p90.yaml` has **289 SPIs**. The retained `proof_p90_260712.ipynb` uses the older catalogue.

See the [local data review](docs/operations/local-data-audit-260923.md) for dependencies and cleanup candidates. New workstreams may use new descriptive folders; no fixed folder registry is required. Temporary review/render output belongs in ignored `tmp/`.
