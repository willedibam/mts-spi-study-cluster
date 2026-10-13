# MTS–SPI study

Research code, experiments and notebooks for SPI–SPI representations and order-parameter inference.

- [Workstream context](docs/context/INDEX.md): current questions, evidence and reproduction pointers.
- [Cluster operations](HPC.md) and [storage layout](docs/operations/gadi-storage-layout.md).
- `notebooks/embeddings/` and `notebooks/inference/`: retained study notebooks.
- `notebooks/cases/`, `notebooks/optimization/`, `research/methods/spi_benchmark/notebooks/` and `notebooks/presentation/`: examples, optimisation, benchmarks and presentation material.
- [`research/`](research/README.md): one folder per research stream and study (being populated; `cases/` first).
- `src/mtsspi/`: the library (runner, corpora, generators, features). `scripts/`: command-line entry points and plotting. `configs/`, `jobs/` and `tests/`: experiment definitions, cluster launchers and tests.

Local datasets and feature banks live together under `data/`; see the [data layout](data/README.md). Analysis outputs live under `results/` with matching study groups. These directories are ignored apart from explicitly tracked documentation and selected results; a clone does not include the full datasets. Temporary review/render output belongs in ignored `tmp/`.

## Running pyspi over a corpus

```bash
uv sync --locked --inexact                      # the environment; pyspi is the sibling checkout ../pyspi-fork
python -m mtsspi.run_corpus --config configs/generate/test.yaml --list
python -m mtsspi.run_corpus --config configs/generate/test.yaml --job-index 1 --n-jobs 4
```

A corpus config either generates its recordings (`mts_classes:`) or names a prepared archive (`source:`); both run under the same contract, and each recording's `meta.json` records the config, code and pyspi version that produced it. In a worktree or source snapshot, set `PYTHONPATH=src:.` so that the copy you are in is the one imported.

## October 2026 restructure

On 13 October 2026 the repository was restructured around pyspi 3.0: one runner (`mtsspi.run_corpus`) replaced the synthetic and external runners, their launchers and the workarounds for seeding and memory that pyspi now handles; the package was renamed from `src` to `mtsspi`; `main` became the trunk; and `uv.lock` became the single environment definition. Results computed before it used pyspi computation r7; later ones use r9 and must not be pooled with them. The commit before the restructure is tagged `pre-restructure-261013`.
