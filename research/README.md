# Research

One folder per research stream, one folder per study inside it. A study holds what is specific to it: its corpus and analysis configs, its SPI sets, its scripts, tests and notebooks, and a README stating the question, the status and how to reproduce it. Code used by more than one study lives in the library (`src/mtsspi/`), not here.

| Stream | Scope |
| --- | --- |
| [`cases/`](cases/README.md) | Small worked cases that isolate one contrast between statistics |
| `inference/` | Order-parameter and phase-transition inference |
| `proof/` | The p90 proof-of-concept embeddings, validation and control baselines |
| `representation/` | The SPI–SPI representation and what it captures, including the Neurotycho transfer work and the learned baselines |
| `empirical/` | Measured data |
| `zenodo/` | The Zenodo MTS corpus map |
| [`methods/`](methods/README.md) | Computing SPIs at scale |

`cases/` and `methods/` have been moved (October 2026); the other streams' files are still under `scripts/`, `configs/`, `tests/` and `notebooks/`. Large inputs and outputs stay in the ignored `data/` and `results/` directories, under the study's name.

Study folders are importable (`python -m research.cases.refresh_case_figures`), so their names use underscores.
