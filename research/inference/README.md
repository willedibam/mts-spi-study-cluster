# Inference

Can an unlabelled SPI–SPI coordinate track a system's order parameter across a phase transition? Each study is one dynamical system, with its physics scouts (to find a well-behaved transition), its corpus configs and its analysis.

Presentation entry points: [visual introduction with feature interpretation](benchmarks/notebooks/order-coordinate-three-systems.ipynb) (collective period-two CML, variable-noise harmonic locking and version-consistent ER/small-world hardware), [lean benchmarks](benchmarks/notebooks/order-parameter-benchmarks-lean.ipynb), [extended benchmarks with cross-frequency locking and feature interpretation](benchmarks/notebooks/order-parameter-benchmarks-lean-extended.ipynb), [full comparison](benchmarks/notebooks/order-parameter-benchmark-comparison.ipynb), and [secondary candidates](benchmarks/notebooks/order-parameter-benchmarks-lean-2.ipynb). Channel-pair sampling is [under methods](../methods/pair_sampling/notebooks/channel-pair-sampling.ipynb). Paths and runtime references were checked after the research-layout move; data/results remain under the repository-level artifact directories.

| Study | System |
| --- | --- |
| `kuramoto/` | Kuramoto oscillators: synchronisation |
| `stuart_landau/` | Stuart–Landau oscillators: locking boundary |
| `desai_zwanzig/` | Desai–Zwanzig mean-field bistability |
| `miller_huse/` | Miller–Huse coupled maps (Ising-like spin order) |
| `cml/` | Quadratic and 2D logistic coupled map lattices: period doubling |
| `locking/` | Cross-frequency locking and the feature sensitivity behind the locking appendix |
| `candidate_search/` | Systems scouted for a sharper transition and not taken forward: dependence-character transitions (generalised synchronisation, copulas, tail alignment), TASEP, Vicsek, finite regime, Lorenz-96, Rössler, Hindmarsh–Rose, ZGB, breathing chimera, dial network; `transition_sharpness.py` compares them |
| `benchmarks/` | The cross-system comparison: lean and extended benchmark notebooks, simple order-parameter baselines |

Within a study, `configs/` holds corpus configs (`python -m mtsspi.run_corpus --config ...`), `configs/scout/` and `configs/analysis/` the configs read by that study's scout and analysis scripts, `notes/` its protocols and results, `jobs/` its Gadi launchers. The stream's context is [`docs/context/order-parameter-benchmarks.md`](../../docs/context/order-parameter-benchmarks.md) and [`docs/context/cml-order-parameter-inference.md`](../../docs/context/cml-order-parameter-inference.md).
