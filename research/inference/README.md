# Inference

Can an unlabelled SPI–SPI coordinate track a system's order parameter across a phase transition? Each study is one dynamical system, with its physics scouts (to find a well-behaved transition), its corpus configs and its analysis.

| Study | System |
| --- | --- |
| `kuramoto/` | Kuramoto oscillators: synchronisation |
| `stuart_landau/` | Stuart–Landau oscillators: locking boundary |
| `desai_zwanzig/` | Desai–Zwanzig mean-field bistability |
| `miller_huse/` | Miller–Huse coupled maps (Ising-like spin order) |
| `cml/` | Quadratic and 2D logistic coupled map lattices: period doubling |
| `locking/` | Cross-frequency locking and the feature sensitivity behind the locking appendix |
| `dependence_transition/` | Transitions in dependence character at fixed strength: copulas, tail alignment, marginal controls |
| `system_scouts/` | Systems scouted and not taken forward: TASEP, Vicsek, finite regime, Lorenz-96, Rössler, Hindmarsh–Rose, ZGB, breathing chimera, dial network |
| `benchmarks/` | The cross-system comparison: lean and extended benchmark notebooks, simple order-parameter baselines |

Within a study, `configs/` holds corpus configs (`python -m mtsspi.run_corpus --config ...`), `configs/scout/` and `configs/analysis/` the configs read by that study's scout and analysis scripts, `notes/` its protocols and results, `jobs/` its Gadi launchers. The stream's context is [`docs/context/order-parameter-benchmarks.md`](../../docs/context/order-parameter-benchmarks.md) and [`docs/context/cml-order-parameter-inference.md`](../../docs/context/cml-order-parameter-inference.md).
