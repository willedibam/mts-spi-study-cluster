# Inference notebooks

New model figures should follow the concise [benchmark figure style](../../docs/benchmark-figure-style.md).

- `order-parameter-benchmarks-lean.ipynb` is the meeting-facing selection: Kuramoto, Stuart–Landau, Miller–Huse, quadratic CML, 2D logistic CML and Rössler, with equations, exact sweep grids, source links and three-control composites. It retains the 2D CML sensitivity/failure plots and excludes Desai–Zwanzig/Vicsek. Rebuild with `python -m scripts.build_lean_order_parameter_notebook`; illustrative snapshot preparation is `python -m scripts.prepare_lean_benchmark_snapshots`. The original comparison notebook is unchanged.
