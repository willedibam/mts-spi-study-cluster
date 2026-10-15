# Inference notebooks

New model figures should follow the concise [benchmark figure style](../../../../docs/benchmark-figure-style.md).

- `order-parameter-benchmarks-lean.ipynb` is the meeting-facing selection: Kuramoto, Stuart–Landau, Miller–Huse, quadratic CML, 2D logistic CML and Rössler, with equations, exact sweep grids, source links and three-control composites. It retains the 2D CML sensitivity/failure plots and excludes Desai–Zwanzig/Vicsek. Rebuild with `python -m research.inference.benchmarks.build_lean_order_parameter_notebook`; illustrative snapshot preparation is `python -m research.inference.benchmarks.prepare_lean_benchmark_snapshots`. The original comparison notebook is unchanged.

- `order-coordinate-three-systems.ipynb` is the compact visual narrative: CML2D independent confirmation, quadratic-CML reorganisation, the M24 varying-independent-sensor-noise locking example, and hardware r9 ER fit/small-world projection. It reads frozen results through `research.inference.benchmarks.three_systems` and `coordinate_features`, with matched top16 loading galleries and original-edge explanations; executing it validates source hashes and score replays and writes PNG/SVG/PDF plus a provenance receipt under `results/order-parameter-inference/three-systems-261015/`. No extraction or model refitting.
