# Benchmark figure style

Follow the general [figure style](figure-style.md). These conventions apply to [`order-parameter-benchmark-comparison.ipynb`](../notebooks/inference/order-parameter-benchmark-comparison.ipynb):

- Physical truth `Q`: near-black circles. Frozen `q`: one consistent colour and square markers. Use Viridis-derived colours when distinguishing `M`.
- For run-averaged curves, use lightly shaded (`alpha≈0.16`) 95% bootstrap intervals across independent runs; label alternatives explicitly.
- Mark known boundaries when useful with a thin grey dotted line.

Canonical colours: `Q="#222222"`; `M={8:"#440154", 16:"#31688e", 20:"#26828e", 32:"#35b779"}`.
