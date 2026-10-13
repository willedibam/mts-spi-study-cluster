# Methods

Computing SPIs at scale: what is expensive, and how to compute less without changing the answer.

| Study | Content |
| --- | --- |
| [`pair_sampling/`](pair_sampling/README.md) | Retrospective test of channel-pair (dyad) sampling: how many dyads reproduce the SPI–SPI representation, on a 48-record large-M panel (CML, MEG, VAR) |
| `pyspi_scaling/` | Reference copies of pyspi's notebooks on workers within a recording, dyad sampling, and both together (the source is `pyspi-fork/demos/optimisation/`) |
| `spi_benchmark/` | Per-SPI wall time across M and T, and the complexity fits behind the p90 selection |

The pyspi side of this work (workers, `compute_pairs`, the budget rule, measured timings) is documented in pyspi itself: README "Running at scale" and `bench/pair_sampling/README.md`.
