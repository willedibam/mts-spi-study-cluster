# SPI sets

Corpus configs name an SPI set bundled with pyspi (`pyspi_config: benchmarked_p90`) or give a path to one of the study-specific sets here (`cases/`, `subsets/`, `test.yaml`).

`benchmarked_p90.yaml` is not for new runs. It is the frozen 289-SPI catalogue that results computed before October 2026 (pyspi computation r7) were made with, and the analysis scripts of those studies read it for their SPI order. pyspi's bundled `benchmarked_p90` has 283 SPIs: the same set without six `dspli`/`dswpli` `_max` SPIs that are constant on every input. It goes when those studies are retired.
