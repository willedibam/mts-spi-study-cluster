# SPI sets

Corpus configs name an SPI set bundled with pyspi (`pyspi_config: benchmarked_p90`) or give a path to one of the study-specific sets here (`cases/`, `subsets/`, `test.yaml`).

`benchmarked_p90.yaml` is not for new runs. It is the frozen 289-SPI catalogue that results computed before October 2026 (pyspi computation r7) were made with, and the analysis scripts of those studies read it for their SPI order. pyspi's bundled `benchmarked_p90` has 283 SPIs: the same set without six `dspli`/`dswpli` `_max` SPIs that are constant on every input. It goes when those studies are retired.

New SPIs reach existing corpora as companion corpora, not by changing a catalogue: a corpus YAML identical to the source corpus except for `name`, `base_output_dir` and `pyspi_config`, run into its own output directory and joined per recording with `mtsspi.companion.join_companion`, which checks that both sides hold the same data and disjoint SPIs. `subsets/phasealigned.yaml` is phase-aligned correlation P alone (`phasealigned_welch_nperseg-auto`, pyspi `v3` from `c3de193`); the runner computes it for the whole 1,053-record Zenodo archive locally in under a minute. Most existing corpora are r7 and P exists only from r9, so joining P onto them needs `allow_mixed_computation=True`, and the analysis must record both versions (the join's `provenance`).
