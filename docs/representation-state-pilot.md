# State pilot: matched statistical and raw neural baselines

This is an exploratory test of future finite-population coherence under changes
in observation count and duration. It is a calibration/falsification pilot for
the representation-learning programme, not evidence of a transferable general
dynamical state or a publication-tier result.

## Data and labels

The [generation proposal](../configs/analysis/representation-stage-b-proposal-260907.yaml)
produced 192 independent 32-oscillator Kuramoto masters: 96 training-pool and 96
evaluation masters, balanced over eight reduced couplings. Each master contains
1,000 observed cosine samples and a label from the *following* 1,000 samples of
hidden full-population phase coherence. Hidden phases, coupling and frequencies
are excluded from predictor inputs. Coupling indices only balance sampling,
validation folds and conditional bootstrap strata.

Training uses one M16/T1000 observation per selected master. Evaluation uses all
six M8/16/32 × T500/1000 views. Channels are nested in a random sensor order;
windows are suffixes ending at the same input endpoint. Every view shares its
master's future target. Physical population size is always 32. SPIs must be
recomputed on each view; slicing a larger MPI is not a substitute.

The 16/32/64-label budgets include **all inner tuning labels**. Five deterministic
nested training subsets use seeds 11, 23, 47, 71 and 101. No additional training
views, unlabelled pretraining, target adaptation, or test-selected early stopping
are provided to any method.

## Models and controls

The [execution protocol](../configs/analysis/representation-stage-b-260907.yaml)
reuses the exact generation/sampling specification and explicitly records the
pre-comparison neural correction described below.

- p90 catalogue controls: m (mean/SD/quantiles per SPI), z (all ordered edges,
  Pearson SPI-pair correlations), m+z, m+g, m+g+z and validity-only.
- All statistical feature transformations, clipping, PCA and ridge selection are
  fitted within each training fold. Alpha selection uses two folds and final
  refitting uses all labels in the selected budget.
- Simple controls: direct analytic-phase coherence, direct mean absolute
  correlation, separately calibrated versions of each, their two-feature ridge
  combination, and a training-mean predictor. Direct versions require no labels.
  Hilbert transforms use only observed cosines, including finite-window edge
  effects. No hidden phase input is supplied to these controls.
- Raw neural comparator: shared temporal convolutions, two attention blocks over
  channels **at each aligned time patch**, a post-interaction temporal convolution,
  mean/max pooling, and a small regression head. No channel-identity embedding is
  used because sensors are exchangeable oscillators in this pilot.
- AdamW learning rate and weight decay use the same two inner folds. The final
  epoch count is the rounded median of the selected candidate's best inner-fold
  epochs. All predictions are clipped to [0,1].

This small encoder is a credible first variable-channel comparator, not an
exhaustive evaluation of pretrained models. Its learning curves must be examined
before interpreting poor results: frequent best epochs at the cap indicate an
unresolved optimization-budget limitation. Published encoders and pretraining
comparisons belong in the next stage if the scientific task warrants investment.

## Neural implementation diagnostic and correction

The original proposal used dropout 0.1. A disjoint four-example cosine diagnostic
showed tiny stochastic training loss but deterministic inference MAE 0.0728 at
learning rate 0.001 and 0.0251 at 0.0001. Dropout preceding max pooling can change
the distribution of the pooled feature at inference. Setting dropout to zero,
with all other architectural components and tuned AdamW weight decay retained,
reduced diagnostic MAE to 5.61e-6 after 200 epochs. The corrected configuration
was chosen before any held-out pilot neural score had completed. The original
diagnostics and partial inner-CV log are retained, not used as an additional
claim-bearing comparison.

Tests cover variable M/T, channel-permutation invariance, genuine cross-channel
influence before pooling, finite gradients, train-only ridge selection, nested
views, equal label accounting and grouped report construction. The optimization
diagnostic demonstrates fitting ability, not statistical generalization.

## Reproduction and outputs

From the repository root, using the project's Python environment:

```bash
python -m scripts.build_representation_state_data \
  --config configs/analysis/representation-stage-b-proposal-260907.yaml \
  --output data/representation_stage_b_260907
python -m scripts.check_representation_state_neural \
  --config configs/analysis/representation-stage-b-260907.yaml \
  --output results/representation_stage_b_260907/neural-optimization-check-final.json \
  --device mps
python -m scripts.run_representation_state_pilot \
  --config configs/analysis/representation-stage-b-260907.yaml \
  --data data/representation_stage_b_260907 \
  --output results/representation_stage_b_260907/final/neural \
  --methods neural --device mps
```

Omit `--methods neural --device mps` and choose another output directory to run
the simple controls. Supported neural devices are CPU, MPS and CUDA; runtime and
device identity are recorded. Do not silently mix devices or library versions
inside a resumed curve. Builders refuse to overwrite data; model outputs resume
only if protocol, code, data and prediction hashes match. Existing data should be
verified and reused instead of regenerated merely to reproduce fitted models.

Gadi extraction uses `configs/external/representation-stage-b-260907.yaml` and
the existing external-corpus farm, starting with two records and a representative
single-node scout. The archive contains observed inputs only, with empty class
tags. Its source digest is
`b3065e3421bdae69cca077dff994d0eaecc93ecc8150a7b491f7251739d809c4`.
The source corpus and new MPI files live under
`/g/data/ql44/we2614/representation_stage_b_260907/`, not inode-constrained Scratch.

`scripts/build_representation_state_bank.py` verifies each member's raw hash,
catalogue, shape, normalization, SPI ordering and pyspi version before extracting
all statistical views into one aligned bank. Supply it to the same runner using
`--feature-bank ... --methods m z m+z m+g m+g+z validity`.

`scripts/report_representation_state_pilot.py` merges completed curves only: every
method needs all three label budgets and five subset seeds. Confidence intervals
resample evaluation masters within coupling, averaging their observation views
and fitted subsets. They are exploratory and conditional on the fitted models;
they do not quantify variation across freshly drawn training populations.

## Decision rule

Judge z against the strongest simple control as well as catalogue-matched m and
the raw encoder. Beating only an undertrained neural baseline is insufficient.
Approximately 10% relative MAE reduction is a provisional resource-allocation
gate, not a field-established scientific effect threshold. A clear failure may
instead justify a specific corrective test, with its rationale recorded first.

If simple coherence remains sufficient, retain this as a control/limitation and
seek a task where cross-statistic interaction information has a reason to matter.
Do not make the benchmark artificially hostile to a known physical observable.
Cross-generator transfer of a shared property and a credible real observation
shift are subsequent scientific requirements, not consequences of fitting this
finite-population pilot.
