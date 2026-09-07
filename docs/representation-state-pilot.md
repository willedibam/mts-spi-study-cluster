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

A saved MPS checkpoint also reproduced 12 predictions across all observation
cells on CPU within 1.20e-7. Current checkpoint metadata includes PyTorch's
`TorchVersion` string subclass: weights-only loading needs the narrow
`torch.serialization.safe_globals([torch.torch_version.TorchVersion])` context.

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

The generated evaluation masters have hidden past-versus-future coherence MAE
0.00587. This supports describing the target as a persistent macroscopic state,
not difficult long-horizon forecasting. Hidden past coherence is a diagnostic
only, unavailable to predictors and not a lower bound on achievable error.

## Completed neural/simple comparison

All 15 neural refits and 120 inner fits completed in 1,662 seconds on local MPS.
The model has 94,017 parameters. Joint-shift MAE at 16/32/64 labels is
0.08899/0.08176/0.07645, versus 0.04259/0.04208/0.03973 for the two-observable
ridge. Paired conditional differences (neural minus observables) are
0.04641 [0.04149,0.05100], 0.03968 [0.03468,0.04435], and
0.03672 [0.03125,0.04203]. The same-cell neural curve is
0.07419/0.06801/0.06347. Eighteen of 120 candidate folds hit the epoch cap;
none of the 30 selected folds did. This evaluates one bounded neural pipeline,
not unrestricted neural learnability or pretrained-model performance.

Artifacts: `results/representation_stage_b_260907/final/neural-simple-report/`
contains the complete paired report, figure and numeric results.

An additional **post hoc, zero-label physics diagnostic** checked the exact
unclipped squared-coherence correction under uniform finite-population sampling.
Using analytic phases, clipping and a square root does not preserve the identity's
unbiasedness guarantee. It did not improve prediction: joint-shift MAE 0.04648
versus 0.04441 uncorrected. Retained in `sampling-bias-diagnostic.json`; script
`check_representation_state_sampling_bias.py` includes an exhaustive subset
identity check. It uses the known physical N=32 and is not a general real-data
procedure or an addition to the frozen primary method list.

## Frozen-initialization diagnostic and pending extraction

After the initial neural comparison, a frozen-initialization control extracted
the 128-dimensional pooled vector before the readout. It uses the same initial
weights for each matched seed, no pretraining or fitted input transformation, and
the same train-only PCA/ridge protocol as the statistical controls. All 15 fits
completed; joint-shift MAE 0.06851/0.06421/0.06015 beats end-to-end neural training
but loses to the two physical observables. The paired neural-minus-frozen gaps
are 0.02048 [0.01641,0.02455], 0.01755 [0.01354,0.02150], and
0.01629 [0.01287,0.01979]. This is a post-result diagnostic, not a retrospectively
preregistered method. Feature extraction takes about one second per initialization
for all 672 views; feature banks and exact training subsets are retained.

Reproduce using `scripts/run_representation_state_random_control.py`; results and
combined plots are in `results/representation_stage_b_260907/random-control-report/`.
This result makes a win over the end-to-end encoder alone even less persuasive.

Both Gadi scouts passed: 2/2 records in job 178344541 and 48/48 in job
178350735. The node scout took 38:38 walltime at 16 concurrent workers, with
25.7 GiB peak aggregate memory. Median per-record times were approximately
65/114/217/396/915/1771 seconds for M8/T500, M8/T1000, M16/T500,
M16/T1000, M32/T500 and M32/T1000 respectively. These are affordable with
Gadi dataset-level parallelism; comparison with a cheap raw feature pass does
not make p90 an allocation bottleneck. The staged scouts account for elapsed
setup time. Fifty records are complete and audited.

The remaining 622 records are submitted in six homogeneous farms,
178355803–178355808, with 622 workers across 960 reserved CPUs. Larger-M
farms reserve additional CPU capacity for memory. Source/runtime identity is
unchanged; job commit 633eff3 adds optional per-record diagnostic logs.
Submission resources and IDs are in `data/representation_stage_b_260907/production-submissions.tsv`
and `results/representation_stage_b_260907/execution-status.json`.
Audit all 672 outputs before building the aligned feature bank. No Stage B z
performance result exists yet.

Additional post-neural-result controls reuse the existing 82 pooled raw features:
joint-shift MAE is 0.07891/0.06735/0.05841, or 0.07830/0.06677/0.05803
when analytic-phase coherence is appended. Neither improves on the two-observable
ridge. The combined report is `results/representation_stage_b_260907/control-report/`.
It also separates all six M/T cells: at 64 labels, two-observable MAE is 0.0624
for M8/T500 and 0.0171 for M32/T500. Their pooled average must not be described
as invariant performance. Reproduce with `scripts/run_representation_state_raw_control.py`;
these additions use the same labels, PCA/ridge and source-only fitting rules.

## Recovery and interpretation — 2026-09-07

The first production farms 178355803–178355808 all failed before extraction:
`nci-parallel` uses `shell: none`, so the added `>`/`2>&1` logging text reached
Python's argument parser. The completion audits correctly failed. No scientific
result comes from those jobs. Commit fdabcc9 reverts the logging change to the
successful scout launcher. Six replacement farms 178356467–178356472 use the
same records/resources and no per-record shell redirection. Do not duplicate them.
The complete raw manifest, masters, targets and observables are now hash-verified
on Gadi. Comparison-fit archives were staged separately from checkpoints.

`jobs/gadi/run_representation_state_analysis.pbs` runs a full output audit,
builds the feature bank, fits all six statistical methods and merges the report.
Job 178356603 was submitted with afterok dependencies on all six replacement farms. Output root is
`/g/data/ql44/we2614/representation_stage_b_260907/analysis/`.
Gadi uses NumPy2.5.2/scikit-learn1.9.0, whereas local controls used 2.3.5/1.7.2;
identities record these versions. Treat cluster fits as provisional and replay
statistical fits locally from the downloaded feature bank before final comparison.

Scientific decision sequence:

1. Complete the unchanged synchronization pilot. Positive Stage A class results
   and a potentially negative Stage B state result can coexist. This task has a
   strong known coherence observable and does not isolate complex interaction
   structure. A weak trained neural result is not evidence for z.
2. Investigate Stage A attribution with richer per-SPI marginals and a matched
   nonlinear head, validity controls, and cross-SPI alignment disruptions. Keep
   development choices separate from a later fresh confirmation population.
3. Before expanding to another dataset, name an externally meaningful target
   whose changes should alter relationships among linear/nonlinear/lagged
   dependence estimates across edges. This is a proposed mechanism, not verified
   recoverability of coupling laws. Control observable nuisance differences and
   test a shared target on held-out generator families; generator identity alone
   cannot supply this test.
4. Advance a representation-learning paper only if incremental value survives
   these controls and replication. If simple summaries explain the gains, keep
   the result as a representation/benchmark section and narrow the novelty claim.

The motivation is to understand when cross-statistic relationships are useful,
not select only tasks on which z wins. Negative tasks define the scope. Diversity
of dependence statistics already has precedent (Cliff et al., 2023); incremental
utility of their relationships is the claim requiring new evidence.
