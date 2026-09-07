# Interaction-share pilot: execution and current results

Protocol: `configs/analysis/interaction-share-260908.yaml`. This is a fresh
exploratory comparison following the raw-only generator audit, not confirmation.
See [scientific design](interaction-share-pilot.md) for the target and its limits.

## Completed on 2026-09-08

- Generated 320 independent masters: 60 source-training and 100 evaluation per
  family. Training contributes 120 views; evaluation contributes 400 nested
  views. Physical N=32; source M16/T1000; shift M8/T500. Generator seed 26090817
  is separate from both feasibility samples.
- Verified all 520 observed arrays against stored masters; all 200 evaluation
  master pairs are nested; four exact simulator replays spanning both splits
  and families verify input/future-label separation. Data manifest binds hashes.
- Raw controls complete: 30 fits per method (two directions x three label budgets
  x five subsets). Every validation label counts in the budget. Raw fits were computed at ea0bf5e.
  A subsequent fold-PCA cache removes redundant work across ridge strengths;
  every candidate score and selected setting exactly replays across all 30
  pooled-raw fits (`pca-cache-verification.json`). No target-family
  labels, unlabelled pretraining or observation augmentation enter training.
- Neural training-only fit on four source-pool masters achieved MAE 2.52e-6.
  The existing 94,017-parameter temporal/channel encoder is unchanged; its runner
  now filters the source family and retains both evaluation families. Sixteen
  tests pass, including source-family isolation and PCA/PLS/reference selection
  isolation. Sixteen isotonic warnings in random-data tests concern uncertain
  monotonic direction; they are warnings, not failed tests.

## Fresh raw-control results

Primary joint-shift MAE, averaged equally across transfer directions:

| Method | 10 labels | 20 labels | 40 labels |
|---|---:|---:|---:|
| Source median | .2561 | .2560 | .2560 |
| Channel-wise memory | .1863 | .1848 | .1805 |
| Calibrated linear estimate | .1485 | .1457 | .1379 |
| Calibrated linear+tanh estimate | .1483 | .1451 | .1394 |
| Pooled raw summaries, selected PCA/ridge | .2277 | .2207 | .1806 |

For the linear reference at 40 labels, source-family/source-observation MAE
averages .0664, family shift alone .0822, and joint shift .1379. This identifies
observation loss as a substantial part of the task, not merely array compatibility.

The raw models choose ridge fraction from {.0001,.001,.01,.1} using two-fold
source-only calibration MAE. The strongest regularization is selected in 24/30
linear and 29/30 nonlinear fits. Thus regularization largely removes the apparent
nonlinear disadvantage in the initial fixed-ridge audit. A wider tuning search
is not justified by this result alone. Isotonic direction uses source labels only.

At 40 labels, linear-to-tanh / tanh-to-linear joint MAEs are .1425/.1334 for the
linear reference and .1472/.1317 for the nonlinear reference. These are similar
practical controls; the result does not establish that nonlinearity is irrelevant.
The initial 20-label audit's .1156 linear score did not reproduce at that magnitude
in the larger fresh population. Different samples and tuning protocols prevent a
strict before/after performance attribution; preserve both records.

**Outlook:** the target remains estimable under partial observation and neither
family-transfer direction is trivial. This is sufficient to test z, not evidence
for its utility or the applied importance of the exact target. The earlier
negative synchronization result still stands. A convincing next result would be
an incremental z or m+z gain over catalogue-matched marginals and practical raw
controls, consistent across labelled subsets and explained by an alignment test.
Beating only the neural baseline would be weaker evidence.

## Runtime and cluster boundary

Full neural fits are running locally on MPS in
`results/interaction_share_260908/neural/{linear,tanh}/`; source-family jobs run
sequentially. Logs are `logs/interaction-share-neural-{linear,tanh}.log`.
Thirty final fits are required; partial methods are not ranked by the reporter.

Initial Gadi login/data-mover attempts timed out. Diagnostic SSH subsequently
showed successful public-key authentication followed by a stall when the client
switched session QoS. Both hosts complete commands with `ssh -o IPQoS=none`.
Routing uses tunnel interface utun6; this supports a QoS/tunnel interaction,
not a verified Gadi outage. Use the option per connection, including rsync/scp;
no global SSH or network settings were changed. Live allocation is 88.34 KSU,
with no queued jobs at this check; gdata has 27.37k/70k inodes and 7.19/100 GiB.

Both isolated local p90 smoke records passed the source/output audit in 116/118s.
Each contains all 289 catalogue entries, 41,616 z coordinates and 6,647 rich
marginals. Three multitaper group-delay SPIs return no finite edges in each;
undefined features remain NaN. This is estimator missingness, not a failed farm.
Gadi smoke job `178390828` passed both records in 6:10; the dependent 48-record
node gate `178390842` is running at code ea0bf5e / pyspi 65317c9. All ten applicable non-neural tests
also pass on Gadi. Full extraction and statistical fitting follow these gates.

## Reproduction and resumption

Run from the repository root. Existing fit outputs resume only when identity and
prediction hashes match; data builders require a new output directory.

```sh
.venv/bin/python -m scripts.build_interaction_share_data --config configs/analysis/interaction-share-260908.yaml --output data/interaction_share_260908
.venv/bin/python -m scripts.run_interaction_share_pilot --config configs/analysis/interaction-share-260908.yaml --data data/interaction_share_260908 --output results/interaction_share_260908/raw --methods median memory linear nonlinear u-pca
.venv/bin/python -m scripts.run_representation_state_pilot --config configs/analysis/interaction-share-260908.yaml --data data/interaction_share_260908 --output results/interaction_share_260908/neural/linear --methods neural --device mps --source-family linear
.venv/bin/python -m scripts.run_representation_state_pilot --config configs/analysis/interaction-share-260908.yaml --data data/interaction_share_260908 --output results/interaction_share_260908/neural/tanh --methods neural --device mps --source-family tanh
```

Pin numerical libraries to two threads locally; production pyspi uses one thread
per record. `data/interaction_share_260908/manifest.json` is the data authority.
The named archive SHA-256 is
`cde6c9c2a6896d8bb7e406dd56216564375bb249362794967a71bb17307143fb`.
The external Gadi config is `configs/external/interaction-share-260908.yaml`.
Generated `smoke-indices.txt`, `node-indices.txt`, `source-indices.txt` and
`shift-indices.txt` select 2, 48, 320 and 200 records respectively. Local smoke config lives alongside the data;
it must not replace the production config or silently mix environment provenance.

For production submission, follow `AGENTS_CLUSTER_CONTEXT.md`: verify allocation,
queue, storage and both repository commits; sync committed code and the source
archive through the data mover. Validate the archive on Gadi, then run the two
record smoke and representative-node gate. Use the existing external corpus farm
launcher with separate source/shift index files, selecting resources from measured
runtime tails. No new scientific generator or observation grid is needed.

The audited extraction-to-fit worker is `jobs/gadi/run_interaction_share_analysis.pbs`,
submitted with dependencies on both production farms. Once all 520 MPIs pass
their source/config/output audit:

```sh
.venv/bin/python -m scripts.build_representation_state_bank --config configs/analysis/interaction-share-260908.yaml --data data/interaction_share_260908 --mpi-root /g/data/ql44/we2614/interaction_share_260908/mpis/interaction-share-260908 --output results/interaction_share_260908/features.npz
.venv/bin/python -m scripts.run_interaction_share_pilot --config configs/analysis/interaction-share-260908.yaml --data data/interaction_share_260908 --output results/interaction_share_260908/statistical --feature-bank results/interaction_share_260908/features.npz --methods m-pca z-pca m+z-pca m-pls z-pls m+z-pls validity-pca
.venv/bin/python -m scripts.report_interaction_share_pilot --config configs/analysis/interaction-share-260908.yaml --data data/interaction_share_260908 --inputs results/interaction_share_260908/raw results/interaction_share_260908/neural results/interaction_share_260908/statistical --output results/interaction_share_260908/report
```

Rich marginals contain 23 summaries per SPI. All 289 p90 SPIs enter the bank;
unified ordered z has 41,616 coordinates before training-only filtering. The same
PCA caps {1,2,4,8,16,32}/ridge grid and PLS components {1,2,4} apply to m, z and
m+z; PLS follows the same imputation, clipping and block weighting and does not
rescale coordinates a second time. The reporter preserves all four family/shape
cells and uses paired master-level bootstrap intervals conditional on fitted models.
