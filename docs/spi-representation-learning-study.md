# SPI–SPI representation learning: proposed study

Status: historical rationale and method menu, updated 2026-09-09. Subsequent
covariance, co-organization, learned-pooling and faster-regime studies are complete.
See [current scientific framing](spi-representation-scientific-framing.md) and
[verified transfer findings](oscillatory-transfer-findings.md) for the present
claim and next evidence gap. The stage descriptions below preserve the earlier
design; [state-pilot execution](representation-state-pilot.md) records that pilot's
neural implementation checks, data contract and reproduction commands.
This is not a preregistered benchmark or evidence of superiority. The existing
proof confirmation set has already been inspected; new decisions made using it
are exploratory and require fresh confirmation.

## Administration and readiness

The scientific direction is sufficiently specified to begin bounded exploration,
but the pilot is not a finalized paper benchmark or a frozen confirmation.
The completed work includes numeric baseline extraction, source alignment,
geometric auditing and a budgeted supervised catalogue-matched screen. The new
runner is `scripts/run_representation_screen.py`; the historical cross-M,T runner
remains v2-specific and was not changed. Stage B now includes raw neural training
with the same label budgets and inner validation as the statistical readouts.

Use one study coordinator and one versioned protocol per experiment. Keep this
document as the rationale/method menu; runnable configurations must state actual
choices and must not silently inherit exploratory notebook defaults. An experiment
record should bind question, target, allowed data, independent group IDs, views,
split manifest, representations, preprocessing, hyperparameter search, random
seeds, metrics, compute budget and the decision its result informs. Do not turn
the context index into a run diary; retain durable conclusions in the indexed
representation-evaluation context and machine-readable run metadata with outputs.

| Stage | Work | Advance only when |
|---|---|---|
| A: descriptor/pipeline screen | Reuse existing proof MPIs to compare m, z, m+z, m+g, m+g+z with a shared grouped evaluation runner and regularized linear heads; reconstruct unified z explicitly. Use Zenodo for coverage/geometry, not automatic source-tag classification. | Source/schema alignment, split isolation, validity handling and score differences are understood. Saturated generator classification is not sufficient to reject or establish the scientific hypothesis. |
| B: state/observation pilot | Choose one independently defined macroscopic target; fixed physical master systems; independently generated train/validation/test masters; specified sensor subsets and time windows. Include one competent raw temporal-plus-channel-interaction model. | The target survives the chosen views often enough to be inferable; performance has room to differ; a practically relevant gain or a specific remediable failure appears. |
| C: focused learning/replication | Add only the extension indicated by B, plus a published variable-channel encoder and stronger held-out system/configuration tests. Assess one or two real-data candidates against the same problem contract. | The effect replicates beyond the discovery setting and cannot be explained by catalogue breadth, preprocessing, validity or tuning differences. |
| D: frozen confirmation | Freeze target, splits, models, tuning rules, primary metric and effect criterion; acquire fresh independent confirmation data; execute without tuning on it. | Report the result, including a null or adverse result. Any subsequent redesign starts a new discovery/confirmation cycle. |

Before Stage B, explicitly resolve these six items:
1. Target definition and measurement: regime/macroscopic state, source of labels,
   time interval and relationship to observed input; no simulator-label shortcut.
2. Independent unit: master realization/subject/asset and all derived views,
   with training/validation/test grouping before any feature fitting.
3. Observation shift: sensor counts, sampling scheme, spatial coverage, duration,
   sampling frequency, noise, and what training sees. Begin with coverage and
   duration separately; make their combination a later stress test.
4. Resource regime: zero-target-label transfer versus target adaptation;
   supervised-only versus a fixed training-only unlabelled pretraining pool.
   Count validation/tuning labels in the budget, or declare their separate fixed
   cost. Do not compare a two-label head against a model whose tuning used many
   unreported labelled examples.
5. Primary outcome: held-out task error/score versus independent labelled groups,
   exact learning-curve aggregation, paired uncertainty, and a minimum worthwhile
   improvement. The threshold comes from the task's usefulness/label cost, not
   statistical significance alone; estimate confirmation precision from pilot
   variability rather than selecting sample size to obtain a desired p-value.
6. Baseline and compute contract: same p90 catalogue for statistical controls;
   faithful input processing/augmentations and adequate training for the raw
   encoder; frozen tuning limits and small measured CPU/GPU scouts before scale.

Use decision checkpoints, not exhaustive factorial sweeps. At each checkpoint,
produce one comparison table, learning curves, uncertainty on method differences,
validity/size controls, and a short failure analysis. Record inconclusive results
as inconclusive. A null on one easy task does not reject a broad representation
claim; repeated failure on relevant, adequately powered tasks does justify ending
that direction. Keep untouched confirmation data out of checkpoint selection.

Initial priority: complete Stage A and specify Stage B together. Delay new matrix
metrics, large SPI graph networks, larger M and broad real-data searches until
they address an observed limitation.

Historical Stage B checkpoint (superseded by the current findings linked above):
Stage A was complete. Stage B had generated its independent masters, passed
implementation checks and completed all matched neural/simple learning curves.
The two-record Gadi p90 scout then had one verified completed record; the larger
record was still computing after a transient SSH interruption. A post hoc
frozen-initialization encoder plus ridge also beat end-to-end neural training
but lost to the physical observables; see the execution document. Published encoders and learned geometry remain conditional Stage C work.

## Scope and operational catalogue

Use `configs/pyspi/benchmarked_p90.yaml` (289 SPIs) throughout. “Full catalogue”
in this study means all of this operational p90 catalogue, not `full.yaml`.
Pin configuration hash, pyspi computation version, normalization, RNG seed and
MPI direction convention. Runtime filtering was not task-label selection, but
the resulting catalogue is not an exhaustive library of all dependence measures.
All paired statistical baselines use the same MPIs. Compare smaller SPI subsets
only as explicit cost/coverage ablations; report distinct SPI computations,
not just the number of retained SPI pairs.

Primary question: do cross-SPI edge relationships support more label-efficient
and transferable inference than individual-SPI summaries or raw neural encoders?
Secondary question: can learning suppress changes caused by observation while
retaining independently validated dynamical distinctions?

Paper-level refinement: “Can a representation of relationships among dependence
measures retain an independently defined dynamical regime/macroscopic state across
sensor-coverage and recording-length shifts, with fewer labelled independent
realizations than catalogue-matched summaries and variable-channel learned encoders?”
Cross-statistic agreement is the proposed explanation, not an established premise.
Distinguish regime inference from full latent-state or mechanism identification.
Separate zero-target-label transfer from few-shot target adaptation; report both
source labelled-data budget and target adaptation budget. Simulations with cheap
automatically available labels justify mechanistic tests, but artificially hiding
their labels is not alone a persuasive practical motivation for label efficiency.
An actual limited-independent-system application, expensive target measurement, or
other defensible resource constraint must eventually support that part of the claim.

The nearest competition extends beyond generic transformers: Vermani et al.,
ICLR 2025, *Meta-Dynamical State Space Models for Integrative Neural Data Analysis*,
learns a family of dynamics across heterogeneous recordings and tests few-shot
reconstruction/forecasting. Its target differs from whole-record state classification,
so it is adjacent positioning unless the study pivots to dynamical reconstruction.
https://proceedings.iclr.cc/paper_files/paper/2025/hash/d3222559698f41247261b7a6c2bbaedc-Abstract-Conference.html

Recommendation remains conditional: invest first in a bounded comparison that can
reject the cross-statistic hypothesis. An ML-venue case would require a substantial,
replicated learning/transfer result and explanatory ablations (a new architecture
or a theorem is not mandatory). A Nature Computational Science case would benefit
from a consequential scientific application beyond method compatibility. There is
no demonstrated basis yet for a publication-tier prediction; do not preserve Pearson
z at the expense of a better supported learned/marginal/graph or hybrid formulation.

User preference: M up to 32 is acceptable for the initial study. Larger M is
possible with p90/Gadi dataset parallelism if a scientific question requires it;
increase independent realizations first when estimating label efficiency. More
dataset workers do not lower a single dataset's M-dependent runtime/memory.

## What the geometry is

For ordered off-diagonal edge vectors, stack `E[e,a] = A_a[e]`, with
`e=1,...,q`, `q=M(M-1)`. On a valid common SPI panel, let `mu` and `sigma`
be column means and population standard deviations, and
`U=(E-mu)/(sqrt(q)*sigma)`. Pearson gives `C=U.T@U`, `diag(C)=1`;
`z=upper(C)`. Spearman has the analogous construction after ranking each column.
The histogram-MI option has no such general PSD guarantee.

- `C` belongs to the closed set of correlation matrices (the elliptope).
  It is not necessarily on the positive-definite manifold.
- `rank(C) <= min(K,q-1)`. At `K=289`, M=8 allows at most rank 55 and
  M=16 at most 239. Duplicate/proportional SPIs and symmetric edge duplication
  impose further dependence. Rank bounds do not count independent dyads.
- With identical complete SPI coordinates,
  `||C_i-C_j||_F^2 = 2*||z_i-z_j||_2^2`. Unscaled Euclidean z is already
  a legitimate matrix distance. A curved geometry is not automatically better.
- The K labelled unit vectors (columns of U) encode angles between statistical
  response patterns across edges. Their ambient dimension changes with M,
  but their K-by-K Gram matrix has fixed coordinates. Gram matrices identify
  configurations up to a common isometry of their spans; many such isometries
  do not correspond to a physically realizable transformation of raw MTS.
- Shared edge permutations leave C unchanged, including permutations not
  induced by node relabelling. Thus incidence/topology is lost at this pooling
  stage. This is stronger than the desired channel-permutation invariance.
- Eigenvalues alone discard SPI identity. They are useful diagnostics/ablations,
  not the preferred complete representation.
- `diag(sigma) @ C @ diag(sigma)` recovers the empirical covariance of edge
  descriptors. Consequently `(mu,sigma,z)` contains their first two moments;
  it still omits non-Gaussian joint structure and graph incidence.

Direction and semantics: the primary representation includes both `i,j` and `j,i`
entries, aligned across SPIs. This retains sensitivity to relative directed
patterns but not complete orientation information: transposing every MPI gives
the same z (a common edge permutation). This is an aggregation-stage identity,
not a claim about time-reversing the raw MTS. Likewise, no explicit spatial
coordinates/incidence are retained, but spatial organization may indirectly
influence z by altering the estimated dependencies. “Insensitive to all spatial
structure” and “identifies coupling mechanisms” are both too strong.

Pearson summarizes standardized linear association of two SPI edge vectors, not
equality of values or agreement on a causal mechanism. A symmetric variable U
can have `corr(U,U^2)=0` despite deterministic nonlinear dependence. Curvature,
mixtures, tails and magnitude can therefore be lost; this is a task-dependent
bottleneck, not evidence the baseline is intrinsically flawed. More expressive
alternatives must justify their estimation variance with finite dependent dyads.
Cheap first extension: m+z. Later: fixed-dimensional kernel/learned summaries of
selected bivariate SPI scatters, or set pooling over complete q_e vectors. A
single MI/distance-correlation scalar still does not retain a scatter distribution;
all bivariate marginals do not generally determine its K-dimensional joint law.
Even a complete distribution of q_e remains blind to graph incidence. See
https://www.jmlr.org/papers/v11/sriperumbudur10a.html for characteristic distribution
embeddings; population injectivity does not imply lossless finite approximations
or transfer of iid sample bounds to channel-sharing dyads.

Matrix geometry requires a shared valid SPI panel or an explicit missing-data
model. Arbitrary entrywise median/zero imputation, Fisher transforms, and
independently filtering SPI-pair coordinates do not generally preserve correlation
geometry. There is a useful exception for the current whole-SPI invalidity pattern:
zero correlations to invalid SPIs and unit diagonals produce the PSD block
completion `diag(C_valid,I_missing)` after reordering. This is a feasible sensitivity,
but invents independence for missing coordinates and can encode missingness through
the added unit eigenvalues; keep the mask and evaluate validity/size shortcuts.
Full-corpus complete-panel selection is descriptive only: benchmark selection
must use training data, and unseen invalid entries still need a policy.
Do not discard difficult test records silently or claim imputation-free coverage.

## Catalogue-matched controls (implemented first)

All entries below use the original signed ordered off-diagonal values. No
per-MPI centring/rescaling occurs before marginal extraction; raw time-series
normalization remains that of the original pyspi computation.

| View | Definition | Question |
|---|---|---|
| m | Per SPI: mean, population SD, q10/q25/median/q75/q90; 2,023 coordinates | Is catalogue breadth enough without cross-SPI alignment? |
| z | Existing unified-v3 Pearson; 41,616 coordinates | Are relationships among statistics useful? |
| m+z | Concatenated blocks, with training-fitted block balancing as a sensitivity | Does each recover information lost by the other? |
| g | Per SPI: nine weighted matrix/graph summaries; 2,601 coordinates | Does incidence or within-MPI direction structure matter? |
| m+g, m+g+z | Nested comparisons | Does z add value beyond marginal and topology summaries? |

The graph block is deliberately small: standard deviation and q10/q90 of
row-mean and column-mean weights (six); reverse-edge Pearson correlation;
largest singular-value energy fraction; and singular-energy effective rank
divided by M. Diagonals are set to zero only for the graph calculation.
Rows/columns are named neutrally, without assuming causal source/target semantics.
These are generic weighted-matrix descriptors, not a full HCGA implementation.
They avoid imposing shortest-path or threshold semantics on arbitrary signed SPIs.
Size-normalized formulas still require empirical size-sensitivity checks.

Extraction preserves NaNs. A nonfinite off-diagonal invalidates that SPI's block;
constant finite matrices retain meaningful marginal summaries, but undefined
correlation and zero-energy normalized spectral summaries stay NaN. Include a
validity-only baseline and a common-validity sensitivity, because constant/failing
estimators affect the views differently. Final selection/imputation/scaling is
training-only; the extraction artifact performs none of these operations.

Use `src/mpi_representation_baselines.py` and
`python -m scripts.extract_mpi_representation_baselines --root ROOT --output STEM`.
ROOT contains dataset directories with `spi_mpis.npz` and `meta.json`.
Output: numeric NPZ, feature names, correlation-valid masks, dataset IDs, M/T,
and JSON binding each source MPI/metadata hash to the extraction code. The
per-record rank audit uses that record's valid principal block; different blocks
are not an aligned representation. Align the existing z artifact by dataset ID,
SPI schema, source hashes and computation provenance, never by incidental row order.

## Learning ladder

1. Start with training-fitted imputation, centre/standard-scale sensitivity,
   optional PCA, and regularized linear classification/regression. Tune each
   representation using the same nested folds and search budget, not the same
   numerical regularization constant across unequal scales.
2. Compare dimension budgets d=8/16/32/64, capped by training rank. Report both
   dimension-matched and best validation-selected results. Keep the existing
   centre-only z baseline: standardization can overweight unreliable pairs.
3. Add a two-layer MLP on training-fitted PCA coordinates only if linear models
   leave an interpretable limitation. Separate a learned geometry result from
   an improvement due only to a more flexible final classifier.
4. Learn observation stability using paired views before attempting a large
   SPI graph network. Let `S_view` be mean outer products of paired-view
   differences and `S_total` the covariance of training views. Solve
   `min tr(W.T S_view W), subject to W.T S_total W=I` in a regularized PCA
   subspace. This selects directions stable across views while preventing
   collapse. It is a baseline related to slow feature analysis/multiview
   learning, not a new principle. Fit all whitening/regularization on training
   groups. Paired-view variance is not pure estimator noise unless the views
   actually preserve the relevant underlying property.
5. A nonlinear extension can use paired-view agreement plus VICReg-style
   variance/covariance penalties. Stability alone can preserve subject or
   simulator identity and suppress meaningful rare states; external tasks
   and held-out systems decide whether the embedding is useful.
6. Optional structured compression: `B=W.T C W` for K-by-d W shared across
   datasets. It preserves PSD, learns combinations of named SPI coordinates,
   and can retain d(d+1)/2 entries. Compare it with ordinary vector PCA/MLP;
   parameter count and d are different notions of compression. It cannot
   recover mean/scale/topology omitted from C. Keep SPI identities if using
   message passing on C; an unlabelled graph/eigenvalue-only model adds an
   unjustified invariance to relabelling statistical methods.

Fixed-geometry controls: unscaled Frobenius first, then principal square-root
embedding `vec_sym(sqrt(C))` (off-diagonals weighted by sqrt(2)). This is defined
on PSD matrices and costs an eigendecomposition per record. Bures–Wasserstein
distance also supports PSD matrices but pairwise computation is costlier and
transport paths need not remain correlation matrices. Log-Euclidean/SPD networks
need shrinkage such as `(1-alpha)C+alpha I`; alpha changes the representation,
especially at low rank, so fit it without test labels and report sensitivity.
There is no reason yet to prefer any of these distances over Frobenius.

## Credible raw neural comparison

Use a shared temporal CNN to create tokens `[record,channel,time_patch,feature]`.
Perform cross-channel attention/message passing at aligned time patches before
global temporal pooling. Add temporal mixing after interaction if needed,
then masked pooling over time/channels to obtain a fixed d-dimensional vector.
Temporal positions remain identifiable; arbitrary input row numbers are not
learned channel identities. Physical sensor coordinates can be an additional
metadata-aware sensitivity if the task makes them available to all methods.

A small pilot can use 64-wide shared temporal features, two interaction blocks
and a 64-dimensional readout. Its size is provisional, not a claimed optimum.
Give it the same allowed channel-subset/duration augmentations as z learning.
Separate (a) supervised from scratch, (b) self-supervised on a fixed training-only
unlabelled pool with a frozen head, and (c) fine-tuning. Published variable-channel
methods such as Brüsch et al. or UniTS provide stronger subsequent comparators.
Do not handicap the neural model by averaging independently computed univariate
summaries before it can observe synchronized dependencies.

An optional learned set over full edge descriptors q_e bridges the handcrafted
summaries and raw encoder. It can learn higher moments/interactions without
retaining incidence; a GNN on the original M-node multi-SPI graph retains it.
These comparisons isolate the pooling bottleneck from statistical extraction.

## Experiments and interpretation

For label-efficiency curves use nested subsets of independent training groups,
initially 2/4/8 realizations per class where the existing development pool permits.
Keep all views of a master realization in one split. Larger curves need more
independent simulations, not more windows or sensor subsets of existing ones.
Keep the unlabelled pretraining pool fixed while varying labels; varying both
is a separate experiment. Use repeated splits/seeds and paired group-level
uncertainty on score differences. Distinguish number of realizations N, channels M,
duration T, physical elapsed time and sampling frequency.

| Shift | Construction | Defensible conclusion |
|---|---|---|
| Independent instances, same generators | Existing proof development; historical confirmation reused only as exploration | Within-family predictive sufficiency |
| Unseen M values | Hold all records of selected M values out, not only one M,T cell | Extrapolation across size values, with physical-size caveat |
| Observation coverage | Multiple sensor subsets from independent fixed-size master systems | Robustness to partial observation, not physical-size invariance |
| Duration | Recompute SPIs from shorter stationary windows; handle transition windows separately | Performance/uncertainty versus available observation time |
| Held generator families | Same target property defined independently in every family | Transfer of that property, not generator-label recognition |

A shared target might be future global synchronization/coherence across
appropriate oscillator families, or an independently measured predictive/state
observable. A control parameter is not automatically an equivalent target across
systems. Simple coherence/correlation or spectral baselines are mandatory where
the target itself is synchronization. Current Kuramoto evidence already shows
that such a simple statistic can be stronger; do not design an easy win around z.

Channel deletion requires rerunning SPIs on the reduced raw recording: covariance
shrinkage, precision and other multivariate estimators need not equal submatrices
of the original MPI. Taking MPI submatrices tests a different operation. Different
time windows always require re-estimation. Bootstrap time blocks jointly across
channels; resampling individual edges as independent observations is invalid.

Mechanism controls:

- Same channel permutation: exact correctness check, not a scientific result.
- Independent node permutations for different SPIs: preserve each MPI's edge
  distribution and unlabelled topology but disrupt cross-SPI edge alignment.
  Use as a pooling-stage sensitivity, not a physically realizable raw-data null.
- Shared arbitrary edge permutation across all SPIs: preserve z and marginals
  while potentially changing incidence. This exposes z's topology blind spot.
- Independent circular time shifts per channel, on suitable stationary data:
  preserve each channel's samples and discrete Fourier magnitudes while changing
  temporal alignment. They do not guarantee destruction of every lagged dependence,
  especially in periodic systems. Recompute and measure what changed.
- Matched-marginal/spectrum simulation controls: verify the matching, rather than
  asserting that a nonlinear generator or surrogate has identical nuisance features.
- Compare raw embedding r, z and r+z using matched training/tuning. Improvement
  establishes usable complementarity under the chosen decoder and data budget,
  not unique information absent from the raw signal.

Primary outcomes: paired held-out score differences and area under a prespecified
learning curve versus log N. Secondary outcomes: retrieval with independent
relevance labels, dimension-performance curves, calibration, worst-shift loss,
view stability and M/T/validity leakage. No contrastive objective or geometric
distance can itself establish scientific meaning of neighbours.

## Compute and decision gates

### Executed Stage A — 2026-09-07

Protocol: `configs/analysis/representation-stage-a-260907.yaml`. Train on
M=16,T=1000, instances 0–9; sample 2/4/8 labelled recordings per each of 14 classes,
with five nested-subset seeds. Two-fold C selection and all preprocessing fit
inside the label budget. Evaluate previously inspected historical instances 10–29
over nine cells; this is exploratory. The primary joint shift has four cells,
1,120 rows and 280 class/instance groups for conditional paired resampling.

`scripts/build_representation_screen_bank.py` binds source-bank hashes, downloads
via a generated rsync manifest, checks every MPI/meta hash, and constructs unified
z plus m/g/validity. The local bank has 2,660 rows (140 training-pool, 2,520
evaluation), all 289 p90 SPIs, normalization false, computation 3.0.0.r7. Artifact:
`data/representation_stage_a_260907/proof-unified-controls.npz`, SHA-256
`5e928d75...1c4b6a5`. The transferred 2.22-GB source set is a local mirror; no
pyspi estimator was rerun. Original banks and results remain intact.

The original standard-m/g, centre-z screen found large m+g failures under shift.
Diagnosis at n=2/seed11: `lmfit_Lasso::reciprocity` had training SD 1.96e-6;
after source standardization its median absolute joint-shift coordinate was 554.
Some other finite raw summaries also had extreme out-of-source values. These
failures must not be advertised as evidence that graph structure is harmful.
`representation-stage-a-clipped-260907.yaml` specifies a post-initial-screen
sensitivity: clip every feature at training mean ±5 training SD, then balance
blocks. No SPI was selectively removed and no clipping-threshold search occurred.

A further post-screen control reused the repository's existing 82-feature pooled
raw statistics (single-channel temporal/marginal plus pairwise correlation/lag-one/
correlation-spectrum summaries), with identical splits, fitting and clipping.
Config: `representation-stage-a-raw-control-260907.yaml`; raw source hashes are
bound in `data/representation_stage_a_260907/raw-controls.json`.

| Joint-shift balanced accuracy | n=2/class | n=4/class | n=8/class |
|---|---:|---:|---:|
| p90 marginal m, clipped | .7914 | .8318 | .8546 |
| unified z, clipped | .8905 | .9123 | .9168 |
| m+z, clipped | .8573 | .8734 | .8768 |
| m+g, clipped | .8023 | .8411 | .8411 |
| m+g+z, clipped | .8536 | .8736 | .8729 |
| raw pooled u, clipped | .7625 | .8257 | .8568 |
| u+z, clipped | .8479 | .8548 | .9088 |
| validity-only, clipped | .4768 | .4995 | .5359 |

At n=8, z−m is .0621 (conditional paired 95% interval .0509–.0732); z−u
is .0600 (.0504–.0695). Normalized learning-curve area versus log n favours z
over m by .0806 (.0721–.0893), and over u by .0903 (.0819–.0987). These
unadjusted descriptive intervals resample evaluation groups, conditional on the
fitted models; the five overlapping training subsets are not five new training
populations. Seed ranges and all cell/class errors are retained in results.json.

Interpretation: cross-SPI values warrant further investigation in this screen;
concatenation does not beat z alone. Gains over m are concentrated in several VAR
and CML classes, and are not universal (brownian-defect is worse). Validity masks
are themselves informative, so this does not isolate a purely value-based
interaction mechanism. Nor is z always less degraded by shift: at n=8, its
same-cell-to-joint loss is .0618 versus m's .0582. Higher shifted accuracy is
not equivalent to a smaller transfer penalty. No neural, real-state, new-generator
or new-confirmation superiority is established.

Reports and plots: `results/representation_stage_a_260907/{screen,clipped-sensitivity,raw-control-sensitivity}/report.md`.
Each directory also contains predictions, split manifest, full scores, source/code
hashes and PNG/SVG curves. Rebuild reports with
`python -m scripts.report_representation_screen PATH/results.json`.
There were zero convergence warnings; the three fits took 21.5/25.4/13.4 seconds
locally with two BLAS threads, excluding extraction/transfer. All 26 relevant
tests passed, including end-to-end external raw-control alignment and train-only
clipping. The statistical learner/prescribed PCA cap has not been exhaustively tuned.

Stage B: `configs/analysis/representation-stage-b-proposal-260907.yaml` now fixes
N_full=32, a common input endpoint, future full-population coherence target,
M={8,16,32}, T={500,1000}, 96 training and 96 evaluation masters, label budgets
16/32/64 and explicit ridge/small raw-encoder tuning. All 672 raw views now exist;
the two-record p90 scout is submitted, with wider extraction gated on its results. Raw-only checks on 16 diagnostic trajectories (two burn times,
four couplings, two seeds) validated nested view construction and finite signals;
future target range was .2747–.8849, and the largest paired burn-window mean-target
difference was .0127. This small panel does not establish stationarity or predictive
utility. Evidence: `stage-b-raw-view-check-final.json` in the results root.

Existing MPI extraction is cheap relative to pyspi and needs no new cluster farm.
The 1,053 Zenodo MPI archives are mirrored locally; reuse them for controls and
geometry diagnostics. They are a breadth audit, not a ready-made new independent
supervised benchmark. Reconstruct unified z for the proof from its MPIs before
comparing it with new p90 controls; historical symmetric scores are distinct.

Follow `AGENTS_CLUSTER_CONTEXT.md` for new Gadi work: two-dataset smoke,
representative one-node timings, homogeneous dataset farms with one pyspi worker
and BLAS thread per dataset. Multiple summaries reuse each costly MPI computation.
Report aggregate CPU/SU cost, storage, peak memory and wall-clock time separately;
massive parallelism reduces elapsed time without erasing resource cost. GPU neural
training needs its own measured allocation; CPU farm parallelism is not a GPU plan.

Completed local extraction on 2026-09-06:
`results/representation_p90_260906/zenodo-mpi-baselines.{npz,json}`. All 1,053
records have 289 SPIs; p90 hash `bc4bafa1...`, computation `3.0.0.r7`, z-scoring,
seed 1729. Marginal/graph matrices are 1053-by-2023 and 1053-by-2601. Extraction
and rank audit took 38.3 seconds locally with one BLAS thread; NPZ is 27.7 MB.
No record has a complete 289-SPI valid Pearson block; valid SPI counts range
197–284 (median 265). There are 148 SPIs valid throughout this corpus (a
descriptive intersection, not a frozen train-selected panel). Every record's
valid principal block is numerically singular at relative eigenvalue tolerance
1e-8; numerical ranks min/median/max are 19/89/255. Minimum eigenvalue across
records is -4.85e-14, consistent with numerical roundoff. These are geometry and
coverage findings, not a performance comparison. Five scientific-contract tests
plus fourteen existing SPI-contract tests passed at implementation.
All 1,053 dataset IDs, ordered SPI names and MPI/metadata hashes match the existing
unified-v3 z bank. Nine evenly spaced recomputations differ by at most 5.76e-17
(BLAS roundoff near zero); see `alignment-verification.json` alongside the outputs.

Proceed to learned geometry if cross-SPI alignment adds value, or if a specific
observation-instability failure offers a justified learning target. If m matches z,
the statistical library may carry most of the benefit. If m+z wins, retain both and
make a complementarity claim. If g adds value, revisit topology loss. A fixed new
distance with a small accuracy gain is usually a sensitivity section; a replicated
observation-stable representation with meaningful transfer could be a larger paper.

## Attribution boundary for a positive result

A win over seven per-SPI summaries would establish value beyond those summaries,
not identify cross-statistic interaction organization by itself. This distinction
is material because the catalogue includes algebraically related statistics.
For an edge statistic V and its square, the SPI-pair feature is

\[
\operatorname{corr}(V,V^2)=\frac{\mu_3-\mu_1\mu_2}
{\sqrt{(\mu_2-\mu_1^2)(\mu_4-\mu_2^2)}},\qquad
\mu_j=\mathbb E[V^j].
\]

This is a first-principles identity: this particular z coordinate measures
higher marginal moments of one underlying edge statistic. It need not indicate
an interaction between distinct dependence mechanisms. Even if those moments
are added to m, a linear head need not cheaply recover their nonlinear ratio.
Consequently an observed gain can reflect a useful precomputed nonlinear basis,
beyond limited marginal summaries or a restricted readout, as well as genuinely
informative cross-statistic alignment. All are possible utility claims, but they
are different explanations.

Before making the stronger organization claim, test richer marginal descriptors
(including skewness and kurtosis), a comparably tuned nonlinear marginal readout,
and a catalogue sensitivity removing exact algebraic duplicates. Independent
SPI-wise edge permutations preserve marginal distributions and destroy alignment;
a shared permutation preserves z while destroying general graph incidence.
Neither null alone proves that named SPI mechanisms provide independent physical
information. These are targeted follow-ups after a useful pilot effect, not a
reason to expand the current grid indiscriminately.

## Primary literature

- Cliff et al., 2023: https://arxiv.org/html/2201.11941v2 (existing MPI correlations).
- Lin et al., 2015: https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html (second-order pooling).
- Peach et al., 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC8085611/ (HCGA).
- Brüsch et al., 2023: https://arxiv.org/abs/2307.09614 (variable-channel learning).
- Thanwerdas and Pennec: https://arxiv.org/abs/2103.04621 (full-rank correlation geometry).
- Bhatia et al.: https://arxiv.org/abs/1712.01504 (Bures–Wasserstein matrices).
- Huang and Van Gool, AAAI 2017: https://ojs.aaai.org/index.php/AAAI/article/view/10866 (SPDNet).
- Wiskott and Sejnowski, 2002: https://www.ini.rub.de/PEOPLE/wiskott/Projects/LearningInvariances.html (slow feature learning).
- Bardes et al., ICLR 2022: https://arxiv.org/abs/2105.04906 (VICReg).
- Mardt et al., 2018: https://www.nature.com/articles/s41467-017-02388-1 (VAMPnets; relevant only with genuine time-lagged trajectories, not independently generated parameter sweeps).

## Scoped contribution and publication criteria — clarified 2026-09-07

The aim is to discover and explain a useful regime of inductive bias, not show
universal superiority. Purposeful exploratory search is legitimate; the separation
between exploration and fresh verification determines how strong the claim can be.
A correctly specified VAR reference winning its native task is informative, but
not grounds to dismiss a representation intended for unknown model families.

A strong scoped contribution could establish an important problem, a non-obvious
reason cross-statistic relationships help, a practically meaningful gain under
honest label/pretraining/compute accounting, and replication sufficient for that
specific claim. A large architecture, formal theorem, real-world dataset and broad
benchmark are not all mandatory components of every strong paper. Clear writing
cannot substitute for significance or novelty, but can make a focused contribution
legible. There is no fixed percentage gain or dataset count that guarantees a venue.

[ICLR 2026's reviewer guide](https://iclr.cc/Conferences/2026/ReviewerGuide) explicitly
allows impactful new knowledge without state-of-the-art results. [NeurIPS 2026](https://neurips.cc/Conferences/2026/ReviewerGuidelines)
recognizes insights, efficiency and well-reasoned combinations of existing techniques;
its use-inspired guidance emphasizes important use cases and non-ML alternatives.
[JMLR](https://www.jmlr.org/author-info.html) asks for advances in ML understanding
of broader interest and excludes pure applications. [Nature Computational Science](https://www.nature.com/natcomputsci/natcomputsci/natcomputsci/about/aims)
emphasizes computational advances and scientific insight or challenging applications.
These are different contribution profiles, not a ladder of universal benchmark scores.
No acceptance prediction follows from this guidance.

The PCA objection is partly addressed, not dismissed: PCA32 was already present;
training-only cap selection now failed to improve z on Stage B. Population
intrinsic dimension is not established by p>>N, and PCA maximizes variance rather
than target information. Supervised compression or independent-source unsupervised
pretraining remain distinct, untested possibilities. The next proposal is
[one local-sensitivity property](interaction-share-pilot.md), not another unrestricted
system catalogue. Its prototype changes no existing generator functions and has
raw target and partial-observation reference evidence so far. The initial
fixed-sum memory shortcut was diagnosed and corrected before z evaluation;
this establishes a usable probe, not a representation advantage.
