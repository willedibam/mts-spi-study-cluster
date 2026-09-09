# Co-organization pilot: verified findings

The fresh pilot supports a **scoped, useful representation result**: relationships
among p90 dependence measures predict whether phase and amplitude-envelope groups
coincide, and retain useful classification performance after sensor and duration
reduction. This is stronger evidence than matching a near-chance neural baseline.
It is not clinical state prediction, direct causal-network recovery, information
inaccessible from raw data, or superiority over neural learning in general.

The [protocol](oscillatory-coorganization-pilot.md) was frozen before320independent
recordings were generated:120source training-pool recordings in three disjoint
40label cohorts, plus200independent evaluation recordings. N32 is fixed; models
train onM16/T1000 and evaluate there and on nestedM8/T500. Conditions have the same
single-channel process laws and each coupling mode's latent group sizes; their
cross-mode correspondence changes. This does not guarantee identical realized
spectra or all individual-SPI distributions. No pretraining or augmentation.

## Main result

Balanced accuracy after joint M/T reduction, averaged over three training cohorts:

| Representation/readout | 10labels | 20labels | 40labels |
|---|---:|---:|---:|
| z + PLS | **88.8%** | **96.0%** | **98.0%** |
| z + PCA/ridge | 83.0% | 96.0% | 98.5% |
| Rich individual-SPI summaries | 64.3% | 74.0% | 80.8% |
| Normalized individual-SPI shapes | 65.3% | 70.0% | 73.7% |
| Rich summaries + z | 78.7% | 91.3% | 97.2% |
| Normalized shapes + z | 78.2% | 89.7% | 95.7% |
| Direct phase/envelope agreement | 96.3% | 96.3% | 98.0% |
| Raw moment/covariance/window summaries | 60.5% | 78.2% | 78.0% |
| Aligned raw neural encoder | 49.5% | 50.3% | 50.2% |
| Temporal-pair raw neural encoder | 48.8% | 48.5% | 47.0% |
| Constant prediction | 50.0% | 50.0% | 50.0% |

At40labels, zPLS−richm is **+17.17percentage points**, conditional pointwise95%CI
[13.50,21.17]. z−normalizedshapes is+24.33[20.67,28.00]. Adding z to richm gives
+16.33[12.67,20.50]; adding it to normalizedshapes gives+22.00[18.50,25.67].
Standalonez beats both distribution baselines in all nine budget/cohort cases.
At10labels, cohort zBA ranges80.5–95.0%, so small-budget variability remains.

This is not only a threshold effect: at40labels reduced zPLS AUROC=.9986 and
Brier=.0431, versus richm .9215/.1341 and normalizedshapes .9513/.1683.
At originalM/T, z reaches100%BA and richm98.8% at40labels; the important separation
therefore concerns scarce labels and transfer, not a large in-domain ceiling gap.
PCA/ridge remains competitive; no gain from a new nonlinear learned geometry is
demonstrated. Selected PCA caps rise from4 at10labels to16–32 at larger budgets,
so these results do not establish that one principal component suffices.

The direct phase/envelope control is a task-informed specialist operating on the
same observed recordings, with groups inferred implicitly through measured
coupling, never supplied latent assignments. At40labels it matches z's mean BA;
z−direct CI[-2.17,2.33]percentage points is not formal equivalence. It is better
on average at10labels and has lower Brier score at40. This limits any claim of
unique information or superior optimality; it does not erase the generic
catalogue's useful performance and advantage over the tested marginal summaries.

![Verified learning curves](../results/oscillatory_coorganization_pilot_260909/learning-curves.png)

Intervals resample independent test masters within class, conditional on the
fitted models; they are pointwise and do not account for every exploratory
comparison. The three source cohorts provide a separate view of training
variability, not a population-wide confidence guarantee. Both observation views
of a test master are dependent. The binary conditions are clearly separated
idealized organizations; real-world prevalence and intermediate organizations
have not been tested.

## Bounded sensitivities and mechanism check

These were declared after the pilot result and must remain supplementary:

- Frozen random aligned encoder + PCA/PLS heads also stays near chance (40label
  reducedBA51.0/50.7%). Independent feature extractions agree exactly and all18
  readouts replay. Thus no useful initial representation was exposed by these
  heads. The two end-to-end models and random controls are small chosen
  architectures, not published biosignal foundation-model benchmarks.
- A nonlinear PCA/RBF kernel-ridge head does not close the gap: reduced40label
  richm72.0%, normalizedshapes66.7%, z91.5%. This fixed training-scale bandwidth
  control is not an exhaustive search over nonlinear marginal learners.
- Removing all1713z coordinates involving the six power-envelope-correlation
  SPIs reduces40label BA from98.0% to89.0%. A fixed36coordinate PEC×PLV panel
  obtains92.0/94.0/94.5% at10/20/40labels. Explicit envelope measures contribute,
  but are not necessary for all predictive signal. The small specialist panel
  is strong at10labels but does not reproduce fullz's40label performance.
- Three independent per-SPI dyad-shuffle controls reduce40label joint-shift BA
  to49.5%,52.3%,50.0% (seeds503/509/521). All520records preserve each SPI's
  edge multiset, reciprocity and z validity; a shared permutation changes z by
  at most4.06e-16. All27saved fits pass prediction-hash and split audits; null503 bank hash and
  all nine local fit/CV replays pass (prediction discrepancy<=1.28e-15).
  This supports cross-SPI correspondence as necessary for the tested readout's
  performance under this intervention. Full results: `mechanism/alignment-report/`.
  Such collections need not be realizable raw processes; interpret as a
  representation-level intervention, not a physiological manipulation.

## Verification and reproducibility

All520MPI outputs pass provenance audits. BankSHA
`5e898b49c519594802f681b0272278807dd155de3e520244519e7e87d00e1ef2`;
240–282SPIs valid per record. Eight local raw/MPI-to-feature replays pass.
All117statistical/raw fits preserve their selected hyperparameters and predictions
to<=1.80e-14, CV scores<=5.67e-15. Eighteen neural checkpoints/splits replay onCPU
within2.38e-7; no selected600epochcap. One inspected source fold worsened in both
validationMAE andMSE as training fit improved; no evidence supports a stopping
metric repair there.

A validity-control fold had rank2 but exactly zero feature–target covariance.
Local NIPALS divided by zero whereas Gadi roundoff produced a finite fit. The
shared fitter now returns the source mean when covariance is numerically zero;
the regression test and full replays pass. This repair leaves all reported
predictions/selections unchanged to numerical precision. Original Gadi outputs
are retained beside corrected local readouts.

Gadi production178482803/804 completed in7:11/1:30; analysis178482814 in3:35,
allExit0. Summed per-record p90 work35.06corehours, made affordable by dataset-level
parallelism; this is not a claim that extraction is cheaper than the specialist.
Results`results/oscillatory_coorganization_pilot_260909/verified-report/` are
authoritative. `report_oscillatory_pilot.py` and `plot_oscillatory_pilot.py`
reproduce tables/figure; verification JSONs and historical outputs sit alongside.

## Current assessment

This is a well-defined success for the proposed inductive bias: represent the
organization of multiple dependence modes, rather than their magnitudes alone.
The covariance-modulation failure supplies a useful contrasting limitation.
Together they support a scientifically intelligible account of when this
representation helps; they do not require pretending it should win everywhere.

The strongest present contribution is a method/representation result in a
controlled oscillator setting. Exact clinical usefulness, transfer across
distinct generator families, robustness to sensor mixing/geometry, and stronger
domain-trained or augmented neural comparators remain open. A narrowly motivated
simulation study can be valuable, but current evidence does not independently
establish a broad high-impact ML claim. No additional same-generator scale-up is
justified; finish the separate covariance curve.


## Learned aggregation follow-up, completed2026-09-09

The declared [pooling comparison](oscillatory-pooling-transfer.md) uses the same
record-standardized SPI edge vectors that can reproduce z, plus validity masks.
A shared per-edge MLP followed by mean/SD pooling learns a64dimensional record
representation;21,697parameters, same source cohorts/budgets/CV/grid. This model
can access standardized distribution information beyond correlations; it has the
same catalogue inputs, not exactly the same compressed information as z. GadiCPU
training was declared before results, rather than the earlier raw encoders' MPS.

| Reduced-observation balanced accuracy | 10labels | 20labels | 40labels |
|---|---:|---:|---:|
| zPLS | 88.8% | 96.0% | 98.0% |
| Learned SPI edge pooling | 65.8% | 96.5% | 98.8% |

Learned pooling reaches100% original-observation BA at20/40labels. Reduced zPLS
minus learned pooling is+23.0percentage points at10labels, conditional pointwise
95%CI[20.0,25.67]; at20/40labels -.50[-2.17,1.00]/-.83[-2.00,.17]. These do not
establish formal equivalence at larger budgets or a precise label-efficiency ratio.
At10labels, pooling cohort reducedBA=.520/.605/.850 and AUROC=.7231/.8935/.9440.
The large training-cohort variability is not captured by a bootstrap conditional
on those fitted models. Two selected folds, both10label seed23, reach600epochs;
last100epoch rawvalidationMAE improves .00083/.00567. Retain the declared ceiling
and disclose this optimization limit; no automatic doubling or claim that more
training could not improve the10label result.

All9checkpoints/splits pass;72local CPU prediction replays differ by<=4.18e-7.
Edge bank520record Gram recovery<=1.083e-7; eight MPI replays are bitexact.
Gadi job178541908 Exit0(1:42,6cores,3.73GiBpeak); no duplicate local fits.
Results `pooling-report/`, checkpoints `learned-pooling/`, verification
`learned-pooling-verification.json` under the original pilot result root. Original
`verified-report/` remains unchanged; supplemental comparison is declared after
its results. Faster-regime predictions remain pending.

The useful conclusion is explicit statistical relationships expose the target
under the smallest tested label budget, while learning an aggregation of the
same catalogue inputs succeeds with20–40labels. This strengthens evidence for
catalogue-based relational information, and limits a claim of uniquely superior
Pearson compression or neural inaccessibility. The64dimensional learned embedding
is a possible extension, not a reason to change the frozen transfer test.

## PCA readout clarification requested by the user

PCA/ridge reducedBA10/20/40=.830/.960/.985 versusPLS.8883/.960/.980 and learned
pooling.6583/.965/.9883. Original-observation PCA=.9567/.9983/1.0. At10labels,
PCA minus learned pooling is+17.17percentage points, conditional pointwise95%CI
[13.33,20.83]; PCA minusPLS=-5.83[-8.34,-3.33]. At20/40labels no substantial
separation is resolved. These are supplementary contrasts of existing predictions,
not new fits; `pca-pooling-report/` retains the results and input hashes.

PCA axes ignore targets, but dimension-cap/ridge selection uses source labels;
this is not an entirely unsupervised model-selection pipeline. PCA is fitted on
the available training records only, without extra unlabeled exposure. Requested
caps at10/20/40labels were4/16/(16or32), respectively. The fitter truncates each
cap to min(training_rows-1,available_features), so inner-fold effective dimensions
can be smaller than final-refit dimensions. These cap choices do not identify
an exact intrinsic dimension or directly validate16versus32components in a fold
that cannot fit that many. This was the declared adaptive-cap algorithm; no test
leakage follows, and the completed results remain intact. A future fixed-dimension
claim would require candidates feasible in every inner fold or a separate frozen
unlabeled basis. No additional PCA sweep is authorized by these results alone.

The user prefers considering PCA, without a concrete decision. Retain both
prespecified readouts through the frozen dynamics-transfer evaluation; do not
silently change the primary analysis according to which test result is better.
