# Prospective NeuroTycho state-transfer pilot

Declared after the four-date source scout and its precision repair, before any
SPI classification or propofol waveform evaluation. This is a bounded application
pilot, not a population-level clinical validation or a state-of-the-art neural
benchmark. The [source protocol](neurotycho-source-pilot.md) fixes labels, geometry,
preprocessing, quality rules and known limitations. Use the corrected float64
observations exclusively for SPI extraction.

## Question

Does cross-statistic agreement help transfer awake-eyes-closed versus sustained
anaesthetized discrimination across animal and anaesthetic/acquisition-period
changes, beyond spectral features and individual-SPI summaries? Pearson z is one
candidate representation; learned SPI pooling tests whether a richer aggregation
of the same statistical coordinates is useful. A positive outcome is not assumed.

## Split and evaluation

- Sources: all11KTMDdates passing the frozen quality gate,16windows/state/date.
  Train only on full M16/T2000 views. Never divide windows from one animal between
  inner training and validation. For each prospective target animal, exclude all
  its KTMD dates: Chibi leaves9source dates; George leaves8, each from3animals.
- Targets: both PFdates for Chibi and both PFdates for George. Same16windows/state
  selection and quality rules. No target-label, target-feature-distribution or
  target-batch calibration enters fitting, preprocessing or model selection.
- Primary: balanced accuracy at threshold.5 on M16/T2000, average dates within
  animal then average the two animals. Report every date and animal, AUROC and
  Brier alongside it. The M8/final4s view is secondary joint sensor/analysis-window
  reduction; it inherits28sfiltercontext and is not a strict recording-access test.
- Three neural initialization seeds11/23/47; report individual runs and mean
  performance, not an undeclared probability ensemble. Statistical fits are
  deterministic apart from fixed PCAseed1729. No window bootstrap presented as
  population uncertainty: only2targetanimals/4dates exist. No low-subject learning
  curve is supported by this cohort. Agent and calendar period remain confounded.

## Comparators and what each tests

| Representation | Readout | Purpose |
|---|---|---|
| Spectrum:70absolute+relative or40relative-only features | Logistic | Strong domain baseline; choose view inside source CV |
| m:23summaries per SPI | PCA/ridge | Information in each dependence measure's distribution |
| m+g: m plus9graph summaries per SPI | PCA/ridge | Additional node-incidence/graph organization |
| Pearson z | PCA/ridge and PLS | Unsupervised versus supervised dimension reduction |
| m+z | PCA/ridge | Complementarity of magnitudes/distributions and agreement |
| SPI validity mask | PCA/ridge | Diagnostic for failure-pattern shortcuts |
| Normalized ordered SPI edges + validity | Learned SPI pooling | Learned aggregation of the fixed statistical coordinates |
| Raw standardized ECoG | Aligned temporal/channel encoder | Learning from waveforms before invariant pooling |
| More source windows + source augmentation | Same raw encoder | Stronger control using the same labelled dates/animals |

Use p90's289fixed identities and both directed off-diagonal entries. m is the
existing23-coordinate rich marginal block; g is the existing9-coordinate graph
block in `mpi_representation_baselines.py`. Do not identify these graph summaries
with measured anatomical connectivity or causal coupling. No extra catalogue,
new generator, architecture grid, hand-selected SPI panel or fusion grid here.

## Source-only fitting contract

Inner leave-one-source-animal-out, three folds. Select by mean balanced Brier
(equal state contribution within each validation animal), with deterministic
ties favouring fewer components/stronger regularization. This prospective
probability-score criterion differs from the exploratory scout's BA selection;
it is common to the transfer comparisons and is fixed before PF outcomes.

For SPI feature blocks, reuse `representation_screen.py` training-only transforms:
at least95percent finite values, median imputation, variance threshold1e−8,
center-only z, standardized other blocks, clipping at5training SD, each block
balanced to unit total training variance. PCA caps2/8/32, ridge alpha.01/.1/1/10/100;
PLS1/2/4components with existing numerical-rank/zero-covariance protections. Reuse
the fixed transform across ridge candidates. No PCA or imputation fits on PF.
Clip linear regression scores to[0,1] for Brier/reporting. Spectrum uses the two
declared views and C.01/.1/1/10, with training-only imputation/scaling. Primary
windows give equal date/state counts; retain and report any QC imbalance.

Learned SPI pooling uses the existing578→32→32edge MLP, mean/SD edge pooling and
64→32→1head. Raw input is B×T×M, with shared temporal convolutions32channels,
kernel31/stride4/padding15, then64channels,kernel15/stride4/padding7. Two aligned
channel-attention blocks have width64,4heads,feedforward128; post-attention temporal
convolution width64,kernel9/stride1/padding4. Pool mean/max over channels/time and
use128→32→1head. GELU and dropout.1 for the raw encoder. The receptive field is
adapted to250Hz ECoG, rather than silently copying100Hz synthetic settings.

Both neural models: sigmoid output, binary cross-entropy training, AdamW,
learning rates1e−4/1e−3, weight decay1e−4/.01, batch16, max400epochs, min30epochs,
patience30 on source-validation balanced Brier. Final epoch count is the median
best epoch of the selected setting across the three source folds. Save histories,
selected settings, hashes, hardware and checkpoints; replay predictions separately.
No pretraining. This is a comparison with these tested encoders, not all neural
methods. BIOT's magnitude-STFT and channel-mapping limitations are audited in the
source protocol; no pretrained human-EEG weights are being presented as an
unmodified macaque ECoG comparator.

## Extra-window raw control and stopping

State intervals already carry labels: extra windows do not require extra labelled
animals/dates. Therefore a favourable matched-window result must also face one
stronger raw control. Use **all** nonoverlapping8s source windows starting30s after
each labelled start and ending before the30s end margin; same filtering/context
and quality criteria. Each source date/state has equal total loss weight so long
intervals do not dominate. During training, independently choose M8or16 and final
4or8s with probability.5each; choose8channels uniformly when reduced. The source
animal grouping and hyperparameter/epoch selection stay unchanged. This control
changes available windows and augmentation together; it tests a stronger competitor,
not an isolated attribution. Report its extra extraction/training cost and exposure.

Fit and inspect source training/validation behaviour before opening PF. If an
implementation/optimization defect is demonstrated, fix it with a versioned source
audit; do not count a win over a failed neural learner as a representation gain.
Do not alter models or thresholds after seeing PF to manufacture superiority.
After the prespecified comparisons and verification, assess whether evidence
supports useful agreement, complementarity, a richer pooling benefit, or no added
utility. More drugs/datasets are not an automatic response to disappointment.
