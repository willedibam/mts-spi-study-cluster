# Premise, framing and the character–strength direction

Distilled 2026-10-06 from the user's unrefined notes; this is the user's intended framing, not a list of established results. Evidence boundaries are stated where existing runs constrain a claim. The [manuscript blueprint](../spi-spi-manuscript-blueprint.md) (11 September) leads with comparability across recording sizes; the user's position is that comparability is a side benefit and the framing below is the crux. That difference is unresolved in the blueprint.

## Background in three steps

Multivariate time-series analysis, in neuroscience especially, moved from regional activation under stimuli, to the strength of connectivity between regions (functional connectivity), to broader statistics (information-theoretic, spectral, causal) and network-science summaries (degree, modularity, centrality).

## Problems

1. **A statistic is an inductive bias.** Covariance presumes linear interaction and cannot register nonlinear dependence; contemporaneous mutual information presumes zero lag and cannot register lagged dependence. Once chosen, the view of the system cannot change, which limits analysis exactly when the organization of the system — the nature, type or character of its interactions — changes.
2. **Most methods refine strength.** They improve the resolution of coupling strength between node pairs under an assumption fixed in advance. They can say how strongly a pair interacts under that assumption, never what kind of interaction it is.
3. **The alternatives need prior knowledge.** Using several statistics (as pyspi enables; Cliff et al., [arXiv:2201.11941](https://arxiv.org/abs/2201.11941)) relaxes one prior but the selection is itself made under prior knowledge, and a bespoke statistic that wins on one task trails on another.

**Aim.** Outside deliberately constructed contrasts between bespoke statistics, there is no general-purpose method for characterizing the mechanism of dependence within a system and how it evolves. SPI–SPI is proposed as a first one.

## What SPI–SPI is claimed to do

It asks how notions of dependence themselves covary across the channel pairs of one recording: z_ab = corr(A^(a)_ij, A^(b)_ij). This defines a geometry among scientific notions of dependence and supplies common reference points across otherwise incomparable systems. Each pair is an interpretable question about the data:

- {Pearson, Spearman}: coupled for linear (Gaussian) data, decoupled under monotone nonlinear or heavy-tailed data — a crude signature of nonlinearity.
- {Euclidean distance, cross-Euclidean distance}: coupled for contemporaneous dynamics, decoupled when dependence is lagged — a signature of time lag.

Case studies: [`r_rho_mi_260622.ipynb`](../../notebooks/cases/r_rho_mi_260622.ipynb), [`pdist-euclid_dtw.ipynb`](../../notebooks/cases/pdist-euclid_dtw.ipynb) (rough, not pedagogical). Aggregating over the combinatorial space of pairs gives a signature of dependence character. Intended use: given two groups of recordings (young and old, patient and control, or an engineering or sensor analogue), say how the nature of dependence differs — more nonlinear, more variably lagged, more variable in frequency — rather than only how strong it is. Representing recordings of different M and T in one space follows from the construction; it is a benefit, not the motivation.

Acknowledged crudeness: Pearson correlation is a clumsy comparator of two statistics, and aggregation over all pairs is blunt.

## Evidence boundaries to keep attached

- z is not strength-blind; strength "bleeds" in. It is invariant to a positive affine change of each SPI's edge profile and to nothing else.
- Per-SPI means are not strength-only. With the full catalogue, supervised mean readouts recover character targets, and the unsupervised mean-SPI PC1 matched z-PC1 on homogeneous recordings ([locking result](../research/order-parameter-benchmarks/cross-frequency-locking-261006.md); [ruled-out record](cross-mt-transfer-ruled-out.md)). The supported contrast is unsupervised access under per-recording nuisances that move every dependence estimate together, not absence of information. On the proof classes (2026-10-06, [record](cross-mt-transfer.md#strength-as-a-per-recording-nuisance-on-proof-classes-2026-10-06)) equalising strength left the mean vector as clustered as before; strength varying between recordings degraded its geometry, clearly only within CML, and never its class information.
- "No general-purpose method exists" and "potentially novel" are literature claims not yet verified by a dedicated search.

## Future direction: a character axis alongside strength (not started)

Existing methods sit at positions along a strength axis fixed by their assumptions. SPI–SPI sits mostly along character and gives up strength resolution. A two-dimensional character–strength representation could cover both.

The proposal is to build named axes on [0, 1] that are directly interpretable — linear to nonlinear, fixed to variable lag, frequency variability — by fitting, per channel pair, a restricted model and a flexible one and scoring their agreement (for linearity: a Gaussian model against a flexible or learned one), then aggregating over pairs. Each axis would replace the correlation aggregator in z_ab with a model-comparison score chosen to expose one quality.

Open design questions (assistant's assessment, 2026-10-06):

- A model-agreement score must be calibrated against estimator variance, or short or noisy recordings read as "linear" by default. Established precedents exist per axis and should anchor it: surrogate-data tests and Gaussian-versus-nonparametric MI gaps for nonlinearity, lagged-versus-instantaneous decompositions for lag.
- Strength and character are not orthogonal: at weak coupling every character score is undefined in practice. The representation needs an explicit rule for pairs below a strength floor.
- Candidate further axes: instantaneous to lagged; symmetric to directed; within-frequency to cross-frequency (the locking benchmark is a ready test bed); amplitude- to phase-mediated; stationary to intermittent; pairwise to higher-order.
- Validation should reuse generators with a known character parameter before any real-data claim.
