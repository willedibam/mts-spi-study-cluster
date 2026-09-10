# Technical guide to the NeuroTycho representation and transfer models

*Implementation guide, 11 September 2026. This explains the existing experiment; it does not revise the [frozen protocol](neurotycho-transfer-pilot.md) or [configuration](../configs/analysis/neurotycho-transfer-260910.yaml). Results and job status belong in the study reports, not this guide.*

The central question is whether relationships among dependence measures provide useful state information that specified learners can extract and transfer efficiently. Every model predicts the same binary label: awake with eyes closed (0), or sustained anaesthetized (1).

The model families differ principally in **what is supplied to the learner**:

| Family | Supplied information | Learned part |
|---|---|---|
| Statistical descriptors | Fixed summaries of spectra or MPI matrices | Projection and a small prediction rule |
| Learned SPI pooling | The joint collection of standardized SPI values across links | Nonlinear link features, their aggregation and a prediction rule |
| Matched raw encoder | Standardized waveform windows | Temporal filters, channel interactions and a prediction rule |
| Enriched raw encoder | More windows from the same source recordings, with training augmentation | The same architecture as the matched raw encoder |

Learned pooling is also a neural network. “Raw neural” identifies its input, not a separate mathematical category of learning.

## 1. Data, notation and the actual unit of replication

Let $X\in\mathbb R^{T\times M}$ denote one time-series window, where $T$ is the number of time samples and $M$ the number of bipolar channels. A model outputs a score $p(X)\in[0,1]$; classification uses the fixed rule $p(X)\geq0.5$.

| Quantity | Meaning in this pilot |
|---|---|
| Source recordings | 11 dates from four macaques, recorded under the KTMD protocol |
| Matched source pool | 352 selected windows: 16 per state per date |
| Dense source pool | 3,556 windows, before excluding the evaluation animal |
| Main observation | $M=16,T=2000$: eight seconds at 250 Hz |
| Reduced observation | First eight bipolar channels, final 1,000 samples: four seconds |
| Target recordings | Four propofol dates across Chibi and George |
| Target pool | 128 windows, each represented at both observation sizes |
| Statistical catalogue | $K=289$ p90 SPIs |
| Ordered links | $E=M(M-1)$: 240 at $M=16$, 56 at $M=8$ |

A recording date contains many windows, and an animal contributes multiple dates. **352 windows are not 352 independent subjects.** Within-animal windows can share anatomy, instrumentation, temporal history and other nuisance factors. The split therefore groups by animal.

For the model evaluated on Chibi, all Chibi KTMD data are excluded from fitting: three source animals, nine dates and 288 matched windows remain. Excluding George leaves three animals, eight dates and 256 windows. Each final model has its own three-animal source dataset.

The target animals' KTMD recordings were examined during exploratory study development. They are excluded from their final fitted models; they were not untouched animals throughout the entire research process.

### What preprocessing does

Each bipolar channel is the difference between two selected physical electrodes. Sixteen disjoint bipolar channels require 32 physical electrodes. Pair selection uses electrode-map geometry rather than classification results; corresponding channel numbers do not establish anatomical homology between animals.

The processing chain is:

1. Select sustained labelled intervals, with 30-second boundary margins.
2. Read a 28-second segment: the central eight seconds and ten seconds of context on each side.
3. Form bipolar differences and remove a linear trend.
4. Apply a fourth-order 0.5–100 Hz Butterworth bandpass and a 50 Hz notch, using zero-phase filtering.
5. Resample from 1,000 to 250 Hz and retain the central eight seconds.
6. Construct the reduced view from the first eight channels and final four seconds.

Both views inherit the filtering context. This is an **analysis-window and sensor-coverage shift**, not a test in which preprocessing has access to only four or eight seconds of raw recording.

Before SPI extraction and raw-neural input, each observed channel is standardized within its window:

$$
\widetilde X_{tj}=\frac{X_{tj}-\overline X_j}{s_j}.
$$

For a reduced raw-neural observation, normalization is recalculated on that observed crop. This fixed, per-window operation is not target-distribution fitting. Absolute waveform amplitude is removed from these inputs. The conventional spectral baseline is deliberately allowed to retain absolute power from the filtered, pre-normalization signal.

Filtered observations remain float64 through SPI extraction. Neural tensors and the final $z$ coordinates use float32. Rounding the *input waveform* can create exact ties that affect some estimators; rounding an already-computed correlation is a different numerical operation.

Implementation: [windowing and preprocessing](../src/neurotycho_pilot.py).

## 2. From SPIs to the agreement representation $z$

An SPI produces an MPI matrix $A^{(k)}\in\mathbb R^{M\times M}$. Its off-diagonal entries describe pairwise statistical relationships. The interpretation of a directed entry depends on the SPI; directionality alone does not establish physical causation.

Write the same ordered off-diagonal positions of every matrix as a vector:

$$
a_k=\operatorname{vec}_{i\ne j}(A^{(k)})\in\mathbb R^E.
$$

Both $i\rightarrow j$ and $j\rightarrow i$ occur in this vector. The diagonal is excluded. The SPI–SPI feature is

$$
z_{kl}=
\frac{(a_k-\bar a_k\mathbf 1)^\top(a_l-\bar a_l\mathbf 1)}
{\|a_k-\bar a_k\mathbf 1\|_2\,
 \|a_l-\bar a_l\mathbf 1\|_2},\qquad k<l.
$$

Thus $z_{kl}$ asks: **across the links in this recording, do these two statistics assign relatively high and low values to the same links?**

It does not simply ask whether either statistic has large values. It is also not a temporal correlation between two channels: the observations being correlated at this second level are MPI entries.

There are

$$
\binom{289}{2}=41{,}616
$$

possible coordinates. SPI identity fixes coordinate meaning across recordings of different $M,T$. Shape compatibility does not imply that the coordinates have equal estimation precision, or are invariant to changing the observed channels.

### A small example of what agreement adds

Consider six illustrative ordered-link values, corresponding to the number of off-diagonal positions in a three-channel MPI:

$$
a=(0,0,0,1,1,1),\quad
b=(0,0,0,1,1,1),\quad
b'=(0,0,1,0,1,1).
$$

Vectors $b$ and $b'$ have identical distributions: every mean, scale and quantile is the same. However,

$$
\operatorname{corr}(a,b)=1,\qquad
\operatorname{corr}(a,b')=\tfrac13.
$$

Agreement distinguishes their different correspondence with $a$, even though summaries of each vector alone cannot. This is an illustration using MPI entries, not a claim that this construction is generated by particular SPIs or explains the biological dataset.

Conversely, replacing $b$ by $3b+5$ leaves its agreement with $a$ unchanged. If the target is encoded in that magnitude change, this coordinate cannot identify it.

### What is retained and what is lost

- A common permutation of links leaves $z$ unchanged. Channel relabelling is one special case, but arbitrary common link permutations also erase information about which links share a node.
- Independent permutations for different SPIs generally change $z$, because they destroy cross-statistic correspondence.
- Positive affine transformations of each SPI's link values leave Pearson agreement unchanged. A negative rescaling changes the signs of that SPI's agreement coordinates.
- Simultaneously transposing every MPI leaves $z$ unchanged. Both directions are represented, but this representation cannot uniquely identify a global reversal of direction.
- Symmetric SPIs repeat the same undirected value in both orientations. Those repetitions are not additional independent evidence.
- An all-constant or invalid SPI vector has undefined correlations. Invalid coordinates remain masked before source-fitted missing-value handling.

The numerical norm threshold for accepting a nonconstant vector is not a guarantee that its estimated relationships are reliable.

Implementation: [SPI–SPI contract](../src/spi_spi_contract.py).

## 3. The geometric interpretation—and its limits

Standardize every valid SPI column across links to population variance one, and collect the columns in $V\in\mathbb R^{E\times K_v}$, where $K_v$ is the number of valid SPIs. Then

$$
C=\frac1E V^\top V.
$$

This is the Pearson agreement matrix: a Gram matrix with unit diagonal. It is positive semidefinite, with

$$
\operatorname{rank}(C)\leq\min(K_v,E-1).
$$

The subtraction of one comes from centering the columns across links. The rank bounds are 239 for the full observation and 55 for the reduced observation, before accounting for further degeneracies.

Three distinctions matter:

1. **Positive semidefinite is not positive definite.** With many SPIs and relatively few links, singularity is expected. Methods requiring an invertible covariance matrix do not apply automatically.
2. **Within-record matrix geometry differs from across-record feature geometry.** PCA below acts on a matrix whose rows are recordings and columns are $z$ coordinates. The rank bound of each recording's $C$ does not impose the same rank bound on that across-record PCA matrix.
3. **Imputation can leave the Gram-matrix family.** Independently filling missing $z$ coordinates does not guarantee a valid positive-semidefinite agreement matrix. The current models treat these as feature vectors; they do not claim to preserve correlation-matrix geometry after imputation.

For two fully observed matrices with the same SPI identities and unit diagonals,

$$
\|C_1-C_2\|_F^2=2\|z_1-z_2\|_2^2.
$$

So ordinary Euclidean distance between their unscaled upper triangles corresponds directly to a Frobenius matrix distance. This provides geometric intuition, but does not by itself justify a more complicated geometry-aware learner.

## 4. Statistical baselines: what each descriptor contains

### Individual-SPI summaries $m$

For each SPI, the model receives 23 descriptors of its ordered-link values:

- mean and population standard deviation;
- skewness and Pearson kurtosis;
- 19 quantiles at 5%, 10%, …, 95%.

The total is $289\times23=6{,}647$ coordinates before filtering. This preserves magnitude, dispersion and distributional shape within each SPI, but does not retain correspondence between the link values of different SPIs.

Here “magnitude” means the magnitude of an SPI calculated on standardized time series. It does not restore the original waveform amplitude removed before SPI extraction.

Implementation: [rich marginal descriptors](../src/representation_attribution.py).

### Graph summaries $g$

Each MPI contributes nine additional descriptors:

1. Standard deviation of row means.
2. Standard deviation of column means.
3. The 10th and 90th percentiles of row means.
4. The 10th and 90th percentiles of column means.
5. Reciprocity: agreement between the off-diagonal entries of $A$ and $A^\top$.
6. Largest singular-value energy fraction.
7. Normalized effective rank of the singular-value energy distribution.

The percentile items each contain two coordinates, giving nine in total and $289\times9=2{,}601$ features.

For singular values $\sigma_i$, the energy fractions are

$$
p_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2},\qquad
r_{\mathrm{eff}}=\frac{\exp(-\sum_i p_i\log p_i)}{M}.
$$

These descriptors retain selected aspects of node incidence and matrix organization that link histograms discard. They are not a complete topology representation. “Singular-value spectrum” here concerns a matrix decomposition, not temporal frequency.

Implementation: [MPI graph summaries](../src/mpi_representation_baselines.py).

### Conventional spectral baseline

This starts from each channel's temporal power spectrum, estimated by Welch's method: 500-sample Hann segments with 250-sample overlap, at 250 Hz. The nominal bin spacing is 0.5 Hz. Eight- and four-second windows contain seven and three overlapping segments respectively.

Each channel contributes:

- six log absolute band powers;
- six relative band powers;
- normalized spectral entropy;
- frequency below which 95% of spectral power lies.

The bands are 0.5–4, 4–8, 8–13, 13–30, 30–45 and 55–100 Hz. Relative powers use total 0.5–100 Hz power as their denominator; the six retained fractions are not renormalized to sum to one.

Channel mean, SD, and 25th/50th/75th percentiles give $14\times5=70$ features. Dropping the six absolute-power features per pooling block leaves 40.

A logistic classifier learns

$$
p=\sigma(b+w^\top s),\qquad \sigma(u)=\frac1{1+e^{-u}},
$$

with regularization strength selected on source animals. The choice of the 70- or 40-feature view is also source-selected.

This baseline contains no cross-channel coherence or SPI–SPI agreement. Spectral SPIs in p90, their marginal summaries, and relationships involving those SPIs remain separate information sources in the comparison.

### The seven statistical configurations

| Configuration | Raw feature width | Learned projection/readout |
|---|---:|---|
| $m$ | 6,647 | PCA + ridge |
| $m+g$ | 9,248 | PCA + ridge |
| $z$ | 41,616 | PCA + ridge |
| $z$ | 41,616 | PLS regression |
| $m+z$ | 48,263 | PCA + ridge |
| SPI validity indicators | 289 | PCA + ridge |
| Conventional spectra | 70 or 40 | Logistic regression |

The validity model checks whether estimator success/failure patterns themselves predict state. A useful validity model would require careful interpretation of any model that also sees missingness.

## 5. What PCA, ridge and PLS learn

### Training-only preprocessing

Within each fitting fold, for each feature block:

1. Retain coordinates finite in at least 95% of training windows.
2. Fill missing values using the training median.
3. Remove coordinates whose training SD is at most $10^{-8}$.
4. Center each retained coordinate. Standardize $m,g$ and validity coordinates; $z$ is centered without coordinate-wise variance normalization.
5. Clip centered values at ±5 training SD.
6. Rescale each block so its total training variance after those operations is one.

Block balancing prevents a concatenated block from dominating solely because it has more coordinates or larger units. It is a specified modelling choice, not an assertion that the blocks are equally informative.

The resulting transformations are frozen and applied to validation and target windows. No target population statistics enter them.

### PCA + ridge

Let $F\in\mathbb R^{N\times P}$ be the preprocessed training-feature matrix. PCA finds orthonormal directions maximizing variation in $F$, independently of the labels. Keeping $d$ components gives scores $H\in\mathbb R^{N\times d}$.

The ridge head solves

$$
\min_{b,\beta}\sum_{n=1}^N
(y_n-b-h_n^\top\beta)^2+\alpha\|\beta\|_2^2.
$$

The intercept is not penalized. Scores are clipped to $[0,1]$ for probability-error reporting and thresholding.

We select $d\in\{2,8,32\}$ and $\alpha\in\{0.01,0.1,1,10,100\}$. PCA uses the fixed randomized solver seed 1729. Components are capped by the available training dimensions; the transform is reused across ridge strengths.

PCA reduces dimensionality and removes directions outside its retained subspace. Ridge additionally shrinks coefficients within that subspace. They address different aspects of estimation variance. PCA can discard a low-variance but highly predictive direction; neither operation guarantees denoising.

The resulting predictor is linear in PCA scores, and effectively linear in the transformed features before output clipping. The overall mapping from raw time series remains nonlinear because the SPI descriptors themselves are nonlinear.

### PLS

Partial least squares learns feature combinations related to the target, rather than selecting directions solely for large feature variance. For a single centered target, the first direction is guided by $F^\top y$; later components use deflated data to extract additional predictive variation.

This pilot selects 1, 2 or 4 components and uses PLS without an extra scaling step, because preprocessing already specifies feature scaling. Numerical-rank and zero-covariance safeguards handle degenerate inputs. PLS is still a linear regression model in the transformed features; it is not a general nonlinear learner. See the [PLSRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html).

Implementation: [fold transforms](../src/representation_screen.py), [ridge/PLS fitting](../src/interaction_share_learning.py), [animal-grouped selection](../src/neurotycho_statistical.py).

## 6. Learned SPI pooling

For each ordered link $e$, form

$$
u_e=[V_{e1},\ldots,V_{eK},v_1,\ldots,v_K]\in\mathbb R^{578},
$$

where $V$ contains standardized SPI values and $v_k$ flags a valid SPI column. Invalid columns are filled with zero and distinguished by their flags. Standardization is across links within the same recording, not across the dataset.

The model computes

$$
h_e=\phi_\theta(u_e)\in\mathbb R^{32},
\quad \phi_\theta:\ 578\rightarrow32\rightarrow32,
$$

using GELU nonlinearities. It then forms

$$
q=\left[
\frac1E\sum_e h_e,\ 
\sqrt{\frac1E\sum_e(h_e-\bar h)^2+10^{-6}}
\right]\in\mathbb R^{64},
$$

and predicts

$$
p=\sigma\!\left(\rho_\theta(q)\right),
\quad \rho_\theta:\ 64\rightarrow32\rightarrow1.
$$

There are **21,697 trainable parameters**. The same link network processes every link; the 289 SPI identities remain distinguishable as fixed input coordinates.

This follows the shared-map-plus-symmetric-aggregation principle associated with [Deep Sets](https://arxiv.org/abs/1703.06114). Our finite-width mean/SD architecture is a particular implementation; a general set-function representation theorem is not a guarantee that this small network will learn any desired function from this dataset.

### Why it is a useful comparator to Pearson $z$

Pearson uses fixed products followed by averages:

$$
z_{kl}=\frac1E\sum_e V_{ek}V_{el}.
$$

Learned pooling can approximate selected interactions and other nonlinear functions before aggregation. It may retain higher-order joint information and normalized single-SPI distribution information discarded by $z$. It also has to estimate its weights from the available labels.

Consequently, a pooling gain would not isolate “learned correlation” alone. Both the aggregation and the prediction function differ, and the input is richer than a list of pairwise correlations.

The model is invariant to arbitrary link ordering. It has no endpoint identities or explicit sensor topology. Padding stored in feature banks is removed before pooling reduced views; padded zeros are not treated as observed links.

Implementation: [standardization, input packing and pooling network](../src/spi_edge_pool.py).

## 7. Raw waveform encoder

The raw encoder has **139,841 trainable parameters** and receives batches $B\times T\times M$. It learns directly from standardized waveforms without SPI inputs.

### Temporal feature extraction

All channels use the same temporal filters. Internally, channels are temporarily treated as separate examples:

| Stage | Operation | Full-view output shape |
|---|---|---|
| Input reshape | $B\times T\times M\rightarrow BM\times1\times T$ | $BM\times1\times2000$ |
| First convolution | 1→32 channels, kernel31, stride4, padding15; GELU and dropout0.1 | $BM\times32\times500$ |
| Second convolution | 32→64, kernel15, stride4, padding7; GELU and dropout0.1 | $BM\times64\times125$ |
| Restore channel/time structure | Align channels at each temporal location | $B\times125\times M\times64$ |

For $T=1000$, the two temporal lengths are 250 and 63. The network learns filters that can respond to local oscillations, waveform shapes and other temporal patterns; it is not restricted to a prescribed frequency-band decomposition.

### Attention across aligned channels

At each temporal patch $p$, the $M$ channel vectors form an $M\times64$ matrix. Attention operates across those channels, separately at every patch.

One attention head forms learned projections $Q,K,V$, and computes

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The weights determine how channel features are combined, conditional on the observed signals. Four heads use $d_k=16$; their outputs are combined. Each of the two encoder blocks also contains residual connections, layer normalization and a 64→128→64 feedforward network. The underlying attention mechanism is introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762); this pilot uses it across channels rather than as a full temporal language-model architecture.

Keeping channels aligned until this stage allows interactions to depend on simultaneous local features. Pooling every channel over time before interaction would remove that opportunity.

Attention weights are learned computational weights, not identified causal connections.

### Post-interaction processing and pooling

A shared temporal convolution, 64→64 with kernel9, stride1 and padding4, combines nearby post-attention features. Mean and maximum are then taken over channel and temporal positions, producing 128 recording-level values. A 128→32→1 head with GELU produces the logit; a sigmoid produces the probability.

No learned electrode identities or coordinate embeddings are supplied. Shared channel processing is permutation-equivariant, and final pooling is permutation-invariant at evaluation:

$$
f(XP)=f(X)
$$

for a channel-permutation matrix $P$, up to numerical rounding. This helps with arbitrary channel ordering, but does not make the model invariant to dropping channels or changing the recorded anatomy.

### The important temporal limitation

Before pooling, the nominal interior receptive field is

$$
R=31+(15-1)\cdot4+(9-1)\cdot16=215
$$

raw samples, or approximately **0.86 seconds** at 250 Hz. Channel attention broadens access across channels, not across distant temporal patches.

Final pooling uses features from the whole window, so prediction is not based on only one 0.86-second segment. However, the architecture does not explicitly model arbitrary long-range temporal ordering after local feature extraction. It is a particular convolution/attention inductive bias, not an unrestricted neural encoder or pretrained foundation model.

Implementation: [AlignedChannelEncoder](../src/representation_state_neural.py).

## 8. Enriched raw training

The enriched model uses exactly the same architecture and parameter count. Its training data and augmentation differ:

- It draws from the dense set of 3,556 source windows before excluding the target animal.
- Independently for each training batch, it chooses the final four or eight seconds with equal probability.
- Independently, it chooses eight or sixteen channels with equal probability. For eight channels, the subset is sampled separately for each recording in the batch.
- It renormalizes each observed crop and balances the total loss contribution of every date/state.

All four size combinations can therefore occur during training. The matched raw model uses the selected full-size windows without this augmentation. The enriched model's validation windows remain the matched full-size windows of the held-out source animal.

The source observations have already been labelled at the interval level. Extracting more windows supplies more waveform examples, not more labelled animals. Those windows are correlated and should not be counted as new independent subjects.

This is a deliberately stronger comparator. Because extra windows and augmentation change together, their individual effects cannot be isolated from this comparison.

Implementation: [augmentation and training weights](../src/neurotycho_learning.py).

## 9. Optimization, model selection and final fitting

### Neural training

All three neural variants train from scratch with binary cross-entropy:

$$
\mathcal L=-\frac1B\sum_n w_n
\left[y_n\log p_n+(1-y_n)\log(1-p_n)\right].
$$

The implementation uses logits for numerical stability. Weights give equal total mass to every source date/state, normalized so the mean sample weight is one.

AdamW adapts updates using gradient-moment estimates and applies weight decay separately from the adaptive gradient step. This distinction matters: for adaptive optimizers, decoupled weight decay is not generally equivalent to adding an ordinary squared-weight penalty to the loss. See [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

The fixed grid is:

| Setting | Values/rule |
|---|---|
| Learning rate | $10^{-4},10^{-3}$ |
| Weight decay | $10^{-4},10^{-2}$ |
| Batch size | 16 |
| Maximum epochs | 400 |
| Early stopping | After at least30epochs, stop after30epochs without sufficient validation improvement |
| Improvement tolerance | $10^{-8}$ in balanced validation Brier |
| Initialization seeds | 11,23,47 |
| Pretraining | None |

For a candidate setting, each source validation animal is held out in turn. Its best checkpoint is selected by validation Brier. The chosen setting minimizes the mean across the three validation animals.

The final model starts afresh and trains on all three source animals for the **median best epoch** of the selected folds. A final fit can therefore use fewer than30epochs: the minimum30 rule ensures sufficient observation of the learning curve, not that the best checkpoint must occur after epoch30.

Three seeds give three fitted models. Reporting averages their performance metrics; it does not silently create an ensemble by averaging their probabilities.

### Brier, balanced accuracy and AUROC answer different questions

Balanced Brier is

$$
\mathrm{BBrier}
=\frac12\sum_{c\in\{0,1\}}
\frac1{N_c}\sum_{n:y_n=c}(p_n-c)^2.
$$

Lower is better. Perfect probabilities give0; a constant0.5 predictor gives0.25.

Balanced accuracy averages the two class recalls after thresholding. AUROC measures ranking across thresholds. Good AUROC can coexist with poor balanced accuracy at0.5 if the probabilities shift or become miscalibrated.

The primary target metric is balanced accuracy at0.5, with AUROC and Brier also reported. Model selection uses source Brier. These are deliberately different roles; a target-calibrated threshold would change the deployment question.

### The exact grouping logic

For each target animal:

1. Remove its source observations.
2. For each candidate, use three folds: fit on two remaining animals, validate on the third.
3. Choose the candidate using only those validation scores.
4. Fit on all three source animals.
5. Freeze the model and evaluate the target animal's PF dates.

Preprocessing and PCA/PLS are refitted inside each training fold. Otherwise, even an “unsupervised” projection could use information from validation data.

A selected source-CV score is still used for model selection. It is not an unbiased independent estimate of the chosen model's future performance.

The full study has $7\times2=14$ statistical fits and $3\times2\times3=18$ neural fits: **32 final source models**. Each neural fit includes its internal validation runs.

Implementation: [neural fitting driver](../scripts/fit_neurotycho_neural.py), [statistical fitting driver](../scripts/fit_neurotycho_statistical.py), [fixed evaluation](../scripts/evaluate_neurotycho_transfer.py).

## 10. Noise, interpretability and the claims the experiment can support

### Why high-dimensional does not automatically mean hopeless

The $z$ coordinates share SPIs, and many SPIs are related. Their number greatly exceeds the number of independent sources, but their redundancy and regularization can make useful prediction possible. Conversely, redundancy can amplify nuisance variation; 41,616 coordinates do not guarantee a strong signal.

Pearson compression removes both information and potential nuisance. A favourable bias–variance tradeoff is an empirical hypothesis. Near-constant MPI vectors, few observed links and uncertain SPI estimates can make correlations unstable. Link observations share channels, and time samples are autocorrelated; independent-sample correlation standard errors cannot be applied blindly.

A successful numerical check establishes that features were computed consistently. It does not establish that they are accurately estimated properties of the underlying process.

### What can be interpreted

For PCA/ridge, the linear coefficients can be mapped back through the PCA loadings to the transformed SPI coordinates. PLS also yields a linear predictive combination. That makes named SPI-pair contributions inspectable.

However, correlated features can exchange coefficients, and regularization and scaling affect their values. A large coefficient is not a causal mechanism. Stability across source fits matters more than a visually appealing ranking from one fit.

Learned pooling keeps identifiable SPI input coordinates, but nonlinear interactions make a single coefficient interpretation unavailable. Perturbation or sensitivity analyses would be additional analyses, not automatic explanations supplied by the architecture.

### Information accessibility versus learning efficiency

For a deterministic representation $z=f(\widetilde X)$, the data-processing inequality gives

$$
I(Y;z)\leq I(Y;\widetilde X).
$$

The representation cannot create label information absent from its input. Its potential advantage is making relevant information easier to extract with limited training data, compute or model capacity.

Within this pilot, $z$ is a fixed representation; PCA/PLS learn projections/readouts, pooling learns an aggregation of statistical coordinates, and raw encoders learn waveform representations. These are related but distinct forms of representation-based learning.

### How to read possible results

| Observation | Defensible interpretation | What it would not establish |
|---|---|---|
| $z$ improves on $m$ | Cross-statistic correspondence helps this learning setup | That every marginal or nonlinear baseline is inferior |
| $m+z$ improves on both | Their retained information is practically complementary | An information-theoretic uniqueness result |
| Pooling improves on $z$ | A richer learned aggregation is useful | That Pearson is inherently defective or universally dominated |
| $z$ competes with raw learning | Its statistical prior has utility under these conditions | Superiority to all neural methods |
| Enriched raw learning improves | More source exposure plus augmentation helps | Which of those two changes caused the gain |
| AUROC holds but fixed-threshold accuracy falls | Ranking survives better than probability/threshold transfer | That target-calibrated deployment was tested |
| All methods transfer poorly | The specified transfer is difficult for these pipelines | That state labels imply a common, easily recoverable mechanism |

Target results average dates within animal and then the two animals. The cohort does not support broad population-level or clinical conclusions. Anaesthetic and calendar period change together, so their separate effects are not identified.

The scientific value comes from the pattern across these controls and its reproducibility—not from finding one number on which $z$ wins.

## 11. Reading order and implementation map

For intuition, read Sections2,6,7 and10. For reproducing the analysis, add Sections1,4,5,8 and9.

| Topic | Authoritative repository file |
|---|---|
| Study decisions | [Frozen protocol](neurotycho-transfer-pilot.md) |
| Numerical settings | [Configuration](../configs/analysis/neurotycho-transfer-260910.yaml) |
| Windowing, filtering, power spectra | [neurotycho_pilot.py](../src/neurotycho_pilot.py) |
| Ordered-edge Pearson representation | [spi_spi_contract.py](../src/spi_spi_contract.py) |
| Rich SPI summaries | [representation_attribution.py](../src/representation_attribution.py) |
| Graph summaries | [mpi_representation_baselines.py](../src/mpi_representation_baselines.py) |
| Training-only feature transforms | [representation_screen.py](../src/representation_screen.py) |
| Ridge and PLS implementation | [interaction_share_learning.py](../src/interaction_share_learning.py) |
| Grouped statistical selection | [neurotycho_statistical.py](../src/neurotycho_statistical.py) |
| Learned link pooling | [spi_edge_pool.py](../src/spi_edge_pool.py) |
| Raw encoder | [representation_state_neural.py](../src/representation_state_neural.py) |
| Binary training and augmentation | [neurotycho_learning.py](../src/neurotycho_learning.py) |
| Metric aggregation and integrity checks | [neurotycho_evaluation.py](../src/neurotycho_evaluation.py) |
| Full prospective evaluation | [evaluate_neurotycho_transfer.py](../scripts/evaluate_neurotycho_transfer.py) |

Some shared modules also contain earlier experimental models. This guide describes only the architectures and settings selected by the NeuroTycho configuration.
