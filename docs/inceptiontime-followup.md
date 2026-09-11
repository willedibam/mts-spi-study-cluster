# InceptionTime: bounded comparator follow-up

Declared 2026-09-11 after inspection of the existing synthetic and NeuroTycho
results. The user explicitly requested both. This is a post-result baseline
addition, not fresh confirmation. No new generators, labels or pyspi extraction.

## Question and coverage

Does the previously observed benefit survive comparison with a published
multiscale convolutional classifier trained on the same labelled source units?
Use full M=16 observations only; standard InceptionTime has a fixed channel
input, so silently padding M=8 or inventing a pooling adapter would change the
comparator. Synthetic T=1000; NeuroTycho T=2000. This addition does not close the
published-baseline gap for reduced M/T or test the conventional feature libraries
on synthetics.

Synthetic: reuse the original co-organization source cohorts and nested 10/20/40
total-label budgets, including validation labels; identical stratified two-fold
indices. Freeze source fits, then evaluate original, faster-dynamics and direct
phase-mechanism full-size targets (200 independent masters each). NeuroTycho:
reuse matched KTMD windows, exclude the target animal before fitting, and select
using the existing three source-animal folds. Evaluate the four PF dates in two
animals with the original date-then-animal metric averaging. No dense-window
enrichment, target calibration or pretraining.

## Architecture and training

Independent PyTorch implementation of the architecture described by
[Fawaz et al.](https://arxiv.org/abs/1909.04939), checked against the
[author implementation](https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py).
Six modules: a 32-channel bottleneck, parallel 32-filter convolutions with
kernels 40/20/10, and a max-pool/1x1 branch; concatenate, batch-normalize and ReLU.
Residual projections after modules 3 and 6, temporal global averaging and a
two-class linear output. Both residual projections remain learned even at equal
width. Even-kernel SAME padding follows the original left-floor/right-ceiling
convention. Glorot weights; BatchNorm epsilon .001, update fraction .01.
Report parameter counts and individual members as well as the probability mean
of five independently initialized, source-selected members.

This is **InceptionTime architecture with a declared source-CV training protocol**,
not a bitwise reproduction of the old Keras runtime/default training recipe.
Two-logit cross entropy is implemented as BCE on their difference, exactly the
same binary likelihood. Source dates/states are equally weighted for ECoG.
The four-candidate grid includes LR .001/.0001 and decay 0/.0001; zero decay
recovers ordinary Adam updates. Training-loss plateau scheduler follows the
original .5/50/.0001 settings. Batch size 16, source-validation balanced Brier
selection, 100 minimum epochs, 100 patience and 1500 maximum. Refits use median
source-fold best epoch. Adam epsilon 1e-7; floating-point/backend differences
remain possible. These settings provide a credible classifier evaluation; the
training loss/selection and compute differ from earlier synthetic MSE encoders.
No equal-compute claim. Selected epoch ceilings are reported, never hidden.

## Execution and verification

`configs/analysis/inceptiontime-followup-260911.yaml` freezes settings.
`scripts/run_inceptiontime_followup.py` packs existing data, fits one case
(five members) independently, then evaluates only after **all 55 source refits**
exist. Eleven cases: nine synthetic budget/cohort pairs and two ECoG target
animals. Source and target packs are separate; fitting never opens target packs.
Save exact IDs, folds, candidate histories, source predictions, checkpoints,
runtime and content hashes. Resume rejects changed inputs/code/config.

Run local unit checks, then two source-only Gadi smoke fits at the two recording
lengths before production. Use shared immutable code and compact gdata packs.
First production fits establish runtime; do not repeat completed fits. CPU
checkpoint replay, label/group/split checks and independently recomputed metrics
are required before calling results verified. Any source-selected epoch ceiling
or optimization failure qualifies the conclusion; do not tune on test scores.

## Interpretation

InceptionTime receives fixed channel positions, while the earlier raw encoders
were channel-exchangeable. Synthetic populations have no corresponding channel
identity across realizations; this is a meaningful inductive-bias distinction,
not justification for weakening the baseline. Keep the exchangeable models too.
NeuroTycho's strongest enriched raw encoder remains relevant regardless of the
new ranking. A win by z supports this defined comparison; a loss narrows the
claim without invalidating the correspondence intervention. No universal
superiority or information-inaccessible-from-raw-data claim follows either way.

## Verified findings, 2026-09-11

All 55 source fits completed before target evaluation. The independent checker verified every source split, candidate score and selected setting, all 174 metric rows and 200 sampled CPU checkpoint replays. Maximum CPU replay error was 5.96e-7; all GPU source replays were exactly zero. All 55 source-report hashes match the evaluation freeze, and their module hashes match the declared immutable source commits. Report SHA256: `6f11a2736e3414547833eed5bd4d0af8a84c9b37e9580b735981a947d9f715b8`.

### Synthetic comparison

These are full M=16 observations, averaged over the three disjoint source cohorts. Each InceptionTime entry evaluates the probability average of five members; cohorts share the 200 target masters, so these are not 600 independent test systems. Source labels include validation. Existing reduced-observation results remain separate.

| Target regime | Labels | InceptionTime ensemble BA | Ensemble AUROC | z-PCA BA | z-PLS BA |
|---|---:|---:|---:|---:|---:|
| Original | 10 | 50.50% | 0.5261 | 95.67% | 98.00% |
| Original | 20 | 49.83% | 0.5018 | 99.83% | 100.00% |
| Original | 40 | 50.33% | 0.4973 | 100.00% | 100.00% |
| Faster dynamics | 10 | 49.17% | 0.4746 | 98.50% | 100.00% |
| Faster dynamics | 20 | 51.50% | 0.5380 | 100.00% | 100.00% |
| Faster dynamics | 40 | 50.33% | 0.5017 | 100.00% | 100.00% |
| Direct phase mechanism | 10 | 50.67% | 0.5199 | 79.17% | 88.17% |
| Direct phase mechanism | 20 | 48.00% | 0.4695 | 99.50% | 99.50% |
| Direct phase mechanism | 40 | 50.33% | 0.5034 | 100.00% | 100.00% |

Individual-member mean BA also remains near chance (49.23–50.43% across these nine cells), so ensembling does not hide successful synthetic members on average. At 40 labels, rich per-SPI summaries achieve 98.83%, 80.00% and 71.83% BA in original/faster/direct regimes; z-PCA reaches 100% throughout. The observed phase/envelope specialist reaches 100% at full size, and learned SPI pooling is competitive. Those controls remain central: beating an unsuccessful raw learner alone would not establish a distinctive relational contribution. Complete method rows and all individual scores remain in the numerical artifacts.

Four selected validation folds reached 1,500 epochs, all at 10 labels: cohort 23/members 23 and 47, cohort 47/members 47 and 71. No selected fold reached the limit at 20 or 40 labels. Their selected validation Brier ranges .259–.349 and .260–.340, respectively, versus .25 for a constant balanced-class score. Some folds fit training data very closely while validation remains poor. This supports a finite-data generalization difficulty; it does not establish an inability of the architecture to express the target. Channel permutation, temporal scale and training-sample size are plausible contributors, not isolated causal explanations. The 100-epoch minimum concerns how long validation training runs; the best checkpoint and median-epoch final refit can legitimately be earlier.

### NeuroTycho comparison

Full M=16, T=2000 only; the same matched source windows and animal exclusions as the existing pilot. Date metrics are averaged within animal and then across the two target animals. InceptionTime receives no extra windows or augmentation.

| Model/aggregation | BA | Date-averaged AUROC | Brier |
|---|---:|---:|---:|
| z-PCA | 92.97% | 1.0000 | .0616 |
| z-PLS | 94.53% | 1.0000 | .0597 |
| InceptionTime five-member ensemble | 95.31% | .9980 | .0568 |
| InceptionTime mean of five member metrics | 84.38% | .9984 | .1218 |
| Earlier enriched raw encoder, mean of three seeds | 96.35% | 1.0000 | See original pilot |

InceptionTime member BAs are 96.09%, 85.16%, 92.19%, 72.66% and 75.78% for seeds 11/23/47/71/101. The ensemble is the declared primary architecture-level comparator; reporting only the weaker member average would be misleading. Seed means and probability ensembles are distinct quantities, and the earlier raw encoder's three-seed mean is not an ensemble.

The ensemble achieves 100% BA on both Chibi dates and 93.75%/87.50% on George's two dates. All member date-averaged AUROCs are .9971–1.0000 despite threshold-performance variation, consistent with score-transfer sensitivity. Chibi/George source validation Brier ranges .0026–.0078/.0053–.0086 with no selected epoch ceilings. Thus the same implementation learns a strong real-data classifier. The ensemble exceeds z-PCA by three errors and z-PLS by one error across 128 balanced windows; two target animals do not establish a population-level superiority claim. Conversely, the old statement that z outperforms the matched raw neural average cannot be generalized to matched-data neural methods after this result.

### Interpretation and completion

The addition closes the specific published convolutional-comparator gap at full size: z exposes the selected synthetic co-organization target efficiently to simple readouts at 10–40 labels, whereas this source-selected InceptionTime implementation does not generalize. This is a result about the tested inductive biases and learning conditions, not raw-inaccessible information, universal neural inferiority, or proof that Pearson compression is optimal. The synthetic construction, specialist success, learned-pooling evidence and four 10-label ceiling cases must accompany the claim.

On NeuroTycho, z remains a useful fixed statistical descriptor, but neither a general neural-superiority claim nor a necessity-of-dependence interpretation follows. Conventional spectral ranking and stronger neural performance remain counterevidence to those claims. The combined study is more informative as a characterization of a selective representation prior than as an aggregate leaderboard win.

No further fit, larger saturated repeat, generator change or target-informed tuning is warranted by this comparison. Synthesize the existing positive and negative studies now. Conventional feature-library controls remain absent from the central synthetic tasks, and reduced-channel published-neural comparison remains open; do not describe these gaps as completed or automatically expand into a benchmark grid. A separate high-impact learning paper still needs an independent general lesson or consequential application beyond this chosen synthetic task. This result does not decide the paper split.

All fits, candidate histories, predictions and audits are under `results/inceptiontime_followup_260911/`. The final summary binds its baseline input hashes. Reproduce with `PYTHONPATH=. .venv/bin/python scripts/check_inceptiontime_followup.py`, then `PYTHONPATH=. .venv/bin/python scripts/summarize_inceptiontime_followup.py`. Outputs: `evaluation/{report.json,verification.json,comparison.json,synthetic-comparison.csv,full-size-comparison.png,full-size-comparison.svg}` and `source-code-verification.json`. Figure values and labels were visually checked. Summed member fitting/CV time is 6,879.7 seconds (1.91 GPU-hours), excluding queueing, setup and transfer; this is not a matched-hardware cost comparison with pyspi. No source fit was repeated. The completed scheduled follow-up was deleted after verification.
