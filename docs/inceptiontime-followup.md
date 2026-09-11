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
