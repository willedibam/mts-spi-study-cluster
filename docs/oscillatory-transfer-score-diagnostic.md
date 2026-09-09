# Score-transfer diagnostic after the mechanism test

Declared 2026-09-10 after inspecting the completed, verified mechanism-transfer
results, before computing this diagnostic. This is explicitly post hoc. It does
not replace the frozen primary threshold or constitute a new confirmation.

## Why this check is useful

At 40 source labels and M8/T500, z PCA/ridge has BA=.890 but AUROC=.99833;
learned SPI pooling has BA=.93833 and AUROC=.99813. The PCA models make no
false-positive errors, but miss 20–25% of aligned records across the three source
cohorts. Thus a near-perfect ordering coexists with an imperfect source-trained
threshold. We should not interpret the accuracy difference as a demonstrated
difference in the availability of task information. Individual-SPI summaries have
AUROC=.80043 here, so threshold correction need not eliminate their disadvantage.

## One diagnostic, no new fitting or extraction

Use the existing 99 frozen prediction vectors, all three source budgets/cohorts,
and both observation views of the mechanism-transfer cohort. Keep every original
prediction and primary metric unchanged. For each 200-record target batch, apply
one alternative decision rule: assign the highest-scoring half to the aligned
class. The positive fraction .5 is supplied from the balanced experimental
design; it is not estimated using individual target labels.

This rule uses **the unlabelled target batch and its known class prevalence**.
It has a different information budget from the primary source-only fixed rule.
It is not individual-record deployment, general probability calibration, or a
claim that class prevalence would be known in an application. It would be
inappropriate to infer an unknown clinical-state prevalence this way.

Ties at the cutoff receive a uniform fractional assignment equal to the remaining
positive count divided by the number tied. Evaluate expected balanced accuracy
under random tie resolution, rather than exploiting the file's class ordering.
The decision function takes scores only; target labels enter evaluation only.
Report the original BA, alternative expected BA, decision threshold and original
AUROC for every fit, with means over the same three source cohorts. Brier remains
the original frozen score's Brier; do not treat these assignments as calibrated
class probabilities. No threshold optimization, new hyperparameters, new
permutation seeds, or target-trained classifier.

## Interpretation and stopping rule

If z and pooling become comparably accurate, their earlier BA gap is largely a
score-transfer difference under the tested conditions, not a ranking advantage
for pooling. If individual summaries remain worse, threshold shift alone is
insufficient to explain their deficit. If the assumed prevalence is essential,
say so rather than claiming a free correction.

This diagnostic only distinguishes those interpretations. Do not use it to
choose a new primary method or launch a prevalence/calibration grid. After this
check and the consolidated findings, stop the current experiment. A separate
few-shot adaptation or application study would require an independently motivated
question and its own data/validation contract.
