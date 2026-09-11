# Conventional-library follow-up: verified results

Completed 11 September 2026 UTC. The [declaration](neurotycho-library-followup.md)
was committed as `b53baf1` before these new models were run, **after inspection of
the original PF pilot**. These are follow-up results on the same test recordings,
not an independent prospective replication. Original results remain unchanged.

**Pearson z outperformed all three added feature pipelines in full-size balanced
accuracy. MiniRocket was the closest competitor and had essentially the same
Brier score as z-PCA, with perfect ranking.** This strengthens the scoped
fixed-threshold transfer comparison; it does not establish information that
conventional representations cannot recover.

## Comparison

Every new method used the original matched source windows and source-animal
splits: 288 or 256 windows from three animals, excluding the target animal.
Evaluation comprises 128 balanced windows from two PF dates each for Chibi and
George. All observations have 16 bipolar channels and 2,000 samples. No reduced
view was evaluated in this follow-up. No new SPI extraction or neural training
was performed.

| Feature pipeline | Balanced accuracy | AUROC | Brier |
|---|---:|---:|---:|
| **z + PCA, original** | **.9297** | 1.0000 | .0616 |
| **z + PLS, original** | **.9453** | 1.0000 | .0597 |
| MiniRocket + logistic | .8984 | 1.0000 | .0612 |
| tsfresh Efficient + logistic | .8438 | .9990 | .0842 |
| catch22 + logistic | .8359 | 1.0000 | .1392 |
| Spectrum + logistic, original | .8672 | 1.0000 | .0936 |
| Matched raw neural, original | .8307 | .9980 | .1670 |
| Enriched raw neural, original | **.9635** | 1.0000 | **.0284** |

Metrics average dates within each animal and then animals. MiniRocket and neural
rows average the metrics from three seeds; no probability ensemble or best-seed
selection. MiniRocket seed BA is .90625/.890625/.8984375, below both z readouts in
each case. Its Brier is slightly lower than z-PCA's on average; the difference is
small and no formal equivalence or superiority test is claimed.

The z-PCA gain over mean MiniRocket BA is **3.125 percentage points**, equivalent
to four fewer errors over these 128 balanced windows; z-PLS gains **4.6875 points**,
equivalent to six. Against tsfresh, the gains are 8.594 and 10.156 points.
These counts should prevent overstating a small two-animal pilot.

| Pipeline | Chibi BA | George BA | George 31 July | George 3 August |
|---|---:|---:|---:|---:|
| z + PCA | 1.0000 | .8594 | .8438 | .8750 |
| z + PLS | 1.0000 | .8906 | .9062 | .8750 |
| MiniRocket | 1.0000 | .7969 | .8750 | .7188 |
| tsfresh | 1.0000 | .6875 | .8125 | .5625 |
| catch22 | 1.0000 | .6719 | .7500 | .5938 |

All five pipelines score perfectly on both Chibi dates. z-PCA's improvement over
MiniRocket is driven by the second George date; it loses on the first. z-PLS
exceeds MiniRocket on both George dates. Thus the overall advantage is not
uniform evidence across many subjects or sessions.

## What was actually implemented

- **MiniRocket:** aeon 1.5.0 multivariate transform, requested 10,000 features,
  yielding 9,996; 32 maximum dilations/kernel; seeds 11/23/47. Per-window channel
  standardization matches the raw neural input convention. Bias fitting occurs
  separately within each inner training fold and again on the final source set.
- **tsfresh:** 0.21.2 `EfficientFCParameters`, 777 features per channel from
  filtered float64 observations, including amplitude-related features. Both
  target-animal fits selected channel concatenation: 12,432 coordinates before
  training-only filtering. This is not the Minimal catalogue or the unrestricted
  Comprehensive catalogue.
- **catch22:** pycatch22 0.5.0, the standard 22 features, not catch24. The Chibi
  fit selected 352 concatenated channel features; George selected 110 pooled
  features. Pooling uses mean/SD/quartiles/median over channels.

All heads were L2 logistic regression with source-only C selection from
.01/.1/1/10, finite-column filtering, median imputation and standard scaling.
Date/state training weights and source balanced-Brier selection match the
original spectral comparator. Neither target data nor target labels entered
selection or calibration. All ten final source models were saved before their
PF prediction stage.

These are **specified feature-transform + logistic pipelines**, not entire
libraries. In particular, this is not a comparison against every ROCKET variant
or the default MiniRocket/RidgeClassifier pipeline, nor against tsfresh with
every possible nonlinear classifier. All six MiniRocket fits selected C=.01,
the strongest regularization in the declared grid; an optimal value beyond that
grid has not been ruled out. No grid extension was made after observing PF.

The comparison is full-size only. Channel concatenation and ordinary multivariate
MiniRocket should not be described as inherently invariant to channel relabelling
or compatible with arbitrary changes in channel count. Source selection between
pooled and concatenated statistical features addressed this modelling choice
without choosing the better layout on PF.

## Interpretation

This is favourable evidence for **competitive transfer of a source-trained
decision rule using z**, beyond the earlier comparison with SPI marginals and a
custom raw encoder. However, conventional feature representations already rank
the two states perfectly or almost perfectly. The data therefore do not support
the proposed stronger mechanism that those methods fail to expose the relevant
state information. Threshold transfer and probability behaviour are central.

The enriched neural control still wins. It has greater window exposure and
source augmentation, but uses the same labelled animals/dates; it must remain
visible. No label-efficiency curve, population uncertainty, clinical inference
or independent confirmation follows from this extension. The original
animal/agent/period and filtering-context limitations also remain.

No further library expansion is triggered by these outcomes. InceptionTime remains
a relevant future comparator for a broader neural claim, and a fresh test would
strengthen generalization claims. These are distinct future decisions, not
reasons to keep modifying the already-inspected PF benchmark.

## Verification, cost and reproduction

The independent checker verified all 640 target predictions, ten fitted models,
every candidate's source-animal split and source Brier score, selected settings,
source/target feature-name alignment, original target-window labels and final
date/animal aggregates. Reloaded target predictions agree **exactly**. Independently
computed tsfresh means, variances and SDs agree exactly over all source and target
channels. The original pilot report hash is unchanged.

Joint catch22/tsfresh extraction took 609.4 summed task-seconds for 352 source
windows and 172.1 for 128 PF windows, with four local workers. Including source CV,
final fitting and source replay, catch22's two fits took 2.5 s total, tsfresh's two
61.4 s, and MiniRocket's six 76.1 s. These are local timings and jointly extracted
feature costs, not isolated per-library CPU benchmarks or a matched-hardware
comparison with pyspi. They do establish that these were inexpensive controls.

One implementation repair was required: pandas supplied tsfresh feature names
as an object array that the numeric NPZ loader rejected. Commit `f35abb6` stores
Unicode strings. The 352 already-computed numerical feature arrays and all name
values were preserved exactly; originals and a hash-by-hash migration audit remain
under the result root. No extraction values, fitting choices or predictions were
changed to resolve this metadata issue.

Result root: `results/neurotycho_library_followup_260911/`.
Authoritative outputs: `report/report.json`, `report/predictions.npz`,
`report/verification.json`; full source candidates/checkpoints in `fits/`.
Report SHA256: `9831fe94ed7589567134297c93bc72e4b73c27bcd54472aca9f06e61b02ab13b`.
Runtime pins are in `runtime-requirements.txt`; a separate `.venv-neuro-baselines`
was used without changing the shared pyspi environment.

Re-run verification with:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-neuro-baselines/bin/python scripts/check_neurotycho_library_followup.py
```
