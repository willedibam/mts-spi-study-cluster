# Greedy forward SPI selection

## Goal
Select a subset S of SPIs from a pool of K (~300) such that the SPI-SPI feature
vector — Spearman rank correlations between pairs of off-diagonal MPI vectors
within S — maximizes 9-class CV performance on the corpus. S is grown greedily,
one SPI per step, anchored on Pearson_r.

## Inputs
- MPI tensors per MTS: shape (K, M, M). Stored per existing pipeline conventions.
- Class labels: 9 classes (Gaussian noise, Cauchy noise, CML-defect, CML-sti1,
  Kuramoto-fast, Kuramoto-slow, VAR1-strong, VAR1-weak, 1D-wave).
- Grid: realizations across (M, T) cells. Group key = (class, M, T, seed) for CV
  to prevent generator-shared leakage. Confirm group key in repo before running.
- SPI index map: name → integer index. Identify Pearson_r index up front.

## Feature definition
For a subset S = {s_1, ..., s_n} (n ≥ 2), the feature vector for one MTS is

    f_{ab} = spearmanr( v(s_a), v(s_b) )    for all a < b in S,

where v(s) is the off-diagonal vector of MPI[s] (use the directed M(M-1) form
if the SPI is directed, undirected M(M-1)/2 otherwise; do not mix). NaN
correlations (constant vectors) → 0.

## Algorithm (plain greedy forward)
1. S ← {Pearson_r}.
2. For n = 2 ... N_max (default N_max = 20):
     For each candidate c ∈ {all SPIs} \ S:
         S' ← S ∪ {c}
         Build X ∈ R^(N_samples × (|S'| choose 2)).
         Standardize X (fit scaler on train fold only).
         CV: 5-fold stratified GROUPED on (class, M, T, seed).
         Classifier: sklearn LogisticRegression, multi_class='multinomial',
                     solver='lbfgs', C=1.0, max_iter=1000.
         Record mean CV log-loss and CV accuracy across folds.
     c* ← argmin mean CV log-loss.
     S ← S ∪ {c*}.
     Log: n, c*, S, log-loss, accuracy.
3. Stop when mean(Δlog-loss over last 3 steps) < 1e-3, or n = N_max.

## Outputs
- `greedy/` is the folder.
- `selection_trace.csv`: columns [step, added_spi, log_loss, accuracy, S_size].
- `final_S.json`: ordered list of selected SPI names.
- `selection_curve.png`: log-loss + accuracy vs |S|.
- Stopping rule decision logged to `summary.txt`.

## Implementation notes
- Cache per-MTS off-diagonal vectors per SPI index once; recompute pairwise
  Spearman on the fly (cheap with scipy.stats.spearmanr).
- Parallelize over candidates within a step (joblib, n_jobs=-1).
- Set RNG seed (default 42) for fold splitting.
- Skip SPIs whose v(s) is constant for ≥ 10% of MTS (degenerate; will yield
  uniform-zero feature columns).
- Keep the candidate evaluation loop independent — no shared state between
  candidates within a step.

## Out of scope for v1
- Floating selection (SFFS), bootstrap stability, nested CV. Add only if v1
  results warrant it.