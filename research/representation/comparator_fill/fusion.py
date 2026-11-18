"""X ⊕ z, z | X and commonality on the cohort banks, under the cohort protocol of cohort.py.

Blocks: r (Pearson edges), S (Pearson + relative spectra), mean-SPI, z. Per outer fold each block keeps its own
fitted model (cohort.fit_predict: own PCA cap and C); its inner out-of-fold logits on the outer training rows
are cached.
- Stacks. Any set of blocks, single blocks included, is combined by the same non-negative, lightly penalised
  logistic meta-learner fitted on those inner logits, so every compared model is nested and equally calibrated.
  X ⊕ z is the stack of X's blocks and z. The decisive contrast is (S ⊕ mean-SPI) ⊕ z against S ⊕ mean-SPI.
- z | X. z's PCs minus a ridge prediction from all of X's standardised features: cross-fitted by group on the
  outer training rows, fitted on all of them for the test rows. What z carries that X cannot express linearly;
  for a linear readout this is z's partial effect in the joint model, not an independent test.
- Commonality on unit-level cross-validated log-loss D (bits, probabilities clipped at 1e-3):
  U(z; X) = D(X) - D(X ⊕ z), U(X; z) = D(z) - D(X ⊕ z), shared = D(null) - D(X ⊕ z) - U(z; X) - U(X; z),
  with paired participant-bootstrap intervals. Negative shared is synergy (suppression).
Nothing is selected on outer test rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from research.representation.comparator_fill import cohort as C
from research.representation.dataset_screen.common import folds

BLOCKS = dict(r=['pearson'], S=['pearson', 'spectral'], mean_spi=['mean_spi'], z=['z'])
STACKS = dict(r=['r'], S=['S'], mean_spi=['mean_spi'], z=['z'], S_mean_spi=['S', 'mean_spi'],
              r_z=['r', 'z'], S_z=['S', 'z'], mean_spi_z=['mean_spi', 'z'], S_mean_spi_z=['S', 'mean_spi', 'z'])
BASES = dict(r='r', S='S', mean_spi='mean_spi', S_mean_spi='S_mean_spi')  # X for X ⊕ z, z | X and commonality
EPS = 1e-3
PENALTY = 1e-2


def clipped_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


class Meta:
    """Logistic regression on block logits with non-negative weights and a small L2 penalty."""
    def fit(self, F, y):
        def loss(theta):
            eta = theta[0] + F @ theta[1:]
            p = np.clip(expit(eta), 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + PENALTY * np.sum(theta[1:] ** 2)
        k = F.shape[1]
        self.theta = minimize(loss, np.r_[0.0, np.full(k, 1.0 / k)], method='L-BFGS-B',
                              bounds=[(None, None)] + [(0, None)] * k).x
        return self

    def predict(self, F):
        return expit(self.theta[0] + F @ self.theta[1:])


def block_logits(X, y, groups, names, train, test):
    """Inner out-of-fold logits on the training rows, and the full-training fit's logits on the test rows."""
    inner = np.full(len(train), np.nan)
    for a, b in folds(y[train], groups[train]):
        inner[b] = C.fit_predict(X, y, groups, names, train[a], train[b])[0]
    return clipped_logit(inner), clipped_logit(C.fit_predict(X, y, groups, names, train, test)[0])


def residual_z(X, y, groups, names, train, test):
    """z's PCs minus their ridge prediction from X's standardised features: cross-fitted on train, fitted on test.

    The PCA basis and scaling (label-free) are fitted once on the training rows, so every row's residual is in
    the same coordinates; only the ridge is cross-fitted.
    """
    fz = C.Blocks(['z']).fit({'z': X['z'][train]}).transform({'z': X['z']})
    raw = np.hstack([X[n] for n in names])
    fx = np.nan_to_num(StandardScaler().fit(raw[train]).transform(raw))
    ridge = lambda fit, apply: fz[apply] - RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(fx[fit], fz[fit]).predict(fx[apply])
    out = np.zeros_like(fz)
    out[test] = ridge(train, test)
    for a, b in folds(y[train], groups[train]):
        out[train[b]] = ridge(train[a], train[b])
    return out


def log_loss_bits(label, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(label * np.log2(p) + (1 - label) * np.log2(1 - p)))


def commonality(label, p_x, p_z, p_xz, p_null):
    d = {k: log_loss_bits(label, v) for k, v in dict(x=p_x, z=p_z, xz=p_xz, null=p_null).items()}
    u_z, u_x = d['x'] - d['xz'], d['z'] - d['xz']
    return dict(unique_z=u_z, unique_base=u_x, shared=d['null'] - d['xz'] - u_z - u_x)


def bootstrap(label, keys, statistic, n=2000, seed=261102):
    """Paired participant bootstrap: 95% interval of each entry of statistic(rows)."""
    rng = np.random.default_rng(seed)
    people = np.array([k.split('|')[0] for k in keys])
    unique = np.unique(people)
    draws = []
    for _ in range(n):
        pick = np.concatenate([np.flatnonzero(people == u) for u in rng.choice(unique, len(unique))])
        if len(np.unique(label[pick])) == 2:
            draws.append(statistic(pick))
    return {k: [float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5))]
            for k in draws[0]}


def scores(label, p, prior):
    s = C.summary(label, p)
    brier_null = np.mean((prior - label) ** 2)
    return dict(s, log_loss_bits=log_loss_bits(label, p), brier_skill=float(1 - s['brier'] / brier_null))


def main(cohort, root, output, repeats, contrasts=None):
    bank, X, x, _ = C.load(root)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    roles_all = bank['role'].astype(str) if 'role' in bank.files else np.full(len(y_all), '')
    animals_all = groups_all
    blocks = {k: v for k, v in BLOCKS.items() if all(b in X for b in v)}
    stacks = {k: v for k, v in STACKS.items() if all(b in blocks for b in v)}
    bases = {k: v for k, v in BASES.items() if v in stacks}
    report = dict(cohort=cohort, scope=__doc__.splitlines()[0], contrasts={})
    for label, positive, negative in C.CONTRASTS[cohort]:
        if contrasts and label not in contrasts:
            continue
        ix = np.flatnonzero(np.isin(y_all, [positive, negative]))
        y, groups, roles, animals = (y_all[ix] == positive).astype(int), groups_all[ix], roles_all[ix], animals_all[ix]
        Xc = {k: v[ix] for k, v in X.items()}
        outer = C.splits(cohort, y, groups, roles, animals, repeats)
        used = max(r for r, _, _ in outer) + 1
        rows = list(stacks) + [f'z|{k}' for k in bases]
        window_p = {n: np.full((used, len(y)), np.nan) for n in rows}
        prior = np.full((used, len(y)), np.nan)
        weights = {k: [] for k in stacks}
        for r, train, test in outer:
            prior[r, test] = y[train].mean()
            cache = {k: block_logits(Xc, y, groups, v, train, test) for k, v in blocks.items()}
            for k, members in stacks.items():
                meta = Meta().fit(np.column_stack([cache[m][0] for m in members]), y[train])
                window_p[k][r, test] = meta.predict(np.column_stack([cache[m][1] for m in members]))
                weights[k].append(meta.theta[1:].tolist())
            for k, stack in bases.items():
                names = sorted({b for m in stacks[stack] for b in blocks[m]})
                fold = dict(Xc, resid=residual_z(Xc, y, groups, names, train, test))
                window_p[f'z|{k}'][r, test] = C.fit_predict(fold, y, groups, ['resid'], train, test)[0]
            print(cohort, label, 'outer fold done', r, flush=True)
        unit_p = {}
        for n in rows:
            keys, unit_label, unit_p[n] = C.aggregate(cohort, y, groups, animals, window_p[n], used)
        _, _, null = C.aggregate(cohort, y, groups, animals, prior, used)
        entry = dict(windows=int(len(ix)), unit_keys=keys.tolist(), unit_labels=unit_label.tolist(),
                     scores={n: scores(unit_label, unit_p[n], null) for n in rows},
                     meta_weights_mean={k: np.mean(v, axis=0).tolist() for k, v in weights.items()},
                     null_log_loss_bits=log_loss_bits(unit_label, null), bases={},
                     units={n: unit_p[n].tolist() for n in rows})
        for k, stack in bases.items():
            fused = f'{stack}_z'
            pick_all = np.arange(len(unit_label))
            stat = lambda s, a=stack, f=fused: commonality(unit_label[s], unit_p[a][s], unit_p['z'][s], unit_p[f][s], null[s])
            entry['bases'][k] = dict(
                fused_minus_base=dict(auroc_difference=entry['scores'][fused]['auroc'] - entry['scores'][stack]['auroc'],
                                      interval=C.paired_bootstrap(unit_label, unit_p[fused], unit_p[stack], keys)),
                conditional_auroc_interval=bootstrap(unit_label, keys, lambda s, n=f'z|{k}': dict(
                    auroc=C.summary(unit_label[s], unit_p[n][s])['auroc']))['auroc'],
                commonality_bits=stat(pick_all), commonality_interval=bootstrap(unit_label, keys, stat))
            print(cohort, label, k, {a: round(b, 3) for a, b in entry['bases'][k]['commonality_bits'].items()}, flush=True)
        report['contrasts'][label] = entry
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=sorted(C.CONTRASTS))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--contrasts', nargs='*')
    a = parser.parse_args()
    main(a.cohort, a.root, a.output, a.repeats, a.contrasts)
