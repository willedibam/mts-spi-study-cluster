"""Small training-only readouts for the focused interaction-share comparison."""
import itertools
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyRegressor

from src.representation_screen import fit_view, fit_block, ViewTransform


def standardized_marginal_shapes(rich, validity):
    """Match Pearson's positive-affine invariance without cross-SPI alignment.

    Rich blocks are mean, std, skewness, kurtosis and 19 raw quantiles.
    Constant/invalid SPIs have no standardized shape, matching z's domain.
    This is a within-record transform, never a corpus-fitted normalization.
    """
    values = np.asarray(rich).reshape(len(rich), -1, 23)
    valid = np.asarray(validity, dtype=bool) & np.isfinite(values[:, :, 1]) & (values[:, :, 1] > 0)
    scale = np.where(valid, values[:, :, 1], 1)
    quantiles = (values[:, :, 4:] - values[:, :, :1]) / scale[:, :, None]
    shapes = np.concatenate([values[:, :, 2:4], quantiles], axis=2)
    shapes[~valid] = np.nan
    return shapes.reshape(len(rich), -1)


def fit_statistical(bank, view, train, target, preprocessing, head, value, alpha=None):
    if head in ('pca', 'rbf'):
        transform, x = fit_view(bank, view, train, {**preprocessing, 'pca_dimensions':value})
        model = ridge_readout(x, target[train], head, alpha)
    elif head == 'pls':
        blocks = [fit_block(name, bank[name][train], preprocessing) for name in view.split('+')]
        transform = ViewTransform(blocks, None)
        x = transform.transform(bank, train)
        if np.max(np.std(x, axis=0)) < 1e-12:
            model = Ridge(alpha=1).fit(x, target[train])
        else:
            centered = x - x.mean(axis=0)
            centered_y = target[train] - target[train].mean()
            covariance = centered.T @ centered_y
            roundoff = (np.finfo(float).eps * max(centered.shape)
                        * np.linalg.norm(centered) * np.linalg.norm(centered_y))
            if np.linalg.norm(covariance) <= roundoff:
                # No linear target direction exists. NIPALS otherwise divides
                # by a zero score norm, or amplifies BLAS roundoff into a fit.
                model = DummyRegressor(strategy='mean').fit(x, target[train])
                return transform, model
            # Binary validity flags can contain several identical columns.
            # Their numerical rank, not column count, limits usable PLS scores.
            rank = int(np.linalg.matrix_rank(centered))
            components = min(value, len(train)-1, x.shape[1], rank)
            # Preprocessing already specifies scaling/block weights; PLS must not undo it.
            model = PLSRegression(n_components=components, scale=False, max_iter=500).fit(x, target[train])
    else:
        raise ValueError(head)
    return transform, model


def ridge_readout(x, y, head, alpha):
    if head == 'pca':
        return Ridge(alpha=alpha).fit(x, y)
    # Fixed training-only scale rule, no bandwidth search. Center the target so
    # vanishing kernel similarity has a source-mean fallback rather than zero.
    variance = x.shape[1] * x.var()
    gamma = 1 / variance if variance > 0 else 1.0
    return TransformedTargetRegressor(
        regressor=KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma),
        transformer=StandardScaler(with_std=False)).fit(x, y)


def select_statistical(bank, view, train, target, strata, methods, seed, head):
    candidates = (list(itertools.product(methods['pca_caps'], sorted(methods['ridge_alpha_grid'], reverse=True)))
                  if head in ('pca','rbf') else [(k,None) for k in methods['pls_components']])
    folds = [(train[a],train[b]) for a,b in StratifiedKFold(2,shuffle=True,random_state=seed).split(train,strata[train])]
    scores = [[] for _ in candidates]
    for fit, val in folds:
        cached = {}
        for ci,(value,alpha) in enumerate(candidates):
            if head in ('pca','rbf'):
                # Ridge strength does not change preprocessing/PCA. Reuse the
                # identical fold transform instead of recomputing the large SVD.
                if value not in cached:
                    transform, x = fit_view(bank, view, fit,
                                           {**methods['preprocessing'], 'pca_dimensions': value})
                    cached[value] = x, transform.transform(bank, val)
                x, vx = cached[value]
                model = ridge_readout(x, target[fit], head, alpha)
            else:
                transform, model = fit_statistical(bank,view,fit,target,methods['preprocessing'],head,value,alpha)
                vx = transform.transform(bank,val)
            pred=np.clip(model.predict(vx).reshape(-1),0,1)
            scores[ci].append(float(abs(pred-target[val]).mean()))
    chosen=min(range(len(candidates)),key=lambda i:np.mean(scores[i]))
    return candidates[chosen],dict(candidates=[dict(components=k,alpha=a,MAE=s) for (k,a),s in zip(candidates,scores)],
                                   folds=[dict(fit=a.tolist(),validation=b.tolist()) for a,b in folds])


def select_reference(values, train, target, strata, seed):
    """Choose ridge fraction via calibrated source validation; no target-family labels."""
    cv=StratifiedKFold(2,shuffle=True,random_state=seed)
    scores=[[] for _ in range(values.shape[1])]
    folds=[]
    for a,b in cv.split(train,strata[train]):
        fit,val=train[a],train[b]
        for col in range(values.shape[1]):
            model=IsotonicRegression(increasing='auto',out_of_bounds='clip').fit(values[fit,col],target[fit])
            scores[col].append(float(abs(model.predict(values[val,col])-target[val]).mean()))
        folds.append(dict(fit=fit.tolist(),validation=val.tolist()))
    # Stronger regularization wins exact ties.
    col=min(reversed(range(len(scores))),key=lambda i:np.mean(scores[i]))
    model=IsotonicRegression(increasing='auto',out_of_bounds='clip').fit(values[train,col],target[train])
    return col,model,dict(candidates=scores,folds=folds)
