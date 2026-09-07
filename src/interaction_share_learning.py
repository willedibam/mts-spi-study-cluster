"""Small training-only readouts for the focused interaction-share comparison."""
import itertools
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from src.representation_screen import fit_view, fit_block, ViewTransform


def fit_statistical(bank, view, train, target, preprocessing, head, value, alpha=None):
    if head == 'pca':
        transform, x = fit_view(bank, view, train, {**preprocessing, 'pca_dimensions':value})
        model = Ridge(alpha=alpha).fit(x, target[train])
    elif head == 'pls':
        blocks = [fit_block(name, bank[name][train], preprocessing) for name in view.split('+')]
        transform = ViewTransform(blocks, None)
        x = transform.transform(bank, train)
        if np.max(np.std(x, axis=0)) < 1e-12:
            model = Ridge(alpha=1).fit(x, target[train])
        else:
            components = min(value, len(train)-1, x.shape[1])
            # Preprocessing already specifies scaling/block weights; PLS must not undo it.
            model = PLSRegression(n_components=components, scale=False, max_iter=500).fit(x, target[train])
    else:
        raise ValueError(head)
    return transform, model


def select_statistical(bank, view, train, target, strata, methods, seed, head):
    candidates = (list(itertools.product(methods['pca_caps'], sorted(methods['ridge_alpha_grid'], reverse=True)))
                  if head == 'pca' else [(k,None) for k in methods['pls_components']])
    folds = [(train[a],train[b]) for a,b in StratifiedKFold(2,shuffle=True,random_state=seed).split(train,strata[train])]
    scores = [[] for _ in candidates]
    for fit, val in folds:
        for ci,(value,alpha) in enumerate(candidates):
            transform, model = fit_statistical(bank,view,fit,target,methods['preprocessing'],head,value,alpha)
            pred=np.clip(model.predict(transform.transform(bank,val)).reshape(-1),0,1)
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
