"""Post hoc (not in the robust prereg): within-session slope of the anchor drug Δz against window start time.

Δz per window as plot_character_draft.load (zns residual minus its session's mean awake-EC residual); start times from
figwin/start_s.csv (bank manifests). Both live in the main checkout's results/representation/atlas-mi-pair-261117/.
Gate: the session-pooled drug Δz must reproduce the anchor (-.0133).

  python robust_placement.py <atlas-mi-pair results root>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from plot_character_draft import load


def main(root):
    root = Path(root); df, _ = load(root)
    sv = pd.read_csv(root / 'figwin/start_s.csv')
    assert (sv.groupby('id').start_s.nunique() == 1).all()
    df['tm'] = df.id.map(sv.drop_duplicates('id').set_index('id').start_s) / 60
    dr = df[df.state == 'drug'].copy(); assert dr.tm.notna().all()
    print('gate: session-pooled drug Δz %.5f (anchor -.0133), %d windows' % (dr.groupby('unit').d.mean().mean(), len(dr)))
    S = dr.groupby('unit').apply(lambda g: pd.Series(dict(
        animal=g.animal.iloc[0], agent=g.group.iloc[0], span=g.tm.max() - g.tm.min(), mean=g.d.mean(),
        slope=10 * np.polyfit(g.tm, g.d, 1)[0])))
    print(S.round(4).to_string())

    def report(x, s, tag):
        c = lambda v: v - v.groupby(x.unit).transform('mean')
        pooled = 10 * (c(x.tm) * c(x.d)).sum() / (c(x.tm) ** 2).sum()
        A = s.groupby('animal').slope.mean(); t = ttest_1samp(A, 0)
        print('%s: pooled within-session %+.5f per 10 min; animal mean %+.5f, %d/4 > 0, t(3) %.2f, p(two-sided) %.3f'
              % (tag, pooled, A.mean(), (A > 0).sum(), t.statistic, t.pvalue))
    report(dr, S, 'all sessions')
    k = ~S.index.str.startswith('20120730pf')
    report(dr[dr.unit.isin(S.index[k])], S[k], 'without Chibi PF 0730')
    print(S.groupby('agent')[['slope', 'mean', 'span']].mean().round(5))
    print('sessions with slope < 0: %d/%d' % ((S.slope < 0).sum(), len(S)))


if __name__ == '__main__':
    main(sys.argv[1])
