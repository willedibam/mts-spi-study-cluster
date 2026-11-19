"""Declared null scales and decomposition for the PRIOS rows (prereg, "PRIOS sensitivity set"); primary (CAR) variant only.

    python nulls.py <z.npz> [<z.npz> ...]

Per run: subject Δ; the frozen split-half null SD and its robust version (1.4826 × MAD across subjects); d_z = mean Δ / SD(Δ);
Δ split into the change in real z and in mean surrogate z (Δ = Δreal − Δsurr).
"""
import sys, json, numpy as np
from scipy.stats import ttest_1samp


def report(fn, need=20):
    d = np.load(fn); Z = d['Z'][:, 0]; rows = json.loads(str(d['rows']))
    subj = np.array([r['subject'] for r in rows]); state = np.array([r['state'] for r in rows])
    real, surr = Z[:, 0], Z[:, 1:].mean(1); res = real - surr
    keep = [s for s in np.unique(subj) if min(((subj == s) & (state == st)).sum() for st in ('awake', 'anaes')) >= need]
    m = lambda x, s, st: x[(subj == s) & (state == st)].mean()
    D = np.array([m(res, s, 'anaes') - m(res, s, 'awake') for s in keep])
    Dr = np.array([m(real, s, 'anaes') - m(real, s, 'awake') for s in keep]); Ds = Dr - D
    half = []
    for s in keep:
        a = res[(subj == s) & (state == 'awake')]; h = len(a) // 2; half.append(a[:h].mean() - a[h:2 * h].mean())
    half = np.array(half); sd = half.std(ddof=1); rsd = 1.4826 * np.median(np.abs(half - np.median(half)))
    print(f'{fn}\n  included {len(keep)}: ' + ', '.join(f'{s[4:]} {x:+.4f} (split {h:+.4f})' for s, x, h in zip(keep, D, half)))
    print(f'  mean Δ {D.mean():+.4f}, {(D < 0).sum()}/{len(D)} negative, one-sided p {ttest_1samp(D, 0, alternative="less").pvalue:.4f}; '
          f'null SD {sd:.4f} -> {D.mean() / sd:+.1f}; robust null SD {rsd:.4f} -> {D.mean() / rsd:+.1f}; d_z {D.mean() / D.std(ddof=1):+.2f}')
    print(f'  Δ real z {Dr.mean():+.4f} ({(Dr < 0).sum()}/{len(D)} negative), Δ surrogate z {Ds.mean():+.4f}; '
          f'real z awake {np.mean([m(real, s, "awake") for s in keep]):.3f}')


if __name__ == '__main__':
    for f in sys.argv[1:]: report(f)
