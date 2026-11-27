"""Freeze-record numbers from the baseline (wake) state only (transfer protocol: window rule, reliability floor, null).

    python baseline.py <root> <baseline out dir> [<plan target counts from: score.py plan>]

Reads only wake windows (score.py run ... wake:1min) and metadata. Prints: simultaneity check (median over subjects of the
largest aligned |r| of channel 0 against the others, against the same with channel 0 shifted 3 s); median integrated
autocorrelation time tau (samples, 1 + 2 * sum of the ACF up to its first non-positive lag) and T / (2 tau); reliability of the
G and K profiles (mean pairwise profile r between windows of a subject, averaged over subjects); wake real z (ceiling rule);
the split-half null SD (protocol scaling to N3 and N2 window counts, plain SD; and the PRIOS construction: unscaled,
first minus second half, 1.4826 MAD); and the forecast 90% half-width of the subject-mean contrast under no change.
"""
import sys, os, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, '..')]
from extract import windows, recording, segments
from score import plan, STATES
from scipy.stats import t as tdist


def tau(x):
    out = []
    for c in x.T:
        c = c - c.mean(); f = np.fft.rfft(c, 2 * len(c)); a = np.fft.irfft(f * np.conj(f))[:len(c)]; a /= a[0]
        k = np.argmax(a <= 0) if (a <= 0).any() else len(a); out.append(1 + 2 * a[1:k].sum())
    return np.median(out)


def halves(v):
    h = len(v) // 2; return v[:h], v[len(v) - h:]


def main(root, d):
    pl = plan(root).set_index('sub'); S = []
    for fn in sorted(glob.glob(f'{d}/sub-*.npz')):
        sub = os.path.basename(fn)[:-4]; q = np.load(fn); Z = q['Z']; P = q['P']; ch = json.loads(str(q['chans']))
        net = Z[:, 0] - Z[:, 1:].mean(1); fz = np.arctanh(Z[:, 0]) - np.arctanh(Z[:, 1:]).mean(1)
        rel = [np.mean(np.corrcoef(P[:, k])[np.triu_indices(len(P), 1)]) for k in (0, 1)]
        x, nm = recording(root, sub, 'wake', '1min'); a, n = segments(root, sub, 'wake', '1min')[0]; y = x[a:a + n][:, [nm.index(c) for c in ch]]
        al = np.abs(np.corrcoef(y.T)[0, 1:]).max(); sh = max(abs(np.corrcoef(np.roll(y[:, 0], 600), y[:, j])[0, 1]) for j in range(1, len(ch)))
        ta = np.median([tau(w) for w, _ in windows(root, sub, 'wake', '1min', ch)])
        nb = len(net); nh = nb // 2; f1, f2 = halves(net); g1, g2 = halves(fz)
        sc = lambda nt: np.sqrt((1 / nb + 1 / nt) / (2 / nh))
        S.append(dict(sub=sub, M=len(ch), nb=nb, relG=rel[0], relK=rel[1], z=Z[:, 0].mean(), zs=Z[:, 1:].mean(), al=al, sh=sh, tau=ta,
                      split=f2.mean() - f1.mean(), splitF=g2.mean() - g1.mean(), n3=pl.loc[sub, 'sleepN3:10min'], n2=pl.loc[sub, 'sleepN2:10min'],
                      sc3=sc(pl.loc[sub, 'sleepN3:10min']), sc2=sc(pl.loc[sub, 'sleepN2:10min'])))
    s = pd.DataFrame(S); n = len(s); q90 = tdist.ppf(.95, n - 1)
    print(f'subjects {n}; M median {s.M.median():.0f} (range {s.M.min()}-{s.M.max()}); wake windows {s.nb.value_counts().sort_index().to_dict()}')
    print(f'simultaneity: largest aligned |r| ch0 median {s.al.median():.3f}; shifted 3 s {s.sh.median():.3f}; subjects aligned > shifted {(s.al > s.sh).sum()}/{n}')
    print(f'tau median {s.tau.median():.1f} samples (IQR {s.tau.quantile(.25):.1f}-{s.tau.quantile(.75):.1f}); T/(2 tau) {1000 / (2 * s.tau.median()):.1f}')
    print(f'reliability G {s.relG.mean():.3f}, K {s.relK.mean():.3f} (floor .3); subjects below .3: G {(s.relG < .3).sum()}, K {(s.relK < .3).sum()}')
    print(f'wake real z mean {s.z.mean():.3f} (median {s.z.median():.3f}, max {s.z.max():.3f}); surrogate z {s.zs.mean():.3f}; subjects with real z >= .98: {(s.z >= .98).sum()}')
    for lab, col, scc in (('N3', 'split', 'sc3'), ('N2', 'split', 'sc2'), ('N3 Fisher', 'splitF', 'sc3')):
        v = s[col] * s[scc]; sd = v.std(ddof=1); u = s[col]; rsd = 1.4826 * np.median(np.abs(-u - np.median(-u)))
        print(f'{lab}: null SD (protocol, scaled) {sd:.4f}; mean split {s[col].mean():+.4f}; PRIOS construction (unscaled, MAD) {rsd:.4f}; '
              f'forecast 90% half-width of subject mean under no change {q90 * sd / np.sqrt(n):.4f} = {q90 / np.sqrt(n):.2f} null SD')
    s.to_csv(f'{d}/baseline.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
