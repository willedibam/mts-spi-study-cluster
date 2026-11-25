"""Post hoc, written after the ke_*.npz outputs were read (26 November): why the net against 'im' keeps a drug effect while KZ does
not, and whether the fourth-order excess over the matched null is present in every state. Descriptive only; not in the prereg.
Per animal and state (drug pooled over agents): window means of KZ (k04 z against 'im'), the k04 tail fraction (chance 3 / (B + 1)),
the net k04 against the 'im' mean, and the mean log 'im' SD of k04; then the drug and sleep changes from each session's eyes-closed mean.
  python kens_posthoc.py <dir with ke_[AB]_*.npz>
"""
import sys, glob, numpy as np, pandas as pd


def main(d):
    F = []
    for f in sorted(glob.glob(f'{d}/ke_[AB]_*.npz')):
        q = np.load(f); real, mu, sd = (q[k][:, 3].astype(float) for k in ('real', 'im_mean', 'im_sd'))
        F.append(pd.DataFrame(dict(unit=q['unit'], state=q['state'], animal=q['animal'], KZ=((real - mu) / sd).mean(1),
                                   TAIL=(q['im_below'][:, 3] >= int(q['B']) - 2).mean(1), net=(real - mu).mean(1), logsd=np.log(sd).mean(1))))
    df = pd.concat(F, ignore_index=True); B = int(q['B']); cols = ['KZ', 'TAIL', 'net', 'logsd']
    print(f'levels by animal and state (window means; TAIL chance {3 / (B + 1):.3f})')
    print(df.groupby(['animal', 'state'])[cols].mean().round(3).to_string())
    for c in cols:
        df['d_' + c] = df[c] - df.unit.map(df[df.state == 'awake_ec'].groupby('unit')[c].mean())
    U = df.groupby(['unit', 'animal', 'state'])[['d_' + c for c in cols]].mean().reset_index()
    print('\nchange from the session eyes-closed mean (unit means, then animal means)')
    print(U[U.state.isin(['drug', 'sleep'])].groupby(['state', 'animal'])[['d_' + c for c in cols]].mean().round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1])
