"""Per-patient table and post hoc descriptives for human-sleep-result.md (written after the frozen result was read).

    python table.py <windows dir> <result.json>

Table (frozen outputs only): per patient centre, channels, window counts, wake real z, and the N3 and N2 Δ of result.json, raw
and in each contrast's null SD. Post hoc (not in the prereg): per-state median delta fraction and mean |r| over patients, and
the correlation over patients of the N3 Δ with the change in delta fraction and in mean |r|.
"""
import sys, os, glob, json, numpy as np

ST = ['wake:1min', 'sleepN2:10min', 'sleepN3:10min']


def main(d, res):
    r = json.load(open(res)); D3, D2 = r['N3']['subjects'], r['N2']['subjects']; s3, s2 = r['N3']['null_sd'], r['N2']['null_sd']; P = []
    for fn in sorted(glob.glob(f'{d}/sub-*.npz')):
        q = np.load(fn); rows = json.loads(str(q['rows'])); ch = json.loads(str(q['chans'])); st = np.array([x['state'] for x in rows])
        m = {s: st == s for s in ST}; sub = os.path.basename(fn)[:-4]
        P.append(dict(sub=sub, c=ch[0][0], M=len(ch), n={s: int(m[s].sum()) for s in ST}, z=q['Z'][m[ST[0]], 0].mean(),
                      delta={s: np.median(q['C'][m[s], 3]) for s in ST if m[s].any()}, absr={s: q['C'][m[s], 0].mean() for s in ST if m[s].any()}))
    f = lambda D, k, sd: f'{D[k]:+.4f} | {D[k] / sd:+.2f}' if k in D else '– | –'
    print('| Patient | Centre | M | Windows W/N2/N3 | Wake z | N3 Δ raw | N3 Δ SD | N2 Δ raw | N2 Δ SD |\n|---|---|---|---|---|---|---|---|---|')
    for p in P:
        print(f"| {p['sub']} | {p['c']} | {p['M']} | {'/'.join(str(p['n'][s]) for s in ST)} | {p['z']:.3f} | {f(D3, p['sub'], s3)} | {f(D2, p['sub'], s2)} |")
    print('\npost hoc:')
    for s in ST:
        v = [p['delta'][s] for p in P if s in p['delta']]; a = [p['absr'][s] for p in P if s in p['absr']]
        print(f'  {s}: delta fraction median {np.median(v):.3f} (IQR {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}); mean |r| median {np.median(a):.3f}')
    p3 = [p for p in P if p['sub'] in D3]; y = [D3[p['sub']] for p in p3]
    for k in ('delta', 'absr'):
        x = [p[k][ST[2]] - p[k][ST[0]] for p in p3]; print(f'  corr over patients of N3 Δ with change in {k}: {np.corrcoef(x, y)[0, 1]:+.2f} (n {len(y)})')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
