"""Bipolar 10-s banks from the common-average banks, and a gate against a re-cut from the raw archives.

  python bank.py derive <CAR bank dir> <pairs.json> <out bank dir>
  python bank.py gate <CAR bank dir> <pairs.json> <raw archive dir> <unit> [<unit> ...]
  python bank.py mapcheck <pairs.json> <CAR bank dir> [<CAR bank dir> ...]

derive: for every row, channel e_i − e_j of each of the animal's 32 pairs, taken from the CAR window columns of
those electrodes (column = position in the unit's kept_channels). The common average is subtracted from every
channel and the filters of prepare.py are linear and per-channel, so this equals the bipolar derivation of the
filtered raw data exactly (to float rounding) for any pair of kept electrodes. Rows, ids, start_s, order and the
stage split are those of the CAR bank; M = 32; nothing is re-screened.

gate: for every row of the named units, re-cut from the raw archive with prepare.continuous on each pair electrode
(the long-arm pass 1: nan_to_num, 50 Hz notch, 0.5 Hz high-pass, 40 Hz low-pass, 1000 -> 100 Hz), sample
k = round(start_s * 100), 1000 samples, and difference; prints max |derived − re-cut| / SD per unit.
"""
import sys, io, json, zipfile, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # research/representation/atlas, for prepare.py
import prepare as P

L = int(P.LONG * P.FS)


def load(bank, pairs):
    man = json.load(open(f'{bank}/manifest.json')); W = np.load(f'{bank}/windows.npz')
    kept = {u['unit']: u['kept_channels'] for u in man['preprocessing']['units']}
    PR = {a: v['pairs'] for a, v in json.load(open(pairs))['animals'].items()}
    return man, W, kept, PR


def columns(kept, pairs):
    ix = {e: c for c, e in enumerate(kept)}
    missing = sorted({e for p in pairs for e in p} - set(ix))
    if missing: raise ValueError(f'pair electrodes not kept: {missing}')
    return np.array([ix[i] for i, _ in pairs]), np.array([ix[j] for _, j in pairs])


def derive(bank, pairs, out):
    man, W, kept, PR = load(bank, pairs); arrays, rows = {}, []
    for r in man['rows']:
        a, b = columns(kept[r['unit']], PR[r['animal']]); w = W[r['id']].astype(np.float64)
        arrays[r['id']] = w[:, a] - w[:, b]; rows.append(dict(r, M=len(a)))
    units = [dict(u, kept_channels=[f'{i}-{j}' for i, j in PR[next(r['animal'] for r in man['rows'] if r['unit'] == u['unit'])]],
                  car_kept_channels=u['kept_channels']) for u in man['preprocessing']['units']
             if any(r['unit'] == u['unit'] for r in man['rows'])]
    rec = dict(man['preprocessing'], units=units, reference='bipolar: lower − higher electrode of 32 nearest-neighbour pairs per animal (reference/pairs.json), from the CAR bank',
               derived_from=str(bank), pairs_json=json.load(open(pairs)))
    P.write_bank(Path(out), man['dataset'], man['arm'], man['T'], rows, arrays, rec)


def gate(bank, pairs, raw, units):
    man, W, kept, PR = load(bank, pairs); worst = 0.0
    for unit in units:
        rows = [r for r in man['rows'] if r['unit'] == unit]; pr = PR[rows[0]['animal']]
        a, b = columns(kept[unit], pr); E = sorted({e for p in pr for e in p}); gaps = []
        with zipfile.ZipFile(f'{raw}/{unit}.zip') as z:
            full = {n.rsplit('/', 2)[-2]: n.rsplit('/', 1)[0] for n in z.namelist() if n.endswith('/Condition.mat')}
            for s in sorted({r['session'] for r in rows}):
                low = {e: P.continuous(np.nan_to_num(P.mat_vector(z.read(f'{full[s]}/ECoG_ch{e}.mat'))[:, None]), P.NT_FS, P.NT_MAINS)[:, 0] for e in E}
                for r in (r for r in rows if r['session'] == s):
                    k = int(round(r['start_s'] * P.FS)); w = W[r['id']].astype(np.float64)
                    cut = np.column_stack([low[i][k:k + L] - low[j][k:k + L] for i, j in pr])
                    der = w[:, a] - w[:, b]; gaps.append(np.abs(der - cut).max() / der.std())
        worst = max(worst, max(gaps)); print(f'{unit}: {len(rows)} windows, max |derived − re-cut| / SD = {max(gaps):.2e}', flush=True)
    print('identity gate:', 'PASS' if worst < 1e-6 else 'FAIL', f'(worst {worst:.2e}, threshold 1e-6)')


def mapcheck(banks, pairs):
    """Drug-blind check that map electrode n is ECoG_ch{n}: on awake eyes-closed CAR windows only, the Spearman rho
    between map distance and the window-mean correlation of each electrode pair (electrodes kept in every unit of the
    animal), and the mean correlation of the 32 selected pairs against all pairs. Gate: in every animal the selected
    pairs' mean r exceeds the 90th percentile of r over all electrode pairs (rho is reported, not gated: the common
    average flattens distance decay)."""
    from scipy.stats import spearmanr
    A = json.load(open(pairs))['animals']; R, worst = {}, 1.0
    for bank in banks:
        man, W, kept, _ = load(bank, pairs)
        for r in (r for r in man['rows'] if r['state'] == 'awake_ec'):
            R.setdefault(r['animal'], []).append((kept[r['unit']], np.corrcoef(W[r['id']].T)))
    for a, L in sorted(R.items()):
        E = sorted(set.intersection(*[set(k) for k, _ in L]))
        C = np.mean([c[np.ix_(*[[k.index(e) for e in E]] * 2)] for k, c in L], 0)
        xy = np.array(A[a]['xy'])[np.array(E) - 1]; D = np.linalg.norm(xy[:, None] - xy[None], axis=-1); iu = np.triu_indices(len(E), 1)
        rho = spearmanr(D[iu], C[iu]).statistic; ix = {e: n for n, e in enumerate(E)}
        pr = np.array([C[ix[i], ix[j]] for i, j in A[a]['pairs']]); sel = pr.mean(); pct = (C[iu] < sel).mean(); worst = min(worst, pct)
        low = np.argsort(pr)[:3]; print(f'  lowest pair r: ' + ', '.join(f"{A[a]['pairs'][q][0]}-{A[a]['pairs'][q][1]} {pr[q]:+.2f} (pct {(C[iu] < pr[q]).mean():.2f})" for q in low))
        print(f'{a}: {len(L)} awake_ec windows, {len(E)} electrodes; rho(distance, r) = {rho:+.2f}; mean r selected pairs {sel:+.2f} '
              f'(percentile {pct:.3f} of all pairs; all-pair mean {C[iu].mean():+.2f}, 90th pct {np.quantile(C[iu], .9):+.2f})')
    print('map gate:', 'PASS' if worst > .9 else 'FAIL', f'(lowest percentile {worst:.3f}, threshold .90)')


if __name__ == '__main__':
    if sys.argv[1] == 'derive': derive(*sys.argv[2:5])
    elif sys.argv[1] == 'mapcheck': mapcheck(sys.argv[3:], sys.argv[2])
    elif sys.argv[1] == 'gate': gate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
