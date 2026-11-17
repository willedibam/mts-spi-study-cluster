import json, numpy as np, warnings; warnings.filterwarnings('ignore')
d = json.load(open('surr_pairs_nt.json')); rows = d['rows']; res = d['res']
unit = np.array([r['unit'] for r in rows]); agent = np.array([r['agent'] for r in rows]); state = np.array([r['state'] for r in rows])
AG = {'KTMD': 'ket-med', 'KT': 'ketamine', 'MD': 'medetom', 'PF': 'propofol', 'SLP': 'sleep'}
def get(v, k): return np.array([x[v][k] for x in res])
def adjust(y, X):
    m = np.isfinite(y); yy, XX = y.copy(), X.copy()
    for u in np.unique(unit):
        for s in np.unique(state[unit == u]):
            g = m & (unit == u) & (state == s); yy[g] -= yy[g].mean(); XX[g] -= XX[g].mean(0)
    return y - X @ np.linalg.lstsq(XX[m], yy[m], rcond=None)[0]
def contrast(v, s, ag=None):
    out = []
    for u in np.unique(unit):
        if ag and agent[unit == u][0] != ag: continue
        a, c = (unit == u) & (state == s), (unit == u) & (state == 'awake_ec')
        if a.any() and c.any(): out.append(np.nanmean(v[a]) - np.nanmean(v[c]))
    return np.array(out)
ctrl = np.c_[get('real', 'mean_abs_r'), get('real', 'sd_abs_r')]
for key, lab in (('gauss_vs_ksg', 'Gaussian MI vs KSG MI (Theiler)'), ('envelope_orth', 'envelope: all vs orthogonalised'), ('coh_vs_icoh', '|coherency| vs |imag coherency|')):
    print(f'\n### {lab}   (16-channel subset; Δ vs own awake eyes-closed; units positive/n)')
    print(f"{'':22s} awake  " + ' '.join(f'{AG[a]:>17s}' for a in AG) + '   EO null')
    for nm, v in (('real', get('real', key)), ('surr marginal', get('marginal', key)), ('surr spectrum', get('spectrum', key)),
                  ('real-surr marg', get('real', key) - get('marginal', key)), ('real-surr spec', get('real', key) - get('spectrum', key)),
                  ('real-surr spec, adj', adjust(get('real', key) - get('spectrum', key), ctrl))):
        eo = contrast(v, 'awake_eo'); sd = np.nanstd(eo, ddof=1); cells = []
        for a in AG:
            c = contrast(v, 'sleep' if a == 'SLP' else 'drug', a); cells.append(f"{np.nanmean(c):+.3f}({np.nanmean(c)/sd:+5.1f},{int((c>0).sum())}/{len(c)})")
        print(f"{nm:22s} {np.nanmean(v[state=='awake_ec']):+.2f}  " + ' '.join(f'{x:>17s}' for x in cells) + f"  {np.nanmean(eo):+.3f}±{sd:.3f}")
    v = get('real', key) - get('spectrum', key)
    print('   recovery (real-surr spec) as fraction of drug shift: ' + ' '.join(f"{AG[a]} {np.nanmean(contrast(v,'recovery_ec',a))/np.nanmean(contrast(v,'drug',a)):+.2f}" for a in ('KTMD','KT','MD','PF')))
