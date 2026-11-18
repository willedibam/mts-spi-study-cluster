import json, numpy as np, style, matplotlib.pyplot as plt
d = json.load(open('surr_pairs_nt.json')); rows = d['rows']; res = d['res']
unit = np.array([r['unit'] for r in rows]); agent = np.array([r['agent'] for r in rows]); state = np.array([r['state'] for r in rows])
def contrast(v, s, ag=None):
    out = []
    for u in np.unique(unit):
        if ag and agent[unit == u][0] != ag: continue
        a, c = (unit == u) & (state == s), (unit == u) & (state == 'awake_ec')
        if a.any() and c.any(): out.append(np.nanmean(v[a]) - np.nanmean(v[c]))
    return np.array(out)
real = np.array([x['real']['gauss_vs_ksg'] for x in res]); surr = np.array([x['spectrum']['gauss_vs_ksg'] for x in res])
groups = [('eyes open', 'awake_eo', None, '#969696'), ('sleep', 'sleep', 'SLP', '#2171b5'), ('ketamine', 'drug', 'KT', '#238b45'),
          ('medetomidine', 'drug', 'MD', '#6a51a3'), ('ket-med', 'drug', 'KTMD', '#cb181d'), ('propofol', 'drug', 'PF', '#fd8d3c')]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
for ax, (v, title) in zip(axes, ((real, 'real'), (real - surr, 'real − surrogate'))):
    eo = contrast(v, 'awake_eo'); ax.axhspan(eo.mean() - 2 * eo.std(ddof=1), eo.mean() + 2 * eo.std(ddof=1), color='#f0f0f0', zorder=0)
    ax.axhline(0, color='#bdbdbd', lw=.8, zorder=1)
    for k, (lab, s, ag, col) in enumerate(groups):
        c = contrast(v, s, ag); xs = k + np.linspace(-.15, .15, len(c))
        ax.plot(xs, c, 'o', color=col, ms=3.5, mec='none', alpha=.8); ax.plot([k - .25, k + .25], [c.mean()] * 2, color=col, lw=1.7)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], rotation=35, ha='right'); ax.set_title(title)
axes[0].set_ylabel(r'$\Delta z$(Gaussian MI, kNN MI) vs own wake')
fig.savefig('fig/form_gauss_ksg.png'); fig.savefig('fig/form_gauss_ksg.svg')
