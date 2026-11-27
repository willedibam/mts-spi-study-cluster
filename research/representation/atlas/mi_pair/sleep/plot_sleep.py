"""Figure for human-sleep-result.md: panel (c) of the MI figure extended with human rows, the frozen decision in null-SD units,
and depth (delta fraction) across species. Written after the frozen result was read; panel c is post hoc.

    python plot_sleep.py <macaque zns dir> <PRIOS S-full z.npz> <sleep result.json> <sleep windows dir or table json> <out stem>

a  Δ(z_observed − z_surrogate), raw, state minus eyes-closed wake: macaque as panel (c) of plot_character.py (dot per animal,
   tick = session mean); human PRIOS propofol (S-full, dot per patient, as human/nulls.py); human N2 and N3 (10-min set; dot
   per patient, tick = mean, bar = t 90% CI). Shaded: the raw equivalence margin ±.0021.
b  The frozen decision: human patient Δ in each contrast's null SD (10-min and processing-matched 1-min rows), mean and 90% CI;
   shaded ±.675 (the margin); reference lines at the PRIOS propofol S-full effect (−2.7 robust null SD), the macaque drug session
   mean (−3.29) and the macaque sleep session mean (−0.24, eyes-open SD). Units differ across datasets.
c  Post hoc: unit Δ (as a) against the rise in delta fraction (prepare.delta_fraction, 0.5–4 Hz share of 0.5–40 Hz at 100 Hz)
   over that unit's own wake: macaque sessions (vs eyes-closed wake), human patients (N2, N3 vs wake). Rises are within unit, so montage level cancels.
"""
import sys, os, json, glob
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from scipy import stats
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE.parent)]
from plot_character import read, session_effects, BLUE, ORANGE, GREY
from plot_character_draft import load

GREEN = '#3C7A4A'
MAC = (('KT', 'Ketamine'), ('KTMD', 'Ket. + med.'), ('MD', 'Medetomidine'), ('PF', 'Propofol'), ('SLP', 'Sleep'))


def prios(fn, need=20):
    d = np.load(fn); Z = d['Z'][:, 0]; rows = json.loads(str(d['rows']))
    subj = np.array([r['subject'] for r in rows]); st = np.array([r['state'] for r in rows]); res = Z[:, 0] - Z[:, 1:].mean(1)
    keep = [s for s in np.unique(subj) if min(((subj == s) & (st == x)).sum() for x in ('awake', 'anaes')) >= need]
    return np.array([res[(subj == s) & (st == 'anaes')].mean() - res[(subj == s) & (st == 'awake')].mean() for s in keep])


def ci(v):
    v = np.asarray(v); h = stats.t.ppf(.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return v.mean(), v.mean() - h, v.mean() + h


def human_delta(src):
    if src.endswith('.json'): return json.load(open(src))
    out = {}
    for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
        q = np.load(fn); st = np.array([r['state'] for r in json.loads(str(q['rows']))])
        out[os.path.basename(fn)[:-4]] = {s: float(np.median(q['C'][st == s, 3])) for s in set(st)}
    return out


def draw(zdir, pfn, rfn, src, out):
    zdir = Path(zdir); df, _ = read(zdir); ses = session_effects(df); R = json.load(open(rfn)); P = prios(pfn)
    dl, _ = load(zdir.parent, zdir); H = human_delta(src); rng = np.random.default_rng(0)
    style = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "mathtext.fontset": "cm",
             "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.0, 3.1), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1, .95]})
        # a: raw residual change, macaque then human
        a.axvspan(-R['baseline']['margin_raw'], R['baseline']['margin_raw'], color='#E9ECEE', lw=0, zorder=0)
        a.axvline(0, color=GREY, lw=.7, ls='--', zorder=0)
        labels = []
        for j, (g, lab) in enumerate(MAC):
            part = ses[ses.agent == g]
            for _, an in part.groupby('animal'): a.scatter(an.residual.mean(), j, color=BLUE, alpha=.72, s=22, lw=0)
            a.scatter(part.residual.mean(), j + .2, marker='|', s=65, color='black'); labels.append(lab)
        hum = [('Propofol', P, ORANGE), ('N2 sleep', np.array(list(R['N2']['subjects'].values())), GREEN),
               ('N3 sleep', np.array(list(R['N3']['subjects'].values())), GREEN)]
        for i, (lab, v, col) in enumerate(hum):
            j = len(MAC) + .6 + i; m, lo, hi = ci(v)
            a.scatter(v, j + rng.uniform(-.17, .17, len(v)), color=col, alpha=.45 if len(v) > 10 else .8, s=7 if len(v) > 10 else 18, lw=0)
            a.plot([lo, hi], [j + .3, j + .3], color='black', lw=1); a.scatter(m, j + .3, marker='|', s=65, color='black'); labels.append(lab)
        a.axhline(len(MAC) - .2, color=GREY, lw=.5)
        a.set_yticks(list(range(len(MAC))) + [len(MAC) + .6 + i for i in range(3)], labels); a.set_ylim(len(MAC) + 3.0, -.65)
        a.text(-.36, .985, 'Macaque', transform=a.transAxes, fontsize=7, color=BLUE, va='top', ha='left', weight='bold')
        a.text(-.36, .40, 'Human', transform=a.transAxes, fontsize=7, color=GREEN, va='top', ha='left', weight='bold')
        a.set_xlim(-.065, .035)
        a.set_xlabel(r'$\Delta(z_{\rm observed}-z_{\rm surrogate})$, state $-$ wake')
        # b: frozen decision in null SD
        ms = R['baseline']['margin_sd']; b.axvspan(-ms, ms, color='#E9ECEE', lw=0, zorder=0); b.axvline(0, color=GREY, lw=.7, ls='--', zorder=0)
        rows = [('N3', 'N3, 10 min'), ('N3_1min', 'N3, 1 min'), ('N2', 'N2, 10 min'), ('N2_1min', 'N2, 1 min')]
        for j, (k, lab) in enumerate(rows):
            v = np.array(list(R[k]['subjects'].values())) / R[k]['null_sd']; m, lo, hi = ci(v)
            b.scatter(v, j + rng.uniform(-.18, .18, len(v)), color=GREEN, alpha=.4, s=6, lw=0)
            b.plot([lo, hi], [j + .32, j + .32], color='black', lw=1.2); b.scatter(m, j + .32, marker='|', s=65, color='black')
        for x, lab, col, ha in ((-2.7, 'PRIOS propofol', ORANGE, 'left'), (-3.29, 'macaque drugs', BLUE, 'right'), (-.24, 'macaque sleep', BLUE, 'right')):
            b.axvline(x, color=col, lw=.8, ls=':' if 'sleep' in lab else '-', zorder=0)
            b.text(x + (.12 if ha == 'left' else -.12), -1.45, lab, rotation=90, fontsize=6, color=col, ha=ha, va='top')
        b.set_yticks(range(len(rows)), [l for _, l in rows]); b.set_ylim(len(rows) - .4, -1.5); b.set_xlim(-8, 6)
        b.set_xlabel('Patient $\\Delta$ (wake split-half null SD)')
        # c: post hoc, Δ against the within-unit rise in delta fraction (state minus that unit's own wake)
        dm = dl.groupby(['unit', 'group']).delta.mean().unstack(); rise = {g: (dm[g] - dm['awake_ec']).dropna() for g in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
        c.axhline(0, color=GREY, lw=.7, ls='--', zorder=0)
        for k, key, face, lab in (('N2', 'sleepN2:10min', 'white', 'Human N2'), ('N3', 'sleepN3:10min', GREEN, 'Human N3')):
            sub = [s for s in R[k]['subjects'] if key in H[s] and 'wake:1min' in H[s]]
            c.scatter([H[s][key] - H[s]['wake:1min'] for s in sub], [R[k]['subjects'][s] for s in sub], s=9, facecolor=face, edgecolor=GREEN, lw=.6, alpha=.7, label=lab)
        md = ses[ses.agent != 'SLP']; sl = ses[ses.agent == 'SLP']
        c.scatter([rise[g][u] for g, u in zip(md.agent, md.unit)], md.residual, s=16, color=BLUE, lw=0, label='Macaque drugs')
        c.scatter([rise['SLP'][u] for u in sl.unit], sl.residual, s=16, facecolor='white', edgecolor=BLUE, lw=.9, label='Macaque sleep')
        c.set_xlabel('Rise in delta fraction over own wake'); c.set_ylabel(r'$\Delta(z_{\rm observed}-z_{\rm surrogate})$')
        c.legend(loc='lower right', fontsize=6.5, handletextpad=.1, borderaxespad=.2)
        for ax, t in zip((a, b, c), ('a   Beyond the spectral null (raw)', 'b   Frozen decision, human sleep', 'c   Against spectral depth (post hoc)')):
            ax.set_title(t, loc='left', fontsize=9, weight='bold', pad=9)
        for ext in ('pdf', 'png', 'svg'): fig.savefig(f'{out}.{ext}', dpi=220, bbox_inches='tight')
    print('PRIOS S-full', np.round(P, 4), 'mean', round(P.mean(), 4))
    print('macaque per group session mean', ses.groupby('agent').residual.mean().round(4).to_dict())
    x = np.array([rise[g][u] for g, u in zip(ses.agent, ses.unit)]); y = ses.residual.values; b1, b0 = np.polyfit(x, y, 1)
    for k, key in (('N2', 'sleepN2:10min'), ('N3', 'sleepN3:10min')):
        sub = [s for s in R[k]['subjects'] if key in H[s]]; xr = np.median([H[s][key] - H[s]['wake:1min'] for s in sub])
        print(f'macaque line (all 30 sessions) Δ = {b0:+.4f} {b1:+.4f}·rise, r {np.corrcoef(x, y)[0, 1]:+.2f}; at human {k} median rise {xr:.3f}: predicted {b0 + b1 * xr:+.4f}, observed {R[k]["mean_raw"]:+.4f}')
    print('within-unit delta rise, median', {g: round(float(v.median()), 3) for g, v in rise.items()})


if __name__ == '__main__':
    draw(*sys.argv[1:6])
