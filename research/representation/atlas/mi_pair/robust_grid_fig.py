"""Rate x window map of the MI-pair drug Δz (post hoc, descriptive). Reads the pooled rows of lowrate_summary.txt.

(a) N against window length per rate (grey), with N_eff = T / max(2 tau_1/e, 1/fs) for drug and awake; each computed arm at
    (window, N), coloured by pooled net Δz; running and proposed cells hollow.
(b) The same arms as a rate x window grid with net Δz and t(3).
(c) Real − surrogate z within each state: drug (filled) and awake eyes-closed (hollow), against sampling rate.
(d) Drug − awake change of raw z, real against surrogate, per arm: why raw drug z rises at low rates while Δz stays < 0.
<stem>_agents: panel (c) of mi-figure.png summarised per arm, Δz by agent (raw and in the arm's own eyes-open SD).

  python robust_grid_fig.py <lowrate_summary.txt> <stem>
"""
import re
import sys
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mtsspi.figures import SANS, SANS_PREAMBLE

mpl.rcParams.update({
    "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
    "text.latex.preamble": SANS_PREAMBLE, **SANS,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.grid": False, "legend.frameon": False,
    "lines.linewidth": 1.7, "lines.markersize": 2.7,
    "figure.constrained_layout.use": True,
    "figure.dpi": 180, "savefig.dpi": 600, "savefig.bbox": "tight",
})
INK, GREY = '#2B2B2B', '#B9BFC3'
TAU = {'drug': .105, 'awake': .031}  # median tau_1/e (s), robust/acf.txt: KTMD drug, awake EC
# arm: (rate Hz, window s, band label); the 100 Hz anchor is the 1,260-window bank, the rest are low-rate-session arms
ARMS = {'100Hz': (100, 10, '0.5-40'), 'L20': (100, 10, '0.5-8'), 'L10': (100, 10, '0.5-4'),
        'S20': (20, 10, 'sub'), 'D20': (20, 10, '0.5-8'), 'S10': (10, 10, 'sub'), 'D10': (10, 10, '0.5-4'),
        'S20L': (20, 50, 'sub'), 'D20L': (20, 50, '0.5-8'), 'S10L': (10, 100, 'sub'), 'D10L': (10, 100, '0.5-4')}
RUNNING = [(100, 10, 'C10'), (100, 50, 'W50')]
PROPOSED = [(20, 25), (50, 10), (50, 25), (50, 50), (100, 25)]
CONDITIONAL = [(200, 10)]


def parse(f):
    out = {}
    for line in open(f):
        m = re.match(r'(\S+)\s+ns\s+EO SD .*t\(3\) ([-+.\d]+), p ([\d.]+); pooled ([-+.\d]+)', line)
        if m and m.group(1) in ARMS:
            out.setdefault(m.group(1), {}).update(t=float(m.group(2)), dz=float(m.group(4)))
        m = re.match(r'(\S+)\s+T (\d+) n_win \d+ \| ns real/surr awake_ec z ([\d.]+)/([\d.]+); drug z ([\d.]+)/([\d.]+)', line)
        if m and m.group(1) in ARMS:
            ar, as_, dr, ds = map(float, m.groups()[2:])
            out.setdefault(m.group(1), {}).update(T=int(m.group(2)), ar=ar, as_=as_, dr=dr, ds=ds)
    return out


OFF = {'D10L': (-17, 3), 'S20L': (3, -6), 'S10L': (3, 2), 'D20L': (3, -7), 'D20': (3, 0), 'D10': (3, 1)}


def fmt(v):
    return f'{v:+.4f}'.replace('0.', '.').replace('-', '\u2212')


AGENTS = ('KT', 'KTMD', 'MD', 'PF', 'SLP')
ROWS = ['100Hz', 'L20', 'L10', 'S20', 'D20', 'S10', 'D10', 'S20L', 'D20L', 'S10L', 'D10L']
ROWLAB = {'100Hz': 'anchor 100 Hz, 10 s', 'L20': '100 Hz, 0.5-8 Hz, 10 s', 'L10': '100 Hz, 0.5-4 Hz, 10 s',
          'S20': '20 Hz sub, 10 s', 'D20': '20 Hz dec, 10 s', 'S10': '10 Hz sub, 10 s', 'D10': '10 Hz dec, 10 s',
          'S20L': '20 Hz sub, 50 s', 'D20L': '20 Hz dec, 50 s', 'S10L': '10 Hz sub, 100 s', 'D10L': '10 Hz dec, 100 s'}


def parse_agents(f):
    out = {}
    for line in open(f):
        m = re.match(r'(\S+)\s+ns\s+EO SD ([\d.]+) .*?\| (KT .*?) \| animals', line)
        if m and m.group(1) in ARMS:
            ag = dict((a, (float(v), float(sd))) for a, v, sd in re.findall(r'(\w+) ([-+.\d]+) \(([-+.\d]+) SD', m.group(3)))
            out[m.group(1)] = dict(eo=float(m.group(2)), ag=ag)
    return out


def agents_figure(f, stem):
    A, R = parse_agents(f), parse(f); norm = TwoSlopeNorm(vcenter=0, vmin=-.02, vmax=.006); cmap = plt.get_cmap('RdBu_r')
    rows = [k for k in ROWS if k in A]; cols = list(AGENTS) + ['pooled']
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for i, k in enumerate(rows):
        for j, a in enumerate(cols):
            v, sd = (R[k]['dz'], R[k]['dz'] / A[k]['eo']) if a == 'pooled' else A[k]['ag'][a]
            ax.add_patch(plt.Rectangle((j - .48, i - .46), .96, .92, color=cmap(norm(v)), lw=0))
            ax.text(j, i, f'{fmt(v)}\n({sd:+.1f} SD)', ha='center', va='center', fontsize=6, color='white' if norm(v) < .12 else INK)
        ax.text(len(cols) - .4, i, f"EO SD {A[k]['eo']:.4f}", va='center', fontsize=6, color='#7A8085')
    for y in (2.5, 6.5):
        ax.axhline(y, color=INK, lw=.6)
    ax.set(xlim=(-.5, len(cols) - .5), ylim=(len(rows) - .5, -.5))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(['Ket.', 'Ket.+med.', 'Medet.', 'Prop.', 'Sleep', 'pooled drug'])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([ROWLAB[k] for k in rows], fontsize=7)
    ax.xaxis.tick_top(); ax.spines[['left', 'bottom']].set_visible(False); ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=ax, shrink=.5, pad=.1, aspect=15)
    cb.set_label(r'$\Delta z$ (raw; SD = own eyes-open SD)', fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=180 if ext == 'png' else 600)


def neff(T, fs, tau):
    return T / max(2 * tau, 1 / fs)


def figure(f, stem):
    R = parse(f); norm = TwoSlopeNorm(vcenter=0, vmin=-.016, vmax=.004); cmap = plt.get_cmap('RdBu_r')
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.2))
    a, b, c, d = ax.ravel()
    # (a) N and N_eff against window
    Ts = np.array([2, 5, 10, 25, 50, 100, 200.])
    for hz in (10, 20, 50, 100, 200):
        a.plot(Ts, Ts * hz, color=GREY, lw=.9, ls=(0, (3, 1.5)))
        a.text(230, 200 * hz, f'$N$, {hz} Hz', fontsize=6.5, va='center', color='#7A8085', clip_on=False)
    for s, col in (('drug', '#7B4B94'), ('awake', '#8C969C')):
        a.plot(Ts, [neff(t, 100, TAU[s]) for t in Ts], color=col, lw=1.2)
        a.text(2.1, neff(2, 100, TAU[s]) * (1.6 if s == 'awake' else .62), rf'$N_\mathrm{{eff}}$ {s}', color=col, fontsize=7)
    for k, (hz, w, _) in ARMS.items():
        if k in R:
            a.scatter(w * (1.06 if k.startswith('D') else .94 if k.startswith('S') else 1), w * hz, s=34,
                      c=[R[k]['dz']], cmap=cmap, norm=norm, edgecolor=INK, lw=.5, zorder=3)
    for hz, w, _ in RUNNING:
        a.scatter(w, w * hz, s=34, facecolor='none', edgecolor=INK, lw=.9, zorder=3, hatch='////')
    for hz, w in PROPOSED + CONDITIONAL:
        a.scatter(w, w * hz, s=34, facecolor='none', edgecolor=INK, lw=.8, ls=':' if (hz, w) in CONDITIONAL else '-', zorder=3)
    a.set(xscale='log', yscale='log', xlim=(1.8, 220), ylim=(8, 6e4), xlabel='window (s)', ylabel='samples per window')
    a.set_xticks(Ts); a.set_xticklabels([f'{t:g}' for t in Ts]); a.minorticks_off()
    a.set_title(r'(a) $N$ per rate (grey) and $N_\mathrm{eff}$ at 100 Hz', loc='left', fontsize=8.5)
    # (b) grid
    rates, wins = [10, 20, 50, 100, 200], [10, 25, 50, 100]
    for i, hz in enumerate(rates):
        for j, w in enumerate(wins):
            done = [k for k, (h, ww, _) in ARMS.items() if h == hz and ww == w and k in R and k != '100Hz' and not k.startswith('L')]
            if hz == 100 and w == 10:
                done = ['100Hz', 'L20', 'L10']
            if done:
                v = R['100Hz']['dz'] if hz == 100 else np.mean([R[k]['dz'] for k in done])  # 100 Hz cell: anchor alone
                b.add_patch(plt.Rectangle((j - .48, i - .48), .96, .96, color=cmap(norm(v)), lw=0))
                txt = '\n'.join(f"{k if k != '100Hz' else 'anchor'} {fmt(R[k]['dz'])} ({R[k]['t']:+.1f})" for k in done)
                b.text(j, i, txt, ha='center', va='center', fontsize=5.3, color=INK)
            elif (hz, w) in [(h, ww) for h, ww, _ in RUNNING]:
                b.add_patch(plt.Rectangle((j - .48, i - .48), .96, .96, fill=False, hatch='////', ec=GREY, lw=.6))
                b.text(j, i, 'running\n' + [n for h, ww, n in RUNNING if (h, ww) == (hz, w)][0], ha='center', va='center', fontsize=6.5)
            elif (hz, w) in PROPOSED + CONDITIONAL:
                b.add_patch(plt.Rectangle((j - .48, i - .48), .96, .96, fill=False, ec=INK, lw=.6,
                                          ls=':' if (hz, w) in CONDITIONAL else '-'))
                b.text(j, i, f"{'if 50 Hz $<$ 100' if (hz, w) in CONDITIONAL else 'proposed'}\nT {hz * w}", ha='center', va='center',
                       fontsize=6.5)
    b.set(xlim=(-.5, len(wins) - .5), ylim=(-.5, len(rates) - .5), xlabel='window (s)', ylabel='sampling rate (Hz)')
    b.set_xticks(range(len(wins))); b.set_xticklabels(wins); b.set_yticks(range(len(rates))); b.set_yticklabels(rates)
    b.spines[['left', 'bottom']].set_visible(False); b.tick_params(length=0)
    b.set_title(r'(b) pooled drug $\Delta z$ ($t_{(3)}$)', loc='left', fontsize=8.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=[a, b], shrink=.6, pad=.01)
    cb.set_label(r'pooled drug $\Delta z$', fontsize=8); cb.ax.tick_params(labelsize=7)
    # (c) real - surrogate within state against rate
    for k, (hz, w, band) in ARMS.items():
        if k not in R:
            continue
        x = hz * (1.07 if k.startswith('D') else .93 if k.startswith('S') else 1) * (1.0 if w == 10 else 1.15)
        mk = 'o' if w == 10 else 's'
        if k in ('L20', 'L10'):
            mk, x = '^', hz * (1.07 if k == 'L20' else 1.14)
        c.plot(x, R[k]['dr'] - R[k]['ds'], mk, color='#7B4B94', ms=5)
        c.plot(x, R[k]['ar'] - R[k]['as_'], mk, mfc='none', mec='#7A8085', ms=5, mew=.9)
    c.axhline(0, color=GREY, lw=.8)
    c.set(xscale='log', xlim=(8, 130), xlabel='sampling rate (Hz)', ylabel=r'$z_\mathrm{real} - z_\mathrm{surr}$ within state')
    c.set_xticks([10, 20, 50, 100]); c.set_xticklabels([10, 20, 50, 100]); c.minorticks_off()
    for lab, mk, mf in (('drug', 'o', '#7B4B94'), ('awake EC', 'o', 'none')):
        c.plot([], [], mk, mfc=mf, mec='#7B4B94' if mf != 'none' else '#7A8085', ms=5, label=lab)
    c.plot([], [], 'o', color=INK, mfc='none', ms=4, label='10 s'); c.plot([], [], 's', color=INK, mfc='none', ms=4, label='50-100 s')
    c.plot([], [], '^', color=INK, mfc='none', ms=4, label='100 Hz, low-passed')
    c.legend(fontsize=6.5, loc='lower center', ncol=2, handletextpad=.3, columnspacing=.8)
    c.set_title('(c) departure from the surrogate, by state', loc='left', fontsize=8.5)
    # (d) drug - awake change in raw z: real against surrogate
    keys = [k for k in ARMS if k in R]
    lo = min(min(R[k]['dr'] - R[k]['ar'], R[k]['ds'] - R[k]['as_']) for k in keys) - .004
    hi = max(max(R[k]['dr'] - R[k]['ar'], R[k]['ds'] - R[k]['as_']) for k in keys) + .004
    d.plot([lo, hi], [lo, hi], color=GREY, lw=.8)
    for k in keys:
        x, y = R[k]['ds'] - R[k]['as_'], R[k]['dr'] - R[k]['ar']
        d.scatter(x, y, s=26, c=[R[k]['dz']], cmap=cmap, norm=norm, edgecolor=INK, lw=.5, zorder=3)
        d.annotate('anchor' if k == '100Hz' else k, (x, y), xytext=OFF.get(k, (3, -2)), textcoords='offset points', fontsize=5.8)
    d.axhline(0, color=GREY, lw=.5, ls=':'); d.axvline(0, color=GREY, lw=.5, ls=':')
    d.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal', xlabel=r'surrogate $z$: drug $-$ awake EC', ylabel=r'real $z$: drug $-$ awake EC')
    d.set_title(r'(d) below the diagonal: $\Delta z < 0$', loc='left', fontsize=8.5)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=180 if ext == 'png' else 600)


if __name__ == '__main__':
    figure(sys.argv[1], sys.argv[2])
    agents_figure(sys.argv[1], sys.argv[2] + '_agents')
