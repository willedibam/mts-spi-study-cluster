"""Candidates for fig:mi. pairs (a): the 10 off-band pairs of the figwin windows with the largest K - K_surr, transients
kept and flagged (a sample beyond 5 robust SDs), plus the best clean pair of (b)'s big window; each as its glyph and
traces. windows (b): ketamine + medetomidine windows at stated places in the state's Δz distribution, drawn as (b)'s.

    python fig_candidates.py pairs <results/representation/atlas-mi-pair-261117> <output.png>
    python fig_candidates.py windows <results/representation/atlas-mi-pair-261117> <output.png>
"""
import sys; from pathlib import Path
sys.path.insert(0, "."); import numpy as np, matplotlib.pyplot as plt, plot_character_draft as P
from scipy.stats import chi2, kurtosis
mode, root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); df, edges = P.load(root); chan = {}
if mode == "windows":
    w = df[df.group == "KTMD"].sort_values("sd").reset_index(drop=True)
    w["off"] = [int(P.band(edges, x)[2].sum()) for x in w.id]; med = w.sd.median()
    pick_ = lambda i: w.iloc[int(i)]
    near = w[(w.sd - med).abs() <= .5]                    # display-selected: most off-band pairs near the median
    cand = [(w[w.id == P.PANEL_B[0][1]].iloc[0], "median (current)"), (pick_(np.argmin(np.abs(w.sd - w.sd.mean()))), "nearest the mean"),
            (near.loc[near.off.idxmax()], r"most off-band within $\pm$.5 SD of median"), (pick_(round(.25 * (len(w) - 1))), "25th percentile"),
            (pick_(round(.1 * (len(w) - 1))), "10th percentile"), (pick_(0), "most extreme")]
    P.figure_style(); plt.rcParams.update({"font.size": 6})
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0), layout="constrained"); ink = P.COLOR["KTMD"]
    for ax, (r, why) in zip(axs.flat, cand):
        (g, k), _ = edges[r.id]; c, hw, o = P.band(edges, r.id); xx = np.linspace(-.05, .6, 100)
        ax.fill_between(xx, np.polyval(c, xx) - hw, np.polyval(c, xx) + hw, color=P.LIGHT, lw=0)
        ax.plot(xx, np.polyval(c, xx), color="white", lw=1)
        ax.scatter(g[~o], k[~o], s=5, lw=.2, ec="white", color=P.GREY, alpha=.6)
        ax.scatter(g[o], k[o], s=14, lw=.35, ec="white", color=ink)
        ax.set(xlim=(-.02, .42), ylim=(-.17, .52), xticks=[0, .2, .4], yticks=[0, .2, .4])
        ax.set_title(rf"\textbf{{{why}}}" + "\n" + f"{r.animal} {r.id[:8]} {r.id.split(':', 1)[1]}", fontsize=6.3, color=ink)
        ax.text(.97, .04, rf"$\Delta z$ \textbf{{{r.sd:.1f}}} SD".replace("-", "--") + "\n" + rf"$z$ {r.real:.3f}, surr. {r.null:.3f}; {r.off} off band",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=P.COLOR["ink"])
        print(why, r.id, round(r.sd, 2), r.off, round(r.real, 3), round(r.null, 3))
    print("median", round(med, 2), "mean", round(w.sd.mean(), 2), "wake median off band",
          int(np.median([P.band(edges, x)[2].sum() for x in df[df.group == "awake_eo"].id])))
    fig.savefig(out, dpi=400); sys.exit()
for p_ in (root / "zns").glob("zns_[AB]_*.npz"):
    with np.load(p_) as d_: chan.update(zip(d_["id"], d_["chan"]))
W = np.load(root / "figwin/windows.npz")
rob = lambda v: np.abs(v - np.median(v)).max() / (1.4826 * np.median(np.abs(v - np.median(v))))
iu = np.triu_indices(32, 1); rows = []
for wid in [k for k in W.files if "|" not in k and k in edges]:
    (g, k), (gs, ks) = edges[wid]; _, _, off = P.band(edges, wid)
    lab = list(W[wid + "|chan"]); X = W[wid][:, [lab.index(v) for v in chan[wid]]].astype(float)
    grp = df[df.id == wid].group.iloc[0]
    for pk in np.flatnonzero(off):
        a, b = iu[0][pk], iu[1][pk]; x = X[:, [a, b]]
        u = np.column_stack([P.normal_score(x[:, 0]), P.normal_score(x[:, 1])]); r = np.corrcoef(u.T)[0, 1]
        q = (u[:, 0]**2 - 2*r*u[:, 0]*u[:, 1] + u[:, 1]**2) / (1 - r**2)
        rows.append(dict(wid=wid, grp=grp, pk=pk, ea=chan[wid][a], eb=chan[wid][b], G=g[pk], K=k[pk], Ks=ks[pk], r=r,
                         far=100 * (q > chi2.ppf(.99, 2)).mean(), kurt=kurtosis(u[:, 0] - u[:, 1]),
                         rob=max(rob(x[:, 0]), rob(x[:, 1])), x=x, u=u))
rows.sort(key=lambda d: -(d["K"] - d["Ks"]))
pick = rows[:10]
bigw = P.PANEL_B[0][1]
for d in rows:                                   # (b)'s big window: its best clean pair
    if d["wid"] == bigw and d["rob"] < 5 and d not in pick:
        pick.append(d); break
P.figure_style(); plt.rcParams.update({"font.size": 6})
nc = 4; nr = -(-len(pick) // nc)
fig = plt.figure(figsize=(7.2, 2.35 * nr))
for i, d in enumerate(pick):
    c0, r0 = i % nc, i // nc; x0 = c0 / nc; y0 = 1 - (r0 + 1) / nr; hh = 1 / nr
    col = P.COLOR.get(d["grp"], P.COLOR["ink"])
    ga = fig.add_axes([x0 + .03, y0 + .35 * hh, .19, .19 * 7.2 / (2.35 * nr)]); P.glyph(ga, d["u"], col)
    tr = fig.add_axes([x0 + .03, y0 + .05 * hh, .19, .27 * hh])
    xc = d["x"] - np.median(d["x"], 0); xc /= np.abs(xc).max(0)
    for j in range(2):
        tr.plot(xc[:, j] * .45 + (.5 - j), color=col, lw=.35)
    tr.axis("off"); tr.set(xlim=(0, 1000), ylim=(-1, 1))
    tag = "transient" if d["rob"] >= 5 else "clean"
    who = ("Chibi " if "chibi" in d["wid"] else "George ") + d["wid"][:8]
    sub = "  [(b)'s window]" if d["wid"] == bigw else ("  [current (a)]" if (d["wid"], d["pk"]) == P.GLYPH_PAIR else "")
    fig.text(x0 + .125, y0 + .985 * hh, rf"\textbf{{{i + 1}.  el.\,{d['ea']}--{d['eb']}}}" + sub, ha="center", va="top", fontsize=6, color=col)
    fig.text(x0 + .125, y0 + .935 * hh, f"{P.NAME[d['grp']]}, {d['wid'].split(':', 1)[1]}, {who}", ha="center", va="top", fontsize=4.8, color=P.GREY)
    fig.text(x0 + .125, y0 + .335 * hh,
             f"$K-K_s$ {d['K'] - d['Ks']:.2f}   $>$99\\%: {d['far']:.1f}\\%   kurt {d['kurt']:.1f}   max {d['rob']:.0f} rSD ({tag})",
             ha="center", va="bottom", fontsize=4.6, color="#C8553D" if tag == "transient" else P.COLOR["ink"])
    print(i + 1, d["wid"], d["pk"], d["ea"], d["eb"], round(d["G"], 2), round(d["K"], 2), round(d["Ks"], 2), round(d["far"], 1), round(d["kurt"], 1), round(d["rob"], 1))
fig.savefig(out, dpi=400)
