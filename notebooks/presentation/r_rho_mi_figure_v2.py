import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from src.utils import project_root

RUN = project_root() / "data/r_rho_mi/260624_g-roll"     # beta = pi
ABBR = {"linear": "L", "monotonic": "M", "non-monotonic": "NM"}
SPI = {"r": "#0072B2", "rho": "#E69F00", "MI": "#009E73"}
PAIRCOL = {"L-M": "#9467bd", "L-NM": "#8c564b", "M-NM": "#e377c2"}
SAME = "#c8c8c8"; ACCENT = "#d62728"
FEAT = ["corr(r,rho)", "corr(r,MI)", "corr(rho,MI)"]
FEATCOL = {"corr(r,rho)": "#5b6dbf", "corr(r,MI)": "#c98a3a", "corr(rho,MI)": "#5aa86f"}
ITERS = ["iter1_L", "iter2_LM", "iter3_LMNM"]
PLANES = [("r", "rho"), ("r", "MI"), ("rho", "MI")]

def _km(f):
    km = {}
    for k in f:
        kl = k.lower()
        if "cov" in kl or "empirical" in kl: km["r"] = k
        elif "spearman" in kl: km["rho"] = k
        elif "mi_" in kl or "kraskov" in kl: km["MI"] = k
    return km
def load(d):
    d = Path(d); npz = np.load(d/"spi_mpis.npz"); km = _km(npz.files)
    return np.load(d/"timeseries.npy"), {k: npz[km[k]] for k in ("r","rho","MI")}, json.loads((d/"meta.json").read_text())["generator"]
def linf(x): return float(np.sqrt(np.clip(1-np.exp(-2*max(x,0)),0,1)))
def od(M): iu = np.triu_indices(M.shape[0],1); return M[iu]
def fc(a,b): return float(np.corrcoef(a,b)[0,1])
def find_pair(meta,tx,ty,snr="high"):
    types,noise = meta["types"],meta["noise_stds"]; want={tx,ty}
    c=[(i,j) for i in range(len(types)) for j in range(i+1,len(types)) if {ABBR[types[i]],ABBR[types[j]]}==want]
    c.sort(key=lambda ij:noise[ij[0]]+noise[ij[1]]); return {"high":c[0],"low":c[-1],"median":c[len(c)//2]}[snr]
def glyph(ax,kind,color="k"):
    t=np.linspace(-1,1,80); y={"line":t,"S":np.tanh(2.5*t),"U":t**2*2-1}[kind]
    ax.plot(t,y,color=color,lw=1.8); ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.3,1.3); ax.axis("off")
def sq(ax): ax.set_box_aspect(1)

fig = plt.figure(figsize=(12.5, 16.5))
g = fig.add_gridspec(4, 1, height_ratios=[1.9, 1.5, 3.0, 1.4], hspace=0.42)

# ============ Panel A: dependency alphabet ============
ts3, mp3, meta3 = load(RUN/"iter3_LMNM/M32_T2000_I0"); types3 = meta3["types"]
order = np.argsort([{"linear":0,"monotonic":1,"non-monotonic":2}[t] for t in types3], kind="stable")
gsA = g[0].subgridspec(2, 4, width_ratios=[1.25,1,1,1], height_ratios=[1,1], hspace=0.15, wspace=0.42)
axM = fig.add_subplot(gsA[:,0])
axM.imshow(ts3[:400, order].T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
axM.set_title("an MTS  (channels x time)", fontsize=10); axM.set_xlabel("t", fontsize=9); axM.set_xticks([])
for b in (11.5,23.5): axM.axhline(b, color="k", lw=0.7)
for cen,lab in [(5.5,"L"),(17.5,"M"),(27.5,"NM")]: axM.text(-26,cen,lab,fontsize=11,fontweight="bold",va="center")
A_bars = {}
for c,(tx,ty,gl) in enumerate([("L","L","line"),("L","M","S"),("L","NM","U")]):
    i,j = find_pair(meta3,tx,ty,"high")
    axp = fig.add_subplot(gsA[0,c+1]); axb = fig.add_subplot(gsA[1,c+1]); A_bars[(tx,ty)] = axb
    axp.scatter(ts3[:,i],ts3[:,j],s=3,alpha=0.22,color="#34495e",linewidths=0); sq(axp)
    axp.set_xticks([]); axp.set_yticks([]); axp.set_title(f"{tx}-{ty}",fontsize=10,pad=2)
    gi = axp.inset_axes([0.64,0.64,0.34,0.34]); glyph(gi,gl,PAIRCOL.get(f"{tx}-{ty}","#444"))
    r,rho,mi_ = mp3["r"][i,j],mp3["rho"][i,j],linf(mp3["MI"][i,j])
    axb.bar(["r","ρ","MI"],[r,rho,mi_],color=[SPI["r"],SPI["rho"],SPI["MI"]],width=0.72); sq(axb)
    axb.axhline(0,color="grey",lw=0.5); axb.set_ylim(-0.25,1.08); axb.set_yticks([0,0.5,1.0]); axb.tick_params(labelsize=8)
    for s in ("top","right"): axb.spines[s].set_visible(False)
    if (tx,ty)==("L","M"):   # callout: the subtle monotone-nl r-drop
        axb.annotate("r<ρ", xy=(0,r), xytext=(0.2,0.45), fontsize=9, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
fig.text(0.06,0.975,"A",fontsize=17,fontweight="bold")
fig.text(0.13,0.975,"Each channel pair has a relationship; r/ρ/MI form a nested capture hierarchy",fontsize=10.5)

# ============ Panel B: type-ordered MPI heatmaps (iter3) ============
gsB = g[1].subgridspec(1, 5, width_ratios=[0.4,1,1,1,0.4], wspace=0.45)
blocks = {t:[k for k,a in enumerate([ABBR[types3[m]] for m in order]) if a==t] for t in ("L","M","NM")}
for c,key in enumerate(("r","rho","MI")):
    ax = fig.add_subplot(gsB[0,c+1]); Mre = mp3[key][np.ix_(order,order)].astype(float); np.fill_diagonal(Mre,np.nan)
    im = ax.imshow(Mre, cmap="RdBu_r", vmin=-1, vmax=1) if key!="MI" else ax.imshow(Mre, cmap="viridis"); sq(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_title(key, fontsize=11)
    for idx in blocks.values(): ax.axhline(idx[-1]+0.5,color="k",lw=0.5); ax.axvline(idx[-1]+0.5,color="k",lw=0.5)
    tk=[np.mean(idx) for idx in blocks.values()]; ax.set_xticks(tk); ax.set_yticks(tk)
    ax.set_xticklabels(blocks.keys(),fontsize=8); ax.set_yticklabels(blocks.keys(),fontsize=8)
fig.text(0.06,0.70,"B",fontsize=17,fontweight="bold")
fig.text(0.13,0.70,"Same data as matrices: the (L,NM) block is ~0 in r but bright in MI",fontsize=10.5)

# ============ Panel C: 3x3 SPI-SPI planes (square) ============
gsC = g[2].subgridspec(3, 3, hspace=0.18, wspace=0.30)
exem = find_pair(meta3,"L","NM","high"); C_target=None
for ri,it in enumerate(ITERS):
    _,mp,meta = load(RUN/f"{it}/M32_T2000_I0"); types=meta["types"]; iu=np.triu_indices(len(types),1)
    labs=np.array(["-".join(sorted([ABBR[types[i]],ABBR[types[j]]])) for i,j in zip(*iu)])
    same=np.array([l.split("-")[0]==l.split("-")[1] for l in labs])
    for ci,(sx,sy) in enumerate(PLANES):
        ax=fig.add_subplot(gsC[ri,ci]); sq(ax); xv,yv=mp[sx][iu],mp[sy][iu]
        ax.scatter(xv[same],yv[same],s=11,facecolors="none",edgecolors=SAME,linewidths=0.6,alpha=0.7)
        for lab,col in PAIRCOL.items():
            m=labs==lab
            if m.any(): ax.scatter(xv[m],yv[m],s=15,color=col,alpha=0.8)
        if sx=="r" and sy=="rho": ax.plot([-0.2,1.05],[-0.2,1.05],ls=":",color="grey",lw=0.6)
        ax.text(0.05,0.95,f"$f$={fc(xv,yv):+.2f}",transform=ax.transAxes,va="top",fontsize=8.5,
                bbox=dict(boxstyle="round",fc="white",ec="grey",alpha=0.85))
        ax.tick_params(labelsize=7)
        if ri==2: ax.set_xlabel(sx,fontsize=9)
        if ci==0: ax.set_ylabel(f"{it.replace('_','-')}\n\n{sy}",fontsize=9)
        else: ax.set_ylabel(sy,fontsize=9)
        if ri==0: ax.set_title(f"({sx},{sy})",fontsize=9.5)
        if it=="iter3_LMNM" and (sx,sy)==("rho","MI"):
            a,b=exem; ax.scatter([mp["rho"][a,b]],[mp["MI"][a,b]],s=80,facecolors="none",edgecolors=ACCENT,linewidths=2,zorder=5)
            C_target=(ax,mp["rho"][a,b],mp["MI"][a,b])
fig.text(0.06,0.585,"C",fontsize=17,fontweight="bold")
fig.text(0.13,0.585,"Pairs populate the SPI–SPI planes across iterations; $f_{ij}$ = their correlation",fontsize=10.5)

# ============ Panel D: compact raincloud ============
gsD = g[3].subgridspec(1, 3, width_ratios=[0.5,2.0,0.5])
axD = fig.add_subplot(gsD[0,1])
samp = {p: [] for p in FEAT}
for it in ITERS:
    per={p:[] for p in FEAT}
    for d in sorted((RUN/it).glob("M32_T2000_I*")):
        _,mp,_=load(d); rv,sv,iv=od(mp["r"]),od(mp["rho"]),od(mp["MI"])
        per["corr(r,rho)"].append(fc(rv,sv)); per["corr(r,MI)"].append(fc(rv,iv)); per["corr(rho,MI)"].append(fc(sv,iv))
    for p in FEAT: samp[p].append(np.array(per[p]))
offs={"corr(r,rho)":-0.22,"corr(r,MI)":0,"corr(rho,MI)":0.22}; rng=np.random.default_rng(0)
for p in FEAT:
    col=FEATCOL[p]
    for ii,data in enumerate(samp[p]):
        pos=ii+offs[p]
        if np.std(data)>1e-9:
            for bb in axD.violinplot([data],positions=[pos],widths=0.16,showextrema=False)["bodies"]:
                v=bb.get_paths()[0].vertices; v[:,0]=np.clip(v[:,0],pos,np.inf); bb.set_facecolor(col); bb.set_alpha(0.3); bb.set_edgecolor("none")
        axD.boxplot([data],positions=[pos],widths=0.05,showfliers=False,patch_artist=True,
                    boxprops=dict(facecolor="white",edgecolor=col),medianprops=dict(color=col),whiskerprops=dict(color=col),capprops=dict(color=col))
        jit=pos-0.05-np.abs(rng.normal(0,0.011,len(data))); axD.scatter(jit,data,s=8,color=col,alpha=0.7,linewidths=0,zorder=3)
    axD.scatter([],[],color=col,label=p)
y0=samp["corr(rho,MI)"][2][0]; xD=2+offs["corr(rho,MI)"]-0.05
axD.scatter([xD],[y0],s=80,facecolors="none",edgecolors=ACCENT,linewidths=2,zorder=6)
axD.set_xticks(range(3)); axD.set_xticklabels([it.replace("_","-") for it in ITERS],fontsize=9); axD.set_ylim(0,1.08); axD.set_xlim(-0.5,2.45)
axD.set_ylabel(r"$f_{ij}$",fontsize=11); axD.legend(fontsize=7.5,loc="lower left",ncol=1); axD.grid(True,axis="y",alpha=0.3)
fig.text(0.06,0.235,"D",fontsize=17,fontweight="bold")
fig.text(0.13,0.235,"The staggered signature over 30 instances",fontsize=10.5)

# ============ Telescope (one exemplar L-NM pair) ============
if C_target:
    axc,rx,my = C_target
    fig.add_artist(ConnectionPatch(xyA=(xD,y0),coordsA=axD.transData,xyB=(0.5,-0.12),coordsB=axc.transAxes,
                                   color=ACCENT,lw=1.0,alpha=0.75,arrowstyle="-",linestyle="--"))
    fig.add_artist(ConnectionPatch(xyA=(rx,my),coordsA=axc.transData,xyB=(0.5,-0.22),coordsB=A_bars[("L","NM")].transAxes,
                                   color=ACCENT,lw=1.0,alpha=0.75,arrowstyle="-",linestyle="--"))

fig.suptitle("r–ρ–MI case study (v2: square panels + MPI, β=π)", y=0.995, fontsize=13)
fig.savefig("/tmp/figure_v2.png", dpi=105, bbox_inches="tight")
print("saved /tmp/figure_v2.png")
