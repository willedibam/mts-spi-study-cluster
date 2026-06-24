import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from src.utils import project_root

RUN = project_root() / "data/r_rho_mi/260624_g-roll"     # beta = pi
ABBR = {"linear": "L", "monotonic": "M", "non-monotonic": "NM"}
SPI = {"r": "#0072B2", "rho": "#E69F00", "MI": "#009E73"}          # SPIs own the bright triad
PAIRCOL = {"L-M": "#9467bd", "L-NM": "#8c564b", "M-NM": "#e377c2"}  # pair-types: distinct from SPI hues
SAME = "#c8c8c8"; ACCENT = "#d62728"
FEAT = ["corr(r,rho)", "corr(r,MI)", "corr(rho,MI)"]
FEATCOL = {"corr(r,rho)": "#5b6dbf", "corr(r,MI)": "#c98a3a", "corr(rho,MI)": "#5aa86f"}
ITERS = ["iter1_L", "iter2_LM", "iter3_LMNM"]

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
    ax.plot(t,y,color=color,lw=2.0); ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.3,1.3); ax.axis("off")

fig = plt.figure(figsize=(13.5, 16))
outer = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.5, 0.85], hspace=0.30)

# ---------- Panel A: dependency alphabet + capture hierarchy ----------
ts3, mp3, meta3 = load(RUN/"iter3_LMNM/M32_T2000_I0"); types3 = meta3["types"]
order = np.argsort([{"linear":0,"monotonic":1,"non-monotonic":2}[t] for t in types3], kind="stable")
gsA = outer[0].subgridspec(2, 4, width_ratios=[1.35,1,1,1], height_ratios=[1.25,1], hspace=0.45, wspace=0.32)
axM = fig.add_subplot(gsA[:,0])
axM.imshow(ts3[:500, order].T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
axM.set_title("an MTS (iter3)", fontsize=10); axM.set_xlabel("t"); axM.set_yticks([])
for b,lab in [(11.5,"L"),(23.5,"M")]: axM.axhline(b,color="k",lw=0.7)
for cen,lab in [(5.5,"L"),(17.5,"M"),(27.5,"NM")]: axM.text(-30,cen,lab,fontsize=11,fontweight="bold",va="center")
archs = [("L","L","line"),("L","M","S"),("L","NM","U")]
axA_bars = {}
for c,(tx,ty,gl) in enumerate(archs):
    i,j = find_pair(meta3,tx,ty,"high")
    axp = fig.add_subplot(gsA[0,c+1]); axb = fig.add_subplot(gsA[1,c+1]); axA_bars[(tx,ty)] = axb
    axp.scatter(ts3[:,i],ts3[:,j],s=3,alpha=0.22,color="#34495e",linewidths=0)
    axp.set_xticks([]); axp.set_yticks([]); axp.set_title(f"{tx}-{ty}",fontsize=10)
    gi = axp.inset_axes([0.66,0.66,0.34,0.34]); glyph(gi,gl,PAIRCOL.get(f"{tx}-{ty}","#444"))
    r,rho,mi_ = mp3["r"][i,j],mp3["rho"][i,j],linf(mp3["MI"][i,j])
    axb.bar(["r","ρ","MI"],[r,rho,mi_],color=[SPI["r"],SPI["rho"],SPI["MI"]],width=0.72)
    axb.axhline(0,color="grey",lw=0.5); axb.set_ylim(-0.3,1.08); axb.set_yticks([0,0.5,1.0])
    for sp in ("top","right"): axb.spines[sp].set_visible(False)
axM.annotate("", xy=(1.02,0.5), xytext=(0.92,0.5), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->",lw=1.2))
fig.text(0.07,0.965,"A",fontsize=18,fontweight="bold")
fig.text(0.30,0.965,"Each channel pair has a relationship; r/ρ/MI detect a nested hierarchy of it",fontsize=11)

# ---------- Panel C: SPI-SPI planes, 3 iters x 2 planes ----------
gsC = outer[1].subgridspec(3, 2, hspace=0.30, wspace=0.22)
exec_pair = find_pair(meta3,"L","NM","high")
axC_target = None
for ri,it in enumerate(ITERS):
    _,mp,meta = load(RUN/f"{it}/M32_T2000_I0"); types=meta["types"]; iu=np.triu_indices(len(types),1)
    labs=np.array(["-".join(sorted([ABBR[types[i]],ABBR[types[j]]])) for i,j in zip(*iu)])
    same=np.array([l.split("-")[0]==l.split("-")[1] for l in labs])
    for ci,(sx,sy) in enumerate([("r","rho"),("rho","MI")]):
        ax=fig.add_subplot(gsC[ri,ci]); xv,yv=mp[sx][iu],mp[sy][iu]
        ax.scatter(xv[same],yv[same],s=14,facecolors="none",edgecolors=SAME,linewidths=0.7,alpha=0.7)
        for lab,col in PAIRCOL.items():
            m=labs==lab
            if m.any(): ax.scatter(xv[m],yv[m],s=20,color=col,alpha=0.8)
        if sx=="r": ax.plot([-0.2,1.05],[-0.2,1.05],ls=":",color="grey",lw=0.7)
        ax.text(0.04,0.95,f"$f_{{{sx},{sy}}}$={fc(xv,yv):+.2f}",transform=ax.transAxes,va="top",fontsize=9,
                bbox=dict(boxstyle="round",fc="white",alpha=0.8))
        ax.set_xlabel(sx); ax.set_ylabel(sy); ax.tick_params(labelsize=8)
        if ri==0: ax.set_title(f"({sx}, {sy}) plane",fontsize=10)
        if ci==0: ax.text(-0.32,0.5,it.replace("_","-"),transform=ax.transAxes,rotation=90,va="center",fontsize=10,fontweight="bold")
        if it=="iter3_LMNM" and sx=="rho":
            a,b=exec_pair; ax.scatter([mp["rho"][a,b]],[mp["MI"][a,b]],s=90,facecolors="none",edgecolors=ACCENT,linewidths=2,zorder=5)
            axC_target=(ax,mp["rho"][a,b],mp["MI"][a,b])
fig.text(0.07,0.635,"C",fontsize=18,fontweight="bold")
fig.text(0.30,0.635,"Pairs populate the SPI–SPI planes; $f_{ij}$ is their correlation",fontsize=11)

# ---------- Panel D: staggered raincloud ----------
axD = fig.add_subplot(outer[2])
samp = {p: [] for p in FEAT}
for it in ITERS:
    per={p:[] for p in FEAT}
    for d in sorted((RUN/it).glob("M32_T2000_I*")):
        _,mp,_=load(d); rv,sv,iv=od(mp["r"]),od(mp["rho"]),od(mp["MI"])
        per["corr(r,rho)"].append(fc(rv,sv)); per["corr(r,MI)"].append(fc(rv,iv)); per["corr(rho,MI)"].append(fc(sv,iv))
    for p in FEAT: samp[p].append(np.array(per[p]))
offs={"corr(r,rho)":-0.25,"corr(r,MI)":0,"corr(rho,MI)":0.25}; rng=np.random.default_rng(0)
axD_target=None
for p in FEAT:
    col=FEATCOL[p]
    for ii,data in enumerate(samp[p]):
        pos=ii+offs[p]
        if np.std(data)>1e-9:
            for bb in axD.violinplot([data],positions=[pos],widths=0.18,showextrema=False)["bodies"]:
                v=bb.get_paths()[0].vertices; v[:,0]=np.clip(v[:,0],pos,np.inf); bb.set_facecolor(col); bb.set_alpha(0.3); bb.set_edgecolor("none")
        axD.boxplot([data],positions=[pos],widths=0.05,showfliers=False,patch_artist=True,
                    boxprops=dict(facecolor="white",edgecolor=col),medianprops=dict(color=col),whiskerprops=dict(color=col),capprops=dict(color=col))
        jit=pos-0.055-np.abs(rng.normal(0,0.012,len(data))); axD.scatter(jit,data,s=9,color=col,alpha=0.7,linewidths=0,zorder=3)
        if it=="x": pass
    axD.scatter([],[],color=col,label=p)
# exemplar dot: iter3 corr(rho,MI), instance I0 (first in sorted order)
y0=samp["corr(rho,MI)"][2][0]; xpos=2+offs["corr(rho,MI)"]-0.055
axD.scatter([xpos],[y0],s=90,facecolors="none",edgecolors=ACCENT,linewidths=2,zorder=6); axD_target=(xpos,y0)
axD.set_xticks(range(3)); axD.set_xticklabels([it.replace("_","-") for it in ITERS]); axD.set_ylim(0,1.08); axD.set_xlim(-0.6,2.5)
axD.set_ylabel(r"$f_{ij}$"); axD.legend(fontsize=8,loc="lower left"); axD.grid(True,axis="y",alpha=0.3)
fig.text(0.07,0.28,"D",fontsize=18,fontweight="bold")
fig.text(0.30,0.28,"The staggered signature over 30 instances",fontsize=11)

# ---------- Telescope links (one exemplar thread: an L-NM pair) ----------
if axC_target and axD_target:
    axc,rx,my=axC_target
    fig.add_artist(ConnectionPatch(xyA=axD_target,coordsA=axD.transData,xyB=(0.5,1.02),coordsB=axc.transAxes,
                                   color=ACCENT,lw=1.1,alpha=0.8,arrowstyle="-",linestyle="--"))
    fig.add_artist(ConnectionPatch(xyA=(rx,my),coordsA=axc.transData,xyB=(0.5,-0.18),coordsB=axA_bars[("L","NM")].transAxes,
                                   color=ACCENT,lw=1.1,alpha=0.8,arrowstyle="-",linestyle="--"))

fig.suptitle("r–ρ–MI case study (first-pass figure, beta=pi)  —  A: alphabet  C: feature space  D: signature", y=0.995, fontsize=13)
fig.savefig("/tmp/figure_v1.png", dpi=110, bbox_inches="tight")
print("saved /tmp/figure_v1.png  | exemplar L-NM pair:", exec_pair, "| iter3 corr(rho,MI) I0 =", round(y0,3))
