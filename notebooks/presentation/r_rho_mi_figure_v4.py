import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from src.utils import project_root

# v4 over v3 (see r_rho_mi_figure_plan.md):
#  - Panel A becomes a CONSTRUCTION + alphabet panel (double height): shared AR(1) mother z ->
#    per-channel filter glyphs (noisy phase-portraits: shape = filter family, scatter = SNR) ->
#    real type-ordered MTS heatmap -> 4 bracket-picked real phase plots + r/rho/MI bars.
#    Mother + filter glyphs are an illustrative regeneration (generative model); phase plots/bars
#    are REAL pyspi data. Fan-in lines + channel brackets are Illustrator (anchors only here).
#  - B kept on the shared sequential magnitude scale (r, rho, MI->Linfoot); spacers trimmed.
#  - C 3x2 planes + a legend/capture-hierarchy key in the freed third column (reclaims wasted width).
#  - D widened.

RUN = project_root() / "data/r_rho_mi/260624_g-roll"     # beta = pi (locked)
ABBR = {"linear": "L", "monotonic": "M", "non-monotonic": "NM"}
SPI = {"r": "#0072B2", "rho": "#E69F00", "MI": "#009E73"}
PAIRCOL = {"L-M": "#9467bd", "L-NM": "#8c564b", "M-NM": "#e377c2"}
TYPECOL = {"L": "#444444", "M": "#9467bd", "NM": "#8c564b"}
SAME = "#c8c8c8"; ACCENT = "#d62728"
FEAT = ["corr(r,rho)", "corr(r,MI)", "corr(rho,MI)"]
FEATCOL = {"corr(r,rho)": "#5b6dbf", "corr(r,MI)": "#c98a3a", "corr(rho,MI)": "#5aa86f"}
ITERS = ["iter1_L", "iter2_LM", "iter3_LMNM"]
PLANES = [("r", "rho"), ("rho", "MI")]
HEAT_CMAP = "viridis" #magma
BETA = np.pi

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
def linf(x): return np.sqrt(np.clip(1-np.exp(-2*np.maximum(x,0)),0,1))
def od(M): iu = np.triu_indices(M.shape[0],1); return M[iu]
def fc(a,b): return float(np.corrcoef(a,b)[0,1])
def find_pair(meta,tx,ty,snr="high"):
    types,noise = meta["types"],meta["noise_stds"]; want={tx,ty}
    c=[(i,j) for i in range(len(types)) for j in range(i+1,len(types)) if {ABBR[types[i]],ABBR[types[j]]}==want]
    c.sort(key=lambda ij:noise[ij[0]]+noise[ij[1]]); return {"high":c[0],"low":c[-1],"median":c[len(c)//2]}[snr]
def glyph(ax,kind,color="k",lw=1.8):
    t=np.linspace(-1,1,80); y={"line":t,"S":np.tanh(2.5*t),"U":t**2*2-1}[kind]
    ax.plot(t,y,color=color,lw=lw); ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.3,1.3); ax.axis("off")
def sq(ax): ax.set_box_aspect(1)

# ---- illustrative generative model (mother -> filtered channels), matching generate_filter_roll_mts
def _ar1(T,a=0.8,rng=None):
    rng = rng or np.random.default_rng(7); m=np.zeros(T)
    for t in range(1,T): m[t]=a*m[t-1]+rng.normal(0,1)
    return (m-m.mean())/m.std()
def _unit(g): return (g-g.mean())/(g.std()+1e-12)
def _filter(z,typ):
    if typ=="L": return _unit(z)
    if typ=="M": return _unit(1/(1+np.exp(-BETA*z)))
    return _unit((z-z.mean())**2)
GLYPH_OF = {"L":"line","M":"S","NM":"U"}

fig = plt.figure(figsize=(12.5, 17.0))
g = fig.add_gridspec(4, 1, height_ratios=[3.3, 1.55, 2.15, 1.2], hspace=0.40)
_captions = []
def band(ax, letter, text): _captions.append((ax, letter, text))

# ============ Panel A: construction + dependency alphabet ============
ts3, mp3, meta3 = load(RUN/"iter3_LMNM/M32_T2000_I0"); types3 = meta3["types"]
order = np.argsort([{"linear":0,"monotonic":1,"non-monotonic":2}[t] for t in types3], kind="stable")
gsA = g[0].subgridspec(1, 4, width_ratios=[1.15, 1.45, 1.0, 0.95], wspace=0.34)

# -- col 0: mother z (top) + 5 filter glyphs (noisy phase-portraits: shape=family, scatter=SNR)
Tdemo = 400; zdemo = _ar1(Tdemo, a=0.8, rng=np.random.default_rng(7))
SEL = [("linear",       "L",  0.05),
       ("sigmoid hi-SNR","M", 0.05),
       ("sigmoid lo-SNR","M", 0.45),
       ("quad hi-SNR",   "NM", 0.05),
       ("quad med-SNR",  "NM", 0.22)]
gsA0 = gsA[0,0].subgridspec(6, 1, height_ratios=[1.15,1,1,1,1,1], hspace=0.30)
axZ = fig.add_subplot(gsA0[0,0])
axZ.plot(zdemo, color="#222", lw=0.7); axZ.set_title("shared mother  $z_t$", fontsize=8.5, pad=2)
axZ.set_xticks([]); axZ.set_yticks([])
for s in ("top","right"): axZ.spines[s].set_visible(False)
rngn = np.random.default_rng(3)
for k,(lab,typ,ns) in enumerate(SEL):
    axf = fig.add_subplot(gsA0[k+1,0])
    gz = _filter(zdemo, typ); x = gz + rngn.normal(0, ns, Tdemo)
    o = np.argsort(zdemo)
    axf.scatter(zdemo, x, s=2, alpha=0.18, color="#888", linewidths=0)
    axf.plot(zdemo[o], gz[o], color=TYPECOL[typ], lw=1.6)
    axf.set_xlim(-2.6,2.6); axf.set_ylim(-2.8,2.8); axf.set_xticks([]); axf.set_yticks([])
    axf.set_ylabel(lab, fontsize=6.8, rotation=0, ha="right", va="center", labelpad=2)
    for s in axf.spines.values(): s.set_visible(False)
fig.text(0.085, 0.905, "filters  $g(z;\\beta)$  +  noise(SNR)", fontsize=8, rotation=90, va="top", color="#555")

# -- col 1: real type-ordered MTS heatmap (cropped T)
axM = fig.add_subplot(gsA[0,1])
axM.imshow(ts3[:Tdemo, order].T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
axM.set_title("an MTS  (channels × time)", fontsize=9.5); axM.set_xlabel("t", fontsize=9); axM.set_xticks([])
for b in (11.5,23.5): axM.axhline(b, color="k", lw=0.7)
for cen,lab in [(5.5,"L"),(17.5,"M"),(27.5,"NM")]: axM.text(-2.5,cen,lab,fontsize=10,fontweight="bold",va="center",ha="right")
axM.set_yticks([])

# -- cols 2,3: 4 bracket-picked real phase plots + r/rho/MI bars
gsP = gsA[0,2].subgridspec(4, 1, hspace=0.32)
gsB4 = gsA[0,3].subgridspec(4, 1, hspace=0.32)
A_bars = {}
for k,(tx,ty) in enumerate([("L","L"),("L","M"),("L","NM"),("M","NM")]):
    i,j = find_pair(meta3,tx,ty,"high")
    axp = fig.add_subplot(gsP[k,0]); axb = fig.add_subplot(gsB4[k,0]); A_bars[(tx,ty)] = axb
    axp.scatter(ts3[:,i],ts3[:,j],s=3,alpha=0.22,color="#34495e",linewidths=0); sq(axp)
    axp.set_xticks([]); axp.set_yticks([]); axp.set_ylabel(f"{tx}-{ty}",fontsize=9,rotation=0,ha="right",va="center",labelpad=3)
    gi = axp.inset_axes([0.66,0.66,0.32,0.32]); glyph(gi, GLYPH_OF["NM"] if {tx,ty}=={"M","NM"} else GLYPH_OF[ty if tx=="L" else tx], PAIRCOL.get(f"{tx}-{ty}","#444"),lw=1.4)
    r,rho,mi_ = mp3["r"][i,j],mp3["rho"][i,j],linf(mp3["MI"][i,j])
    axb.bar(["r","ρ","MI"],[r,rho,mi_],color=[SPI["r"],SPI["rho"],SPI["MI"]],width=0.74); sq(axb)
    axb.axhline(0,color="grey",lw=0.5); axb.set_ylim(-0.25,1.08); axb.set_yticks([0,0.5,1.0]); axb.tick_params(labelsize=7)
    for s in ("top","right"): axb.spines[s].set_visible(False)
    if k==0: axp.set_title("pick a pair", fontsize=8.5); axb.set_title("r / ρ / MI", fontsize=8.5)
    if (tx,ty)==("L","M"):
        axb.annotate("r<ρ", xy=(0,r), xytext=(0.25,0.42), fontsize=8.5, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
band(axZ,"A","Construction: a shared latent, filtered per channel → each pair inherits a relationship (r/ρ/MI capture hierarchy)")

# ============ Panel B: type-ordered MPI heatmaps (iter3), shared magnitude scale ============
gsB = g[1].subgridspec(1, 7, width_ratios=[0.3,1,0.32,1,0.32,1,0.09], wspace=0.0)
abbr_ord = [ABBR[types3[m]] for m in order]
blocks = {t:[k for k,a in enumerate(abbr_ord) if a==t] for t in ("L","M","NM")}
def span(t): idx=blocks[t]; return idx[0]-0.5, idx[-1]+0.5
mats = {"r": mp3["r"], "rho": mp3["rho"], "MI": linf(mp3["MI"])}
im=None; axB0=None
for c,key in enumerate(("r","rho","MI")):
    ax = fig.add_subplot(gsB[0,1+2*c]); axB0 = axB0 or ax; Mre = mats[key][np.ix_(order,order)].astype(float); np.fill_diagonal(Mre,np.nan)
    im = ax.imshow(Mre, cmap=HEAT_CMAP, vmin=0, vmax=1); sq(ax)
    ax.set_title(key, fontsize=11)
    for idx in blocks.values(): ax.axhline(idx[-1]+0.5,color="w",lw=0.5); ax.axvline(idx[-1]+0.5,color="w",lw=0.5)
    tk=[np.mean(idx) for idx in blocks.values()]; ax.set_xticks(tk); ax.set_yticks(tk)
    ax.set_xticklabels(blocks.keys(),fontsize=8); ax.set_yticklabels(blocks.keys(),fontsize=8)
    for rt in ("L","M"):
        y0,y1 = span(rt); x0,x1 = span("NM")
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor=ACCENT, lw=1.6, zorder=4))
    lbl = "≈0" if key in ("r","rho") else "high"
    cx = np.mean(span("NM")); cy = np.mean(span("L"))
    ax.text(cx, cy, lbl, color="w", fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=5)
cax = fig.add_subplot(gsB[0,6]); fig.colorbar(im, cax=cax).set_label("dependence (r, ρ, MI→Linfoot)", fontsize=8)
band(axB0,"B","Same data as matrices: the (L,NM)/(M,NM) blocks are ~0 in r,ρ but bright in MI")

# ============ Panel C: 3x2 SPI-SPI planes + legend/key column ============
gsC = g[2].subgridspec(3, 3, width_ratios=[1,1,0.85], hspace=0.22, wspace=0.28)
exem = find_pair(meta3,"L","NM","high"); C_target=None; axC0=None
EVENTS = {(1,("r","rho")):"ρ splits from r", (2,("rho","MI")):"MI splits from ρ"}
for ri,it in enumerate(ITERS):
    _,mp,meta = load(RUN/f"{it}/M32_T2000_I0"); types=meta["types"]; iu=np.triu_indices(len(types),1)
    labs=np.array(["-".join(sorted([ABBR[types[i]],ABBR[types[j]]])) for i,j in zip(*iu)])
    same=np.array([l.split("-")[0]==l.split("-")[1] for l in labs])
    for ci,(sx,sy) in enumerate(PLANES):
        ax=fig.add_subplot(gsC[ri,ci]); axC0 = axC0 or ax; sq(ax); xv,yv=mp[sx][iu],mp[sy][iu]
        ax.scatter(xv[same],yv[same],s=13,facecolors="none",edgecolors=SAME,linewidths=0.6,alpha=0.7)
        for lab,col in PAIRCOL.items():
            m=labs==lab
            if m.any(): ax.scatter(xv[m],yv[m],s=18,color=col,alpha=0.8)
        if sx=="r" and sy=="rho": ax.plot([-0.2,1.05],[-0.2,1.05],ls=":",color="grey",lw=0.6)
        ax.text(0.05,0.95,f"$f$={fc(xv,yv):+.2f}",transform=ax.transAxes,va="top",fontsize=9,
                bbox=dict(boxstyle="round",fc="white",ec="grey",alpha=0.85))
        ax.tick_params(labelsize=7)
        if ri==2: ax.set_xlabel(sx,fontsize=9)
        if ci==0: ax.set_ylabel(f"{it.replace('_','-')}\n\n{sy}",fontsize=9)
        else: ax.set_ylabel(sy,fontsize=9)
        if ri==0: ax.set_title(f"({sx},{sy})",fontsize=10)
        if (ri,(sx,sy)) in EVENTS:
            ax.annotate(EVENTS[(ri,(sx,sy))], xy=(0.5,0.5), xytext=(0.42,0.12), textcoords="axes fraction",
                        fontsize=8.5, color=ACCENT, fontweight="bold", ha="center")
        if it=="iter3_LMNM" and (sx,sy)==("r","rho"):
            ax.text(0.5,0.80,"r,ρ co-fail → recorrelate",transform=ax.transAxes,fontsize=7,color="#555",ha="center",style="italic")
        if it=="iter3_LMNM" and (sx,sy)==("rho","MI"):
            a,b=exem; ax.scatter([mp["rho"][a,b]],[mp["MI"][a,b]],s=80,facecolors="none",edgecolors=ACCENT,linewidths=2,zorder=5)
            C_target=(ax,mp["rho"][a,b],mp["MI"][a,b])
# legend / capture-hierarchy key (fills the freed third column)
axL = fig.add_subplot(gsC[:,2]); axL.axis("off")
axL.set_xlim(0,1); axL.set_ylim(0,1); axL.set_autoscale_on(False)   # all legend artists in axes coords
TT = axL.transAxes
axL.text(0.0,0.98,"capture hierarchy",fontsize=9.5,fontweight="bold",va="top",transform=TT)
axL.text(0.0,0.92,"$r \\subset \\rho \\subset$ MI",fontsize=11,va="top",transform=TT)
for yy,(s,nm) in zip([0.83,0.78,0.73],[("r","Pearson — linear"),("rho","Spearman — monotone"),("MI","MI — any")]):
    axL.add_patch(Rectangle((0.0,yy),0.05,0.03,color=SPI[s],transform=TT))
    axL.text(0.08,yy+0.015,nm,fontsize=8,va="center",transform=TT)
axL.text(0.0,0.66,"pair filter family",fontsize=9.5,fontweight="bold",va="top",transform=TT)
for yy,(kind,nm) in zip([0.53,0.45,0.37],[("line","L: linear"),("S","M: monotone-nl"),("U","NM: non-monotone")]):
    gi = axL.inset_axes([0.0,yy,0.13,0.07]); glyph(gi,kind,"#333",lw=1.3)
    axL.text(0.17,yy+0.035,nm,fontsize=8,va="center",transform=TT)
axL.text(0.0,0.27,"pair type",fontsize=9.5,fontweight="bold",va="top",transform=TT)
axL.scatter([0.025],[0.20],s=20,facecolors="none",edgecolors=SAME,transform=TT)
axL.text(0.08,0.20,"same-type (X-X)",fontsize=8,va="center",transform=TT)
for yy,(lab,col) in zip([0.14,0.09,0.04],PAIRCOL.items()):
    axL.scatter([0.025],[yy],s=20,color=col,transform=TT); axL.text(0.08,yy,lab,fontsize=8,va="center",transform=TT)
band(axC0,"C","Pairs populate the SPI–SPI planes across iterations; $f_{ij}$ = their correlation")

# ============ Panel D: raincloud ============
gsD = g[3].subgridspec(1, 3, width_ratios=[0.12,2.0,0.12])
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
band(axD,"D","The staggered signature over 30 instances")

# ============ Telescope (one exemplar L-NM pair); adjacent hops only ============
if C_target:
    axc,rx,my = C_target
    fig.add_artist(ConnectionPatch(xyA=(xD,y0),coordsA=axD.transData,xyB=(0.5,-0.12),coordsB=axc.transAxes,
                                   color=ACCENT,lw=1.0,alpha=0.7,arrowstyle="-",linestyle="--"))
    fig.add_artist(ConnectionPatch(xyA=(rx,my),coordsA=axc.transData,xyB=(0.5,-0.22),coordsB=A_bars[("L","NM")].transAxes,
                                   color=ACCENT,lw=1.0,alpha=0.7,arrowstyle="-",linestyle="--"))

fig.canvas.draw()
for ax,letter,text in _captions:
    p = ax.get_position(); y = p.y1 + 0.010
    fig.text(0.055, y, letter, fontsize=17, fontweight="bold", va="bottom")
    fig.text(0.105, y, text, fontsize=10.0, va="bottom")

fig.savefig("notebooks/presentation/r_rho_mi_figure_v4.png", dpi=110, bbox_inches="tight")
print("saved notebooks/presentation/r_rho_mi_figure_v4.png")
