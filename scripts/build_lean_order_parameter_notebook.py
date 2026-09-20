"""Build a lean, executed presentation without modifying the source notebook."""
import argparse
import ast
import hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'notebooks/inference/order-parameter-benchmark-comparison.ipynb'


class Quiet(ast.NodeTransformer):
    """Remove console/tables, retain actual plotting and displayed images."""
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name):
            if node.value.func.id=='print':
                return ast.Pass()
            if node.value.func.id=='display' and not any(
                isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='Image'
                for x in node.value.args):
                return ast.Pass()
        return self.generic_visit(node)


def build():
    original=nbformat.read(SOURCE,as_version=4)
    cells=[]
    def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip()))
    def code(s): cells.append(nbformat.v4.new_code_cell(s.strip()))
    def reuse(index):
        s=ast.unparse(ast.fix_missing_locations(Quiet().visit(ast.parse(original.cells[index].source))))
        # No figure writes into the original figure directories.
        s=s.replace('notebooks/inference/figures','notebooks/inference/figures/lean')
        code(s)
    md(r'''# Physical order and regime tracking with SPI–SPI

For each recording $X\in\mathbb R^{M\times T}$, all 289 p90 SPIs produce pairwise-channel matrices. Flatten their off-diagonal entries as $v_a$, then form $z_{ab}=\operatorname{corr}(v_a,v_b)$. A development-only validity mask, imputation, centring and first principal component define the frozen scalar $q=w^\top(\widetilde z-\mu)/s$. Neither control nor physical target $Q$ fits this representation. Display sign is arbitrary; separate systems use separate fitted coordinates.

The question is **across-control tracking of an independently defined physical quantity**, not discovering its formula, numerical calibration, or recovering its fluctuations through time. Black circles show $Q$; coloured squares show $q$ on a separate axis. Colours identify $M$; opacity identifies $T$. Bands are 95% bootstrap intervals for the mean across seeds. Published infinite-size boundaries are references, not assumed finite-system thresholds.

The three snapshots are quick visual checks, not evidence selected for a good match. They show 100 consecutive samples from $T=1000$ inputs, robust-scaled per channel; row order in dispersed observations is **not** spatial adjacency. Full fields use raw values. Regenerated illustrations use fixed seeds and are not additional inference trials. [Full comparison and provenance](order-parameter-benchmark-comparison.ipynb).''')
    reuse(2)
    code('''import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from IPython.display import Image, display
from scripts.lean_benchmark_figures import snapshots, composite, corpus_examples
from scripts.cml2d_figure_diagnostics import export_snapshots
T_ALPHA={100:.38,500:.68,1000:1.0}
COLORS[6]="#31688e"
''')
    md(r'''## 1. Kuramoto — phase coherence

**System**

$$\dot\theta_i=\omega_i+\frac K N\sum_j\sin(\theta_j-\theta_i).$$

**Params**

- $\theta_i$: oscillator phase; $N$: population size.
- $\omega_i$: fixed Gaussian natural frequencies, mean 1 and SD 1.
- $K$: all-to-all coupling; $\kappa=K/K_c$, with continuum $K_c=\sqrt{8/\pi}$ for this frequency law.

**Phase diagram paper:** [Xu, Wang & Skardal (2020)](https://arxiv.org/html/2007.02383v1#S2) — Sec. II, discussion following Eq. (7): Gaussian synchronization onset at $K_c=2/[\pi g(0)]$; Sec. IV.1 treats continuous transitions.

**Order-param paper:** [Same paper](https://arxiv.org/html/2007.02383v1#S2) — Eq. (2), complex phase-coherence order parameter.

**Control-parameter sweep:** $\kappa\in[.625,1.65]$, 16 nonuniform points: $.625,.725,.825,.8875,.9425,.9775,.9925,1.0075,1.0225,1.0575,1.0875,1.175,1.25,1.375,1.525,1.65$. Existing result: $N=256,M=20,T=1000$. New full-population pilot: $M=N=32,T=1000$, same grid, eight development and eight held seeds. No completed multi-$T$ sensitivity bank is available.

**Order-parameter**

$$R(t)=\left|N^{-1}\sum_j e^{i\theta_j(t)}\right|,\qquad Q=\langle R(t)\rangle_{\rm future}.$$

The length of the average unit arrow: scattered phases cancel, aligned phases reinforce. Increasing coupling promotes coherence. The finite population rounds the continuum onset at $\kappa=1$.

**Results:** partial-observation retrospective $|\rho(q,Q)|=.956$. New full-population held-seed pilot: $\rho=.826$ (95% seed-bootstrap CI $.706$–$.941$), control-mean $\rho=.997$, all 128 held rows eligible. It misses the prechosen PC1-dominance screen, not the coverage or seed-stability checks; see the qualification below. Different $N$ arms are not a fixed-physics $M$ sensitivity.''')
    reuse(6)
    md(r'''### Full-population result: $M=N=32,T=1000$

The $1.5$ cutoff is a **workflow heuristic**, inherited from the [earlier CML protocol](../../docs/research/order-parameter-benchmarks/cml2d-protocol-260911.md), requiring $\lambda_1/\lambda_2\geq1.5$. It is not a physical law, a calibrated significance test, or a necessary condition for tracking $Q$. Here PC1/PC2 explain $27.76\%/18.78\%$, giving $1.478$; minimum leave-seed loading cosine is $.897$ (required $.8$), and coverage passes. The original screen remains unmet; it does **not** establish absence of order-related information. The plots show exploratory held-seed tracking by the unchanged, target-blind PC1—not a separately confirmed result or proof of a uniquely one-dimensional representation.''')
    code('''kur_full_root=ROOT/"data/order_parameter/kuramoto_full_observation_260916/primary"
kfull=None
if (kur_full_root/"analysis/summary.json").exists():
    report=json.loads((kur_full_root/"analysis/summary.json").read_text())
    if (kur_full_root/"analysis/scores.csv").exists():
        kfull=pd.read_csv(kur_full_root/"analysis/scores.csv").query("role == 'evaluation' and eligible")
        fig,axes=plt.subplots(1,2,figsize=(8.8,3.35),constrained_layout=True)
        dual_tracking(axes[0],kfull,control="control",Q="Q_reference",q="q",q_sign=report["display_sign"],
            q_color=COLORS[32],boundary=1,title=r"Full-population pilot: $M=N=32,T=1000$")
        points=recovery_scatter(axes[1],kfull,Q="Q_reference",q="q",control="control",q_sign=report["display_sign"],title=rf"Held-seed tracking: $|\\rho|={report['results'][0]['rho']:.3f}$")
        axes[0].set_xlabel(r"reduced coupling $\\kappa$")
        fig.colorbar(points,ax=axes[1],label=r"reduced coupling $\\kappa$")
        if not report["passes"]: fig.suptitle(r"Exploratory tracking; PC1-dominance screen not met ($1.478 < 1.5$)")
        fig.savefig(FIGURE_DIR/"kuramoto-full-headline.png",dpi=300,bbox_inches="tight")
        fig.savefig(FIGURE_DIR/"kuramoto-full-headline.svg",bbox_inches="tight")
        plt.show()
    else:
        print("Full-population analysis failed:",report.get("reason",report["status"]))
else:
    print("M=N=32 p90 results pending. Available completed arm: M=20, T=1000, N=256; no multi-T results.")''')
    code('''def tracking(ax):
    dual_tracking(ax,kuramoto,control="kappa",Q="Q",q="q",q_sign=-1,q_color=COLORS[20],boundary=1,
        title=r"Retrospective partial observation: $M=20,N=256$; $|\\rho|=0.956$")
composite(snapshots(ROOT,"kuramoto",[.625,1.0075,1.65]),[.625,1.0075,1.65],r"\\kappa",tracking,
    title="Kuramoto: phase alignment",output=FIGURE_DIR/"kuramoto-composite",
    field_label="All 256 oscillators (not a spatial lattice)",trace_label=r"Global $R(t)$")
plt.show()''')
    md(r'''### Full-population snapshots: $M=N=32$

Same 16 controls; 16 fresh Gaussian populations, eight development and eight held-out seeds; $T=1000$, $\Delta t_{\rm sample}=.1$, integration step .02, burn 200, disjoint 10,000-sample future $Q$. Every oscillator contributes one cosine channel: full **oscillator** coverage, not both quadratures. Snapshots use the first held seed and fixed control anchors, not selection for a good-looking result. The bottom row repeats the held-seed ensemble result, not within-time tracking.''')
    code('''if kfull is not None and len(kfull):
    def tracking(ax):
        dual_tracking(ax,kfull,control="control",Q="Q_reference",q="q",q_sign=report["display_sign"],
            q_color=COLORS[32],boundary=1,title="M=N=32: held-seed tracking" + ("" if report["passes"] else "; dominance screen unmet"))
        ax.set_xlabel(r"reduced coupling $\\kappa$")
    composite(corpus_examples(kur_full_root,[.625,1.0075,1.65],260916009),[.625,1.0075,1.65],r"\\kappa",tracking,
        title="Kuramoto: all 32 oscillators observed",output=FIGURE_DIR/"kuramoto-full-composite")
    plt.show()''')
    md(r'''## 2. Stuart–Landau — collective oscillation amplitude

**System**

$$\dot z_j=(1-|z_j|^2+i\omega_j)z_j+K(Z-z_j),\qquad Z=N^{-1}\sum_jz_j.$$

**Params**

- $z_j$: complex amplitude and phase; $-|z_j|^2z_j$ saturates amplitude.
- $K=.8$: coupling to the population mean $Z$.
- $\omega_j$: uniform midpoint grid on $[2-\gamma,2+\gamma]$; $\gamma$ controls frequency spread.

**Phase diagram paper:** [Matthews & Strogatz (1990)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.65.1701) — Fig. 2(a), $(K,\gamma)$ phase diagram; Fig. 2(b), unsteady-region detail. The locking–unsteady line is numerical.

**Order-param paper:** [Same paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.65.1701) — Eq. (2), complex mean field; Fig. 1, its time-varying magnitude at $K=.8$.

**Control-parameter sweep:** broad $\gamma\in[.55,1.25]$, nine nonuniform points $.55,.65,.725,.775,.85,.95,1.05,1.15,1.25$; fine $\gamma\in[.680,.770]$, step $.005$, 19 points. Full oscillator coverage $M=N\in\{8,16,32\}$, real-part channels, $T\in\{100,500,1000\}$; fine confirmation uses $M=N=32,T=1000$ and eight fresh seeds.

**Order-parameter**

$$R(t)=|Z(t)|,\qquad Q=\langle R(t)\rangle_{\rm future}.$$

The collective oscillation amplitude. Increasing frequency spread disrupts locking; $R(t)$ can itself oscillate. Its mean $Q$ and variability $\operatorname{sd}_t(R)$ describe different aspects, both shown below.

**Results:** fine-boundary confirmation $|\rho(q,Q)|=.886$; $q$ and $Q$ share the steepest sampled interval $.720$–$.725$. This is a finite-$N$ result, not an exact thermodynamic boundary.''')
    reuse(9);reuse(10);reuse(12);reuse(8)
    code('''sl_headline=sl_primary.query("M == 32")
def tracking(ax):
    dual_tracking(ax,sl_headline,control="gamma",Q="Q_R_mean",q="q",q_sign=1,q_color=COLORS[32],
        title=r"Across-control collective amplitude: $M=N=32,T=1000$")
composite(snapshots(ROOT,"stuart-landau",[.55,.725,1.25]),[.55,.725,1.25],r"\\gamma",tracking,
    title="Stuart--Landau: broad collective-regime sweep",output=FIGURE_DIR/"stuart-landau-composite",trace_label=r"Global $|Z(t)|$")
plt.show()''')
    md(r'''## 3. Miller–Huse — sign-domain symmetry breaking

**System**

$$x_{ij}(t+1)=(1-4g)f(x_{ij}(t))+g\sum_{(k,l)\in\mathrm{nn}(i,j)}f(x_{kl}(t)),\qquad f(x)=\begin{cases}-2-3x&x<-1/3,\\3x&|x|\leq1/3,\\2-3x&x>1/3.\end{cases}$$

**Params**

- $x_{ij}\in[-1,1]$: chaotic site state; piecewise-map slope magnitude fixed at $\mu=3$.
- $g$: coupling to four nearest neighbours; synchronous updates and periodic boundaries.
- $L=128$: square-lattice side; $N=L^2=16384$ sites.

**Phase diagram paper:** [Marcq, Chaté & Manneville (1997)](https://doi.org/10.1103/PhysRevE.55.2606) — pp. 2610–2611, order/susceptibility across coupling; finite-size critical estimates in Table I. Published $g_c=.20534(2)$; this is a one-control cut.

**Order-param paper:** [Same paper](https://doi.org/10.1103/PhysRevE.55.2606) — Eqs. (6)–(7), p. 2610: instantaneous spin magnetization and its mean absolute value. Original model: [Miller & Huse (1993)](https://doi.org/10.1103/PhysRevE.48.2528).

**Control-parameter sweep:** plotted confirmation $g\in[.185,.225]$, nine nonuniform points $.185,.195,.20125,.20325,.20470,.20517,.20570,.20875,.225$. Dispersed $M\in\{8,16,32\}$, $T=1000$; eight fresh seeds; disjoint two-million-step reference. No multi-$T$ results are available.

**Order-parameter**

$$Q=\left\langle\left|L^{-2}\sum_{ij}\operatorname{sign}x_{ij}(t)\right|\right\rangle_t.$$

The imbalance between positive and negative domains. Stronger coupling allows one sign to dominate; the absolute value prevents whole-system sign flips from cancelling order.

**Results:** $\rho=.888$ pooled across $M$ in the 215/216-row exclusion sensitivity. The strict confirmation failed one row; this is not pristine confirmation.''')
    reuse(14)
    code('''mh_headline=mh.query("M == 32")
def tracking(ax):
    dual_tracking(ax,mh_headline,control="g",Q="Q_spin_abs",q="q",q_sign=1,q_color=COLORS[32],boundary=.20534,
        title="Confirmation exclusion sensitivity; strict gate failed")
composite(snapshots(ROOT,"miller-huse",[.185,.20517,.225]),[.185,.20517,.225],"g",tracking,
    title="Miller--Huse: chaotic sites, ordered sign domains",output=FIGURE_DIR/"miller-huse-composite",
    lattice=True,field_label="L=128 lattice; white circles mark sensors",trace_label=r"Global $m_s(t)$")
plt.show()
''')
    md(r'''## 4. Quadratic/Kaneko CML — pattern and temporal complexity diagnostics

**System**

$$x_i(t+1)=(1-\epsilon)f_\alpha(x_i(t))+\frac\epsilon2[f_\alpha(x_{i-1}(t))+f_\alpha(x_{i+1}(t))],\qquad f_\alpha(x)=1-\alpha x^2.$$

**Params**

- $\alpha$: local map nonlinearity.
- $\epsilon=.3$: nearest-neighbour mixing on a periodic **one-dimensional ring**.
- $N=512$: physical ring size; observations are dispersed sites.

**Phase diagram paper:** [Kaneko (1989)](https://doi.org/10.1016/0167-2789(89)90227-3) — pattern-selection, defect and pattern-competition/intermittency discussions. This is qualitative regime evidence, not an exact numerical boundary for our cut. [Long-transient caveat for the quadratic lattice](https://chaos.phys.msu.ru/loskutov/PDF/TMPh_quadratic_cml.PDF).

**Order-param paper:** no canonical order-parameter source for these exact two diagnostics. [Kaneko (1989)](https://doi.org/10.1016/0167-2789(89)90227-3) discusses entropy/spectral quantification of the regimes; our precise operational definitions are below and in [the diagnostic implementation](../../src/cml_order_parameter.py). They are not claimed as Kaneko's published order parameters.

**Control-parameter sweep:** $\alpha\in[1.60,2.00]$, step $.01$, 41 points; $M\in\{8,16,32\}$, $T=1000$. Four development and four held seeds; two-million-step burn, 20,000-step future diagnostics. No multi-$T$ results are available.

**Order-parameter / physical diagnostics**

$$Q_H=-\frac{\sum_{k=1}^{B}p_k\log p_k}{\log B},\qquad Q_P=\sum_{k/\pi\in[.25,.45]}p_k^{\rm space}.$$

Here $p_k$ is normalized non-DC temporal power averaged over sites, and $p_k^{\rm space}$ is normalized spatial power averaged over time. $Q_H$ measures narrow-band versus broadband activity; $Q_P$ measures concentration in a fixed range of spatial wavelengths.

**Results:** exploratory regime tracking with the same frozen $q_1$ for both diagnostics, not canonical order-parameter recovery. The reorganisation near $1.74$–$1.76$ depends on basin and finite time; full-ring images are site-versus-time plots.''')
    code('''cml_headline=cml_held.query("M == 32")
cml_targets=[("Q_selected_band_power","Selected spatial-band power","band-power"),
             ("Q_temporal_entropy","Temporal spectral entropy","temporal-entropy")]
fig,axes=plt.subplots(1,2,figsize=(9,3.35),constrained_layout=True)
for ax,(target,label,_) in zip(axes,cml_targets):
    dual_tracking(ax,cml_headline,control="alpha",Q=target,q="q1",q_sign=-1,
        q_color=COLORS[32],Q_label=label,title=label+": M=32, T=1000")
    ax.set_ylabel(label)
    ax.get_legend().set_loc("center left")
    ax.axvspan(1.74,1.76,color="#d95f02",alpha=.1,lw=0)
plt.show()
for target,label,_ in cml_targets:
    fig=size_tracking(cml_held,control="alpha",Q=target,q="q1",q_sign=-1,boundary=None,title=label+": observation-size sensitivity")
    for ax in fig.axes:
        if ax.get_ylabel()==r"physical $Q$": ax.set_ylabel(label)
    plt.show()''')
    reuse(16)
    code('''cml_headline=cml_held.query("M == 32")
for target,label,slug in [("Q_selected_band_power","Selected spatial-band power","band-power"),
                          ("Q_temporal_entropy","Temporal spectral entropy","temporal-entropy")]:
    def tracking(ax):
        dual_tracking(ax,cml_headline,control="alpha",Q=target,q="q1",q_sign=-1,q_color=COLORS[32],
            Q_label=label,title=label+": same frozen q (exploratory)")
        ax.set_ylabel(label)
        ax.get_legend().set_loc("center left")
        ax.axvspan(1.74,1.76,color="#d95f02",alpha=.10,lw=0)
    composite(snapshots(ROOT,"quadratic-cml",[1.6,1.75,2.0]),[1.6,1.75,2.0],r"\\alpha",tracking,
        title="Quadratic CML: "+label,output=FIGURE_DIR/("quadratic-cml-"+slug),field_label="Full 512-site ring: space-time")
    plt.show()
''')
    md(r'''## 5. Two-dimensional logistic CML — collective period two

**System**

$$x_{ij}(t+1)=(1-4g)f_r(x_{ij}(t))+g\sum_{(k,l)\in\mathrm{nn}(i,j)}f_r(x_{kl}(t)),\qquad f_r(x)=rx(1-x).$$

**Params**

- $r$: local logistic-map parameter; $g=.2$: four-neighbour coupling.
- $L$: periodic square-lattice side; $N=L^2$; synchronous updates.
- Primary $L=256$, $N=65536$; small full-observation diagnostics use $L=6,8$.

**Phase diagram paper:** [Marcq, Chaté & Manneville (2006)](https://arxiv.org/html/nlin/0605004) — Fig. 1, collective bifurcation diagram at $g=.2$; Fig. 3 and Sec. 2, extrapolated boundary $r_c=3.86212(12)$; Sec. 3, other cuts in $(r,g)$.

**Order-param paper:** [Same paper](https://arxiv.org/html/nlin/0605004) — Eqs. (3)–(4), global mean and period-two order parameter.

**Control-parameter sweep:** $r\in[3.84,3.89]$, 17 nonuniform confirmation points: $3.84,3.845,3.85,3.854,3.858,3.86006,3.86212,3.86306,3.864,3.865,3.866,3.868,3.87,3.8725,3.875,3.8825,3.89$ (nine pilot points plus midpoints). Primary $M=32,T=1000$; pilot $M\in\{8,16,32\}$, $T\in\{100,500,1000\}$; independent paired confirmation $M\in\{16,32\}$, $T\in\{500,1000\}$. Eight pilot / 32 fresh confirmation seeds; 200,000-step burn and disjoint million-step reference.

**Order-parameter**

$$\bar x(t)=L^{-2}\sum_{ij}x_{ij}(t),\qquad Q=\langle|\bar x(2t+1)-\bar x(2t)|\rangle_t.$$

The separation between alternating global-mean levels. Increasing $r$ merges the two levels while individual sites remain chaotic. Mean activity itself is not $Q$; the quoted boundary is numerical and infinite-size, not exact for finite $L$.

**Results:** frozen dispersed-view confirmation $\rho=.883$, all 1408 primary/secondary records valid. Contiguous-patch transfer and small full-lattice tests failed their respective gates; these remain labelled diagnostics.''')
    code('''CML2D_ROOT=ROOT/"data/order_parameter/cml2d_period_doubling_260911"
cml2d_snapshot_dir=FIGURE_DIR/"cml2d-period-doubling"
cml2d_scores=pd.read_csv(CML2D_ROOT/"primary-analysis/scores.csv")''')
    reuse(42);reuse(43);reuse(37);reuse(32)
    md(r'''### Three-control composite

The top row is the global-mean time trace, not $Q$. The lattice images are later saved fields at the same control and seed, not the exact first 100 displayed steps. The bottom row is the independent across-control confirmation. Changing $M$ or $T$ can shift $q$; no exact invariance or arbitrary-patch transfer is claimed.''')
    reuse(36)
    md(r'''### Full small lattices: diagnostic failures, not successful inference

$L=6$ gives $M=N=36$; $L=8$ gives $M=N=64$. Both use the same 17 controls, $T=1000$, eight development and 32 held seeds, and the same burn/reference lengths. $L=6$ retains all rows but its fitted PC1 fails seed-stability; plotted $q$ is **descriptive only**. $L=8$ has constant-channel records and stopped before p90. Grey MTS rows mask numerical constancy. These finite lattices often settle onto periodic attractors, so they are not small replicas of the large-lattice chaotic transition.''')
    reuse(47);reuse(49);reuse(51)
    md(r'''## 6. Coupled Rössler oscillators — mean-frequency entrainment

**System**

$$\dot x_i=-\omega_i y_i-z_i+C(x_j-x_i),\qquad \dot y_i=\omega_i x_i+.15y_i,\qquad \dot z_i=.2+z_i(x_i-10),\quad j\ne i.$$

**Params**

- $(x_i,y_i,z_i)$: state of oscillator $i\in\{1,2\}$; local parameters fixed at $.15,.2,10$.
- $\omega_{1,2}=1\pm\Delta\omega$, with $\Delta\omega=.015$: intrinsic frequency mismatch.
- $C$: diffusive coupling through the two $x$ coordinates.

**Phase diagram paper:** [Rosenblum, Pikovsky & Kurths (1996)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.76.1804) — Fig. 2, frequency-entrainment region in $(C,\Delta\omega)$; Fig. 1(a), phase drift versus locking at fixed $\Delta\omega=.015$.

**Order-param paper:** [Same paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.76.1804) — Fig. 2 and accompanying text, mean-frequency difference $\Delta\Omega$. This is a dynamical physical quantity, not a thermodynamic order parameter.

**Control-parameter sweep:** $C\in[.015,.040]$, step $.00125$, 21 points. All six state coordinates observed: $M=N_{\rm state}=6$ (two oscillators), $T=1000$; sample interval $.2$, integration step $.01$, burn 2000, disjoint reference duration 100,000. Independent confirmation uses 32 fresh seeds. No multi-$M,T$ results are available.

**Order-parameter / physical quantity**

$$\phi_i=\operatorname{unwrap}\operatorname{atan2}(y_i,x_i),\qquad Q=|\langle\dot\phi_1\rangle-\langle\dot\phi_2\rangle|.$$

The difference in average rotation rates. Stronger coupling drives it toward zero without requiring identical chaotic amplitudes.

**Results:** frozen confirmation $\rho=.848$; $Q$ and $q$ share the steepest sampled interval $.02625$–$.0275$. Heatmap colours saturate at $\pm3$ robust units to keep $x,y$ visible alongside rare $z$ bursts; inputs are unchanged.''')
    code('''rossler_root=ROOT/"data/order_parameter/finite_regime_260915/rossler/confirmation/primary"
rossler=pd.read_csv(rossler_root/"analysis/scores.csv").query("eligible")
rossler_summary=json.loads((rossler_root/"analysis/summary.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(8.8,3.35),constrained_layout=True)
dual_tracking(axes[0],rossler,control="control",Q="Q_reference",q="q",q_sign=rossler_summary["display_sign"],
    q_color=COLORS[6],title=r"Independent confirmation: $|\\rho|=0.848$",Q_label="frequency mismatch Q")
axes[0].set(xlabel=r"Coupling $C$",ylabel="frequency mismatch Q")
axes[0].axvspan(.02625,.0275,color="#31688e",alpha=.1,lw=0)
points=recovery_scatter(axes[1],rossler,Q="Q_reference",q="q",control="control",q_sign=rossler_summary["display_sign"],title="Held-out physical-quantity recovery")
fig.colorbar(points,ax=axes[1],label=r"Coupling $C$")
plt.show()''')
    code('''def tracking(ax):
    dual_tracking(ax,rossler,control="control",Q="Q_reference",q="q",q_sign=rossler_summary["display_sign"],
        q_color=COLORS[6],title=r"Independent confirmation: $|\\rho(q,Q)|=0.848$",Q_label=r"frequency mismatch $Q$")
    ax.set(xlabel=r"Coupling $C$",ylabel=r"frequency mismatch $Q$")
    ax.axvspan(.02625,.0275,color="#31688e",alpha=.1,lw=0)
composite(corpus_examples(rossler_root,[.015,.0275,.04],2609151001),[.015,.0275,.04],"C",tracking,
    title="Rössler: all six state coordinates",output=FIGURE_DIR/"rossler-composite",
    row_labels=[r"$x_1$",r"$y_1$",r"$z_1$",r"$x_2$",r"$y_2$",r"$z_2$"],robust_limit=3)
plt.show()''')
    return nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'source_notebook_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-execute',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'notebooks/inference/order-parameter-benchmarks-lean.ipynb')
    args=p.parse_args();n=build()
    if not args.no_execute:
        NotebookClient(n,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(n,args.output)
    print(args.output)
