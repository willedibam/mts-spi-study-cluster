"""Read-only, result-driven comparison-notebook cells for the CML2D benchmark."""
import nbformat
from scripts.cml2d_confirmation_cells import build_cells as confirmation_cells
from scripts.cml2d_full_observation_cells import build_cells as full_observation_cells


def build_cells():
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    return [md(r'''# 7. Collective period doubling in a two-dimensional CML

This is **not** the earlier one-dimensional Kaneko sweep or an Ising spin model.
The published synchronous periodic-square model is
\[
x_{ij}^{t+1}=(1-4g)f_r(x_{ij}^t)+g\sum_{\text{four neighbours}}f_r(x^t),
\quad f_r(x)=rx(1-x),\quad g=.2.
\]
Its collective period-1/period-2 boundary has numerical thermodynamic estimate
$r_c=3.86212(12)$, not an exact analytical solution. The published order parameter is
\[
Q=\langle|\bar x(2t+1)-\bar x(2t)|\rangle_t,\quad
\bar x(t)=N^{-1}\sum_i x_i(t).
\]
It measures alternating **global** activity; the absolute value comes after
spatial averaging. Microscopic sites remain chaotic in either collective phase.
Finite lattices have a positive fluctuation floor on the period-one side.
[Model, observable and critical study](https://arxiv.org/html/nlin/0605004).

Physical size $N=L^2$ is distinct from recorded $M$. Dispersed fixed sites are
primary; nested contiguous patches are a sensitivity. Site locations are drawn
independently of Q and q.
All observations retain every time step. The long-reference Q uses the portion
after the maximal 2,000-step observation; matched-window Q is a separate
diagnostic. This is an across-control benchmark, not a new requirement that
the earlier systems recover spontaneous within-trajectory switching.
[Protocol and execution notes](../../docs/research/order-parameter-benchmarks/cml2d-protocol-260911.md).'''),
    code(r'''CML2D_ROOT = ROOT / "data/order_parameter/cml2d_period_doubling_260911"
cml2d_physics = pd.read_csv(CML2D_ROOT / "physics-analysis/physics.csv")
cml2d_sizes = cml2d_physics.copy()
if (CML2D_ROOT / "convergence-analysis/physics.csv").exists():
    cml2d_convergence = pd.read_csv(CML2D_ROOT / "convergence-analysis/physics.csv")
    cml2d_sizes = pd.concat([cml2d_sizes,cml2d_convergence.query("burn == 20000 and record_steps == 42000")])
display(cml2d_sizes.groupby(["L", "r"]).agg(
    mean_Q=("Q", "mean"), minimum_Q=("Q", "min"), maximum_Q=("Q", "max"),
    maximum_half_difference=("half_difference", "max"),
).round(5))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
for L, group in cml2d_sizes.groupby("L"):
    curve = group.groupby("r").agg(Q=("Q", "mean"), lo=("Q", "min"), hi=("Q", "max"), drift=("half_difference", "max"))
    axes[0].plot(curve.index, curve.Q, "o-", ms=3, label=rf"$L={L}, N={L*L}$")
    axes[0].fill_between(curve.index, curve.lo, curve.hi, alpha=.12)
    axes[1].plot(curve.index, curve.drift, "o-", ms=3)
for ax in axes:
    ax.axvline(3.86212, color=".5", ls=":", lw=1)
    ax.set_xlabel(r"logistic parameter $r$"); paper_axis(ax)
axes[0].set(ylabel=r"collective $Q$", title="Size audit: seed range, not confidence interval")
axes[0].legend(frameon=False, fontsize=7)
axes[1].set(ylabel="absolute half-mean difference", title="Time-precision diagnostic")
plt.show()
if (CML2D_ROOT / "convergence-analysis/physics.csv").exists():
    cml2d_convergence = pd.read_csv(CML2D_ROOT / "convergence-analysis/physics.csv")
    display(cml2d_convergence[["L","r","start","burn","record_steps","Q","half_difference","block_mean_se"]].round(5))
if (CML2D_ROOT / "primary-physics/physics.csv").exists():
    cml2d_primary_physics = pd.read_csv(CML2D_ROOT / "primary-physics/physics.csv")
    print("Primary physics: N=65,536; 200,000-step burn; disjoint 1,000,000-step reference; eight seeds.")
    display(cml2d_primary_physics.groupby("r").agg(
        mean_Q=("Q","mean"), minimum_Q=("Q","min"), maximum_Q=("Q","max"),
        max_half_difference=("half_difference","max"), max_block_SE=("block_mean_se","max"),
    ).round(5))
    print("Near-boundary block dependence remains: these are finite-run reference values, not exact equilibrium estimates.")
    raw = pd.read_csv(CML2D_ROOT / "primary-physics/observations.csv").merge(
        cml2d_primary_physics[["path","Q"]],on="path",validate="many_to_one")
    raw_rows=[]
    for (view,M),group in raw.query("seed >= 26091115 and T == 1000").groupby(["view","M"]):
        raw_rows.append({"view":view,"M":M,
            "sampled Q vs future Q":spearman(group.sample_Q,group.Q),
            "mean absolute correlation vs future Q":spearman(group.mean_abs_correlation,group.Q)})
    print("Raw-sensor audit, held seeds, T=1000 — these are NOT SPI–SPI results:")
    display(pd.DataFrame(raw_rows).round(4))
    print("No clear raw-observability need for M=64; no M=64 p90 extraction is implied.")'''),
    md(r'''### Three headline MTS snapshots

Start, published-boundary neighbourhood, and end of the tested range:
$r=3.84,3.86212,3.89$. All use the **first held seed, 26091115**, the same
32 dispersed sensor identities, and the first 100 consecutive steps of the
actual primary input ($T=1000$, burn 200,000). No appearance-based selection
or temporal decimation. The black traces show the **global mean over those
exact same steps**, with common vertical limits.

Heatmaps use per-process robust display scaling and a common colour limit;
adjacent sensor rows are not spatial neighbours. These are brief input
illustrations, not estimates of Q. Full 1,000-step heatmaps are also exported.
Below each MTS panel, the lattice heatmap shows the **saved late-time field
from that same run and r value**, with its 32 sensors marked by white circles.
The three lattices share a raw [0,1] colour scale; they are later snapshots,
not the time of the MTS panels.'''),
    code(r'''from scripts.cml2d_figure_diagnostics import export_snapshots
cml2d_snapshot_dir = ROOT / "notebooks/inference/figures/cml2d-period-doubling"
cml2d_snapshot_manifest = export_snapshots(CML2D_ROOT, cml2d_snapshot_dir)
from IPython.display import Image
display(Image(filename=str(cml2d_snapshot_dir / "headline-mts-snapshots.png")))
print("Input/master hashes and exact snapshot timing: ", cml2d_snapshot_dir / "manifest.json")'''),
    md(r'''### Layout preview only — add the actual order-tracking result

The existing figure above is unchanged. The upper three rows retain the same
single-run illustrations. The new bottom panel shows physical $Q$ and frozen
$q$ across $r$: means over **32 independent confirmation seeds**, with 95%
bootstrap intervals. These are fresh seeds, not the illustrative seed above.
Its horizontal axis is control $r$, **not time**. Each $q$ is one coordinate
per $M=32,T=1000$ recording; $Q$ is the long-reference full-system order.
Separate y axes retain their different units; no $q$-to-$Q$ calibration is implied.'''),
    code(r'''cml2d_preview_root = ROOT / "data/order_parameter/cml2d_confirmation_260911/primary-analysis"
cml2d_preview_frame = pd.read_csv(cml2d_preview_root / "scores.csv").query("eligible")
cml2d_preview_sign = json.loads((cml2d_preview_root / "summary.json").read_text())["display_sign"]
def cml2d_preview_tracking(ax):
    dual_tracking(ax, cml2d_preview_frame, control="r", Q="Q_reference", q="q",
        q_sign=cml2d_preview_sign, q_color=COLORS[32], boundary=3.86212,
        title=r"Order tracking across $r$ (not time): 32 independent confirmation seeds")
    ax.set_xlabel(r"Control parameter $r$")
cml2d_preview_path = export_snapshots(CML2D_ROOT, cml2d_snapshot_dir,
    tracking_panel=cml2d_preview_tracking)
display(Image(filename=str(cml2d_preview_path)))'''),
    code(r'''CML2D_SPI = CML2D_ROOT / "primary-analysis"
if not (CML2D_SPI / "summary.json").exists():
    print("Physics/observation results only so far. No CML2D SPI-SPI recovery result yet.")
else:
    cml2d_spi_summary = json.loads((CML2D_SPI / "summary.json").read_text())
    display(pd.DataFrame(cml2d_spi_summary["results"]).round(4))
    print("Evidence:", cml2d_spi_summary["status"])
    print("Row gate:", cml2d_spi_summary["passes_row_gate"], "Geometry:", cml2d_spi_summary["geometry"])
    cml2d_scores = pd.read_csv(CML2D_SPI / "scores.csv")
    cml2d_fit = cml2d_scores.query("role == 'development' and eligible")
    cml2d_sign = -1 if spearman(cml2d_fit.q,cml2d_fit.Q_reference)<0 else 1
    cml2d_held = cml2d_scores.query("role == 'evaluation' and eligible").copy()
    from src.order_parameter_analysis import clustered_bootstrap_spearman
    if cml2d_held.seed.nunique() >= 2:
        uncertainty=[]
        for target in ["Q_reference","Q_window"]:
            overall, within = clustered_bootstrap_spearman(
                cml2d_sign*cml2d_held.q, cml2d_held[target],
                cml2d_held.r, cml2d_held.seed, n_resamples=2000, seed=260911,
            )
            residual=cml2d_held[["q",target]]-cml2d_held.groupby("r")[["q",target]].transform("mean")
            uncertainty.append({"target":target,
                "overall rho":spearman(cml2d_sign*cml2d_held.q,cml2d_held[target]),
                "overall 95% interval":np.nanquantile(overall,[.025,.975]).round(4).tolist(),
                "within r rho":spearman(cml2d_sign*residual.q,residual[target]),
                "within r 95% interval":np.nanquantile(within,[.025,.975]).round(4).tolist()})
        print("Seed-cluster bootstrap; the same-window within-control check is a secondary diagnostic, not longitudinal tracking:")
        display(pd.DataFrame(uncertainty).round(4))
        print("Only four held seed clusters: uncertainty is conditional on this control grid and frozen coordinate.")
    fig, axes = plt.subplots(1,2,figsize=(8.8,3.2),constrained_layout=True)
    dual_tracking(axes[0], cml2d_held, control="r", Q="Q_reference", q="q", q_sign=cml2d_sign,
                  q_color=COLORS[32],boundary=3.86212,title="Exploratory held-seed order tracking")
    intervals=cml2d_spi_summary["results"][0]["steepest_intervals"]
    if intervals["q"] == intervals["Q_reference"]:
        axes[0].axvspan(*intervals["q"],color="#31688e",alpha=.10,lw=0)
        print("Shaded shared steepest interval:",intervals["q"],"— finite-run curve, not a new critical-point estimate.")
    points=recovery_scatter(axes[1],cml2d_held,Q="Q_reference",q="q",control="r",q_sign=cml2d_sign)
    fig.colorbar(points,ax=axes[1],label=CONTROL_LABELS["r"])
    plt.show()
    cml2d_banks=[cml2d_scores]
    for name in ["sensitivity-analysis", "short-T-analysis", "contiguous-analysis"]:
        if (CML2D_ROOT / name / "summary.json").exists():
            result=json.loads((CML2D_ROOT / name / "summary.json").read_text())
            print(name, "row gate:",result["passes_row_gate"])
            display(pd.DataFrame(result["results"]).round(4))
            scores=pd.read_csv(CML2D_ROOT / name / "scores.csv")
            display(scores.groupby(["view","M","T"]).agg(
                total_rows=("eligible","size"), eligible_rows=("eligible","sum"),
                max_selected_missingness=("selected_missingness","max"),
            ).round(4))
            paired=scores.query("role == 'evaluation' and eligible").merge(
                cml2d_held[["seed","r","q"]].rename(columns={"q":"q_primary"}),
                on=["seed","r"],validate="many_to_one")
            agreement=[]
            for (view,M,T),part in paired.groupby(["view","M","T"]):
                difference=cml2d_sign*(part.q-part.q_primary)
                agreement.append({"view":view,"M":M,"T":T,"paired rows":len(part),
                    "rho with primary q":spearman(part.q,part.q_primary),
                    "mean q shift":difference.mean(),"median abs q difference":np.median(abs(difference))})
            print("Paired agreement with primary: differences use its frozen development-score SD units.")
            display(pd.DataFrame(agreement).round(4))
            cml2d_banks.append(scores)
            scores=pd.concat(cml2d_banks,ignore_index=True).drop_duplicates(["seed","r","view","M","T"])
            plotted=scores.query("role == 'evaluation' and eligible").copy()
            if name == "contiguous-analysis": plotted=plotted.query("M == 32 and T == 1000")
            if plotted.empty:
                print("No eligible evaluation curves to display.");continue
            groups=list(plotted.groupby(["view","M"]))
            qlo,qhi=(cml2d_sign*plotted.q).agg(["min","max"])
            padding=max(.1,.05*(qhi-qlo))
            fig,axes=plt.subplots(1,len(groups),figsize=(4*len(groups),3.2),squeeze=False,constrained_layout=True)
            for ax,((view,M),group) in zip(axes.flat,groups):
                right=ax.twinx()
                reference=cml2d_held.groupby("r").Q_reference.mean()
                line_Q,=ax.plot(reference.index,reference,color=Q_COLOR,lw=1.9,marker="o",ms=2.6,label=r"physical $Q$")
                reference_band=bootstrap_curve(cml2d_held,"r","Q_reference",seed=260911)
                ax.fill_between(reference_band.r,reference_band.lower,reference_band.upper,color=Q_FILL,alpha=.16,linewidth=0)
                lines=[line_Q]
                for T,part in group.groupby("T"):
                    curve=bootstrap_curve(part.assign(display_q=cml2d_sign*part.q),"r","display_q",seed=260911)
                    expected=np.sort(scores.loc[(scores["view"]==view)&(scores["M"]==M)&(scores["T"]==T),"r"].unique())
                    curve=curve.set_index("r").reindex(expected)
                    counts=part.groupby("r").size().reindex(expected,fill_value=0)
                    curve.loc[counts<2,["lower","upper"]]=np.nan
                    curve=curve.reset_index()
                    line_q,=right.plot(curve.r,curve["mean"],marker="s",lw=1.7,ms=2.4,
                        label=rf"frozen $q$, $T={T}$",color=COLORS[M],alpha=T_ALPHA[T])
                    lines.append(line_q)
                    right.fill_between(curve.r,curve.lower,curve.upper,color=COLORS[M],alpha=.12*T_ALPHA[T],linewidth=0)
                ax.axvline(3.86212,color=".5",ls=":")
                ax.set(xlabel="r",ylabel="physical Q (black)",title=f"{view}, M={M}")
                right.set_ylabel("same frozen q",color=COLORS[M])
                right.tick_params(axis="y",colors=COLORS[M])
                ax.set_ylim(-.01,.36);right.set_ylim(qlo-padding,qhi+padding)
                ax.legend(lines,[line.get_label() for line in lines],frameon=False,fontsize=7)
                paper_axis(ax);paper_axis(right,right=True)
            title={"sensitivity-analysis":"Nested M,T sensitivity",
                   "short-T-analysis":"Short-T stress test alongside longer windows",
                   "contiguous-analysis":"Sampling-layout sensitivity"}[name]
            failure = " — contiguous gate failed; descriptive curves" if name == "contiguous-analysis" else " — joint row gate failed; descriptive curves"
            fig.suptitle(title + ("" if result["passes_row_gate"] else failure))
            plt.show()'''),
    md(r'''### What contributes to q?

PCA does not erase attribution: for retained SPI pairs $j$,
$q=\sum_j w_j(\widetilde z_j-\mu_j)/s$ exactly. Loadings describe sensitivity
to a unit feature change; $w_j\Delta\overline z_j/s$ gives the exact additive
contribution to a specified change in mean q. Correlated features make neither
quantity a unique or causal importance measure. Squared loading mass is a
description of the fitted direction, not independent predictive information.

The following **post-hoc descriptive audit** reports the ten largest absolute
loadings, overall concentration, and SPI-level loading mass (each pair's mass
split equally between its two SPIs). Partner counts are included because an
SPI appearing in more retained pairs has more opportunities to contribute.
The endpoint decomposition uses the four held seeds and is not used to select
features, revise q, or guide the confirmation model.'''),
    code(r'''from scripts.cml2d_figure_diagnostics import feature_audit
cml2d_attribution = feature_audit(CML2D_ROOT)
display(pd.Series(cml2d_attribution["summary"],name="Frozen coordinate audit"))
display(cml2d_attribution["pairs"].head(10).round(6))
display(cml2d_attribution["spis"].head(10).round(6))'''),
    md(r'''### Why a large physical system with sparse observations?

$N=L^2$ determines the simulated physics; $M$ is the number of recorded
processes, and $T$ the recording duration. A sufficiently large N realises the
collective transition; q sees only the sparse M-by-T recording, while the full
system supplies an independent physical reference. Here M32 observes 1/2,048
of the lattice sites (about 0.049%). This supports **partial-observation
order tracking**, not reconstruction of every unobserved degree of freedom.

It is useful supporting evidence for the common variable-M,T representation:
access to the entire system is unnecessary in this demonstrated setting.
Fewer recorded sites do not imply a proportional reduction in information
about Q, because collective information can be redundant across sites.
Sampling layout must be stated; the dispersed result does not imply arbitrary
local patches work. No stronger headline or additional application claim is
needed to motivate this construction.

[Detailed results and execution ledger](../../docs/research/order-parameter-benchmarks/cml2d-results-260911.md).

**Verdict.** This is a positive exploratory example of unsupervised order-coordinate
inference and tracking across a known collective transition. The primary
dispersed $M=32,T=1000$ coordinate has held-seed $\rho=0.898$ with future Q,
and control-mean $\rho=0.983$. The same frozen map remains useful across the
tested M,T views; $M\geq16,T\geq500$ gives the clearest paired agreement.
Smaller/shorter views lose precision, and 12 of 216 T=100 records are excluded
by the prespecified numerical-quality rule. There is no per-arm re-fitting.

**Sampling matters.** The contiguous arm fails the frozen validity gate
(10 of 40 records excluded), and its eligible scores do not reliably recover Q.
Its curves above are descriptive, not a passed transfer result. This does not
show that patch data contain no order information: the raw baselines retain it.
Use dispersed observation for the demonstrated benchmark; larger N does not
make local and dispersed measurements interchangeable.

The supported claim is recovery of the changing order trend, not discovery of
Q's formula, calibrated numerical Q prediction, or longitudinal tracking.
The same-window within-control association is a secondary diagnostic, distinct
from prediction of the long future mean. The simple correlation baseline is
stronger in the primary test. No new thermodynamic critical-point estimate or
universal observation-size invariance is claimed.

**Independent confirmation is a separate stage**, not included in
these pilot results: 32 fresh seeds, nine original plus eight interleaved
controls, and the identical frozen q; primary M32/T1000, with a restricted
paired M16/32,T500/1000 sensitivity on the original controls. More seeds
improve uncertainty and reproducibility; more control values improve resolution
along the sweep. Its availability and results are reported below.
[Prospective confirmation protocol](../../docs/research/order-parameter-benchmarks/cml2d-confirmation-protocol-260911.md).''')] + confirmation_cells() + full_observation_cells()
