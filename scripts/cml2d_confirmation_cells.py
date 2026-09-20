"""Read-only notebook presentation of the independently frozen CML confirmation."""
import nbformat


def build_cells():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    return [md(r'''### Independent fresh-seed confirmation

This stage is distinct from the pilot above: 32 entirely new seed clusters,
nine original plus eight interleaved control values (544 physical masters),
the **unchanged pilot feature mask, imputation, centring, PCA direction and
score scale**, and no new fitting. Primary: dispersed M32/T1000. A restricted
paired M16/32,T500/1000 comparison uses the original nine controls.

The prospectively specified per-row rule is at most5% missing selected
features; each arm must exclude at most10% of rows and retain at least24 of32
records in every control/M/T cell. Confidence intervals resample the32 seed
clusters. The generic scorer retains its historical “exploratory” label;
the separate sealed confirmation report below records this stage's status.
[Protocol](../../docs/research/order-parameter-benchmarks/cml2d-confirmation-protocol-260911.md).
'''), code(r'''CML2D_CONFIRM_ROOT = ROOT / "data/order_parameter/cml2d_confirmation_260911"
cml2d_confirm_reports = {}
for arm in ["primary", "sensitivity"]:
    path=CML2D_CONFIRM_ROOT / f"{arm}-analysis/confirmation-report.json"
    if path.exists():
        cml2d_confirm_reports[arm]=json.loads(path.read_text())
        print(arm, cml2d_confirm_reports[arm]["status"])
        display(pd.Series(cml2d_confirm_reports[arm]["gate"],name="Prospective quality gate"))
        display(pd.DataFrame(cml2d_confirm_reports[arm]["endpoints"]).round(4))
    else:
        print(arm, "confirmation results are not yet available locally.")
if "primary" in cml2d_confirm_reports:
    cml2d_confirm_summary=json.loads((CML2D_CONFIRM_ROOT / "primary-analysis/summary.json").read_text())
    cml2d_confirm_sign=cml2d_confirm_summary["display_sign"]
    cml2d_confirm_all=pd.read_csv(CML2D_CONFIRM_ROOT / "primary-analysis/scores.csv")
    cml2d_confirm=cml2d_confirm_all.query("eligible").copy()
    display(pd.DataFrame(cml2d_confirm_summary["results"]).round(4))
    fig,axes=plt.subplots(1,2,figsize=(9,3.2),constrained_layout=True)
    dual_tracking(axes[0],cml2d_confirm,control="r",Q="Q_reference",q="q",
        q_sign=cml2d_confirm_sign,q_color=COLORS[32],boundary=3.86212,
        title="Independent confirmation: 32 fresh seeds")
    points=recovery_scatter(axes[1],cml2d_confirm,Q="Q_reference",q="q",control="r",q_sign=cml2d_confirm_sign)
    fig.colorbar(points,ax=axes[1],label=CONTROL_LABELS["r"])
    if not cml2d_confirm_reports["primary"]["gate"]["passes"]:
        fig.suptitle("Quality gate failed: descriptive results only")
    plt.show()
    physics=pd.read_csv(CML2D_CONFIRM_ROOT / "primary-analysis/physics-reference-audit.csv")
    print("Reference precision: finite-run Q, not a new equilibrium critical-point estimate.")
    display(physics.groupby("r").agg(mean_Q=("Q","mean"),SD_Q=("Q","std"),
        max_half_difference=("half_difference","max"),max_block_SE=("block_mean_se","max")).round(5))
'''), code(r'''if {"primary","sensitivity"}.issubset(cml2d_confirm_reports):
    # Derive the pilot grid without importing a CLI module that sets an Agg backend.
    OLD_CONTROLS = np.sort(cml2d_scores.r.unique()).tolist()
    secondary=pd.read_csv(CML2D_CONFIRM_ROOT / "sensitivity-analysis/scores.csv")
    primary_old=cml2d_confirm_all[cml2d_confirm_all.r.isin(OLD_CONTROLS)]
    views=pd.concat([primary_old,secondary],ignore_index=True).query("eligible").copy()
    paired=secondary.query("eligible").merge(
        primary_old.query("eligible")[["seed","r","q"]].rename(columns={"q":"q_primary"}),
        on=["seed","r"],validate="many_to_one")
    agreement=[]
    for (M,T),part in paired.groupby(["M","T"]):
        shift=cml2d_confirm_sign*(part.q-part.q_primary)
        agreement.append(dict(M=M,T=T,paired_rows=len(part),rho_with_primary=spearman(part.q,part.q_primary),
            mean_shift=shift.mean(),median_absolute_difference=np.median(abs(shift))))
    print("Paired coordinate agreement on the original control grid; original frozen q units, no per-arm rescaling.")
    display(pd.DataFrame(agreement).round(4))
    fig,axes=plt.subplots(1,2,figsize=(8,3.2),constrained_layout=True)
    limits=(cml2d_confirm_sign*views.q).agg(["min","max"]).to_numpy()
    pad=max(.1,.05*np.ptp(limits))
    for ax,M in zip(axes,[16,32]):
        right=ax.twinx()
        truth=bootstrap_curve(primary_old.query("eligible"),"r","Q_reference",seed=260911)
        line,=ax.plot(truth.r,truth["mean"],color=Q_COLOR,marker="o",ms=2.6,lw=1.9,label=r"physical $Q$")
        ax.fill_between(truth.r,truth.lower,truth.upper,color=Q_FILL,alpha=.16,linewidth=0)
        lines=[line]
        for T in [500,1000]:
            part=views.query("M == @M and T == @T")
            curve=bootstrap_curve(part.assign(display_q=cml2d_confirm_sign*part.q),"r","display_q",seed=260911)
            curve=curve.set_index("r").reindex(OLD_CONTROLS).reset_index()
            line,=right.plot(curve.r,curve["mean"],color=COLORS[M],alpha=T_ALPHA[T],
                marker="s",ms=2.4,lw=1.7,label=rf"frozen $q$, $T={T}$")
            right.fill_between(curve.r,curve.lower,curve.upper,color=COLORS[M],alpha=.12*T_ALPHA[T],linewidth=0)
            lines.append(line)
        ax.axvline(3.86212,color=".5",ls=":",lw=1)
        ax.set(xlabel=CONTROL_LABELS["r"],ylabel=r"physical $Q$",title=rf"$M={M}$",ylim=(-.01,.36))
        right.set(ylabel="same frozen q",ylim=(limits[0]-pad,limits[1]+pad))
        right.yaxis.label.set_color(COLORS[M]);right.tick_params(axis="y",colors=COLORS[M])
        ax.legend(lines,[line.get_label() for line in lines],frameon=False,fontsize=7)
        paper_axis(ax);paper_axis(right,right=True)
    passed=all(report["gate"]["passes"] for report in cml2d_confirm_reports.values())
    fig.suptitle("Independent paired M,T confirmation" + ("" if passed else " — failed arm is descriptive only"))
    plt.show()
    audit=CML2D_CONFIRM_ROOT / "integrity-audit.json"
    if audit.exists():
        print("Independent integrity and reconstruction audit:")
        display(pd.Series(json.loads(audit.read_text())))
'''), md(r'''**Confirmed interpretation.** With the original map frozen, the independent
M32/T1000 result is $\rho=0.883$ (95% seed-bootstrap interval $[0.868,0.898]$),
and control-mean $\rho=1.000$. On the eight new control values alone,
$\rho=0.880$. Both q and Q change most steeply over $r=3.86212$–$3.86306$.
All 1,408 views pass the quality rules with no exclusions or missing selected
features. This supports **unsupervised inference of an order-related coordinate
that tracks changing collective order across this studied boundary**.

The same frozen coordinate remains informative across the tested M,T values,
but small shifts and localisation differences remain; longer T does not
monotonically improve q–Q correlation. Within-control future-Q recovery is
weak ($\rho=0.078$, interval including zero), and simple baselines are stronger.
No numerical Q calibration, longitudinal tracking or arbitrary-layout
invariance is claimed. [Full confirmation results](../../docs/research/order-parameter-benchmarks/cml2d-confirmation-results-260911.md).''')]
