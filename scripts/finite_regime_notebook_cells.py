"""Compact, read-only appendix for finite-system physical-regime scouts."""
import nbformat


def build_cells():
    return [nbformat.v4.new_markdown_cell(r'''---

## Finite full-observation follow-up: physical regime indicators

This appendix tests established physical or dynamical quantities, not only thermodynamic order parameters. It separates exploratory pilots from independent frozen confirmations and asks whether target-blind SPI–SPI coordinates track interpretable changes across documented control boundaries. It does not establish clinical transfer, numerical calibration of Q, or superiority to purpose-built statistics. Full observation means every natural scalar state coordinate, with no synthetic duplicate channels.

| Candidate | Full observation and physical quantity | Control-space interpretation |
|---|---|---|
| Rössler pair | Six state coordinates of two oscillators; mean angular-frequency mismatch | Coupling at fixed intrinsic mismatch, across a frequency-entrainment tongue; finite-time locking estimates, not an exact threshold |
| Open TASEP | All 32 or 64 site occupations; particle density | Entry/exit-rate phase diagram, crossing alpha=beta=.2; exact finite-size stationary reference, rounded rather than discontinuous at finite N |
| Lorenz–96 | All eight state coordinates; crisis-induced residence/switching statistics, conditional on classifier validation | Negative-forcing intercept near a published attractor-merging crisis; operational region classifier must be independently checked |
| Hindmarsh–Rose | All three state coordinates; large spikes per complete burst | A local spike-deletion boundary with coexisting five-/six-spike bursting; preparation branches remain explicit |

FHN relaxation synchronization remains a reserve: inconsistencies in the published coupling convention/scale currently prevent a faithful reproduction. This is a source-specification issue, not a demonstrated physical or SPI failure. [Protocol, sources, exclusions and execution record](../../docs/research/order-parameter-benchmarks/finite-full-observation-260915.md).

The panels below separate physical contrast, observation/representation validity and q recovery. Failed-gate coordinates, where available, are explicitly descriptive. No simulation or SPI extraction runs from this notebook.'''),
        nbformat.v4.new_code_cell(r'''finite_root = ROOT / "data/order_parameter/finite_regime_260915"
finite_specs = [
    ("rossler", None, "Rössler: full six-state observation", "Coupling C", None),
    ("tasep", 32, "TASEP: M=N=32", "Entry rate alpha", .2),
    ("tasep", 64, "TASEP: M=N=64", "Entry rate alpha", .2),
    ("lorenz96", None, "Lorenz–96: M=N=8", "Forcing F", -6.4717),
    ("hindmarsh-rose", None, "Hindmarsh–Rose: full three-state observation", "Intrinsic parameter b", None),
]
if (finite_root/"rossler/confirmation/primary/analysis/summary.json").exists():
    finite_specs.append(("rossler", "confirmation", "Rössler: frozen 32-seed confirmation", "Coupling C", None))
available = [s for s in finite_specs if (finite_root/s[0]/"physics-analysis/physics.csv").exists()]
if not available:
    print("Finite-system scouts in progress; no new claim-bearing q result available yet.")
else:
    finite_results = []
    ncols = min(2,len(available)); nrows = (len(available)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows,ncols,figsize=(5.5*ncols,3.3*nrows),
                             squeeze=False,constrained_layout=True)
    for ax in list(axes.flat)[len(available):]: ax.set_visible(False)
    for ax, (system, size, title, control_label, boundary) in zip(axes.flat,available):
        system_root = finite_root/system
        frame = pd.read_csv(system_root/"physics-analysis/physics.csv")
        gate = json.loads((system_root/"physics-analysis/physics-gate.json").read_text())
        if size == "confirmation":
            arm = system_root/"confirmation/primary"
            frame = pd.read_csv(arm/"physics.csv")
            gate = json.loads((arm/"physics-gate.json").read_text())
            verdict = gate["passes"]
            analysis_root = arm/"analysis"
        elif system in ("rossler", "lorenz96", "hindmarsh-rose"):
            if system == "hindmarsh-rose":
                for preparation, group in frame.query("arm == 'preparation'").groupby("preparation"):
                    branch_curve=group.groupby("control").Q.mean()
                    ax.plot(branch_curve.index,branch_curve.values,"--",lw=.8,alpha=.55,label=f"{preparation} preparation")
            frame = frame.query("arm == 'primary'")
            verdict = gate["passes"]
            analysis_root = system_root/"primary/analysis"
        else:
            frame = frame.query("N == @size")
            verdict = gate["arms"][str(size)]["passes"]
            analysis_root = system_root/f"N{size}/analysis"
        curve = bootstrap_curve(frame,"control","Q",seed=2711)
        ax.plot(curve.control,curve["mean"],"ko-",ms=3,label="physical Q")
        if system == "hindmarsh-rose":
            ax.scatter(frame.control,frame.Q,s=7,color=".3",alpha=.3)
        ax.fill_between(curve.control,curve.lower,curve.upper,color=".5",alpha=.15)
        if "Q_exact" in frame:
            exact = frame.groupby("control").Q_exact.first()
            ax.plot(exact.index,exact.values,color=".4",ls="--",lw=1,label="exact finite-N mean")
        label = "physics passed; q pending" if verdict else "physical/input gate failed; no q"
        summary_path = analysis_root/"summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            result = summary.get("results", [{}])[0]
            rho = result.get("rho",float("nan"))
            label = (f"{'frozen' if size == 'confirmation' else 'held-seed'} q: rho={rho:.2f}" if summary["passes"]
                     else "FAILED validity gate; q descriptive only")
            if summary["passes"] and system in ("lorenz96", "hindmarsh-rose"):
                label += "; weak recovery"
            if (analysis_root/"scores.csv").exists():
                scores = pd.read_csv(analysis_root/"scores.csv").query("role == 'evaluation' and eligible")
                if len(scores):
                    qcurve = bootstrap_curve(scores.assign(q_display=summary["display_sign"]*scores.q),
                                             "control","q_display",seed=3613)
                    right=ax.twinx(); color="#4477aa" if size is None else COLORS.get(size,"#aa4499")
                    right.plot(qcurve.control,qcurve["mean"],"s-",color=color,ms=3,label="q")
                    right.fill_between(qcurve.control,qcurve.lower,qcurve.upper,color=color,alpha=.15)
                    right.set_ylabel("q (development scale)",color=color)
            finite_results.append({"System":title,"Geometry / coverage":"pass" if summary["passes"] else "FAIL",
                "Held rows":result.get("rows"),"rho(q,Q)":round(rho,3),
                "95% seed-bootstrap CI":"["+", ".join(f"{v:.3f}" for v in result.get("rho_ci",[]))+"]",
                "Excluded rows":summary.get("excluded_rows")})
        if boundary is not None: ax.axvline(boundary,color=".5",ls=":",lw=1)
        ax.set(title=f"{title}\n{label}",xlabel=control_label,ylabel="physical Q")
        ax.legend(frameon=False,fontsize=7);paper_axis(ax)
    fig.savefig(FIGURE_DIR / "finite-regime-pilots.png", dpi=200, bbox_inches="tight")
    plt.show()
    if finite_results: display(pd.DataFrame(finite_results))
    print("Pilot physical curves: eight exploratory seeds; q: four held seeds. Rössler confirmation: 32 fresh seeds for both curves. Shading: bootstrap 95% intervals for seed means, not transition-location intervals. HR dots retain per-run five/six-spike branches; its mean is an ensemble summary, not a unique Q at that control.")
'''), nbformat.v4.new_code_cell(r'''tasep_confirmation = finite_root/"tasep/confirmation-N32/primary/analysis"
if (tasep_confirmation/"summary.json").exists():
    summary=json.loads((tasep_confirmation/"summary.json").read_text())
    scores=pd.read_csv(tasep_confirmation/"scores.csv").query("role == 'evaluation' and eligible")
    result=summary["results"][0]
    fig,ax=plt.subplots(figsize=(6.2,3.4),constrained_layout=True)
    right=ax.twinx(); color=COLORS[32]
    for field,target,tint,marker in [("Q_reference",ax,"black","o"),("q_display",right,color,"s")]:
        curve=bootstrap_curve(scores.assign(q_display=summary["display_sign"]*scores.q),"control",field,seed=4211)
        target.plot(curve.control,curve["mean"],marker+"-",color=tint,ms=3)
        target.fill_between(curve.control,curve.lower,curve.upper,color=tint,alpha=.15)
    ax.axvline(.2,color=".5",ls=":",lw=1)
    status="frozen confirmation" if summary["passes"] else "FAILED gate; descriptive only"
    ax.set(xlabel="Entry rate alpha",ylabel="Physical density Q",title=f"TASEP M=N=32: {status}\n32 fresh seeds; rho={result['rho']:.3f}")
    right.set_ylabel("q (unchanged development scale)",color=color); paper_axis(ax)
    fig.savefig(FIGURE_DIR/"tasep32-frozen-confirmation.png",dpi=200,bbox_inches="tight")
    plt.show()
    print(f"{result['rows']}/672 eligible; 95% seed-bootstrap CI {result['rho_ci']}; finite-N crossover, not a discontinuity. The pilot above remains separate.")
'''), nbformat.v4.new_markdown_cell(r'''**Results.** Rössler's independent, frozen 32-seed confirmation retains all 672 observations and gives rho=.848 [95% CI .835,.862]; q and Q have the same steepest sampled interval, C=.02625–.0275. This supports unsupervised across-control inference/tracking of the established frequency-mismatch quantity from all six physical state coordinates. It does not establish within-regime fluctuation recovery, unseen-control interpolation or an exact asymptotic locking threshold. The separate pilot (.861) remains visible: its q slope maximum was earlier, underscoring finite-seed localization uncertainty. All frozen arrays, source/observation identities, eligibility and statistical reconstruction were independently audited.

TASEP N=32 also confirms across-control recovery of finite-system density: all 672 fresh observations pass, frozen rho=.746 [.705,.787], control-mean rho=.968. Its q/Q steepest intervals remain adjacent rather than identical (.200–.205 versus .195–.200), and within-control CI includes zero. The full N=64 size diagnostic also passes (168/168 valid, 84 held): rho=.813 [.779,.864], control-mean rho=.943; q/Q steepest intervals .190–.195 / .195–.200. This is a finite-size crossover across the established low-/high-density coexistence line, not a precisely localized finite-N discontinuity. N=64 uses its own development-only coordinate; this comparison is not frozen cross-size transfer. The separate N=32 exploratory pilot (.762 [.715,.801]) is retained above. No further N=64 confirmation is needed for this bounded size diagnostic.

**Retrospective frozen size-transfer check.** Applying the original N=32 model directly to the existing N=64 features, without refitting, changing scale/sign or running new SPIs, also passes: 168/168 valid, zero selected missingness; on 84 held rows rho=.782 [95% CI .726,.867], control-mean rho=.922. Every original model array remains identical. This is additional evidence for one common coordinate across these full-observation sizes, not a new independent confirmation: native N=64 results were already examined, and physical N changes together with observed M. The unchanged row/coverage gates were specified before this secondary calculation. Evidence: `data/order_parameter/finite_regime_260915/tasep/N64/frozen-N32-analysis`.

Lorenz–96 (.331) and Hindmarsh–Rose (.161) have stable, valid coordinates but weak Q recovery; these are informative negative/weak results, not implementation failures. We do not select another component using Q.

Rössler's raw mean-absolute-correlation baseline is essentially tied in confirmation (absolute rho=.849); the same-window frequency mismatch has rho=.804. TASEP's direct sample density is stronger in confirmation (.828), and raw correlation is much stronger for Lorenz–96 (.797). HR's same-window burst-count baseline matches the future target on the 68/84 held windows with a defined count; 16 windows lack a complete count. Thus weak q recovery does not demonstrate absence of information in the observed time series. The purpose remains a common representation's physical-regime sensitivity, not beating specialized observables.''')]
