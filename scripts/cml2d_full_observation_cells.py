"""Small-system diagnostic notebook sections, including clearly labelled failures."""
import nbformat


def build_cells():
    cells = [nbformat.v4.new_markdown_cell(r'''## Full observation: small physical lattices

These user-requested diagnostics change the physical lattice as well as observing all sites: they are not smaller views of L=256. Both use the same 17 controls, g=0.2, T=1000, 200,000-step burn and disjoint million-step Q reference. Each L has eight development seeds and 32 held-out evaluation seeds. The main question is whether a separately learned, target-blind coordinate tracks that small system's own Q; unchanged transfer of the L=256 coordinate is a distinct secondary test. The published infinite-size boundary is a reference, not an assumed finite-size transition. [Prespecified diagnostic and execution record](../../docs/research/order-parameter-benchmarks/cml2d-full-observation-260914.md).

Figures retain the large-system layout: exact 100-step global mean, first 100 input steps, later saved lattice field, and held-out across-r order tracking. Every site is observed, so no sensor markers are needed. Heatmap rows use fixed row-major indexing; wrapping between lattice rows is not spatial adjacency. Numerically constant sites (temporal SD <=1e-8) are masked grey for display, rather than magnifying roundoff; raw recordings are unchanged. Q and q have separate units. Where eligible scores exist despite a failed geometry gate, q is plotted explicitly as descriptive only; otherwise the panel shows physical Q alone. The L=8 illustration is read directly from its physical master because its raw-input gate stopped export.''')]
    for L in (6, 8):
        cells.append(nbformat.v4.new_markdown_cell(f'### L={L}: full observation, M=N={L*L}'))
        cells.append(nbformat.v4.new_code_cell(f'''small_L = {L}
small_root = ROOT / "data/order_parameter/cml2d_full_observation_260914" / f"L{{small_L}}"
small_report_path = small_root / "analysis/diagnostic-report.json"
if not (small_root / "physics.csv").exists():
    print(f"L={{small_L}}: local physical results not yet available.")
else:
    small_physics = pd.read_csv(small_root / "physics.csv").query("seed >= 260914101")
    display(small_physics.groupby("r").agg(mean_Q=("Q","mean"), seed_SD=("Q","std"), max_half_difference=("half_difference","max")).round(5))
    raw_failures = small_physics.minimum_channel_sd.le(1e-8)
    print(f"Raw-channel failures: {{raw_failures.sum()}}/{{len(small_physics)}} evaluation recordings.")
    small_report = json.loads(small_report_path.read_text()) if small_report_path.exists() else None
    has_q = small_report is not None and small_report["gate"]["passes"]
    if small_report is not None:
        display(small_report)
    else:
        print("L=8: raw-input gate stopped p90." if raw_failures.any() else "L=6: independent p90 run pending.")
    show_descriptive_q = small_report is not None and (small_root / "analysis/scores.csv").exists()
    if show_descriptive_q:
        small_summary = json.loads((small_root / "analysis/summary.json").read_text())
        small_held = pd.read_csv(small_root / "analysis/scores.csv").query("role == 'evaluation' and eligible")
        show_descriptive_q = len(small_held) > 0
    def small_tracking_panel(ax):
        if show_descriptive_q:
            verdict = "held-out small-system inference" if has_q else "FAILED stability gate; q descriptive only"
            dual_tracking(ax, small_held, control="r", Q="Q_reference", q="q",
                q_sign=small_summary["display_sign"], q_color=COLORS[32], boundary=3.86212,
                title=rf"$L={{small_L}}, M=N={{small_L**2}}$: {{verdict}}")
        else:
            curve = bootstrap_curve(small_physics, "r", "Q")
            ax.plot(curve.r, curve["mean"], "o-", color=Q_COLOR, ms=3)
            ax.fill_between(curve.r, curve.lower, curve.upper, color=Q_FILL, alpha=.16)
            ax.axvline(3.86212, color=".5", ls=":")
            state = "raw-input gate stopped p90" if raw_failures.any() else "p90 pending" if small_report is None else "SPI–SPI gate failed"
            ax.set(ylabel="physical Q", title=f"L={{small_L}}: {{state}}; physical order only (32 evaluation seeds)")
            paper_axis(ax)
        ax.set_xlabel(r"Control parameter $r$")
    small_figure_dir = ROOT / "notebooks/inference/figures/cml2d-period-doubling" / f"L{{small_L}}-full"
    small_figure = export_snapshots(small_root, small_figure_dir,
        tracking_panel=small_tracking_panel, seed=260914101, M=small_L**2, full=True)
    display(Image(filename=str(small_figure)))
    transfer_path = small_root / "transfer-analysis/diagnostic-report.json"
    if transfer_path.exists():
        print("Separate secondary question: unchanged transfer of the L=256 model")
        display(json.loads(transfer_path.read_text()))
'''))
    cells.append(nbformat.v4.new_markdown_cell(r'''### Structural diagnosis and possible workarounds

The small-system SPI diagnostic is complete. L=6 has all 680 valid rows, but its separately fitted coordinate fails leave-seed stability: the minimum loading cosine is 0.224 (required 0.8), despite PC1 explaining 97.65% of development variance. Only 40 meta-features survive; two non-period-four development examples from one seed drive the direction. Its plotted q is descriptive, not accepted recovery. Unchanged transfer of the large-L model fails for every recording (45.6–99.95% selected-feature missingness). L=8 stopped earlier on constant channels. These are scientific failures after successful execution; no successful small-full-system claim is made.

The separate physics-only follow-up audits all original recordings, compares odd/even lattice sizes L=5 through 64, and tests preparations, longer burns and perturbations. Short-cycle stability and finite-time Lyapunov estimates distinguish stable periodic behaviour from microscopic chaos or numerical locking. A first successful tested size is not a universal minimum-size theorem. No observation noise or channel removal is used to rescue the original full-observation experiment.

The completed scout supports a structural, attractor-dependent limitation rather than a simple failure to burn in: L=6 remains periodic after the longer burn and tested preparations. Odd L=5 and L=7 do not provide a clean substitute for the intended large-system transition; L=7 has sharp changes between different periodic/chaotic states. L=24,32,64 have positive late-time Lyapunov estimates in all 120 tested random-start cases and consistently decreasing mean Q(r), with finite-size rounding. L=24 is the first *tested* size with that consistency, not a proven minimum; its N=576 also exceeds the original feasible full-observation sizes. L=12 and L=16 retain some late-time periodic outcomes and substantial reference drift. No successful M=N<=64 workaround is established by this scout. Lyapunov estimates are measured after the Q reference, whereas recurrence is measured in the early observation window; disagreement can indicate a transient or attractor change rather than a contradiction.'''))
    cells.append(nbformat.v4.new_code_cell(r'''structure_root = ROOT / "data/order_parameter/cml2d_structure_260914"
audit_path = structure_root / "existing-bank-audit.csv"
audit_members = sorted((structure_root / "audit").glob("audit-*.json"))
if audit_path.exists() or len(audit_members) == 1360:
    structure_audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame([json.loads(p.read_text()) for p in audit_members])
    assert len(structure_audit) == 1360 and structure_audit.case_index.nunique() == 1360
    print("Completed audit of all original recordings; period 0 means no recurrence detected within 32 steps, not proof of chaos.")
    display(pd.crosstab(structure_audit.L, structure_audit.micro_period))
    display(structure_audit.assign(stable_cycle=lambda d:d.floquet_radius.lt(1),
        has_constant=lambda d:d.constant_channels.gt(0)).groupby("L").agg(
        records=("seed","size"), stable_cycles=("stable_cycle","sum"),
        recordings_with_constant_sites=("has_constant","sum")))
if not (structure_root / "complete.json").exists():
    print("Size/preparation/perturbation scout pending. No minimum size or workaround established yet.")
else:
    structure_scout = pd.read_csv(structure_root / "structural-scout.csv")
    base = structure_scout.query("preparation == 'random'")
    fig, axes = plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    size_colors = plt.get_cmap("tab20")(np.linspace(0,1,base.L.nunique()))
    for color, (L, group) in zip(size_colors, base.groupby("L")):
        curve=group.groupby("r")[["Q","lyapunov_mean"]].mean()
        axes[0].plot(curve.index,curve.Q,"o-",ms=3,color=color,label=f"L={L}")
        axes[1].plot(curve.index,curve.lyapunov_mean,"o-",ms=3,color=color)
    for ax in axes:
        ax.axvline(3.86212,color=".5",ls=":",lw=.8)
        ax.set_xlabel("r");paper_axis(ax)
    axes[0].set(ylabel="physical Q",title="Size / geometry: eight random starts")
    axes[0].legend(frameon=False,ncol=2,fontsize=6)
    axes[1].axhline(0,color=".5",lw=.8)
    axes[1].set(ylabel="finite-time largest Lyapunov estimate",title="Microscopic divergence / contraction")
    for suffix in ("png","svg"):
        fig.savefig(ROOT / f"notebooks/inference/figures/cml2d-period-doubling/structural-size-diagnostic.{suffix}",dpi=300,bbox_inches="tight")
    plt.show()
    display(structure_scout.query("L in [6,8]").groupby(["L","preparation","r"]).agg(
        mean_Q=("Q","mean"), seed_SD=("Q","std"),
        mean_Lyapunov=("lyapunov_mean","mean"),
        minimum_Q=("Q","min"),maximum_Q=("Q","max")).round(6))
'''))
    return cells
