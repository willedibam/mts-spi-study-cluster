"""Copy the executed lean notebook; add input-only baseline comparisons."""
import hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/inference/order-parameter-benchmarks-lean.ipynb'
TARGET = SOURCE.with_name('order-parameter-benchmarks-baselines.ipynb')


def build():
    nb = nbformat.read(SOURCE, 4)
    # Fail explicitly if the source layout changes; never replace unrelated cells.
    assert 'for ax, M in zip' in nb.cells[12].source and 'T_ALPHA' in nb.cells[12].source
    assert 'primary_old' in nb.cells[25].source and 'Independent paired M,T confirmation' in nb.cells[25].source
    assert 'CML2D_SPI =' in nb.cells[26].source and 'Sampling-layout sensitivity' in nb.cells[26].source
    # Modify this copy only. Existing scientific text and qualification stays intact.
    nb.cells[0].source = r'''# Order/regime tracking: comparison with simple input statistics

This is a separate copy of the lean notebook. The original physical targets, SPI--SPI coordinates, splits, exclusions and failed-gate qualifications are unchanged. These are **retrospective baseline comparisons**, not a new prespecified superiority test.

For the zero-lag Pearson MPI $C_{ij}=\operatorname{corr}_t(X_i,X_j)$, compare $\bar C=\frac{2}{M(M-1)}\sum_{i<j}C_{ij}$ (orange dashed) and $\overline{|C|}=\frac{2}{M(M-1)}\sum_{i<j}|C_{ij}|$ (purple dotted). The diagonal is excluded. Both are computed from exactly the observed $M\times T$ input, with no global state or future target. Opposite signs cancel in the first statistic, not the second; the two are not interchangeable.

**Reading the curves:** physical $Q$ retains its original left-axis units. The right axis shows $q$ and both correlations after separate, target-free affine standardization for display, using the reference cohort's mean and SD. The same constants are used across an M/T/layout family; no per-control or per-T alignment, no fit to $Q$, and no baseline sign flip is used. Original q display signs are retained. This display scaling is not a learned prediction or cross-size calibration. Colours still identify M for q; opacity still identifies T. Bands show 95% intervals of seed means. The q-vs-Q scatter plots retain their original units. Heatmaps and physics-only diagnostics remain unchanged.

The summary compares **individual held recordings**, not just control means, on an identical finite-row intersection. Also reported: control-mean association, within-control residual association, and paired seed-cluster bootstrap intervals for $|\rho(\mathrm{baseline},Q)|-|\rho(q,Q)|$. Intervals are descriptive, unadjusted for multiple comparisons. A high association is not recovery of Q's numerical units, conditional incremental information, or tracking fluctuations within a recording. Failure-gate panels remain failures regardless of association.

Additional input-only statistics: average per-channel normalized Fourier spectral entropy (DC removed); Hilbert-phase coherence of demeaned channels; mean channel standard deviation; and the largest correlation-matrix eigenvalue divided by M. Hilbert phase is a domain-sensitive oscillator baseline, not automatically a physical phase for arbitrary signals. The local entropy baseline is not the global Kaneko target formula. No baseline uses the control value as an input. None is tuned against Q.

The retained original introduction and system descriptions follow the summary.''' + '\n\n' + nb.cells[0].source
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            cell.source = cell.source.replace('notebooks/inference/figures/lean', 'notebooks/inference/figures/baselines')
            cell.source = cell.source.replace('pd.read_csv(', 'load_scores(')
    nb.cells[1].source = '''import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT/'src').exists(): ROOT=ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.plot_order_parameter_baselines import load_scores
''' + nb.cells[1].source
    nb.cells[2].source += '''
from scripts.plot_order_parameter_baselines import install, summary_figure
from scripts.order_parameter_simple_baselines import OUT as BASELINE_DIR
kuramoto=load_scores(BASELINE_DIR/'kuramoto_partial.csv')
install(globals())
'''
    nb.cells[12].source = '''fig=baseline_mt(sl,control='gamma',Q='Q_R_mean',q_sign=1,
    title='Sample-length sensitivity: same display scale across M and T')
plt.show()'''
    # Retain preceding bank loading, row pairing and audit calculations.
    s = nb.cells[25].source
    start, end = s.index('    fig, axes = plt.subplots'), s.index('    audit = ')
    nb.cells[25].source = s[:start] + '''    fig=baseline_mt(views,control='r',Q='Q_reference',q_sign=cml2d_confirm_sign,
        boundary=3.86212,title='Independent paired M,T confirmation',reference=primary_old.query('eligible'))
    plt.show()
''' + s[end:]
    s = nb.cells[26].source
    start = s.index('            groups = list(')
    end = s.index('            plt.show()', start) + len('            plt.show()')
    nb.cells[26].source = s[:start] + '''            title={'sensitivity-analysis':'Nested M,T sensitivity','short-T-analysis':'Short-T stress test','contiguous-analysis':'Sampling-layout sensitivity'}[name]
            title += '' if result['passes_row_gate'] else ' -- failed gate; descriptive only'
            fig=baseline_mt(plotted,control='r',Q='Q_reference',q_sign=cml2d_sign,
                boundary=3.86212,title=title,reference=cml2d_held)
            plt.show()''' + s[end:]
    # A high-value comparison table/heatmap, rather than six more lines on every panel.
    summary = nbformat.v4.new_code_cell('''fig=summary_figure()
fig.savefig(FIGURE_DIR/'baseline-summary.png',dpi=220,bbox_inches='tight')
plt.show()
baseline_metrics=load_scores(BASELINE_DIR/'metrics.csv')
display(baseline_metrics.query("feature in ['q','mean_correlation','mean_abs_correlation']")
    [['system','feature','n','clusters','abs_rho','ci_low','ci_high','delta_vs_q','delta_low','delta_high','within_control_rho']].round(3))
''')
    nb.cells.insert(3, summary)
    nb.cells.append(nbformat.v4.new_markdown_cell('''## Interpretation

**On these tasks, the current q results do not demonstrate an advantage over simple statistics.** Mean absolute Pearson correlation exceeds q for both Kuramoto arms, both Stuart--Landau sweeps, both Kaneko targets and large-lattice 2D CML; it is statistically unresolved against q for Miller--Huse and Rössler in the paired bootstrap comparison. Miller--Huse's signed mean correlation is poor, but channel variability and spectral entropy perform well. The alternatives are not uniformly interchangeable: signed cancellation matters, especially for Miller--Huse and Kaneko.

The summary uses each headline observation size (M32/T1000 where available; partial Kuramoto M20, Rössler M6), not a pool across M/T. Thus, for example, headline Miller--Huse q rho is .931, whereas the original pooled-M result is .888. All comparisons in a row use identical recordings. Failed contiguous/L6 results are descriptive only; contiguous correlation numbers use just the same 14 eligible rows as q and cannot establish whole-grid success. Within-control residual correlations retain raw coordinate signs, which are arbitrary for q, and are not longitudinal tracking.

These comparisons test whether simple summaries already recover the same across-control association. They do not test whether q adds conditional information after a baseline, nor whether the full z representation has useful information beyond its first principal component. Beating q does not prove that q or z contains no additional information. Conversely, a smooth q curve or a marginal win over one weak baseline is not evidence that SPI--SPI is necessary.

Use the strongest simple competitor per task as an exploratory reference, not a post-selected universal winner. Preserve the distinction between independently confirmed systems, retrospective analyses and failed geometry/input gates. No additional simulation or p90 calculation was run for this audit. The L8 physics-only failure has no usable q and therefore no q-versus-baseline comparison. Full numeric results and paired intervals are saved in `data/order_parameter/simple_baselines_260917/metrics.csv`; raw-score CSVs include all M/T arms and retain excluded rows before plotting filters.
'''))
    return nb


if __name__ == '__main__':
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    nb = build()
    NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(nb, TARGET)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    print(TARGET)
