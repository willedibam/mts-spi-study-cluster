"""Build the separately owned, concise sequel; never modify the original notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'notebooks/inference/order-parameter-benchmarks-lean-2.ipynb'


def build():
    # Read completed results without modifying the other agent's source/artifacts.
    import pandas as pd
    from scripts.lean2_candidate_readout import figure
    source=ROOT/'results/order-parameter-inference/cross-frequency-locking-261006/scores.csv'
    if source.exists():
        scores=pd.read_csv(source).rename(columns={'Q_lock':'Q'});scores['system']='crossfreq'
        fit=scores.role.eq('development')
        for col in ['z_PC1','mean_PC1']:
            scores[col]=(scores[col]-scores.loc[fit,col].mean())/scores.loc[fit,col].std()
        out=ROOT/'results/order-parameter-inference/lean2-regime-candidates-261006/crossfreq'
        out.mkdir(parents=True,exist_ok=True);figure(scores,out)
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cells=[md(r'''# Physical regime tracking: new candidates and baselines

For each recording, $z_{ab}=\operatorname{corr}_{i\ne j}(A^{(a)}_{ij},A^{(b)}_{ij})$ compares the channel-pair profiles of the 289 p90 SPIs. Separate target-blind PCA fits use development seeds only: centered SPI–SPI PC1 (primary), standardized SPI–SPI PC1 (sensitivity), and standardized **mean-SPI PC1**. **Mean absolute Pearson** is the first baseline. Q is measured independently on a future physical trajectory; neither Q nor control fits PCA. PC signs are arbitrary and oriented only for display. This tests across-control recovery, not within-record temporal tracking or absence of information from the complete mean-SPI vector.

Figure titles use held-seed Spearman rank agreement $|\rho_s|$; linear Pearson agreement is separately recorded in the metrics. A monotone but differently shaped curve can rank well without matching Q's shape.

New physics scouts and subsequent p90 results are separate from [the original benchmarks](order-parameter-benchmarks-lean.ipynb) and the copula toy. Cross-frequency results below reuse the completed experiment owned by the other agent; its ongoing confirmation is not included. [Protocol and provenance](../../docs/research/order-parameter-benchmarks/lean2-regime-candidates-261006.md).'''),
        code('''from pathlib import Path
from IPython.display import display, Image, Markdown
ROOT = Path.cwd().resolve()
while not (ROOT / 'src').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
RESULTS = ROOT / 'results/order-parameter-inference'
NEW = RESULTS / 'lean2-regime-candidates-261006'
def show(path):
    if path.exists():
        display(Image(filename=str(path)))
    else:
        display(Markdown('Not available yet: no result is inferred from a pending run.'))'''),
        md(r'''## 2:1 resonant cross-frequency locking

$\dot\phi_i=\omega_i+\epsilon\operatorname{Im}(X_1e^{-i\phi_i})+\gamma\operatorname{Im}(Ye^{-2i\phi_i})$, $\dot\psi_j=\nu_j+\epsilon\operatorname{Im}(Ye^{-i\psi_j})+\gamma\operatorname{Im}(X_2e^{-i\psi_j})$, with $X_k=\langle e^{ik\phi}\rangle$ and $Y=\langle e^{i\psi}\rangle$; independent phase noise is added. Two internally coherent communities oscillate near frequencies 1 and 2.3. Cross-coupling $\gamma$ can lock their **frequency ratio**, while ordinary zero-lag correlation stays small. A third non-resonant community supplies a fixed reference of unrelated channel pairs.

$Q=|\langle e^{i(\arg Y-\arg X_2)}\rangle_t|$ measures 2:1 locking on a disjoint future window. $M=N=24$, $T=1000$, $\epsilon=0.5$, $\gamma=0:0.01:0.20$, 16 seeds (8 development / 8 evaluation). The coherent-phase reduction gives $\gamma_c\approx0.102$, not an exact finite noisy threshold. [Komarov & Pikovsky, PRE 92, 012906 (2015), Eq. 10](https://arxiv.org/abs/1502.06193) supplies the two-community resonant model; the third community, narrow Gaussian frequencies and noise are explicit experimental modifications. **Clean recordings beat the correlation baseline near locking, but do not establish a clear advantage over mean-SPI PC1.**'''),
        code("show(NEW / 'crossfreq/baseline-comparison.png')"),
        md(r'''### Existing recording-noise sensitivity (not another physical transition)

Independent sensor noise is either fixed or varied between recordings, independently of $\gamma$. With variable noise, centered SPI–SPI PC1 tracks Q while mean-SPI PC1 follows noise. This is an **unsupervised-accessibility** result, not loss of Q-information from all SPI means: supervised mean readouts still recover Q. Standardizing SPI–SPI coordinates also weakens this advantage. Confirmation belongs to the other agent.'''),
        code("show(RESULTS / 'cross-frequency-locking-snr-261006/snr-comparison.png')"),
        md(r'''## Physics gates for the two new systems

Each coarse scout uses four independent initial/connectivity realizations per control and size, $M=16,T=1000$. Q comes from the full system, not just the observed sensors. Bands below span the four realizations; they are **not confidence intervals**. These checks choose a physical interval before p90, not a favorable representation result.'''),
        code("show(NEW / 'physics/physics-scout.png')"),
        md(r'''## Complex Ginzburg–Landau: phase turbulence → defect turbulence

$\partial_tA=(1+ic_1)\partial_x^2A+A-(1-ic_3)|A|^2A$ on a periodic one-dimensional domain. $c_1=3.5$ controls linear dispersion; $c_3$ controls nonlinear phase rotation. In phase turbulence the complex amplitude stays away from zero; defect turbulence contains amplitude zeros and phase slips. $Q=D/(L\,\Delta t)$ is the **space-time defect density**, a literature-established order parameter—not mean synchrony. Count local winding defects of either sign, not only net winding changes.

Initial sweep $c_3=0.60,0.70,0.75,0.80,0.90,1.00$; $L=128,512$, $\Delta x=0.5$ ($N=256,1024$ complex sites), $\Delta t=0.025$, burn 2,000, future truth window 4,000. Observe $\operatorname{Re}A$ at 16 evenly spaced sites every 0.5 time units. The phase/defect boundary near $c_3\sim0.7$–$0.77$ is a rare-event numerical boundary, **not** the analytic Benjamin–Feir line $c_3=1/c_1$. [Torcini, Frauenkron & Grassberger, sections I–III](https://arxiv.org/html/chao-dyn/9608003). Zero measured defects in a finite window is only an upper-resolution limit, not proof of zero infinite-time density.'''),
        md(r'''**Finer test:** $c_3=0.74:0.01:0.90$ (17 values), 16 fresh seeds; $L=512,N=1024,M=16,T=1000$, $\Delta t=0.0125$, burn 4,000 and future truth window 8,000. Time/spatial refinement preserves the density curve, with an unresolved ~7% middle-point ensemble/grid difference; this is not a precision critical-point estimate.'''),
        code("show(NEW / 'cgle/analysis/baseline-comparison.png')"),
        md(r'''**Result: not a baseline-advantage example.** All 136 held recordings pass. Rank recovery is weaker for SPI–SPI PC1 (.904) than mean absolute Pearson (.949) and mean-SPI PC1 (.970); paired seed-bootstrap differences are negative. Its linear agreement is better (.846 versus .746/.757), but the means retain substantial information: a supervised mean-vector readout reaches .983 linear agreement. This distinction is curve shape versus information, not a hidden success under another metric.'''),
        md(r'''### Secondary diagnostic: collapse of minimum amplitude

$Q_{\min}=\min_{x,t\;\mathrm{in\ the\ future\ window}}|A(x,t)|$ estimates the lower edge of the amplitude distribution. The paper's Fig. 4 discusses its sharp collapse as defects become possible. This is **not interchangeable with canonical defect density**: a finite minimum depends on domain size, window length and numerical resolution. Its observed collapse need not locate the infinite-size critical point. Proposed before viewing SPI results and included at the user's request. The following comparison reuses exactly the same fitted coordinates; only their arbitrary display signs are reversed for the falling target.'''),
        code("show(NEW / 'cgle/analysis/minimum-amplitude-comparison.png')"),
        md(r'''**Result:** the minimum collapses near $c_3=0.77$–$0.78$ in these finite windows, but none of the three descriptors reproduces that sharp collapse closely. SPI–SPI is weaker than both baselines in rank and linear agreement. A sharp target alone does not make this a strong demonstration of SPI–SPI advantage.'''),
        md(r'''## Driven random rate network: reliable → chaotic response

$dx_i=[-x_i+\sum_jJ_{ij}\tanh x_j]dt+\sqrt{2}\sigma\,dW_i$, with independent neuronal input noise, $J_{ij}\sim\mathcal N(0,g^2/N)$ and $J_{ii}=0$. Gain $g$ amplifies recurrent feedback; input-noise variance is fixed at $\sigma^2=0.125$. $Q=\lambda_{\max}$ is the **largest conditional Lyapunov exponent**: negative means perturbations decay under the same input, positive means chaotic sensitivity. It is an established physical diagnostic, not a thermodynamic order parameter, and its zero crossing need not be a jump.

Initial sweep $g=1.10,1.30,1.45,1.60,1.80,2.00$; $N=16,64,256$, $M=16$, time step 0.02, burn 1,000, sampling interval 0.5, future truth window 3,000. Connectivity is fixed across g within each seed. Tangent vectors are used only to measure Q and never become observed channels. [Schuecker, Goedeke & Helias, PRX 8, 041029 (2018)](https://arxiv.org/html/1603.01880) reports the large-N boundary around $g\approx1.48$ for this input variance; it is not assumed for N=16. Numerical/finite-size checks precede a p90 interpretation.'''),
        md(r'''**Finer test:** all tested N16 cases remained non-chaotic; N256 crossed but its boundary varied substantially between networks. N1024 gives a much more consistent crossing. The p90 cohort therefore uses $N=1024,M=16,T=1000$, $g=1.35:0.025:1.75$ (17 values), 16 fresh connectivity seeds; $\Delta t=0.01$, burn 1,000 and future truth window 4,000. Size and interval were selected from physics alone, before SPI results. The dotted vertical line is the independently reproduced **infinite-N reference** $g_c=1.475568$, not the measured finite-network boundary; the horizontal line marks $Q=0$.'''),
        code("show(NEW / 'rate/analysis/baseline-comparison.png')"),
        md(r'''**Result: PC1 fails to recover the transition.** All 136 held recordings pass. SPI–SPI PC1 has $|\rho_s|=.053$, versus .675 for mean absolute Pearson and .764 for mean-SPI PC1. Standardizing z does not fix this (.071). A supervised SPI-mean readout reaches .956, so the recordings contain usable information about Q.

A labelled post-result diagnosis finds that network-seed group means explain ~45% of held z-PC1 variation, versus ~4% for control group means. z-PC4 does track Q (.795), showing that failure of PC1 is not absence of information from z. PC4 was **not** selected in advance and is not substituted as a successful unsupervised readout. Neither new system establishes the sought advantage over both baselines; retain these as bounded negative tests, not reasons to scale unchanged.''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.write(nb,TARGET)


if __name__=='__main__':build()
