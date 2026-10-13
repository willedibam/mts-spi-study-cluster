"""Copy the user's current lean notebook verbatim, then execute only the new appendix."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'research/inference/benchmarks/notebooks/order-parameter-benchmarks-lean.ipynb'
TARGET = ROOT / 'research/inference/benchmarks/notebooks/order-parameter-benchmarks-lean-extended.ipynb'
RESULTS = ROOT / 'results/order-parameter-inference'
PREVIEW = RESULTS / 'locking-preview-261007'


def appendix():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = []
    def figure(path):
        cells.append(code(f"_appendix_show({str(path)!r})"))
    cells.append(md(r'''---

# Additional benchmarks: baselines and dependence-character interpretation

The preceding notebook is preserved unchanged. This appendix separates **physical transition sensitivity**, **advantage over a baseline**, and **which signal PC1 prioritizes**. The primary readout is centered SPI–SPI PC1; mean absolute Pearson and standardized per-SPI-mean PC1 are the baselines. Each readout is fitted on development recordings without Q; display signs are arbitrary. Individual-feature explanations and the family-removal experiment are retrospective diagnostics, not new confirmed methods. [Protocol and complete results](../../research/inference/locking/notes/cross-frequency-locking-261006.md).'''))
    cells.append(code('''from pathlib import Path
from IPython.display import Image, display
_appendix_root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
                     if (p / 'src/mtsspi/spi_spi_contract.py').is_file())
def _appendix_show(relative_path):
    path = _appendix_root / 'results/order-parameter-inference' / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path)))'''))
    cells.append(md(r'''## 7. Resonant oscillator communities — 2:1 cross-frequency locking

**System.** Define $Z_{A,k}=n^{-1}\sum_j e^{ik\phi_j}$, $Z_B=n^{-1}\sum_j e^{i\psi_j}$ and $Z_C=n^{-1}\sum_j e^{i\chi_j}$:

$$d\phi_i=[\omega_i+\varepsilon\operatorname{Im}(Z_{A,1}e^{-i\phi_i})+\gamma\operatorname{Im}(Z_Be^{-2i\phi_i})]dt+\sigma dW_i,$$
$$d\psi_i=[\nu_i+\varepsilon\operatorname{Im}(Z_Be^{-i\psi_i})+\gamma\operatorname{Im}(Z_{A,2}e^{-i\psi_i})]dt+\sigma dV_i,$$
$$d\chi_i=[\xi_i+\varepsilon\operatorname{Im}(Z_Ce^{-i\chi_i})]dt+\sigma dU_i.$$

**Parameters.** Slow A and fast B have mean natural frequencies 1 and 2.3 (detuning $\delta=.3$); C has frequency 1.37 and no cross-coupling. Frequency spread .05, internal coupling $\varepsilon=.5$ and independent phase noise $\sigma=.1$ keep each community coherent but non-identical. Zero coupling-phase shifts isolate locking rather than the paper's frustrated/chaotic regimes. C supplies pairs that remain unrelated; its particular frequency is a design choice, not a physical constant.

**Published model and regime:** [Komarov & Pikovsky, PRE 92, 012906 (2015), Eqs. 10–12](https://arxiv.org/html/1502.06193v1). Our Gaussian-quantile frequencies, noise, finite populations and third community are explicit modifications. The published diagrams are not exact phase boundaries for this modified system.

**Sweep and observations.** $\gamma=0:.01:.20$ (21 values), 16 independent realizations per control (8 development / 8 held). $n=8$ initially, $n=16$ in confirmation, hence $M=N=24,48$. Observe $\sin(\mathrm{phase})$, $T=1000$, sample interval .5, channel standardization; integration step .02 and burn 500. Every record uses full p90 (289 SPIs).

**Physical quantity.** The harmonic phase-locking index on a disjoint future window of 1,000 time units is

$$Q=\left|\left\langle e^{i[\arg Z_B(t)-\arg Z_{A,2}(t)]}\right\rangle_{\rm future}\right|.$$

For coherent A, $\arg Z_{A,2}\approx2\arg Z_{A,1}$. Q is a harmonic-locking observable, not a thermodynamic order parameter. The coherent, noise-free reduction $\dot\Phi\approx\delta-\gamma(|Z_{A,2}|+2|Z_B|)\sin\Phi$ predicts $\gamma_c\approx.102$; this is an approximate reference, not an exact finite-noise threshold.'''))
    figure('locking-composites-261008/latex/physics.png')
    cells.append(md(r'''**Physical interpretation.** Phase slips disappear on the observed time scale and the fast/slow frequency ratio pins to two. Q rises before locking, then approaches one: a ramp into a sharp corner and plateau, not a flat–jump–flat discontinuity. Ordinary A–B Pearson remains near zero because harmonic locking need not produce a linear scatter. The physics figure uses fresh illustrative realizations; its relative-phase label uses the coherent-population shorthand above.'''))
    cells.append(md(r'''### Clean recordings and variable recording quality

**Hypothesis:** locking reorganizes dependence while zero-lag correlation changes little. Independent sensor noise is added as $X_i=\operatorname{standardize}[\sin(\theta_i)/\operatorname{sd}(\sin\theta_i)+\eta e_i]$, with $e_i$ independent white noise and $\eta\sim U[.3,.9]$ drawn independently of control for each recording. This is a chosen recording-quality nuisance, not an instrument calibration; it leaves the underlying Q unchanged. “Clean” means no observation noise; the dynamical phase noise remains.

**Reading the composites.** Top: an observed slow A channel against an observed fast B channel, using all 1,000 samples (state portraits, not latent phase angles). Middle: the same recordings' first 200 samples, with A/B/C blocks; heatmap colour is clipped at ±3 channel SD. The system is fully observed, so no separate lattice is needed. Snapshot controls are fixed at $\gamma=0,.10,.20$; clean uses the first held seed, variants use the held recording nearest the nuisance midpoint, never Q or q. Bottom: three methods over the **whole held sweep**, not three single-record estimates. Black Q is left; blue readouts are right. Each right axis spans its own curve and 10–90% instance band, with padding; this aligns visible ranges without changing scores, fitting to Q or clipping uncertainty. Independent scales can magnify a small Pearson drift: read its ticks, not just curve height.

**Displayed SPI–SPI readout: centered PC1 throughout.** Standardization divides each feature by its development SD and gives low-variance coordinates equal scale; centering alone preserves their relative variance. Standardized PC1 has better clean rank agreement (.894 versus .832), but is substantially more noise-sensitive at M48 (.478 versus .855). We retain the original centered primary consistently rather than choose a winner for each panel. Robustness was the hypothesis, not a general guarantee; the alternative results remain archived but are omitted from these composites.'''))
    figure('locking-composites-261008/clean-composite.png')
    cells.append(md(r'''**Clean interpretation, M24.** Both catalogue PC1s register locking; mean absolute Pearson moves only about .302→.305. Its pooled rank association (.73) rewards that tiny drift but does not imply a comparably clear boundary. Centered SPI–SPI does not clearly beat mean-SPI PC1 or reproduce the entire pre-locking Q ramp.'''))
    figure('locking-composites-261008/random-noise-m24-composite.png')
    cells.append(md(r'''**Varying independent sensor noise, M24.** Same physical model, size and control grid as clean, with fresh realizations and varying observation quality. Held $|\rho_s(q,Q)|$ is **.87 / .03 / .02** for centered SPI–SPI / mean absolute Pearson / mean-SPI PC1. Noise obscures the single-pair portraits; the MTS and analysis use these noisy observations, not denoised or hidden trajectories. Centered z prioritizes locking while the mean-based PC1 prioritizes noise.'''))
    figure('locking-composites-261008/random-noise-m48-composite.png')
    cells.append(md(r'''**Fresh confirmation, M48.** The same noise distribution at twice the oscillator count gives **.855 / .133 / .115** on 168 held records. This confirms the bounded unsupervised-accessibility result, not information absence from the mean vector or invariance to arbitrary measurement noise. The observation noise is drawn independently per recording; the three displayed mid-noise examples do not represent its full variation.'''))
    cells.append(md(r'''### Boundary tests: other sources of between-recording variation

These are **variants of the same oscillator model**, not additional system families. Each M48 arm uses 21 controls × 16 fresh realizations, with the same development/held split and separate PCA fits. Independent sensor noise is now fixed at $\eta=.5$.

**Common mode (not “common node”):** add the same white signal $s e_0(t)$ to every observed channel, with $s\sim U[0,.8]$ per recording. This mimics shared pickup/reference contamination, adds instantaneous dependence and does not change the underlying oscillators or Q. It asks whether robustness to independent noise extends to shared contamination.'''))
    figure('locking-composites-261008/common-mode-composite.png')
    cells.append(md(r'''**Common-mode interpretation: failure.** Centered SPI–SPI PC1 has held Q association .059 but nuisance association .994. Standardized PC1 also fails (.043 with Q). The shared signal changes the mixture of dependence types, not merely every SPI's scale; therefore insensitivity was not guaranteed. Both SPI–SPI and mean-SPI PC1 follow the nuisance. Rescaling the axes does not restore a transition.'''))
    cells.append(md(r'''**Internal coupling:** draw $\varepsilon\sim U[.15,.6]$ per recording, independently of $\gamma$. This changes within-community coherence and the physical locking boundary, while keeping observation-noise SD fixed. It tests dynamical heterogeneity, not just recording quality; Q itself can change with $\varepsilon$. There is no single exact vertical boundary for the ensemble, so no fixed transition line is drawn below.'''))
    figure('locking-composites-261008/internal-coupling-composite.png')
    cells.append(md(r'''**Internal-coupling interpretation.** Centered SPI–SPI tracks Q (.886 versus .527 for mean-SPI PC1), but also associates with the varying coupling (.235); some of that association is physical because coupling changes Q. This is better than the predicted contamination, not proof of complete nuisance rejection. The original full-catalogue centered readout is retained, as in every other composite.'''))
    figure('locking-composites-261008/latex/component-scatter-random-noise.png')
    cells.append(md(r'''**Why PC1 differs.** These are the same held points recoloured by Q and sensor noise. Mean-SPI PC1 follows noise; its PC2 contains the transition ($|\rho|=.72$). SPI–SPI puts Q first (.855) and noise second. A supervised mean-vector predictor reaches .900. The advantage is the **ordering of variation**, not that only SPI–SPI contains physical information. The coordinates were fitted without Q or nuisance labels.'''))
    cells.extend(ranked_feature_cells())
    cells.extend(correlation_composition_cells())
    cells.extend(non_double_correlation_cells())
    cells.extend(m24_edge_profile_cells())
    cells.append(md(r'''### Individual SPI–SPI features: what changes?

For fixed $q=\sum_jw_j(z_j-\bar z_j)$, rank transition contributions $\Delta q_j=w_j(\bar z_{j,\gamma=.12}-\bar z_{j,\gamma=.08})$ using development recordings, then inspect held curves. This is explanatory use of the known boundary, not target-driven PCA fitting. Below: one development-ranked example per identifier-family pair, excluding all lagged-correlation endpoints. These identifier groups are not the published SPI modules. All six loading signs persist across eight leave-development-seed fits; this is not a guarantee of feature-rank stability.'''))
    figure('locking-preview-261007/nonlag-features.png')
    cells.append(md(r'''**Interpretation.** LCSS/DTW versus transfer entropy contrasts sequence-alignment similarity/distance with predictive dependence; their agreement across channel pairs changes through locking. Held $|\rho_s(z_{ab},Q)|$ is .883 for LCSS×TE and .877 for DTW×TE. Stochastic interaction×Johansen trace gives .899, but Johansen statistics on bounded oscillatory data must not be interpreted as evidence of cointegration. These are illustrative correlated features, not unique causal drivers. The strongest individual contributors overall are redundant lagged-correlation variants; their top four account for only .62% of summed absolute contributions. [Lagged-feature preview](../../results/order-parameter-inference/locking-preview-261007/individual-features.png).'''))
    figure('locking-preview-261007/nonlag-edge-profiles.png')
    cells.append(md(r'''**Edge-level interpretation.** Each point is a directed channel pair; pink denotes A–B. After locking, A–B transfer-entropy estimates rise while their LCSS similarity changes little and their covariance remains near zero. Thus different statistics cease to agree about which pairs are dependent: LCSS×TE changes from about .59 to .01, covariance×TE from .65 to .34 in these examples. Records were chosen nearest fixed noise SD .6 at three fixed controls, not for favourable outcomes. Their exact cached MPI hashes and z values were verified. These three recordings illustrate the mechanism, not a population test; TE estimates are not by themselves causal-flow evidence.'''))
    cells.append(md(r'''### Retrospective sensitivity: remove lagged-correlation families

Remove every z coordinate touching a `corr_*` or `xcorr*` SPI, then refit centered PC1 using the same development records. Separately fit standardized PC1 to the means of the 60 lagged-correlation, cross-correlation, coherence-magnitude and PLV SPIs. Catalogue-name masks were fixed before evaluating this diagnostic; no estimator or control was tuned to improve its outcome. All 168 held records per arm pass the common coverage gate. The full-p90 fit above is unchanged.'''))
    figure('locking-preview-261007/family-sensitivity.png')
    cells.append(md(r'''**Interpretation: an important limitation.** Under sensor noise, removing lagged-correlation coordinates reduces held Q agreement from **.855 to .041**; the new PC1 instead follows noise (.966). The lagged/spectral mean-PC1 also follows noise (Q agreement .065). Under internal-coupling variation, the reduced SPI–SPI PC1 still tracks Q (.801 versus full .886); common mode remains a failure. Features involving lagged correlation account for ~62% of summed absolute development transition contributions in the noise arm, including cross-family pairs; redundancy affects this weighting. Strong individual non-lagged features therefore do **not** imply that PC1 of the remaining representation will track Q. This supports a distributed, catalogue-dependent ordering of variance—not a proven nonlinear-only axis. The ablation is retrospective and is not promoted as a replacement method.'''))
    cells.append(md(r'''## 8. NeuroTycho anaesthetic induction — exploratory real-data counterexample

**System and observations.** Macaque ECoG during ketamine–medetomidine (11 dates) or propofol (4 dates): four animals, 16 bipolar channels, 2,239 eight-second windows at 250 Hz ($T=2000$), full p90. Windows tile induction and later periods; plotting uses a five-window rolling median. Readouts are fitted without the displayed animal, with display sign oriented from fitting animals only.

**Reference, not an order parameter.** The horizontal axis is time since injection, not a measured physical control parameter. Truth consists of event/state annotations; no continuous Q or consciousness-onset boundary is established. This is a state-separation/trajectory check, not validation of order-parameter recovery. The source and processing are recorded in [the experiment report](../../research/inference/locking/notes/cross-frequency-locking-261006.md#real-data-attempt-neurotycho-anaesthetic-induction-exploratory-6-october).'''))
    figure('locking-composites-261008/latex/induction-trajectories.png')
    cells.append(md(r'''**Interpretation.** The baselines perform better. Mean date-level awake-versus-anaesthetized AUROC (ketamine–medetomidine / propofol) is **.959/.993** for mean absolute Pearson, **.995/.998** for mean-SPI PC1 and **.721/.696** for centered SPI–SPI PC1. The dominant centered-z variation is more animal-related than state-related. This is consistent with prominent strength changes and between-animal heterogeneity, not proof of their precise physiological cause. Windows/dates are not independent biological replicates; there are four animals, and one animal shows little leave-animal-out change. These results do not generalize to anaesthesia broadly.

**Overall reading.** The locking benchmark demonstrates a bounded unsupervised-accessibility advantage under specified variability, with interpretable changes between SPI profiles. Neither the family sensitivity nor the real-data attempt supports universal superiority, a strength-independent representation, or automatic discovery of the scientifically relevant transition.'''))
    return cells


def ranked_feature_cells():
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    return [md(r'''### Unfiltered feature rankings — varying independent sensor noise, M24

The original M24 noise experiment is used here (168 development / 168 held recordings; centered PC1). “Unfiltered” means **no family filtering or deduplication**; the original development-complete, nonconstant eligibility gate retains 25,417 coordinates. Both rankings use development data only; plotted blue means and 10–90% bands use held data. Black is the independently measured Q on the right axis. Feature axes differ; signs are not flipped. The Q-based ranking is a supervised explanatory diagnostic, not an unsupervised choice of q or a new validation cohort. Full identifiers, weights and signed development/held associations are in [the ranking file](../../results/order-parameter-inference/locking-features-m24-261009/all-feature-rankings.csv).

**PC1 weight:** rank by $|w_j|$ in the unchanged centered $q=\sum_jw_j(z_j-\bar z_j)$. This identifies the coordinate's coefficient in PC1, not its causal importance or unique contribution among redundant features.'''),
            *weight_bar_cells(),
            code("_appendix_show('locking-features-m24-261009/top16-weight.png')"),
            md(r'''**Interpretation.** Large weights mainly join short-lag correlations to squared long-lag correlations. They represent large changes in agreement across edge profiles and can respond to the frequency adjustments accompanying locking. The leading feature has $|w|=.0504$, held $|\rho_s(z_{ab},Q)|=.895$ and development SD .477. Its large coefficient does not establish a nonlinear-only mechanism.'''),
            md(r'''**Association with physical Q:** rank by development $|\rho_s(z_{ab},Q)|$, not by held performance. No family restriction is applied.'''),
            code("_appendix_show('locking-features-m24-261009/top16-rho.png')"),
            md(r'''**Interpretation.** There is **no overlap between these two top-16 lists**. The development-leading Q feature is squared lag-4 Pearson × biased distance correlation: held $|\rho_s|=.952$, but $|w|=.0060$ and development SD .052. Nearby signed variants reach held .960–.963. They compare a delayed linear/rank-based notion of dependence with a more general distance-based one; high association alone does not isolate the physical cause. Centered PCA maximizes variance of z, not association with Q, so a smaller-amplitude feature may track Q better yet receive less weight. For PCA, $\operatorname{Cov}(z_j,q)=\lambda_1w_j$; neither this covariance nor $w_j$ is $\operatorname{corr}(z_j,Q)$. Negative association produces a falling feature as Q rises. High rank agreement also does not require identical curve shape or exact threshold localization. The existing M48 examples below remain separately labelled.''')]


def weight_bar_cells():
    return [nbf.v4.new_markdown_cell(r'''**Top 20 coefficients.** Sorted horizontal bars show $|w_j|$; colour and labels retain the sign. Black segments show the range across eight leave-development-seed-out fits with the original validity mask (not confidence intervals). Near-ties are redundant variants, not separate discoveries. Coefficient magnitude is neither explained variance nor a causal attribution. [Weights and held/seed diagnostics](../../results/order-parameter-inference/locking-features-m24-261009/top20-weights.csv).'''),
            nbf.v4.new_code_cell("_appendix_show('locking-features-m24-261009/top20-weights.png')")]


def correlation_composition_cells():
    return [nbf.v4.new_markdown_cell(r'''### Where correlation–correlation coordinates sit

One signed-weight raincloud for all 25,417 eligible coordinates; only the scatter is coloured by family. Orange means both endpoints are classical Pearson/Spearman/Kendall, lagged or cross-correlation variants (including squared versions and empirical covariance, which is Pearson-equivalent on these standardized channels). Other covariance estimators, distance/envelope correlation and coherence are not in this explicitly named group. The grey cloud/box summarize everyone; every point is shown, with seeded jitter. These are **PC1 coefficients**, not distributions of the feature values across recordings. Global PC1 sign is arbitrary.'''),
            nbf.v4.new_code_cell("_appendix_show('locking-features-m24-261009/weight-raincloud.png')"),
            nbf.v4.new_markdown_cell(r'''**Result.** Correlation–correlation features are 1,431/25,417 (5.63%); they occupy all top20, 45/top100 and 259/top1000, but their median rank is 5,812 and they extend to rank25,390. They account for 23.06% of squared loading norm, **not explained-variance attribution**. Broadening the group to all covariance estimators gives 2,264 coordinates and 25.86% of squared norm. [Complete group ranks](../../results/order-parameter-inference/locking-features-m24-261009/correlation-pair-rankings.csv); [top20 with at most one correlation endpoint](../../results/order-parameter-inference/locking-features-m24-261009/top20-at-most-one-correlation.csv).

**Lag interpretation.** Lag20 occurs in 68/top100, versus 18 for lag2; these counts overlap. With sampling interval .5, lag20 is ten physical time units, not a coupling delay. Frequency pulling strongly changes long-lag phase offsets; lag2 contrasts positive within-A with negative within-B correlation. Three estimators and squared variants replicate that contrast, so high ranks are not independent mechanisms. The longest configured lag is not established as optimal: no longer-lag sweep has been performed, and oscillatory sensitivity is not monotonic in lag.

**Beyond two correlation endpoints.** The highest remaining coordinates compare Johansen statistics with different lag/trend settings (ranks26–29), lag20 magnitude with conditional entropy/time-lagged MI, or LCSS with transfer entropy. Johansen sensitivity here is not evidence of cointegration in bounded oscillators. The implementation's `xme_*_k1` estimates one-step conditional entropy; its Gaussian version and Gaussian time-lagged MI are closely related to squared one-step Pearson, not independent proof of nonlinear dependence. LCSS×TE contrasts sequence similarity with additional predictive information and matches the edge-level interpretation below. Duplicate LCSS constraint variants should not be counted as separate discoveries. The PCA/readouts are unchanged; this is descriptive interpretation, not a refitted non-correlation model.''')]


def insert_correlation_composition(target=TARGET):
    before=target.read_bytes();notebook=nbf.read(target,as_version=4)
    if any(c.cell_type=='code' and 'weight-raincloud.png' in c.source for c in notebook.cells):
        raise ValueError('Composition already inserted; preserve edits.')
    index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='markdown' and c.source.startswith('### Individual SPI–SPI features:'))
    old=deepcopy(notebook.cells);added=correlation_composition_cells();notebook.cells[index:index]=added
    assert notebook.cells[:index]==old[:index] and notebook.cells[index+len(added):]==old[index:]
    assert target.read_bytes()==before;nbf.validate(notebook);nbf.write(notebook,target)


def non_double_correlation_cells():
    return [nbf.v4.new_markdown_cell(r'''### Top weights with at most one classical-correlation endpoint

Same M24 varying-independent-sensor-noise PC1; no refitting or selection using Q. Titles give the **global rank among all 25,417 eligible coordinates** by absolute weight. Blue is held mean z with 10–90% instance bands; black is Q on its own right axis. Johansen labels use d for deterministic setting (0: constant, 1: linear trend), and l for the number of lagged differences. These are statistic settings, not measured physical delays or demonstrated cointegration.'''),
            nbf.v4.new_code_cell("_appendix_show('locking-features-m24-261009/top16-weight-non-double-correlation.png')")]


def insert_non_double_correlation(target=TARGET):
    before=target.read_bytes();notebook=nbf.read(target,as_version=4)
    if any(c.cell_type=='code' and 'top16-weight-non-double-correlation.png' in c.source for c in notebook.cells):
        raise ValueError('Filtered grid already inserted; preserve edits.')
    index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='markdown' and c.source.startswith('### Individual SPI–SPI features:'))
    old=deepcopy(notebook.cells);added=non_double_correlation_cells();notebook.cells[index:index]=added
    assert notebook.cells[:index]==old[:index] and notebook.cells[index+len(added):]==old[index:]
    assert target.read_bytes()==before;nbf.validate(notebook);nbf.write(notebook,target)


def m24_edge_profile_cells():
    return [nbf.v4.new_markdown_cell(r'''### Matched edge profiles — M24, varying independent sensor noise

The same three held recordings are used in every row: fixed $\gamma=0,.10,.14$, selected nearest sensor-noise SD $\eta=.6$, not by Q or feature value. Columns are before, near and after the approximate boundary; they are illustrative single recordings, not population estimates. Every point is a directed channel pair (symmetric entries overlap). Six colours/markers identify A–A, B–B, C–C, A–B, A–C and B–C; larger black-outlined markers are group means. Each row has identical axis limits across controls and includes every point. Titles report the global PC1-weight rank and the exact edge correlation $z_{ab}$.

**First set:** Johansen short/long-history comparison (rank26), LCSS–TE (rank42), and lag2 Pearson–squared lag20 Pearson (rank7). All matrices are the original archived p90 outputs, verified against source hashes and the stored SPI–SPI feature bank. No estimators or readouts were refitted.'''),
            nbf.v4.new_code_cell("_appendix_show('locking-features-m24-261009/edge-profiles-mechanisms.png')"),
            nbf.v4.new_markdown_cell(r'''**Second set:** squared lag20 Kendall against one-step kernel conditional entropy (rank37) and Gaussian time-lagged MI (rank41). This compares long-delay alignment with short-horizon uncertainty/shared information. Entropy and MI can give opposite profile changes because conditional entropy subtracts shared information from target entropy; Gaussian MI remains a function of one-step correlation. These panels alone do not establish nonlinear or causal information transfer. [Record selection, matrix hashes and replay audit](../../results/order-parameter-inference/locking-features-m24-261009/edge-profile-audit.json).'''),
            nbf.v4.new_code_cell("_appendix_show('locking-features-m24-261009/edge-profiles-information.png')")]


def insert_m24_edge_profiles(target=TARGET):
    before=target.read_bytes();notebook=nbf.read(target,as_version=4)
    if any(c.cell_type=='code' and 'locking-features-m24-261009/edge-profiles-mechanisms.png' in c.source for c in notebook.cells):
        raise ValueError('M24 profiles already inserted; preserve edits.')
    index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='markdown' and c.source.startswith('### Individual SPI–SPI features:'))
    old=deepcopy(notebook.cells);added=m24_edge_profile_cells();notebook.cells[index:index]=added
    assert notebook.cells[:index]==old[:index] and notebook.cells[index+len(added):]==old[index:]
    assert target.read_bytes()==before;nbf.validate(notebook);nbf.write(notebook,target)


def repair_font_and_add_weight_plot(target=TARGET):
    """Surgical requested repair: font fallback, stale matching warnings, two new cells."""
    before=target.read_bytes();notebook=nbf.read(target,as_version=4)
    old="'font.serif': ['Computer Modern Roman']"
    new="'font.serif': ['Computer Modern Roman', 'DejaVu Serif']"
    warnings={"findfont: Generic family 'serif' not found because none of the following families were found: Computer Modern Roman",
              "findfont: Font family ['serif'] not found. Falling back to DejaVu Sans."}
    removed=0;changed=[]
    for i,c in enumerate(notebook.cells):
        if c.cell_type!='code':continue
        if old in c.source:c.source=c.source.replace(old,new);changed.append(i)
        outputs=[]
        for output in c.outputs:
            if output.output_type=='stream':
                lines=output.text.splitlines(keepends=True)
                kept=[line for line in lines if line.strip() not in warnings]
                removed+=len(lines)-len(kept);output.text=''.join(kept)
                if not output.text:continue
            outputs.append(output)
        c.outputs=outputs
    if not any(c.cell_type=='code' and 'top20-weights.png' in c.source for c in notebook.cells):
        index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='code' and 'top16-weight.png' in c.source)
        notebook.cells[index:index]=weight_bar_cells()
    previous=notebook.metadata.get('font_fallback_repair',{})
    notebook.metadata['font_fallback_repair']=dict(font_cells=sorted(set(changed+previous.get('font_cells',[]))),
        removed_stale_warning_lines=removed+previous.get('removed_stale_warning_lines',0),
        reason='Computer Modern remains the TeX font; installed DejaVu Serif supports native font lookup.',date='2026-10-09')
    assert target.read_bytes()==before
    nbf.validate(notebook);nbf.write(notebook,target)
    print(dict(font_cells=changed,removed_warning_lines=removed))


def insert_ranked_feature_cells(target=TARGET):
    """Insert only the requested galleries; preserve every other existing cell/output."""
    before=target.read_bytes();notebook=nbf.read(target,as_version=4)
    if any(c.cell_type=='markdown' and '### Unfiltered feature rankings' in c.source for c in notebook.cells):
        raise ValueError('Ranked galleries already inserted; preserve user edits.')
    index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='markdown' and c.source.startswith('### Individual SPI–SPI features:'))
    old=deepcopy(notebook.cells);added=ranked_feature_cells();notebook.cells[index:index]=added
    assert notebook.cells[:index]==old[:index] and notebook.cells[index+len(added):]==old[index:]
    assert target.read_bytes()==before
    nbf.write(notebook,target)


def build(source=SOURCE):
    original = nbf.read(source, as_version=4)
    result = deepcopy(original)
    extra = appendix()
    result.cells.extend(extra)
    result.metadata['interpretation_appendix'] = dict(
        source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        preserved_cells=len(original.cells), date='2026-10-08',
        reproduction='python -m research.inference.benchmarks.build_lean_interpretation_appendix')
    assert result.cells[:len(original.cells)] == original.cells
    return result, original, extra


def save(target=TARGET, replace_generated=False):
    if target.exists() and not replace_generated:
        raise FileExistsError(f'Preserve existing copy: {target}')
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result, original, extra = build()
    if target.exists():
        existing = nbf.read(target, as_version=4)
        previous = json.loads((PREVIEW/'notebook-audit.json').read_text()) if (PREVIEW/'notebook-audit.json').exists() else {}
        unedited_previous = hashlib.sha256(target.read_bytes()).hexdigest() == previous.get('target_sha256')
        if existing.cells[:len(original.cells)] != original.cells or (not unedited_previous and [c.source for c in existing.cells] != [c.source for c in result.cells]):
            raise ValueError('Existing copy differs from this generated content; preserve its edits.')
    # Do not rerun inherited cells: they can write shared figures and have user-edited outputs.
    executable = nbf.v4.new_notebook(cells=deepcopy(extra), metadata=deepcopy(original.metadata))
    NotebookClient(executable, timeout=120, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    result.cells = deepcopy(original.cells) + executable.cells
    nbf.validate(result)
    assert result.cells[:len(original.cells)] == original.cells
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    images = sum('image/png' in o.get('data', {}) for c in executable.cells if c.cell_type=='code' for o in c.outputs)
    errors = [o for c in result.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    assert not errors and images == 18
    nbf.write(result, target)
    loaded = nbf.read(target, as_version=4)
    assert loaded.cells[:len(original.cells)] == original.cells
    receipt = dict(source_sha256=before, target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   target=str(target.relative_to(ROOT)), preserved_cells=len(original.cells),
                   appended_cells=len(executable.cells), appended_images=images, errors=0,
                   inherited_outputs='Preserved byte-content logically, not rerun; original file hash unchanged.')
    (PREVIEW/'notebook-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


def refresh_figure_outputs(target=TARGET):
    """Refresh display cells only, preserving user reruns/edits everywhere else."""
    replacements={
        'cross-frequency-locking-261006/physics.png':'locking-composites-261008/latex/physics.png',
        'cross-frequency-locking-confirm-261006/component-scatter-random-noise.png':'locking-composites-261008/latex/component-scatter-random-noise.png',
        'neurotycho-induction-261006/induction-trajectories.png':'locking-composites-261008/latex/induction-trajectories.png'}
    before_bytes=target.read_bytes()
    original=nbf.read(target,as_version=4);result=deepcopy(original)
    helper_index=next(i for i,c in enumerate(result.cells) if c.cell_type=='code' and 'def _appendix_show(' in c.source)
    helper=result.cells[helper_index]
    indices=[i for i,c in enumerate(result.cells) if c.cell_type=='code' and c.source.strip().startswith('_appendix_show(')]
    assert len(indices) in (11,13,14,15,16,18)
    for i in indices:
        for old,new in replacements.items():result.cells[i].source=result.cells[i].source.replace(old,new)
    executable=nbf.v4.new_notebook(cells=[deepcopy(helper)]+[deepcopy(result.cells[i]) for i in indices],metadata=deepcopy(result.metadata))
    NotebookClient(executable,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    result.cells[helper_index]=executable.cells[0]
    for i,c in zip(indices,executable.cells[1:]):result.cells[i]=c
    for i in range(len(result.cells)):
        if i not in indices and i!=helper_index:assert result.cells[i]==original.cells[i]
    assert all(o.output_type!='error' for c in executable.cells for o in c.get('outputs',[]))
    prefix_hash=hashlib.sha256(json.dumps(original.cells[:37],sort_keys=True).encode()).hexdigest()
    result.metadata['typography_refresh']={'font':'LaTeX Computer Modern','changed_figure_cells':indices,
                                           'preserved_prefix_sha256':prefix_hash,'date':'2026-10-08'}
    nbf.validate(result)
    assert target.read_bytes()==before_bytes,'Notebook changed during refresh; preserve concurrent edits.'
    nbf.write(result,target)
    receipt={'before_sha256':hashlib.sha256(before_bytes).hexdigest(),
             'after_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
             'preserved_prefix_sha256':prefix_hash,'refreshed_figures':len(indices),
             'other_cells_unchanged':True,'font':'LaTeX Computer Modern'}
    (PREVIEW/'typography-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    save()
