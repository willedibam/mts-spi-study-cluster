"""Build and execute the order-parameter composite notebook (iteration 2, 2026-10-21)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT/'research/inference/benchmarks/notebooks/order-parameter-composites_261021.ipynb'


def cells():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md(r'''# Order-parameter composites: unsupervised $q$ tracking a physical $Q$

The paper's section-4 figure (iteration 4, 2026-10-21). The left column is the main case, 2:1 cross-frequency locking between resonant oscillator communities (the M24 varying-sensor-noise arm). The right column stacks three further systems: the measured electronic oscillators (ER sweep), the fine Stuart–Landau locked-to-unsteady sweep, and the collective period-two 2D coupled map lattice. In every panel $q$ is the centred PC1 of SPI–SPI $z$, fitted without $Q$, control or nuisance labels; $Q$ is computed independently from the physics. Code: [`composite_figures.py`](../composite_figures.py) (registry `SYSTEMS`). Every panel is also exported on its own (SVG, PDF, 600-dpi PNG) under `results/order-parameter-inference/composites-261021/panels/<system>/`. Text is typeset by LaTeX in Computer Modern when `latex` and `dvipng` are available, with a native Computer Modern-math fallback otherwise.

**Reading the figure.** (a) Control against $Q$ (black, left axis, physical units) and $q$ (teal, right axis). Lines are means over held instances, and bands are their 10–90% range (not confidence intervals). The hardware has one recording per setting. $q$ is a PC score with arbitrary scale and sign, so its axis carries no ticks. Each inset is a robust-scaled MTS heatmap (boxed; locking communities A/B/C marked on the first) of a short window of one recording, with a physical glyph above it and a leader line to its control on $Q$. Legends give $|\rho_s|$ with $Q$. (b) $Q$ against mean-$|r|$ and mean-MPI PC1, without ticks. (c) $z_{ab}$ for named feature pairs against control on a fixed $[-1,1]$ range; triangles mark the snapshot controls. (d) Edge planes: SPI $a$ against SPI $b$ across all $M(M-1)$ ordered channel pairs of the snapshot recordings, in the same columns as (c); $z$ is written in the corner that is emptiest across the row. (e–g) Compact panels: heatmaps with glyphs above the curves. $Q$ (left axis) is overlaid with $q$, mean-$|r|$ and mean-MPI PC1, each affinely rescaled so its mean curve spans $[0,1]$. Bands are drawn for $Q$ and $q$ only. The mean-MPI sign is arbitrary and is shown correlating positively with $q$; display only, no statistic changes.

**Trajectories or bands.** Bands are the default. In locking every control–seed pair is an independent realisation, so faint lines joining equal seed indices would suggest paths that do not exist. Bands also keep the overlaid right-column panels legible. A trajectory version is kept below. It shows something bands hide: in Stuart–Landau, Kuramoto and CML2D, where seeds are paired across controls, individual instances step more sharply than their mean.

**What is claimed.** $q$ tracks $Q$ across controls. This is not a calibration of $Q$, not a mechanism, and not superiority over simple statistics: in hardware, Stuart–Landau and CML2D the baselines track $Q$ as well or better by rank. Feature names follow what calibration supports.

**Snapshots** follow fixed rules, never $Q$ or $q$. Locking: the held recordings with sensor noise nearest $\eta=.6$ at $\gamma=0,.10,.20$ (start, transition, end). The $\gamma=.20$ MPI was fetched read-only from Gadi and matches the archived hash. Hardware: ER settings 0, 23 (the last setting with phase slips) and 87. Stuart–Landau: instance 0 at $\gamma=.68$, $.725$ (the steepest interval) and $.77$. Kuramoto: the first held seed. CML2D: the fixed pilot seed of the three-systems figure; the curves are the 32-seed confirmation bank.

**Glyphs (defaults).** Locking: the wrapped slow–fast phase $\Phi(t)=\arg Z_B-\arg Z_{A,2}$, the quantity whose time-averaged phasor on a later window defines $Q$. Hardware: the wrapped carrier phase of the four late-locking channels (12/18/21/24) relative to the most coherent core channel. Stuart–Landau and Kuramoto: the global order-parameter magnitude $R(t)$. CML2D: the whole 256 × 256 lattice (archived state after the reference window), with the 32 observed sites circled. Every glyph is regenerated from the archived generator and checked against the archived input. Heatmaps show 80 samples (locking), 0.08 s (hardware), 150 and 200 samples (Stuart–Landau, Kuramoto) and 40 steps (CML2D), so individual oscillations are resolved.'''),
        code('''import io
import sys
from pathlib import Path
ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p/'src/mtsspi/spi_spi_contract.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from PIL import Image as PILImage
from IPython.display import Image, display
from research.inference.benchmarks import composite_figures as cf
pd.set_option('display.width', 220)

def show(path, width=1500):
    """Display a downsampled preview; the saved PNG/SVG/PDF files are full resolution."""
    image = PILImage.open(path)
    image.thumbnail((width, 10*width))
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    display(Image(data=buffer.getvalue()))

systems = {name: cf.load(name) for name in ('locking', 'hardware', 'stuart-landau', 'kuramoto', 'cml2d')}
locking, hardware, sl, kuramoto, cml = systems.values()
pd.DataFrame({name: {k: d.metrics[k] for k in ('q', 'mean-|r|', 'mean-MPI')} for name, d in systems.items()}).round(3)'''),
        md(r'''## Feature contract and replay checks

$z$ must use all $M(M-1)$ ordered off-diagonal entries (`unified_ordered_v3`). For each locking and hardware snapshot recording, the whole stored $z$ row (41,616 coordinates) is rebuilt from the archived MPIs with the library builder and compared with the bank; each displayed edge profile is replayed. The legacy upper-symmetrised value is listed alongside: it differs for the directed LCSS × TE pair, so the check can tell the contracts apart. The locking replay reproduces the archived noisy observations exactly. Stuart–Landau and Kuramoto glyphs come from regenerated trajectories that match the archived inputs (to float32 precision for Stuart–Landau). CML2D $q$ values are replayed from the frozen model by `three_systems.load_cml`.'''),
        code('''checks = pd.DataFrame([dict(system=name, record=c['row_id'], **{k: v for k, v in c.items() if k != 'row_id'})
                       for name, d in systems.items() for c in d.provenance.get('snapshot_checks', [])])
checks['row_contract'] = checks.row_contract.map(lambda c: c if isinstance(c, float) else f"passes={c['passes']}, max error {c['max_error']:.1e}")
display(checks)
edges = pd.DataFrame([dict(system=d.spec.name, **r) for d in (locking, hardware) for r in d.provenance['edge_receipts']])
edges.assign(feature=edges.feature.str.replace('\\n', ' ', regex=False)).round(4)'''),
        md(r'''## Composite, default: bands

The main headline grows to balance the column heights.'''),
        code('''show(cf.save(cf.composite_stacked(locking, [hardware, sl, cml]), 'composite-stacked-band'))'''),
        md(r'''## The same with faint instance trajectories'''),
        code('''show(cf.save(cf.composite_stacked(locking, [hardware, sl, cml], spread='trajectories'), 'composite-stacked-trajectories'))'''),
        md(r'''## Alternative middle panel: Kuramoto instead of Stuart–Landau

At $M=N=32$ the Kuramoto sweep is a gradual finite-size crossover with wide seed spread. Its $q$ is the weakest of the candidate systems (.83), the PC1-dominance screen was not met, and mean-$|r|$ (1.00) and mean-MPI (.99) are near-perfect, because synchronisation strength is the order parameter here. Stuart–Landau is recommended for the panel; see the notes below.'''),
        code('''show(cf.save(cf.composite_stacked(locking, [hardware, kuramoto, cml]), 'composite-stacked-kuramoto'))'''),
        md(r'''## Glyph options

All variants at the three snapshots. Locking: wrapped $\Phi$ (default), spiral ($\Phi$ as angle, time as radius), Lissajous ($\cos\arg Z_A$ against $\cos\arg Z_B$ over 60 time units). Hardware: wrapped late-channel phase (default), $R(t)$, phase raster. CML2D: lattice (default) or the global mean $\bar x(t)$ over 40 steps.'''),
        code('''show(cf.save(cf.glyph_gallery([locking, hardware, sl, kuramoto, cml]), 'glyph-gallery'), width=700)'''),
        md(r'''## Hardware: which physical quantity?

No universal order parameter is known for this apparatus. **Default:** the across-channel range of mean carrier frequency (log axis), whose collapse marks the cessation of phase slips at 23→24. **Alternatives:** mean carrier phase coherence $R_1$ (almost a monotone function of mean-$|r|$: rank correlation 1.00 on ER), the temporal SD of $R(t)$, or the fraction of frequency-locked channels (needs a threshold). With $R_1$, $q$'s rank association is similar, but $Q$ no longer shows the step that $q$ shows at slip cessation.'''),
        code('''hardware_r1 = cf.load('hardware', Q_source='R1', Q_label='Carrier coherence $R_1$', Q_log=False, legend_loc='lower right')
show(cf.save(cf.compare_Q([hardware, hardware_r1], ['Frequency range', 'Mean carrier coherence']), 'hardware-Q-alternatives'), width=1000)'''),
        md(r'''## Baselines: where simple statistics do as well

| system | $q$ | mean-$\lvert r\rvert$ | mean-MPI PC1 | rows |
|---|---|---|---|---|
| locking (held) | .871 | .032 | .020 | 168 |
| hardware ER, settings ≤ 87 | .892 | .950 | .982 | 22 |
| Stuart–Landau fine | .886 | .915 | .883 | 152 |
| Kuramoto, $M=N=32$ | .826 | .998 | .994 | 128 |
| CML2D confirmation | .883 | .966 | .973 | 544 |

Values are $|\rho_s|$ with $Q$. Only locking separates $q$ from both baselines; there the advantage is the ordering of variation under recording-specific noise, not information absent from the means (a supervised mean-vector readout reaches .900). In hardware the descriptive difference is localisation: $q$ crosses 43% of its displayed range over the slip-cessation interval 23→24, against 4.5% for mean-$|r|$ and 8.9% for mean-MPI ([baselines notebook](order-parameter-benchmarks-baselines.ipynb), section 7).'''),
        md(r'''## Notes on the open questions

**Locking glyph: wrapped $\Phi$ is recommended.** It is the quantity $Q$ summarises ($Q=|\langle e^{i\Phi}\rangle|$ on a later window), so the glyph and the curve read as one object. It separates all three snapshots at a glance (continuous sawtooth, intermittent slips, flat band) and is scale-free. It also shares its visual language with the hardware glyph. The Lissajous is the most familiar picture of 2:1 locking, but over any fixed window it shows $\gamma=.10$ as a filled box like $\gamma=0$: it hides partial locking, exactly the transition state. It also depends on the window length and on the latent community mean fields rather than anything observed. The spiral encodes the same information as $\Phi$ but is denser and harder to read at inset size.

**Stuart–Landau over Kuramoto.** (i) The fine Stuart–Landau sweep is an independent confirmation: a frozen coordinate on 152 fresh rows, with every pre-outcome gate passed. The $M=N=32$ Kuramoto result is exploratory, since its PC1-dominance screen was not met. (ii) The Stuart–Landau transition is sharp, and $q$, mean $R$ and $\mathrm{sd}_t(R)$ have the same steepest interval (.720–.725). At $N=32$ Kuramoto is a gradual crossover, with $q$'s steepest interval adjacent to $Q$'s. (iii) Locked-to-unsteady changes the temporal character of the collective dynamics (constant against oscillating $R$), which fits the paper's character-first framing. Kuramoto synchronisation is strength itself, so mean-$|r|$ reaches .998 and $q$ (.826) looks weakest exactly where the framing says it should not be used. Kuramoto is probably the better-known model (inferred from its standing as the canonical synchronisation model, not measured), but recognition does not outweigh (i)–(iii). The Stuart–Landau path is Matthews & Strogatz's $K=.8$ cut, which keeps a classical reference.

**CML2D: what to show.** $Q$ is the period-two amplitude of the spatially averaged state on a disjoint future window. The default glyph is the whole lattice with the observed sites circled. It shows why $Q$ falls: one period-two phase fills the lattice at $r=3.84$, large domains of both phases coexist near $3.862$, and fine domains at $3.89$ average the collective oscillation away. It also shows that 32 scattered sensors sample this structure. Caveat: the lattice is the archived state after the reference window, not simultaneous with the observed window, and comes from the pilot seed. The alternative, the global mean $\bar x(t)$ (a large zigzag when ordered), is $Q$ itself and so partly redundant with the curve.

**Edge-plane groups.** Within-community pairs are split into A–A and B–B (two blues), and C–C joins the pairs involving the reference community C. For LCSS × TE the split adds little: A–A, B–B and C–C sit together (LCSS ≈ .9, TE ≈ .05), and the change is in the A–B pairs (TE .00 → .06 at almost constant LCSS ≈ .78). The split matters for the other two locking pairs, so one scheme serves any edge row. For Pearson lag 2 × lag 20², A–A and B–B move apart: at γ = 0.14 the A–A centroid is (.35, .00) and B–B is (−.44, .56), the documented within-community redistribution. For Johansen, B–B separates on the trace axis.

**Statistics placement.** $|\rho_s|$ with $Q$ sits in the legends: each number is bound to its curve's colour and symbol, and the legend is needed anyway to identify four overlaid curves. A grey one-liner would repeat the legend's names, and defining $Q$ in-panel needs symbols that only the caption can define.

**Hardware $Q$ and range.** Keep the carrier frequency range on settings ≤ 87. It measures the calibrated event, the cessation of phase slips, and it steps exactly where $q$ steps. $R_1$ is effectively a strength measure (rank correlation 1.00 with mean-$|r|$), so choosing it would turn $Q$ into a baseline. Beyond 0.435 V the grid is sparse (six settings to 3.5 V), $q$ dips at 0.875 V without a change in $Q$, and the calibrated spectral explanation fails at settings 175–437. The caption should say that $q$ is fitted on all 29 ER recordings and give the full-sweep association (.822).

**Which systems have $M<32$?** Locking: $M=24$ (with an $M=48$ confirmation); hardware: $M=28$. Fine Stuart–Landau and full Kuramoto: $M=N=32$ only. The broad Stuart–Landau sweep ($\gamma=.55$–$1.25$) has separate $M=N=8,16,32$ systems at $T=100/500/1000$. CML2D has paired $M=16$ views at $T=500$ and $1000$ (288 rows each, nested sensors from the same masters, 9 old-grid controls) beside the primary $M=32$. All panels use $M=32$ or the full system.'''),
        md(r'''## Which axes do the leading locking features read?

The filtered top 16 of the locking PC1 hold three families: nine Johansen short- against long-history statistic pairs, six squared lag-20 rank correlations against one-step entropy or time-lagged MI, and two LCSS × TE variants. The table adds Pearson lag 2 × lag 20². Retrospective and descriptive: held records, no refitting.'''),
        code('''summary, redundancy, blocks = cf.locking_feature_audit()
display(summary.round(3))
families = {'Johansen': redundancy.index.str.startswith('coint'), 'lag-20 x one-step': redundancy.index.str.contains('tau-20-sq x'),
            'LCSS x TE': redundancy.index.str.startswith('lcss'), 'Pearson lag 2 x lag 20^2': redundancy.index.str.startswith('corr_pearson_tau-2 ')}
display(pd.DataFrame({f: {g: redundancy.loc[m, n].abs().to_numpy()[~(m[:, None] & n[None, :] & (redundancy.index.to_numpy()[:, None] == redundancy.index.to_numpy()[None, :]))[m][:, n]].mean()
                          for g, n in families.items()} for f, m in families.items()}).round(2))
blocks.assign(pair=blocks.pair.str.replace('_', ' ')).round(2)'''),
        md(r'''**Reading.** The first table gives each feature's association with $Q$, sensor noise and mean-$|r|$. The second gives the mean $|r|$ between feature families across held records. The third decomposes $z$ on the four local snapshot MPIs.

- **One within-community lag-profile axis.** Johansen, lag-20² × one-step dependence and Pearson lag 2 × lag 20² are near-copies across records (family-mean $|r|$ ≈ .85–1.0). Their change is carried by A–A and B–B pairs: removing the A–B pairs barely moves $z$, while the A–A/B–B-only $z$ flips sign. As B's frequency is pulled to twice A's, within-community dependence at long lags reorganises. This is temporal redistribution, compatible with frequency pulling, not a reading of the A–B coupling.
- **Why so many lag-20² × entropy/TLMI pairs.** One-step conditional entropy and lag-1 MI are short-lag dependence measures, so each pair contrasts long-lag with short-lag dependence across pairs, the same contrast as Pearson lag 2 × lag 20². Pearson, Spearman and Kendall at lag 20 are interchangeable here, and PCA spreads loading over near-duplicates, so one contrast fills six slots. Gaussian lag-1 MI is related to squared lag-1 Pearson (rank .75–.79 across edges) but is not a copy, so the correlation filter is not simply leaking.
- **LCSS × TE is the one feature that reads the coupled pairs.** Without the A–B pairs its $z$ stays at .55–.66 at every $\gamma$; the full $z$ falls from .62 to −.14 only because of the A–B pairs. Between A and B, TE rises from about 0 to .05–.09 at locking while LCSS stays at about .78, as expected for 2:1 locking: one oscillator becomes predictive of the other without waveform alignment. It also has the lowest noise association (.08) at $|\rho_s(Q)|$ .89. Caveats: the TE values are small KSG estimates on noisy data, and frequency- and spectrum-matched controls are still pending.'''),
        md(r'''## Which axes do the leading locking features read?

The filtered top 16 hold three families: Johansen short- against long-history statistics, squared lag-20 rank correlation against one-step entropy or lag-1 MI, and LCSS × TE. Pearson lag 2 × lag 20² is added for reference. The tables give association with $Q$, noise and mean-$|r|$; correlation across held records; and $z$ on the local MPIs with the A–B pairs removed or with A–A/B–B pairs only. Retrospective and descriptive.'''),
        code('''summary, redundancy, blocks = cf.locking_feature_audit()
display(summary.round(3))
display(redundancy.round(2).set_axis(range(len(redundancy)), axis=1))
blocks.round(2)'''),
        md(r'''- **Johansen, lag-20² × one-step and Pearson lag 2 × lag 20² are one axis.** They correlate at $|r|$ ≈ .85–1.0 across held records, and their change is carried by A–A and B–B pairs, not A–B pairs. This is within-community redistribution of long-lag dependence as B's frequency is pulled to twice A's. It is compatible with frequency pulling and does not read the coupling itself. Six slots go to the lag-20² family because Pearson, Spearman and Kendall at lag 20 are interchangeable here, and PCA spreads loading over near-duplicates.
- **LCSS × TE reads the coupled pairs.** Without the A–B pairs its $z$ stays at .55–.66 at every $\gamma$. Between A and B, TE rises from about 0 to .05–.09 at locking while LCSS stays at about .78: predictive dependence without waveform alignment, as expected for 2:1 locking. It has the lowest noise association (.08). The TE values are small KSG estimates, and frequency-matched controls are still pending.'''),
        md(r'''## Export individual panels'''),
        code('''written = {'locking': cf.export_panels(locking)}
for name in ('hardware', 'stuart-landau', 'kuramoto', 'cml2d'):
    written[name] = cf.export_panels(systems[name], stacked=True)
receipt = cf.receipt(list(systems.values()))
print({k: len(v) for k, v in written.items()}, 'panels (each as SVG, PDF and 600-dpi PNG) under', cf.OUT/'panels')
print('provenance:', receipt.relative_to(ROOT))'''),
        md(r'''## Open choices

Decided (2026-10-21): Stuart–Landau over Kuramoto; wrapped $\Phi$ and the other default glyphs; bands; locking snapshots at $\gamma=0,.10,.20$.

1. **Hardware $Q$ and range.** Frequency range on settings ≤ 87 (recommended), $R_1$, or the full sweep.
2. **Feature titles.** Spell out the SPIs: LCSS similarity (unconstrained) × KSG transfer entropy (4 neighbours, history 1); Johansen max-eigenvalue statistic (constant term, 10 lagged differences) × trace statistic (linear trend, 1 lagged difference); Pearson $r$ at lag 2 × squared Pearson $r$ at lag 20 (samples, i.e. 1 and 10 time units); kernel stochastic interaction × interdependence score; Pearson $r$ at 2 ms × 20 ms.
3. **$Q$ definitions.** In the caption or methods rather than in-panel: each needs symbols that only the caption can define.'''),
    ]


def build(target=TARGET):
    nb = nbf.v4.new_notebook(cells=cells())
    nb.metadata['kernelspec'] = {'display_name': '.venv (3.12.12)', 'language': 'python', 'name': 'python3'}
    NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': str(target.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
    assert not errors, errors
    nbf.validate(nb)
    nbf.write(nb, target)
    return target


if __name__ == '__main__':
    print(build())
