# Pearson, Spearman and mutual information

## Current case (2026-10-28): `form_ladder_261028.py`; figure `form_overview_261029.py` (main; its helpers still come from the row-layout draft `form_figure_261028.py`), viewed and regenerated, with separate panel files, in `notebooks/form_figure.ipynb`

One 4/4/4 channel layout, 2 × 2 over the maps. Twelve channels in three blocks of four observe one AR(1) latent ($a=0.8$). Block 1 is linear; block 2 is linear or monotone ($\sinh 2w$); block 3 is linear or folded ($w^2$). The "both" condition is the 4 linear / 4 monotone / 4 fold recording shown in the figure. Generator `mtsspi.generators.generate_form_ladder_mts(maps=...)`; T = 2,000, 20 recordings per condition, the same draws in every condition. Outputs in `results/cases/form-ladder-261028/`.

| $z$ (ordered entries) | linear | monotone | fold | both |
| --- | --- | --- | --- | --- |
| $r$–$\rho$ | 1.00 | 0.50 ± 0.12 | 0.995 | 0.87 ± 0.06 |
| $\rho$–MI | 0.94 | 0.94 | 0.62 ± 0.01 | 0.62 |
| $\rho$–Linfoot MI | 1.00 | 0.99 | 0.62 | 0.62 |

- **Why sinh rather than $e^{w}$.** sinh is odd, so monotone–fold pairs have $r=\rho=0$ in population. With $e^{1.5w}$ they get $r\approx0.35$ while $\rho\approx0$, a spurious fourth pair type in a mixed recording. $\mathrm{corr}(w,\sinh 2w)=2e^{2}/\sqrt{(e^{8}-1)/2}\approx0.38$.
- **The two effects do not add.** In "both", the fold pairs (low on both $r$ and $\rho$) add variance that $r$ and $\rho$ share, so the monotone break is diluted (0.87 against 0.50).
- **Noise enters before the map** ($x=g(w)$, $w=(u+s\varepsilon)/\sqrt{1+s^2}$). Then a monotone g changes the marginals but not the copula, so $\rho$ and MI are exactly unchanged and only $r$ moves (verified: identical $\rho$ matrices; $z(\rho,\mathrm{MI})$ flat). With noise added *after* the map, as in the archive, the textbook reading fails: under $u^3$, noise floods the map's flat bulk and $\rho$ drops more than $r$.
- **Why not the sigmoid.** For Gaussian input, $\mathrm{corr}(u,\sigma(\beta u))\ge\sqrt{2/\pi}\approx0.80$ for every $\beta$. At unsaturated strengths it moves $z(r,\rho)$ only to 0.95 ($\beta=15$). The archive got its contrast by saturating every pair near $r=1$, which leaves $\rho$ with no cross-pair variance and makes $z$ ill-conditioned.
- **What sets the size of the $r$–$\rho$ break:** form contrast relative to strength spread. Latent noise is log-spaced, $s\in[0.2,1]$, within each block (linear pairs $r\approx0.50$–$0.96$). Wider spreads or milder maps give smaller breaks: $u^3$ gives 0.90 / 0.82 / 0.58 for loadings 0.5–0.95 / $s$ 0.2–1 / loadings 0.8–0.95. Random geometric noise draws break harder on average, but vary between recordings (SD 0.2–0.28).
- **$z(\rho,\mathrm{MI})<1$ under linear** is the curvature of the Gaussian $\mathrm{MI}=-\tfrac12\log(1-\rho^2)$ over the strength range (population 0.944 for a comparable profile), not estimator error; Linfoot MI gives 1.00. The case reads changes relative to this baseline, so raw MI is kept in $z$, as in the representation.
- **Fold:** $r$ and $\rho$ both lose the linear–fold pairs and agree on which pairs are strong, so $z(r,\rho)$ returns to 0.995. The single-map conditions therefore occupy separate corners of the ($z(r,\rho)$, $z(\rho,\mathrm{MI})$) plane.
- **Superseded (same day):** a half-linear / half-mapped design with $e^{1.5w}$ (linear 1.00/0.95, monotone 0.52/0.95, fold 0.99/0.55).

## Earlier material

- `configs/`: generated corpora, named by date and generator setting. `260924_beta.yaml` is the current one.
- `spis/`: the SPI sets these corpora are run with (`r_rho_mi_260924.yaml` for the current corpus).
- `notebooks/`: `r_rho_mi_261020.ipynb` is the current figure viewer (Fig. 1-style edge planes, interpretation and flags; outputs in `results/cases/r-rho-mi-261020/`). `r_rho_mi_261016.ipynb` holds the construction, matrices and pair-type decomposition; the others are earlier explorations.
- `gen_sweep_configs.py`: writes a parameter sweep of configs from a base config.

```bash
python -m mtsspi.run_corpus --config research/cases/r_rho_mi/configs/260924_beta.yaml --count-only
python -m research.cases.refresh_case_figures --help
```
