# CML order-parameter inference

## Question and current scope

For the diffusively coupled quadratic map
`x_i(t+1)=(1-eps)f_alpha(x_i)+eps/2[f_alpha(x_{i-1})+f_alpha(x_{i+1})]`,
`f_alpha(x)=1-alpha*x^2`, ask whether a scalar learned only from SPI–SPI
meta-features infers changing order as alpha varies at `eps=0.3`.

The user restricted this stage to the existing alpha sweep and precomputed SPI
features. No pyspi or new trajectories are computed. Full-lattice validation is
a later cluster recommendation, not current evidence.

## Verified archive and split

- Alpha `1.60:0.01:2.00`; 20 independent alpha-specific seeds; observed
  `M=20`, `T=1000`; physical `N=100`; central crop; burn-in 2000; 820/820
  datasets complete.
- Existing 39,567-dimensional SPI–SPI feature rows are loaded, not recomputed.
- Strict internal split: PCA10 + Isomap1 (15 neighbours) fitted without alpha or
  observables on instances 0--9 at even-grid alpha; evaluated on instances
  10--19 at unseen odd-grid alpha. Coordinate sign is conventional.
- This is leakage-controlled internally but not pristine external confirmation:
  the archive informed earlier exploration.

## Literature-grounded interpretation

- Pattern selection gives sharp spatial spectral peaks; intermittency adds
  broadband power; fully developed spatiotemporal chaos loses the peaks:
  https://csc.ucdavis.edu/~chaos/chaos/pubs/pstc-title.html
- Kaneko's exact transition study uses pattern distributions/transitions,
  static/dynamical entropies, lifetimes, spectra, and Lyapunov spectra:
  https://doi.org/10.1016/0167-2789(89)90227-3
- The explicit zigzag-collapse disorder parameter is `Delta=1-Q(1)`, with
  domain-length distribution `Q(D)` and associated `S_p`, `S_d`. It belongs to
  the weak-coupling zigzag branch; applicability at `eps=0.3` is unsupported.
- Exact-model work warns that long-lived mixed states at `a=1.8, eps=.3` can be
  supertransient, whereas `a=1.88` is fully developed chaos:
  https://chaos.phys.msu.ru/loskutov/PDF/TMPh_quadratic_cml.PDF

## Frozen current-data primary and alternatives

- Primary `Q_sel`: fraction of non-DC spatial power in crop modes `k/pi=0.3`
  and `0.4`. The two modes were selected from low-alpha discovery fields only
  and frozen before internal confirmation. It is a window-level operational
  spectral-order coordinate, not a full-lattice thermodynamic observable.
- Mode-agnostic sensitivity:
  `Q_spec=sqrt((m*sum_k(p_k**2)-1)/(m-1))`, normalized spectral concentration.
- Other tested quantities: maximum mode share; spatial spectral entropy;
  length-4 spatial-pattern static/dynamical entropy; Kaneko `Delta`, `S_p`,
  `S_d`; period-2 activity; temporal spectral entropy; thresholded activity.
- Baselines: raw mean/std, lag-1/lag-2 correlation, neighbour correlation,
  period-2 activity, temporal spectral entropy.
- Turbulent fraction is not promoted: the saved data provide no objective
  laminar/burst rule and the result changes materially with threshold.

## Existing-data result

- `Q_sel`: mean 0.612 at alpha 1.60 and 0.314 at 2.00; difference 0.298,
  seed-bootstrap 95% CI `[0.221, 0.377]`.
- Largest observed mean drop of both `Q_sel` and `q` is at alpha `1.71` on the
  available 0.01 grid; this is descriptive, not a critical-point estimate.
- Internal confirmation: `|rho(q,Q_sel)|=0.911`, stratified 95% CI
  `[0.852,0.950]`; alpha-centroid `|rho|=0.955`.
- After subtracting each alpha-cell mean: `|rho|=0.611`, 95% CI
  `[0.519,0.692]`. Agreement is not only common monotonic dependence on alpha.
- Simple summaries are competitive: temporal entropy and recurrence baselines
  have overall `|rho|` about 0.91. This supports capability, not superiority or
  necessity. The SPI coordinate is more aligned with temporal disorder than
  with the mode-agnostic spectral magnitude.

## Claim and remaining limitation

Defensible now: "Within the existing finite CML sweep, an unlabeled SPI–SPI
meta-feature recovers a changing, literature-grounded spectral-order coordinate
associated with the pattern-to-turbulence region."

Calling `Q_sel` an operational order parameter is a plausible scoped pilot
interpretation. The unqualified physical sentence remains unproved because the
archive has only a short crop, one physical size, no T/N scaling, and cannot
exclude supertransients.

Later cluster design: full-ring `N={128,256,512}`, prefixes
`T={8192,16384}`, burn-in 65536, 24--48 seeds, alpha 0.01 globally and 0.002
over 1.68--1.80. Freeze modes and SPI transform first; require T convergence,
sensible N behavior, stationary blocks, and seed-clustered held-out q--Q
association. Current notebook:
`notebooks/embeddings/cml-order-parameter-inference.ipynb`.
