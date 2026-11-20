# Chatterjee's xi on the axis-screen windows (post hoc)

Criteria stated and committed before the run. Code: [xi_screen.py](xi_screen.py); job: [jobs/xi.pbs](jobs/xi.pbs). Catalogue entry: `xi` in [catalogue.md](catalogue.md). Everything here is post hoc relative to the frozen axis screen ([../axis_screen/prereg.md](../axis_screen/prereg.md)).

## Question

Does Chatterjee's ξ, a rank statistic that tends to 1 only when one channel is a function of the other (including non-monotone functions) and has a fixed Gaussian value ξ_G(ρ) = (3/π) arcsin((1 + ρ²)/2) − ½ ([identities.json](identities.json)), show a drug-specific departure from its Gaussian value beyond IAAFT? And does it carry anything that the MI anchor, MinCovDet or distance correlation do not?

## What is computed

- **Windows and nulls:** every window of the 10-s arm (primary) and the 2-s arm (secondary), normal-scored, all channels. The screen's seven versions (real, 3 spectrum-final joint IAAFT draws, stationary, zero-phase and bias Gaussian nulls) are regenerated with its code and seeds.
- **New rows:**
  - `xi`: symmetrised ξ at lag 0;
  - `xi_g`: ξ_G of the window's r;
  - `xi_asym`: |ξ(i→j) − ξ(j→i)|;
  - `xi1`: symmetrised ξ at lag 1.
- **Existing rows:** the 33 screen statistics and KSG are read from the screen's stored profiles, not recomputed.
- **Reproduction check:** before anything is read, the regenerated r profile must match the stored one, and the 33 × 33 block of z must match the screen's, both within 1e-4 in every window and version. If either fails, nothing is scored and the mismatch is reported.
- **Scoring:** the frozen scorer, unchanged except for two things. The `xi` rows are declared higher-order (F1, real − mean of 3 IAAFT); `xi_g` is a function of r and stays linear. The coordinates below are added to its named list. Every gate, threshold and control is the screen's. The F1 Benjamini-Hochberg family now also holds the ξ coordinates, so the existing coordinates' q values are reported from the screen, not from this run.

## Coordinates

**Primary:** `xi_g|xi` (F1). Under any Gaussian process its z is fixed by r up to estimator noise, so its real − IAAFT net reads departure of ξ's edge ordering from the Gaussian-copula ordering. No direction is predicted.

**Secondary**, each reported with its gates:

| Coordinate | Role |
|---|---|
| `r2\|xi` | same contrast without the map (saturation check against the primary) |
| `rho\|xi` | monotone (Spearman) against functional rank dependence; non-monotone dependence |
| `dcorr\|xi` | two general dependence statistics; different weighting |
| `mcd2\|xi` | link to the MinCovDet axis |
| `xi\|ksg` | link to the MI anchor (32 channels) |
| `xi_g\|xi_asym` | asymmetry of functional dependence; zero in expectation under a Gaussian |
| `xi\|xi1` | lag 0 against lag 1; diagnostic |

**Levels** (drug − awake eyes-closed, real, IAAFT and net, animal t(3), per agent):
- mean over edges of ξ − ξ_G: positive means more functional dependence than the Gaussian copula with the same r;
- mean ξ_asym;
- mean ξ.

**Screen:** every F1 coordinate involving a ξ row (37 statistics give 666 all-channel pairs plus 37 against KSG) goes through the screen's BH step at q ≤ .05 with every animal mean the same sign. Rows that pass are listed and read with the reading guide.

## Outcomes and readings

- **Primary passes every gate.**
  - If the window-level r with the MI net (`r_with_mi_net`, within drug sessions) is ≥ .5, ξ sees the anchor's feature: the same copula-level departure, now from a rank statistic that ignores marginals by construction. It goes into the MI family, not a new axis.
  - If r < .5 and `mcd2|xi` fails, it is a candidate new character axis: drug-specific departure from Gaussian-copula functional dependence. It is named only after a synthetic check of which generator moves it (shared gain, covariance-regime mixture, non-monotone coupling), because the level sign alone does not separate them.
- **Screened but fails gates:** reported with the failing gates, not read.
- **Primary not screened:** ξ adds nothing the screen does not already carry at this power.
- **Ceiling or reliability failure** (z₀ > .98 or ξ edge-profile reliability < .3 in awake eyes-closed): uninformative at this window length. At T 200 the ξ SD is about .05 for every ρ, so the 2-s arm is expected to fail reliability.
- **Level:** read only with the primary.
  - A positive ξ − ξ_G net with a negative primary means more functional dependence than Gaussian, unevenly over edges.
  - A level near 0 with a moving primary means the ordering changes without a net excess.

## Cost

ξ costs about 0.1 s per window-version at M 126, T 1,000. The rest is regenerating the IAAFT and Gaussian-null versions and reading 8.7 GB of profiles per arm.
- **Expected:** about 25 SU (10 s) and 10 SU (2 s) on one 48-core node.
- **Worst case:** 0.1 kSU.
