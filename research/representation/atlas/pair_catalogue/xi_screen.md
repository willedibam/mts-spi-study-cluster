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

## Result (20 November)

Outputs: `results/representation/atlas-pair-catalogue-xi-261120/` (`score-long.json`, `score-short.json`, `netcorr-long.json`); the npz files are on Gadi scratch, `atlas/pair_catalogue/`. Jobs 184750735 (10 s) and 184750736 (2 s): 12.8 SU in all, exit 0.

**Reproduction passed.** In every window and version of both arms, the regenerated r profile matched the stored one to 3e-8 and the 33 × 33 z block to 6e-8.

**Primary `xi_g|xi`: uninformative under the stated rule.**
- 10 s: Δ −.0061 (t₃ −3.5, −5.0 null SD, 23/23 sessions, eyes-open ≈ 0), but z₀ = .9802, just over the .98 ceiling, so it is not usable and not screened.
- 2 s: Δ −.018 (−7.9 SD); passes every gate except reliability (Kin2 changes sign after disattenuation). ξ's reliability falls by .17 under drug, and the window-mean profile Δ is a third of the per-window one.
- The ceiling rule misfires here. With the exact Gaussian map, z₀ is 1 minus estimator noise by construction, so a correctly mapped pair is the one most likely to exceed .98 even though its null SD is well defined. Proposal: judge ceiling for mapped pairs by the null SD, not by z₀ (see catalogue guide proposals).

**Screen (10 s): 9 F1 axes involve a ξ row. Only one is ξ-specific.**

| Coordinates | Δ | Reading |
|---|---|---|
| `mcd2\|xi`, `mcd2\|xi1`, `mcd2\|xi_g` | −.053 to −.056 | the MinCovDet axis; ξ is a stand-in for r (`r2\|mcd2` −.056) |
| `icoh_bb\|xi`, `icoh_lo\|xi`, `icoh_bb\|xi1`, `icoh_lo\|xi1` | +.020 to +.027 | the existing imaginary-coherence-against-static family (`dcorr\|icoh_bb`, `tau2\|icoh_bb`, `l1\|icoh_bb` are axes); window-level r with `icoh_bb\|r2` net .92 |
| `xi_g\|ksg` | −.013 | the MI anchor with the arcsin map (r with the MI net .92) |
| `mi_g\|xi` | −.0078 | ξ-specific: see below |

**`mi_g|xi`: the primary's contrast with the −½ log(1 − r²) map. Passes every gate at 10 s.**
- Statistics: Δ −.0078, t₃ −5.1, −6.0 null SD, q 1e-4, 23/23 sessions, eyes-open +.0008, z₀ .965, reliability .93/.89.
- Window-mean profile Δ −.0053 (68% of the per-window Δ). Sleep ratio .10.
- Agents: ketamine −.015, ketamine–medetomidine −.008, propofol −.005, medetomidine −.003 (small, as in the MinCovDet and band exceptions).
- At 2 s: −.019 (−8.6 SD), fails reliability.
- **Post hoc, exploratory (`xi_screen.py netcorr`):** window-level correlation within drug sessions, n = 230.
  - With the primary's net: .90.
  - With the MI anchor's net: .43.
  - With the MinCovDet axis (`r2|mcd2`) net: .20.
  - With the ξ − ξ_G level net: −.71.
- Reading: under drug, beyond IAAFT, ξ's edge ordering departs from what r predicts under a Gaussian copula, and windows with a larger ξ excess over ξ_G show the larger departure (r −.71), so the excess is uneven over edges.
  - It shares under half its window variance with the MI anchor (rMI .43 < .5), so it is not placed in the MI family.
  - It is nearly independent of the MinCovDet axis at window level.
- The stated rule ("`mcd2|xi` fails") cannot test the MinCovDet link: `mcd2|xi` moves exactly as `r2|mcd2` does, because it reads MinCovDet against any linear-like partner. The window-level correlation (.20) replaces it, and is post hoc.
- Status: **candidate, not named.** It is map-dependent (the stated primary is ceiling-excluded), small (Δ −.008), and its generator is untested. Naming needs the synthetic check stated above (shared gain, covariance-regime mixture, non-monotone coupling at matched r). The shared gain is the live alternative: in the smoke it raised ξ above ξ_G by +.049.

**Levels (drug − awake eyes-closed, net of IAAFT).**
- 10 s: mean ξ − ξ_G +.013 (t₃ 6.9; 4/4 animals; all agents, medetomidine smallest +.005). Asymmetry +.0012 (t₃ 4.0).
- 2 s: mean ξ − ξ_G +.022 (t₃ 8.5). Asymmetry +.003.
- Read with the primary (positive level, negative ordering): more functional dependence than the Gaussian copula gives, unevenly over edges. The level's sign does not separate a gain from a regime mixture.

**2-s axes involving ξ:** only `xi_g|ksg` (anchor) and `pec|xi1`, `pec_log|xi1` (envelope against lag-1 ξ; diagnostic row, not read).

**Cost:** 12.8 SU (expected about 35, worst case 0.1 kSU). The IAAFT regeneration was faster than estimated.
