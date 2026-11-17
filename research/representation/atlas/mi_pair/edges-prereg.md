# Frozen before any output is read: MI-pair recompute and edge-level concentration (critic 1, 17 November 2026)

Code: [`edges.py`](edges.py) and `jobs/edges_*.pbs`, with checksums in `edges-code.sha256` (frozen at submission). The scorer `edges_score.py` is added to the checksum file before any output is fetched. Exactness: on synthetic data, `edges.one_recompute` gives the same four z values as `review/zgk_ns.one` to every printed digit.

## Why

The user's reading is that a falling `z(mi_gaussian, mi_kraskov)` means the dependence became more nonlinear. Three existing results stop that from following by definition (`findings.md`):
- the kNN-minus-Gaussian level falls beyond IAAFT;
- a linear Gaussian network with a shared on/off gain lowers z (−0.068);
- in the one-animal hub test, the hub edges gained *less* non-Gaussian dependence than strength-matched controls.

A fall in z says that the two statistics rank pairs more differently; it does not say that dependence rose. The edge lens tests where the divergence sits and which way those edges move. It can turn the result into a more specific claim, or show that no such claim is available.

## Part R: recompute the lost arrays

- **Runs.** The ten runs repeat the original invocations, with seed = row index within each per-agent call:
  - stage A, `neurotycho-long-stage-a`: PF, KT, KTMD, MD and SLP, all with `rowmode first` (`rows[0]['M']` subset);
  - stage B, `neurotycho-long-stage-b`: PF, KT, KTMD and MD with `first`, and SLP with `row` (the per-row fix was applied only before that run; `review/EDITS-LOG.txt`).
- **Gate.** Scored with `review/sumall.py` logic, normal-scored, against the reported values:

  | Quantity | Reported | Tolerance |
  |---|---|---|
  | KT | −.0159 | ±.0005 |
  | KTMD | −.0155 | ±.0005 |
  | MD | −.0062 | ±.0005 |
  | PF | −.0117 | ±.0005 |
  | SLP | −.0010 | ±.0005 |
  | Eyes-open session SD | .0020 | ±.0002 |
  | Sessions down | 23/23 drug, 7/7 sleep | exact |

  - Pass: "re-run, matches to tolerance". It is not called bit-identical, because the originals ran on macOS and cannot be compared.
  - Fail: report the differences. The recomputed arrays become the record, and `findings.md` is corrected.

## Part E: edge arm on all channels

- **Data.**
  - Windows: all 1,260 10-s windows (stages A and B), states awake_ec, awake_eo, drug, sleep and recovery_ec.
  - Channels: every kept channel (126 or 127), average-referenced as stored, normal-scored per window.
  - Null: three spectrum-final joint IAAFT draws over all channels; seed = stage offset + 1000 × row + b.
  - This joint all-channel null is stricter than the 32-channel null of the headline (it roughly halved the 2-s residual), so its global values are not the headline's.
- **Per edge** (electrode pair i < j, by kept-channel label):
  - g: Gaussian MI from r;
  - k: KSG MI (k = 4, AUTO Theiler window);
  - k_s: mean KSG over the three draws.
  - **Excess** e = k − k_s.
- **Per session u.**
  - Δe = mean over drug windows − mean over awake_ec windows.
  - Control ΔeEO = awake_eo − awake_ec, in the same session.
  - Sleep sessions use sleep in place of drug.
- **Matched strength.** Edges are binned into deciles of their awake_ec mean g within the session. ẽ = Δe − that decile's mean Δe. This removes "stronger pairs change more".
- **Held out.** Within each state, windows split into even and odd by order. Select on one half, measure on the other, both directions, then average.

### Primary decisions, at the animal level

The primary set is KT, KTMD and PF pooled, over 4 animals (Chibi, George, Kin2, Su); each animal's value is the mean over its sessions. MD and sleep are reported the same way, descriptively.

**E1. Concentration.** Take the top 1% of edges by ẽ on the selection half. C is their share of the summed positive ẽ on the held-out half; uniform is .01.
- CONCENTRATED: C_drug > C_EO in 4/4 animals, and median C_drug ≥ .05.
- DIFFUSE EXCESS: C_drug > C_EO in 4/4, but median < .05.
- NOT CONCENTRATED: otherwise.
- The same is reported for the negative tail (edges losing excess), descriptively.

**E2. Reproducible locus within animal.** Take the Spearman correlation of ẽ maps between drug sessions of one animal, on shared electrode pairs (one map per session, all windows). Compare it with the same for EO maps.
- REPRODUCIBLE: mean drug ρ exceeds mean EO ρ by ≥ .10 in every animal with ≥ 2 drug sessions.
- Across animals there is no electrode correspondence, so no cross-animal locus is claimed.

**E3. Link to z.** Edge contribution to z: p_e = ĝ_e k̂_e, using the profiles standardised within the window. Take its residual (real minus the same quantity with k_s), Δ drug − awake_ec, so that Δz_res = mean over edges of Δp. Report Spearman(Δp, ẽ), held out, and whether the edges with the most negative Δp have ẽ above or below the median. Both outcomes are named in advance:
- (a) the edges that drive the z fall *gain* excess;
- (b) they *fail to gain* while matched pairs gain. This is the one-animal hub test's result.

### Descriptive

- **Node level:**
  - node score = mean ẽ over incident edges;
  - share of the top 5 nodes;
  - R² of an a_i + a_j fit to the ẽ map.
- **Global check:** z_res over all channels per agent.
- **Edge geometry:** inter-electrode index distance of the top-1% edges, using adjacency in label order as a proxy. Electrode coordinates are not in the manifest.

## Not decided here

- Physiology against artefact.
- The form (decoupling against shared gain).
- Any cross-feature (all-SPI) edge pass. That runs only if E1 or E2 holds out, and needs its own rule.
