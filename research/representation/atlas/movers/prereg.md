# Frozen before any output exists: whole-atlas movers beyond the spectral null and their edge carriers (critic 1, 17 November 2026)

Code: [`movers.py`](movers.py) (phases 1 and 2, chained on Gadi) and [`movers_score.py`](movers_score.py) (phase 3, local). Checksums are in `code.sha256`. Reading rules: [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md).

## Question

- Which of the ~37,000 z coordinates change under anaesthesia beyond what a stationary Gaussian process with each state's spectrum gives?
- What character does each one read?
- Is each change carried by a few edges or nodes, or broadly?

Most of every drug's z change is spectral (81–89%, findings.md), so every quantity here is net of the pyspi null banks.

## Data

- **Arm.** NeuroTycho 2-s arm: 1,260 windows, 126–127 channels, 272 SPIs, pyspi r13.
- **Real bank.** `atlas/pyspi/neurotycho-short`; reduced z in `neurotycho-short-reduced.npz`.
- **Null banks.** `null-stationary`: the state's mean cross-spectrum, one draw per window. `null-bias`: a null redrawn from a null.
- **Net change** per session u and contrast s against awake_ec: net = (real − stationary) + (bias − stationary), as in `discovery.py null-decompose`.
- **Halves.** Within each session and state, windows by row order; even (H0) and odd (H1).

## Phase 1: movers (feature level)

- **Usable coordinates.**
  - Finite in every bank.
  - Both SPIs have awake-reference reliability ≥ .3 (`neurotycho-short-reliability-all.json`).
  - |z₀| < .98, where z₀ is the real awake_ec mean.
- **Primary set.** KT, KTMD and PF drug sessions: 19 sessions, 4 animals.
- **Selection, on H0:**
  - |mean net| ≥ .04 over the 19 sessions;
  - the same sign in ≥ 15/19 sessions;
  - each animal's mean has that sign.
- **Confirmation, on H1:**
  - the same sign;
  - |mean net| ≥ .02;
  - ≥ 15/19 sessions of that sign.
- **Negative control.** The identical pipeline on awake_eo − awake_ec for the same 19 sessions. The estimated false-discovery proportion is n_EO / n_drug. Movers are reported with it, and nothing is claimed when it exceeds .2.
- **Ranking.** Confirmed movers are ranked by |H1 mean net|.
- **Each mover is labelled:**
  - its `assumptions.classify` class if it is a minimal pair;
  - otherwise its shortest chain of minimal links (`assumptions.links`);
  - its SPI module pair.
- **Per agent and for MD and sleep:** the H1 mean net, descriptively.

## Phase 2: edge carriers

- **Features decomposed:**
  - the top 60 confirmed movers;
  - the MI pair `mi_gaussian|mi_kraskov_NN-4_DCE-AUTO` as a named anchor, whether or not it moves.
- **Per window and bank.** Each SPI's ordered off-diagonal profile is centred and scaled to unit norm, as in `corr_upper`. Edge contribution q_ij = â_ij b̂_ij + â_ji b̂_ji, over unordered pairs, so that Σ q = z.
- **Per session, contrast and half.** Δq_net is the per-edge version of net. Stored for drug (or sleep) and for awake_eo, on H0, H1 and all windows.

## Phase 3: scoring

### How broad against concentrated is decided

This answers "how can you tell".

- **Held-out edge share S_e.** Rank edges by contribution in the direction of the feature's net Δz on H0. S_e is the share of the H1 net Δz carried by the top 1% of those edges.
  - Under broad carriage, selection on one half does not carry over, and S_e ≈ .01: every edge carries its 1%.
  - Tiers, on the median over primary sessions:
    - **BROAD:** S_e < .02;
    - **ENRICHED:** .02 ≤ S_e < .05;
    - **CONCENTRATED:** S_e ≥ .05.
- **Held-out node share S_n.** The same, with node contributions c_i = ½ Σ_j q_ij (Σ c = z), and the top 5% of nodes (6 nodes; uniform .05).
  - **BROAD:** < .10;
  - **ENRICHED:** .10–.25;
  - **CONCENTRATED:** ≥ .25.
- **Reported alongside, not decisive:**
  - E50, the fraction of edges needed to reach half of the in-sample Δz (uniform .5);
  - the participation ratio of |Δq|.

### Carriers and substrate

- **Carrier reproducibility.** The Jaccard overlap of the top-1% edges and the top-5% nodes between sessions of one animal, for same-agent and different-agent pairs, against awake_eo maps.
- **Shared substrate.** Within each session, the correlation between movers' Δq_net maps. Movers are clustered on the mean over sessions, to see whether many features move through the same edges.
- **Nodes.** The electrode labels of recurring top nodes, per animal. There are no electrode coordinates in the bank, so there is no anatomy claim. Label distance is reported as a proxy only.

### Interpretation

- A mover gets a character reading only when its pair is minimal and its class has a reading in `assumptions.READINGS`, and with that class's confound listed.
- Compound movers get their candidate links and a literal reading.

## Not decided here

- Physiology against artefact.
- Any mechanism.
- Anything at 10 s: there are no pyspi null banks at 10 s.

## Revision 1 (17 November, before any output existed)

### Why

The share statistics in Phase 3 were invalid. On a synthetic atlas, a planted *broad* change scored CONCENTRATED: a held-out top-1% share of .09, against .01 uniform. The reason is that per-edge contributions to a correlation are products of standardised profiles. They are heavy-tailed and repeat across windows under any structured change, and shares of a signed sum are also unstable (values above 1 and below 0).

### Broad against concentrated is decided by a drop test

This replaces S_e and S_n as the decision.

- **Method.**
  - Select the top 1% of edges by contribution to the net Δz on one half, in the direction of Δz.
  - On the other half, recompute net Δz with those edges removed, in all three banks. The correlation re-standardises the profiles itself.
  - Compare with 20 random drops matched on wake leverage (deciles of |â b̂| in awake_ec).
  - f is the retained fraction, pooled as Σ over sessions of Δz_drop / Σ Δz_full, both directions averaged.
  - Nodes: drop the edges incident to the top 6 nodes, against 20 random six-node sets. This is an edge subset, not a channel removal.
- **Decision**, calibrated on three synthetics (concentrated .60 against random 1.03; broad 1.09; broad on a skewed profile 1.16):
  - **LOCALISED:** f < .8 and below every random draw;
  - **BROAD:** f ≥ .9, or inside the random range;
  - **PARTIAL:** otherwise.
- **Not tiered:** a feature whose held-out |net| is below the confirmation bar (.02).
- **Shares are descriptive only.**

### Decomposition set

- Every confirmed minimal pair.
- The remaining confirmed movers by |H1 net|, with at most 3 per SPI and 60 in all.
- The MI anchor.
- Phase 1 also reports how many confirmed movers each SPI takes part in, and the count by class.

### Carrier reproducibility

Edges are matched by electrode-label pairs, so sessions with different channel lists are compared on the edges they share.

### Run order

1. Phase 1 alone. Stop if the eyes-open false-discovery estimate exceeds .2.
2. Phase 2, the drop test (a second pass over the shards). Time one unit and estimate the SU before submitting.

### Reading limit

The stationary Gaussian null is looser than IAAFT. Movers involving rank statistics, cointegration or `xcorr_mean` may reflect marginal shape or slow within-window drift, and are flagged as such.
