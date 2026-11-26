# kshift side by side: how much of each completed MI-pair drug effect survives the shift-corrected KSG

Post-hoc secondary, declared on 26 November 2026 in [`preregs/claar-prereg.md`](preregs/claar-prereg.md), [`preregs/human-prereg.md`](preregs/human-prereg.md) and [`preregs/kauvar-prereg.md`](preregs/kauvar-prereg.md) (commit d8caa4f8), before any kshift value was computed. Code: [`kshift/kz.py`](kshift/kz.py) (k4, kshift and k2te on the same windows and IAAFT draws) and [`kshift/kscore.py`](kshift/kscore.py) (gate, the frozen scorers rerun per estimator, R table). No frozen primary or verdict changes.

## Submission (26 November)

Snapshot `operations/sources/mi-kshift-d8caa4f8` (src and `research/representation/atlas` only, 413 files), job `jobs/kshift.pbs`, 48 cores and 64 GB each. Claar and Kauvar windows were copied from the session-notes archive to `atlas/kshift-inputs/` on scratch (62 files, sha256 matched). PRIOS reads its windows on gdata. Outputs go to `atlas-mi-pair-261117/kshift/<study>/` on gdata.

| Study | Job | Windows | Expected kSU | Worst kSU |
|---|---|---|---|---|
| Claar (primary, induction, late) | 185283890 | 983 + 406 + 235 | 0.1 | 0.38 (4 h) |
| PRIOS post-deviation primary | 185283891 | 1,064 | 0.03 | 0.38 (4 h) |
| Kauvar | 185283892 | 600 | 0.05 | 0.38 (4 h) |
| PRIOS S-full | 185283893 | 3,911 (all channels, 10 draws) | 0.35 | 0.96 (10 h) |

Before submission:
- `kz.py check`: the per-edge k4, kshift and k2te equal `ksgbias.edge_all` exactly.
- k4 gate, k4 only, so no kshift value was printed. It was exact on the machine that made each stored z: macOS for 4 Claar primary, 2 induction, 2 late and 3 Kauvar windows; a Gadi login node for 2 PRIOS windows.
- The same PRIOS windows on macOS gave a real-window gap of 1.8e-4, with the surrogates exact. Normal-scored windows share one quantile set, so their distance ties are exact and different hardware breaks them differently (inferred from that pattern). Claar and Kauvar run on Gadi may therefore show a gap of this size on real windows. R is computed against the same run's k4.

## Results (26 November)

All four jobs exited 0. Walltime and charge (48 cores × 2 SU/h): Claar 21 min, 34 SU; PRIOS 4.7 min, 7.5 SU; Kauvar 13 min, 21 SU; PRIOS S-full 2 h 12 min, 212 SU; 0.27 kSU in all. Outputs, with checksums, are copied to `results/representation/atlas-mi-pair-261117/kshift/<study>/` (score.txt is the full log: the gates, the frozen scorers run on each estimator's view, and the tables below).

Reproduction on Gadi, over every window:
- PRIOS was exact: real and surrogate gaps 0 (primary on the gate windows, S-full on all 3,911 windows).
- Claar showed a maximum real-window gap of 2.9e-5 (primary), 2.4e-5 (induction) and 1.8e-5 (late), and Kauvar 1.7e-4. Every surrogate gap was ≤ 1.9e-15. This is the platform-tie pattern declared in the preregs (the stored Claar and Kauvar z were made on macOS). The gate prints FAIL at 1e-6 for this reason only.
- The frozen scorers run on the Gadi k4 column reprint every frozen line to the printed decimal: Claar primary −0.0191 [−0.0260, −0.0132], 16/17, −7.3 null SD, REPLICATES; Δ_ind −0.0063, 9/9, −2.4, PRESENT; Δ_late −0.0000, 6/13, DRIFT NEGLIGIBLE; PRIOS −0.0083, 5/5, p .0095, REPLICATES; Kauvar +0.0000 [−0.0053, +0.0051], null SD .0055, ABSENT. PRIOS S-full −0.0083, 6/7, p .0226 (7 subjects included), REPLICATES.

R = Δ(kshift)/Δ(k4) is taken against the same run's k4. The own null SD is the split-half SD across units of that estimator's residuals. "Sign = k4" counts units whose Δ has the same sign under both estimators. The real/surrogate split is the per-unit Δ of the real z and of the surrogate mean z (residual = real − surrogate). k2te is a candidate only and is descriptive.

| Study, estimand | Estimator | Mean Δ | Own null SD | Δ / null SD | R | Units < 0 | Test | Sign = k4 | Δ real z | Δ surr z | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Claar primary, iso − awake | k4 | −0.0191 | .0026 | −7.3 | 1 | 16/17 | sign p .0001 | | −0.0190 | +0.0001 | REPLICATES (frozen) |
| | kshift | −0.0173 | .0026 | −6.7 | 0.91 | 16/17 | sign p .0001 | 17/17 | −0.0169 | +0.0004 | **SURVIVES** |
| | k2te | −0.0188 | .0023 | −8.0 | 0.98 | 16/17 | sign p .0001 | 17/17 | −0.0181 | +0.0006 | |
| Claar Addendum A, Δ_ind | k4 | −0.0063 | .0026 | −2.4 | 1 | 9/9 | sign p .0020 | | −0.0075 | −0.0012 | PRESENT (frozen) |
| | kshift | −0.0061 | .0026 | −2.4 | 0.97 | 9/9 | sign p .0020 | 9/9 | −0.0074 | −0.0013 | **SURVIVES** |
| | k2te | −0.0048 | .0023 | −2.1 | 0.77 | 9/9 | sign p .0020 | 9/9 | −0.0059 | −0.0010 | |
| Claar Addendum B, Δ_late | k4 | −0.0000 | | | | 6/13 | | | | | Δ_late/Δ_ind +0.01, DRIFT NEGLIGIBLE (frozen) |
| | kshift | −0.0000 [−0.0006, +0.0004] | | | | 7/13 | | 12/13 | | | +0.01: drift does not explain |
| | k2te | −0.0000 | | | | 6/13 | | 13/13 | | | +0.01 |
| PRIOS post-deviation primary, propofol − awake | k4 | −0.0083 | .0137 | −0.6 | 1 | 5/5 | t(4) −3.81, p .0095 | | +0.0151 | +0.0235 | REPLICATES (frozen) |
| | kshift | −0.0097 | .0158 | −0.6 | 1.17 | 5/5 | t(4) −5.71, p .0023 | 5/5 | +0.0191 | +0.0288 | **SURVIVES** |
| | k2te | −0.0133 | .0139 | −1.0 | 1.59 | 5/5 | t(4) −5.91, p .0021 | 5/5 | +0.0136 | +0.0269 | |
| Kauvar, infusion − reference | k4 | +0.0000 [−0.0053, +0.0051] | .0055 | +0.0 | undefined | 3/7 | sign p .77 | | +0.0191 | +0.0191 | ABSENT (frozen) |
| | kshift | −0.0007 [−0.0053, +0.0035] | .0082 | −0.1 | undefined | 3/7 | sign p .77 | 7/7 | +0.0165 | +0.0173 | **ABSENT holds** |
| | k2te | +0.0001 [−0.0044, +0.0048] | .0045 | +0.0 | undefined | 3/7 | sign p .77 | 7/7 | +0.0170 | +0.0169 | ABSENT holds |
| PRIOS S-full (all good channels, 10 draws), propofol − awake | k4 | −0.0083 | .0126 | −0.7 | 1 | 6/7 | t(6) −2.52, p .0226 | | +0.0217 | +0.0300 | REPLICATES (frozen) |
| | kshift | −0.0110 | .0131 | −0.8 | 1.32 | 7/7 | t(6) −2.87, p .0142 | 6/7 | +0.0241 | +0.0350 | **SURVIVES** |
| | k2te | −0.0139 | .0120 | −1.2 | 1.68 | 7/7 | t(6) −3.53, p .0062 | 6/7 | +0.0165 | +0.0304 | |

Per-unit Δ under k4 → kshift, the real/surrogate split per unit and the BI secondaries are in each score.txt. The rows under each study:
- **Claar primary.** Every mouse keeps its sign. The one positive mouse (569068) stays positive (+0.0041 → +0.0039). The largest shrinkage is in 551399 (−0.0569 → −0.0439) and 571619 (−0.0118 → −0.0085). The effect remains carried by the real windows (Δ real z −0.0169, surrogates +0.0004). The burst secondaries are unchanged: within-iso Spearman(residual, BI) median +0.78 with 17/17 positive (k4 +0.75); the continuous-like iso windows (BI ≥ .85, 7 sessions) give +0.0003 (k4 −0.0004), so under both estimators the effect is borne by burst-suppression windows.
- **Claar burst caveat.** It does not arise: the declared ambiguity was for a low R, and R is 0.91. The subtracted shift offset is larger under isoflurane (+0.0202 real) than awake (+0.0084), and the same on the IAAFT surrogates (+0.0199, +0.0088). Joint IAAFT keeps the linear cross-correlation at all lags, so this equality shows only that the subtracted part is structure the surrogates reproduce (spectrum and linear lagged dependence), not nonlinear shared burst envelopes beyond 2.5 s; it does not show that the offset is pure independence bias (inferred).
- **Claar Addendum A.** The frozen `claar_ind_sum.py` hard-codes the k4 null SD .0026 and prints kshift −2.3 (PRESENT) and k2te −1.9 (ABSENT). The table uses each estimator's own null SD (kshift .0026, −2.4; k2te .0023, −2.1); kshift's own SD rounds to the same .0026, and −0.0061/.0026 = −2.35 is printed as −2.3 by the scorer and −2.4 in the table. The kshift reading is the same on either scale. k2te sits on the −2 boundary.
- **Claar Addendum B.** kshift Δ_late is −0.0000, far from Δ_ind/2 = −0.0031, so drift does not explain the induction effect under kshift either.
- **PRIOS.** The frozen REPLICATES rule holds under kshift (mean < 0, p .0023, no subject > 0) with R 1.17. The T = 200 caveat does not bite: R is above 1, not pushed down. Under both estimators the residual falls because the surrogate z rises more than the real z (Δ real z +0.0191, Δ surr z +0.0288 under kshift). This is the surrogate-driven fall already recorded for the k4 primary, so kshift does not change what the PRIOS effect is made of. The split-half null SD (.0158; −0.6 SD) is as uninformative as under k4 at n = 5 (robust null SD .0033, −2.9). Shift offsets are larger at T = 200 than in Claar (+0.0438 awake, +0.0449 propofol, real) and again equal real vs surrogate (+0.0428).
- **Kauvar.** R is undefined (the table's arithmetic value is −82, meaningless at Δ(k4) = +0.0000). kshift Δ_inf is −0.0007, −0.1 own null SD, CI lower −0.0053 > −.010: ABSENT holds. kshift's own null SD (.0082) is half again as wide as k4's (.0055), which loosens the band, but on the k4 scale the mean is also −0.1 SD, so the reading does not depend on it. Per-subject signs agree with k4 in 7/7. Offsets are small and equal across states and real vs surrogate (+0.0049 to +0.0056).
- **PRIOS S-full.** The frozen REPLICATES rule holds under kshift with R 1.32 (7/7 negative, p .0142). The one subject that is positive under k4, PRIOS01 (+0.0011), turns negative under kshift (−0.0018), so kshift removes the one exception the rule allowed; the other six keep their sign (k4 → kshift: PRIOS02 −0.0038 → −0.0080, PRIOS03 −0.0105 → −0.0125, PRIOS04 −0.0041 → −0.0031, PRIOS05 −0.0092 → −0.0134, PRIOS06 −0.0260 → −0.0316, PRIOS09 −0.0054 → −0.0062). The effect is surrogate-driven as in the primary (Δ real z +0.0241, Δ surr z +0.0350). Split-half null SD .0131 (−0.8 SD); robust null SD .0030 (−3.6; k4 −2.7). Offsets +0.0433 awake, +0.0439 propofol (real), +0.0425 and +0.0428 on surrogates.

Not rescored: PRIOS S-A, S-B, S-AB and ds004541 (declared). NeuroTycho 2-s needs no compute: [`robustness-result.md`](robustness-result.md) already has the 32-channel 2-s kshift rows (own bank R 1.29, which fails the calibration rule; cut arm R 1.15, passing at −2.04 SE). Those rows use the same `neurotycho-short` bank (1,260 windows) at 32 channels. The old 2-s arm (16-channel subsets, 126-channel joint IAAFT, the atlas pipeline before `edges.py recompute`; [`preregs/robust-prereg.md`](preregs/robust-prereg.md)) is not rescored with kshift. That needs its own gate against the old pipeline; it is left as a follow-up.
