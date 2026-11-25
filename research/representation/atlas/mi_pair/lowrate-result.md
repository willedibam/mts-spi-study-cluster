# Result: the macaque MI-pair drug fall at 20 and 10 Hz (25 November 2026)

- **Rule:** [preregs/lowrate-prereg.md](preregs/lowrate-prereg.md), frozen at f1157a0c before scoring; one declared deviation (figure (d), below).
- **Code:** [`lowrate/`](lowrate/) (`build.py`, `score.py`, `summarise.py`, `jobs/`); estimator and null are `edges.py` recompute's, unchanged. Full table: [`lowrate/summary.txt`](lowrate/summary.txt) (`summarise.py <scores> <100 Hz zns> D20 S20 L20 D10 S10 L10 D20L S20L D10L S10L`).
- **Gadi:** build 185156131 (8.2 SU) and its rerun (9.1 SU), gates A = 0.0 and B passed; scoring 185158076 (10-s arms, 132.5 SU) and 185158077 (long arms, 211.7 SU). 361 SU in all.
- **Outputs:** gdata `/g/data/ql44/we2614/mts-spi-study/representation/atlas-mi-pair-lowrate-261125/{banks,scores}/` with SHA256SUMS (all 40 score files verified locally); local `results/representation/atlas-mi-pair-lowrate-261125/`.
- **Dropped:** the 2-s arm (40 or 20 samples per window), as frozen.

## Arms

Pooled drug Δ = drug − awake eyes-closed per session, pooled over agents; SD = the arm's own eyes-open session SD; t(3) one-sided over Chibi, George, Kin2, Su. No arm's eyes-open or sleep control reaches 2 own SD.

| Arm | Rate (Hz) | Window (s) | T | Band (Hz) | ns Δ | ns SD | ns t(3) | raw Δ | raw SD | raw t(3) | K < 0 awake / drug |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor | 100 | 10 | 1000 | 0.5–40 | −.0132 | −6.7 | −6.11 | −.0135 | −8.8 | −5.37 | .17 / .33 |
| L20 | 100 | 10 | 1000 | 0.5–8 | −.0113 | −4.6 | −4.30 | −.0099 | −2.4 | −4.86 | .32 / .47 |
| L10 | 100 | 10 | 1000 | 0.5–4 | −.0091 | −2.8 | −3.59 | −.0055 | −1.2 | −2.75 | .44 / .51 |
| D20 | 20 | 10 | 200 | 0.5–8 | −.0060 | −1.5 | −1.88 | −.0028 | −0.8 | −1.44 | .13 / .12 |
| S20 | 20 | 10 | 200 | aliased | −.0074 | −1.0 | −2.74 | −.0075 | −1.3 | −2.46 | .16 / .13 |
| D10 | 10 | 10 | 100 | 0.5–4 | −.0012 | −0.2 | −0.63 | +.0026 | +0.4 | +2.74 | .10 / .09 |
| S10 | 10 | 10 | 100 | aliased | −.0004 | −0.1 | −0.06 | −.0003 | −0.0 | −0.10 | .14 / .11 |
| D20L | 20 | 50 | 1000 | 0.5–8 | −.0038 | −1.6 | −2.68 | −.0025 | −0.7 | −1.47 | .12 / .12 |
| S20L | 20 | 50 | 1000 | aliased | −.0050 | −1.7 | −5.71 | −.0058 | −2.0 | −3.31 | .15 / .13 |
| D10L | 10 | 100 | 1000 | 0.5–4 | −.0024 | −0.6 | −1.35 | −.0001 | −0.0 | −0.08 | .10 / .12 |
| S10L | 10 | 100 | 1000 | aliased | −.0051 | −1.7 | −3.24 | −.0054 | −0.8 | −2.96 | .15 / .13 |

Band retention B (prereg rule, D against S at the same N and duration):

| Pair | ns B, D − S t(3) | reading | raw B, t(3) | reading |
|---|---|---|---|---|
| D20 / S20 | .81, +1.63 | band-neutral | .38, +2.27 | inconclusive |
| D10 / S10 | 3.33, −0.17 | inconclusive | −8.30, +1.29 | inconclusive |
| D20L / S20L | .76, +2.04 | band-neutral | .43, +9.04 | weaker in slow band |
| D10L / S10L | .47, +9.61 | **weaker in slow band** | .03, +7.06 | weaker in slow band |

## Reading

- **Retraction.** The interim reading given to the coordinator and user, that the 10-s loss is mainly N (D20 = S20, L20 retains, spread ∝ 1/√T), is falsified by the long arms it was staked on. At T = 1000 the 20 Hz arms keep about a third of what L20 keeps in the same band at the same N (D20L −.0038 against L20 −.0113), and the 10 Hz arms about a quarter (D10L −.0024 against L10 −.0091).
- **N explains the noise, not the signal.** Eyes-open SD shrinks as predicted (D20 .0041 → D20L .0024; S20 .0074 → S20L .0029), but Δ does not grow (D20 −.0060 → D20L −.0038; S20 −.0074 → S20L −.0050).
- **Sampling rate orders the arms.** Every 100 Hz arm, whatever its band (0.5–40, 0.5–8, 0.5–4 Hz), has ns Δ −.009 to −.013; every 20 or 10 Hz arm, whatever its N (100–1000), has −.000 to −.007. The 2-s anchor (100 Hz, N = 200, 16 channels: −.018 to −.024, 22/23 sessions; `findings.md`) is a second N-matched contrast with D20/S20, though not channel-matched.
- **The sign survives at matched N outside KSG's negative regime.** S20L (t(3) −5.71) and S10L (−3.24) are negative in 4/4 animals with K < 0 at .13–.15, at about 40% of the anchor's raw Δ (−.0058 and −.0054 against −.0135). By agent it is carried by ketamine, which keeps 60–120% of its anchor Δ in every arm (ns −.0096 to −.0196 against −.0159); ketamine–medetomidine, propofol and medetomidine lose most or all of theirs below 100 Hz (KTMD −.0155 at the anchor, +.0020 to −.0063 in the 20/10 Hz arms).
- **"Band" is the wrong label for the D-vs-S readings.** The 10 Hz long pair reads weaker in the slow band (the fall needs content above 4 Hz) and the 20 Hz long pair is neutral on the primary (B .76, at the edge; its raw row is not promoted). But L10, which is the same 0.5–4 Hz signal sampled at 100 Hz, keeps 69% of the anchor's fall. So above-cutoff content matters when sampling is sparse and not when it is dense: D against S measured a band × rate interaction, not where the fall lives. The band question is asked at fixed 100 Hz by the band sweep ([preregs/bands-prereg.md](preregs/bands-prereg.md)).
- **Open confound: duration.** The long arms change rate and window duration together; the missing cell is 100 Hz at 50-s windows. S20 giving similar Δ at 10 and 50 s (−.0074, −.0050), and the 2-s anchor holding, argue against duration but do not test it at 100 Hz.
- **Mechanism (inference, untested).** The nonstat nulls put more than half the anchor's departure in a mixture over ~1-s stretches. One reading consistent with every arm: KSG registers that mixture through neighbour sets drawn from many samples within each stretch (100 per second at 100 Hz, 20 or 10 below), so the 100 Hz fall is real structure that the estimator sees only when it is densely sampled. No numerical match with the nonstat fractions is claimed. Under this reading the anchor is not an artefact, but its size is an estimator-and-sampling quantity, not a rate-invariant property of the signal.

## Diagnostics

- **K < 0** (normal-scored, real): .09–.16 in every 20/10 Hz arm with no drug rise, against .17 → .33 awake → drug at the anchor and .32–.51 in the oversampled L arms. The arms that lose the fall are not in KSG's negative regime; the arms that keep it are.
- **Theiler cap** (T // 10): hit on ≤ 0.2% of edges in every arm and state. Median window is 0.1–0.3 s at every rate (2–5 samples at 20/10 Hz, 10–24 in the L arms), so the Theiler window does not distinguish the arms.
- **IAAFT fit on long windows:** real K < 0 is below the surrogate's in every state of every long arm (gap .01–.06; equal at the anchor), and the awake residual is a little more negative (D20L −.0022, D10L −.0024 against the anchor's −.0014). The stationary surrogate fits 50–100-s windows less well, as the prereg anticipated.
- **D10L is not drug-specific:** its sleep Δ (−.0034) exceeds its drug Δ (−.0024), though neither reaches 2 own SD.

## Figures

`plot_character_draft.py --rate 20|10 --lowrate results/representation/atlas-mi-pair-lowrate-261125` draws the anchor figure on D20 and D10 (`results/.../mi-figure-{20,10}hz.{svg,png,pdf}`; local, regenerable). Panels (b) and (c) re-pick windows by the median rule; (c)'s own-SD scale widens ∝ 1/√T and its drug shift shrinks except for ketamine; (d)'s macaque mark has 4/4 animals below zero at 20 Hz and 1/4 at 10 Hz. The default figure's SVG and PNG are byte-identical to before the `--rate` change; the PDF is not byte-reproducible run to run whatever the code.

**Deviation 1, figure (d) extension.** The prereg triggers extending (d) to mouse and human at low rate on a "stronger" or "weaker" band reading; D10L/S10L reads weaker. It was not run: L10 shows the D-vs-S contrast did not isolate band (above), so the extension would carry a band label the data do not support. Whether to run it is the coordinator's decision.

## Recommendation

- **Supplement, not main text.** Report that the macaque fall is a 100 Hz result: at 20 and 10 Hz it keeps its sign at matched N (S20L, S10L 4/4) at about 40% of the raw size, and only ketamine is rate-robust. Do not describe the MI pair as rate-invariant, and do not attribute the low-rate loss to N or to band.
- **Next cells, if pursued:** (i) 100 Hz on 50-s windows (T = 5000), the missing duration cell; (ii) a Theiler-window-in-seconds sweep at fixed rate, for the KSG-bias workstream; (iii) the band sweep, now running (addendum below).

## Band-sweep gate, decided before any band score was read

The bands prereg reads LP2, LP4 and LP8 only if D20L recovers the fall (t(3) ≤ −2.35). D20L meets that by the letter on normal scores (−2.68) but not in substance (−1.6 own SD; raw t −1.47; a third of L20's Δ in the same band). **LP2, LP4 and LP8 are therefore descriptive only.** Its "broadband" outcome (LP_f and HP_f both retain) can no longer separate a shared gain mixture from a rate-carried effect, since every 100 Hz arm so far retains whatever its band; it is reported descriptively. The HP ladder is read as frozen, and an HP or LP arm that is lost at 100 Hz is the informative outcome. Jobs 185180701–3 (snapshot 99cc8632), not yet returned when this was committed.
