# Incumbent nonlinearity level against z (draft text, 24 November 2026)

Draft text for the paper, not in `main.tex`. Numbers: [`level_vs_z.py`](level_vs_z.py) → `results/representation/atlas-mi-pair-261117/level_vs_z.txt` (KSG k = 4, from the local `zns` arrays) and `est/score.txt` (k = 10 and kernel MI, nats only).

## Table (supplement)

Drug (or sleep) minus eyes-closed wake, real minus joint IAAFT, normal-scored, 10-s macaque, 32-channel subsets. Each readout in its own eyes-open session SD (z .0020, level .0023 nats); k = 10 and kernel level in nats, because their per-window arrays are on gdata only and their SDs were not computed.

| Agent (sessions) | z, k = 4 | Level, k = 4 | Level, k = 10 (nats) | Level, kernel (nats) |
|---|---|---|---|---|
| Ketamine (4) | −8.0 SD, 4/4 down | **+6.4 SD, 0/4 down** | +.019 | +.011 |
| Ketamine–medetomidine (11) | −7.8, 11/11 | −3.7, 11/11 | +.000 | −.004 |
| Medetomidine (4) | −3.1, 4/4 | −0.9, 2/4 | +.001 | +.001 |
| Propofol (4) | −5.9, 4/4 | −6.5, 4/4 | −.004 | −.010 |
| Sleep (7) | −0.5, 7/7 | −0.4, 5/7 | −.000 | −.001 |

z falls under all three estimators for every agent (`est/score.txt`; kernel MI under medetomidine 3/4).

## Results paragraph (draft)

The established test for nonlinear dependence compares the mean amount of model-free MI beyond the Gaussian (linear) value with the same quantity in surrogates that keep each channel's spectrum and the cross-spectrum (Paluš 1995; Hlinka et al. 2011). Applied to the same windows, edges and surrogate draws, this level does not give a consistent answer across anaesthetics: beyond the surrogate it rises under ketamine (+6.4 eyes-open SD, 0/4 sessions down) and falls under propofol (−6.5 SD, 4/4) and ketamine–medetomidine (−3.7 SD, 11/11), with ketamine rising and propofol falling under all three MI estimators. A fall below the surrogate is also hard to read as less non-Gaussian dependence, because for normal-scored marginals the true MI cannot fall below its Gaussian value; it more likely reflects estimator bias that the surrogate does not fully cancel. The SPI–SPI coordinate falls under every agent (−3.1 to −8.0 SD, 23/23 sessions), every estimator, and much less in sleep (−0.5 SD). The two readouts answer different questions: the level asks whether there is more non-Gaussian dependence, z asks whether its organisation across edges has changed. Only the second has a stable answer here.

## Methods sentence (draft)

As the established comparison we computed the nonlinearity level, the mean over edges of model-free minus Gaussian MI on normal-scored windows, minus the same quantity in the window's joint IAAFT surrogate (Paluš 1995; Hlinka et al. 2011), with the same estimators, edges and surrogate draws as z, and expressed both as drug minus eyes-closed wake in units of the eyes-open session SD.

## Caveats

- **Estimator dependence is not shown.** Ketamine–medetomidine and medetomidine change sign between estimators only at values near zero (k = 10: +.0003, +.0007 nats against a k = 4 SD of .0023). So the claim is that the level's sign depends on the agent, not on the estimator, until the k = 10 and kernel SDs are computed (needs the `est_*.npz` arrays copied from gdata, no new compute).
- **Animals:** the ketamine level rise is one animal (Chibi +12.8 SD, George +0.1). The z fall under ketamine is also Chibi-heavy (−13.6 against −2.5) but has the same sign in both.
- **Recovery:** z stays below wake (−1.6 to −2.1 SD), the level returns (−0.3 to +0.6 SD).
- **Not done:** an edge-wise (Hlinka-style) count of nonlinear edges. The arrays hold one surrogate draw per window, too few for a per-edge threshold; it would need a new multi-draw Gadi run.
- **Framing:** this does not show that z beats the established test; they target different properties. It shows that the established property gives an agent-dependent sign that is hard to interpret, while organisation across edges moves the same way under every agent.
