# Frozen test: is the anaesthetic z drop amplitude-linked decoupling or shared gain?

Frozen 16 November 2026, before any gate value was computed on real windows in this configuration. Code: `amplitude/code/` (gate.py, gate_calib.py, gate_run.py, gate_sum.py).

## Question

The z(Gaussian MI, kNN MI) residual beyond IAAFT falls under anaesthesia (macaque, mouse). Two physical readings fit a fall:
- **Local-event decoupling:** high-amplitude episodes that are not shared loosen coupled pairs.
- **Shared gain:** a common amplitude envelope makes coupling stronger when amplitude is high, through signal-to-noise.

The readout below separates them in calibration.

## Readout (as calibrated)

- **gate** (`mech.feats` definition, vectorised in `gate.py`): per edge, the correlation over 0.5-s blocks of block Fisher r with the block mean log-SD of the two channels. Block r is oriented by the sign of the edge's window r.
  - The orientation was added at calibration. Without it, negative edges cancel positive ones in multichannel data. The earlier stage-A `ampcoup.py` readout was signed, so its "nothing consistent" may be that artefact.
- Averaged over a fixed edge set: the top quartile of |r| in the session's reference-state windows. It is chosen once per session and used for every window, state and surrogate.
- **Residual:** real − mean over the joint all-channel IAAFT draws (spectrum-final) of the normal-scored window.
- **Secondary readouts, not diagnostic in calibration:** imbal and switch. They are reported only.

## Calibration (32 channels, T = 1000, 14 windows each; `gate_calib.py`)

| Kind | gate residual (window SD) |
|---|---|
| Linear stationary | +0.002 (0.036) |
| Local events | −0.344 (0.042) |
| Shared gain plus noise floor | +0.199 (0.021) |
| Mixture | +0.105 (0.035) |

imbal is negative and switch positive in every non-linear kind, so they are not diagnostic. A mixture can cancel: NEITHER does not rule out both mechanisms acting together.

## Confirmatory test: mouse isoflurane induction (Claar, DANDI 000458)

No amplitude readout has been computed on these windows before.

- **Windows:** Addendum A induction target windows (BI ≥ awake p5 and δ ≥ awake p95), the 9 mice with ≥ 3 target windows, with the second session of the repeat mouse excluded. Reference: the primary awake windows (`claar_sel`).
- **Preparation and surrogates:** the same as `claar_z.py` variant ns, i.e. average reference then normal-scored, with 3 IAAFT draws using the same seeds (crc32(session:index)·10 + d). So the gate and z residuals share surrogates.
- **Unit:** per mouse, Δ = mean gate residual over target induction windows − mean over awake windows.
- **Null:** awake split half (later half − earlier half by time), SD across the 9 mice.
- **Decision (n = 9; 8 of 9 has one-sided p = .0195 in each direction):**
  - **DECOUPLING:** ≥ 8 of 9 negative, mean < −2 null SD, and mouse-bootstrap 95% CI upper < 0.
  - **GAIN:** ≥ 8 of 9 positive, mean > +2 null SD, and CI lower > 0.
  - **NEITHER:** |mean| < 2 null SD and the CI within ±0.05 (a quarter of the calibrated gain size).
  - **Otherwise INCONCLUSIVE.**
- **Real-data sign control:** isoflurane-epoch windows (`claar_sel` state iso, burst suppression, which is read as shared gain) minus awake, per mouse with ≥ 3 iso windows. Predicted positive, i.e. mean > 0 with CI lower > 0. If not, the readout is flagged as not behaving on real data, and the primary verdict is reported but not interpreted.
- **Secondary link:** within each mouse, across induction target windows, Spearman(gate residual, z residual) on the same surrogates. Report the median and the sign count. Prior: no window-level link (macaque, earlier signed readout).

## Disclosed extension: macaque NeuroTycho 10-s windows (partly seen before)

- **Already seen:**
  - Stage A had an earlier raw, signed, 1-s-block gate ("nothing consistent").
  - George propofol strong-pair gate was −6.5 / −7.5 surrogate SD in the hub test (positive pairs, so orientation is moot there).
  - `amp10_nt.npz` was not reopened.
- **Windows and surrogates:** nt10 and nt10b, 32-channel subset S(M), normal-scored, 1 IAAFT draw with the same seed as `zns_*` (row index in each agent file). The z residual is therefore paired.
- **Edges:** the top quartile of |r| over the unit's awake eyes-closed windows.
- **Unit:** Δ = drug (or sleep) − awake eyes-closed. Null: the SD of eyes-open − eyes-closed over 30 units.
- **Reported:** per agent; pooled drug sessions with the same rule as the mouse at n = 23 (k = 17); sleep; and George propofol separately.
- **Link:** within-session Spearman across drug windows.

## Prior [S]

The mouse induction drop deepens with burst index, and the macaque George propofol pairs read as local-event decoupling. I expect DECOUPLING in the macaque, and lean GAIN or INCONCLUSIVE in the mouse induction, since slow waves are largely shared. Low confidence.

## Result (16 November 2026) [V]

Output `amplitude/gate_result.txt`; per-window readouts `amplitude/gate_*.npz`.

**The sign control failed, so by the frozen rule the primary is reported but not interpreted.**
- The isoflurane burst-suppression epoch minus awake gives gate −0.162 [−0.237, −0.096], with 1/17 mice positive. The prediction was positive (shared gain).
- **Primary, mouse induction: INCONCLUSIVE.** 6/9 negative (8 needed), mean −0.081 [−0.159, −0.012], −2.3 null SD (split-half SD 0.035).
- **Macaque, disclosed extension, pooled drug: NEITHER by the rule.** 16/23 negative, mean −0.029 [−0.049, −0.011], −1.2 eyes-open SD (null SD 0.025). The CI excludes 0 but stays within ±0.05.

  | Agent | Δgate | Eyes-open SD | Negative |
  |---|---|---|---|
  | Ketamine | −0.066 | −2.6 | 3/4 |
  | Ketamine–medetomidine | −0.025 | −1.0 | 9/11 |
  | Medetomidine | +0.010 | +0.4 | 1/4 |
  | Propofol | −0.043 | −1.7 | 3/4 |
  | Sleep | −0.005 | −0.2 | 4/7 |

  George propofol −0.098 and −0.073 reproduces the hub-test direction, but that unit was seen before.
- **No window-level link to the z residual:**
  - Macaque: within-session Spearman median +0.05, 12/23 positive.
  - Mouse: median +0.39 over only 4 mice with ≥ 5 target windows.
- imbal and switch: no consistent change.

**Post hoc diagnostics (not preregistered):**
- **The average reference is not the cause.** Synthetic calibration with CAR is unchanged: local −0.33, gain +0.20, linear ≈ 0. On the mouse as-recorded reference, burst suppression is more negative still (−0.458, 0/17 positive; raw gate change −0.43).
- **Within the isoflurane epoch, burstier windows (lower burst index) have more negative gate:** Spearman(BI, gate residual) median +0.62 as recorded, +0.10 with CAR.
- **Correlated noise floor explains the failure (synthetic, `gate_calib_floor.py`).** Shared gain over a noise floor whose common-mode share is 0 / 0.5 / 0.9 gives gate +0.21 / −0.05 / −0.66 as recorded, and +0.21 / +0.15 / +0.03 after the average reference. So gate measures whether loud blocks are less spatially coherent than quiet ones, relative to the floor's coherence. Shared gain over a coherent floor and local events both give it a negative sign. The real CAR value (−0.16) means a floor that CAR does not fully remove (non-uniform volume conduction or reference) or genuinely local bursts; the readout cannot say which.

**Reading [I]:**
- The readout is not diagnostic on real data. It cannot separate local-event decoupling from shared gain without knowing the floor's spatial coherence. This test therefore did not decide the mechanism.
- Under the drugs the gate residual is mostly negative, but by the frozen rule that is not interpreted.
- "Amplitude-linked decoupling" is not established as the form of the anaesthetic z drop. The form stays unresolved.
- **Claims that rested on the same calibration need downgrading:**
  - The propofol form "amplitude-linked decoupling" (`docs/context/premise-and-character-axes.md`).
  - The hub-test reading "not shared bursty gain (gate negative)" in `review-nature-level.md`.
  - "Burst suppression is shared gain" (claar notes, commit 9f5ec744 message). It rested on a synthetic sufficiency result: shared on/off gain lowers z by −0.068. That shows gain *can* produce the drop, not that real bursts are shared gain. The defensible wording is "the drop tracks burst index".
- Do not spend more on amplitude-stratified readouts unless a readout is first validated on a real known case with a measured floor.

Compute: about 15 min local (1389 + 983 mouse and 1260 macaque windows, numpy only).
