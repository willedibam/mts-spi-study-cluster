# Stuart–Landau temporal order audit, 2026-09-10

## Outcome and interpretation

Completed the 28-case baseline bank (27 new simulations, one verified existing
archive reused) plus five matched integration/burn sensitivities. The new runs
reproduce the distinction between constant collective amplitude and large
collective oscillations, and the finite-size shift of their boundary. They do
not add a new SPI–SPI confirmation or diagnose chaos from spectral appearance.

The physical object is the complex collective order Z(t)=mean_i w_i(t), with
amplitude R(t)=|Z(t)|. A time average of R compresses temporal behavior; it is
not interchangeable with the order-parameter trajectory. This is central to
the Matthews–Strogatz model, not an invented alternative order parameter.
Reference: https://doi.org/10.1103/PhysRevLett.65.1701 (1990).

## Protocol

K=.8, frequencies on a uniform grid carrier+[-gamma,gamma], carrier2, RK4 dt=.02;
burn200 time units, 8000 observations at sample_dt=.1 (800 further time units).
N=32 or800; gamma=.70,.74,.75,.80,.90,1.0,1.2; seeds910001/910002.
Seeds change initial states, frequency permutations, and sensors; they do NOT
produce independent randomly drawn frequency populations. N32 observes all
oscillators, N800 observes a fixed random subset32. Store full complex Z,
activity, complex small observations, frequency values and code hash.

The observation frame matters: lab-frame real channels have a carrier waveform
even when R is constant, whereas rotating-frame locked channels are nearly
constant. Neither representation is silently substituted for the other.
Very small rotating-frame residual variation is not evidence of collective
oscillation. Spectral entropy is left undefined when SD(R)<=1e-8 rather than
normalizing negligible residual power into an apparently complex spectrum.

## Evidence

| Control | N32 | N800 | Interpretation |
|---|---|---|---|
| gamma=.70 | MeanR=.5952; SD near machine precision | MeanR=.6471; SD near machine precision | Locked collective amplitude |
| gamma=.74 | MeanR≈.3044; SD≈.1245 | MeanR=.5206; SD≈1e-9 | Same control lies on different sides of finite-size onset |
| gamma=.75 | MeanR≈.2915; SD≈.1326 | MeanR≈.3778; SD≈.1130 | Collective amplitude fluctuates strongly at both sizes |
| gamma=.80 | MeanR≈.2213; SD≈.1580 | MeanR≈.2889; SD≈.1831 | Large oscillations; mean alone discards their dynamics |

These are descriptive two-start summaries, not confidence intervals or a new
precise critical-point estimate. Higher-gamma traces have more complex temporal
structure, but neither their appearance nor spectral entropy proves chaos.

Matched seed910001 timestep checks (.02→.01): at N32/gamma.8, meanR changes
by -1.02e-7 and SD by -3.82e-8; at N800/gamma.75 the changes are +3.08e-9 and
-1.81e-9. This supports integration accuracy for these observables/cells only.

Increasing burn200→1000 changes meanR by -.00223 at N32/gamma.8 and -.00106
at N800/gamma.75; SD changes are +.000344 and -.000694. The original block
variation should not be attributed to timestep error, nor should every change
of finite-window mean be called nonstationarity: oscillation phase matters.

At N800/gamma1.2/seed910002, however, meanR falls .0441→.0210 and SD falls
.0360→.0136 with the longer burn. Do not treat this short-burn endpoint as a
settled incoherent-state reference. Existing benchmark claims have their own
protocols; this audit does not retroactively replace their gates or estimates.

## Artifacts and next use

- Generator: scripts/scout_stuart_landau_streaming.py; baseline runner:
  scripts/run_stuart_landau_temporal_scout.py. The runner verifies parameters,
  source hash and global-trace integrity before reusing archives.
- Analysis: scripts/analyze_stuart_landau_temporal_scout.py, with optional
  --sensitivity-dir. Two unit tests check carrier invariance of amplitude and
  recovery of a known modulation frequency; no chaos classification exists.
- data/order_parameter/stuart_landau_dynamics_260910/ contains baseline NPZs;
  convergence/ contains the five matched follow-ups.
- analysis/temporal_order.csv and integration_sensitivity.csv preserve numbers.
- [Order trajectories](../../../data/order_parameter/stuart_landau_dynamics_260910/analysis/order_traces.png)
- [Mean, variability and descriptive spectra](../../../data/order_parameter/stuart_landau_dynamics_260910/analysis/temporal_order.png)

Use the existing fine locking-boundary benchmark as the clean claim-bearing
example. Preserve R variability/trajectory in its explanation. Further broad-
regime exploration should first settle the gamma1.2 transient, not expand N
or fit a descriptor to the current unsettled endpoint. A common representation
need not outperform the purpose-built order statistics to demonstrate tracking.
