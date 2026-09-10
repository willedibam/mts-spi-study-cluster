# Vicsek narrow physics refinement (frozen before simulation)

## Question

Does the strong polarization contrast seen between eta=.44 and .476 resolve
into a reproducible local transition curve, and how much start/seed/window
dependence remains? This is physics-only exploration, not SPI confirmation or
a proof of a thermodynamic discontinuity. Keep prior negative/ambiguous results.

## Design

Use configs/scout/vicsek-narrow-physics-260910.yaml: N32768, angular noise,
density2, speed.5, burn20000 steps, record100000 steps, store every25th step.
Controls .460,.468,.476,.484; two new seeds910204/910205; ordered/random
starts for each seed/control (16 cases). No control/start chosen from SPI output.
Seed is a blocking factor across controls and starts, not 16 independent seeds.
The .476 point repeats the previous long gate with new seeds. The wider .44
anchor remains available from that gate, not silently relabeled as new evidence.

The completed long gate had half-record mean differences .0049/.0060 at eta.44
but .0249/.0139 at .476; start differences were .00011 and .01270 respectively.
This motivates independent seeds and a narrow sweep, not a claim that all
near-boundary windows are stationary. Hold N/time fixed to isolate this question.

## Analysis and decision

Retain every case. Report seed/start-resolved curves, eight consecutive block
means, half-record shifts, distributions and Binder values. Do not declare a
first-order transition merely from one steep interval or a negative Binder.
Do not call a failure to observe a jump evidence against the published model.
No seed-level uncertainty claim from treating time blocks or nested views as
independent masters. Finite-window Q remains meaningful, but a stationary-Q
interpretation needs the stated uncertainty/start dependence to support it.

Use the fixed x/y particle and field views already recorded; compare M8/16/32
and T100/500/1000/2000 with the known stride25 (different physical durations
from old stride1 records). Do not choose a view by its agreement with Q. This
does not freeze a future p90 sampling rate or authorize assuming SPI validity
from channel variance alone. No p90 farm in this stage.

## Bounded execution

The unchanged numerical generator and module-mode launcher passed Gadi tests,
two smoke cases and the four-case long run. Increase only the single-node
case-count guard from4 to16, one process per requested CPU. Request16 CPUs,
64GB,2h: maximum64 SU, expected about29 SU from the prior53m39s run.
Together with7.21 SU spent, the ceiling is71.21 SU, below the unchanged1000-SU
overnight cap. No submissions after2026-09-10 04:51 UTC. Collect jobs already
running after the cutoff; do not automatically retry a timeout after it.

Pin a source commit and keep each job's snapshot immutable. Do not publish
unrelated local commits merely to deploy these three benchmark files.
