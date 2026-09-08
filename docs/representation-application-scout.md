# Application scout: baseline-relative structural change

2026-09-08. Decision: pursue population-based structural monitoring as a candidate
application, but do not launch a new p90 farm yet. The completed interaction-share
results justify investigating dependence sensitivity, not assuming useful damage
information or superiority to spectral methods. This scout establishes a real
problem and audits data eligibility; it establishes no SPI–SPI performance gain.

## The question and why it matters

**Can cross-statistic agreement improve transfer of baseline-relative structural
change detection between related structures with different sensor coverage,
without target damage labels, beyond calibrated modal and spectral features?**

The practical need is to share scarce damage examples across structures while
separating structural changes from environmental variation. This is an existing
research problem, not a motivation invented for z. A published laboratory study
already uses domain adaptation of modal features for cross-configuration transfer.
[Giglioni et al., MSSP 2025](https://doi.org/10.1016/j.ymssp.2024.112151).

The plausible contribution is a useful descriptor requiring less manual matching
of dependence features across sensing configurations. That is a hypothesis.
Natural frequencies already provide fixed-dimensional features from different
numbers of sensors, so size compatibility is not novel. Sensor-aware models must
retain available coordinates; stripping them to favour z would be unfair.
Because z discards incidence, the first target should be global change detection,
not damage localization or identification of which structural component failed.

## What was actually inspected

The public [Sheffield version-1 dataset](https://doi.org/10.15131/shef.data.27732792.v1)
provides multichannel acceleration records and condition metadata. I downloaded
the complete B3 file (358 MB, verified against its published MD5), inspected all
four recording indices using bounded byte-range downloads, and ran a raw spectral
window check. I did not run pyspi or train a damage classifier.

These are direct observations of the exported files, not a reproduction of the
paper's selected analysis subset:

| Configuration | Exported recordings | Acquisition dates | Audit concern |
|---|---:|---:|---|
| B1 | 148 | 4 | Additional scour and normal records outside the paper's main table |
| B2 | 337 | 5 | Temperature coverage differs substantially by condition |
| B3 | 70 | 1 | 20 M2 records rather than the paper's selected 10; few condition blocks |
| B4 | 440 | 5 | Normal tags <=28°C; M1–M4 tags >=29°C |

In B4, the rule `filename temperature > 28` separates all 201 normal and 41
M1–M4 recordings. This is a **descriptive confound check**, not a held-out model
score. The tags must be checked against measured temperature when full B4 data
are retrieved. In B3, the tags match the measured temperature metadata exactly.
Temperature-only discrimination would not prove an acceleration classifier uses
temperature, but it prevents attributing an unqualified score to damage.
Do not silently pool additional conditions, normalize label spellings, or treat
S0, waterlogging, and added masses as interchangeable physical damage states.

B3 contains 25 arrays: A1–A22 and X/Y/Z. The scout uses the 20 deck channels
A1–A20. The stored arrays have 24,600 samples per record, with 256 Hz spacing
verified from the time vectors: about 96 seconds, not the paper's approximate
20-second acquisition description. The time vectors have 24,636–24,648 entries;
only their verified shared prefix is used. The README's `Damage` key is actually
`damage`; `scale` contains per-record time vectors, not scalar elapsed times.
These export differences require explicit handling, not an assumption that the
data are unusable or that the original study was incorrect.

A reference-anchored spectral peak check on B3 found substantial disagreement
between short central windows and full-record estimates, especially for the
higher reference bands. Increasing the window from 4 to 16 seconds did not fix
all peak switches. This elementary peak picker is **not** an operational modal
identification algorithm; its instability is not evidence against competent
modal baselines or in favour of z. It rules out adopting the old synthetic
T=1000 convention without a physical sampling/window check.

Repeated recordings are observations under a small number of imposed physical
conditions. Acquisition blocks and structural configurations must be kept
together when their reuse would leak the target deployment setting. Counts of
windows do not establish counts of independently labelled structures or damage
interventions. This dataset can support a limited laboratory transfer study;
it is not a large independent-system sample-efficiency benchmark.

## A better matched candidate, with an access limitation

[Morleo et al., EWSHM 2026](https://doi.org/10.58286/33886)
crosses baseline and incremental cut states with controlled temperature levels.
The paper describes 45 six-minute histories, with acceleration sampled at
2048 Hz, from three baseline configurations built using two physical designs.
Its third baseline is an already-damaged version of the second design, so its
labels describe further change from that baseline. Treat related configurations
as sharing a physical origin. Ten derived segments per history are not ten
independent experiments. The paper's data-availability paragraph says public
ORDA release is planned; meanwhile access is by contacting the authors.
[Full paper](https://www.ndt.net/article/ewshm2026/papers/EWSHM_2026_1655.pdf).

The publisher page, full paper, Crossref record and Figshare search were checked.
No public raw archive was verified in this scout. A request draft is in
[bridge-data-access-request.md](bridge-data-access-request.md); no message has
been sent. This candidate improves experimental control, but its small physical
population still limits claims about generalization to new real bridges.

## One conditional pilot, not a benchmark grid

1. **Eligibility first.** Confirm raw synchronized response channels, sampling
   conventions, physical-specimen identifiers, damage-application blocks and
   temperature coverage within each condition. Use a baseline-relative label.
   If condition and environment cannot be separated, do not describe the task
   as robust damage inference. No new simulator is needed for this gate.
2. **One transfer question.** Hold out a physical design; keep its related
   configurations and all segments out of source training/tuning. Allow a
   separately designated target baseline calibration set equally to every method,
   with its records and environmental coverage counted explicitly. Target change
   labels are evaluation-only. With only two designs, this is an exploratory
   two-direction transfer demonstration, not population-level confirmation.
3. **Compact comparisons.** Use z, rich SPI summaries, and their concatenation
   with matched regularized heads. Mandatory raw comparators are spectral
   summaries, appropriately estimated modal features, and selected cross-spectral
   or transmissibility features; include baseline normalization and measured
   temperature where available. Add the existing aligned temporal/channel
   encoder after this input/split gate, without using a weak peak picker as the
   specialist comparator. The previous neural results do not validate its
   optimization on a new physical signal distribution.
4. **One observation stress.** After the full-coverage comparison, use one
   prespecified spatially balanced sensor reduction. Select channels without
   change labels; all methods receive the same recordings. Retain the sampling
   rate needed for the physical modes. Choose duration from baseline-only
   estimator reliability, and count multiple windows as one parent history.
5. **Decision.** Report both directions, each temperature and each physical
   change condition, including negatives. Quantify the incremental value of z
   beyond the strongest comparator and calibration exposure. Do not estimate
   an independent-structure learning curve or precise rare-false-alarm rate from
   a handful of specimens. Expand only if the pilot shows a useful effect that
   survives these controls and a separate physical replication is obtainable.

These steps define the comparison and its failure conditions. Exact sample counts
and windows remain contingent on the data audit; this is not a frozen predictive
benchmark or a claim that z will win. A domain-specific contribution may be
valuable without broad neural superiority, but neither a top ML venue nor a
high-impact application is established by the present evidence.

## Other routes screened

- [DOSE-I](https://zenodo.org/records/18483292) has direct behavioural sedation
  annotations but only two EEG channels. Primary z is degenerate for symmetric
  SPIs at M=2. Adding heterogeneous physiological channels merely to increase M
  would introduce a different multimodal problem, not repair that EEG test.
- The public [GABA-anesthesia resource](https://physionet.org/content/eeg-gaba-anesthesia/1.0.0/)
  exposes illustrative derived traces/spectrograms from a very small set of cases;
  it is not the required raw multichannel, independently labelled cohort.
- [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) is a genuine accessible
  seizure benchmark, but seizure waveform/spectral information and established
  channel-flexible models make it a less targeted next mechanism test. This does
  not establish that z cannot help there.
- Motor-imagery left/right labels depend materially on spatial information that
  z discards. Existing [MOABB comparisons](https://arxiv.org/abs/2404.15319) also
  require strong covariance/Riemannian baselines; it is not a clean default pivot.
- [BIOT](https://arxiv.org/abs/2305.10351) and
  [REVE](https://arxiv.org/abs/2510.21585) explicitly address heterogeneous EEG
  formats. They reinforce that variable-channel compatibility is not sufficient
  novelty. Their published performance is not a measured comparator result here.

## Reproduction and evidence

```bash
.venv/bin/python -m scripts.fetch_bridge_scout
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m scripts.audit_bridge_transfer_data
.venv/bin/python -m pytest -q tests/test_bridge_scout.py
```

Data root: `data/representation_application_scout_260908/`. Results:
`results/representation_application_scout_260908/bridge-audit.json` and
`B3-spectral-scout.npz`; `temperature-support.png` and SVG visualize the recorded temperature tags. The audit includes label/temperature counts, acquisition
runs, file hashes, the B3 time-axis check and window sensitivity. Tail indices
were verified against the complete B3 file. Restricted NumPy deserialization and
non-executing index parsing are covered by two tests. No Gadi jobs were submitted.
