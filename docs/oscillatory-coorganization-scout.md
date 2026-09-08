# Oscillatory co-organization: bounded exploratory scout

Decision2026-09-09, after negative covariance-modulation utility results and the
user's clarification: test organization of two dependence modes, not another
scalar strength parameter. Existing pointwise neural diagnostic continues.

Question: can a fixed SPI catalogue represent whether phase-coherent groups and
groups with cofluctuating amplitude envelopes coincide, across observation size
and recording nuisances? This is a common-drive model of functional organization,
not a biophysical model of anesthesia, direct causal coupling, or a clinical task.
The binary label is defined from latent group assignment, never from estimated z.

Motivation: phase and envelope coupling are distinct intrinsic coupling modes.
Rokos et al2021 discuss this distinction and study phase-based network changes
and cognitive recovery after anesthesia
(https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.706693/full).
That paper does not establish the clinical utility of their cross-network
agreement or of SPI-SPI. It motivates the question, not a promised application.
In particular, z cannot retain anatomical localization after edge pooling.

## Frozen first scout

- N32, four groups of eight channels for each mode. Coincident condition uses the
  same partition. Crossed condition has two channels in every phase/envelope
  group intersection. Each mode separately has identical group-size distribution
  and an isomorphic population co-membership graph. Their correspondence changes.
- Channel i emits exp(log-envelope)*cos(phase)+measurement noise. Phase-group
  drivers are independent noisy oscillators with a shared carrier frequency;
  envelope-group drivers are independent stationary Gaussian AR1 processes.
  Independent local phase/envelope fluctuations, offsets, gains and sensor noise
  are shared in distribution between conditions. Channels are randomly ordered.
- Each single-channel process law is identical across conditions conditional on
  nuisance parameters, because the two driver families are independent and their
  groups exchangeable. This is a distributional claim, not exact realized power
  spectra or histograms. Other individual-SPI distributions may change because
  they mix both modes; empirical rich-marginal controls remain necessary.
- 24 paired nuisance/innovation seeds, two conditions, sourceM16T1000 and nested
  M8T500:48independent-condition recordings/96views in24paired blocks. Do not
  count paired conditions or nested views as independent experimental blocks.
  This is a feasibility scout, not a learning curve or confirmatory experiment.
- Fs100Hz, carrier uniform6–12Hz; phase innovationSD .10–.18rad/sample;
  log-envelopeSD .25–.55, AR coefficient .94–.98; shared-envelope variance
  fraction .8; local phaseSD .15–.35; additive noiseSD .05–.15; per-channel
  gains exp(U[-.3,.3]). Masterseed260909121. No parameter search in this scout.
- Observed-data controls: fixed3–20Hz bandpass+Hilbert transform; phase-locking
  magnitude, log-envelope correlation, their individual edge summaries, and
  Pearson/Spearman agreement. Retain pooled autospectra and raw moment controls.
  No true phase, envelope or group assignments enter these estimates.
- Raw gate: observed phase/envelope agreement must distinguish conditions at
  both observation sizes (descriptive AUROC>=.8 in the expected direction), and
  all views must be finite. These thresholds check observability, not z success.
  Inspect single-mode summaries and spectra; do not alter generator to erase
  legitimate competitors if they also expose the target.
- If raw gate passes, use p90 on these96views with existing Gadi smoke/node-gate
  workflow. Characterize catalogue signal versus direct controls before deciding
  on a fresh independent learning pilot. No extensive hyperparameter grid and
  no full representation-learning claim from this scout. Any new pilot must
  freeze its split/readouts before new data and keep this scout separate.

A positive finding would show a principled applicability regime of agreement
features. A negative finding would separate lack of relevant catalogue coverage,
estimation problems, and normalization/readout limitations where possible.
The clinical significance of the exact binary organization remains unestablished;
do not present a manufactured partition classification as anesthesia prediction.
