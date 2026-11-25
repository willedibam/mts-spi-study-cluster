# Off-band count: linear-Gaussian twin through the pipeline (frozen 26 November 2026)

Addendum to [`offband-prereg.md`](offband-prereg.md), frozen and pushed before any twin count on real windows was computed. Code: `offband_twin.py` (`run` on Gadi, `score` locally), `jobs/offband_twin.pbs`. Result: a section of [`../offband-result.md`](../offband-result.md).

## Question

Eyes-open wake has 7.9 off-band pairs per window above the surrogate-against-surrogate null (step 2). Is that a property of the recordings, or a mismatch between how the pipeline treats a recording and its surrogate? A known linear, stationary, Gaussian input run through the same steps answers it: it has no departure to find, so any excess it shows is the pipeline's.

## Seen before freezing

- Synthetic only. On a near-unit-root 32-channel VAR(1), 300 twins have a mean lag-0 correlation within .027 of the window's and a lag-1 autocorrelation of .993 against .998. The tapered sample covariance makes very persistent channels slightly less persistent in the twin. The padded twin's end jump is 73 one-step increments (median), against .29 for the periodic twin. One `edge_mi` takes about 16 s on the Mac.
- A scorer plumbing test with each recording as its own twin and t1 draws 0–3 as its draws reproduces the step-2 E per state. No twin of a real window has been made.

## Twin

- Per window, start from the pipeline's own recording v = `ns` of the 32-channel subset. Zero-pad v (demeaned) to 2T, multiply each rFFT bin by one complex normal shared by all channels (real normal at DC and Nyquist), invert, and keep the first T samples.
- This is a stationary Gaussian process whose auto- and cross-covariances at lags below T are the window's biased sample covariances. The kept segment is not periodic, so its ends mismatch as a recording's do. This is the artefact Fourier-based surrogates carry from a recording ([Schreiber & Schmitz 2000](https://arxiv.org/abs/chao-dyn/9909037), Physica D 142:346; verified on arXiv).
- The twin enters where the recording does:
  - `ns`, then G and K by `edge_mi`;
  - B = 4 spectrum-final joint IAAFT draws;
  - residuals about each one's own G–K line;
  - L = 95th percentile of |draw-0 residual|.
- Twin count c_t = #{|e_t| > L}. Its null mean m_t is the mean count of draws 1–3 against the same L. E_t = c_t − m_t, averaged over R = 3 twins per window.
- Periodic control: the same with no padding (T-periodic twin), 3 draws. Its surrogate comes from nearly its own ensemble, so it checks the code path, not the question.
- Seeds, with g = the window's index in `offband_null.windows()` order: twin `[g, 31, r]`, its draws `[g, 32, r, b]`, periodic `[g, 33]`, its draws `[g, 34, b]`. All are disjoint from `k` and `[k, b]`.
- All 1,260 windows. End mismatch can grow with autocorrelation, so it could also inflate part of the drug excess.
- Gate, in every window: the recording K and draw 0's K reproduce the stored `zns` K within 1e-4, and each draw's G equals its twin's. If it fails, stop and report.

## Decision (eyes-open wake; E_t over its 300 windows, 95% interval from 2,000 session-bootstrap resamples)

- **Excluded:** upper bound < 2.0 (at most a quarter of +7.9). The wake excess is a property of the recordings, not of the pipeline.
- **Artefact:** point estimate ≥ 4.0 (at least half). The wake excess is mostly the pipeline's, and the text must not call wake a departure from Gaussian dependence.
- **Partial:** otherwise. Report the share E_t / E_rec.
- **Periodic control:** |mean E| over all windows ≤ 1.5 means calibrated. Otherwise it is a declared deviation, and twin E is read against it.

## Secondary (descriptive unless stated)

- Per state: E_rec, E_t (with interval), periodic E, corrected E_rec − E_t, twin count and null, median residual SD ratio (recording against twin).
- On corrected E: the step-2 animal test (one-sided t(3) on drug-pooled animal means of session Δ against eyes-closed wake, α .05) and the 3× ratio rule (drug against wake and sleep).
- Fidelity per state (median per window):
  - Spearman(G twin, G recording) over the 496 pairs;
  - mean |G twin − G rec| / mean G rec;
  - mean Theiler window, twin against recording;
  - end jump, recording against twin.

  A state whose median Spearman is below .9 is flagged as poorly matched.
- End jump = per-channel (v₀ − v_{T−1})² over the mean squared one-step increment, averaged over channels. This is a variant of Schreiber & Schmitz's γ_jump, normalised by the increment rather than the variance because leakage acts at the sampling scale. Report the within-state Spearman with E, for the recording and for the twin.

## What this cannot decide

- **Non-stationarity versus nonlinearity.** A non-excluded wake excess remains a departure from linear-stationary-Gaussian dependence, and this check does not say which. The `nonstat/` test is calibrated to show that it cannot separate gain from a coupling nonlinearity.
- **The twin's covariance is the window's own sample estimate.** It is tapered at long lags, so it is not the true process covariance.
