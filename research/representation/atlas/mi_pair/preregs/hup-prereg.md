# Seizure-onset-zone edges against matched non-onset edges: HUP iEEG (OpenNeuro ds004100)

Frozen 24 November 2026, before any MI was computed on HUP data. Inspected before freezing: public metadata, `participants.tsv`, every interictal `channels.tsv` and `ieeg.json`, and every `electrodes.tsv`, and one outcome-blind feasibility check of the matching rule on the extracted windows of HUP060 and HUP065 (coordinates and |r| only; no kNN MI). That check showed SOZ sets are spread out (median spacing about 30 mm) and weakly correlated (median |r| .08 and .29), so the first draft's compact-cluster proposal could not match them; the proposal below replaced it (audit in [`../../../../../docs/dataset-candidates-character.md`](../../../../../docs/dataset-candidates-character.md)). Code: [`../hup/`](../hup/). Changes after freezing go under "Deviations", dated, with the reason.

## Claim under test

In macaque, mouse and human anaesthesia, z(Gaussian MI, kNN MI) beyond a window-matched joint IAAFT surrogate falls: model-free dependence beyond the linear-Gaussian expectation becomes more uneven across pairs (`../../findings.md`, "Gaussian against model-free MI"). Signals recorded inside the epileptogenic zone show more nonlinear interdependence beyond surrogates than signals outside it (Andrzejak, Schindler & Rummel 2012, PRE 86:046206; abstract checked, full text not re-read). This test asks whether the same SPI–SPI reading separates edges *within* the clinically marked seizure-onset zone (SOZ) from edges within matched non-SOZ contact sets of the same patient and run. It is a spatial pathology contrast on the same axis as anaesthesia, not a state change.

Predictions, fixed now:
- **Primary (SPI–SPI):** the SOZ set's residual z (real − surrogate) is lower than that of its matched non-SOZ sets (D_z < 0). The sign is a hypothesis carried over from anaesthesia; Andrzejak's prior is about level, not about across-edge agreement, so it does not fix this sign.
- **Secondary, decided (level, the direct test of the prior):** the SOZ set's mean excess of kNN over Gaussian MI beyond the surrogate is higher than that of its matched sets (D_lev > 0).

## Data and windows

- Subjects: interictal runs (`task-interictal`, 2 per subject, 300 s, 256–1024 Hz). Channels: `status` good in every interictal run of the subject. SOZ contacts: `status_description` containing `soz`; resected non-SOZ contacts: containing `resect` and not `soz`.
- Chain per run (atlas long arm, `prepare.continuous`): notch 60 Hz, 0.5-Hz high-pass, 40-Hz low-pass, resample to 100 Hz. Atlas bad-channel screen (`prepare.bad_channels`) on both runs concatenated.
- Reference: common average over the screened channels, then drop one channel for rank: the last screened channel in channel order that is neither SOZ nor resected.
- Windows: non-overlapping 10 s (T = 1000) from 10 s after run start to 10 s before run end (28 per run), each linearly detrended. Exclusion: atlas artefact rule (`window_ok`) with the run's own robust scale after CAR; any flat run ≥ 1 s at source rate. No cap: every passing window is used.
- Per window: normal-score each channel; Gaussian MI −½ log(1 − r²) and KSG MI (k = 4, AUTO Theiler window) for every pair of the analysed channels (`edges.edge_mi`). Three spectrum-final joint IAAFT draws over all analysed channels, seed `crc32("<subject>:<window>") × 10 + draw`; per-edge Gaussian and KSG MI on each draw. Computed once; sets below are edge selections.

## Sets

- **SOZ set S:** every screened SOZ contact with coordinates. Size n_S.
- **Pool:** screened contacts that are neither SOZ nor resected and have coordinates (`electrodes.tsv`, fsaverage space).
- **Matched non-SOZ sets N_j:** each candidate draws k uniformly from {1, 2, 3, 5, 8, all}, starts from a random pool contact and repeatedly adds one contact chosen uniformly among the k pool contacts nearest (minimum Euclidean distance to any member) to the current set, until n_S contacts; proposals therefore range from compact clusters to uniform random sets. A candidate is accepted if (a) its median pairwise distance is within ±25% of S's, (b) the median over its edges of window-mean |r| (real data, Gaussian side only) is within ±.05 of S's, and (c) the interquartile range of the same is within ±.05 of S's. Duplicates are discarded. Up to 20,000 candidates; stop at 200 accepted. If fewer than 20 are accepted, the search is repeated once with ±40% and ±.08; the level used is reported per subject. RNG seed `crc32(subject)` (+1 for the resected-pool variant). No kNN MI enters the matching. Feasibility check: 199 and 200 sets at the first level for HUP060 and HUP065.
- **Inclusion:** n_S ≥ 8 (≥ 28 edges) and ≥ 20 accepted matched sets. Metadata predict 31 patients at n_S ≥ 8 before screening.

## Statistics

- Per set A and window w: z_real = corr over A's edges of (Gaussian MI, KSG MI); z_surr = mean over the 3 draws of the same on each draw; res = z_real − z_surr. Level: lev = mean over A's edges of (KSG − Gaussian), real, minus the 3-draw mean of the same on surrogates. Set value: mean over windows (both runs pooled).
- Per subject: D_z = res(S) − mean_j res(N_j); D_lev likewise.
- **Null scale (no-change baseline):** the same statistic with the first accepted matched set N_1 standing in for S, against N_2 … N_J. Null SD = SD across subjects of that pseudo-D. Effects are reported in null SDs.

## Decision rule

- **Primary, D_z:** SUPPORTED if mean D_z < 0, one-sided t-test on subject D_z p < .05 (less), and one-sided sign test p < .05 (fewer positives). OPPOSITE if mean D_z > 0 and one-sided t (greater) p < .05. Otherwise NOT DETECTED, reported as mean D_z in null SDs with its 95% interval.
- **Secondary, D_lev:** the mirror rule with prediction D_lev > 0 (SUPPORTED: mean > 0, t greater p < .05, sign test p < .05; OPPOSITE: mean < 0, t less p < .05).
- A SUPPORTED primary is read as "edges inside the clinical SOZ show model-free dependence beyond the Gaussian expectation more unevenly across pairs than matched edges elsewhere"; any reading as epileptogenic mechanism, rather than the clinical label, needs the confound checks below.

## Pre-stated confounds and descriptive checks (all reported, none decided)

- **Interictal spikes.** SOZ contacts carry more epileptiform transients, which are joint bursts by construction. Check: D_z and D_lev using, per set, only windows in which no set channel exceeds 5 robust SDs (run scale); sets with < 10 such windows are dropped. A result carried by spikes is a statement about joint bursts, not coupling form outside bursts.
- **Run replication:** D_z and D_lev per run; sign agreement and Spearman across subjects between runs.
- **Floor:** n_S ≥ 6 (metadata predict 40 patients).
- **Pool:** resected non-SOZ contacts admitted to the pool.
- **Surrogate-free:** D on z_real alone, to show how much the surrogate removes.
- **Strata:** ECoG against SEEG; Engel class 1 against the rest.
- **Not addressed:** the SOZ label is clinical and imperfect; matched sets lie in other brain regions with other cytoarchitecture and spectra (the surrogate removes linear-spectral differences, not regional nonlinear ones); fsaverage coordinates distort individual distances.

## Cost

Expected 0.6–1.3 KSU (about 300,000 edges over 47 patients with ≥ 4 SOZ contacts, 56 windows, 4 MI evaluations each, 16–35 ms per evaluation); worst case 2.6 KSU.

## Run

Gadi, `jobs/hup.pbs`, ten array indices over `hup/groups.txt` (47 patients with ≥ 4 SOZ contacts in metadata, balanced by edge count), from the source snapshot `operations/sources/hup-b398da13` (job 185084113[], submitted 24 November; logs in scratch `operations/logs/atlas-mi-pair-hup-261124/`; windows in scratch `representation/atlas-mi-pair-hup-261124/windows/`); then `score.py sets` and `score.py summary`. Per-edge arrays on gdata `representation/atlas-mi-pair-261117/hup/edges/`.

## Deviations

None yet.
