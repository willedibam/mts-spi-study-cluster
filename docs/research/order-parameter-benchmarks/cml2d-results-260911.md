# Collective period-doubling CML: execution and results

## Current status

The pilot and subsequent **independent confirmation are both complete**. See the [final confirmation report](cml2d-confirmation-results-260911.md) for the544 fresh-seed/control simulations and1408-view results. The sections below retain the pilot and execution history; earlier queued/running notes are historical.

Physics reproduction and eight numerical/export/target-blind-fit tests pass. Complete:72 primary simulations and688 paired benchmark observations have full-p90 outputs, plus the two preliminary runtime-smoke records. The frozen primary and dispersed M,T arms pass their prespecified gates; the contiguous transfer arm fails. Section7 of the comparison notebook presents both outcomes. All jobs, the short-record retry and durable archive are finished. Total NCI charge319.87SU, below the500SU ceiling. No M64 extraction or new recurring automation was performed.

## Completed physics

All42 size-gate archives and12 convergence archives pass source/config hash, finite-array and shape checks. Eight local regression tests pass, including the independent stencil oracle, affine conjugacy of the full 2D update, all-step sensor alignment, nested export/hash checks, and fit invariance to changes in physical targets and evaluation features.

| L | Physical N | Mean Q at r3.84 | Mean Q at r3.89 |
|---|---:|---:|---:|
| 64 | 4096 | .33308 | .05274 |
| 128 | 16384 | .33351 | .02844 |
| 256 | 65536 | .33374 | .01411 |
| 512 | 262144 | not run | .00704 (one seed) |

The first three rows average two seed-labelled runs, with20k burn and42k record; the L512 row is a separate one-seed size anchor at the same horizon. Q is computed over40k future steps after2k observed steps. These are finite-system order curves, not independent proof of a thermodynamic critical point. The42 local cases used164.54 summed simulation seconds (not cluster SU).

At L256 after200k burn and402k record, random/preordered starts give:

| r | Random Q | Preordered Q | Absolute start difference |
|---|---:|---:|---:|
| 3.86212 | .22025 | .20478 | .01547 |
| 3.866 | .05973 | .06353 | .00380 |

Time-block uncertainty persists: the largest first/second-half difference for these four runs is.03062, and largest descriptive block-mean SE is.00980. Agreement of means does not establish mixing. This motivates a longer scalar reference, not declaring the quoted asymptotic boundary numerically exact. Local and Gadi runs implement the same equation/source; chaotic trajectories are not assumed to be bitwise continuations across architectures.

At L256/M32/T1000, the mean sampled-order estimate at r3.89 is about.261 for contiguous patches and.0654 for dispersed sites, versus.0111 from the full lattice on those input windows. Local paired-step amplitudes remain large while spatial averaging cancels much of them. This supports the prespecified dispersed primary observation; it does not establish a universally optimal layout or justify choosing patches by eventual q–Q correlation. Every tested M8/16/32/64,T100/500/1000 raw view has channel SD>1e-8. This is not a full-p90 validity guarantee. No M64 extraction is currently justified.

For a uniformly sampled fixed set of M sites, define the local paired-step difference d_i=x_i(2t+1)-x_i(2t). Conditional on the full field, ordinary sampling without replacement gives Var(sample mean d - full mean d)= (1-M/N)S_d^2/M, where S_d^2 uses denominator N-1. Thus increasing N at fixed M does not eliminate observation error. Jensen's inequality also gives E_sensor[Q_M]>=Q_N on a matched window: the sampled absolute-order statistic has an upward sampling bias. This first-principles statement concerns Q_M, not z; it neither guarantees nor rules out order information in SPI–SPI. Contiguous patches have a different covariance-dependent sampling error.

The primary held-seed raw audit (36 records, T1000) gives mean-absolute-input- correlation versus futureQ rho=.9459/.9784/.9689/.9609 for dispersed M8/16/32/64. Sampled-Q rho=.8517/.8669/.8680/.8958 respectively. Thus the observations clearly contain across-control information already at standard M; there is no demonstrated need for M64 p90. This is not a proof that its z would be identical or that M16 is statistically optimal. Contiguous M32 is not uninformative: the same two baseline rhos are.8046/.7879, despite its biased absolute level. A contiguous SPI sensitivity must therefore be measured, not declared a failure from the sampled-Q bias alone.

## Chosen primary pilot

L256, nine controls, eight fresh seeds (four development/four evaluation), 200k burn,2k retained observation and one-million-step disjoint scalar reference. Long reference Q and matched-window Q are reported separately. All-step float64 observations; initial p90 only M32/T1000 dispersed. Frozen fit and eligibility gates are in the [protocol](cml2d-protocol-260911.md). This is an exploratory held-seed pilot, not independent confirmation.

L512 lowers the disordered finite-size background but does not solve critical relaxation or small-M sampling error. The choice of L256 is a precision/cost decision, not a claim that it reproduces the exact infinite-lattice curve.

## Jobs and provenance

| Job | Scope | Status/accounting |
|---|---|---|
| 178709040 | 12 convergence/large-L physics cases | Complete, exit0;4m31s wall;1.81 SU;5.86GB peak |
| 178709102 | Two endpoint p90 smoke records | Complete, exit0;24m08s wall;3.22 SU;7.69GB peak |
| 178709304 | 72 primary physical masters | Complete, exit0;16m46s wall;26.83 SU;22.96GB peak |
| 178709341 | Export primary observations | Complete, exit0;9s wall |
| 178709795 | Audit primary physics | Complete, exit0;10s wall |
| 178709971 | Primary full-p90,72 records | Complete, exit0;26m46s wall;128.48 SU;132.81GB peak |
| 178710032 | Frozen primary analysis | Complete, exit0;16s wall |

Smoke runtimes:1436.0s at r3.84 and1221.9s at r3.89, M32/T1000, full289 catalogue both records. Nine/eight reported errors concern group-delay or automatic-order spectral-Granger measures. Unified z finite fractions are .86616/.82788 before development hygiene. No family is manually excluded, and no coordinate is fit to these two records. These are sufficient pipeline and runtime checks to proceed to the frozen primary validity/geometry gates, not a claim that every estimator is numerically valid.

The primary archive and all72 M32/T1000 records are validated locally against the manifest, scientific source hash, shape and finite/nonconstant-channel checks. Full-lattice mean Q across eight seeds at the nine r controls is .333768,.310432,.275083,.196179,.116860,.063955,.035896,.025122,.014077. At r3.86212 and3.864, maximum half-mean differences remain.05152/.04982; maximum block SE is.01905/.01604. Longer records improve the reference but do not prove equilibration. These finite-run targets remain unchanged. Nevertheless, the eight-seed Q ranges do not overlap between any adjacent controls on this grid. The across-control physical ordering is therefore clear in these runs; the temporal caveat concerns equilibrium precision, not an ambiguous direction of the observed transition. Stored `binder` uses1-<m^4>/(3<m^2>^2), equivalent to -U_paper/3; it is not used to fit q or estimate a new critical boundary.

Primary corpus commit62b73e789d322dd92a1ac2cf7b95fba98ead4abb; observation SHA256 `84445f8908a209cddac932f94b1d557842c8d3ea016282fa342e01472f5653dd`. Frozen analysis/target-blind regression test commit169e24e. Pyspi fork remains 65317c9c1fd5f12358b8ede09b7576ef001a76dd. Remote worktree now62b73e7; do not update it during extraction. Completed cluster cost~31.9SU; primary completed cost is128.48SU, leaving total completed cost~160.4SU.

## Primary SPI–SPI result

All289 p90 SPIs were attempted. Development-only hygiene retains19,444 of 41,616 unified_ordered_v3 pairs, spanning253 SPIs. PC1 explains.802723 of development variance; PC2 explains.068053. Leave-development-seed-out loading cosines are .998985–.999457. Both frozen geometry and row gates pass; no rows are excluded and selected-feature missingness is zero in all72 records.

On the36 held-seed M32/T1000 dispersed records:

| Quantity | Spearman rho |
|---|---:|
| Frozen q vs disjoint million-step Q | .898327 |
| Frozen q vs matched-window Q | .907851 |
| Control-mean q vs control-mean Q | .983333 |
| Within-control q vs future Q | .056113 |
| Raw mean absolute correlation vs future Q | .968855 |
| Sampled Q_M vs future Q | .867954 |

Both q and Q have their steepest sampled derivative on r=3.86212–3.864. This is agreement on the finite-run curve, not a new thermodynamic critical- point estimate. The result supports an unsupervised coordinate tracking the changing physical order across the control sweep, not recovery of spontaneous within-control fluctuations or numerical Q calibration. The specialist raw correlation baseline is stronger. This is exploratory held-seed evidence, not independently confirmed discovery of the order parameter's formula.

Primary results justify the prespecified nested M,T and bounded layout arms. The primary transform remains frozen; no feature, geometry, control or sensor selection is changed in response to these scores.

Seed-cluster bootstrap (2000 draws, seed260911, four held clusters) gives 95% intervals [.8516,.9546] for future-Q overall rho and [-.5439,.5715] for its within-control rho. A secondary same-window check is materially different: within-control rho(q,Q_window)=.5542, interval[.0328,.7253]; pooled same-window rho has interval[.8306,.9587]. This indicates moderate realization/window-level association after removing r, not demonstrated tracking along a time series. It should not be conflated with prediction of the much longer future mean. These intervals are conditional on this grid and frozen coordinate and are limited by only four independent held seed clusters.

All72 primary MPI archives are now local. Independent re-fitting of feature hygiene and PCA reproduces all72 scores to maximum absolute error7.8e-15, with the same feature mask and component cosine~1. Core runner, compute, feature-contract and analysis dependencies match the pinned branch; the pyspi checkout remained clean at65317c9 throughout primary extraction.

Primary per-record runtime min/median/p90/max is1384.25/1456.85/1540.58/1577.60s. Reported estimator-error counts are9 for62 records,8 for5 records and3 for5 records; every record still contains the full289-SPI attempted catalogue.

## Paired follow-up execution

Export178710557 completed, exit0,35s. Source/config commit 9ed8aa48f70cbd78d94c713375a16a4543d92c74 now pins the remote worktree; do not change it during extraction/analysis. The three committed external configs bind the generated archives by SHA256. No new master simulations are run.

| Job | Additional arm | Rows | Requested walltime |
|---|---|---:|---:|
| 178710730 | dispersed M8,T500 | 72 | 8min |
| 178710731 | dispersed M8,T1000 | 72 | 10min |
| 178710732 | dispersed M16,T500 | 72 | 12min |
| 178710733 | dispersed M16,T1000 | 72 | 15min |
| 178710734 | dispersed M32,T500 | 72 | 20min |
| 178710735 | dispersed M8,T100 | 72 | 5min |
| 178710737 | dispersed M16,T100 | 72 | 5min |
| 178710738 | dispersed M32,T100 | 72 | 5min |
| 178710739 | contiguous M32,T1000, five anchors | 40 | 40min |
| 178710740 | frozen nested-M/T analysis | 360 | after first five farms |
| 178710741 | frozen short-T analysis | 216 | after three short farms |
| 178710742 | frozen layout analysis | 40 | after contiguous farm |

The eight dispersed farms each reserve96cores/380GB for72 workers; the contiguous farm reserves48cores/190GB for40 workers. This uses measured same-shape memory (primary fleet peak132.81GB and two-record smoke7.69GB), with headroom, rather than increasing observed M. Worst-case additional farm charge320SU plus small analysis jobs leaves total staged maximum below500SU. The short-T MPI directory is `short-T/mpi/short-t` (runner slug is lowercase). Expected analyses are `sensitivity-analysis`, `short-T-analysis` and `contiguous-analysis`; all use the unchanged `primary-analysis/model.npz`.

The five main-sensitivity farms completed with exit0 and walltimes 1m47s/2m36s/3m41s/7m27s/11m31s in table order. Short-T M8/M16 completed in 1m25s/2m02s. The original M32/T100 job failed its completion audit at4m15s: 38 records were complete and34 unfinished after the240s task cutoff. Logs explicitly report subprocesses running out of time; this is not a scientific validity result. Finished compute times were203.2–233.6s, excluding startup/ shutdown overhead. Retry178710892 uses only the34 unfinished indices, with 34cores/136GB,600s per-task timeout and12min job maximum. No completed record is recomputed and no estimator/target/gate changes. Replacement short-T analysis178710894 depends on this retry; original178710741 never ran after its failed dependency. The retry adds at most13.6SU, still within500SU overall. Do not read `.status` files as text: nci-parallel stores them in binary format; use the human-readable `.log` and output completion audit.

Retry178710892 completed successfully in4m26s; short-T analysis178710894 completed in18s. Main sensitivity analysis178710740 completed in29s. Their frozen model arrays are identical to the primary model; paired future targets are identical. Main T>=500 views have no selected-feature missingness and no excluded rows. The short-T arm passes its prespecified gate after12/216 target-blind row exclusions (5.56%); maximum selected missingness.08198. At least two eligible records remain in every role/control/M/T cell.

| Observed M | T100: held rho (eligible n) | T500: held rho | T1000: held rho |
|---:|---:|---:|---:|
| 8 | .6633 (35) | .7127 (36) | .7681 (36) |
| 16 | .7785 (34) | .8257 (36) | .8332 (36) |
| 32 | .7931 (31) | .8566 (36) | .8983 (36) |

All values compare the same frozen q with disjoint future Q. Control-mean rhos at T500/T1000 are.8833/.9500 (M8),1.0000/.9833 (M16),.9833/.9833 (M32). At T100 they are.8167/.9667/.9333. Transition-slope intervals can move one sampled grid step, so precise localization is not invariant to M,T.

For the five T>=500 follow-up cells in table order M8T500,M8T1000,M16T500, M16T1000,M32T500, paired agreement with primary q is rho=.7171/.8275/.9050/.8942/.9372. Mean shifts stay within.092 original development-score SD; median absolute differences are.3022/.2540/.1411/ .1545/.1317. This supports a useful common frozen coordinate, especially at M>=16, not exact observation-size invariance. No arm-specific re-centring, scaling, feature selection, PCA fit or sign reversal was used.

At T100, eligible totals are71/72 (M8),69/72 (M16),64/72 (M32), including both roles. Paired held q agreement with the primary is.7014/.7357/.8157; mean shifts are-.2066/-.1817/-.2399 SD and median absolute differences .3634/.2841/.2236 SD. Short records therefore show systematic displacement as well as loss of precision; common dimensionality is not exact invariance.

All648 dispersed input-member hashes have been checked against exact M,T prefixes of the primary raw archive. Their core computation identities agree. This validates that the comparison changes observations, not trajectories or the simulation truth. Durable cluster-data archive178711159 is queued after all three analyses; it will preserve working copies and verify a tar archive on gdata, outside Scratch's expiry policy. Archive code95d1bcd is fetched but the scoring worktree remains pinned9ed8aa4.

## Contiguous layout: failed transfer, not absence of information

All40 records completed (178710739,26m59s,43.17SU); analysis178710742 completed successfully in29s. The scientific validity gate fails:10/40 rows exceed5% selected-feature missingness, with maximum.23874. At r3.84 only1/4 development and0/4 evaluation rows remain, also violating minimum coverage. The14 eligible held rows have descriptive rho(q,Q)=-.1692. Their mean q stays around.97–1.07 while full-system Q falls; this is not reliable order recovery. Do not promote the reported steepest-interval match or control-mean rho from this failed, incomplete arm as a positive tracking result.

On exactly these14 paired cases, dispersed primary q has rho=.7714 with Q; cross-layout q agreement is-.2440, mean shift1.2839 SD, median absolute difference1.6932 SD. Raw mean absolute correlation and sampled local Q still have rhos.8286/.8549 with global Q. Thus patch data retain order information, but the dispersed-trained feature mask/readout does not transfer. This is not a proof that a separately developed patch-based z coordinate is impossible; no such refit or threshold relaxation was attempted.

The contiguous archive and all40 member hashes are verified locally; the frozen model is identical to primary and selected missingness/eligibility is independently recomputed from the saved feature bank. Together with the648 dispersed member checks, all688 benchmark inputs have been checked. Use fixed dispersed sampling for the demonstrated proof of concept. Increasing physical N alone does not make different observation geometries interchangeable.

## Durable outputs and final scope

Archive178711159 completed with exit0,54s,0.03SU. GNU tar's comparison against the completed source tree passed. The1.5GB archive of cluster-generated data is `/g/data/ql44/we2614/mts-spi-archives/cml2d_period_doubling_260911.tar`, SHA256 `766918b0bb2551d9423b552774b3cce9534fde887e54f049de8fa45359421ab7`. Working copies remain on Scratch; local numerical scouts and notebook artifacts remain in the repository workspace. No data were deleted.

The supported statement is unsupervised inference of an order-sensitive coordinate and across-control tracking of a canonical physical quantity under the stated observation protocol. Numerical Q calibration, longitudinal tracking, a new thermodynamic boundary estimate, exact M,T invariance, and sampling-layout invariance are not established. This is a viable additional proof-of-concept example, not evidence of superiority over specialist statistics.

Numerical source SHA256: `e9d2864b0f84a29e13642fa33ec8682a9d98f32111278b0b7d50f4ecc0600d72`. Initial isolated source commit d619b02658cbfb704c6450b17f50e01818c66428; primary protocol/config commit4dfb67e5f6befe24ee082bd5b37337381b0984e6 on `codex/cml2d-period-doubling`. Unrelated local commits were not pushed. Remote source worktree `/scratch/ql44/we2614/cml2d-source-d619b02`; data root `/scratch/ql44/we2614/mts-spi-data/order_parameter/cml2d_period_doubling_260911/`. Local data/analysis root `data/order_parameter/cml2d_period_doubling_260911/`.

## Presentation refinement and feature audit

Section7 now matches the other systems: colour encodes M using the canonical palette; T100/500/1000 uses opacity .38/.68/1.0, with square q markers and black circle Q markers. The same frozen q units and common limits remain unchanged.

Three headline snapshots use the first held seed26091115 at r3.84,3.86212,3.89, the same32 dispersed sensors, first100 consecutive input steps after200k burn. FullT1000 versions are also exported. Robust per-process scaling is estimated from each complete input, with a common colour limit over all three; this is display-only, not a change to p90. No best-looking seed/window selection. PNG/SVG and input/master hashes: `notebooks/inference/figures/cml2d-period-doubling/`. Reproduce via `scripts/cml2d_figure_diagnostics.py` through the notebook builder.

The descriptive fixed-loading audit retains19444 pairs spanning253 SPIs. Top10/top100 pairs contain .8354%/6.3245% of squared loading mass; inverse concentration is4736.76 (not a count of statistically independent features). The two largest-loading pairs are pec_orth--pec_orth_abs and pec_orth_log--pec_orth_log_abs. Splitting each pair's mass equally between its endpoints gives pec_orth_abs the largest SPI aggregate,2.2395%, followed by pec_orth_log_abs,1.5770%. Partner counts and mean mass per partner are shown. This is not causal importance or an ablation; correlated features can substitute. The exact held high-minus-low endpoint contribution sum is -2.4719955118 in canonical q units; all scores reconstruct within6.9e-15. Full pair/SPI tables: `data/order_parameter/cml2d_period_doubling_260911/feature-audit/`. No feature ranking is used to alter the frozen confirmation coordinate.

N>>M framing: large physical N realises the collective dynamics; the measured M-by-T view alone supplies q, and the global state supplies reference Q. M32/N65536 observes1/2048 of sites. This demonstrates partial-observation order tracking, not a proportional information-theoretic reduction, arbitrary layout invariance, or recovery of all unobserved dynamics. It supports the variable-observation-size motivation without replacing the original headline.

## Fresh-seed confirmation (separate from the completed pilot)

[Prospective protocol](cml2d-confirmation-protocol-260911.md):32 fresh seeds, 17 controls (nine original plus eight midpoints),544 physical masters/primary p90 records. Restricted paired M16/32,T500/1000 adds864 observations on the original nine controls;1408 new p90 views in total, all with the original frozen transform. No M64 or contiguous rescue. More seeds address uncertainty; interleaved controls address sweep resolution. The completed pilot is adequate as exploratory evidence; this batch strengthens independent confirmation. User has removed the earlier self-imposed compute ceiling; no ceiling is carried over. L and observation sizes remain scientifically motivated.

Submitted finite dependency graph (no recurring automation):

| Stage | PBS job | Last checked state |
|---|---|---|
| 544 physics masters | 178753573 | Queued |
| Frozen export | 178753574 | Dependency hold |
| M32/T1000 primary p90 | 178753576 | Dependency hold |
| M16/T500 secondary p90 | 178753577 | Dependency hold |
| M16/T1000 secondary p90 | 178753578 | Dependency hold |
| M32/T500 secondary p90 | 178753579 | Dependency hold |
| Primary confirmation report | 178753580 | Dependency hold |
| Paired-grid secondary report | 178753581 | Dependency hold |

Source `5f76768a0b9be7d8b3f5d07fcd1c03f67d8d6f38` on `codex/cml2d-period-doubling`; source worktree `/scratch/ql44/we2614/cml2d-confirm-source-5f76768`. Physics output `/scratch/ql44/we2614/mts-spi-data/order_parameter/cml2d_confirmation_260911/physics`; corpora/outputs `/g/data/ql44/we2614/mts-spi-data/order_parameter/cml2d_confirmation_260911`. The latter stores `submission.json` with full job IDs and commit, then generated hash-bound corpora and frozen-input identities. Gdata was chosen to avoid Scratch inode pressure (188,585/202,000 before launch versus47,465/70,000 on gdata); no prior data were removed. Pyspi remained clean at65317c9c1fd5f12358b8ede09b7576ef001a76dd. Ten numerical/export/frozen-fit/design/gate tests pass locally and on Gadi. Notebook42 cells executes without errors; headline snapshot and M/T colour/ opacity figures visually checked. No confirmation outcomes are available yet.

Next: inspect `primary-analysis/confirmation-report.json` and `sensitivity-analysis/confirmation-report.json` after the dependency graph finishes; verify masks/models/inputs against the sealed identities, review reference drift and paired q agreement, then append confirmation separately from the pilot. The generic scorer's historical 'exploratory' summary label is retained; the prospectively specified confirmation gate/report supplies the independent-confirmation status, not a retrospective relabelling of the pilot.

## Confirmation launcher failure and repair

The first physics job178753573 exited1 after8m37s; all seven dependents were terminated without running. The simulation itself did not fail: the launcher extracted its source into node-local JobFS on the head node, unavailable to other nodes. Logs explicitly report `ModuleNotFoundError: No module named 'scripts'` for those workers; some head-node cases completed. Failed-attempt charge165.44SU. The prior ledger's queued status is historical, not current.

Repair commit0b813b8b5a114a9b5acc32f3980f6c907c265f38 stages an immutable snapshot on shared Scratch and checks an import on every allocated node. `RESUME=1` verifies case identity, source/config hashes, shapes and finite/bounded arrays before reusing each existing archive; only missing indices are run. Scientific simulation source, seeds, control grid, horizons, observations and frozen q are unchanged. No completed files are overwritten. A separate two-node smoke178771550 gates the replacement finite dependency graph. Original `submission.json` is preserved; replacement IDs go to `submission-retry.json`.

Figure assessment: sensor-by-time heatmaps are valid input illustrations, but adjacent dispersed-sensor rows are not spatial neighbours; robust display normalisation is not a visual measurement of absolute Q. This is now explicit in the notebook caption. A256x256 lattice frame is easy to display but cannot by itself demonstrate a temporal period-doubling transition. No larger lattice, new feature ablation or additional scientific arm is justified by this question; finish the fixed confirmation first. Short aligned global-mean traces would be a more direct optional explanatory addition than a row of static lattice maps.

There are46 completed master archives; the retry schedules the remaining498 after validation. Replacement submission is recorded: two-node smoke178771550 then physics178771601, export178771602, primaryp90178771603, M16T500178771604, M16T1000178771605, M32T500178771606, primary report178771607 and secondary report178771608. Source0b813b8, unchanged scientific source/config hashes and frozen model. Ten tests pass remotely; notebook42 cells executes after the caption clarification. This records submitted recovery, not completed results.

## Matched visual context and confirmation continuation

User clarified that the global-mean traces and optional lattice image are wanted. Added the exact first100 global-mean steps above the three unchanged MTS examples, with common vertical limits. Downloaded only the corresponding pilot masters case004/028/068 and verified their manifest SHA256 plus exact sensor-input equality. The small boundary-control lattice panel uses the saved final field at record index1001999, after the reference, with its32 sensor positions marked; it is explicitly not simultaneous with the MTS panels. No regenerated trajectory or extra simulation is used. Updated PNG/SVG and timing/hash manifest are in the existing figure directory; visual QA passed.

Shared-source smoke178771550 and physics retry178771601 completed successfully (29s/9m47s); export178771602 completed3m28s. All544 masters and1408 views exist. Primary archive SHA3d3010d8503cee0f5464f8401b02e56575d590b03d6f9e85bc1d202e50435539; secondary SHA73ace585e6272b7ba3f199ebff71096f19f513fdf91bafd629dd1fef8aad2e03. The32-seed Q curve decreases across all17 controls; at r3.86306, max half-mean difference.07912 and max descriptive block SE.02313 retain the finite-run interpretation. No targets or horizons were adjusted in response.

A second launcher error stopped p90 jobs178771603–606 before extraction: `run_dataset_farm.pbs` expects synthetic `mts_classes`, not exported archives. No p90 results were produced and reports607/608 did not run. Fixed only the routing to the existing pilot-tested `run_external_corpus_farm.pbs`, using committed shape-index files and validating both archives/counts first. Repair sourcee9cf5ccfae6d2e2f739fa5d712c98c645b258bfe; eleven tests pass remotely, including a new regression assertion for archive-runner routing. Replacement jobs: primary178774140, M16T500178774141, M16T1000178774142, M32T500178774143, primary report178774144, secondary report178774145. `submission-p90-retry.json` preserves these IDs without replacing earlier ledgers. No physics/export rerun, model changes or new experiment was introduced.

## Final confirmation completion

All final stages finished successfully: primary extraction17877414028m24s; secondary1787741415m34s/1429m13s/14313m15s; reports1441m40s/1451m57s; integrity audit17877613940s; physics archival1787768643m08s. All1408 records pass the prospective quality gates, zero selected missingness and zero exclusions. Main rho.883018[.867676,.898475], controlmean1.0; new-midpoint-only rho.879546; q/Q steepest interval3.86212–3.86306. See the final report for paired M,T, baselines and finite-reference limitations.

The separate audit verifies all1408 member hashes,864 exact prefixes/shared references, every frozen model array and score(maxerror2.22e-16), execution-core identity, and eight predetermined MPI replays. Local checks additionally verify both corpus hashes, every prefix, both copied models and all six bootstrap overall/within-control intervals to1e-12. Notebook46 cells executes without errors. The two confirmation plots, matched global/MTS figure and lattice panel were visually checked. A presentation-only backend issue was corrected: the notebook now derives the original control grid from pilot scores instead of importing the CLI module that switches Matplotlib to noninteractive Agg.

Physics archive4.9GB passed tar comparison; SHA256 `51bb6c94934290426ce9e506557d8af2c606842a8159288cf803aa1da6fae707`. Full corpora/features/MPIs/reports already reside on gdata. No data deleted. Actual confirmation charge including both failed launcher attempts, smoke, all successful stages and archival:1179.91SU; with the319.87SU pilot,1499.78SU. No recurring automation or new scientific experiment remains to run.
