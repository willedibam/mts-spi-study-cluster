# Gadi storage audit and organisation, 22–23 September 2026

Scope: project `ql44`, primarily `/scratch/ql44/we2614` and `/g/data/ql44/we2614`. Counts include directories and symlinks; byte estimates in the inventory are logical sizes. Dates are file modification dates (UTC in the CSV), not last access or necessarily generation dates. Symlinks were not traversed when counting storage. [Full run inventory before cleanup](gadi-storage-inventory-260922.csv).

## Quotas and maintenance

Before this maintenance: Scratch 192,745/202,000 inodes and 179.74 GiB/1 TiB; gdata 68,857/70,000 inodes and 34.03 GiB/100 GiB. The preceding read-only check recorded 192,722 Scratch inodes. Project quotas include other users: `bf7108` accounts for approximately 2,202 Scratch entries and 55.58 GiB; its files are outside this cleanup.

Authorised work: preserve recoverability and remove EEML's loose environment; archive all five July r0/r1 cohorts; remove the obsolete MTS environment remnants, TASEP, Ising and superseded execution smokes; organise inference artifacts by system; flag non-`benchmarked_p90` results rather than silently delete them.

The first cleanup completed successfully at 00:38 AEST on September 23. At that checkpoint Scratch was 116,434/202,000 inodes (57.6%), 171.60 GiB; gdata is 64,531/70,000 (92.2%), 36.45 GiB. Approximately 76,300 Scratch inodes were recovered. All six archive comparisons completed and SHA-256 checksums were recorded; the archives total 4,122,380,337 bytes. Removed the obsolete MTS remnants, Ising trash, both Scratch TASEP roots, gdata TASEP N32/N64 banks, 16 superseded smoke artifacts and 39 TASEP/Ising job-output files. The current MTS environment passed `import src, pyspi`; every relocation/link resolved. Archive/check/delete journal: `/g/data/ql44/we2614/mts-spi-study/operations/maintenance/storage-cleanup-260922/actions.jsonl`. The later verified Zenodo archive removed 3,160 loose entries and its obsolete alias; its recovery archive is 269,015,813 bytes with SHA-256 `b10e689b0544e22822a492032bd72aea4b1d38892ff602c710f2a16efe78e7e7`. Successful jobs: archival `179610308` (28m22s), retained-dataset catalogue audit `179610451` (1m33s), finalisation/reorganisation `179610782` (1m03s), superseded Zenodo archive `179613075` (1m41s). Archives are compared against their source before source removal, then SHA-256 recorded. No historical scientific result or source code is edited.

## Final checkpoint, 23 September

The final project cleanup is complete. Project quota usage at the final check: Scratch **155,063/202,000 inodes (76.8%)**, 180.62 GiB; gdata **61,080/70,000 (87.3%)**, 35.84 GiB. These are project-wide totals, not isolated cleanup deltas. Baseten grew from 84 entries in the initial inventory to 38,844 in the final physical count, explaining most of the Scratch increase since the earlier checkpoint; Baseten was not modified by this cleanup.

Removed the authorised 820-record short-burn CML sweep (3,282 entries). Archived and compared 1,135 old PBS log files before removing their loose copies: 1,378,611,909 source bytes became a 52,963,118-byte archive at `mts-spi-study/archives/operations/logs/pbs-through-20260923.tar.zst`, SHA-256 `c5f717631bb3e1d37ed85b056765f739b359f36c87acf8d268e4e5e456fe73a6`. New jobs retain a clean live log directory.

The main pass removed 284 obsolete aliases; the subsequent simplification removed 23 source-navigation/data-backlink aliases and redundant documentation aliases. The gdata Kuramoto folder had contained only a source-checkout link leading back to the whole project, not Kuramoto data; that misleading view was removed. Actual Kuramoto banks remain on Scratch. Large-M artifacts moved to `mts-spi-study/pyspi-optimisation/large-m-260917/`. New workstreams/subfolders are allowed without a whitelist or mandatory registry. The index is an audit snapshot, not a permanent requirement for ordinary moves.

Validation: all 13 registered checkout states were preserved during relocation; the one uv-created environment retained identical package names/versions and passed imports/entrypoint checks; representative MPI files loaded from eight banks; five path-resolution tests passed; all 3,780 historical proof paths resolved locally; the final layout audit and an arbitrary new-workstream output passed. PBS jobs `179634981` (1m55s) and `179637201` (1m59s) exited zero. Earlier verification attempts stopped before archival because of test discovery and the now-removed category whitelist; both were corrected. No pyspi computation was rerun.

Current Gadi launcher/path changes were deployed from `codex/gadi-storage-layout` (through commit `8fc2bc1`, based on the previous Gadi main-checkout commit `18ef949`). Matching path edits are in the local working tree, preserving pre-existing user edits. Integrate the migration branch when next reconciling the local main branch with the cluster; do not force-reset either checkout. Historical generated YAML/metadata remain provenance records. Config/source hashes are part of completion identity, so a path edit alone must not trigger overwriting a retained scientific result; replay needs explicit translated paths and the correct original identity.

## Canonical project/function layout

The September 23 second pass replaces the earlier compatibility-heavy layout. Read the [storage contract](gadi-storage-layout.md) for placement rules, the project `documentation/storage-index.csv` for physical owners and `documentation/path-map.json` for historical lookup. The two stores share naming conventions, not blanket symlink mirrors. Categories are extensible; no mandatory folder registry is required. Redundant per-system source links and snapshot data links back to the project were removed to prevent recursive navigation. Scientific payloads stay on their original filesystem.

```text
mts-spi-study/
  proof/
  order-parameter-inference/<system>/
  zenodo/7118947/
  representation/{synthetic,empirical,cross-mt,baselines}/
  pyspi-optimisation/large-m-260917/
  archives/<workstream>/
  operations/{sources,logs,maintenance}/
  environments/
  dev/                         # only when needed on Scratch
  legacy-non-p90/
  documentation/
eeml-2026-application/archives/{environment-260922,july-2026}/
baseten-research/
tusz/
```

MTS archives live on gdata, with one explicit Scratch archive link. The sole uv-created MTS environment lives on Scratch, with one explicit gdata environment link. Source checkout `.venv` links all share it. EEML owns its environment recovery archive and July cohorts. Shared caches use `.cache/`; active editor/runtime sockets remain in `tmp/`.

The obsolete store-root `mts-spi-data`, `mts-spi-data-v2`, `mts-spi-archives`, feature wrappers, environment aliases, individual workstream aliases and stray script/config aliases are removed. Registered Git worktrees use physical canonical paths. Current Gadi launchers/configurations are updated; frozen source snapshots and result metadata retain historical text, resolvable through the path map. Historical replay requires translated paths rather than assuming old absolute names still exist.

Shared `multi` proof inputs remain in `proof/`; cross-M/T-specific data, features and analyses move to `representation/cross-mt/`. Inference systems remain distinct. CML2D's Scratch physics and gdata MPI components are complementary, not duplicates; their physical locations are indexed separately. The user additionally authorised deletion of the 820-record short-burn `cml_param_sweep_260508` bank. Its compact derived feature bank is retained; the raw/MPIs will no longer be available for replay from Gadi.

## Dataset iteration chains

An iteration is not necessarily a duplicate or a safe deletion. Replacement evidence is distinguished here from fresh confirmation, different physical observations and controls.

| Chain | What changed | Interpretation / treatment |
|---|---|---|
| Zenodo unseeded → seed1729 | Same 1,053 source-member hashes; deterministic estimator RNG introduced after reproducibility findings | Genuine corrected replacement. Keep seed1729; provisional bank was archived and verified (3,160 loose entries removed). After clarifying that the initial layout approval did not explicitly cover this archival, the user explicitly chose to keep the verified archive only on 23 September. |
| NeuroTycho float32 scout → float64 scout | Premature float32 storage introduced ties and broke 11 information-estimator configurations; corrected from cached raw data | Genuine corrected replacement; old 13-entry scout is labelled obsolete. The float64 four-date scout and seven-date `extra` bank are complementary parts of the 11-date source cohort, not replacements. See [precision diagnosis](../neurotycho-source-pilot.md). |
| Legacy six-class CML panel → current proof + CML additions | 297→289 SPI catalogue; four missing CML classes recomputed, two shared classes represented in base proof | Scientific replacement of the old catalogue panel; old panel flagged. All 1,800 raw inputs checked across these banks are byte-distinct. |
| Legacy proof 80/90/95 → August p90 proof / cross-M,T | Old 267/297/312-SPI banks replaced by the current catalogue and protocol | Old archives flagged. Current 900 base + 360 additions form development; 2,520 confirmation rows are deliberately separate held-out instances. |
| Kuramoto benchmark → confirmation → final confirmation | 880→1,152→1,536 records; Gaussian-paired coupling grids have 10→12→16 distinct settings, with no matching coupling/seed pairs across stages | Genuine successive experiment designs, not copies. Gaussian random terminal data support current retrospective analysis; logistic/regular arms are historical alternatives. The later 256-row full-observation bank changes N/observation and remains separate. |
| Quadratic CML alpha sweep → long-burn large-lattice development | 820 short-burn M20/T1000 records versus 1,224 large-lattice/small-system records with much longer burn-in | Later scientific refinement, not an identical rerun. The user authorised deletion of the old 820-row bank on September 23; derived features remain. Keep the long-burn development bank. Two-dimensional CML is a different system. |
| CML2D pilot → independent confirmation | 688 pilot/sensitivity/patch records → 1,408 fresh-primary/paired-sensitivity records with new seeds and control midpoints | Development and independent confirmation both needed. Small-L full-observation and structural audits are separate negative/control branches. |
| Stuart–Landau development → broad confirmation → fine boundary | 1,440→1,296→152 records; broad M/T/observation checks followed by a narrower gamma interval | Successive evidence, not interchangeable banks. Fine boundary does not replace broad robustness. |
| Miller–Huse development → confirmation; physics primary → stationarity/long truth | New datasets plus separate numerical/physical validation stages | Preserve development, confirmation and validation. Later truth checks do not replace MPI inputs. |
| Desai–Zwanzig discarded-Euler / archived-NaN-CI folders → accepted fine-boundary data/analysis | Names suggest abandoned numerical or reporting iterations; exact rationale was not independently verified in this audit | Keep labelled pending a scientific retirement decision. T-sensitivity and finite-size validation are separate arms. |
| Interaction-share pilot → continuous-parameter confirmation → phase controls | Fresh continuous controls/disjoint cohorts, then common/independent Fourier-phase surrogate recordings | Extension and mechanism controls, not duplicate data. See [confirmation](../interaction-share-confirmation.md) and [phase-control design](../interaction-share-phase-control.md). |
| Oscillatory scout → pilot → faster-regime / direct-mechanism transfer | 96-view feasibility, 520-view source study, two distinct 400-view target banks | Pilot supersedes the scout's feasibility role; target banks test different shifts with frozen source models and should remain separate. |
| HCP one-person scouts/profiles → 24-person cohort; raw → prepared → T4000 bank | Population/observation refinement plus preprocessing stages | Different scope/stages. Raw and prepared files are not redundant copies; cohort preparation has 48 runs, 590 clean blocks and 1,180 dependent views. No claim that predictive study is complete. |
| NeuroTycho matched/enriched neural, library and InceptionTime follow-ups | Different models/exposure on existing source/target observations | Model-result iterations, not new independent datasets. Keep separate model/analysis folders and shared inputs. |

## What is inside mts-spi-data

| Family/run | Original location; entries | Dates | Meaning and recommended treatment |
|---|---|---|---|
| Five July `26072*_r*` cohorts | Scratch; 42,152 | Jul 27–28 | EEML chain/fork/collider studies: nonlinear r1 (4,500 datasets), linear VAR r0 (3,000), observation r1b (3,000), demo (24), probe (9). Archived and loose copies removed. Legacy 297-SPI catalogue: also flag the archives for possible later deletion. |
| `embeddings/cml-embedding` | Scratch; 2,167 | Jun 29–30 | Six CML regimes × nine M/T cells × ten instances = 540 datasets; legacy 297-SPI panel. Flag for deletion. Four classes were recomputed under p90, while defect-turbulence and STI-I are in the p90 proof bank. |
| `embeddings/kuramoto_k_sweep_260616` | Scratch; 1,642 | Jun 16–17 | 410 legacy coupling-sweep datasets, 297-SPIs. Flag for deletion; scientifically different from the later N256/global-Q benchmark. |
| `embeddings/kuramoto_explosive_260616` | Scratch; 3,802 | Jun 16–17 | 820 raw series but only 670 metadata/SPI results; incomplete older explosive-synchronisation sweep, legacy 297-SPIs. Flag for deletion. |
| `embeddings/cml_param_sweep_260508` | gdata `mts-spi-data-v2`; 3,282 | Aug 24 | 820 p90 datasets; old filename date is misleading. One-dimensional quadratic-CML alpha sweep, eps=.3, short burn 2,000, M20/T1000. User authorised deletion on September 23. Compact derived features are retained. Distinct long-burn quadratic-CML and CML2D banks are unaffected. |
| `embeddings/multi_p90_260701` | gdata `mts-spi-data-v2`; 3,611 | Aug 24 | Current 900-row, ten-class proof development bank: M8/16/32, T500/1000/2000, instances 0–9. Keep. Scratch path is already a symlink. |
| `embeddings/cross_mt_cml_development_260824` | gdata `mts-spi-data-v2`; 1,445 | Aug 24 | 360 additional development rows, four CML classes, completing the 14-class proof. Keep; not a duplicate of base proof. |
| `embeddings/cross_mt_confirmation_260824` | gdata `mts-spi-data-v2`; 10,095 | Aug 24–25 | 2,520 held-out rows, 14 classes × nine M/T cells × instances 10–29. Keep separate from development. |
| `kuramoto_order_benchmark` | Scratch; 4,409 | Aug 22 | 880 development datasets, Gaussian/logistic frequency populations and paired/independent controls. p90; preserve while retiring redundant branches is considered. |
| `kuramoto_confirmation` | Scratch; 5,767 | Aug 22 | Earlier 1,152-row confirmation/eligibility bank with random/regular frequency variants. Later terminal confirmation supersedes its inferential role, but the runs are not established byte duplicates. Archive candidate, not automatic deletion. |
| `kuramoto_final_confirmation` | Scratch; 7,687 | Aug 22 | 1,536 terminal rows across six frequency/sampling arms. The Gaussian random paired arm provides the 512-row full-catalogue retrospective analysis. Keep these; logistic/regular arms are historical alternatives, candidates for archiving. |
| `kuramoto_full_observation_260916` | Scratch; 785 | Sep 16 | New 256-row M=N32 bank. Distinct observation contract, seeds and controls; keep as the current full-observation example. |
| `miller_huse_development` / `confirmation` | Scratch; 1,442 / 1,082 | Sep 2 | 288 / 216 p90 datasets; distinct development and confirmation. Physics primary/audit/stationarity/long-truth folders document validation rather than duplicate MPI farms. Keep and group. |
| `stuart_landau_development` / `confirmation` | Scratch; 7,203 / 6,483 | Sep 1–2 | 1,440 / 1,296 p90 datasets spanning full/partial observation and M/T. Preserve paired sensitivities and failed joint-gate evidence. |
| `stuart_landau_locking_boundary_confirmation` | Scratch; 762 | Sep 3 | 152 fine-boundary p90 datasets. Refines the control interval; does not replace the broader M/T benchmark. Keep. |
| `quadratic_cml_development` | Scratch; 6,125 | Sep 2 | 1,224 p90 datasets: 984 N512 partial observations and 240 small fully observed controls, long burn. Later and physically distinct from the 820-row short-burn alpha sweep. Keep; do not conflate with 2D CML. |
| `cml2d_period_doubling_260911` | Scratch; 2,234 | Sep 11 | 688 pilot/paired sensitivity/patch records plus two execution-smoke records. Keep scientific arms, remove smoke. |
| `cml2d_confirmation_260911` | Scratch + gdata; 567 + 4,268 | Sep 11 | 544 fresh primary records plus 864 paired M/T sensitivities. Scratch physics and gdata MPI outputs are complementary. Keep and unify browsing. |
| `cml2d_full_observation_260914` / `cml2d_structure_260914` | Scratch + gdata; 1,382 + 2,084 / 1,990 | Sep 14–15 | Small-lattice full-observation failure and structural/attractor audit. Distinct negative evidence, not duplicates of successful large-L runs. Keep; archive later if inactive. |
| Desai–Zwanzig | Scratch; several folders | Sep 3–4 | Primary fine-boundary p90, T500/T100 sensitivity and finite-size physics validation. Different arms; group. Folders labelled discarded Euler / archived NaN-CI look superseded; their exact diagnostic rationale was not independently verified, so they remain retained and labelled. |
| `finite_regime_260915` | gdata; 6,897 before deletion | Sep 15 | Rössler, Lorenz96, Hindmarsh–Rose and TASEP. TASEP deleted as requested; keep the other scientific banks, including negative outcomes. |
| `vicsek_observation_260910` | Scratch; 55 | Sep 10 | Long/narrow physics-only scouts; not a p90 feature corpus. Keep scientific evidence, remove execution smokes. No catalogue means not applicable here, not a non-p90 feature result. |
| `zenodo_7118947` | gdata; 6,328 | Aug 25–26 | Two 1,053-row p90 runs: provisional unseeded and authoritative seed1729. The latter supersedes the former; random estimators mean these are not necessarily identical. Seed1729 remains live; the provisional bank is now archived and its loose copy removed. Scratch root is a symlink. |

The other September gdata families (`interaction_share*`, `covariance_modulation*`, `oscillatory_*`, NeuroTycho, HCP and large-M pair baselines) belong to SPI–SPI representation evaluation rather than this inference run hierarchy. Their audited MPI metadata use current p90; they remain separate. No retained metadata outside the authorised TASEP/Ising removal trees mentions either system.

## Catalogue policy and apparent duplication

The accepted catalogue identity observed in current metadata is `configs/pyspi/benchmarked_p90.yaml`, SHA-256 `bc4bafa16b4add8bb6283db490a9fa3d8dde4fde47093147d32579b989397965`, 289 SPIs. Folder names are not authoritative: `mts-spi-data-v2` contains current pyspi-v3/p90 datasets. Conversely, `benchmarked90_amortized_config.yaml` is an older 297-SPI catalogue and is flagged even though its name contains “90”. Physics-only datasets and source code are outside this catalogue deletion criterion.

`spi-spi-direction-v2/cml_embedding.npz` explicitly records the old 297-SPI catalogue and missing hash/version provenance: flag alongside its source panel. `cml.npz` and `proof.npz` explicitly record current p90 provenance: preserve. Old and new feature representations of the same accepted p90 datasets need not be redundant: directional-v2 and unified-v3 have different feature contracts. Do not delete a compact feature bank just because its raw inputs are shared.

The already archived `mts-spi-archives/legacy-proof-v2/proof_benchmarked{80,90,95}_260603.tar.zst` banks are legacy-catalogue candidates, with negligible inode impact (three payloads). Current proof and CML additions remain necessary to reproduce the cross-M/T and September 21 baseline analyses. Before smoke removal, the exhaustive audit read 28,793 metadata records outside the authorised July/TASEP/Ising removal trees without errors: 27,171 exact-hash 289-SPI p90 records, 1,620 legacy 297-SPI records, and two 290-SPI/no-hash p90 smoke records. All 1,620 legacy records belong to the three old CML/June-Kuramoto banks above. See the [catalogue audit](gadi-catalogue-audit-260922.csv). The audit excludes already authorised July/TASEP/Ising removals; archived July and legacy proof metadata were audited separately in full. All 10,533 July records use legacy 297-SPIs. The three older proof archives each contain 900 records with 267/297/312 SPIs for the 80/90/95 catalogues respectively; all are flagged for deletion. The existing CML2D pilot archive contains 690 current-p90 records and is retained.

SHA-256 comparison of all 1,800 raw inputs in the old 540-row CML panel, 900-row current proof and 360-row current CML additions found no identical raw files. These are related experiments/recomputations, not established duplicate files. The three Gaussian-paired Kuramoto stages use disjoint coupling grids (10/12/16 controls) and have zero matching `(coupling, generator seed)` pairs; preserve their separate provenance. Zenodo's two banks have matching stored source-member hashes for all 1,053 inputs; their SPI calculation seeds differ (unset versus 1729), so the input duplication does not establish identical SPI outputs.

Among 15,165 retained metadata records with explicit generator identity and seed, six repeated generation-specification groups (14 records total) were found; every repeated group consists of discarded/preproduction smokes and its production counterpart. This checks recorded specifications, not full-byte equivalence or every external-archive dataset. No other repeated specifications were found under that comparison.

## Environment recovery

The discovered checkout `.venv` links share one physical uv-created environment, now under `/scratch/ql44/we2614/mts-spi-study/environments/mts-spi-v3-631de27`. The obsolete nine-entry Python 3.11 remnant was removed. Moving the shared environment requires repairing embedded launcher/activation paths and verifying unchanged package versions and imports.

EEML's separate 26,639-entry environment is archived before removal. `storage-cleanup-260922/eeml-rebuild/` preserves the actual installed versions, `pyvenv.cfg`, project `uv.lock`, `pyproject.toml`, setup guide and Git commit. Its actual installed package set differs from the broader project dependency declaration, so the verified environment archive is the exact recovery fallback. Restore at the original path after loading `python3/3.12.1`; consult the saved README. A fresh network installation has not been tested.

## Remaining decisions

Non-p90 datasets and feature/archive artifacts are flagged in the catalogue CSV; they have not been deleted solely on catalogue grounds. The three loose legacy CML/June-Kuramoto banks account for 7,611 Scratch entries, plus one old CML feature archive. July and legacy proof tarballs are also flagged, but deleting those recovers few inodes.

The provisional unseeded Zenodo bank is archived at `/g/data/ql44/we2614/mts-spi-study/archives/zenodo/7118947/unseeded-p90-260825.tar.zst`; the authoritative seed1729 bank remains live. A pointer and restoration instructions are in `mts-spi-study/zenodo/7118947/runs/ARCHIVED-UNSEEDED.md`. The old provisional directory/alias is deliberately absent; the relocation/catalogue CSVs mark it as requiring restoration. At the first-cleanup checkpoint gdata was 92.2% of its inode quota; check headroom before another large loose-file farm. No other p90 scientific bank was discarded based only on age, similar naming or a failed scientific gate.

Recovery instructions and archive checksums: `/g/data/ql44/we2614/mts-spi-study/operations/maintenance/storage-cleanup-260922/README.md`. The full pre-cleanup inventory and metadata evidence are retained there as compressed JSONL. July cohort recovery now belongs to `eeml-2026-application/archives/`.
