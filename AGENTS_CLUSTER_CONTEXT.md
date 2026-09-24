# NCI Gadi context

This file is repository/account-specific. See `HPC.md` for the generic group guide.

Storage audited on 2026-09-22; allocation, quotas and checkout state rechecked on 2026-09-24. Queue limits below were last verified on 2026-08-28 and must be rechecked before a production submission.

## Account and layout

- Login: `we2614@gadi.nci.org.au`; project/default group: `ql44`.
- Repositories: `/home/562/we2614/mts-spi-study-cluster` and sibling `../pyspi-fork`.
- Main links: `.venv -> /scratch/ql44/we2614/mts-spi-study/environments/mts-spi-v3-631de27`, `data -> /scratch/ql44/we2614/mts-spi-study`, `logs -> /scratch/ql44/we2614/mts-spi-study/operations/logs/pbs`. Read the [storage contract](docs/operations/gadi-storage-layout.md) before adding or moving cluster artifacts. The project `documentation/storage-index.csv` is a migration snapshot, not a mandatory registry; `python scripts/gadi_storage_path.py OLD_PATH` resolves historical paths. The dataset-farm submitter checks the layout and output placement before submission.
- Allocation: Scratch 1 TiB / 202k inodes; gdata 100 GiB / 70k inodes. Live 2026-09-24 checkpoint: Scratch 211.23 GiB / 141,948 inodes (70.3%); gdata 35.84 GiB / 61,086 (87.3%). These are project-wide totals. Baseten grew from 84 to 38,844 entries between earlier audit checkpoints and was not modified by the MTS cleanup. Check current byte and inode quotas before large runs. See the [storage audit](docs/operations/gadi-storage-audit-260922.md).
- Canonical project roots are `/g/data/ql44/we2614/mts-spi-study/` and `/scratch/ql44/we2614/mts-spi-study/`: `proof/`, `order-parameter-inference/<system>/`, `zenodo/7118947/`, `representation/`, `pyspi-optimisation/`, `archives/<workstream>/`, `operations/{sources,logs,maintenance}/`, `environments/`, optional Scratch `dev/`, `legacy-non-p90/` and `documentation/`. Cross-M/T data/features/analysis belong to `representation/cross-mt/`; the shared multi-system bank stays in `proof/`. Mirror naming conventions, not contents. Create new workstreams/subfolders when scientifically appropriate; no fixed category allowlist or mandatory registry. Prefer direct moves and shallow folders. Large-M estimator/pair-baseline work lives at `pyspi-optimisation/large-m-260917/`. Keep source checkouts only in `operations/sources/`; do not add per-system source aliases or snapshot `data` links back to the whole project. Consult the index before assuming a run exists on a particular store. No individual workstreams, `mts-spi-data*`, `mts-spi-archives`, `venvs` or compatibility aliases at store roots.
- One physical uv-created MTS environment lives at `/scratch/ql44/we2614/mts-spi-study/environments/mts-spi-v3-631de27`; checkout `.venv` links point directly to it. Gdata exposes that environment through a deliberate one-way link. Scientific MTS archives are in `/g/data/ql44/we2614/mts-spi-study/archives/`; Scratch exposes them through one link. Frozen source worktrees retain historical text; use current launchers or explicitly translated replay configs rather than assuming old embedded paths still exist. Code/worktree registration must use physical canonical paths.
- EEML's archived environment/rebuild record and five July cohorts belong to `/g/data/ql44/we2614/eeml-2026-application/archives/{environment-260922,july-2026}/`. EEML, TUSZ and Baseten remain separate projects. Shared machine caches belong to Scratch `.cache/`; active editor/runtime sockets remain in `tmp/`. Maintenance receipts are in `mts-spi-study/operations/maintenance/`. Keep one physical copy and record a retirement decision before archival/deletion.
- The old 820-record short-burn `cml_param_sweep_260508` data bank was deleted as authorised; its compact derived feature bank and the distinct long-burn quadratic-CML/CML2D banks are retained. TASEP, Ising and superseded execution smokes were deleted. Non-`benchmarked_p90` SPI outputs remain flagged, not automatically deleted; physics-only validation is exempt.
- Zenodo seed1729 remains live at `mts-spi-study/zenodo/7118947/runs/authoritative-seed1729/`. The earlier unseeded bank is archived at `mts-spi-study/archives/zenodo/7118947/unseeded-p90-260825.tar.zst`; the user explicitly chose archive-only retention. Historical unseeded reproduction requires extraction.
- Passwordless public-key SSH is configured. Use `gadi-dm.nci.org.au`, not a login node, for `scp`/`rsync`. Login nodes are for light management; standard compute nodes have no external network, so synchronise Git before submission or use `copyq` for networked work.
- PBS jobs implicitly mount the charged project's Scratch. Add `#PBS -l storage=gdata/ql44` (and any other project filesystems used) when a job needs gdata; an omitted storage mount appears as a missing path inside the job.

## Git and environment

- Local repositories are authoritative during development. Commit the intended files, push, then fetch/fast-forward the matching Gadi branches; never pull over uncommitted work.
- Storage-path updates were deployed from `codex/gadi-storage-layout` (`8fc2bc1`, based on prior Gadi commit `18ef949`) into the existing cluster checkout. The migration lineage was integrated into the local `refactor-lagged-warping` branch on September 23, preserving the newer local code and user notebook removals. Future cluster updates can follow that merged history; do not force-reset.
- Gadi tracks main-repo branch `refactor-lagged-warping` and pyspi branch `v3`. Verify both commits before each production submission.
- Read-only preflight on 2026-09-24: Gadi main checkout `791964f`, pyspi `65317c9`, both clean; local main was ahead at `a7f0a3a`. No user PBS jobs were listed. No synchronisation or submission was performed.
- The active Scratch Python 3.12 environment has editable main/pyspi-v3 installs. The obsolete broken v2 environment was removed on 2026-08-22. Require `import src, pyspi`, the fast tests, and a one-dataset smoke test after any rebuild.
- These experiments use `configs/pyspi/benchmarked_p90.yaml` (289 SPIs) and one pyspi worker per dataset. The exact-GP additive-noise-model SPI is intentionally disabled: six `M=20,T=1000` tasks each exceeded 18 minutes inside it in job `177018028`.

## Compute allocation and charging

- `nci_account` reports the spendable quarterly allocation. Live 2026.q3 balance on 2026-09-24: 124.51 KSU granted, 85.49 KSU used, 39.02 KSU available, zero reserved. The grant can change; query again before submission.
- `normal` has 48 cores and 192 GiB physical RAM per Cascade Lake node (PBS maximum 190 GiB), 4 GiB/core for charging, 2 SU/resource-hour, and a 20,736-core maximum request. Requests above one node use whole 48-core nodes.
- Charge is based on actual walltime and the greater of requested CPU or memory-equivalent cores, but PBS must be able to reserve the requested maximum before starting. A 2,016-core job costs about 4.032 KSU per wall-hour before any memory uplift.
- Live PBS: `max_array_size=10`; `normal` allows 1,000 queued jobs/project and its execution queue allows 300/project. These job-count limits are not useful dataset-level parallelism.

## Dataset-level parallelism

- Use one multi-node PBS allocation with `nci-parallel`, not thousands of one-core PBS jobs or the old array wrappers. Each command processes one dataset with `--n-jobs 1`; pin BLAS/OpenMP threads to one.
- `nci-parallel` dynamically assigns the next dataset to a free core. For homogeneous tasks lasting minutes, concurrency can approach the task count (rounded to whole 48-core nodes above one node); for heterogeneous tasks, use measured runtime tails to avoid paying for many idle cores near completion.
- Keep farms reasonably homogeneous in expected M/T/config cost; use separate farms or index ranges when task classes have materially different runtimes.
- Default progression: 2-dataset/2-core smoke test; representative 48-core node test; then choose 192, 480, 960, 2,016, or more cores from measured runtime variance, memory, queueing and remaining KSU. Maximum concurrency is not automatically minimum time-to-result.
- Use `jobs/gadi/submit_dataset_farm.sh`; its PBS worker is `jobs/gadi/run_dataset_farm.pbs`. The launcher defaults to p90, one core/task, no CSV/heatmaps, resumable `--skip-existing`, and persistent job log/status files.
- Set `TASK_TIMEOUT` for homogeneous production farms after measuring the representative runtime tail; the status file identifies timed-out indices, and a resubmission skips completed outputs via `--skip-existing`. Leave it unset for heterogeneous timing scouts.
- Batch pyspi runs keep Calculator INFO/progress output off; warnings, errors and per-SPI timings remain in metadata. Do not multiplex hundreds of progress bars through `nci-parallel`.
- Keep generated artifacts numeric: `timeseries.npy`, compressed `spi_mpis.npz`, compressed `ground_truth.npz`, and small JSON/log files. Do not generate heatmaps or CSV tables for farms.
- Measured `M=20,T=1000` p90 pilot `177019354`: median `603 s`, maximum `774 s` per dataset; six tasks peaked at 11.4 GB total. Corner gates `177020665` and `177020346` measured maxima of `1716 s` for `M=20,T=2000` and `1908 s` for `M=32,T=1000`, with three-task peak memory of 13.6 and 9.46 GB respectively. Request 8 GB/core for those two classes and 4 GB/core for `M=20,T<=1000` and `M=8`; `M=20` remains the production observation size.
- Production job `177026144` used one 1,776-core allocation for 880 concurrent `M=20,T=1000` commands, requested 7,104 GB, and finished in `00:17:04` with `177:41:03` CPU time and 936 GiB peak aggregate memory. Per-dataset runtime was `617/686/834/866/908/1004 s` at min/median/p90/p95/p99/max. The 880 outputs occupy about 750 MiB and 4,401 inodes. This validates the farm architecture, but recheck Scratch inode headroom before repeating it (47.93k at verification).

## Operational checks

- Before submit: correct branches/commit, clean intended tree, working venv, dataset count/dry run, output path, two-task smoke test.
- Multi-node workers cannot see another node's `$PBS_JOBFS`. Put immutable code snapshots on shared Scratch/gdata (or explicitly stage on every node), and smoke-test imports on one process per node before the farm. CML confirmation178753573 exposed this: head-node workers ran while other nodes failed to import `scripts`; the shared-snapshot launcher and validated resume preserve completed cases.
- During/after: `qstat -swx`, task status/log archive, output completeness, PBS CPU/memory efficiency, `nci_account`, Scratch inode use.
- Do not scale a scientifically unvalidated generator merely because compute is available.

Sources: live `qmgr`, `nci_account`, filesystem and module queries; NCI [connecting](https://opus.nci.org.au/spaces/Help/pages/230491359/Connecting+to+Gadi), [file transfer](https://opus.nci.org.au/spaces/Help/pages/236880317/File+Transfer), [job submission](https://opus.nci.org.au/spaces/Help/pages/236880320/Job+Submission), [queue structure](https://opus.nci.org.au/spaces/Help/pages/236880996/Queue+Structure+on+Gadi), [queue limits](https://opus.nci.org.au/spaces/Help/pages/236881198/Queue+Limits), and [nci-parallel](https://opus.nci.org.au/spaces/Help/pages/248840680/Nci-parallel) documentation.

# USyd Physics cluster context

Verified live on 2026-08-28. This is a small USyd Physics PBS Pro cluster; its access controls and capacity are unrelated to NCI allocations.

## Access, repositories and storage

- Login: `wedi0306@headnode.physics.usyd.edu.au`; home: `/suphys/wedi0306`. Public-key SSH works directly from the current workstation. Physics' published off-site entry point is `gateway.physics.usyd.edu.au`, and direct hosts may be IP-filtered; gateway key/host setup has not been verified here.
- Public-key SSH from the headnode to compute hosts is not configured. Inspect and run work through PBS; do not bypass the scheduler.
- Main repo: `/suphys/wedi0306/mts-spi-study-cluster -> /import/taiji1/wedi0306/mts-spi-study-cluster`; pyspi repo: `/suphys/wedi0306/pyspi-fork`. The main repo's `.venv` and `data` therefore live on `/import/taiji1` and currently use 5.4 GiB and 11 GiB.
- Home is shared NFS with a per-user quota: 42.8/51.2 GiB soft (52.2 GiB hard), 207k/500k files soft. Use `quota -s`; filling home can prevent login.
- The only user data root currently visible is `/import/taiji1/wedi0306`, on a cross-mounted NFS filesystem with 7.5 TiB globally free (93% used). No per-user quota or backup guarantee is exposed; ask Physics Support and treat it as non-backed-up until confirmed.
- Keep small durable files in home/Git and large environments, data and results on an assigned `/import/<disk>/<user>` data disk. A stable symlink may expose storage where code expects it, but verify with `readlink -f`, `df -h` and `quota -s` before a large write.

## PBS resources and account access

- PBS Pro 23.06; omitting `-q` enters `defaultQ`, which routes to the CPU-only `physics` queue. Server defaults are 1 CPU, 1 GiB and 1 hour; `physics` defaults to 2 hours. Always request CPU, total memory and walltime explicitly. Maximum array size is 950.
- `defaultQ`/`physics` can schedule this account on CPU portions of `nodegpu01`, `nodegpu02` and `h100g01`. The user-wide running cap is 48 CPUs; GPU requests are prohibited in this queue.
- `l40s` is open to this account on two 64-CPU, ~247-GiB hosts with one L40S GPU each. The queue permits at most one running job and 48 CPUs per user, with a 72-hour maximum; the user's queued-GPU cap is one even though the queue-level per-job maximum is two.
- `h100` is open to this account on one 96-CPU, ~247-GiB host with one H100 GPU. The queue permits at most one running job, 48 CPUs and one GPU per user, with a 72-hour maximum.
- `taiji` explicitly lists this account and targets `taiji01` (32 CPUs, ~247 GiB). No user CPU cap or maximum walltime is published in its queue attributes; confirm unusually large/long requests with Physics Support.
- `cmt` (two 168-CPU, ~1-TiB nodes) and the `jasper` route (one 168-CPU, ~1.5-TiB/GPU node) are visible but do not list this account. Their execution queues are route-only, so do not treat visibility as access.
- There is no visible SU budget or `nci_account` equivalent. Limits are queue/user caps; fair-share/accounting policy is not exposed to users.

## Environment and operation

- Main branch `refactor-lagged-warping` and pyspi branch `v3` track origin. The Python 3.12.12 environment imports editable `src` from the main repo and `pyspi` from `/suphys/wedi0306/pyspi-fork`; preserve the existing untracked configs, logs and benchmark outputs.
- The login node has external network access and environment modules including Intel oneAPI, OpenMPI, NVIDIA HPC SDK, scientific libraries and PBS. Compute-node external network access is unverified; stage dependencies before submission.
- Use `qsub`, `qstat -swx <jobid>` and `qdel <jobid>`. Pin BLAS/OpenMP threads to the CPUs requested. Use `jobs/physics/run.pbs` for one dataset and `jobs/physics/run_array.pbs` for arrays; smoke-test before scaling.

Sources: live `qmgr`, `qstat`, `pbsnodes`, filesystem, quota, SSH and module queries; Physics [network and storage FAQ](https://www.physics.sydney.edu.au/computing/faq.html).
