# NCI Gadi context

Verified live on 2026-08-22. Recheck queue/allocation values before a production submission.

## Account and layout

- Login: `we2614@gadi.nci.org.au`; project/default group: `ql44`.
- Repositories: `/home/562/we2614/mts-spi-study-cluster` and sibling `../pyspi-fork`.
- Main links: `.venv -> /scratch/ql44/we2614/venvs/mts-spi-v3-631de27`, `data -> /scratch/ql44/we2614/mts-spi-data`, `logs -> /scratch/ql44/we2614/mts-spi-logs`.
- Allocation: Scratch 1 TiB / 202k inodes; gdata 100 GiB / 70k inodes. Python environments consume tens of thousands of inodes, so check both byte and inode headroom before a large farm. Keep durable code in Git; Scratch is working storage.
- Passwordless public-key SSH is configured. Compute nodes have no internet; synchronise Git from a login node before submission.

## Git and environment

- Local repositories are authoritative during development. Commit the intended files, push, then fetch/fast-forward the matching Gadi branches; never pull over uncommitted work.
- Gadi tracks main-repo branch `refactor-lagged-warping` and pyspi branch `v3`. Verify both commits before each production submission.
- The active Scratch Python 3.12 environment has editable main/pyspi-v3 installs. The obsolete broken v2 environment was removed on 2026-08-22. Require `import src, pyspi`, the fast tests, and a one-dataset smoke test after any rebuild.
- These experiments use `configs/pyspi/benchmarked_p90.yaml` (289 SPIs) and one pyspi worker per dataset. The exact-GP additive-noise-model SPI is intentionally disabled: six `M=20,T=1000` tasks each exceeded 18 minutes inside it in job `177018028`.

## Compute allocation and charging

- `nci_account` reports the spendable quarterly allocation. Last live check: 43.8 KSU granted, 17.0 KSU used, 26.8 KSU available for 2026.q3. A confirmed allocation request raises the q3 total to 143 KSU, but it is not spendable until the next NCI allocation sync appears in `nci_account`; do not budget against the email alone.
- `normal` has 48 cores and 192 GiB per Cascade Lake node, 4 GiB/core, 2 SU/core-hour, and a 20,736-core maximum request. Requests above one node use whole 48-core nodes.
- Charge is based on actual walltime and the greater of requested CPU or memory-equivalent cores, but PBS must be able to reserve the requested maximum before starting. A 2,016-core job costs about 4.032 KSU per wall-hour before any memory uplift.
- Live PBS: `max_array_size=10`; `normal` allows 1,000 queued jobs/project, with scheduling thresholds of 300/project and 200/user. These job-count limits are not useful dataset-level parallelism.

## Dataset-level parallelism

- Use one multi-node PBS allocation with `nci-parallel`, not thousands of one-core PBS jobs or the old array wrappers. Each command processes one dataset with `--n-jobs 1`; pin BLAS/OpenMP threads to one.
- `nci-parallel` dynamically assigns the next dataset to a free core. For homogeneous tasks lasting minutes, concurrency can approach the task count (rounded to whole 48-core nodes above one node); for heterogeneous tasks, use measured runtime tails to avoid paying for many idle cores near completion.
- Keep farms reasonably homogeneous in expected M/T/config cost; use separate farms or index ranges when task classes have materially different runtimes.
- Default progression: 2-dataset/2-core smoke test; representative 48-core node test; then choose 192, 480, 960, 2,016, or more cores from measured runtime variance, memory, queueing and remaining KSU. Maximum concurrency is not automatically minimum time-to-result.
- Use `jobs/gadi/submit_dataset_farm.sh`; its PBS worker is `jobs/gadi/run_dataset_farm.pbs`. The launcher defaults to p90, one core/task, no CSV/heatmaps, resumable `--skip-existing`, and persistent job log/status files.
- Set `TASK_TIMEOUT` for homogeneous production farms after measuring the representative runtime tail; the status file identifies timed-out indices, and a resubmission skips completed outputs via `--skip-existing`. Leave it unset for heterogeneous timing scouts.
- Batch pyspi runs keep Calculator INFO/progress output off; warnings, errors and per-SPI timings remain in metadata. Do not multiplex hundreds of progress bars through `nci-parallel`.
- Keep generated artifacts numeric: `timeseries.npy`, compressed `spi_mpis.npz`, compressed `ground_truth.npz`, and small JSON/log files. Do not generate heatmaps or CSV tables for farms.
- Measured `M=20,T=1000` p90 pilot `177019354`: median `603 s`, maximum `774 s` per dataset; six tasks peaked at 11.4 GB total. Keep 4 GB/core until the larger `M,T` scout is measured.

## Operational checks

- Before submit: correct branches/commit, clean intended tree, working venv, dataset count/dry run, output path, two-task smoke test.
- During/after: `qstat -swx`, task status/log archive, output completeness, PBS CPU/memory efficiency, `nci_account`, Scratch inode use.
- Do not scale a scientifically unvalidated generator merely because compute is available.

Sources: live `qmgr`, `nci_account`, filesystem and module queries; NCI [queue structure](https://opus.nci.org.au/spaces/Help/pages/236880996/Queue+Structure+on+Gadi) and [nci-parallel](https://opus.nci.org.au/spaces/Help/pages/248840680/Nci-parallel) documentation; USyd [embarrassingly parallel Gadi guide](https://sydney-informatics-hub.github.io/usyd-gadi-onboarding-guide/notebooks/13_parallel_jobs.html).
