# NCI Gadi context

Verified live on 2026-08-22. Recheck queue/allocation values before a production submission.

## Account and layout

- Login: `we2614@gadi.nci.org.au`; project/default group: `ql44`.
- Repositories: `/home/562/we2614/mts-spi-study-cluster` and sibling `../pyspi-fork`.
- Main links: `.venv -> /scratch/ql44/we2614/venvs/mts-spi`, `data -> /scratch/ql44/we2614/mts-spi-data`, `logs -> /scratch/ql44/we2614/mts-spi-logs`.
- Allocation: Scratch 1 TiB / 202k inodes; gdata 100 GiB / 70k inodes. Scratch currently uses about 71 GiB and 139k inodes. Keep durable code in Git; Scratch is working storage.
- Passwordless public-key SSH is configured. Compute nodes have no internet; synchronise Git from a login node before submission.

## Git and environment

- Local repositories are authoritative during development. Commit the intended files, push, then fetch/fast-forward the matching Gadi branches; never pull over uncommitted work.
- Gadi was on main-repo branch `refactor-lagged-warping` and pyspi branch `v2`; local pyspi development is on `v3`. Verify branches each time.
- The existing Scratch Python 3.12 environment currently has broken editable installs. Rebuild it after repository sync and require `import src, pyspi` plus a one-dataset smoke test before PBS production.
- These experiments use `configs/pyspi/benchmarked_p90.yaml` and one pyspi worker per dataset.

## Compute allocation and charging

- `nci_account` reports the spendable quarterly allocation. Last live check: 43.8 KSU granted, 17.0 KSU used, 26.8 KSU available for 2026.q3. A confirmed allocation request raises the q3 total to 143 KSU, but it is not spendable until the next NCI allocation sync appears in `nci_account`; do not budget against the email alone.
- `normal` has 48 cores and 192 GiB per Cascade Lake node, 4 GiB/core, 2 SU/core-hour, and a 20,736-core maximum request. Requests above one node use whole 48-core nodes.
- Charge is based on actual walltime and the greater of requested CPU or memory-equivalent cores, but PBS must be able to reserve the requested maximum before starting. A 2,016-core job costs about 4.032 KSU per wall-hour before any memory uplift.
- Live PBS: `max_array_size=10`; `normal` allows 1,000 queued jobs/project, with scheduling thresholds of 300/project and 200/user. These job-count limits are not useful dataset-level parallelism.

## Dataset-level parallelism

- Use one multi-node PBS allocation with `nci-parallel`, not thousands of one-core PBS jobs or the old array wrappers. Each command processes one dataset with `--n-jobs 1`; pin BLAS/OpenMP threads to one.
- `nci-parallel` dynamically assigns the next dataset to a free core. Request fewer workers than datasets when runtimes vary, avoiding an expensive long-tail of idle nodes. NCI specifically recommends no more than about 200 concurrent workers for 2,000 heterogeneous tasks; scale beyond that only after a representative timing/memory pilot (and be generous with resource allocation).
- Keep farms reasonably homogeneous in expected M/T/config cost; use separate farms or index ranges when task classes have materially different runtimes.
- Default progression: 2-dataset/2-core smoke test; representative 48-core node test; then choose 192, 480, 960, or more cores (be generous) from measured runtime variance, memory, scheduling and remaining KSU. Maximum concurrency is not automatically minimum time-to-result.
- Use `jobs/gadi/submit_dataset_farm.sh`; its PBS worker is `jobs/gadi/run_dataset_farm.pbs`. The launcher defaults to p90, one core/task, no CSV/heatmaps, resumable `--skip-existing`, and persistent job log/status files.

## Operational checks

- Before submit: correct branches/commit, clean intended tree, working venv, dataset count/dry run, output path, two-task smoke test.
- During/after: `qstat -swx`, task status/log archive, output completeness, PBS CPU/memory efficiency, `nci_account`, Scratch inode use.
- Do not scale a scientifically unvalidated generator merely because compute is available.

Sources: live `qmgr`, `nci_account`, filesystem and module queries; NCI [queue structure](https://opus.nci.org.au/spaces/Help/pages/236880996/Queue+Structure+on+Gadi) and [nci-parallel](https://opus.nci.org.au/spaces/Help/pages/248840680/Nci-parallel) documentation; USyd [embarrassingly parallel Gadi guide](https://sydney-informatics-hub.github.io/usyd-gadi-onboarding-guide/notebooks/13_parallel_jobs.html).
