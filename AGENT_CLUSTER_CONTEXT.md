# IDE Agent Cluster Context (USYD Physics)

Last verified: 2026-04-22
Verified from: headnode.physics.usyd.edu.au
Primary user: wedi0306

## 1) Environment identity

- Username: wedi0306
- UID/GID: 1785800272 / 10000
- Groups: linuxusers, theory
- Host role observed: headnode
- Shell umask: 0022
- SELinux context (session): unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023

## 2) Filesystem and storage architecture

### 2.1 Mount points

- Home mount: /suphys/wedi0306
  - Backend: homes.physics.usyd.edu.au:/home1/wedi0306 (nfs4)
  - Capacity snapshot: 8.7T total, 3.6T used, 5.2T free (41% used)
- Bulk mount: /import/taiji1
  - Backend: taiji01.physics.usyd.edu.au:/export/taiji1 (nfs4)
  - Capacity snapshot: 107T total, 101T used, 5.9T free (95% used)

### 2.2 Project path topology

- /suphys/wedi0306/mts-spi-study-cluster is a symlink to:
  - /import/taiji1/wedi0306/mts-spi-study-cluster
- /suphys/wedi0306/pyspi-fork is a regular directory on /suphys (not symlinked)

Implication:
- Heavy paths under mts-spi-study-cluster (for example .venv and data) are already physically on /import/taiji1 due to repo-root symlinking.

### 2.3 Current heavy path locations and size

- /suphys/wedi0306/mts-spi-study-cluster/.venv resolves to /import/taiji1/wedi0306/mts-spi-study-cluster/.venv
  - Size snapshot: about 7.9G
- /suphys/wedi0306/mts-spi-study-cluster/data resolves to /import/taiji1/wedi0306/mts-spi-study-cluster/data
  - Size snapshot: about 12G

### 2.4 Quota and writable checks

- Home quota snapshot (/suphys):
  - Space used: 36017M
  - Soft limit: 51200M
  - Hard limit: 52224M
  - Files used: 174k
  - File soft/hard limits: 500k / 510k
- Write probe succeeded on both:
  - /suphys/wedi0306
  - /import/taiji1/wedi0306

## 3) Permissions and access model

- /suphys/wedi0306
  - Mode: 711 (drwx--x--x)
  - Owner/group: wedi0306:linuxusers
- /import/taiji1/wedi0306
  - Mode: 710 (drwx--x---)
  - Owner/group: wedi0306:linuxusers
- /suphys/wedi0306/pyspi-fork
  - Mode: 755 (drwxr-xr-x)
  - Owner/group: wedi0306:linuxusers
- In mts-spi-study-cluster:
  - .venv mode: 755
  - data mode: 755
  - ACLs: default owner/group/other only, no extended ACL entries observed

## 4) Scheduler architecture (PBS Pro)

### 4.1 Scheduler/tooling presence

- Present: qsub, qstat, pbsnodes, module
- Not present: sinfo, squeue, sacct, sbatch

Conclusion:
- This environment is PBS-first, not Slurm.

### 4.2 Server defaults and policy

From qstat -Bf:
- server_host: headnode.physics.usyd.edu.au
- default_queue: defaultQ
- default_qsub_arguments: -V -q defaultQ
- resources_default.ncpus: 1
- resources_default.mem: 1gb
- resources_default.walltime: 01:00:00
- max_array_size: 950
- acl_user_enable: True

### 4.3 Queue model observed

- defaultQ is a Route queue (routes to physics)
- physics is the primary CPU execution queue
- Other execution queues seen: l40s, h100, cmt, taiji, jasper_cpu, jasper_gpu, jasper_small

Key queue constraints observed:
- physics:
  - resources_max.ncpus=48
  - resources_max.ngpus=0
  - default walltime=02:00:00
- l40s:
  - resources_max.ncpus=48
  - resources_max.ngpus=2
  - resources_min.ngpus=1
  - resources_max.walltime=72:00:00
- h100:
  - resources_max.ncpus=48
  - resources_max.ngpus=1
  - resources_min.ngpus=1
  - resources_max.walltime=72:00:00
- jasper_small:
  - resources_max.ncpus=28
  - resources_max.mem=200gb
  - resources_max.ngpus=0
  - resources_max.walltime=08:00:00
- jasper_cpu and jasper_gpu:
  - from_route_only=True

## 5) Node class/resource observations (pbsnodes)

Observed slice-level resource classes:
- physics/l40s slices: about 16 ncpus and about 63GB mem, with GPU slices exposing ngpus=1 in l40s
- physics/h100 slices: about 24 ncpus and about 63GB mem, with GPU slice exposing ngpus=1 in h100
- cmt slices: about 84 ncpus and about 513GB mem
- taiji slices: about 16 ncpus and about 126GB mem
- jasper slices: about 42 ncpus and about 386GB mem; one GPU-capable slice for jasper_gpu

## 6) Runtime/toolchain defaults

- No modules currently loaded by default
- System python in unactivated shell:
  - /bin/python
  - Python 3.9.25
- uv:
  - ~/.local/bin/uv
  - version 0.10.4

Project venv details:
- venv python: Python 3.12.12
- venv python symlink points to uv-managed CPython binary
- venv pip shim missing (no .venv/bin/pip observed)
- Use uv pip in this environment

## 7) Repo state anchors (at verification time)

- mts-spi-study-cluster:
  - Branch: refactor-lagged-warping
  - Commit: e3a8d77
  - Remote: git@github.com:willedibam/mts-spi-study-cluster.git
  - Working tree: clean
- pyspi-fork:
  - Branch: v2
  - Commit: df84c1b
  - Remote: https://github.com/willedibam/pyspi.git
  - Working tree: clean

## 8) Python package wiring between repos

From editable install metadata in mts venv:
- pyspi import mapping resolves to:
  - /suphys/wedi0306/pyspi-fork/pyspi
- mts package mapping resolves to:
  - /import/taiji1/wedi0306/mts-spi-study-cluster/src

Operational implication:
- Pulling either repo updates runtime behavior immediately (editable install), without reinstall, unless dependency metadata changes.

## 9) Project PBS scripts and policy fit

- Total PBS scripts in repo: 19
- Resource style in physics scripts is generally select=1 with ncpus in low-single digits and mem around 2-24gb.

Array-cap mismatches with server max_array_size=950:
- jobs/physics/run_eeml.pbs uses #PBS -J 1-3000
- jobs/physics/run_proof.pbs uses #PBS -J 1-1350

These should be chunked or split to submit successfully on this cluster.

## 10) Agent operating guidance

- Treat /suphys as quota-constrained home and /import/taiji1 as bulk storage.
- Prefer writing large artifacts, caches, and environments to /import paths.
- For CPU jobs, submit explicitly to physics unless a different queue (e.g. taiji) is intended.
- Ensure PBS array ranges stay <= 950 unless split across multiple submissions.
- Keep BLAS/OpenMP threads pinned when parallelizing pyspi workers (see scripts/activate.sh).
- Rebuild/reinstall venv only when needed:
  - dependency set changed (for example pyproject.toml updates)
  - python version changed
  - environment corrupted

## 11) Known unknowns

- Per-user quota visibility for /import/taiji1 was not exposed by quota command in this session.
- Fairshare/accounting internals were not extracted (admin-level policy view not queried).

## 12) Related project docs

- Workflow reference: [WORKFLOW.md](WORKFLOW.md)
- Python project dependencies: [pyproject.toml](pyproject.toml)
- Activation helper with thread pinning: [scripts/activate.sh](scripts/activate.sh)
