# Cluster context

**Account sections are pre-filled and verified — treat them as current until the stated date ages out.** **Repo sections are `<placeholders>` — fill them on drop-in.** Anything that would stay true for a different repo on the same cluster belongs in an account section; anything that changes with the checkout, environment or workload belongs in a repo section. Re-date a section in the session you change it; when a live query contradicts this file, fix the file. 

## Fill on drop-in

Repo path on each cluster; symlink targets for `.venv`/`data`/`logs`; tracked branches; environment contents and smoke test; job scripts; artifact policy; measured cost. Everything else is already here.

---

# NCI Gadi — account verified 2026-09-22

## Access and storage

- Login `we2614@gadi.nci.org.au`, project `ql44`, home `/home/562/we2614`. Passwordless key SSH works unattended. `~/.ssh/config` defines host `gadi.nci.org.au` (no short alias, and **no `gadi-dm` entry** — add one before relying on it).
- Transfers use `gadi-dm.nci.org.au`, never a login or compute node. Login nodes have external network; **standard compute nodes do not** — sync Git and stage dependencies before submission, or use `copyq`.
- Roots: `/scratch/ql44/we2614` (working, **100-day expiry**), `/g/data/ql44/we2614` (retained), `mdss` (tape, for bulky archives).
- Live quota (`lquota`, `nci_account`), 2026-09-22:

  | Filesystem | Bytes | Inodes |
  |---|---|---|
  | scratch | 176.08 GiB / 1.00 TiB | **192,637 / 202,000** (hard 212,100) |
  | gdata | 34.03 GiB / 100 GiB | **68,857 / 70,000** (hard 73,500) |

- **Inodes bind on this account long before bytes.** Check both before any farm. Typical composition: bulk data ~3 inodes per unit of work (`.npz`+`.json`+`.npy`), each Python venv ~40k, `uv` cache ~39k, and other projects' checkouts on the same scratch count against this quota.
- `UV_CACHE_DIR=/scratch/ql44/we2614/uv-cache` (set in `~/.bashrc`). `uv cache clean` is safe for existing venvs — wheels are hardlinked in, so the venv survives — but it reclaims mostly **inodes, not bytes**: the 2026-09-21 clear freed 8,075 inodes and only 0.06 GiB despite uv reporting 5.4 GiB. Rebuild caches from a login node; compute nodes cannot fetch wheels.
- Jobs must declare filesystems: `#PBS -l storage=gdata/ql44+scratch/ql44`. An omitted mount appears as a missing path at runtime, not a submission error.
- Beyond the scheduler: `lquota`, `nci_account`, `nci-file-expiry`, `nci-files-report`, `nqstat`, `mdss`, `netcp`/`netmv`, `switchproj` (all in `/opt/nci/bin`).

## Queues, nodes and charging

- Budget `nci_account`, 2026.q3: **124.51 KSU granted, 85.38 used, 39.13 available.** The grant is revised within a period — re-read it, never subtract remembered spend.
- Charging: walltime × max(requested cores, memory-equivalent cores). PBS must reserve the requested maximum before starting, so oversizing delays as well as costs. Above one node, requests round to whole nodes.
- **`server max_array_size = 10`** — arrays are not viable parallelism here. `normal` allows 1,000 queued jobs/project. Use a task farm instead.
- CPU node inventory (`pbsnodes -a`): 3,274 × 48c (Cascade Lake, `normal`, 192 GB / 190 GB requestable, 4 GB/core, 2 SU/resource-hour); 732 × 104c (Sapphire Rapids, `normalsr`); 814 × 28c (Broadwell, `normalbw`); 203 × 32c (`normalsl`). **`normal` is the default, not automatically the best fit** — `normalsr` more than doubles cores per node.
- Modules: `python3/3.12.13` (3.9.2–3.12.13 available), `nci-parallel/1.0.0`, `cuda/12.8.0`, `pytorch/2.12.0`. Load explicit versions and record them for production runs.

## GPUs

| Queue | Nodes | Per node | Accessible |
|---|---|---|---|
| `gpuvolta` | 160 | 48c, 384 GB, 4× V100 32 GB | yes |
| `gpuvolta` (bigmem) | 30 | 48c, 1 TB, 4× V100 32 GB | yes |
| `dgxa100` | **2** | 128c, 2 TB, 8× A100 80 GB | yes |
| `gpursaa` / `analysis` | 4 / 2 | 56c / 40c | other projects' |

No ACL block on `gpuvolta`/`dgxa100` for `ql44`. **GPU charge rates and walltime caps are not exposed by `qstat -Qf` — check NCI docs before committing budget.** V100 is Volta: no bf16, no FlashAttention-2, so it is a poor fit for modern LLM training; A100 capacity is 16 GPUs cluster-wide and contested. Start on CPU unless the code demonstrably uses the GPU.

## Task-level parallelism

- One multi-node allocation driven by `nci-parallel`, chosen because the array cap (10) and job-count caps rule out per-task jobs. Each command handles one unit of work; **pin BLAS/OpenMP to one thread** or the farm oversubscribes every core.
- Keep a farm homogeneous in expected cost — dynamic dispatch approaches full concurrency for homogeneous minute-scale tasks, but a heterogeneous farm pays for idle cores across the tail. Split distinct cost classes.
- Progression: 2 tasks / 2 cores, then one full node, then production width from measured variance, memory, queueing and remaining SU. Maximum concurrency is not minimum time-to-result, and an unvalidated workload should not be scaled because compute is available.
- Can be generous in allocating nodes and codes, e.g. hundreds or even thousands if required.
- Set a per-task timeout only after measuring the tail; leave it unset for timing scouts. Resubmission must skip completed outputs.
- Keep farm artifacts numeric and compact. No plots, wide tables or per-task progress streams — they corrupt logs and burn the inode budget.
- **Multi-node traps (both cost real jobs):** `$PBS_JOBFS` is node-local and invisible to other nodes, so code and data snapshots must sit on shared scratch/gdata or be staged per node; and imports succeeding on the head node can fail on the others — smoke-test one process per node before launching.

## This repo on Gadi

- Checkout `<path>`; dependency repos `<paths>`. Links: `.venv -> <path>`, `data -> <path>`, `logs -> <path>` (confirm with `readlink -f`; a symlink does not inherit quota, expiry or backup policy).
- Branches: `<repo> -> <branch>`, each verified by resolved commit before a production submission. Local working copy is authoritative: commit, push, then fast-forward on the cluster — never pull over uncommitted work there.
- Environment `<interpreter>`, `<editable installs and sources>`. After any rebuild: required imports, fast tests, one end-to-end smoke run. An environment that imports is not a validated environment.
- Launcher `<submit script>`, worker `<job script>`; defaults `<config, cores/task, resume, log and status paths>`.
- Known-bad components: `<component>` disabled because `<failure, job ID, measurement>`.

### Measured cost

| Task class | Job ID | Date | Median | p90 | Max | Peak memory | Resource rule |
|---|---|---|---|---|---|---|---|
| `<class>` | `<id>` | `<date>` | `<t>` | `<t>` | `<t>` | `<GB over n tasks>` | `<cores, memory per task>` |

Run of record: `<job, width, walltime, CPU time, peak memory, output bytes and inodes>`.

---

# USyd Physics — inherited, last verified 2026-08-28

**Not re-verified on 2026-09-22: `headnode.physics.usyd.edu.au:22` timed out from off-campus, consistent with direct hosts being IP-filtered.** Published off-site entry is `gateway.physics.usyd.edu.au`; gateway key/host setup is unverified. Re-confirm everything below before relying on it.

- Login `wedi0306@headnode.physics.usyd.edu.au`, home `/suphys/wedi0306` (shared NFS, `quota -s`, ~51 GiB soft / 500k files — a full home blocks login). `~/.ssh/config` defines `headnode`, `cartman`, `karl`.
- Data disk `/import/taiji1/wedi0306`, cross-mounted NFS, no exposed per-user quota and **no backup guarantee** — treat as non-backed-up. Keep small durable files in home/Git, environments and data on the data disk.
- No key SSH from headnode to compute hosts; run everything through PBS.
- PBS Pro 23.06. Omitting `-q` enters `defaultQ` → CPU-only `physics`. Server defaults 1 CPU / 1 GiB / 1 hour — always state CPU, memory and walltime. **Max array size 950**; split larger ranges without renumbering.
- Queues for this account: `physics` (48-CPU user cap, no GPUs); `l40s` (2 × 64c/247 GB/1× L40S, 1 running job, 72 h); `h100` (1 × 96c/247 GB/1× H100, 1 running job, 72 h); `taiji` (`taiji01`, 32c/247 GB, no published caps). `cmt` and `jasper` are visible but not accessible — visibility is not access.
- No SU budget or accounting equivalent; limits are queue/user caps.
- Login node has external network and modules (Intel oneAPI, OpenMPI, NVIDIA HPC SDK, PBS); **compute-node network access unverified** — stage dependencies.

## This repo on Physics

- Checkout `<path>`; links `<.venv, data, logs targets>`; branches `<repo -> branch>`; environment `<interpreter, editable installs>`; job scripts `<one-task and array templates>`.

---

# Operational checks

- **Before submit:** branches and resolved commits correct; intended tree clean; environment imports and smoke-tests; task count confirmed by dry run; output path writable; two-task smoke test passed; byte **and** inode headroom checked.
- **During/after:** `qstat -swx <id>`; task status and log archive; output completeness against expected count; PBS-reported CPU and memory efficiency; `nci_account`; storage bytes and inodes.

Sources: live `nci_account`, `lquota`, `qmgr`, `qstat`, `pbsnodes`, module and filesystem queries (Gadi, 2026-09-22); Physics section inherited from 2026-08-28. NCI [queue structure](https://opus.nci.org.au/spaces/Help/pages/236880996/Queue+Structure+on+Gadi), [queue limits](https://opus.nci.org.au/spaces/Help/pages/236881198/Queue+Limits), [nci-parallel](https://opus.nci.org.au/spaces/Help/pages/248840680/Nci-parallel); Physics [network and storage FAQ](https://www.physics.sydney.edu.au/computing/faq.html).
