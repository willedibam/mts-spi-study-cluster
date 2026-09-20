# HPC quick guide

CPU-first instructions for NCI Gadi and the USyd Physics cluster. Replace values in `<angle brackets>` with your own. Queue limits and storage allocations change; check them before large runs.

## One-time access

You need an account and allocation before SSH will work:

- Gadi: NCI account plus membership of an NCI project.
- Physics: Physics account plus `taiji`/data-disk access. Group members normally inherit these, but confirm with Physics Support if `qsub -q taiji` is rejected.

Create a key on your own computer and install only its public half:

```bash
ssh-keygen -t ed25519
ssh-copy-id <nci-user>@gadi.nci.org.au
ssh-copy-id <physics-user>@headnode.physics.usyd.edu.au
```

Never share or commit the private key. Verify a new server's host-key fingerprint through NCI or Physics Support before accepting it. SSH config is optional, but aliases make interactive and agent access safer and simpler:

```sshconfig
Host gadi
    HostName gadi.nci.org.au
    User <nci-user>
    IdentityFile ~/.ssh/id_ed25519

Host gadi-dm
    HostName gadi-dm.nci.org.au
    User <nci-user>
    IdentityFile ~/.ssh/id_ed25519

Host physics
    HostName headnode.physics.usyd.edu.au
    User <physics-user>
    IdentityFile ~/.ssh/id_ed25519
```

Physics may require off-site users to enter through `gateway.physics.usyd.edu.au`; ask Physics Support whether to add a `ProxyJump`. Test unattended access with `ssh -o BatchMode=yes gadi hostname` and the equivalent Physics alias.

## Layout

| | Gadi | Physics |
|---|---|---|
| Login | `gadi.nci.org.au` | `headnode.physics.usyd.edu.au` |
| Scheduler | PBS Pro | PBS Pro |
| Small files | `$HOME` | `$HOME` (`quota -s`) |
| Working data | `/scratch/<project>/<user>` | `/import/taiji1/<user>` |
| Retained data | `/g/data/<project>/<user>` | Confirm backup/retention with Physics Support |
| Large transfer | `gadi-dm.nci.org.au` | `headnode` or the approved gateway route |
| Allocation check | `nci_account` | Queue limits; no visible SU account |

Gadi Scratch is temporary: files untouched for 100 days enter expiry. Standard Gadi compute nodes have no external network; `copyq` is the exception. Physics login has external network, but compute-node network access should not be assumed.

Keep login nodes for editing, Git, submission and monitoring. Run computation through PBS; do not SSH to compute nodes to bypass the scheduler.

## Storage and symlinks

Large environments, datasets and outputs should live on working/data storage, not home. A symlink can retain a convenient project layout:

```bash
storage_root=/scratch/<project>/<user>/<work>       # Gadi
# storage_root=/import/taiji1/<user>/<work>         # Physics alternative
mkdir -p "$storage_root/data"
ln -s "$storage_root/data" data

readlink -f data
df -h data
```

Do not replace an existing path blindly. On Gadi also run `nci_account`; on Physics run `quota -s`. A symlink does not change the target filesystem's quota, expiry or backup policy.

Use the Gadi data movers for transfers:

```bash
rsync -aP ./results/ gadi-dm:/g/data/<project>/<user>/results/
rsync -aP physics:/import/taiji1/<user>/results/ ./results/
```

## CPU jobs

A minimal Gadi job is:

```bash
#!/bin/bash
#PBS -P <project>
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=01:00:00
#PBS -l jobfs=1GB
#PBS -l storage=scratch/<project>+gdata/<project>
#PBS -l wd
#PBS -j oe

set -euo pipefail
module load python3/<version>
source .venv/bin/activate
python your_program.py
```

Request only filesystems actually used. The charged project's Scratch is implicit, but stating storage explicitly is clearer. Above one `normal` node, CPU requests use whole 48-core nodes. For many independent tasks, prefer one measured multi-node allocation with `nci-parallel`; Gadi arrays are limited to 10 indices.

Use `module avail` to find software and load an explicit version. Record the module and Git versions used for a production run.

Physics repository templates:

- `jobs/physics/run.pbs`: one selected dataset.
- `jobs/physics/run_array.pbs`: one dataset per array index.

Examples:

```bash
qsub -v CONFIG=configs/generate/example.yaml,INDEX=1 jobs/physics/run.pbs
qsub -J 1-100 -v CONFIG=configs/generate/example.yaml jobs/physics/run_array.pbs
```

Physics arrays are limited to 950 indices. Split larger ranges without renumbering them, for example `1-950`, `951-1900`, then the remainder. The runner uses `--skip-existing`, so a repeated index is safe.

## GPUs

Start on CPU unless the code demonstrably benefits from a GPU.

- Physics: use `-q l40s` or `-q h100` and request `ngpus=1`. These queues currently allow one running GPU job per user; group members also inherit `taiji` CPU access.
- Gadi: choose `gpuvolta`, `dgxa100` or `gpuhopper` only after checking current GPU queue limits and matching CPU/GPU ratios.

Example Physics override:

```bash
qsub -q h100 -l select=1:ncpus=4:mem=32gb:ngpus=1 \
  -l walltime=02:00:00 your_gpu_job.pbs
```

The application must explicitly use the GPU; requesting one does not accelerate CPU code.

## Submit and monitor

```bash
qsub job.pbs
qstat -swx <job-id>
qstat -fx <job-id>
qdel <job-id>
```

Before scaling: confirm the Git commit and environment, count/dry-run the workload, run one task, then a small representative batch. Pin BLAS/OpenMP threads to the allocated CPUs and inspect output, runtime and peak memory before requesting more.

Current references: NCI [connecting](https://opus.nci.org.au/spaces/Help/pages/230491359/Connecting+to+Gadi), [file transfer](https://opus.nci.org.au/spaces/Help/pages/236880317/File+Transfer), [job submission](https://opus.nci.org.au/spaces/Help/pages/236880320/Job+Submission), [queue structure](https://opus.nci.org.au/spaces/Help/pages/236880996/Queue+Structure+on+Gadi), and [queue limits](https://opus.nci.org.au/spaces/Help/pages/236881198/Queue+Limits); Physics [network and storage FAQ](https://www.physics.sydney.edu.au/computing/faq.html).
