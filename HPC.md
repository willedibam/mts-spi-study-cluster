# Running on the clusters

How this repository runs on NCI Gadi and the USyd Physics cluster: what is fixed, how to submit, and how to size a request. It holds rules and stable facts only. Balances, quotas and queue limits change, so query them (`nci_account`, `lquota`, `qstat -Qf`); what a particular run did is in its `meta.json` and the Git history, not here.

## Source and environment

- **Source moves one way: local, then `origin`, then the cluster.** Commit and push locally; on a cluster only `git pull --ff-only`. Never edit tracked files there. The trunk is `main`.
- `jobs/gadi/submit_farm.sh` refuses a checkout with uncommitted tracked changes, an unpushed commit, or one behind its remote (`jobs/gadi/require_synced_source.sh`). A source snapshot under `operations/sources/` (not a Git checkout, named by commit) passes and is the way to freeze code for a long run. `ALLOW_UNSYNCED_SOURCE=1` overrides for a deliberate one-off.
- **One environment definition.** `uv.lock` is the environment on every machine (Python 3.12; pyspi is the editable sibling checkout `../pyspi-fork`). After a dependency change: `uv lock` locally, commit, then `uv sync --locked --inexact` in each checkout. `aeon` cannot be locked with pyspi 3.0's numpy; use a separate environment for it.
- **pyspi.** The shared checkout stays on `v3`, the 3.0 release branch. A pyspi development branch is used per job with `PYSPI_SOURCE=<snapshot>`, never by switching the shared checkout; `meta.json` records which pyspi ran.
- **Never extend a corpus with a different pyspi computation version** (`pyspi.COMPUTATION_VERSION`). The runner treats such a recording as incomplete and `--audit` fails on a mixed corpus; give a re-run a new `base_output_dir`.

## Running a corpus

A corpus is one YAML file: generated (`mts_classes:`) or a prepared archive (`source:`). One runner and one launcher serve both.

```bash
python -m mtsspi.run_corpus --config corpus.yaml --count-only        # also --list, --validate-source, --audit
python -m mtsspi.run_corpus --config corpus.yaml --job-index 1 --n-jobs 1 --dry-run
# locally there is no farm: one recording per call, so loop over 1..count
for i in $(seq 1 "$(python -m mtsspi.run_corpus --config corpus.yaml --count-only)"); do
    python -m mtsspi.run_corpus --config corpus.yaml --job-index "$i"; done

# Gadi: one recording per core
bash jobs/gadi/submit_farm.sh corpus.yaml [index-file]
# several pyspi workers per recording, with the pyspi that shares each SPI's pairs between them
PYSPI_SOURCE=$MTS_SCRATCH/operations/sources/pyspi-pair-sampling-fd0de71 \
CORES_PER_RECORDING=8 MEM_GB=<recordings x peak> bash jobs/gadi/submit_farm.sh corpus.yaml
# dyad sampling: the same, with `pair_tolerance: 0.03` in the YAML (that pyspi has compute_pairs)
```

Other options are environment variables: `START_INDEX`/`END_INDEX`, `NCPUS`, `WALLTIME`, `TASK_TIMEOUT`, `WORKERS`. `python -m scripts.select_dataset_indices` writes an index file for a subset of a generated corpus. `jobs/gadi/env.sh` holds the modules, thread settings and the storage roots `$MTS_SCRATCH` and `$MTS_GDATA`, which configs may use in paths.

- **A farm is one PBS allocation** in which `nci-parallel` gives each free core group the next recording. Do not submit thousands of single-core jobs or PBS arrays.
- **Keep a farm homogeneous** in `M`, `T` and SPI config, and submit size classes separately: one oversized recording can exhaust the job's memory and kill every recording in it.
- **Book only the cores in use.** `NCPUS` is the recordings running at once times `CORES_PER_RECORDING`, which is the launcher's default: leave `NCPUS` unset, and never round a job of fewer than 48 cores up to a node. A one-recording job asks for its own workers, not a node. Cores beyond that sit idle until the job ends.
- **Workers or not.** One worker per recording is the most efficient use of a core. Use several when one worker would not finish a recording in good time or when cores exceed recordings; choose the number for run time and by size, never from memory. With the shared `v3` pyspi one worker computes a whole SPI and nothing is gained beyond 8. With `PYSPI_SOURCE=.../pyspi-pair-sampling-fd0de71` the pairs of each pair-loop SPI are shared between the workers, with no flag: a full table scales to a whole node, a sampled one gains little beyond 16. Results do not depend on the worker count.
- **Dyad sampling** only for large `M` (about 100 and up): the default budget caps near 2,500 dyads and below `M` of about 70 it is nearly every dyad. It is representative of the whole matrix, not of individual channels or small groups. `spi_mpis.npz` then has NaN at unsampled entries and `pairs.npz` lists the dyads.
- **Memory is per recording, not per core.** Size `MEM_GB` as the recordings running at once times the measured peak of one, plus a quarter. The measured peak is `job.peak_memory_mib.job` in the `meta.json` of a one-recording job: the whole PBS job's peak, the figure `qstat -xf` reports as `resources_used.mem` (recordings made before 14 October 2026 lack it; use `qstat`). The `parent` and `largest_worker` entries are single processes; parent plus workers times largest is an upper bound two to four times actual use (100 GiB against 28 at 12 workers and 36 against 14 at 4, `M=86`, `T=4096`), because only the worker on an all-channel SPI is large. Sampling does not reduce memory.
- **Threads.** Keep the `*_NUM_THREADS=1` settings from `env.sh`. The runner refuses several workers without them, or more workers than the cores it is bound to.
- **Outputs are numeric and few**: `spi_mpis.npz`, `meta.json`, and for generated corpora `timeseries.npy` and any ground truth. Scratch and gdata inode quotas, not bytes, are the usual limit; check `lquota` before a large farm.

## Sizing a request

Progression: a two-recording smoke on the tiny archive (`$MTS_GDATA/pyspi-optimisation/farm-smoke-261013/tiny.npz`, about 90 s) to check that the configuration runs; one representative recording when time or memory is unknown; then the farm. Prefer a generous memory request to a long smoke.

**Workers per recording** for full tables with the `fd0de71` pyspi. When recordings outnumber the cores, 1 each. Otherwise fix them by size and do not re-measure with more: 8 up to `M=64`, 12 up to about `M=128`, 24 near `M=243`, and 48 only for a lone recording near `M=306`. Above about 16 a recording gains less per worker and holds more memory, so run a job's recordings side by side (4 recordings of 12 workers on 48 cores) before giving one of them more. Measure one recording of each size class at its longest `T` first, with those workers, for its wall time and PBS memory; then release the rest in one job. `WALLTIME` is twice the measured time.

**Workers, memory and charge.** A job is charged 2 SU per hour times the larger of its requested cores and its requested memory over 3.96 GiB, on the request and not on use.

- **Workers:** at least the requested memory over 4 GiB, since those cores are charged anyway. More buys time at a modest extra cost: from 4 to 12 workers at `T=4096` a recording ran 2.1 to 2.5 times faster for between 39% less and 19% more SU.
- **Memory request:** 1.25 times PBS's measured peak at the worker count you will run. For another worker count add 1 to 2 GiB per extra worker at `T=4096`, about 0.2 GiB at `T=1000`.
- **Packing a node:** recordings at once is the smaller of 48 over the workers and 190 GiB over the requested memory. At 12 workers that is four recordings at `M=86` (35 GiB each) and three at `M=107` (52 GiB each).
- **Useful workers:** a recording gains nothing beyond its summed SPI time over its longest unsplit task. For full tables that is 80 to 125 workers at `T=4096` (`M=48` to `107`) and about 40 at `M=64`, `T=1000`; for a sampled `M=243` run it is 12.
- **Idle bookings cost the most:** a recording that needs 29 SU on 4 workers is billed 320 when its job books 48 cores.

Measured on Gadi `normal` (Cascade Lake, 48 cores and 190 GiB per node, 4 GiB per core for charging), p90 (283 SPIs in the `fd0de71` rows, 289 in the others), `T=1000`, single runs (expect tens of percent between nodes):

| Recording | Workers | Wall time | Peak memory |
| --- | ---: | ---: | ---: |
| `M=20` | 1 | about 10 min (maximum 17) | under 4 GiB |
| `M=32`, or `M=20, T=2000` | 1 | about 30 min | under 8 GiB |
| `M=64`, shared `v3` pyspi | 1 / 4 / 8 / 16 | 40 / 11 / 6 / 5 min | 2.7 / 5.1 / 5.9 / 6.2 GiB |
| `M=64`, `fd0de71` pyspi | 1 / 4 / 8 / 16 / 48 | 41 / 10 / 5.0 / 2.6 / 1.3 min | 4 / 4 / 5 / 7 / 14 GiB |
| `M=243`, 2,305 dyads (`pair_tolerance: 0.03`) | 4 | 16 min | 28 GiB |
| `M=243`, 1,000 dyads | 1 / 4 / 8 | 52 / 9 / 5 min | 20 / 42 / 39 GiB |
| `M=243`, 1,000 dyads, `fd0de71` pyspi | 16 / 48 | 3.6 / 3.6 min | 72 / 81 GiB |
| `M=48`, `T=4096`, `fd0de71` pyspi | 1 / 3 | 258 / 87 min | 5 / 7 GiB |
| `M=86`, `T=4096`, `fd0de71` pyspi | 4 / 12 | 201 / 87 min | 14 / 28 GiB |
| `M=107`, `T=4096`, `fd0de71` pyspi | 4 | 342 min | 27 GiB |

The `M=20` and `M=32` rows predate two kernel speed-ups and are upper bounds. A full `M=243` table has not been run on the current code (19 h on one core before them).

## Gadi

- Login `we2614@gadi.nci.org.au`, project `ql44`. Checkouts: `/home/562/we2614/mts-spi-study-cluster` and sibling `pyspi-fork`. `.venv`, `data` and `logs` in the checkout are links into `/scratch/ql44/we2614/mts-spi-study/`.
- Storage follows the [storage layout](docs/operations/gadi-storage-layout.md): Scratch for working files, gdata for retained outputs and archives. Read it before adding or moving cluster artefacts.
- Use `gadi-dm.nci.org.au` for `scp`/`rsync`. Login nodes are for light management; compute nodes have no external network, so synchronise Git before submitting.
- A job that reads gdata needs `#PBS -l storage=gdata/ql44+scratch/ql44`; an omitted mount appears as a missing path inside the job.
- A multi-node farm cannot see another node's `$PBS_JOBFS`: code and environments must be on shared storage. The launcher checks imports on every node before dispatch.
- The project going over a storage quota holds its jobs. Check `lquota` when a job is held without an obvious reason.
- During and after a run: `qstat -swx`, the farm's status file under `logs/`, `--audit` for completeness, and `nci_account`. Audit with the `PYSPI_SOURCE` the run used (`PYSPI_SOURCE=... source jobs/gadi/env.sh`): the stored identity includes the pyspi source path, so a snapshot run audited against the shared checkout reports every recording incomplete.

## USyd Physics

- Login `wedi0306@headnode.physics.usyd.edu.au` (may be unreachable off campus; the published entry point is `gateway.physics.usyd.edu.au`). Checkouts: `~/mts-spi-study-cluster` (a link to `/import/taiji1/wedi0306/`) and `~/pyspi-fork`.
- Home is quota-limited shared NFS; keep environments, data and results on `/import/taiji1/wedi0306`, which has no visible quota or backup guarantee.
- PBS Pro. Always request CPUs, total memory and walltime explicitly. `defaultQ` routes to the CPU queue with a 48-CPU running cap per user and a 950-index array limit; `l40s`, `h100` and `taiji` are open to this account, `cmt` and `jasper` are not.
- One launcher: `qsub -v CONFIG=<corpus.yaml>,INDEX=1 jobs/physics/run.pbs`, or `qsub -J 1-100 -v CONFIG=<corpus.yaml> jobs/physics/run.pbs`. Run work through PBS; do not use compute hosts directly.
