#!/bin/bash
set -euo pipefail

config="${1:?usage: $0 EXPERIMENT_CONFIG}"
test -f "$config"

module purge
module load python3/3.12.1
source .venv/bin/activate

total=$(python -m src.run_experiments --experiment-config "$config" --count-only)
start="${START_INDEX:-1}"
end="${END_INDEX:-$total}"
(( start >= 1 && end >= start && end <= total ))

tasks=$((end - start + 1))
ncpus="${NCPUS:-$(( tasks < 48 ? tasks : 48 ))}"
if (( ncpus > 48 && ncpus % 48 != 0 )); then
    echo "NCPUS above one normal node must be a multiple of 48" >&2
    exit 2
fi

mem_per_cpu_gb="${MEM_PER_CPU_GB:-4}"
mem_gb=$((ncpus * mem_per_cpu_gb))
walltime="${WALLTIME:-01:00:00}"
jobfs_gb="${JOBFS_GB:-10}"
pyspi_config="${PYSPI_CONFIG:-configs/pyspi/benchmarked_p90.yaml}"
test -f "$pyspi_config"
task_timeout="${TASK_TIMEOUT:-}"

variables="EXPERIMENT_CONFIG=$config,START_INDEX=$start,END_INDEX=$end,PYSPI_CONFIG=$pyspi_config"
if [[ -n "$task_timeout" ]]; then
    variables+=",TASK_TIMEOUT=$task_timeout"
fi

echo "[INFO] config=$config datasets=$start..$end/$total ncpus=$ncpus mem=${mem_gb}GB walltime=$walltime task_timeout=${task_timeout:-none}" >&2
qsub \
    -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime,jobfs=${jobfs_gb}GB" \
    -v "$variables" \
    jobs/gadi/run_dataset_farm.pbs
