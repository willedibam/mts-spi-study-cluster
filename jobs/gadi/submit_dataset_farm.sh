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

selector=("$config" --start "$start" --end "$end" --count)
[[ -n "${FILTER_M:-}" ]] && selector+=(--M "$FILTER_M")
[[ -n "${FILTER_T:-}" ]] && selector+=(--T "$FILTER_T")
[[ -n "${INSTANCE_MIN:-}" ]] && selector+=(--instance-min "$INSTANCE_MIN")
[[ -n "${INSTANCE_MAX:-}" ]] && selector+=(--instance-max "$INSTANCE_MAX")
tasks=$(python -m scripts.select_dataset_indices "${selector[@]}")
if (( tasks <= 48 )); then
    default_ncpus=$tasks
else
    default_ncpus=$(( ((tasks + 47) / 48) * 48 ))
fi
ncpus="${NCPUS:-$default_ncpus}"
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
workers="${WORKERS:-}"
if [[ -n "$workers" ]]; then
    (( workers >= 1 && workers <= ncpus ))
fi

variables="EXPERIMENT_CONFIG=$config,START_INDEX=$start,END_INDEX=$end,PYSPI_CONFIG=$pyspi_config"
for name in FILTER_M FILTER_T INSTANCE_MIN INSTANCE_MAX; do
    if [[ -n "${!name:-}" ]]; then
        variables+=",$name=${!name}"
    fi
done
if [[ -n "$task_timeout" ]]; then
    variables+=",TASK_TIMEOUT=$task_timeout"
fi
if [[ -n "$workers" ]]; then
    variables+=",WORKERS=$workers"
fi

echo "[INFO] config=$config selected=$tasks/$total ncpus=$ncpus workers=${workers:-$ncpus} mem=${mem_gb}GB walltime=$walltime task_timeout=${task_timeout:-none}" >&2
qsub \
    -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime,jobfs=${jobfs_gb}GB" \
    -v "$variables" \
    jobs/gadi/run_dataset_farm.pbs
