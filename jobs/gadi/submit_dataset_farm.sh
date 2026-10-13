#!/bin/bash
set -euo pipefail

config="${1:?usage: $0 EXPERIMENT_CONFIG}"
test -f "$config"
source jobs/gadi/require_synced_source.sh

module purge
module load python3/3.12.1
source .venv/bin/activate

# Reject layout drift and accidental output into retired roots before submission.
output_dir=$(python - "$config" <<'PYCONFIG'
import sys
from src.mapping import ExperimentConfig
print(ExperimentConfig.from_file(sys.argv[1]).base_output_dir)
PYCONFIG
)
python scripts/check_gadi_storage_layout.py --output "$output_dir"

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
# pyspi workers per dataset (see run_dataset_farm.pbs): cores are tasks x
# workers, and memory is per dataset, so give MEM_GB from a measured dataset.
cores_per_recording="${CORES_PER_RECORDING:-1}"
[[ "$cores_per_recording" =~ ^[1-9][0-9]*$ ]] || { echo 'CORES_PER_RECORDING must be a positive integer' >&2; exit 2; }
cores=$((tasks * cores_per_recording))
if (( cores <= 48 )); then
    default_ncpus=$cores
else
    default_ncpus=$(( ((cores + 47) / 48) * 48 ))
fi
ncpus="${NCPUS:-$default_ncpus}"
if (( ncpus > 48 && ncpus % 48 != 0 )); then
    echo "NCPUS above one normal node must be a multiple of 48" >&2
    exit 2
fi

mem_per_cpu_gb="${MEM_PER_CPU_GB:-4}"
mem_gb="${MEM_GB:-$((ncpus * mem_per_cpu_gb))}"
walltime="${WALLTIME:-01:00:00}"
jobfs_gb="${JOBFS_GB:-10}"
pyspi_config="${PYSPI_CONFIG:-configs/pyspi/benchmarked_p90.yaml}"
test -f "$pyspi_config"
task_timeout="${TASK_TIMEOUT:-}"
workers="${WORKERS:-}"
if [[ -n "$workers" ]]; then
    (( workers >= 1 && workers * cores_per_recording <= ncpus ))
fi

variables="EXPERIMENT_CONFIG=$config,START_INDEX=$start,END_INDEX=$end,PYSPI_CONFIG=$pyspi_config"
variables+=",CORES_PER_RECORDING=$cores_per_recording"
for name in FILTER_M FILTER_T INSTANCE_MIN INSTANCE_MAX PYSPI_SOURCE; do
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

echo "[INFO] config=$config selected=$tasks/$total ncpus=$ncpus workers=${workers:-$((ncpus / cores_per_recording))} cores_per_recording=$cores_per_recording mem=${mem_gb}GB walltime=$walltime task_timeout=${task_timeout:-none}" >&2
dependency_args=()
[[ -n "${DEPENDENCY:-}" ]] && dependency_args=(-W "depend=$DEPENDENCY")
qsub \
    "${dependency_args[@]}" \
    -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime,jobfs=${jobfs_gb}GB" \
    -v "$variables" \
    jobs/gadi/run_dataset_farm.pbs
