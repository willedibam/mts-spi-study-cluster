#!/bin/bash
set -euo pipefail

config="${1:?usage: $0 CORPUS_CONFIG [INDEX_FILE]}"
index_file="${2:-}"
test -f "$config"
[[ -z "$index_file" ]] || test -f "$index_file"

module purge
module load python3/3.12.1
source .venv/bin/activate

total=$(python -m src.run_external_corpus --config "$config" --count-only)
if [[ -n "$index_file" ]]; then
    tasks=$(awk 'NF { n += 1 } END { print n + 0 }' "$index_file")
else
    start="${START_INDEX:-1}"
    end="${END_INDEX:-$total}"
    (( start >= 1 && end >= start && end <= total ))
    tasks=$((end - start + 1))
fi
(( tasks >= 1 ))

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
(( ncpus >= 1 ))

mem_per_cpu_gb="${MEM_PER_CPU_GB:-4}"
mem_gb=$((ncpus * mem_per_cpu_gb))
walltime="${WALLTIME:-01:00:00}"
jobfs_gb="${JOBFS_GB:-10}"

variables="CORPUS_CONFIG=$config"
if [[ -n "$index_file" ]]; then
    variables+=",INDEX_FILE=$index_file"
else
    variables+=",START_INDEX=${START_INDEX:-1},END_INDEX=${END_INDEX:-$total}"
fi
for name in TASK_TIMEOUT WORKERS; do
    if [[ -n "${!name:-}" ]]; then
        variables+=",$name=${!name}"
    fi
done

echo "[INFO] config=$config selected=$tasks/$total ncpus=$ncpus mem=${mem_gb}GB walltime=$walltime" >&2
qsub \
    -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime,jobfs=${jobfs_gb}GB" \
    -v "$variables" \
    jobs/gadi/run_external_corpus_farm.pbs
