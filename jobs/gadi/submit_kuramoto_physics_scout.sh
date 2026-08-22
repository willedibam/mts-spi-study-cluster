#!/bin/bash
set -euo pipefail

config="${1:?usage: $0 SCOUT_CONFIG}"
test -f "$config"
module purge
module load python3/3.12.1
source .venv/bin/activate
total=$(python scripts/kuramoto_order_parameter_scout.py --config "$config" --count-only)
start="${START_INDEX:-1}"
end="${END_INDEX:-$total}"
(( start >= 1 && end >= start && end <= total ))
tasks=$((end - start + 1))
ncpus="${NCPUS:-$(( tasks < 48 ? tasks : 48 ))}"
if (( ncpus > 48 && ncpus % 48 != 0 )); then
    echo "NCPUS above one normal node must be a multiple of 48" >&2
    exit 2
fi
mem_gb="${MEM_GB:-$ncpus}"
walltime="${WALLTIME:-00:30:00}"
qsub -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime" \
    -v "SCOUT_CONFIG=$config,START_INDEX=$start,END_INDEX=$end" \
    jobs/gadi/run_kuramoto_physics_scout.pbs
