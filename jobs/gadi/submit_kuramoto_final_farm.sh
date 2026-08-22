#!/bin/bash
set -euo pipefail

config="configs/generate/order_parameter/kuramoto-final-confirmation.yaml"
total=$(python -m src.run_experiments --experiment-config "$config" --count-only)
chunk=48

for ((start=1; start<=total; start+=chunk)); do
    end=$((start + chunk - 1))
    (( end > total )) && end=$total
    START_INDEX=$start END_INDEX=$end NCPUS=48 WORKERS=48 \
      MEM_PER_CPU_GB=4 WALLTIME=01:00:00 \
      bash jobs/gadi/submit_dataset_farm.sh "$config"
done
