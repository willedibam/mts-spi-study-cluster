#!/bin/bash
set -euo pipefail

total=256
chunk=48
for ((start=1; start<=total; start+=chunk)); do
    end=$((start + chunk - 1))
    (( end > total )) && end=$total
    qsub -v "START_INDEX=$start,END_INDEX=$end" jobs/gadi/run_kuramoto_final_shift_farm.pbs
done
