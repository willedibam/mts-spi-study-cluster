#!/bin/bash
set -euo pipefail

config="${1:?usage: $0 CONFIG}"
test -f "$config"

ncpus="${NCPUS:-12}"
(( ncpus >= 1 && ncpus <= 48 ))
mem_gb="${MEM_GB:-48}"
walltime="${WALLTIME:-02:00:00}"
threads="${THREADS:-$ncpus}"
(( threads >= 1 && threads <= ncpus ))

qsub \
    -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime" \
    -v "ATLAS_CONFIG=$config,THREADS=$threads" \
    jobs/gadi/run_atlas_analysis.pbs
