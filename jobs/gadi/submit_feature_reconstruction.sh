#!/bin/bash
set -euo pipefail

data_path="${1:?usage: $0 DATA_PATH OUTPUT}"
output="${2:?usage: $0 DATA_PATH OUTPUT}"
test -d "$data_path"

ncpus="${NCPUS:-48}"
(( ncpus >= 1 && ncpus <= 48 ))
mem_gb="${MEM_GB:-192}"
walltime="${WALLTIME:-01:00:00}"
variables="DATA_PATH=$data_path,OUTPUT=$output,WORKERS=${WORKERS:-$ncpus},METRIC=${METRIC:-pearson}"
if [[ -n "${SPI_SUBSET:-}" ]]; then
    test -f "$SPI_SUBSET"
    variables+=",SPI_SUBSET=$SPI_SUBSET"
fi
if [[ "${RECOMPUTE:-0}" == 1 ]]; then
    variables+=",RECOMPUTE=1"
fi

qsub \
    -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime" \
    -v "$variables" \
    jobs/gadi/run_feature_reconstruction.pbs
