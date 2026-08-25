#!/bin/bash
set -euo pipefail

data_path="${1:?usage: $0 DATA_PATH OUTPUT}"
output="${2:?usage: $0 DATA_PATH OUTPUT}"
test -d "$data_path"

ncpus="${NCPUS:-8}"
(( ncpus >= 1 && ncpus <= 48 ))
mem_gb="${MEM_GB:-32}"
walltime="${WALLTIME:-00:30:00}"
feature_contract="${FEATURE_CONTRACT:-unified_ordered_v3}"
case "$feature_contract" in
    unified_ordered_v3|direction_preserving_v2|legacy_symmetrized_v1) ;;
    *) echo "unknown FEATURE_CONTRACT: $feature_contract" >&2; exit 2 ;;
esac
variables="DATA_PATH=$data_path,OUTPUT=$output,WORKERS=${WORKERS:-$ncpus},METRIC=${METRIC:-pearson},FEATURE_CONTRACT=$feature_contract"
if [[ -n "${SPI_SUBSET:-}" ]]; then
    test -f "$SPI_SUBSET"
    variables+=",SPI_SUBSET=$SPI_SUBSET"
fi
if [[ -n "${DATASET_LIMIT:-}" ]]; then
    (( DATASET_LIMIT >= 1 ))
    variables+=",DATASET_LIMIT=$DATASET_LIMIT"
fi
if [[ "${RECOMPUTE:-0}" == 1 ]]; then
    variables+=",RECOMPUTE=1"
fi

qsub \
    -l "ncpus=$ncpus,mem=${mem_gb}GB,walltime=$walltime" \
    -v "$variables" \
    jobs/gadi/run_feature_reconstruction.pbs
