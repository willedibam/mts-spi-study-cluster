#!/bin/bash
# Generic submit wrapper for jobs/gadi/run_pipeline.pbs.
# Handles Gadi's max_array_size=10 cap by submitting N batches of up to 10 subjobs.
#
# Usage (positional):
#   bash jobs/gadi/submit_pipeline.sh <EXPERIMENT_CONFIG> <TOTAL_DATASETS> [CHUNK_SIZE]
#
# Or env-driven (matches submit_cml_param_sweep.sh style):
#   EXPERIMENT_CONFIG=... TOTAL_DATASETS=... CHUNK_SIZE=... \
#       bash jobs/gadi/submit_pipeline.sh
#
# Examples:
#   # Wave redux at M=10/T=1000 (50 datasets, 1 array, chunk=5):
#   bash jobs/gadi/submit_pipeline.sh \
#       configs/generate/embeddings/proof-benchmarked-wave.yaml 50 5
#
#   # Wave redux full grid (450 datasets, 1 array of 10 subjobs at chunk=45):
#   bash jobs/gadi/submit_pipeline.sh \
#       configs/generate/embeddings/proof-benchmarked-wave-full.yaml 450 45
#
# CONCURRENT_DATASETS (default 48, the node core count) and CORES_PER_DATASET
# (default 1) are independent env knobs — see run_pipeline.pbs for guidance.
# --skip-existing in the PBS script makes resubmission idempotent.
set -euo pipefail

EXPERIMENT_CONFIG="${1:-${EXPERIMENT_CONFIG:?need EXPERIMENT_CONFIG (arg 1 or env)}}"
TOTAL_DATASETS="${2:-${TOTAL_DATASETS:?need TOTAL_DATASETS (arg 2 or env)}}"
CHUNK_SIZE="${3:-${CHUNK_SIZE:-200}}"
SUBJOBS_PER_ARRAY="${SUBJOBS_PER_ARRAY:-10}"
CONCURRENT_DATASETS="${CONCURRENT_DATASETS:-48}"
CORES_PER_DATASET="${CORES_PER_DATASET:-1}"
PBS_SCRIPT="${PBS_SCRIPT:-jobs/gadi/run_pipeline.pbs}"

BATCH_DATASETS=$(( CHUNK_SIZE * SUBJOBS_PER_ARRAY ))
N_BATCHES=$(( (TOTAL_DATASETS + BATCH_DATASETS - 1) / BATCH_DATASETS ))

echo "[INFO] config=$EXPERIMENT_CONFIG"
echo "[INFO] total=$TOTAL_DATASETS  chunk=$CHUNK_SIZE  subjobs/array=$SUBJOBS_PER_ARRAY"
echo "[INFO] concurrent_datasets=$CONCURRENT_DATASETS  cores_per_dataset=$CORES_PER_DATASET"
echo "[INFO] datasets/batch=$BATCH_DATASETS  n_batches=$N_BATCHES"
echo "[INFO] PBS script: $PBS_SCRIPT"

for (( b=0; b<N_BATCHES; b++ )); do
    OFFSET=$(( b * BATCH_DATASETS ))
    REMAINING=$(( TOTAL_DATASETS - OFFSET ))
    SUBJOBS=$SUBJOBS_PER_ARRAY
    NEEDED=$(( (REMAINING + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    if [[ $NEEDED -lt $SUBJOBS ]]; then
        SUBJOBS=$NEEDED
    fi
    echo "[INFO] submitting batch $((b+1))/$N_BATCHES  offset=$OFFSET  subjobs=$SUBJOBS"
    VARS="EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},TOTAL_DATASETS=${TOTAL_DATASETS},CHUNK_SIZE=${CHUNK_SIZE},BATCH_OFFSET=${OFFSET},CONCURRENT_DATASETS=${CONCURRENT_DATASETS},CORES_PER_DATASET=${CORES_PER_DATASET}"
    if [[ $SUBJOBS -le 1 ]]; then
        # PBS rejects -J 1-1; submit as a single non-array job (PBS script defaults PBS_ARRAY_INDEX to 1).
        qsub -v "$VARS" "$PBS_SCRIPT"
    else
        qsub -J "1-${SUBJOBS}" -v "$VARS" "$PBS_SCRIPT"
    fi
done

echo "[INFO] submitted $N_BATCHES array job(s). monitor with: qstat -tu \$USER"
