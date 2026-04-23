#!/bin/bash
# Submit all 2,440 CML param-sweep datasets to Gadi in chunked batches.
# Gadi caps PBS array size at 10, so we submit many arrays and each subjob
# processes CHUNK_SIZE datasets sequentially.
#
# Per batch:  SUBJOBS_PER_ARRAY (=10) subjobs * CHUNK_SIZE (=20) datasets = 200
# Per subjob: CHUNK_SIZE * ~15min worst-case = ~5h (walltime set to 10h margin)
#
# Resubmit-safe: --skip-existing inside the PBS script means re-running this
# wrapper after partial completion only picks up the unfinished datasets.
set -euo pipefail

CHUNK_SIZE="${CHUNK_SIZE:-20}"
SUBJOBS_PER_ARRAY="${SUBJOBS_PER_ARRAY:-10}"
TOTAL_DATASETS="${TOTAL_DATASETS:-2440}"
PBS_SCRIPT="${PBS_SCRIPT:-jobs/gadi/run_cml_param_sweep.pbs}"

BATCH_DATASETS=$(( CHUNK_SIZE * SUBJOBS_PER_ARRAY ))
N_BATCHES=$(( (TOTAL_DATASETS + BATCH_DATASETS - 1) / BATCH_DATASETS ))

echo "[INFO] total=$TOTAL_DATASETS  chunk=$CHUNK_SIZE  subjobs_per_array=$SUBJOBS_PER_ARRAY"
echo "[INFO] datasets_per_batch=$BATCH_DATASETS  n_batches=$N_BATCHES"
echo "[INFO] PBS script: $PBS_SCRIPT"

for (( b=0; b<N_BATCHES; b++ )); do
    OFFSET=$(( b * BATCH_DATASETS ))
    echo "[INFO] submitting batch $((b+1))/$N_BATCHES (offset=$OFFSET)"
    qsub \
        -J "1-${SUBJOBS_PER_ARRAY}" \
        -v "CHUNK_SIZE=${CHUNK_SIZE},BATCH_OFFSET=${OFFSET},TOTAL_DATASETS=${TOTAL_DATASETS}" \
        "$PBS_SCRIPT"
done

echo "[INFO] submitted $N_BATCHES array jobs. monitor with: qstat -tu \$USER"
