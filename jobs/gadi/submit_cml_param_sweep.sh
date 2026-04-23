#!/bin/bash
# Submit all 2,440 CML param-sweep datasets to Gadi in chunked whole-node arrays.
#
# Defaults match the PBS script: whole-node (ncpus=48), 12 datasets run
# concurrently per subjob with PYSPI_N_JOBS=4 each. CHUNK_SIZE=122 per subjob,
# 10 subjobs per array (Gadi cap), so each submission covers 1,220 datasets.
# 2 submissions = 2,440 datasets = whole sweep.
#
# Per subjob: 122 datasets / 12 parallel = ~10 waves. At ~15 min/wave (estimate
# to confirm with canary) that's ~2.5h walltime. Walltime=6h is pro-rated.
#
# All flags are overridable via env, e.g.:
#   CHUNK_SIZE=60 SUBJOBS_PER_ARRAY=10 bash jobs/gadi/submit_cml_param_sweep.sh
# Re-running after partial completion is safe: --skip-existing in the PBS
# script skips datasets whose meta.json+calc.csv+spi_mpis.npz exist.
set -euo pipefail

CHUNK_SIZE="${CHUNK_SIZE:-122}"
SUBJOBS_PER_ARRAY="${SUBJOBS_PER_ARRAY:-10}"
TOTAL_DATASETS="${TOTAL_DATASETS:-2440}"
PARALLEL="${PARALLEL:-12}"
PYSPI_JOBS="${PYSPI_JOBS:-4}"
PBS_SCRIPT="${PBS_SCRIPT:-jobs/gadi/run_cml_param_sweep.pbs}"

BATCH_DATASETS=$(( CHUNK_SIZE * SUBJOBS_PER_ARRAY ))
N_BATCHES=$(( (TOTAL_DATASETS + BATCH_DATASETS - 1) / BATCH_DATASETS ))

echo "[INFO] total=$TOTAL_DATASETS  chunk=$CHUNK_SIZE  subjobs/array=$SUBJOBS_PER_ARRAY"
echo "[INFO] parallel/subjob=$PARALLEL  pyspi_jobs=$PYSPI_JOBS  -> $(( PARALLEL * PYSPI_JOBS )) cores/subjob"
echo "[INFO] datasets/batch=$BATCH_DATASETS  n_batches=$N_BATCHES"
echo "[INFO] PBS script: $PBS_SCRIPT"

for (( b=0; b<N_BATCHES; b++ )); do
    OFFSET=$(( b * BATCH_DATASETS ))
    echo "[INFO] submitting batch $((b+1))/$N_BATCHES (offset=$OFFSET)"
    qsub \
        -J "1-${SUBJOBS_PER_ARRAY}" \
        -v "CHUNK_SIZE=${CHUNK_SIZE},BATCH_OFFSET=${OFFSET},TOTAL_DATASETS=${TOTAL_DATASETS},PARALLEL=${PARALLEL},PYSPI_JOBS=${PYSPI_JOBS}" \
        "$PBS_SCRIPT"
done

echo "[INFO] submitted $N_BATCHES array jobs. monitor with: qstat -tu \$USER"
