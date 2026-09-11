#!/bin/bash
# Finite dependency graph, not a recurring monitor. Run from the pinned worktree.
set -euo pipefail
: "${EXPECTED_COMMIT:?}"
source_dir=$PWD
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git -C /home/562/we2614/pyspi-fork rev-parse HEAD)" = 65317c9c1fd5f12358b8ede09b7576ef001a76dd
test -z "$(git -C /home/562/we2614/pyspi-fork status --porcelain)"
physics_dir=/scratch/ql44/we2614/mts-spi-data/order_parameter/cml2d_confirmation_260911/physics
confirm_root=/g/data/ql44/we2614/mts-spi-data/order_parameter/cml2d_confirmation_260911
frozen_dir=/scratch/ql44/we2614/mts-spi-data/order_parameter/cml2d_period_doubling_260911/primary-analysis
test ! -e "$physics_dir"
test ! -e "$confirm_root"
module purge
module load python3/3.12.1
source .venv/bin/activate
python -c "from pathlib import Path; from scripts.cml2d_confirmation import verify_frozen; verify_frozen(Path('$frozen_dir'))"
test "$(python -m scripts.scout_cml2d_period_doubling --config configs/scout/cml2d-period-doubling-confirmation.yaml --count-only)" = 544
python -m pytest -q tests/test_cml2d_period_doubling.py tests/test_cml2d_corpus.py tests/test_cml2d_spi.py tests/test_cml2d_confirmation.py
mkdir -p "$confirm_root" logs

physics=$(qsub -N cml2d-confirm-physics -l ncpus=576,mem=2280GB,walltime=00:40:00 \
    -v "EXPECTED_COMMIT=$EXPECTED_COMMIT,SCOUT_CONFIG=configs/scout/cml2d-period-doubling-confirmation.yaml,OUTPUT_DIR=$physics_dir" jobs/gadi/run_cml2d_physics.pbs)
stage_vars="SOURCE_DIR=$source_dir,EXPECTED_COMMIT=$EXPECTED_COMMIT,CONFIRM_ROOT=$confirm_root,PHYSICS_DIR=$physics_dir,FROZEN_DIR=$frozen_dir"
export_job=$(qsub -N cml2d-confirm-export -W "depend=afterok:$physics" \
    -v "$stage_vars,STAGE=export" jobs/gadi/run_cml2d_confirmation_stage.pbs)
primary=$(qsub -N cml2d-confirm-m32t1000 -W "depend=afterok:$export_job" \
    -l ncpus=576,mem=2280GB,walltime=01:00:00 \
    -v "EXPERIMENT_CONFIG=$confirm_root/primary-corpus.yaml,START_INDEX=1,END_INDEX=544,WORKERS=544,TASK_TIMEOUT=3000" jobs/gadi/run_dataset_farm.pbs)
m16t500=$(qsub -N cml2d-confirm-m16t500 -W "depend=afterok:$export_job" \
    -l ncpus=288,mem=1140GB,walltime=00:20:00 \
    -v "EXPERIMENT_CONFIG=$confirm_root/sensitivity-corpus.yaml,START_INDEX=1,END_INDEX=864,FILTER_M=16,FILTER_T=500,WORKERS=288,TASK_TIMEOUT=900" jobs/gadi/run_dataset_farm.pbs)
m16t1000=$(qsub -N cml2d-confirm-m16t1000 -W "depend=afterok:$export_job" \
    -l ncpus=288,mem=1140GB,walltime=00:25:00 \
    -v "EXPERIMENT_CONFIG=$confirm_root/sensitivity-corpus.yaml,START_INDEX=1,END_INDEX=864,FILTER_M=16,FILTER_T=1000,WORKERS=288,TASK_TIMEOUT=1200" jobs/gadi/run_dataset_farm.pbs)
m32t500=$(qsub -N cml2d-confirm-m32t500 -W "depend=afterok:$export_job" \
    -l ncpus=288,mem=1140GB,walltime=00:30:00 \
    -v "EXPERIMENT_CONFIG=$confirm_root/sensitivity-corpus.yaml,START_INDEX=1,END_INDEX=864,FILTER_M=32,FILTER_T=500,WORKERS=288,TASK_TIMEOUT=1500" jobs/gadi/run_dataset_farm.pbs)
primary_report=$(qsub -N cml2d-confirm-primary-report -W "depend=afterok:$primary" \
    -v "$stage_vars,STAGE=primary" jobs/gadi/run_cml2d_confirmation_stage.pbs)
secondary_report=$(qsub -N cml2d-confirm-MT-report -W "depend=afterok:$m16t500:$m16t1000:$m32t500" \
    -v "$stage_vars,STAGE=sensitivity" jobs/gadi/run_cml2d_confirmation_stage.pbs)
python - "$confirm_root" "$EXPECTED_COMMIT" "$physics" "$export_job" "$primary" "$m16t500" "$m16t1000" "$m32t500" "$primary_report" "$secondary_report" <<'PY'
import json, sys
from pathlib import Path
keys=['physics','export','primary_p90','m16t500_p90','m16t1000_p90','m32t500_p90','primary_report','secondary_report']
record=dict(source_commit=sys.argv[2], jobs=dict(zip(keys,sys.argv[3:],strict=True)))
with (Path(sys.argv[1])/'submission.json').open('x') as handle:
    json.dump(record,handle,indent=2)
print(json.dumps(record,indent=2))
PY
