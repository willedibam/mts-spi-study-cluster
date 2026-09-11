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
resume=${RESUME:-0}
if [[ "$resume" == p90 ]]; then
    test -f "$confirm_root/primary-corpus.yaml"
    test -f "$confirm_root/sensitivity-corpus.yaml"
    test -f "$confirm_root/frozen-input-identity.json"
    test ! -e "$confirm_root/submission-p90-retry.json"
elif [[ "$resume" == 1 ]]; then
    test -d "$physics_dir"
    test -f "$confirm_root/submission.json"
    # Only a failed physics-stage attempt can be resumed by this launcher.
    test ! -e "$confirm_root/primary"
    test ! -e "$confirm_root/sensitivity"
    test ! -e "$confirm_root/submission-retry.json"
else
    test ! -e "$physics_dir"
    test ! -e "$confirm_root"
fi
module purge
module load python3/3.12.1
source .venv/bin/activate
python -c "from pathlib import Path; from scripts.cml2d_confirmation import verify_frozen; verify_frozen(Path('$frozen_dir'))"
test "$(python -m scripts.scout_cml2d_period_doubling --config configs/scout/cml2d-period-doubling-confirmation.yaml --count-only)" = 544
python -m pytest -q tests/test_cml2d_period_doubling.py tests/test_cml2d_corpus.py tests/test_cml2d_spi.py tests/test_cml2d_confirmation.py
mkdir -p "$confirm_root" logs

dependency=()
[[ -n "${PHYSICS_DEPENDENCY:-}" ]] && dependency=(-W "depend=afterok:$PHYSICS_DEPENDENCY")
stage_vars="SOURCE_DIR=$source_dir,EXPECTED_COMMIT=$EXPECTED_COMMIT,CONFIRM_ROOT=$confirm_root,PHYSICS_DIR=$physics_dir,FROZEN_DIR=$frozen_dir"
if [[ "$resume" == p90 ]]; then
    python -m src.run_external_corpus --config "$confirm_root/primary-corpus.yaml" --validate-source
    python -m src.run_external_corpus --config "$confirm_root/sensitivity-corpus.yaml" --validate-source
    test "$(python -m src.run_external_corpus --config "$confirm_root/primary-corpus.yaml" --count-only)" = 544
    test "$(python -m src.run_external_corpus --config "$confirm_root/sensitivity-corpus.yaml" --count-only)" = 864
    physics=reused-178771601
    export_job=reused-178771602
    farm_dependency=()
else
physics=$(qsub -N cml2d-confirm-physics "${dependency[@]}" -l ncpus=576,mem=2280GB,walltime=00:40:00 \
    -v "EXPECTED_COMMIT=$EXPECTED_COMMIT,SCOUT_CONFIG=configs/scout/cml2d-period-doubling-confirmation.yaml,OUTPUT_DIR=$physics_dir,RESUME=$resume" jobs/gadi/run_cml2d_physics.pbs)
export_job=$(qsub -N cml2d-confirm-export -W "depend=afterok:$physics" \
    -v "$stage_vars,STAGE=export" jobs/gadi/run_cml2d_confirmation_stage.pbs)
farm_dependency=(-W "depend=afterok:$export_job")
fi
primary=$(qsub -N cml2d-confirm-m32t1000 "${farm_dependency[@]}" \
    -l ncpus=576,mem=2280GB,walltime=01:00:00 \
    -v "CORPUS_CONFIG=$confirm_root/primary-corpus.yaml,START_INDEX=1,END_INDEX=544,WORKERS=544,TASK_TIMEOUT=3000" jobs/gadi/run_external_corpus_farm.pbs)
m16t500=$(qsub -N cml2d-confirm-m16t500 "${farm_dependency[@]}" \
    -l ncpus=288,mem=1140GB,walltime=00:20:00 \
    -v "CORPUS_CONFIG=$confirm_root/sensitivity-corpus.yaml,INDEX_FILE=$confirm_root/sensitivity/indices-m16-t500.txt,WORKERS=288,TASK_TIMEOUT=900" jobs/gadi/run_external_corpus_farm.pbs)
m16t1000=$(qsub -N cml2d-confirm-m16t1000 "${farm_dependency[@]}" \
    -l ncpus=288,mem=1140GB,walltime=00:25:00 \
    -v "CORPUS_CONFIG=$confirm_root/sensitivity-corpus.yaml,INDEX_FILE=$confirm_root/sensitivity/indices-m16-t1000.txt,WORKERS=288,TASK_TIMEOUT=1200" jobs/gadi/run_external_corpus_farm.pbs)
m32t500=$(qsub -N cml2d-confirm-m32t500 "${farm_dependency[@]}" \
    -l ncpus=288,mem=1140GB,walltime=00:30:00 \
    -v "CORPUS_CONFIG=$confirm_root/sensitivity-corpus.yaml,INDEX_FILE=$confirm_root/sensitivity/indices-m32-t500.txt,WORKERS=288,TASK_TIMEOUT=1500" jobs/gadi/run_external_corpus_farm.pbs)
primary_report=$(qsub -N cml2d-confirm-primary-report -W "depend=afterok:$primary" \
    -v "$stage_vars,STAGE=primary" jobs/gadi/run_cml2d_confirmation_stage.pbs)
secondary_report=$(qsub -N cml2d-confirm-MT-report -W "depend=afterok:$m16t500:$m16t1000:$m32t500" \
    -v "$stage_vars,STAGE=sensitivity" jobs/gadi/run_cml2d_confirmation_stage.pbs)
python - "$confirm_root" "$EXPECTED_COMMIT" "$resume" "$physics" "$export_job" "$primary" "$m16t500" "$m16t1000" "$m32t500" "$primary_report" "$secondary_report" <<'PY'
import json, sys
from pathlib import Path
keys=['physics','export','primary_p90','m16t500_p90','m16t1000_p90','m32t500_p90','primary_report','secondary_report']
record=dict(source_commit=sys.argv[2], resume_stage=sys.argv[3], jobs=dict(zip(keys,sys.argv[4:],strict=True)))
name={'0':'submission.json','1':'submission-retry.json','p90':'submission-p90-retry.json'}[sys.argv[3]]
with (Path(sys.argv[1])/name).open('x') as handle:
    json.dump(record,handle,indent=2)
print(json.dumps(record,indent=2))
PY
