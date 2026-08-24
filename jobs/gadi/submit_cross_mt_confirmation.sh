#!/bin/bash
set -euo pipefail

expected_manifest_sha="${1:?usage: $0 EXPECTED_DEVELOPMENT_MANIFEST_SHA256}"
manifest=/g/data/ql44/we2614/mts-spi-cross-mt-260824/development-manifest.json
config=configs/generate/embeddings/cross-mt-confirmation-260824.yaml
test -f "$manifest"
test -f "$config"
actual_manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
[[ "$actual_manifest_sha" == "$expected_manifest_sha" ]]

python3 - "$manifest" configs/analysis/cross-mt-transfer-260824.yaml <<'PY'
import hashlib
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
protocol_sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert manifest["status"] == "development_frozen_confirmation_unseen"
assert manifest["protocol_sha256"] == protocol_sha
PY

submit() {
    FILTER_M="$1" FILTER_T="$2" WALLTIME="$3" TASK_TIMEOUT="$4" \
        bash jobs/gadi/submit_dataset_farm.sh "$config"
}

submit 8 500 01:00:00 3000
submit 8 1000 01:30:00 4800
submit 8 2000 02:00:00 6600
submit 16 500 01:30:00 4800
submit 16 1000 02:00:00 6600
submit 16 2000 02:30:00 8400
submit 32 500 02:00:00 6600
submit 32 1000 02:30:00 8400
submit 32 2000 03:00:00 10200
